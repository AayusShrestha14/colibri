/* qwen36_chain.h -- Qwen3.6's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by qwen36.c in a COLI_VULKAN build, after the CPU forward it stands in
 * for; COLI_VK_CHAIN decides (backend_vulkan.c, coli_vk_chain_decide).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device, frame A1: the routed MoE output of the layer before joins the residual
 *     (x += routed + gate * shared, chain_ew COMBINE, the CPU's order), the input
 *     RMSNorm, then either the gated attention (q/k/v, per-head q/k norm, RoPE from a
 *     host table, the K/V rows into the device cache, attention with the output gate,
 *     o_proj) or the Gated DeltaNet (qkv/z/b/a, the convolution with its ring, the
 *     recurrence with its state, the gated norm, out_proj), the residual add, the
 *     post-attention RMSNorm and the router logits. Then the frame is waited for.
 *   host: the router's softmax and top-k, the routed experts (the Vulkan expert tier's
 *     device batch and the CPU's share, joined in rank order: moe_ex routed_only), the
 *     K/V rows copied into the host cache.
 *   device, frame A2 (not waited for): the shared expert and its gate, while the host
 *     computes the routed experts.
 * Crossing per layer: the normalized rows (D floats per row) and the router logits
 * (E per row) down, the K/V rows of an attention layer down, the routed sum (D per
 * row) up. A dense model (no routed experts) records the whole forward in one frame.
 * The last frame normalizes the last row and runs lm_head; its logits come back.
 *
 * State and who owns it:
 *   - the residual stream: on the device for the whole forward;
 *   - the attention KV cache: the host's stays canonical (every chain step copies its
 *     new rows back), the device holds a mirror with a watermark per layer, kv_valid:
 *     rows [0, kv_valid) equal the host's. A step that starts at pos_base uploads the
 *     rows [kv_valid, pos_base) first; a CPU step lowers the watermark to its
 *     pos_base; a cache that grows (ensure_kv) is mirrored again from the host;
 *   - the DeltaNet recurrent state and conv ring: on the device while the chain runs
 *     (60 MB on the 35B: too much to copy per token). dn_where says which side holds
 *     the newest copy; the host one is brought back before anything reads it there
 *     (a CPU step, a pinned snapshot) and pushed up after anything writes it there
 *     (reset_recurrent: a fill with zeros on the device, pin_restore: an upload).
 * Prompt-cache and prefix reuse need nothing more: a reused prefix is rows below the
 * watermark and a DeltaNet state that already sits where the next step expects it.
 *
 * The chain declines (the per-matrix path runs, state synced first) under the CUDA
 * expert tier, a qpack container, PILOT prefetch, or geometry outside its shaders
 * (head dim > 256, DeltaNet value head > 128, key head > 256, conv kernel > 9).
 * A device lost while the chain holds the recurrent state: the engine rebuilds that
 * state on the CPU from the prefix record (the ids it was built from; the KV rows are
 * the host's already), stays on the CPU, and the step runs there. Only a state that
 * the ids do not describe (an image) cannot be rebuilt: that stops the engine.
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests. */
#include "vk_chain.h"

#define Q36C_HOST 0
#define Q36C_DEV  1
#define Q36C_BOTH 2

typedef struct {
    int ok, failed;
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* device KV rows (the host's kv_cap) */
    VkcBuf *prm;                               /* every norm / conv / DeltaNet parameter */
    size_t *o_in, *o_post, *o_qn, *o_kn, *o_conv, *o_dn, o_final;
    ColiVkTensor **t_ab, **t_sg;               /* DeltaNet b|a rows, shared gate row (f32) */
    VkcBuf **rec, **ring, **kc, **vc;          /* per layer state */
    int *kv_valid, *attn_ord, n_attn;
    int dn_where, host_zero;
    VkcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *qkv, *z, *ab, *cv, *dny, *h2, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    VkcBuf *h2d, *lgd, *kvd, *outd, *xd, *lfd, *routed, *cs;
    float *host_routed;
    unsigned long long forwards, frames, host_ms_n;
    double wait_ms, host_ms;
} Q36Chain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */
static int q36c_chunk_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}

static void q36c_fatal(const char *what) {
    fprintf(stderr, "[VK] qwen36 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
/* The device was lost with the newest recurrent state on it. The host's KV rows are
 * canonical, the DeltaNet state is not: rebuild it on the CPU by running the `upto`
 * positions the prefix record names (a prefill on the CPU), and leave the chain off. */
static void q36c_recover(Model *m, int upto) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    Cfg *c = &m->c; int D = c->hidden;
    g_vk_chain = 0;
    if (ch) { ch->failed = 1; ch->dn_where = Q36C_HOST; ch->host_zero = 0; }
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q36c_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[VK] qwen36 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = malloc((size_t)upto * sizeof(int));
    float *x = falloc((int64_t)upto * D);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) memcpy(x + (int64_t)s * D, m->embed + (int64_t)ids[s] * D, D * sizeof(float));
    layers_forward_range(m, x, upto, 0, 0, c->n_layers, 0, NULL);
    free(ids); free(x);
}

/* f32 rows [O x I] as a resident fmt 10 tensor */
static ColiVkTensor *q36c_f32_tensor(const float *w, int I, int O) {
    ColiVkTensor *t = NULL;
    return coli_vk_tensor_ensure(&t, w, NULL, 10, I, O, 0) ? t : NULL;
}

static int q36c_geometry_ok(const Cfg *c) {
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) {
            if (c->head_dim > 256 || c->head_dim != c->k_head_dim || c->q_heads % c->kv_heads ||
                c->q_head_dim < c->head_dim || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim) return 0;
        } else {
            if (c->dn_vdim > 128 || c->dn_kdim > 256 || c->dn_convk < 2 || c->dn_convk > 9 ||
                c->dn_vheads % c->dn_kheads) return 0;
        }
    }
    return 1;
}

/* The model's parameters on the device, its tensors resolved; NULL = the chain cannot run. */
static Q36Chain *q36c_setup(Model *m) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    if (ch) return ch->ok ? ch : NULL;
    ch = (Q36Chain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->vkchain = ch;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden;
    if (!q36c_geometry_ok(c)) {
        fprintf(stderr, "[VK] qwen36 chain: a geometry its shaders do not take (head dim %d, DeltaNet %dx%d); per-matrix path\n",
                c->head_dim, c->dn_kdim, c->dn_vdim);
        return NULL;
    }
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t)); ch->o_qn = calloc(L, sizeof(size_t));
    ch->o_kn = calloc(L, sizeof(size_t)); ch->o_conv = calloc(L, sizeof(size_t)); ch->o_dn = calloc(L, sizeof(size_t));
    ch->t_ab = calloc(L, sizeof(void *)); ch->t_sg = calloc(L, sizeof(void *));
    ch->rec = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->attn_ord = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->o_conv || !ch->o_dn || !ch->t_ab || !ch->t_sg ||
        !ch->rec || !ch->ring || !ch->kc || !ch->vc || !ch->kv_valid || !ch->attn_ord) return NULL;
    /* the parameter arena: offsets, then one upload */
    size_t n = 0;
    int vh = c->dn_vheads, conv_dim = c->dn_conv_dim, convk = c->dn_convk;
    for (int i = 0; i < L; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        if (c->is_attn[i]) {
            ch->attn_ord[i] = ch->n_attn++;
            if (m->L[i].qn) { ch->o_qn[i] = n; n += c->head_dim; }
            if (m->L[i].kn) { ch->o_kn[i] = n; n += c->k_head_dim; }
        } else {
            ch->o_conv[i] = n; n += (size_t)conv_dim * convk;
            ch->o_dn[i] = n; n += 2 * (size_t)vh + c->dn_vdim;
        }
    }
    ch->o_final = n; n += D;
    float *arena = calloc(n, sizeof(float));
    if (!arena) return NULL;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        if (c->is_attn[i]) {
            if (l->qn) memcpy(arena + ch->o_qn[i], l->qn, c->head_dim * sizeof(float));
            if (l->kn) memcpy(arena + ch->o_kn[i], l->kn, c->k_head_dim * sizeof(float));
        } else {
            memcpy(arena + ch->o_conv[i], l->dn_conv, (size_t)conv_dim * convk * sizeof(float));
            memcpy(arena + ch->o_dn[i], l->dn_alog, vh * sizeof(float));
            memcpy(arena + ch->o_dn[i] + vh, l->dn_dtbias, vh * sizeof(float));
            memcpy(arena + ch->o_dn[i] + 2 * vh, l->dn_norm, c->dn_vdim * sizeof(float));
        }
    }
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return NULL;
    /* the tensors: the same device copies the per-matrix path uses */
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        if (c->is_attn[i]) ok = vk_qw_tensor(&l->q) && vk_qw_tensor(&l->k) && vk_qw_tensor(&l->v) && vk_qw_tensor(&l->o);
        else {
            ok = vk_qw_tensor(&l->dn_qkv) && vk_qw_tensor(&l->dn_z) && vk_qw_tensor(&l->dn_out);
            float *ab = ok ? malloc((size_t)2 * vh * D * sizeof(float)) : NULL;
            if (ab) {
                memcpy(ab, l->dn_b, (size_t)vh * D * sizeof(float));
                memcpy(ab + (size_t)vh * D, l->dn_a, (size_t)vh * D * sizeof(float));
                ch->t_ab[i] = q36c_f32_tensor(ab, D, 2 * vh);
                free(ab);
            }
            ok = ok && ch->t_ab[i];
            if (ok) {
                ch->rec[i] = vkc_buf((size_t)vh * c->dn_kdim * c->dn_vdim * sizeof(float), VKC_DEV);
                ch->ring[i] = vkc_buf((size_t)conv_dim * (convk - 1) * sizeof(float), VKC_DEV);
                ok = ch->rec[i] && ch->ring[i];
            }
        }
        if (ok && c->n_experts > 0) ok = vk_qw_tensor(&l->gate) != NULL;
        if (ok && c->shared_inter > 0) ok = vk_qw_tensor(&l->sh_g) && vk_qw_tensor(&l->sh_u) && vk_qw_tensor(&l->sh_d);
        if (ok && l->sh_gate) ok = (ch->t_sg[i] = q36c_f32_tensor(l->sh_gate, D, 1)) != NULL;
    }
    ok = ok && vk_qw_tensor(&m->lm_head);
    if (!ok) { fprintf(stderr, "[VK] qwen36 chain: a matrix did not reach the device; per-matrix path\n"); return NULL; }
    ch->dn_where = Q36C_HOST; ch->host_zero = 0;
    ch->ok = 1;
    fprintf(stderr, "[VK] qwen36 chain: %d layers on the device (%d attention), %.1f MiB of parameters\n",
            L, ch->n_attn, n * 4 / 1048576.0);
    return ch;
}

static int q36c_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* scratch for `rows` rows, and the KV mirrors at the host's capacity */
static int q36c_scratch(Q36Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, H = c->q_heads, KV = c->kv_heads, hd = c->head_dim;
    int qo = H * c->q_head_dim, kvo = KV * c->k_head_dim, vd = c->dn_vheads * c->dn_vdim;
    int E = c->n_experts > 0 ? c->n_experts : 1, SI = c->shared_inter > 0 ? c->shared_inter : 1;
    size_t r = (size_t)rows;
    int ok = q36c_res(&ch->x, r * D, VKC_DEV) && q36c_res(&ch->nrm, r * D, VKC_DEV) && q36c_res(&ch->tmp, r * D, VKC_DEV) &&
             q36c_res(&ch->q, r * qo, VKC_DEV) && q36c_res(&ch->k, r * kvo, VKC_DEV) && q36c_res(&ch->v, r * kvo, VKC_DEV) &&
             q36c_res(&ch->ctx, r * H * hd, VKC_DEV) && q36c_res(&ch->qkv, r * c->dn_conv_dim, VKC_DEV) &&
             q36c_res(&ch->z, r * vd, VKC_DEV) && q36c_res(&ch->ab, r * 2 * c->dn_vheads, VKC_DEV) &&
             q36c_res(&ch->cv, r * c->dn_conv_dim, VKC_DEV) && q36c_res(&ch->dny, r * vd, VKC_DEV) &&
             q36c_res(&ch->h2, r * D, VKC_DEV) && q36c_res(&ch->lg, r * E, VKC_DEV) &&
             q36c_res(&ch->gs, r * SI, VKC_DEV) && q36c_res(&ch->us, r * SI, VKC_DEV) && q36c_res(&ch->hs, r * SI, VKC_DEV) &&
             q36c_res(&ch->ds, r * D, VKC_DEV) && q36c_res(&ch->sgd, r, VKC_DEV) && q36c_res(&ch->fin, D, VKC_DEV) &&
             q36c_res(&ch->h2d, r * D, VKC_DOWN) && q36c_res(&ch->lgd, r * E, VKC_DOWN) &&
             q36c_res(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * 2 * r * kvo, VKC_DOWN) &&
             q36c_res(&ch->outd, c->vocab, VKC_DOWN) && q36c_res(&ch->routed, r * D, VKC_UP) &&
             q36c_res(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    if (ch->cap != m->kv_cap) {        /* the host cache grew: mirror it again */
        for (int i = 0; i < c->n_layers; i++) {
            if (!c->is_attn[i]) continue;
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            ch->kv_valid[i] = 0;
            ch->kc[i] = vkc_buf((size_t)KV * m->kv_cap * c->k_head_dim * sizeof(float), VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)KV * m->kv_cap * c->k_head_dim * sizeof(float), VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}

/* ---- state between the host and the device ----------------------------------- */
/* The host's DeltaNet state brought up to date (before anything reads it there). */
static void q36c_sync_host(Model *m) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    if (!ch || !ch->ok || ch->dn_where != Q36C_DEV) return;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_attn[i]) continue;
        if (!vkc_read(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) ||
            !vkc_read(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float))) { q36c_recover(m, m->kv_len); return; }
    }
    ch->dn_where = Q36C_BOTH;
}
/* The host wrote its DeltaNet state (zero: reset_recurrent's zeros). */
static void q36c_host_wrote(Model *m, int zero) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    if (!ch || !ch->ok) return;
    ch->dn_where = Q36C_HOST; ch->host_zero = zero;
}
/* A CPU step from pos_base: the host state current before it, the device's stale after. */
static void q36c_cpu_step(Model *m, int pos_base) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    if (!ch || !ch->ok) return;
    q36c_sync_host(m);
    ch->dn_where = Q36C_HOST; ch->host_zero = 0;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
}
/* Record the uploads that make the device state the host's, as the step at pos_base needs it. */
static int q36c_push_state(Q36Chain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q36C_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = 0; i < c->n_layers && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = vkc_zero(ch->rec[i], 0, nr) && vkc_zero(ch->ring[i], 0, nc);
            else ok = vkc_write(ch->rec[i], 0, m->DN_rec[i], nr * sizeof(float)) &&
                      vkc_write(ch->ring[i], 0, m->DN_conv[i], nc * sizeof(float));
        }
        ch->dn_where = Q36C_BOTH;
    }
    int kvd = c->k_head_dim, KV = c->kv_heads;
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (!c->is_attn[i] || ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KV && ok; h++) {
            size_t src = ((size_t)h * m->max_t + t0) * kvd, dst = ((size_t)h * ch->cap + t0) * kvd;
            ok = vkc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * kvd * sizeof(float)) &&
                 vkc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * kvd * sizeof(float));
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* ---- one layer's pieces ---------------------------------------------------------- */
static int q36c_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, VKC_NORM_ADD1, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int q36c_attention(Q36Chain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int H = c->q_heads, KV = c->kv_heads, hd = c->head_dim, qdim = c->q_head_dim, kvd = c->k_head_dim;
    int qo = H * qdim, kvo = KV * kvd, half = c->rotary_dim / 2, gate_dim = qdim > hd ? qdim - hd : 0;
    int ok = vkc_matmul(vk_qw_tensor(&l->q), ch->nrm, 0, ch->q, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->k), ch->nrm, 0, ch->k, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->v), ch->nrm, 0, ch->v, 0, n);
    if (ok && l->qn) { VkcNorm p = {n * H, hd, H, 0, qo, qdim, 0, qo, qdim, (int)ch->o_qn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->q, ch->prm, ch->q, &p); }
    if (ok && half) { VkcRope p = {n * H, H, 0, qo, qdim, half, 0, 2 * half}; ok = vkc_rope(ch->q, ch->cs, &p); }
    if (ok && l->kn) { VkcNorm p = {n * KV, kvd, KV, 0, kvo, kvd, 0, kvo, kvd, (int)ch->o_kn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
                       ok = vkc_norm(ch->k, ch->prm, ch->k, &p); }
    if (ok && half) { VkcRope p = {n * KV, KV, 0, kvo, kvd, half, 0, 2 * half}; ok = vkc_rope(ch->k, ch->cs, &p); }
    if (!ok) return 0;
    /* the new rows into the device cache, and to the host's */
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * KV);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < KV; h++)
        rg[s * KV + h] = (VkcRegion){((size_t)h * ch->cap + pb + s) * kvd, (size_t)s * kvo + (size_t)h * kvd, (size_t)kvd};
    size_t ko = (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
    ok = vkc_copy_regions(ch->kc[i], ch->k, rg, n * KV) && vkc_copy_regions(ch->vc[i], ch->v, rg, n * KV) &&
         vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) && vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    free(rg);
    VkcAttn a = {n, H, KV, hd, pb, ch->cap, 0, qo, qdim, hd, qo, qdim, gate_dim > 0, 0, H * hd, 0, 0,
                 1.f / sqrtf((float)hd), 0, 0};
    return ok && vkc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, NULL, &a) &&
           vkc_matmul(vk_qw_tensor(&l->o), ch->ctx, 0, ch->tmp, 0, n);
}
static int q36c_deltanet(Q36Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c;
    int vh = c->dn_vheads, CD = c->dn_conv_dim, vd = vh * c->dn_vdim;
    int ok = vkc_matmul(vk_qw_tensor(&l->dn_qkv), ch->nrm, 0, ch->qkv, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->dn_z), ch->nrm, 0, ch->z, 0, n) &&
             vkc_matmul(ch->t_ab[i], ch->nrm, 0, ch->ab, 0, n);
    VkcDnConv cp = {n, CD, c->dn_convk, 0, CD, 0, CD, -1, 0, (int)ch->o_conv[i], 0, 0};
    VkcDnRec rp = {n, vh, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, 0, CD, 0, 2 * vh, vh, 2 * vh,
                   0, vd, 0, vd, -1, 0, c->eps, 1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
    return ok && vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, NULL, &cp) &&
           vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, NULL, &rp) &&
           vkc_matmul(vk_qw_tensor(&l->dn_out), ch->dny, 0, ch->tmp, 0, n);
}
static int q36c_shared(Q36Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int ok = 1, SI = c->shared_inter;
    if (SI > 0) {
        VkcEw p = {VKC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
        ok = vkc_matmul(vk_qw_tensor(&l->sh_g), ch->h2, 0, ch->gs, 0, n) &&
             vkc_matmul(vk_qw_tensor(&l->sh_u), ch->h2, 0, ch->us, 0, n) &&
             vkc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &p) &&
             vkc_matmul(vk_qw_tensor(&l->sh_d), ch->hs, 0, ch->ds, 0, n);
    }
    if (ok && ch->t_sg[i]) ok = vkc_matmul(ch->t_sg[i], ch->h2, 0, ch->sgd, 0, n);
    return ok;
}
/* x += routed + gate * shared, as moe() leaves `out` and layers_forward_range adds it */
static int q36c_combine(Q36Chain *ch, Model *m, int i, int n) {
    Cfg *c = &m->c;
    int flags = (c->n_experts > 0 ? 1 : 0) | (c->shared_inter > 0 ? 2 : 0) | (m->L[i].sh_gate && c->shared_inter > 0 ? 4 : 0);
    VkcEw p = {VKC_EW_COMBINE, n * c->hidden, c->hidden, 1, flags, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->x, ch->x, ch->routed, ch->ds, ch->sgd, &p);
}

/* Every layer for S rows from host rows xh, the last row's logits into `logit`; xh gets
 * the final rows back when want_x. 0 = not taken (nothing on the device changed). */
static int q36c_forward(Model *m, float *xh, int S, int pos_base, FILE *lf, int want_x, float *logit) {
    if (!g_vk_chain || qt_ready() || qq_active() || g_pilot) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    Q36Chain *ch = q36c_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, E = c->n_experts;
    int CH = q36c_chunk_rows(), rows = S < CH ? S : CH;
    int kvo = c->kv_heads * c->k_head_dim, half = c->rotary_dim / 2;
    if (!q36c_scratch(ch, m, rows) || (lf && !q36c_res(&ch->lfd, (size_t)L * 3 * D, VKC_DOWN)) ||
        (want_x && !q36c_res(&ch->xd, (size_t)rows * D, VKC_DOWN))) {
        fprintf(stderr, "[VK] qwen36 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !q36c_push_state(ch, m, pb)) goto lost;
        if (half) {   /* the CPU's own angles, cosines and sines (rope_head_partial / _mrope) */
            float *cs = (float *)vkc_ptr(ch->cs);
            for (int s = 0; s < n; s++) {
                int p3[3]; int mr = m->mpos || m->rope_delta;
                if (mr) mrope_at(m, pb + s, p3);
                for (int j = 0; j < half; j++) {
                    float inv = powf(c->theta, -2.0f * j / c->rotary_dim);
                    int axis = !mr ? 0 : (j % 3 == 1 && j < 3 * c->mrope_section[1]) ? 1 : (j % 3 == 2 && j < 3 * c->mrope_section[2]) ? 2 : 0;
                    float ang = (mr ? p3[axis] : pb + s) * inv;
                    cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
                }
            }
        }
        int ok = 1, pending = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {
                ok = q36c_combine(ch, m, i - 1, n);
                if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(i - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
            }
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps);
            ok = ok && (c->is_attn[i] ? q36c_attention(ch, m, l, i, n, pb) : q36c_deltanet(ch, m, l, i, n));
            if (ok && lf) ok = vkc_copy(ch->lfd, (size_t)i * 3 * D, ch->tmp, (size_t)(n - 1) * D, D);
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = ok && vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)i * 3 + 1) * D, ch->x, (size_t)(n - 1) * D, D);
            ok = ok && q36c_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps);
            if (ok && E > 0) {
                ok = vkc_matmul(vk_qw_tensor(&l->gate), ch->h2, 0, ch->lg, 0, n) &&
                     vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
                double t0 = tm_now();
                ok = ok && vkc_submit(1);                      /* A1 */
                ch->frames++; ch->wait_ms += tm_now() - t0;
                if (!ok) break;
                if (c->is_attn[i]) {                           /* the rows into the host's cache */
                    const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
                    for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
                    }
                }
                /* A2: the shared expert, while the host computes the routed experts */
                ok = vkc_begin() && q36c_shared(ch, m, l, i, n) && vkc_submit(0);
                ch->frames++;
                double t1 = tm_now();
                moe_ex(m, l, i, (float *)vkc_ptr(ch->h2d), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), 1);
                memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
                double t2 = tm_now();
                tm_add(n, 2, t2 - t1); ch->host_ms += t2 - t1;
                ok = ok && vkc_begin();
            } else if (ok) ok = q36c_shared(ch, m, l, i, n);   /* a dense model: no host step */
            pending = 1;
        }
        if (ok) ok = q36c_combine(ch, m, L - 1, n);
        if (ok && lf) ok = vkc_copy(ch->lfd, ((size_t)(L - 1) * 3 + 2) * D, ch->x, (size_t)(n - 1) * D, D);
        if (ok && want_x) ok = vkc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        if (ok && last) ok = q36c_norm(ch->x, (size_t)(n - 1) * D, ch->prm, ch->o_final, ch->fin, 0, 1, D, c->eps) &&
                             vkc_matmul(vk_qw_tensor(&m->lm_head), ch->fin, 0, ch->outd, 0, 1);
        double t0 = tm_now();
        ok = ok && vkc_submit(1);
        ch->frames++; ch->wait_ms += tm_now() - t0;
        if (!ok) goto lost;
        if (E == 0) for (int i = 0; i < L; i++) {             /* a dense model reads its K/V rows back here */
            if (!c->is_attn[i]) continue;
            const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * 2 * ch->rows * kvo;
            for (int s = 0; s < n; s++) for (int h = 0; h < c->kv_heads; h++) {
                size_t dst = ((size_t)h * m->max_t + pb + s) * c->k_head_dim, src = (size_t)s * kvo + (size_t)h * c->k_head_dim;
                memcpy(m->K[i] + dst, kv + src, c->k_head_dim * sizeof(float));
                memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->k_head_dim * sizeof(float));
            }
        }
        if (want_x) memcpy(xh + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q36C_DEV; ch->host_zero = 0;
        if (last) memcpy(logit, vkc_ptr(ch->outd), (size_t)c->vocab * sizeof(float));
    }
    if (lf) fwrite(vkc_ptr(ch->lfd), sizeof(float), (size_t)L * 3 * D, lf);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    q36c_recover(m, pos_base);
    return 0;
}

static void q36c_report(Model *m) {
    Q36Chain *ch = (Q36Chain *)m->vkchain;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] qwen36 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}
