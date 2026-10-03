/* olmoe_chain.h -- OLMoE's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by olmoe.c in a COLI_VULKAN build, after the CPU forward it stands in
 * for; COLI_VK_CHAIN decides (backend_vulkan.c, coli_vk_chain_decide).
 *
 * OLMoE is the qwen36 chain's Qwen3-Coder case: every layer attention and MoE, no
 * output gate, no shared expert. What runs where, per layer, for one block of rows
 * (decode: one row):
 *   device, one frame: the routed MoE output of the layer before joins the residual
 *     (x += routed, chain_ew COMBINE, as layers_forward_range adds it), the input
 *     RMSNorm, q/k/v, OLMoE's q and k RMSNorms over the whole projection (not per
 *     head), RoPE per head from a host table (the CPU's own powf/cosf/sinf), the new
 *     K/V rows into the device cache, attention, o_proj, the residual add, the
 *     post-attention RMSNorm and the router logits. Then the frame is waited for.
 *   host: the router's softmax and top-k, the routed experts (the expert tier's device
 *     batch and the CPU's share, joined in rank order: moe_routed, the code moe()
 *     runs), the K/V rows copied into the host cache; with PILOT, the prefetch of the
 *     next layers' experts from the residual rows, as the CPU path does it (S <= 8:
 *     the rows after attention come down with the frame, the rows after the MoE are
 *     those plus the routed sum, the same float add the device makes).
 * There is no shared expert, so nothing runs on the device while the host computes the
 * routed experts. Crossing per layer: the normalized rows (D floats a row), the router
 * logits (E a row) and the K/V rows down, the routed sum (D a row) up. The last frame
 * normalizes the last row and runs lm_head; its logits come back.
 *
 * State: OLMoE is attention only. The host's KV cache stays canonical (every chain
 * step copies its new rows back); the device holds a mirror per layer with a watermark,
 * kv_valid: rows [0, kv_valid) equal the host's. A step from pos_base uploads the rows
 * [kv_valid, pos_base) first; a CPU step lowers the watermark to its pos_base; a host
 * cache allocated anew (generate, a larger max_t) is mirrored again. Prompt-cache and
 * prefix reuse need nothing more: a reused prefix is rows below the watermark.
 *
 * The chain declines (the per-matrix path runs, the watermark lowered) for a geometry
 * outside chain_attn.comp (head dim above 256 or odd). A device lost mid-step costs
 * nothing to rebuild: the KV rows below pos_base are the host's, so the CPU redoes the
 * step from the embedding rows (which the chain never overwrites) and runs from there.
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests. */
#include "vk_chain.h"

/* COLI_VK_CHAIN unset on an integrated GPU with the expert tier (coli_vk_chain_decide):
 * on. Measured on a Radeon 780M with OLMoE-1B-7B (docs/vulkan.md, "OLMoE and Inkling"):
 * against the tier alone the chain decodes 17.3 tok/s to 12.8 and prefills 512 tokens
 * in 5.5 s to 6.4 (the CPU alone decodes 23.1 tok/s: its f32 trunk) */
#ifndef OLMOE_CHAIN_IGPU
#define OLMOE_CHAIN_IGPU COLI_VK_CHAIN_ON
#endif

typedef struct {
    int ok, failed;
    int rows;                                  /* scratch capacity in rows */
    int cap;                                   /* device KV rows (the host's max_t) */
    float **hostK;                             /* the host cache the mirror copies */
    VkcBuf *prm;                               /* every norm weight */
    size_t *o_in, *o_post, *o_qn, *o_kn, o_final;
    VkcBuf **kc, **vc;
    int *kv_valid;
    VkcBuf *x, *nrm, *tmp, *q, *k, *v, *ctx, *h2, *lg, *fin;
    VkcBuf *h2d, *lgd, *kvd, *outd, *xd, *xpd, *routed, *cs;
    float *host_routed, *xpost;
    unsigned long long forwards;
    double host_ms;
} OlmChain;

static int olc_chunk_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}

/* A resident f32 matrix [O x I] on the device: the copy the per-matrix path uses
 * (matmul_res), uploaded here if it has not been yet. NULL: refused (it stays so). */
static ColiVkTensor *olc_tensor(void **vk, const float *w, int I, int O) {
    if (*vk == (void *)&g_vk_refused || !w) return NULL;
    if (*vk) return (ColiVkTensor *)*vk;
    if (!coli_vk_tensor_ensure((ColiVkTensor **)vk, w, NULL, 10, I, O, 0)) { *vk = &g_vk_refused; return NULL; }
    return (ColiVkTensor *)*vk;
}

/* The model's parameters on the device, its tensors resolved; NULL = the chain cannot run. */
static OlmChain *olc_setup(Model *m) {
    OlmChain *ch = (OlmChain *)m->vkchain;
    if (ch) return ch->ok ? ch : NULL;
    ch = (OlmChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->vkchain = ch;
    Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, E = c->n_experts;
    if (c->head_dim > 256 || (c->head_dim & 1) || c->head_dim * c->n_heads != D) {
        fprintf(stderr, "[VK] olmoe chain: a geometry its shaders do not take (head dim %d, %d heads, hidden %d); per-matrix path\n",
                c->head_dim, c->n_heads, D);
        return NULL;
    }
    ch->o_in = calloc(L, sizeof(size_t)); ch->o_post = calloc(L, sizeof(size_t));
    ch->o_qn = calloc(L, sizeof(size_t)); ch->o_kn = calloc(L, sizeof(size_t));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int));
    if (!ch->o_in || !ch->o_post || !ch->o_qn || !ch->o_kn || !ch->kc || !ch->vc || !ch->kv_valid) return NULL;
    /* the parameter arena: offsets, then one upload */
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        ch->o_qn[i] = n; n += D; ch->o_kn[i] = n; n += D;
    }
    ch->o_final = n; n += D;
    float *arena = calloc(n, sizeof(float));
    if (!arena) return NULL;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(arena + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(arena + ch->o_qn[i], l->qn, D * sizeof(float));
        memcpy(arena + ch->o_kn[i], l->kn, D * sizeof(float));
    }
    memcpy(arena + ch->o_final, m->final_norm, D * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return NULL;
    /* the tensors: the device copies the per-matrix path uses (matmul_res) */
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        ok = olc_tensor(&l->vk_q, l->q, D, D) && olc_tensor(&l->vk_k, l->k, D, D) &&
             olc_tensor(&l->vk_v, l->v, D, D) && olc_tensor(&l->vk_o, l->o, D, D) &&
             olc_tensor(&l->vk_gate, l->gate, D, E);
    }
    ok = ok && olc_tensor(&m->vk_lm_head, m->lm_head, D, c->vocab);
    if (!ok) { fprintf(stderr, "[VK] olmoe chain: a matrix did not reach the device; per-matrix path\n"); return NULL; }
    ch->ok = 1;
    fprintf(stderr, "[VK] olmoe chain: %d layers on the device, %.1f MiB of parameters\n", L, n * 4 / 1048576.0);
    return ch;
}

static int olc_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* scratch for `rows` rows, and the KV mirrors at the host's capacity */
static int olc_scratch(OlmChain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, H = c->n_heads, hd = c->head_dim;
    size_t r = (size_t)rows;
    int ok = olc_res(&ch->x, r * D, VKC_DEV) && olc_res(&ch->nrm, r * D, VKC_DEV) && olc_res(&ch->tmp, r * D, VKC_DEV) &&
             olc_res(&ch->q, r * D, VKC_DEV) && olc_res(&ch->k, r * D, VKC_DEV) && olc_res(&ch->v, r * D, VKC_DEV) &&
             olc_res(&ch->ctx, r * D, VKC_DEV) && olc_res(&ch->h2, r * D, VKC_DEV) && olc_res(&ch->lg, r * E, VKC_DEV) &&
             olc_res(&ch->fin, D, VKC_DEV) &&
             olc_res(&ch->h2d, r * D, VKC_DOWN) && olc_res(&ch->lgd, r * E, VKC_DOWN) && olc_res(&ch->kvd, 2 * r * D, VKC_DOWN) &&
             olc_res(&ch->outd, c->vocab, VKC_DOWN) && olc_res(&ch->routed, r * D, VKC_UP) && olc_res(&ch->cs, r * hd, VKC_UP);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float)), *xp = hr ? realloc(ch->xpost, r * D * sizeof(float)) : NULL;
        if (hr) ch->host_routed = hr;
        if (!hr || !xp) return 0;
        ch->xpost = xp; ch->rows = rows;
    }
    if (ch->cap != m->max_t || ch->hostK != m->K) {   /* the host cache is new: mirror it again */
        for (int i = 0; i < c->n_layers; i++) {
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); ch->kc[i] = ch->vc[i] = NULL;
            ch->kv_valid[i] = 0;
            ch->kc[i] = vkc_buf((size_t)H * m->max_t * hd * sizeof(float), VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)H * m->max_t * hd * sizeof(float), VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i]) { ch->cap = 0; ch->hostK = NULL; return 0; }
        }
        ch->cap = m->max_t; ch->hostK = m->K;
    }
    return 1;
}

/* A CPU step from pos_base: the rows it writes into the host cache are not the device's. */
static void olc_cpu_step(Model *m, int pos_base) {
    OlmChain *ch = (OlmChain *)m->vkchain;
    if (!ch || !ch->ok) return;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
}
/* Record the uploads that make the device cache the host's below pos_base. */
static int olc_push_kv(OlmChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int hd = c->head_dim, ok = 1;
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < c->n_heads && ok; h++) {
            size_t off = ((size_t)h * m->max_t + t0) * hd;
            ok = vkc_write(ch->kc[i], off, m->K[i] + off, (size_t)n * hd * sizeof(float)) &&
                 vkc_write(ch->vc[i], off, m->V[i] + off, (size_t)n * hd * sizeof(float));
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* One layer's attention for n rows from pb: nrm in, the o_proj rows in tmp. */
static int olc_attention(OlmChain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int D = c->hidden, H = c->n_heads, hd = c->head_dim, half = hd / 2;
    int ok = vkc_matmul((ColiVkTensor *)l->vk_q, ch->nrm, 0, ch->q, 0, n) &&
             vkc_matmul((ColiVkTensor *)l->vk_k, ch->nrm, 0, ch->k, 0, n) &&
             vkc_matmul((ColiVkTensor *)l->vk_v, ch->nrm, 0, ch->v, 0, n);
    /* q and k normalized over the whole projection, then RoPE per head */
    VkcNorm qn = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_qn[i], 0, 0, c->eps, 1.f};
    VkcNorm kn = {n, D, 1, 0, D, D, 0, D, D, (int)ch->o_kn[i], 0, 0, c->eps, 1.f};
    VkcRope rp = {n * H, H, 0, D, hd, half, 0, 2 * half};
    ok = ok && vkc_norm(ch->q, ch->prm, ch->q, &qn) && vkc_norm(ch->k, ch->prm, ch->k, &kn) &&
         vkc_rope(ch->q, ch->cs, &rp) && vkc_rope(ch->k, ch->cs, &rp);
    if (!ok) return 0;
    /* the new rows into the device cache, and down for the host's */
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * H);
    if (!rg) return 0;
    for (int s = 0; s < n; s++) for (int h = 0; h < H; h++)
        rg[s * H + h] = (VkcRegion){((size_t)h * ch->cap + pb + s) * hd, (size_t)s * D + (size_t)h * hd, (size_t)hd};
    ok = vkc_copy_regions(ch->kc[i], ch->k, rg, n * H) && vkc_copy_regions(ch->vc[i], ch->v, rg, n * H) &&
         vkc_copy(ch->kvd, 0, ch->k, 0, (size_t)n * D) && vkc_copy(ch->kvd, (size_t)ch->rows * D, ch->v, 0, (size_t)n * D);
    free(rg);
    VkcAttn a = {n, H, H, hd, pb, ch->cap, 0, D, hd, 0, 0, 0, 0, 0, D, 0, 0, 1.f / sqrtf((float)hd), 0, 0};
    return ok && vkc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, NULL, NULL, &a) &&
           vkc_matmul((ColiVkTensor *)l->vk_o, ch->ctx, 0, ch->tmp, 0, n);
}
static int olc_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}

/* The device was lost in a frame of the step from pos_base. */
static void olc_lost(Model *m, int pos_base) {
    OlmChain *ch = (OlmChain *)m->vkchain;
    g_vk_chain = 0;
    if (ch) ch->failed = 1;
    fprintf(stderr, "[VK] olmoe chain: the device was lost; the CPU redoes the step from position %d "
                    "(the KV rows below it are the host's) and runs from here on\n", pos_base);
}

/* Every layer for S rows from host rows xh, the last row's logits into `logit`; xh gets
 * the final rows back when want_x. 0 = not taken: the host cache below pos_base and xh
 * are as they were, and the caller runs the step on the CPU. */
static int olc_forward(Model *m, float *xh, int S, int pos_base, int want_x, float *logit) {
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    OlmChain *ch = olc_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int D = c->hidden, L = c->n_layers, E = c->n_experts, H = c->n_heads, hd = c->head_dim;
    int CH = olc_chunk_rows(), rows = S < CH ? S : CH, half = hd / 2;
    int pilot = g_pilot >= 1 && S <= 8;          /* layers_forward_range's PILOT condition */
    if (pos_base + S > m->max_t || !olc_scratch(ch, m, rows) ||
        (want_x && !olc_res(&ch->xd, (size_t)rows * D, VKC_DOWN)) ||
        (pilot && !olc_res(&ch->xpd, (size_t)rows * D, VKC_DOWN))) {
        fprintf(stderr, "[VK] olmoe chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    float *xfin = want_x ? malloc((size_t)S * D * sizeof(float)) : NULL;
    if (want_x && !xfin) return 0;
    vkc_gemm_rows(-1);
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        if (!vkc_begin() || !vkc_write(ch->x, 0, xh + (size_t)c0 * D, (size_t)n * D * sizeof(float)) ||
            !olc_push_kv(ch, m, pb)) goto lost;
        float *cs = (float *)vkc_ptr(ch->cs);   /* rope_head's angles, cosines and sines */
        for (int s = 0; s < n; s++)
            for (int j = 0; j < half; j++) {
                float inv = powf(c->theta, -2.0f * j / c->head_dim);
                float ang = (pb + s) * inv;
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        int ok = 1;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            VkcEw add = {VKC_EW_ADD, n * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            if (i > 0) {   /* x += routed: the layer before's MoE output */
                VkcEw cb = {VKC_EW_COMBINE, n * D, D, 1, 1, 1, 0, 0, 0, 0, 0, 1.f};
                ok = vkc_ew(ch->x, ch->x, ch->routed, NULL, NULL, &cb);
            }
            ok = ok && olc_norm(ch->x, 0, ch->prm, ch->o_in[i], ch->nrm, 0, n, D, c->eps) &&
                 olc_attention(ch, m, l, i, n, pb) && vkc_ew(ch->x, ch->x, ch->tmp, NULL, NULL, &add);
            if (ok && pilot) ok = vkc_copy(ch->xpd, 0, ch->x, 0, (size_t)n * D);
            ok = ok && olc_norm(ch->x, 0, ch->prm, ch->o_post[i], ch->h2, 0, n, D, c->eps) &&
                 vkc_matmul((ColiVkTensor *)l->vk_gate, ch->h2, 0, ch->lg, 0, n) &&
                 vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->h2d, 0, ch->h2, 0, (size_t)n * D);
            double t0 = now_s();
            ok = ok && vkc_submit(1);
            g_prof_attn_s += now_s() - t0;
            if (!ok) break;
            const float *kv = (const float *)vkc_ptr(ch->kvd);   /* the rows into the host's cache */
            for (int s = 0; s < n; s++) for (int h = 0; h < H; h++) {
                size_t dst = ((size_t)h * m->max_t + pb + s) * hd, src = (size_t)s * D + (size_t)h * hd;
                memcpy(m->K[i] + dst, kv + src, hd * sizeof(float));
                memcpy(m->V[i] + dst, kv + (size_t)ch->rows * D + src, hd * sizeof(float));
            }
            const float *xa = pilot ? (const float *)vkc_ptr(ch->xpd) : NULL;
            if (pilot && i + 1 < L) pilot_prefetch(m, i + 1, xa, n);
            double t1 = now_s();
            moe_routed(m, i, (float *)vkc_ptr(ch->h2d), n, (float *)vkc_ptr(ch->lgd), ch->host_routed);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * D * sizeof(float));
            double t2 = now_s();
            g_prof_moe_s += t2 - t1; ch->host_ms += (t2 - t1) * 1e3;
            if (pilot && g_pilot >= 2 && i + 2 < L) {   /* the residual after the MoE, as the device will add it */
                for (size_t j = 0; j < (size_t)n * D; j++) ch->xpost[j] = xa[j] + ch->host_routed[j];
                pilot_prefetch(m, i + 2, ch->xpost, n);
                if (g_pilot >= 3 && i + 3 < L) pilot_prefetch(m, i + 3, ch->xpost, n);
            }
            ok = vkc_begin();
        }
        if (ok) { VkcEw cb = {VKC_EW_COMBINE, n * D, D, 1, 1, 1, 0, 0, 0, 0, 0, 1.f};
                  ok = vkc_ew(ch->x, ch->x, ch->routed, NULL, NULL, &cb); }
        if (ok && want_x) ok = vkc_copy(ch->xd, 0, ch->x, 0, (size_t)n * D);
        int last = c0 + n == S;
        if (ok && last) ok = olc_norm(ch->x, (size_t)(n - 1) * D, ch->prm, ch->o_final, ch->fin, 0, 1, D, c->eps) &&
                             vkc_matmul((ColiVkTensor *)m->vk_lm_head, ch->fin, 0, ch->outd, 0, 1);
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        g_prof_head_s += now_s() - t0;
        if (!ok) goto lost;
        if (want_x) memcpy(xfin + (size_t)c0 * D, vkc_ptr(ch->xd), (size_t)n * D * sizeof(float));
        for (int i = 0; i < L; i++) ch->kv_valid[i] = pb + n;
        if (last) memcpy(logit, vkc_ptr(ch->outd), (size_t)c->vocab * sizeof(float));
    }
    if (want_x) { memcpy(xh, xfin, (size_t)S * D * sizeof(float)); free(xfin); }
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    free(xfin);
    olc_lost(m, pos_base);
    return 0;
}

static void olc_report(Model *m) {
    OlmChain *ch = (OlmChain *)m->vkchain;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] olmoe chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}

/* COLI_VK_CHAIN, decided after the tier (model_init): the chain's pipelines, and its
 * teardown registered after the tier's so that it runs first at exit. */
static void olc_start(Model *m) {
    (void)m;
    if (!g_vk_ready) return;
    /* measured on a Radeon 780M (docs/vulkan.md, "OLMoE and Inkling") */
    g_vk_chain = coli_vk_chain_decide("olmoe", vkt_ready(), OLMOE_CHAIN_IGPU);
    if (g_vk_chain && !vkc_init()) g_vk_chain = 0;
    if (g_vk_chain) atexit(vkc_shutdown);
}
