/* glm53_chain.h -- GLM-5.3 Flash's layers as a dense chain on the Vulkan device
 * (vk_chain.h). Included once by glm53.c in a COLI_VULKAN build, after run_layers, the
 * CPU forward it stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide, this
 * engine's integrated GPU default off: not measured on a GLM-5.3 checkpoint).
 *
 * The residual is hc_mult streams per position (mHC). What runs where, per layer, for
 * one block of rows (decode: one row):
 *   device, frame A1: the previous layer's FFN branch joins the streams (the routed sum
 *     from the host plus the shared expert, then mHC's write back); the attention site:
 *     mHC's mix (hc_fn, the split with Sinkhorn, the collapse), the input RMSNorm, then
 *     the KDA layer (the q/k/v, decay, beta and gate projections, the short convolution
 *     with its window, the delta rule with its state, the output norm and gate, o) or the
 *     MLA layer (vkc_mla_qkv, the k-pooled indexer: index keys with their LayerNorm, the
 *     pool gates, each completed pool's key, every row's pools; vkc_mla_attn over the
 *     selection), mHC's write back; the FFN site's mix, collapse and post-attention
 *     RMSNorm, and on a dense layer its MLP (clamped SwiGLU) and write back with no host
 *     step. Then the frame is waited for.
 *   host: the router and the routed experts (ffn_layer without the shared expert: the
 *     expert tier's batch and the CPU's share), the new MLA rows into the host's cache.
 *   device, frame A2 (not waited for): the shared expert (clamped SwiGLU).
 * The final streams come back to the host, which collapses them and runs the final norm
 * and the head as before.
 *
 * State and who owns it (one session at a time, `owner`):
 *   - the MLA caches (latent, index keys, pool gates): the host's stay canonical (every
 *     step copies its new rows back); the device mirrors them behind a watermark that
 *     every host write lowers (a CPU forward of the session, a pin restore rewinding
 *     `filled`); the pools' keys have a watermark of their own.
 *   - the KDA state and convolution windows: on the device while the chain runs (too
 *     large to copy per token). `where` says which side holds the newest copy; the host's
 *     is brought back before anything reads it there (a pin, a state capture, a CPU
 *     forward of the session, another session taking the device) and pushed up after
 *     anything writes it there (a restored pin or state).
 * A device lost while the device holds the newest KDA state: the state is rebuilt on the
 * CPU from the input rows of the positions since the host's copy was last current (the
 * chain keeps them: the embedding rows, or the vision tower's), the forward runs again on
 * the CPU and the CPU runs from there. COLI_VK_CHAIN_FAULT=n fakes the loss at the n-th
 * frame.
 *
 * The chain declines (the CPU path runs, the state synced first) for a layer range (a
 * segment), a session above the device's limits, and matrices or a geometry its shaders
 * do not take (kv_lora above 1024, qk_rope above 0 is not GLM-5.3's and is declined too,
 * a KDA head above 256 key or 128 value floats, an indexer above 64 heads, 4096 query
 * floats or k-pooling below 2). */
#include "vk_chain.h"

#define G53C_HOST 0
#define G53C_DEV  1
#define G53C_BOTH 2

typedef struct {
    int ok, failed, rows, cap, pcap;
    const GSession *owner;
    int where;                                /* G53C_HOST / DEV / BOTH for owner's KDA state */
    VkcBuf *prm;
    size_t *o_in, *o_post, *o_hca, *o_hcf, *o_qn, *o_kn, *o_ikw, *o_ikb, *o_ape, *o_conv, *o_kda;
    ColiVkTensor **fna, **fnf;                /* the mHC mix matrices, f32 */
    VkcMla *mla;
    VkcMlaCache *kv; VkcBuf **ik, **ig, **pk, **win, **st;
    int *kv_valid, *pool_valid;
    VkcMlaScratch sc;
    VkcBuf *xs, *xn, *col, *nrm, *br, *mix, *hp, *gs, *us, *hs, *ds, *qkv3, *cm, *kf, *kb, *kg, *klow, *ky;
    VkcBuf *ikd, *iq, *ihw, *igd, *isc, *sel;
    VkcBuf *h2d, *kvd, *xd, *routed;
    size_t kvd_layer;
    float *host_routed;
    /* the input rows since the host's KDA state was last current: [rec_base, rec_base + rec_len) */
    float *rec; int rec_base, rec_len, rec_cap, rec_ok;
    unsigned long long forwards;
    double host_ms;
} G53Chain;

static G53Chain *g_g53c;
static int g_vk_chain = 0;
static int g_g53c_inited = 0;

static int g53c_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}

/* A Mat's device copy (the per-matrix path's own, where it made one); f32 as fmt 10. */
static ColiVkTensor *g53c_tensor(const Mat *wc) {
    Mat *w = (Mat *)wc;
    if (w->vk) return (ColiVkTensor *)w->vk;
    int fmt; const void *p;
    switch (w->fmt) {
    case 0: fmt = 10; p = w->f; break;
    case 1: fmt = 1; p = w->q8; break;
    case 4: if (w->gs < 8 || w->gs % 8) return NULL; fmt = 4; p = w->q4; break;
    default: return NULL;
    }
    if (!p || w->rows < 1 || w->columns < 1) return NULL;
    return coli_vk_tensor_ensure((ColiVkTensor **)&w->vk, p, w->fmt ? w->s : NULL, fmt, w->columns, w->rows,
                                 w->fmt == 4 ? w->gs : 0) ? (ColiVkTensor *)w->vk : NULL;
}

/* ---- the KDA state between the host and the device --------------------------------- */
static void g53c_rec_reset(G53Chain *ch, int base) { ch->rec_base = base; ch->rec_len = 0; ch->rec_ok = 1; }
/* The host's copy of s's KDA state made current (before anything reads it there). */
static void g53c_sync_host(const GModel *m, const GSession *s) {
    G53Chain *ch = g_g53c;
    if (!ch || !ch->ok || ch->owner != s || ch->where != G53C_DEV || !s) return;
    const Cfg *c = &m->c;
    size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)3 * c->kda_proj * c->conv_k;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->is_full[i]) continue;
        if (!vkc_read(ch->st[i], 0, s->layer[i].kda_state, ns * sizeof(float)) ||
            !vkc_read(ch->win[i], 0, s->layer[i].kda_window, nw * sizeof(float))) {
            fprintf(stderr, "[VK] glm53 chain: the device was lost holding the KDA state; it is rebuilt at the next step\n");
            return;   /* where stays DEV: the next forward rebuilds from the record */
        }
    }
    ch->where = G53C_BOTH;
    g53c_rec_reset(ch, s->filled);
}
/* The host wrote s's KDA state (a restored pin or state). */
static void g53c_host_wrote(const GSession *s) {
    G53Chain *ch = g_g53c;
    if (ch && ch->owner == s) ch->where = G53C_HOST;
}
/* s goes away: whatever the device holds for it goes with it. */
static void g53c_session_gone(const GSession *s) {
    G53Chain *ch = g_g53c;
    if (ch && ch->owner == s) { ch->owner = NULL; ch->where = G53C_HOST; }
}
/* A CPU forward of s from `start`: its state current on the host first, the device's
 * stale after (and the MLA rows from start on). */
static void g53c_cpu_step(const GModel *m, const GSession *s, int start) {
    G53Chain *ch = g_g53c;
    if (!ch || !ch->ok || ch->owner != s) return;
    g53c_sync_host(m, s);
    ch->where = G53C_HOST;
    for (int i = 0; i < m->c.n_layers; i++) {
        if (ch->kv_valid[i] > start) ch->kv_valid[i] = start;
        if (ch->pool_valid[i] > start / m->c.index_kpool) ch->pool_valid[i] = start / m->c.index_kpool;
    }
}

/* ---- setup ----------------------------------------------------------------------------- */
static int g53c_setup(GModel *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, D = c->hidden, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, P = c->kda_proj;
    const char *why = NULL;
    if (m->layer_begin != 0 || m->layer_end != L || !m->has_io) why = "a layer range";
    else if (c->kv_lora > 1024 || c->qk_rope != 0 || c->qk_nope > 1024 || c->q_lora < 1) why = "an attention geometry its shaders do not take";
    else if (c->index_nh > 64 || c->index_nh * c->index_hd > 4096 || c->index_kpool < 2 || c->index_topk % c->index_kpool ||
             c->index_topk / c->index_kpool > 1024) why = "an indexer its shaders do not take";
    else if (P > 0 && (c->kda_hd > 128 || c->conv_k > 8)) why = "a KDA head its shaders do not take";
    else if (H < 1 || H > 8 || c->hc_iters < 1) why = "hyper-connections its shaders do not take";
    if (why) { fprintf(stderr, "[VK] glm53 chain: %s; per-matrix path\n", why); return 0; }
    G53Chain *ch = calloc(1, sizeof *ch);
    if (!ch) return 0;
    size_t **offs[] = {&ch->o_in, &ch->o_post, &ch->o_hca, &ch->o_hcf, &ch->o_qn, &ch->o_kn, &ch->o_ikw, &ch->o_ikb,
                       &ch->o_ape, &ch->o_conv, &ch->o_kda};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = calloc(L, sizeof(size_t)))) return 0;
    ch->fna = calloc(L, sizeof(void *)); ch->fnf = calloc(L, sizeof(void *));
    ch->mla = calloc(L, sizeof(VkcMla)); ch->kv = calloc(L, sizeof(VkcMlaCache));
    ch->ik = calloc(L, sizeof(void *)); ch->ig = calloc(L, sizeof(void *)); ch->pk = calloc(L, sizeof(void *));
    ch->win = calloc(L, sizeof(void *)); ch->st = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->pool_valid = calloc(L, sizeof(int));
    if (!ch->fna || !ch->fnf || !ch->mla || !ch->kv || !ch->ik || !ch->ig || !ch->pk || !ch->win || !ch->st ||
        !ch->kv_valid || !ch->pool_valid) return 0;
    g_g53c = ch;
    /* the parameter arena */
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_in[i] = n; n += D; ch->o_post[i] = n; n += D;
        ch->o_hca[i] = n; n += 3 + nm; ch->o_hcf[i] = n; n += 3 + nm;
        if (c->is_full[i]) {
            ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->kv_lora;
            ch->o_ikw[i] = n; n += c->index_hd; ch->o_ikb[i] = n; n += c->index_hd;
            ch->o_ape[i] = n; n += (size_t)c->index_kpool * c->index_hd;
        } else {
            ch->o_conv[i] = n; n += (size_t)3 * P * c->conv_k;
            ch->o_kda[i] = n; n += (size_t)c->kda_heads + P + c->kda_hd;
        }
    }
    float *a = calloc(n, sizeof(float));
    if (!a) return 0;
    for (int i = 0; i < L; i++) {
        const GLayer *l = &m->layer[i];
        memcpy(a + ch->o_in[i], l->in_ln, D * sizeof(float));
        memcpy(a + ch->o_post[i], l->post_ln, D * sizeof(float));
        memcpy(a + ch->o_hca[i], l->hc_attn_scale, 3 * sizeof(float));
        memcpy(a + ch->o_hca[i] + 3, l->hc_attn_base, nm * sizeof(float));
        memcpy(a + ch->o_hcf[i], l->hc_ffn_scale, 3 * sizeof(float));
        memcpy(a + ch->o_hcf[i] + 3, l->hc_ffn_base, nm * sizeof(float));
        if (c->is_full[i]) {
            memcpy(a + ch->o_qn[i], l->qa_ln, c->q_lora * sizeof(float));
            memcpy(a + ch->o_kn[i], l->kva_ln, c->kv_lora * sizeof(float));
            memcpy(a + ch->o_ikw[i], l->ik_nw, c->index_hd * sizeof(float));
            if (l->ik_nb) memcpy(a + ch->o_ikb[i], l->ik_nb, c->index_hd * sizeof(float));
            memcpy(a + ch->o_ape[i], l->ikpa, (size_t)c->index_kpool * c->index_hd * sizeof(float));
        } else {
            memcpy(a + ch->o_conv[i], l->conv, (size_t)3 * P * c->conv_k * sizeof(float));
            memcpy(a + ch->o_kda[i], l->alog, c->kda_heads * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads, l->dt, P * sizeof(float));
            memcpy(a + ch->o_kda[i] + c->kda_heads + P, l->onorm, c->kda_hd * sizeof(float));
        }
    }
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    for (int i = 0; i < L && ok; i++) {
        const GLayer *l = &m->layer[i];
        ok = coli_vk_tensor_ensure(&ch->fna[i], l->hc_attn_fn, NULL, 10, HD, nm, 0) &&
             coli_vk_tensor_ensure(&ch->fnf[i], l->hc_ffn_fn, NULL, 10, HD, nm, 0);
        if (ok && c->is_full[i]) {
            ch->mla[i] = (VkcMla){c->n_heads, c->qk_nope, 0, c->v_head, c->kv_lora, D, c->q_lora, c->eps,
                                  1.0f / sqrtf((float)c->qk_nope), VKC_ROPE_HALF, g53c_tensor(&l->qa), g53c_tensor(&l->qb),
                                  g53c_tensor(&l->kva), NULL, g53c_tensor(&l->kvb_kt), g53c_tensor(&l->kvb_v),
                                  g53c_tensor(&l->o), ch->prm, ch->o_qn[i], ch->o_kn[i]};
            ok = ch->mla[i].q_a && ch->mla[i].q_b && ch->mla[i].kv_a && ch->mla[i].k_abs && ch->mla[i].v_abs && ch->mla[i].o &&
                 g53c_tensor(&l->iwq) && g53c_tensor(&l->iwk) && g53c_tensor(&l->iwp) && g53c_tensor(&l->ikpg);
        } else if (ok)
            ok = g53c_tensor(&l->kq) && g53c_tensor(&l->kk) && g53c_tensor(&l->kv) && g53c_tensor(&l->ko) &&
                 g53c_tensor(&l->kga) && g53c_tensor(&l->kgb) && g53c_tensor(&l->kfa) && g53c_tensor(&l->kfb) &&
                 g53c_tensor(&l->kb);
        if (ok && i < c->first_dense) ok = g53c_tensor(&l->dg) && g53c_tensor(&l->du) && g53c_tensor(&l->dd);
        if (ok && i >= c->first_dense) ok = g53c_tensor(&l->rg) && g53c_tensor(&l->ru) && g53c_tensor(&l->rd);
        if (ok && !c->is_full[i]) {
            ch->win[i] = vkc_buf((size_t)3 * P * c->conv_k * sizeof(float), VKC_DEV);
            ch->st[i] = vkc_buf((size_t)c->kda_heads * c->kda_hd * c->kda_hd * sizeof(float), VKC_DEV);
            ok = ch->win[i] && ch->st[i];
        }
    }
    if (!ok) { fprintf(stderr, "[VK] glm53 chain: a matrix did not reach the device (or has no device form); per-matrix path\n"); return 0; }
    ch->ok = 1;
    int full = 0; for (int i = 0; i < L; i++) full += c->is_full[i];
    fprintf(stderr, "[VK] glm53 chain: %d layers on the device (%d KDA, %d MLA with the k-pooled indexer, %d dense), "
                    "%d streams, %.1f MiB of parameters\n", L, L - full, full, c->first_dense, H, n * 4 / 1048576.0);
    return 1;
}

static int g53c_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

static int g53c_scratch(G53Chain *ch, const GModel *m, int rows, int ctx) {
    const Cfg *c = &m->c;
    int D = c->hidden, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, P = c->kda_proj, L = c->n_layers;
    int IH = c->index_nh, ID = c->index_hd, MI = c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter;
    int width = coli_sparse_index_width(c->index_topk, c->index_kpool, c->index_kpool_tail);
    int mfull = -1; for (int i = 0; i < L; i++) if (c->is_full[i]) { mfull = i; break; }
    size_t r = (size_t)rows;
    ch->kvd_layer = r * (c->kv_lora + 2 * ID);
    int ok = (mfull < 0 || vkc_mla_scratch(&ch->sc, &ch->mla[mfull], rows)) &&
             g53c_res(&ch->xs, r * HD, VKC_DEV) && g53c_res(&ch->xn, r * HD, VKC_DEV) && g53c_res(&ch->col, r * D, VKC_DEV) &&
             g53c_res(&ch->nrm, r * D, VKC_DEV) && g53c_res(&ch->br, r * D, VKC_DEV) && g53c_res(&ch->mix, r * nm, VKC_DEV) &&
             g53c_res(&ch->hp, r * (2 * H + H * H), VKC_DEV) && g53c_res(&ch->gs, r * MI, VKC_DEV) &&
             g53c_res(&ch->us, r * MI, VKC_DEV) && g53c_res(&ch->hs, r * MI, VKC_DEV) && g53c_res(&ch->ds, r * D, VKC_DEV) &&
             g53c_res(&ch->h2d, r * D, VKC_DOWN) && g53c_res(&ch->kvd, (size_t)L * ch->kvd_layer, VKC_DOWN) &&
             g53c_res(&ch->xd, r * HD, VKC_DOWN) && g53c_res(&ch->routed, r * D, VKC_UP);
    if (ok && P > 0)
        ok = g53c_res(&ch->qkv3, 3 * r * P, VKC_DEV) && g53c_res(&ch->cm, r * 3 * P, VKC_DEV) && g53c_res(&ch->kf, r * P, VKC_DEV) &&
             g53c_res(&ch->kb, r * c->kda_heads, VKC_DEV) && g53c_res(&ch->kg, r * P, VKC_DEV) &&
             g53c_res(&ch->klow, r * c->kda_hd, VKC_DEV) && g53c_res(&ch->ky, r * P, VKC_DEV);
    if (ok && mfull >= 0)
        ok = g53c_res(&ch->ikd, r * ID, VKC_DEV) && g53c_res(&ch->iq, r * IH * ID, VKC_DEV) && g53c_res(&ch->ihw, r * IH, VKC_DEV) &&
             g53c_res(&ch->igd, r * ID, VKC_DEV) && g53c_res(&ch->isc, r * (size_t)(ctx / c->index_kpool + 1), VKC_DEV) &&
             g53c_res(&ch->sel, r * (size_t)(1 + width), VKC_DEV);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *hr = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hr) return 0;
        ch->host_routed = hr; ch->rows = rows;
    }
    return 1;
}

/* The MLA mirror: room for `need` positions (the owner's), filled again when it grows. */
static int g53c_mirror(G53Chain *ch, const GModel *m, int need, int limit) {
    const Cfg *c = &m->c; int L = c->n_layers, ID = c->index_hd, pool = c->index_kpool;
    if (ch->cap >= need) return 1;
    int cap = 256; while (cap < need) cap *= 2;
    if (cap > limit) cap = limit;
    if (cap < need) return 0;
    for (int i = 0; i < L; i++) {
        if (!c->is_full[i]) continue;
        vkc_free(ch->kv[i].lat); vkc_free(ch->ik[i]); vkc_free(ch->ig[i]); vkc_free(ch->pk[i]);
        ch->kv[i] = (VkcMlaCache){NULL, NULL, 0}; ch->ik[i] = ch->ig[i] = ch->pk[i] = NULL;
        ch->kv_valid[i] = ch->pool_valid[i] = 0;
    }
    ch->cap = 0;
    for (int i = 0; i < L; i++) {
        if (!c->is_full[i]) continue;
        ch->kv[i].lat = vkc_buf((size_t)cap * c->kv_lora * sizeof(float), VKC_DEV); ch->kv[i].cap = cap;
        ch->ik[i] = vkc_buf((size_t)cap * ID * sizeof(float), VKC_DEV);
        ch->ig[i] = vkc_buf((size_t)cap * ID * sizeof(float), VKC_DEV);
        ch->pk[i] = vkc_buf((size_t)(cap / pool + 1) * ID * sizeof(float), VKC_DEV);
        if (!ch->kv[i].lat || !ch->ik[i] || !ch->ig[i] || !ch->pk[i]) return 0;
    }
    ch->cap = cap;
    return 1;
}

/* Record the uploads that make the device the owner's below `start`. */
static int g53c_push(G53Chain *ch, const GModel *m, const GSession *s, int start) {
    const Cfg *c = &m->c; int ok = 1, K = c->kv_lora, ID = c->index_hd;
    if (ch->where == G53C_HOST) {
        size_t ns = (size_t)c->kda_heads * c->kda_hd * c->kda_hd, nw = (size_t)3 * c->kda_proj * c->conv_k;
        for (int i = 0; i < c->n_layers && ok; i++)
            if (!c->is_full[i])
                ok = vkc_write(ch->st[i], 0, s->layer[i].kda_state, ns * sizeof(float)) &&
                     vkc_write(ch->win[i], 0, s->layer[i].kda_window, nw * sizeof(float));
        ch->where = G53C_BOTH;
    }
    for (int i = 0; i < c->n_layers && ok; i++) {
        if (!c->is_full[i]) continue;
        const GLayerState *ls = &s->layer[i];
        int t0 = ch->kv_valid[i], n = start - t0;
        if (ch->pool_valid[i] > start / c->index_kpool) ch->pool_valid[i] = start / c->index_kpool;   /* rows from start change */
        if (n <= 0) continue;
        ok = vkc_write(ch->kv[i].lat, (size_t)t0 * K, ls->latent + (size_t)t0 * K, (size_t)n * K * sizeof(float)) &&
             vkc_write(ch->ik[i], (size_t)t0 * ID, ls->ikeys + (size_t)t0 * ID, (size_t)n * ID * sizeof(float)) &&
             vkc_write(ch->ig[i], (size_t)t0 * ID, ls->igates + (size_t)t0 * ID, (size_t)n * ID * sizeof(float));
        ch->kv_valid[i] = start;
    }
    return ok;
}
static void g53c_pull(G53Chain *ch, const GModel *m, GSession *s, int from, int to, int pb, int n) {
    const Cfg *c = &m->c; int K = c->kv_lora, ID = c->index_hd;
    for (int i = from; i < to; i++) {
        if (!c->is_full[i]) continue;
        const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)i * ch->kvd_layer;
        GLayerState *ls = &s->layer[i];
        memcpy(ls->latent + (size_t)pb * K, kv, (size_t)n * K * sizeof(float));
        memcpy(ls->ikeys + (size_t)pb * ID, kv + (size_t)n * K, (size_t)n * ID * sizeof(float));
        memcpy(ls->igates + (size_t)pb * ID, kv + (size_t)n * (K + ID), (size_t)n * ID * sizeof(float));
    }
}

/* ---- one layer's pieces ---------------------------------------------------------------- */
/* mHC's entry for a site: the mix, the split, the collapse, the RMSNorm into nrm */
static int g53c_pre(G53Chain *ch, const GModel *m, ColiVkTensor *fn, size_t hco, size_t lno, int n) {
    const Cfg *c = &m->c; int H = c->hc_mult, D = c->hidden, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    VkcMhc sp = {n, H, D, c->hc_iters, 0, HD, 0, nm, 0, hr, 0, D, (int)hco, 0, c->eps, c->hc_eps, 0.f};
    VkcNorm nr = {n, D, 1, 0, D, D, 0, D, D, (int)lno, 0, 0, c->eps, 1.f};
    return vkc_matmul(fn, ch->xs, 0, ch->mix, 0, n) && vkc_mhc(VKC_MHC_SPLIT, ch->xs, ch->mix, ch->hp, ch->prm, NULL, &sp) &&
           vkc_mhc(VKC_MHC_COLLAPSE, ch->xs, NULL, ch->hp, NULL, ch->col, &sp) && vkc_norm(ch->col, ch->prm, ch->nrm, &nr);
}
/* mHC's exit: xn = mix(xs) + post * br, then the two swap */
static int g53c_post(G53Chain *ch, const GModel *m, int n) {
    const Cfg *c = &m->c; int H = c->hc_mult, D = c->hidden, HD = H * D, hr = 2 * H + H * H;
    VkcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->eps, c->hc_eps, 0.f};
    if (!vkc_mhc(VKC_MHC_POST, ch->xs, ch->br, ch->hp, NULL, ch->xn, &po)) return 0;
    VkcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}
static int g53c_mlp(G53Chain *ch, const GModel *m, const Mat *g, const Mat *u, const Mat *d, VkcBuf *out, int n) {
    int I = g->rows;
    VkcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n * I, 0.f, 0.f, m->c.swiglu_limit};
    return vkc_matmul(g53c_tensor(g), ch->nrm, 0, ch->gs, 0, n) && vkc_matmul(g53c_tensor(u), ch->nrm, 0, ch->us, 0, n) &&
           vkc_mhc(VKC_SWIGLU_CLAMP, ch->gs, ch->us, NULL, NULL, ch->hs, &sw) && vkc_matmul(g53c_tensor(d), ch->hs, 0, out, 0, n);
}
static int g53c_kda(G53Chain *ch, const GModel *m, const GLayer *l, int i, int n) {
    const Cfg *c = &m->c; int P = c->kda_proj, HK = c->kda_heads, KD = c->kda_hd, r = ch->rows;
    VkcKdaConv cp = {n, 3 * P, c->conv_k, P, 0, P, r * P, 0, 3 * P, (int)ch->o_conv[i], 0};
    VkcKdaRec rp = {n, HK, KD, P, 0, 3 * P, 0, P, 0, HK, 0, P, 0, P, 0, (int)ch->o_kda[i], c->gate_lb, 1e-6f, c->eps};
    return vkc_matmul(g53c_tensor(&l->kq), ch->nrm, 0, ch->qkv3, 0, n) &&
           vkc_matmul(g53c_tensor(&l->kk), ch->nrm, 0, ch->qkv3, (size_t)r * P, n) &&
           vkc_matmul(g53c_tensor(&l->kv), ch->nrm, 0, ch->qkv3, (size_t)2 * r * P, n) &&
           vkc_matmul(g53c_tensor(&l->kfa), ch->nrm, 0, ch->klow, 0, n) &&
           vkc_matmul(g53c_tensor(&l->kfb), ch->klow, 0, ch->kf, 0, n) &&
           vkc_matmul(g53c_tensor(&l->kb), ch->nrm, 0, ch->kb, 0, n) &&
           vkc_matmul(g53c_tensor(&l->kga), ch->nrm, 0, ch->klow, 0, n) &&
           vkc_matmul(g53c_tensor(&l->kgb), ch->klow, 0, ch->kg, 0, n) &&
           vkc_kda_conv(ch->qkv3, ch->prm, ch->win[i], ch->cm, &cp) &&
           vkc_kda_rec(KD, ch->cm, ch->kf, ch->kb, ch->kg, ch->prm, ch->st[i], ch->ky, &rp) &&
           vkc_matmul(g53c_tensor(&l->ko), ch->ky, 0, ch->br, 0, n);
}
static int g53c_mla(G53Chain *ch, const GModel *m, const GLayer *l, int i, int n, int pb) {
    const Cfg *c = &m->c; int IH = c->index_nh, ID = c->index_hd, pool = c->index_kpool, K = c->kv_lora;
    int width = coli_sparse_index_width(c->index_topk, pool, c->index_kpool_tail);
    size_t ko = (size_t)i * ch->kvd_layer;
    VkcMlaRow ln = {n, 1, 0, ID, 0, 0, ID, 0, pb * ID, ID, 0, 0, 0, (int)ch->o_ikw[i], (int)ch->o_ikb[i], l->ik_nb != NULL, 1e-5f};
    int done = (pb + n) / pool, p0 = ch->pool_valid[i];
    VkcDsaPool kp = {done > p0 ? done - p0 : 0, pool, p0, ID, 0, ID, (int)ch->o_ape[i], 0, 0, ID};
    VkcDsaPick sp = {n, pb, IH, ID, c->index_topk, pool, 0, IH * ID, 0, IH, 0, c->index_kpool_tail, (pb + n) / pool + 1,
                     1 + width, sqrtf((float)IH), 1.f / sqrtf((float)ID)};
    int ok = vkc_mla_qkv(&ch->mla[i], &ch->sc, ch->nrm, 0, n, pb, NULL, &ch->kv[i], ch->kvd, ko) &&
             vkc_matmul(g53c_tensor(&l->iwq), ch->sc.qa, 0, ch->iq, 0, n) &&
             vkc_matmul(g53c_tensor(&l->iwk), ch->nrm, 0, ch->ikd, 0, n) &&
             vkc_mla_lnorm(ch->ikd, ch->prm, ch->ik[i], &ln) &&
             vkc_matmul(g53c_tensor(&l->ikpg), ch->nrm, 0, ch->igd, 0, n) &&
             vkc_copy(ch->ig[i], (size_t)pb * ID, ch->igd, 0, (size_t)n * ID) &&
             vkc_matmul(g53c_tensor(&l->iwp), ch->nrm, 0, ch->ihw, 0, n) &&
             vkc_copy(ch->kvd, ko + (size_t)n * K, ch->ik[i], (size_t)pb * ID, (size_t)n * ID) &&
             vkc_copy(ch->kvd, ko + (size_t)n * (K + ID), ch->igd, 0, (size_t)n * ID) &&
             vkc_dsa_pool_keys(ch->ik[i], ch->ig[i], ch->prm, ch->pk[i], &kp) &&
             vkc_dsa_pool_select(ch->iq, ch->ihw, ch->pk[i], ch->isc, ch->sel, &sp) &&
             vkc_mla_attn(&ch->mla[i], &ch->sc, n, pb, 0, &ch->kv[i], ch->sel, 0, 1 + width, NULL, 0, ch->br, 0);
    if (ok && done > ch->pool_valid[i]) ch->pool_valid[i] = done;
    return ok;
}

/* The device is gone. If it held s's newest KDA state (dev: the state the forward from
 * `start` began with was the device's only), the state is rebuilt on the CPU from the
 * recorded input rows of [rec_base, start), from the host's copy at rec_base. */
static void g53c_recover(GModel *m, GSession *s, int start, int dev) {
    G53Chain *ch = g_g53c;
    g_vk_chain = 0;
    if (!ch) return;
    ch->failed = 1;
    int had = ch->owner == s && dev;
    ch->owner = NULL; ch->where = G53C_HOST;
    if (!had) {
        fprintf(stderr, "[VK] glm53 chain: the device was lost; the host's state is current, the CPU runs from here on\n");
        return;
    }
    const Cfg *c = &m->c; int H = c->hc_mult, D = c->hidden;
    if (!ch->rec_ok || ch->rec_base + ch->rec_len < start) {
        fprintf(stderr, "[VK] glm53 chain: the device was lost with a KDA state its recorded rows do not describe -- stopping "
                        "(COLI_VK_CHAIN=0 keeps the state on the CPU)\n");
        exit(1);
    }
    int n = start - ch->rec_base;
    fprintf(stderr, "[VK] glm53 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", n);
    if (n <= 0) return;
    float *xs = malloc((size_t)n * H * D * sizeof(float)), *xn = malloc((size_t)n * H * D * sizeof(float));
    if (!xs || !xn) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    for (int t = 0; t < n; t++) for (int h = 0; h < H; h++)
        memcpy(xs + ((size_t)t * H + h) * D, ch->rec + (size_t)t * D, (size_t)D * sizeof(float));
    run_layers(m, s, xs, xn, n, ch->rec_base, 0, c->n_layers);
    free(xs); free(xn);
}

/* Every layer for n rows of `streams` (the session's positions start..), the final
 * streams back into it. 0 = not taken: the CPU runs the layers (streams untouched). */
static int g53c_forward(GModel *m, GSession *s, float *streams, int n, int start) {
    G53Chain *ch = g_g53c;
    if (!g_vk_chain || !ch || !ch->ok || ch->failed) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && n <= 2) return 0;
    const Cfg *c = &m->c; int L = c->n_layers, D = c->hidden, H = c->hc_mult, HD = H * D;
    if (ch->owner != s) {                       /* another session takes the device */
        if (ch->owner) g53c_sync_host(m, ch->owner);
        ch->owner = s; ch->where = G53C_HOST;
        for (int i = 0; i < L; i++) ch->kv_valid[i] = ch->pool_valid[i] = 0;
    }
    int CH = g53c_rows(), rows = n < CH ? n : CH, dev0 = ch->where == G53C_DEV;
    if (vkc_lost()) { g53c_recover(m, s, start, dev0); return 0; }
    if (!g53c_mirror(ch, m, start + n, s->cap) || !g53c_scratch(ch, m, rows, start + n)) {
        fprintf(stderr, "[VK] glm53 chain: device memory for %d rows at %d positions refused; per-matrix path\n", rows, start + n);
        g53c_cpu_step(m, s, start);
        ch->failed = 1; g_vk_chain = 0;
        return 0;
    }
    /* the input rows, for a rebuild should the device go: from the position where the
     * host's KDA state is current (here, when the device takes it up now) */
    if (ch->where == G53C_HOST) g53c_rec_reset(ch, start);
    else if (ch->rec_ok && start != ch->rec_base + ch->rec_len) {
        if (start < ch->rec_base + ch->rec_len && start >= ch->rec_base) ch->rec_len = start - ch->rec_base;
        else ch->rec_ok = 0;
    }
    if (ch->rec_ok) {
        if (ch->rec_len + n > ch->rec_cap) {
            int nc = ch->rec_cap ? ch->rec_cap : 256; while (nc < ch->rec_len + n) nc *= 2;
            float *r = realloc(ch->rec, (size_t)nc * D * sizeof(float));
            if (!r) ch->rec_ok = 0; else { ch->rec = r; ch->rec_cap = nc; }
        }
        if (ch->rec_ok) {
            for (int t = 0; t < n; t++) memcpy(ch->rec + (size_t)(ch->rec_len + t) * D, streams + (size_t)t * HD, (size_t)D * sizeof(float));
            ch->rec_len += n;
        }
    }
    vkc_gemm_rows(-1);
    /* the final streams wait here until every chunk is through: a lost device leaves
     * the caller's rows the forward's input, for the CPU to run again */
    float *outs = malloc((size_t)n * HD * sizeof(float));
    if (!outs) return 0;
    for (int c0 = 0; c0 < n; c0 += rows) {
        int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0;
        if (!vkc_begin() || !g53c_push(ch, m, s, pb) || !vkc_write(ch->xs, 0, streams + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)))
            goto lost;
        int ok = 1, pending = 0, pulled = 0;
        for (int i = 0; i < L && ok; i++) {
            const GLayer *l = &m->layer[i];
            if (pending) {                       /* the FFN branch of the layer before: shared + routed, written back */
                VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = vkc_ew(ch->br, ch->ds, ch->routed, NULL, NULL, &add) && g53c_post(ch, m, nr);
                pending = 0;
            }
            double ta = now_s();
            ok = ok && g53c_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->o_in[i], nr) &&
                 (c->is_full[i] ? g53c_mla(ch, m, l, i, nr, pb) : g53c_kda(ch, m, l, i, nr)) &&
                 g53c_post(ch, m, nr) &&
                 g53c_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->o_post[i], nr);
            if (!ok) break;
            if (i < c->first_dense) {
                ok = g53c_mlp(ch, m, &l->dg, &l->du, &l->dd, ch->br, nr) && g53c_post(ch, m, nr);
                m->t_attn += now_s() - ta;
                continue;
            }
            ok = vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && vkc_submit(1);   /* A1 */
            m->t_attn += now_s() - ta;
            if (!ok) break;
            g53c_pull(ch, m, s, pulled, i + 1, pb, nr); pulled = i + 1;
            /* A2: the shared expert, while the host computes the routed experts */
            ok = vkc_begin() && g53c_mlp(ch, m, &l->rg, &l->ru, &l->rd, ch->ds, nr) && vkc_submit(0);
            double t1 = now_s();
            ffn_layer_ex(m, l, i, (const float *)vkc_ptr(ch->h2d), nr, ch->host_routed, 0);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
            double dt = now_s() - t1; ch->host_ms += dt * 1e3; m->t_ffn += dt;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) {
            VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = vkc_ew(ch->br, ch->ds, ch->routed, NULL, NULL, &add) && g53c_post(ch, m, nr);
        }
        ok = ok && vkc_copy(ch->xd, 0, ch->xs, 0, (size_t)nr * HD) && vkc_submit(1);
        if (!ok) goto lost;
        g53c_pull(ch, m, s, pulled, L, pb, nr);
        memcpy(outs + (size_t)c0 * HD, vkc_ptr(ch->xd), (size_t)nr * HD * sizeof(float));
        for (int i = 0; i < L; i++) if (c->is_full[i]) ch->kv_valid[i] = pb + nr;
        ch->where = G53C_DEV;
    }
    memcpy(streams, outs, (size_t)n * HD * sizeof(float));
    free(outs);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    free(outs);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    g53c_recover(m, s, start, dev0);
    return 0;
}

static void g53c_report(void) {
    G53Chain *ch = g_g53c;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] glm53 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}

/* COLI_VK_CHAIN at startup, before the tier sizes itself: the decision, the pipelines, the tensors. */
static void g53c_start(GModel *m) {
    if (!g_vk_ready) return;
    const Cfg *c = &m->c;
    int tier_on = vkt_wanted() && m->streaming && c->n_experts > 0 && c->swiglu_limit > 0.f;
    int on = coli_vk_chain_decide(NULL, tier_on, COLI_VK_CHAIN_OFF);
    const char *e = getenv("COLI_VK_CHAIN");
    char why[160];
    if (e && *e) snprintf(why, sizeof why, "COLI_VK_CHAIN=%s", e);
    else if (coli_vk_device_integrated())
        snprintf(why, sizeof why, "an integrated GPU: not measured on a GLM-5.3 checkpoint; COLI_VK_CHAIN=1 on, 2 prompts only");
    else if (coli_vk_device_shares_ram()) snprintf(why, sizeof why, "a CPU device; COLI_VK_CHAIN=1 turns it on");
    else snprintf(why, sizeof why, "a discrete GPU%s", tier_on ? ", beside the expert tier" : "");
    const char *no = NULL;
    if (on && !(g_g53c_inited = vkc_init())) no = "the chain's pipelines did not come up";
    if (on && !no && !(vkc_mla_ready() && vkc_kda_ready() && vkc_mhc_ready()))
        no = "the MLA, KDA or mHC shaders are missing";
    fprintf(stderr, "[VK] glm53: dense chain %s (%s%s%s)\n",
            on && !no ? (on == COLI_VK_CHAIN_PREFILL ? "on for prompts" : "on") : "off", why, no ? "; " : "", no ? no : "");
    if (!on || no) return;
    if (!g53c_setup(m)) return;
    g_vk_chain = on;
}
/* After the tier's: at exit the chain goes before the device. */
static void g53c_atexit(void) {
    if (g_g53c_inited) atexit(vkc_shutdown);
}
