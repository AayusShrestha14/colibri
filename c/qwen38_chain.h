/* qwen38_chain.h -- Qwen3.8 Flash Next's layers as a dense chain on the Vulkan device
 * (vk_chain.h). Included once by qwen38_core.h in a COLI_VULKAN build, after the CPU
 * forward it stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide).
 *
 * The same shape as qwen36_chain.h, with what Qwen3.8 adds:
 *   - four hyper-connection streams (hyper, S x 4H) stay on the device; every block
 *     reads them through its gated residual (per-stream norm, the low-rank down/up
 *     pair, the stream mix, the inject weights) and writes back hyper += inject x block;
 *   - the PLE layer: the n-gram table rows stay on the host (disk reads, in token
 *     order, with the n-gram history); only their rows go up, then key and value
 *     projections, the gate and the dilated convolution with its ring on the device;
 *   - QSA attention: the K/V/index-key rows on the device (the host's copies kept
 *     canonical as in qwen36), each block's pooled key computed once on the device
 *     when a step completes the block, and the indexer's top-k selection per query
 *     row on the device (chain_qsa.comp), feeding the attention's selection list;
 *   - the MTP head stays on the CPU (its experts are FP8 beside the int4 sidecar,
 *     and it reads two rows per draft): the chain hands it every row's four streams
 *     when the model has one; a verify forward (S = 2) runs here row-wise (GEMV for
 *     every matrix, so its rows get a decode step's bits) and snapshots the DeltaNet
 *     state, the conv rings and the PLE ring after its first row on the device, so a
 *     rejected draft rolls back by swapping device buffers (q38c_rollback).
 * Per layer the host gets the MoE input rows and the router logits, and the K/V and
 * index-key rows of an attention layer; it sends the routed sum back. The shared
 * expert runs on the device (frame A2) while the host computes the routed experts.
 *
 * State ownership as in qwen36: the host's KV/index caches canonical, device mirrors
 * behind a watermark (kv_valid), pooled block keys behind pk_valid; the DeltaNet
 * state, conv rings and PLE ring on the device while the chain runs (dn_where),
 * brought back before the host reads them (prefix cache, pins) and pushed up after
 * the host writes them (reset, restore). The CUDA tier keeps its priority: with it
 * on, the chain stays off. A device lost while it holds that state: the engine rebuilds
 * it on the CPU from the prefix record and continues there (q38c_recover). */
#include "vk_chain.h"

#define Q38C_HOST 0
#define Q38C_DEV  1
#define Q38C_BOTH 2

typedef struct {
    int ok, failed, rows, cap;
    VkcBuf *prm;
    size_t *o_an, *o_mn, *o_qn, *o_kn, *o_iqn, *o_ikn, *o_conv, *o_dn;
    size_t o_fn, o_ple, o_pconv;
    ColiVkTensor **t_sg;
    VkcBuf **rec, **ring, **rec_snap, **ring_snap, **kc, **vc, **ik, **pk;
    VkcBuf *ple_ring;                 /* [2][W][SL]: current at ple_cur, the verify's snapshot at the other */
    int ple_cur;
    int *kv_valid, *pk_valid, *attn_ord, n_attn;
    int dn_where, host_zero, snap_valid;
    VkcBuf *hyper, *hn, *low, *mix, *mixed, *inj_a, *inj_m, *blk, *q, *k, *v, *ip, *ctx, *qsc, *sel;
    VkcBuf *qkv, *z, *ab, *cv, *dny, *lg, *gs, *us, *hs, *ds, *sgd, *fin;
    VkcBuf *emb, *keys, *val, *gated, *normv;
    VkcBuf *mixd, *lgd, *kvd, *outd, *hypd, *find, *routed, *cs, *pcs;
    float *host_routed, *host_emb;
    unsigned long long forwards;
    double host_ms;
} Q38Chain;

static int g_vk_chain = 0;
static int q38c_chunk_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}
static void q38c_fatal(const char *what) {
    fprintf(stderr, "[VK] qwen38 chain: %s -- stopping (COLI_VK_CHAIN=0 keeps the state on the CPU)\n", what);
    exit(1);
}
static int q38c_geometry_ok(const Cfg *c) {
    int attn = 0, dn = 0;
    for (int i = 0; i < c->layers; i++) { if (c->is_attn[i]) attn = 1; else dn = 1; }
    if (attn && (c->head_dim > 256 || c->q_heads % c->kv_heads || (c->rotary_dim & 1) || c->rotary_dim > c->head_dim ||
                 c->idx_kheads != 1 || c->idx_dim > 256 || c->rotary_dim > c->idx_dim || c->idx_ratio < 1)) return 0;
    if (dn && (c->dn_vdim > 128 || c->dn_kdim > 256 || c->dn_convk < 2 || c->dn_convk > 9 || c->dn_vheads % c->dn_kheads)) return 0;
    if (c->ple_layer >= 0 && c->ple_layer < c->layers && (c->ple_convk - 1) * c->ngram_size > 32) return 0;
    return c->hc_count > 0 && c->hc_width == c->hc_count * c->hidden;
}
static ColiVkTensor *q38c_t(const Q38Weight *w) { return q38_vk_tensor(w); }

static Q38Chain *q38c_setup(Model *m) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (ch) return ch->ok ? ch : NULL;
    ch = (Q38Chain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->vkchain = ch;
    Cfg *c = &m->c; int L = c->layers, H = c->hidden, W = c->hc_width;
    if (m->range_begin != 0 || m->range_end != L || !m->lm_head.rows || !q38c_geometry_ok(c)) {
        fprintf(stderr, "[VK] qwen38 chain: a model or geometry its shaders do not take; per-matrix path\n");
        return NULL;
    }
#define Q38C_ARR(f, t) if (!(ch->f = (t *)calloc((size_t)L, sizeof(t)))) return NULL
    Q38C_ARR(o_an, size_t); Q38C_ARR(o_mn, size_t); Q38C_ARR(o_qn, size_t); Q38C_ARR(o_kn, size_t);
    Q38C_ARR(o_iqn, size_t); Q38C_ARR(o_ikn, size_t); Q38C_ARR(o_conv, size_t); Q38C_ARR(o_dn, size_t);
    Q38C_ARR(t_sg, ColiVkTensor *); Q38C_ARR(rec, VkcBuf *); Q38C_ARR(ring, VkcBuf *); Q38C_ARR(rec_snap, VkcBuf *);
    Q38C_ARR(ring_snap, VkcBuf *); Q38C_ARR(kc, VkcBuf *); Q38C_ARR(vc, VkcBuf *); Q38C_ARR(ik, VkcBuf *); Q38C_ARR(pk, VkcBuf *);
    Q38C_ARR(kv_valid, int); Q38C_ARR(pk_valid, int); Q38C_ARR(attn_ord, int);
#undef Q38C_ARR
    int VH = c->dn_vheads, CD = c->dn_conv_dim, CK = c->dn_convk;
    int ple = c->ple_layer >= 0 && c->ple_layer < L;
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_an[i] = n; n += W; ch->o_mn[i] = n; n += W;
        if (c->is_attn[i]) {
            ch->attn_ord[i] = ch->n_attn++;
            ch->o_qn[i] = n; n += c->head_dim; ch->o_kn[i] = n; n += c->head_dim;
            ch->o_iqn[i] = n; n += c->idx_dim; ch->o_ikn[i] = n; n += c->idx_dim;
        } else {
            ch->o_conv[i] = n; n += (size_t)CD * CK;
            ch->o_dn[i] = n; n += 2 * (size_t)VH + c->dn_vdim;
        }
    }
    ch->o_fn = n; n += W;
    if (ple) { ch->o_ple = n; n += 3 * (size_t)W; ch->o_pconv = n; n += (size_t)W * c->ple_convk; }
    float *a = calloc(n, sizeof(float));
    if (!a) return NULL;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_gr.norm, W * sizeof(float));
        memcpy(a + ch->o_mn[i], l->mlp_gr.norm, W * sizeof(float));
        if (c->is_attn[i]) {
            memcpy(a + ch->o_qn[i], l->qn, c->head_dim * sizeof(float)); memcpy(a + ch->o_kn[i], l->kn, c->head_dim * sizeof(float));
            memcpy(a + ch->o_iqn[i], l->idx_qn, c->idx_dim * sizeof(float)); memcpy(a + ch->o_ikn[i], l->idx_kn, c->idx_dim * sizeof(float));
        } else {
            memcpy(a + ch->o_conv[i], l->dn_conv, (size_t)CD * CK * sizeof(float));
            memcpy(a + ch->o_dn[i], l->dn_alog, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + VH, l->dn_dtbias, VH * sizeof(float));
            memcpy(a + ch->o_dn[i] + 2 * VH, l->dn_norm, c->dn_vdim * sizeof(float));
        }
    }
    memcpy(a + ch->o_fn, m->final_gr.norm, W * sizeof(float));
    if (ple) {
        Layer *pl = &m->L[c->ple_layer];
        memcpy(a + ch->o_ple, pl->ple_norm_key, W * sizeof(float));
        memcpy(a + ch->o_ple + W, pl->ple_norm_query, W * sizeof(float));
        memcpy(a + ch->o_ple + 2 * (size_t)W, pl->ple_norm_conv, W * sizeof(float));
        memcpy(a + ch->o_pconv, pl->ple_conv, (size_t)W * c->ple_convk * sizeof(float));
    }
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    if (!ok) return NULL;
    size_t nr = (size_t)VH * c->dn_kdim * c->dn_vdim, nc = (size_t)CD * (CK - 1);
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        ok = q38c_t(&l->attn_gr.down) && q38c_t(&l->attn_gr.up) && q38c_t(&l->attn_gr.inject) &&
             q38c_t(&l->mlp_gr.down) && q38c_t(&l->mlp_gr.up) && q38c_t(&l->mlp_gr.inject) &&
             q38c_t(&l->router) && q38c_t(&l->sh_g) && q38c_t(&l->sh_u) && q38c_t(&l->sh_d);
        if (ok) { ColiVkTensor *t = NULL; ok = coli_vk_tensor_ensure(&t, l->sh_gate, NULL, 10, H, 1, 0); ch->t_sg[i] = t; }
        if (ok && c->is_attn[i]) ok = q38c_t(&l->q) && q38c_t(&l->k) && q38c_t(&l->v) && q38c_t(&l->o) && q38c_t(&l->idx_qk);
        else if (ok) {
            ok = q38c_t(&l->dn_qkv) && q38c_t(&l->dn_z) && q38c_t(&l->dn_b) && q38c_t(&l->dn_a) && q38c_t(&l->dn_out) &&
                 (ch->rec[i] = vkc_buf(nr * 4, VKC_DEV)) && (ch->ring[i] = vkc_buf(nc * 4, VKC_DEV)) &&
                 (ch->rec_snap[i] = vkc_buf(nr * 4, VKC_DEV)) && (ch->ring_snap[i] = vkc_buf(nc * 4, VKC_DEV));
        }
    }
    ok = ok && q38c_t(&m->final_gr.down) && q38c_t(&m->final_gr.up) && q38c_t(&m->lm_head);
    if (ok && ple) {
        Layer *pl = &m->L[c->ple_layer];
        ok = q38c_t(&pl->ple_key) && q38c_t(&pl->ple_value) &&
             (ch->ple_ring = vkc_buf((size_t)2 * W * (c->ple_convk - 1) * c->ngram_size * 4, VKC_DEV));
    }
    if (!ok) { fprintf(stderr, "[VK] qwen38 chain: a matrix did not reach the device; per-matrix path\n"); return NULL; }
    ch->dn_where = Q38C_HOST;
    ch->ok = 1;
    fprintf(stderr, "[VK] qwen38 chain: %d layers on the device (%d QSA), %.1f MiB of parameters\n", L, ch->n_attn, n * 4 / 1048576.0);
    return ch;
}

static int q38c_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }
static int q38c_selrow(const Cfg *c) { return 1 + c->idx_budget + c->idx_ratio - 1; }
static size_t q38c_kv_stride(Q38Chain *ch, const Cfg *c) {   /* per attention layer in kvd: K, V, IK rows */
    return (size_t)ch->rows * (2 * (size_t)c->kv_heads * c->head_dim + c->idx_dim);
}
static int q38c_scratch(Q38Chain *ch, Model *m, int rows) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    int QH = c->q_heads, D = c->head_dim, kvo = c->kv_heads * D, ipw = (c->idx_qheads + 1) * c->idx_dim;
    int V = c->dn_vheads * c->dn_vdim, E = c->experts, SI = c->shared_inter;
    int Ep = c->ngram_heads * c->ngram_head_dim; if (Ep < 1) Ep = 1;
    size_t r = (size_t)rows;
    int need_rows = ch->rows < rows;
    if (need_rows) ch->rows = rows;   /* kvd's stride follows */
    int nbmax = m->kv_cap / (c->idx_ratio > 0 ? c->idx_ratio : 1) + 1;
    int ok = q38c_res(&ch->hyper, r * W, VKC_DEV) && q38c_res(&ch->hn, r * W, VKC_DEV) && q38c_res(&ch->low, r * R, VKC_DEV) &&
             q38c_res(&ch->mix, r * W, VKC_DEV) && q38c_res(&ch->mixed, r * H, VKC_DEV) && q38c_res(&ch->inj_a, r * C, VKC_DEV) &&
             q38c_res(&ch->inj_m, r * C, VKC_DEV) && q38c_res(&ch->blk, r * H, VKC_DEV) && q38c_res(&ch->q, r * QH * 2 * D, VKC_DEV) &&
             q38c_res(&ch->k, r * kvo, VKC_DEV) && q38c_res(&ch->v, r * kvo, VKC_DEV) && q38c_res(&ch->ip, r * ipw, VKC_DEV) &&
             q38c_res(&ch->ctx, r * QH * D, VKC_DEV) && q38c_res(&ch->qsc, r * 2 * (size_t)nbmax, VKC_DEV) &&
             q38c_res(&ch->sel, r * q38c_selrow(c), VKC_DEV) && q38c_res(&ch->qkv, r * c->dn_conv_dim, VKC_DEV) &&
             q38c_res(&ch->z, r * V, VKC_DEV) && q38c_res(&ch->ab, r * 2 * c->dn_vheads, VKC_DEV) &&
             q38c_res(&ch->cv, r * c->dn_conv_dim, VKC_DEV) && q38c_res(&ch->dny, r * V, VKC_DEV) &&
             q38c_res(&ch->lg, r * E, VKC_DEV) && q38c_res(&ch->gs, r * SI, VKC_DEV) && q38c_res(&ch->us, r * SI, VKC_DEV) &&
             q38c_res(&ch->hs, r * SI, VKC_DEV) && q38c_res(&ch->ds, r * H, VKC_DEV) && q38c_res(&ch->sgd, r, VKC_DEV) &&
             q38c_res(&ch->fin, r * H, VKC_DEV) && q38c_res(&ch->keys, r * W, VKC_DEV) && q38c_res(&ch->val, r * H, VKC_DEV) &&
             q38c_res(&ch->gated, r * W, VKC_DEV) && q38c_res(&ch->normv, r * W, VKC_DEV) &&
             q38c_res(&ch->mixd, r * H, VKC_DOWN) && q38c_res(&ch->lgd, r * E, VKC_DOWN) &&
             q38c_res(&ch->kvd, (size_t)(ch->n_attn ? ch->n_attn : 1) * q38c_kv_stride(ch, c), VKC_DOWN) &&
             q38c_res(&ch->outd, 2 * (size_t)c->vocab, VKC_DOWN) && q38c_res(&ch->routed, r * H, VKC_UP) &&
             q38c_res(&ch->emb, r * Ep, VKC_UP) && q38c_res(&ch->cs, r * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP) &&
             q38c_res(&ch->pcs, (size_t)nbmax * (c->rotary_dim > 0 ? c->rotary_dim : 2), VKC_UP);
    if (!ok) return 0;
    if (need_rows) {
        float *hr = realloc(ch->host_routed, r * H * sizeof(float)), *he;
        if (!hr) return 0;
        ch->host_routed = hr;
        he = realloc(ch->host_emb, r * Ep * sizeof(float));
        if (!he) return 0;
        ch->host_emb = he;
    }
    if (ch->cap != m->kv_cap) {
        for (int i = 0; i < c->layers; i++) {
            if (!c->is_attn[i]) continue;
            vkc_free(ch->kc[i]); vkc_free(ch->vc[i]); vkc_free(ch->ik[i]); vkc_free(ch->pk[i]);
            ch->kv_valid[i] = ch->pk_valid[i] = 0;
            ch->kc[i] = vkc_buf((size_t)kvo * m->kv_cap * 4, VKC_DEV);
            ch->vc[i] = vkc_buf((size_t)kvo * m->kv_cap * 4, VKC_DEV);
            ch->ik[i] = vkc_buf((size_t)c->idx_dim * m->kv_cap * 4, VKC_DEV);
            ch->pk[i] = vkc_buf((size_t)c->idx_dim * nbmax * 4, VKC_DEV);
            if (!ch->kc[i] || !ch->vc[i] || !ch->ik[i] || !ch->pk[i]) { ch->cap = 0; return 0; }
        }
        ch->cap = m->kv_cap;
    }
    return 1;
}

/* ---- state between the host and the device ------------------------------------ */
static size_t q38c_ple_cells(const Cfg *c) { return (size_t)c->hc_width * (c->ple_convk - 1) * c->ngram_size; }
static void q38_layer_forward(Model *m,int i,float *hyper,const int *ids,int S,int pos_base,
                              float *mixed,float *inject,float *block);
static void q38_embed_row(Model *m,int id,int abs_pos,float *out);
/* The device was lost with the newest DeltaNet state, conv rings and PLE ring on it.
 * The host's K/V/index rows are canonical, those are not: rebuild them on the CPU by
 * running the `upto` positions the prefix record names (the PLE n-gram history replayed
 * with them), then leave the chain off. The MTP head's own rows live on the host. */
static void q38c_recover(Model *m, int upto) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count;
    g_vk_chain = 0;
    if (ch) { ch->failed = 1; ch->dn_where = Q38C_HOST; ch->host_zero = 0; ch->snap_valid = 0; }
    for (int i = 0; i < c->layers; i++) {
        if (c->is_attn[i]) continue;
        memset(m->DN_rec[i], 0, (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim * sizeof(float));
        memset(m->DN_conv[i], 0, (size_t)c->dn_conv_dim * (c->dn_convk - 1) * sizeof(float));
    }
    if (m->PLE_conv_state) memset(m->PLE_conv_state, 0, q38c_ple_cells(c) * sizeof(float));
    if (m->ple_history) m->ple_history_len = 0;
    if (upto <= 0) return;
    if (m->kvp.tainted || m->kvp.len < upto || !m->kvp.fed)
        q38c_fatal("the device was lost with a recurrent state its token ids do not describe (an image)");
    fprintf(stderr, "[VK] qwen38 chain: the device was lost; rebuilding the state of %d positions on the CPU, "
                    "which runs from here on\n", upto);
    int *ids = (int *)malloc((size_t)upto * sizeof(int));
    float *hyper = falloc((int64_t)upto * W), *mixed = falloc((int64_t)upto * H);
    float *inject = falloc((int64_t)upto * C), *block = falloc((int64_t)upto * H);
    if (!ids) { fprintf(stderr, "OOM rebuilding the state\n"); exit(1); }
    memcpy(ids, m->kvp.fed, (size_t)upto * sizeof(int));
    for (int s = 0; s < upto; s++) {
        float *e = hyper + (int64_t)s * W;
        q38_embed_row(m, ids[s], s, e);
        for (int b = 1; b < C; b++) memcpy(e + (int64_t)b * H, e, (size_t)H * sizeof(float));
    }
    /* the step being run keeps its prefetched n-gram rows and its snapshot request */
    float *pref = m->ple_pref; int pref_rows = m->ple_pref_rows, snap = m->snap_after, rowwise = g_q38_rowwise;
    m->ple_pref = NULL; m->ple_pref_rows = 0; m->snap_after = 0; g_q38_rowwise = 0;
    for (int i = 0; i < c->layers; i++) q38_layer_forward(m, i, hyper, ids, upto, 0, mixed, inject, block);
    free(m->ple_pref);
    m->ple_pref = pref; m->ple_pref_rows = pref_rows; m->snap_after = snap; g_q38_rowwise = rowwise;
    free(ids); free(hyper); free(mixed); free(inject); free(block);
}
static void q38c_sync_host(Model *m) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (!ch || !ch->ok || ch->dn_where != Q38C_DEV) return;
    Cfg *c = &m->c;
    size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
    for (int i = 0; i < c->layers; i++) {
        if (c->is_attn[i]) continue;
        if (!vkc_read(ch->rec[i], 0, m->DN_rec[i], nr * 4) || !vkc_read(ch->ring[i], 0, m->DN_conv[i], nc * 4)) {
            q38c_recover(m, m->kv_len); return;
        }
    }
    if (ch->ple_ring && m->PLE_conv_state &&
        !vkc_read(ch->ple_ring, (size_t)ch->ple_cur * q38c_ple_cells(c), m->PLE_conv_state, q38c_ple_cells(c) * 4)) {
        q38c_recover(m, m->kv_len); return;
    }
    ch->dn_where = Q38C_BOTH;
}
static void q38c_host_wrote(Model *m, int zero) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (!ch || !ch->ok) return;
    ch->dn_where = Q38C_HOST; ch->host_zero = zero; ch->snap_valid = 0;
}
static void q38c_cpu_step(Model *m, int pos_base) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (!ch || !ch->ok) return;
    q38c_sync_host(m);
    ch->dn_where = Q38C_HOST; ch->host_zero = 0; ch->snap_valid = 0;
    for (int i = 0; i < m->c.layers; i++) {
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        int R = m->c.idx_ratio > 0 ? m->c.idx_ratio : 1;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;
    }
}
/* A rejected draft: the state after the verify's first row is the device's snapshot. */
static void q38c_rollback(Model *m) {
    Q38Chain *ch = (Q38Chain *)m->vkchain;
    if (!ch || !ch->ok || !ch->snap_valid || ch->dn_where != Q38C_DEV) return;
    for (int i = 0; i < m->c.layers; i++) {
        if (m->c.is_attn[i]) continue;
        VkcBuf *t = ch->rec[i]; ch->rec[i] = ch->rec_snap[i]; ch->rec_snap[i] = t;
        t = ch->ring[i]; ch->ring[i] = ch->ring_snap[i]; ch->ring_snap[i] = t;
    }
    if (ch->ple_ring) ch->ple_cur ^= 1;
    ch->snap_valid = 0;
}
static int q38c_push_state(Q38Chain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    if (ch->dn_where == Q38C_HOST) {
        size_t nr = (size_t)c->dn_vheads * c->dn_kdim * c->dn_vdim, nc = (size_t)c->dn_conv_dim * (c->dn_convk - 1);
        for (int i = 0; i < c->layers && ok; i++) {
            if (c->is_attn[i]) continue;
            if (ch->host_zero) ok = vkc_zero(ch->rec[i], 0, nr) && vkc_zero(ch->ring[i], 0, nc);
            else ok = vkc_write(ch->rec[i], 0, m->DN_rec[i], nr * 4) && vkc_write(ch->ring[i], 0, m->DN_conv[i], nc * 4);
        }
        if (ok && ch->ple_ring && m->PLE_conv_state) {
            size_t pc = q38c_ple_cells(c), off = (size_t)ch->ple_cur * pc;
            ok = ch->host_zero ? vkc_zero(ch->ple_ring, off, pc) : vkc_write(ch->ple_ring, off, m->PLE_conv_state, pc * 4);
        }
        ch->dn_where = Q38C_BOTH;
    }
    int D = c->head_dim, KVH = c->kv_heads, ID = c->idx_dim, R = c->idx_ratio;
    for (int i = 0; i < c->layers && ok; i++) {
        if (!c->is_attn[i]) continue;
        if (ch->pk_valid[i] > pos_base / R) ch->pk_valid[i] = pos_base / R;   /* blocks this step rewrites */
        if (ch->kv_valid[i] >= pos_base) continue;
        int t0 = ch->kv_valid[i], n = pos_base - t0;
        for (int h = 0; h < KVH && ok; h++) {
            size_t src = ((size_t)h * m->kv_cap + t0) * D, dst = ((size_t)h * ch->cap + t0) * D;
            ok = vkc_write(ch->kc[i], dst, m->K[i] + src, (size_t)n * D * 4) && vkc_write(ch->vc[i], dst, m->V[i] + src, (size_t)n * D * 4);
        }
        ok = ok && vkc_write(ch->ik[i], (size_t)t0 * ID, m->IK[i] + (size_t)t0 * ID, (size_t)n * ID * 4);
        ch->kv_valid[i] = pos_base;
        if (ch->pk_valid[i] > t0 / R) ch->pk_valid[i] = t0 / R;
    }
    return ok;
}

/* ---- pieces ---------------------------------------------------------------------- */
/* the gated residual's read (q38_gr_read): mixed [n][H], inject [n][C] when inj */
static int q38c_gr_read(Q38Chain *ch, Model *m, const GatedResidual *g, size_t norm_off, int n, VkcBuf *inj) {
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, C = c->hc_count, R = c->hc_rank;
    VkcNorm np = {n * C, H, C, 0, W, H, 0, W, H, (int)norm_off, C, VKC_NORM_ADD1, c->eps, 1.f};
    VkcEw lo = {VKC_EW_HC_LOW, n * R, R, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    VkcEw mx = {VKC_EW_HC_MIX, n * H, H, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
    int ok = vkc_norm(ch->hyper, ch->prm, ch->hn, &np) &&
             vkc_matmul(q38c_t(&g->down), ch->hn, 0, ch->low, 0, n) && vkc_ew(ch->low, ch->low, NULL, NULL, NULL, &lo) &&
             vkc_matmul(q38c_t(&g->up), ch->low, 0, ch->mix, 0, n) && vkc_ew(ch->mixed, ch->mix, ch->hn, NULL, NULL, &mx);
    if (ok && inj) {
        VkcEw ij = {VKC_EW_HC_INJ, n * C, C, C, 0, 1, 0, 0, 0, 0, 0, (float)C};
        ok = vkc_matmul(q38c_t(&g->inject), ch->hn, 0, inj, 0, n) && vkc_ew(inj, inj, NULL, NULL, NULL, &ij);
    }
    return ok;
}
static int q38c_gr_apply(Q38Chain *ch, Model *m, VkcBuf *inj, int n) {
    Cfg *c = &m->c;
    VkcEw ap = {VKC_EW_HC_APPLY, n * c->hc_width, c->hidden, c->hc_count, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->hyper, inj, ch->blk, NULL, NULL, &ap);
}
static int q38c_attention(Q38Chain *ch, Model *m, Layer *l, int i, int n, int pb) {
    Cfg *c = &m->c;
    int QH = c->q_heads, KVH = c->kv_heads, D = c->head_dim, kvo = KVH * D, qo = QH * 2 * D, half = c->rotary_dim / 2;
    int IQ = c->idx_qheads, ID = c->idx_dim, R = c->idx_ratio, ipw = (IQ + 1) * ID;
    int ok = vkc_matmul(q38c_t(&l->q), ch->mixed, 0, ch->q, 0, n) && vkc_matmul(q38c_t(&l->k), ch->mixed, 0, ch->k, 0, n) &&
             vkc_matmul(q38c_t(&l->v), ch->mixed, 0, ch->v, 0, n) && vkc_matmul(q38c_t(&l->idx_qk), ch->mixed, 0, ch->ip, 0, n);
    VkcNorm kn = {n * KVH, D, KVH, 0, kvo, D, 0, kvo, D, (int)ch->o_kn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcNorm qn = {n * QH, D, QH, 0, qo, 2 * D, 0, qo, 2 * D, (int)ch->o_qn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    VkcNorm in = {n * IQ, ID, IQ, 0, ipw, ID, 0, ipw, ID, (int)ch->o_iqn[i], 0, VKC_NORM_ADD1, c->eps, 1.f};
    ok = ok && vkc_norm(ch->k, ch->prm, ch->k, &kn) && vkc_norm(ch->q, ch->prm, ch->q, &qn) && vkc_norm(ch->ip, ch->prm, ch->ip, &in);
    if (ok && half) {
        VkcRope rk = {n * KVH, KVH, 0, kvo, D, half, 0, 2 * half}, rq = {n * QH, QH, 0, qo, 2 * D, half, 0, 2 * half};
        VkcRope ri = {n * IQ, IQ, 0, ipw, ID, half, 0, 2 * half};
        ok = vkc_rope(ch->k, ch->cs, &rk) && vkc_rope(ch->q, ch->cs, &rq) && vkc_rope(ch->ip, ch->cs, &ri);
    }
    if (!ok) return 0;
    /* the rows into the device caches and the host's */
    VkcRegion *rg = malloc(sizeof *rg * (size_t)n * (KVH > 1 ? KVH : 1));
    VkcRegion *ri = malloc(sizeof *ri * (size_t)n);
    if (!rg || !ri) { free(rg); free(ri); return 0; }
    for (int s = 0; s < n; s++) {
        for (int h = 0; h < KVH; h++)
            rg[s * KVH + h] = (VkcRegion){((size_t)h * ch->cap + pb + s) * D, (size_t)s * kvo + (size_t)h * D, (size_t)D};
        ri[s] = (VkcRegion){(size_t)(pb + s) * ID, (size_t)s * ipw + (size_t)IQ * ID, (size_t)ID};
    }
    size_t ko = (size_t)ch->attn_ord[i] * q38c_kv_stride(ch, c);
    ok = vkc_copy_regions(ch->kc[i], ch->k, rg, n * KVH) && vkc_copy_regions(ch->vc[i], ch->v, rg, n * KVH) &&
         vkc_copy_regions(ch->ik[i], ch->ip, ri, n) &&
         vkc_copy(ch->kvd, ko, ch->k, 0, (size_t)n * kvo) &&
         vkc_copy(ch->kvd, ko + (size_t)ch->rows * kvo, ch->v, 0, (size_t)n * kvo);
    for (int s = 0; s < n && ok; s++)   /* index keys back, compact */
        ok = vkc_copy(ch->kvd, ko + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ch->ip, (size_t)s * ipw + (size_t)IQ * ID, ID);
    free(rg); free(ri);
    /* the pooled keys of the blocks this step completes (and any behind the watermark) */
    int nb = (pb + n) / R, b0 = ch->pk_valid[i];
    if (ok && nb > b0) {
        float *pc = (float *)vkc_ptr(ch->pcs);
        if (half) for (int b = b0; b < nb; b++) for (int j = 0; j < half; j++) {
            float ang = (float)(b * R) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
            pc[((b - b0) * half + j) * 2] = cosf(ang); pc[((b - b0) * half + j) * 2 + 1] = sinf(ang);
        }
        VkcQsa pp = {0, ID, R, b0, half, 0, 0, 0, 0, 0, 0, 0, 0, c->eps, 0, (int)ch->o_ikn[i], 0, nb - b0};
        ok = vkc_qsa(ch->ik[i], ch->prm, ch->pk[i], ch->pcs, NULL, NULL, &pp);
        ch->pk_valid[i] = nb;
    }
    int nbmax = (int)(vkc_bytes(ch->qsc) / 4 / (2 * (size_t)ch->rows));
    VkcQsa ps = {1, ID, R, 0, 0, n, pb, c->idx_budget, IQ, 0, ipw, nbmax, q38c_selrow(c), c->eps, 0, 0, 0, 0};
    VkcAttn at = {n, QH, KVH, D, pb, ch->cap, 0, qo, 2 * D, D, qo, 2 * D, 1, 0, QH * D, 0, q38c_selrow(c),
                  1.f / sqrtf((float)D), 0, 0};
    return ok && vkc_qsa(ch->ip, NULL, ch->pk[i], NULL, ch->qsc, ch->sel, &ps) &&
           vkc_attn(ch->q, ch->kc[i], ch->vc[i], ch->ctx, ch->q, ch->sel, &at) &&
           vkc_matmul(q38c_t(&l->o), ch->ctx, 0, ch->blk, 0, n);
}
static int q38c_deltanet(Q38Chain *ch, Model *m, Layer *l, int i, int n, int snap_row) {
    Cfg *c = &m->c;
    int VH = c->dn_vheads, CD = c->dn_conv_dim, V = VH * c->dn_vdim;
    int ok = vkc_matmul(q38c_t(&l->dn_qkv), ch->mixed, 0, ch->qkv, 0, n) && vkc_matmul(q38c_t(&l->dn_z), ch->mixed, 0, ch->z, 0, n) &&
             vkc_matmul(q38c_t(&l->dn_b), ch->mixed, 0, ch->ab, 0, n) &&
             vkc_matmul(q38c_t(&l->dn_a), ch->mixed, 0, ch->ab, (size_t)ch->rows * VH, n);
    VkcDnConv cp = {n, CD, c->dn_convk, 0, CD, 0, CD, snap_row, 1, (int)ch->o_conv[i], 0, 0};
    VkcDnRec rp = {n, VH, c->dn_kheads, c->dn_vdim, c->dn_kheads * c->dn_kdim, 0, CD, 0, VH, ch->rows * VH, VH,
                   0, V, 0, V, snap_row, 1, c->eps, 1.f / sqrtf((float)c->dn_kdim), 0, 0, (int)ch->o_dn[i]};
    return ok && vkc_dnconv(ch->qkv, ch->prm, ch->ring[i], ch->cv, ch->ring_snap[i], &cp) &&
           vkc_dnrec(c->dn_kdim, ch->cv, ch->ab, ch->z, ch->rec[i], ch->prm, ch->dny, ch->rec_snap[i], &rp) &&
           vkc_matmul(q38c_t(&l->dn_out), ch->dny, 0, ch->blk, 0, n);
}
/* the n-gram rows of n tokens, in order, as q38_ple fills them (history and the verify's snapshot) */
static void q38c_ple_rows(Model *m, const int *ids, int c0, int n, float *emb) {
    Cfg *c = &m->c; int E = c->ngram_heads * c->ngram_head_dim;
    for (int r = 0; r < n; r++) {
        int s = c0 + r; float *e = emb + (size_t)r * E;
        int64_t p1 = m->ple_history_len >= 1 ? m->ple_history[m->ple_history_len - 1] : c->eos_id;
        int64_t p2 = m->ple_history_len >= 2 ? m->ple_history[m->ple_history_len - 2] : c->eos_id;
        if (m->ple_pref && s < m->ple_pref_rows)
            memcpy(e, m->ple_pref + (int64_t)s * c->ngram_heads * c->ngram_head_dim, (size_t)E * sizeof(float));
        else for (int h = 0; h < c->ngram_heads; h++) {
            int ng = h < c->heads_per_ngram ? 2 : 3;
            q38_ple_row(m, q38_hash_row(m, h, ng, ids[s], p1, p2), e + (int64_t)h * c->ngram_head_dim);
        }
        if (ids[s] == c->eos_id) m->ple_history_len = 0;
        else if (m->ple_history_len == 0) { m->ple_history[0] = ids[s]; m->ple_history_len = 1; }
        else if (m->ple_history_len == 1) { m->ple_history[1] = ids[s]; m->ple_history_len = 2; }
        else { m->ple_history[0] = m->ple_history[1]; m->ple_history[1] = ids[s]; }
        if (s + 1 == m->snap_after) {
            memcpy(m->snap_ple_history, m->ple_history, sizeof(m->snap_ple_history));
            m->snap_ple_history_len = m->ple_history_len;
        }
    }
}
static int q38c_ple(Q38Chain *ch, Model *m, int n, int snap_row) {
    Cfg *c = &m->c; Layer *l = &m->L[c->ple_layer];
    int W = c->hc_width, H = c->hidden, E = c->ngram_heads * c->ngram_head_dim;
    size_t pc = q38c_ple_cells(c);
    int ok = vkc_write(ch->emb, 0, ch->host_emb, (size_t)n * E * 4) &&
             vkc_matmul(q38c_t(&l->ple_key), ch->emb, 0, ch->keys, 0, n) &&
             vkc_matmul(q38c_t(&l->ple_value), ch->emb, 0, ch->val, 0, n);
    VkcPle g = {0, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, -1, 0, c->eps, (int)ch->o_ple, 0, 0};
    VkcPle cv = {1, n, c->hc_count, H, c->ple_convk, c->ngram_size, 0, 0, 0, snap_row,
                 (int)((size_t)(ch->ple_cur ^ 1) * pc), c->eps, 0, (int)ch->o_pconv, (int)((size_t)ch->ple_cur * pc)};
    (void)W;
    return ok && vkc_ple(ch->keys, ch->hyper, ch->val, ch->prm, ch->gated, ch->normv, NULL, NULL, &g) &&
           vkc_ple(NULL, ch->hyper, NULL, NULL, ch->gated, ch->normv, ch->prm, ch->ple_ring, &cv);
}
static int q38c_shared(Q38Chain *ch, Model *m, Layer *l, int i, int n) {
    Cfg *c = &m->c; int SI = c->shared_inter;
    VkcEw sw = {VKC_EW_SWIGLU, n * SI, SI, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_matmul(q38c_t(&l->sh_g), ch->mixed, 0, ch->gs, 0, n) && vkc_matmul(q38c_t(&l->sh_u), ch->mixed, 0, ch->us, 0, n) &&
           vkc_ew(ch->hs, ch->gs, ch->us, NULL, NULL, &sw) && vkc_matmul(q38c_t(&l->sh_d), ch->hs, 0, ch->ds, 0, n) &&
           vkc_matmul(ch->t_sg[i], ch->mixed, 0, ch->sgd, 0, n);
}
/* block = routed + sigmoid(gate) * shared (q38_moe's out), then hyper += inject x block */
static int q38c_moe_apply(Q38Chain *ch, Model *m, int n) {
    Cfg *c = &m->c;
    VkcEw cb = {VKC_EW_COMBINE, n * c->hidden, c->hidden, 1, 1 | 2 | 4 | 8, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(ch->blk, NULL, ch->routed, ch->ds, ch->sgd, &cb) && q38c_gr_apply(ch, m, ch->inj_m, n);
}

/* Every layer for S rows. hyper_h: the rows' streams in, the final ones out when
 * want_streams; mixed_h (or NULL): every row's final mixed row, for the prefill
 * read-out; logit: the last nlogits rows' logits. 0 = not taken (nothing changed). */
static int q38c_forward(Model *m, const int *ids, int S, int pos_base, int nlogits, float *hyper_h,
                        int want_streams, float *mixed_h, float *logit) {
    if (!g_vk_chain || qt_ready()) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode and verify on the CPU */
    Q38Chain *ch = q38c_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int H = c->hidden, W = c->hc_width, L = c->layers, E = c->experts, V = c->vocab;
    int rows = S < q38c_chunk_rows() ? S : q38c_chunk_rows();
    if (!q38c_scratch(ch, m, rows) || (want_streams && !q38c_res(&ch->hypd, (size_t)rows * W, VKC_DOWN)) ||
        (mixed_h && !q38c_res(&ch->find, (size_t)rows * H, VKC_DOWN))) {
        fprintf(stderr, "[VK] qwen38 chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    int kvo = c->kv_heads * c->head_dim, ID = c->idx_dim, half = c->rotary_dim / 2;
    vkc_gemm_rows(g_q38_rowwise ? 0 : -1);   /* a verify's rows get a decode step's bits */
    int snapped = 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0;
        int snap_row = m->snap_after > c0 && m->snap_after <= c0 + n ? m->snap_after - 1 - c0 : -1;
        if (snap_row >= 0) snapped = 1;
        if (!vkc_begin() || !vkc_write(ch->hyper, 0, hyper_h + (size_t)c0 * W, (size_t)n * W * 4) ||
            !q38c_push_state(ch, m, pb)) goto lost;
        if (half) {
            float *cs = (float *)vkc_ptr(ch->cs);
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float ang = (float)(pb + s) / powf(c->theta, (float)(2 * j) / c->rotary_dim);
                cs[(s * half + j) * 2] = cosf(ang); cs[(s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        if (c->ple_layer >= 0 && c->ple_layer < L) q38c_ple_rows(m, ids, c0, n, ch->host_emb);
        int ok = 1, pending = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) ok = q38c_moe_apply(ch, m, n);
            if (ok && i == c->ple_layer) ok = q38c_ple(ch, m, n, snap_row);
            ok = ok && q38c_gr_read(ch, m, &l->attn_gr, ch->o_an[i], n, ch->inj_a);
            ok = ok && (c->is_attn[i] ? q38c_attention(ch, m, l, i, n, pb) : q38c_deltanet(ch, m, l, i, n, snap_row));
            ok = ok && q38c_gr_apply(ch, m, ch->inj_a, n);
            ok = ok && q38c_gr_read(ch, m, &l->mlp_gr, ch->o_mn[i], n, ch->inj_m);
            ok = ok && vkc_matmul(q38c_t(&l->router), ch->mixed, 0, ch->lg, 0, n) &&
                 vkc_copy(ch->lgd, 0, ch->lg, 0, (size_t)n * E) && vkc_copy(ch->mixd, 0, ch->mixed, 0, (size_t)n * H);
            double t0 = now_s();
            ok = ok && vkc_submit(1);                                        /* A1 */
            m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0;
            if (!ok) break;
            if (c->is_attn[i]) {
                const float *kv = (const float *)vkc_ptr(ch->kvd) + (size_t)ch->attn_ord[i] * q38c_kv_stride(ch, c);
                for (int s = 0; s < n; s++) {
                    for (int h = 0; h < c->kv_heads; h++) {
                        size_t dst = ((size_t)h * m->kv_cap + pb + s) * c->head_dim, src = (size_t)s * kvo + (size_t)h * c->head_dim;
                        memcpy(m->K[i] + dst, kv + src, c->head_dim * sizeof(float));
                        memcpy(m->V[i] + dst, kv + (size_t)ch->rows * kvo + src, c->head_dim * sizeof(float));
                    }
                    memcpy(m->IK[i] + (size_t)(pb + s) * ID, kv + 2 * (size_t)ch->rows * kvo + (size_t)s * ID, ID * sizeof(float));
                }
            }
            ok = vkc_begin() && q38c_shared(ch, m, l, i, n) && vkc_submit(0);   /* A2 */
            double t1 = now_s();
            q38_moe_ex(m, l, i, (const float *)vkc_ptr(ch->mixd), n, ch->host_routed, (const float *)vkc_ptr(ch->lgd), 1);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)n * H * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok) ok = q38c_moe_apply(ch, m, n);
        if (ok && want_streams) ok = vkc_copy(ch->hypd, 0, ch->hyper, 0, (size_t)n * W);
        /* the final mixer for the rows that need it: the read-out wants every row, the
         * logits the last nlogits */
        int lo0 = S - nlogits > c0 ? S - nlogits - c0 : 0, lo_n = n - lo0;   /* this chunk's logit rows */
        int need_final = mixed_h || (lo0 < n && c0 + n > S - nlogits);
        if (ok && need_final) ok = q38c_gr_read(ch, m, &m->final_gr, ch->o_fn, n, NULL);
        if (ok && mixed_h) ok = vkc_copy(ch->find, 0, ch->mixed, 0, (size_t)n * H);
        if (ok && need_final && lo_n > 0 && c0 + n > S - nlogits) {
            int dst = c0 + lo0 - (S - nlogits);
            ok = vkc_matmul(q38c_t(&m->lm_head), ch->mixed, (size_t)lo0 * H, ch->outd, (size_t)dst * V, lo_n);
        }
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        m->timers.seconds[Q38_TM_DENSE_MATMUL] += now_s() - t0;
        if (!ok) goto lost;
        if (want_streams) memcpy(hyper_h + (size_t)c0 * W, vkc_ptr(ch->hypd), (size_t)n * W * 4);
        if (mixed_h) memcpy(mixed_h + (size_t)c0 * H, vkc_ptr(ch->find), (size_t)n * H * 4);
        for (int i = 0; i < L; i++) if (c->is_attn[i]) ch->kv_valid[i] = pb + n;
        ch->dn_where = Q38C_DEV; ch->host_zero = 0;
    }
    memcpy(logit, vkc_ptr(ch->outd), (size_t)nlogits * V * sizeof(float));
    ch->snap_valid = snapped;
    vkc_gemm_rows(-1);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    vkc_gemm_rows(-1);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    q38c_recover(m, pos_base);
    return 0;
}

static void q38c_report(Model *m) {
    Q38Chain *ch = m ? (Q38Chain *)m->vkchain : NULL;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] qwen38 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}
