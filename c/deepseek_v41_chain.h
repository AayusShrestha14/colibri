/* deepseek_v41_chain.h -- DeepSeek V4.1 Flash's layers as a dense chain on the Vulkan
 * device (vk_chain.h). Included once by deepseek_v41.c in a COLI_VULKAN build, before
 * forward_full, whose layer loop it stands in for; COLI_VK_CHAIN decides
 * (coli_vk_chain_decide, this engine's integrated GPU default COLI_VK_CHAIN_UNMEASURED:
 * off, not measured on a V4.1 checkpoint).
 *
 * The residual is hc_mult streams per position (mHC), and a site collapses them with the
 * mix the site BEFORE it computed (forward_full: pre_mix). What runs where, per layer, for
 * one block of rows (decode: one row; a prompt in chunks of COLI_VK_CHAIN_ROWS):
 *   device, frame A1: the previous layer's FFN branch (the routed sum from the host plus
 *     the shared expert) written back into the streams (mHC post); on an engram layer the
 *     engram (eng_wkv over the n-gram rows the host looked up, the gate into each stream);
 *     on a DSpark target layer the streams' mean for the draft head; the attention site:
 *     hc_attn_fn, the split with Sinkhorn, the collapse with the previous site's mix, the
 *     RMSNorm; the attention: wq_a, its norm, wq_b, wkv, its norm, RoPE on interleaved
 *     pairs, the new rows into the window ring; on a kv_source layer the compressor (its
 *     rolling group on the device, the pooled latent's norm), the index keys (idx_wk,
 *     norm, RoPE) and the compressed rows' RoPE; on an index source the indexer (idx_wq_b
 *     and its RoPE, weights_proj, the scores against the keys the CPU would read, the
 *     candidate blocks, the top-k); the sparse attention with the sink over the window
 *     and the selection, the inverse RoPE, the grouped wo_a and wo_b; mHC post; the FFN
 *     site's split, collapse and norm. Then the frame is waited for.
 *   host: the router and the routed experts (moe_run_at without the shared expert: the
 *     expert tier's batch and the CPU's share).
 *   device, frame A2 (not waited for): the shared expert (clamped SwiGLU).
 * The final streams and the last FFN site's mix come back to the host, which collapses
 * them and runs the final norm and the head as before.
 *
 * State, and who owns it: the host's stays canonical. Everything a forward changes there
 * (the window ring and its position map, the compressed rows and index keys, the
 * compressor's group, the published index keys) is written back once the forward's last
 * frame is through, as the CPU would have left it (a speculative verify's undo rows
 * included), so a lost device, a rollback, a reset or a CPU forward never has to rebuild
 * anything. The device mirrors it: the window ring holds window + chunk rows (so no row
 * of a chunk overwrites one an earlier row still reads) behind a watermark win_valid;
 * the compressed rows and index keys of each kv_source layer behind kv_valid[layer]; the
 * compressor's group goes up at every forward's start (two groups of head_dim floats).
 * Every host write lowers the watermarks: a CPU forward (v41c_cpu_step), a rejected
 * draft (v41c_rollback), a reset (v41c_reset). A lost device (COLI_VK_CHAIN_FAULT=n
 * fakes it): the forward runs again on the CPU from its input, which the chain leaves
 * untouched until every chunk is through, and the CPU runs from there.
 *
 * Past the device's budget (vk_kvsplit.h) each kv_source layer's compressed rows keep
 * only some blocks on the device: a window over the newest rows and the blocks the
 * indexer picks most; the rest stay in the host's ckv, which holds every row. A layer's
 * sparse attention stages only selected missing rows in bounded device scratch and
 * reads them alongside resident rows in the original list order (vkc_kv_ds), preserving
 * the complete device result regardless of residency. New compressed rows go through a
 * scratch into their slots, and come back into the host's ckv as soon as their layer's
 * frame is through, so the host part of a later chunk finds every row it reads there.
 * The window ring and the index keys stay whole on the device (every row reads them
 * every step). Below the budget the mirrors are whole, as above.
 *
 * The chain declines (the CPU runs the forward, the watermarks follow) under V41_TRACE,
 * for prompts only when COLI_VK_CHAIN=2 and the forward has two rows or fewer, and for a
 * model its shaders or this file do not take: head_dim above 1024, a window plus top-k
 * above 3072 entries, an indexer above 64 heads or 4096 query floats, more than 8 hyper-
 * connection streams, a compressed layer that is not an index source and reads the list
 * of an earlier index source with another ratio (or of none), a candidate-mask consumer
 * with another ratio than the candidate source. */
#include "vk_chain.h"
#include "vk_kvsplit.h"

typedef struct {
    int ok, failed;
    int rows;                                  /* scratch rows */
    int Wd, K, W, rmax;                        /* device ring rows, top-k slots, window, largest ratio */
    VkcBuf *prm;
    size_t *o_an, *o_fn, *o_qn, *o_kn, *o_sink, *o_hca, *o_hcf, *o_cn, *o_ikn, *o_eq, *o_ek;
    ColiVkTensor **fna, **fnf;                 /* the mHC mix matrices (f32) */
    VkcBuf **win, **ckv, **ikey, **ring;
    int *kv_valid, *ccap;
    int win_valid;
    VkcBuf *xs, *xn, *mix, *hpa, *hpf, *col, *nrm, *qa, *qr, *q, *kv, *heads, *grp, *br, *craw, *cscr;
    VkcBuf *iq, *ihw, *isc, *mask, *list, *cs, *erows, *ekv, *gs, *us, *hs, *ds;
    VkcBuf *h2d, *mh, *xd, *hpd, *pull, *routed;
    size_t wcap;                               /* score columns per row */
    float *host_routed;
    unsigned long long forwards;
    double host_ms;
    VkcKvSplit ks;                             /* the compressed rows split past the device's budget (ks.on) */
    int *sli, nsl, rmin, planned;              /* per layer its table (-1 none), tables, the least ratio */
    VkcBuf *cnew, *cdn;                        /* a chunk's new compressed rows, and their copy for the host */
} V41Chain;

static V41Chain *g_v41c;
static int g_vk_chain = 0;
static int g_v41c_inited = 0;

/* The window ring's extra rows at setup: COLI_VK_CHAIN_ROWS when set, else 512; a forward
 * whose chunk is bigger grows the rings first (v41c_grow_win). */
static int v41c_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e && strcmp(e, "auto") ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}
/* v41c_res counts instead of reserving while g_v41c_count >= 0 (the chunk's sizing) */
static long long g_v41c_count = -1;
static int v41c_res(VkcBuf **b, size_t floats, int kind) {
    if (g_v41c_count >= 0) { g_v41c_count += (long long)(floats ? floats : 1) * (long long)sizeof(float); return 1; }
    return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind);
}

/* The device copies the per-matrix path keeps (vk_entry), uploaded here if it has not:
 * fp8 as fmt 12 with each 32x32 tile's scale written out for its rows, bf16 as fmt 11. */
static ColiVkTensor *v41c_w8(const W8 *w) {
    if (!w->q || w->O < 1 || w->I < 1) return NULL;
    VkEntry *e = vk_entry(w->q, 12, w->O, w->I);
    if (!e || e->refused) return NULL;
    if (e->t) return e->t;
    v41_dho_host(w->q);   /* not placed this way (device only): its rows back first */
    int groups = (w->I + FP8_TILE - 1) / FP8_TILE;
    float *scales = malloc((size_t)w->O * groups * sizeof(float));
    if (!scales) return NULL;
    for (int o = 0; o < w->O; o++)
        for (int g = 0; g < groups; g++) scales[(size_t)o * groups + g] = ue8m0(w->s[(size_t)(o / FP8_TILE) * groups + g]);
    int ok = coli_vk_tensor_ensure(&e->t, w->q, scales, 12, w->I, w->O, FP8_TILE);
    free(scales);
    return ok ? e->t : NULL;
}
static ColiVkTensor *v41c_wb(const WB *w) {
    if (!w->w || w->O < 1 || w->I < 1) return NULL;
    VkEntry *e = vk_entry(w->w, 11, w->O, w->I);
    if (!e || e->refused) return NULL;
    if (e->t) return e->t;
    v41_dho_host(w->w);
    return coli_vk_tensor_ensure(&e->t, w->w, NULL, 11, w->I, w->O, 0) ? e->t : NULL;
}

/* ---- the host's writes: the device's copies follow ------------------------------------- */
static void v41c_lower(Model *m, int pos) {
    V41Chain *ch = g_v41c;
    if (!ch || !ch->ok) return;
    if (pos < 0) pos = 0;
    if (ch->win_valid > pos) ch->win_valid = pos;
    for (int i = 0; i < m->c.n_layers; i++) {
        int r = m->c.compress_ratio[i];
        if (m->c.kv_source[i] && r > 0 && ch->kv_valid[i] > pos / r) ch->kv_valid[i] = pos / r;
        if (m->c.kv_source[i] && r > 0 && ch->sli[i] >= 0) vkc_kv_lower(&ch->ks, ch->sli[i], pos / r);
    }
}
static void v41c_cpu_step(Model *m, int start_pos) { v41c_lower(m, start_pos); }      /* a CPU forward from start_pos */
static void v41c_rollback(Model *m, int committed_end) { v41c_lower(m, committed_end); }  /* rows past it rejected */
static void v41c_reset(Model *m) { v41c_lower(m, 0); }

/* ---- setup ----------------------------------------------------------------------------- */
static const char *v41c_unsupported(const Model *m) {
    const Cfg *c = &m->c;
    int L = c->n_layers, K = c->index_topk > 0 ? c->index_topk : 0;
    if (c->head_dim > 1024 || c->head_dim < 1 || c->rope_dim < 2 || (c->rope_dim & 1) || c->rope_dim > c->head_dim)
        return "an attention geometry its shaders do not take";
    if (c->window + K > 3072 || c->window < 1) return "a window and top-k its attention does not take";
    if (c->hc_mult < 1 || c->hc_mult > 8 || c->hc_iters < 1) return "hyper-connections its shaders do not take";
    if ((c->n_heads * c->head_dim) % c->o_groups) return "an output grouping it does not take";
    int any_index = 0;
    for (int i = 0; i < L; i++) any_index |= c->index_source[i] && c->compress_ratio[i] > 0;
    if (any_index && (c->index_n_heads < 1 || c->index_n_heads > 64 || c->index_n_heads * c->index_head_dim > 4096 ||
                      c->index_head_dim < c->rope_dim || K < 1 || K > 4096)) return "an indexer its shaders do not take";
    int pub = -1;   /* the ratio of the last index source's list */
    for (int i = 0; i < L; i++) {
        int r = c->compress_ratio[i];
        if (r <= 0) continue;
        if (c->index_owner[i] < 0) return "a compressed layer with no kv_source before it";
        if (c->kv_source[i] && r > c->max_positions) return "a compression ratio past the context";
        if (c->index_source[i]) pub = r;
        else if (pub != r) return "a compressed layer that reads another ratio's index list";
    }
    int cs = c->candidate_source;
    if (cs >= 0 && cs < L && c->index_source[cs] && c->compress_ratio[cs] > 0) {
        if (c->candidate_block_size < 1) return "candidate blocks it does not take";
        for (int i = cs + 1; i < L; i++)
            if (c->index_source[i] && c->compress_ratio[i] > 0 && c->compress_ratio[i] != c->compress_ratio[cs])
                return "a candidate mask read by a layer of another ratio";
    }
    return NULL;
}
static int v41c_cand_on(const Cfg *c) {
    int cs = c->candidate_source;
    return cs >= 0 && cs < c->n_layers && c->index_source[cs] && c->compress_ratio[cs] > 0;
}

static int v41c_setup(Model *m) {
    Cfg *c = &m->c;
    int L = c->n_layers, D = c->dim, H = c->hc_mult, nm = (2 + H) * H;
    const char *why = v41c_unsupported(m);
    if (why) { fprintf(stderr, "[VK] deepseek_v41 chain: %s; the CPU runs the layers\n", why); return 0; }
    V41Chain *ch = calloc(1, sizeof *ch);
    if (!ch) return 0;
    size_t **offs[] = {&ch->o_an, &ch->o_fn, &ch->o_qn, &ch->o_kn, &ch->o_sink, &ch->o_hca, &ch->o_hcf, &ch->o_cn, &ch->o_ikn,
                       &ch->o_eq, &ch->o_ek};
    for (size_t k = 0; k < sizeof offs / sizeof *offs; k++) if (!(*offs[k] = calloc(L, sizeof(size_t)))) return 0;
    ch->fna = calloc(L, sizeof(void *)); ch->fnf = calloc(L, sizeof(void *));
    ch->win = calloc(L, sizeof(void *)); ch->ckv = calloc(L, sizeof(void *));
    ch->ikey = calloc(L, sizeof(void *)); ch->ring = calloc(L, sizeof(void *));
    ch->kv_valid = calloc(L, sizeof(int)); ch->ccap = calloc(L, sizeof(int)); ch->sli = malloc(L * sizeof(int));
    if (!ch->fna || !ch->fnf || !ch->win || !ch->ckv || !ch->ikey || !ch->ring || !ch->kv_valid || !ch->ccap || !ch->sli) return 0;
    for (int i = 0; i < L; i++) {               /* the split's tables: one per layer that makes compressed rows */
        ch->sli[i] = c->kv_source[i] && c->compress_ratio[i] > 0 ? ch->nsl++ : -1;
        if (ch->sli[i] >= 0 && (ch->rmin == 0 || c->compress_ratio[i] < ch->rmin)) ch->rmin = c->compress_ratio[i];
    }
    g_v41c = ch;
    ch->W = c->window; ch->K = c->index_topk > 0 ? c->index_topk : 0; ch->rmax = 1;
    for (int i = 0; i < L; i++) if (c->kv_source[i] && c->compress_ratio[i] > ch->rmax) ch->rmax = c->compress_ratio[i];
    ch->Wd = ch->W + v41c_rows();
    /* the parameter arena */
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_an[i] = n; n += D; ch->o_fn[i] = n; n += D;
        ch->o_qn[i] = n; n += c->q_lora; ch->o_kn[i] = n; n += c->head_dim;
        ch->o_sink[i] = n; n += c->n_heads;
        ch->o_hca[i] = n; n += 3 + nm; ch->o_hcf[i] = n; n += 3 + nm;
        if (c->kv_source[i]) { ch->o_cn[i] = n; n += c->head_dim; ch->o_ikn[i] = n; n += c->index_head_dim; }
        if (m->L[i].engram_index >= 0) { ch->o_eq[i] = n; n += (size_t)H * D; ch->o_ek[i] = n; n += (size_t)H * D; }
    }
    float *a = calloc(n ? n : 1, sizeof(float));
    if (!a) return 0;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(a + ch->o_an[i], l->attn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_fn[i], l->ffn_norm.w, D * sizeof(float));
        memcpy(a + ch->o_qn[i], l->q_norm.w, c->q_lora * sizeof(float));
        memcpy(a + ch->o_kn[i], l->kv_norm.w, c->head_dim * sizeof(float));
        memcpy(a + ch->o_sink[i], l->attn_sink.w, c->n_heads * sizeof(float));
        memcpy(a + ch->o_hca[i], l->hc_attn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hca[i] + 3, l->hc_attn_base.w, nm * sizeof(float));
        memcpy(a + ch->o_hcf[i], l->hc_ffn_scale.w, 3 * sizeof(float));
        memcpy(a + ch->o_hcf[i] + 3, l->hc_ffn_base.w, nm * sizeof(float));
        if (c->kv_source[i]) {
            memcpy(a + ch->o_cn[i], l->comp_norm.w, c->head_dim * sizeof(float));
            memcpy(a + ch->o_ikn[i], l->idx_knorm.w, c->index_head_dim * sizeof(float));
        }
        if (l->engram_index >= 0) {
            memcpy(a + ch->o_eq[i], l->eng_q.w, (size_t)H * D * sizeof(float));
            memcpy(a + ch->o_ek[i], l->eng_k.w, (size_t)H * D * sizeof(float));
        }
    }
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, a, n * sizeof(float)) && vkc_submit(1);
    free(a);
    /* the tensors */
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        int r = c->compress_ratio[i];
        ok = coli_vk_tensor_ensure(&ch->fna[i], l->hc_attn_fn.w, NULL, 10, H * D, nm, 0) &&
             coli_vk_tensor_ensure(&ch->fnf[i], l->hc_ffn_fn.w, NULL, 10, H * D, nm, 0) &&
             v41c_w8(&l->wq_a) && v41c_w8(&l->wq_b) && v41c_w8(&l->wkv) && v41c_w8(&l->wo_a) && v41c_w8(&l->wo_b) &&
             v41c_w8(&l->sh_w1) && v41c_w8(&l->sh_w3) && v41c_w8(&l->sh_w2);
        if (ok && c->kv_source[i]) ok = v41c_wb(&l->comp_wkv) && (r <= 1 || v41c_wb(&l->comp_wgate)) && v41c_wb(&l->idx_wk);
        if (ok && c->index_source[i] && r > 0) ok = v41c_w8(&l->idx_wq_b) && v41c_wb(&l->idx_wproj);
        if (ok && l->engram_index >= 0) ok = v41c_w8(&l->eng_wkv) != NULL;
        if (ok) ok = (ch->win[i] = vkc_buf((size_t)ch->Wd * c->head_dim * sizeof(float), VKC_DEV)) != NULL;
        if (ok && c->kv_source[i] && r > 1)
            ok = (ch->ring[i] = vkc_buf((size_t)2 * r * c->head_dim * sizeof(float), VKC_DEV)) != NULL;
    }
    if (!ok) { fprintf(stderr, "[VK] deepseek_v41 chain: a matrix or the state did not reach the device; the CPU runs the layers\n"); return 0; }
    ch->ok = 1;
    int nsrc = 0, nidx = 0, neng = 0;
    for (int i = 0; i < L; i++) { nsrc += c->kv_source[i]; nidx += c->index_source[i] && c->compress_ratio[i] > 0; neng += m->L[i].engram_index >= 0; }
    fprintf(stderr, "[VK] deepseek_v41 chain: %d layers on the device (%d compressing, %d indexing, %d engram), %d streams, "
                    "%.1f MiB of parameters\n", L, nsrc, nidx, neng, H, n * 4 / 1048576.0);
    return 1;
}

/* The compressed rows of the kv_source layers, whole or split: planned when the whole
 * mirrors would grow (and at the first forward); once split, the split stays. ctx: the
 * positions the forward reaches. 0 = the device refused the split's rows. */
static int v41c_plan(V41Chain *ch, Model *m, int ctx) {
    Cfg *c = &m->c;
    if (!ch->nsl || ch->ks.on) return 1;
    int L = c->n_layers, hd = c->head_dim, tcap = 1, grow = !ch->planned;
    size_t need = 0, held = 0;
    for (int i = 0; i < L; i++) {
        if (ch->sli[i] < 0) continue;
        int r = c->compress_ratio[i], host = c->max_positions / r, want = ctx / r > 0 ? ctx / r : 1, cap = ch->ccap[i];
        if (host > tcap) tcap = host;
        if (cap < want) { cap = 64; while (cap < want) cap *= 2; if (cap > host) cap = host; grow = 1; }
        need += (size_t)cap * hd * sizeof(float); held += (size_t)ch->ccap[i] * hd * sizeof(float);
    }
    if (!grow) return 1;
    ch->planned = 1;
    /* a window of half the slots, the rest pinned by the indexer's picks; the forward's
     * chunks shrink to what the window takes (v41c_forward) */
    if (vkc_kv_plan_need(&ch->ks, "deepseek_v41", ch->nsl, (size_t)hd * sizeof(float), tcap, 1, 1, held, need) != 2) return 1;
    for (int i = 0; i < L; i++) {
        if (ch->sli[i] < 0) continue;
        vkc_free(ch->ckv[i]);
        if (!(ch->ckv[i] = vkc_buf((size_t)ch->ks.rows * hd * sizeof(float), VKC_DEV))) return 0;
    }
    return 1;
}

/* the compressed rows and index keys of a kv_source layer: room for `rows` rows */
static int v41c_cache(V41Chain *ch, Model *m, int i, int rows) {
    Cfg *c = &m->c;
    if (ch->ccap[i] >= rows) return 1;
    int r = c->compress_ratio[i], host = c->max_positions / (r > 0 ? r : 1);
    int cap = 64; while (cap < rows) cap *= 2;
    if (cap > host) cap = host;
    if (cap < rows) return 0;
    if (!ch->ks.on) {          /* under the split the compressed rows keep their ks.rows */
        vkc_free(ch->ckv[i]);
        ch->ckv[i] = vkc_buf((size_t)cap * c->head_dim * sizeof(float), VKC_DEV);
    }
    vkc_free(ch->ikey[i]);
    ch->ikey[i] = vkc_buf((size_t)cap * c->index_head_dim * sizeof(float), VKC_DEV);
    ch->kv_valid[i] = 0; ch->ccap[i] = 0;
    if (!ch->ckv[i] || !ch->ikey[i]) return 0;
    ch->ccap[i] = cap;
    return 1;
}

static int v41c_scratch(V41Chain *ch, Model *m, int rows, int ctx);
/* Prompt rows per chunk (vkc_chunk_rows, decided at the first forward, after the tier
 * filled): the chain's scratch a row at the model's context (the indexer's scores and
 * mask) at the forward's context, the window ring's row, and the routed experts' outputs. */
static int v41c_chunk_rows(V41Chain *ch, Model *m, int ctx) {
    if (!vkc_chunk_auto()) return v41c_rows();
    size_t wcap = ch->wcap;
    g_v41c_count = 0; v41c_scratch(ch, m, 1, ctx); long long b1 = g_v41c_count;
    g_v41c_count = 0; v41c_scratch(ch, m, 2, ctx); long long b2 = g_v41c_count;
    g_v41c_count = -1; ch->wcap = wcap;
    const Cfg *c = &m->c;
    size_t row = (size_t)(b2 - b1) + (size_t)c->n_layers * c->head_dim * sizeof(float) +
                 (size_t)(2 * c->n_activated + 1) * c->dim * sizeof(float);
    /* The CPU expert path gathers one expert's rows and keeps its gate/up and
     * output beside the rank-order contribution buffer. */
    row += 2 * ((size_t)c->dim + c->moe_inter) * sizeof(float);
    if (ch->ks.on) row += (size_t)(ch->W + ch->K) * sizeof(int);
    return vkc_chunk_rows("deepseek_v41", row);
}
/* The window rings for chunks of `rows`: W + rows rows each (a chunk never overwrites a
 * row one of its earlier queries reads); a ring that grows is mirrored again. */
static int v41c_grow_win(V41Chain *ch, Model *m, int rows) {
    if (ch->W + rows <= ch->Wd) return 1;
    const Cfg *c = &m->c;
    int Wd = ch->W + rows;
    for (int i = 0; i < c->n_layers; i++) {
        vkc_free(ch->win[i]);
        if (!(ch->win[i] = vkc_buf((size_t)Wd * c->head_dim * sizeof(float), VKC_DEV))) return 0;
    }
    ch->Wd = Wd; ch->win_valid = 0;
    return 1;
}
static int v41c_scratch(V41Chain *ch, Model *m, int rows, int ctx) {
    Cfg *c = &m->c;
    int D = c->dim, H = c->hc_mult, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H, nh = c->n_heads, hd = c->head_dim;
    int IH = c->index_n_heads > 0 ? c->index_n_heads : 1, ID = c->index_head_dim > 0 ? c->index_head_dim : 1;
    int ew = m->engram.active ? m->engram.cols * m->engram.head_dim : 1, T = c->n_spec_targets > 0 ? c->n_spec_targets : 1;
    size_t r = (size_t)rows;
    ch->wcap = (size_t)ctx;
    int ok = v41c_res(&ch->xs, r * HD, VKC_DEV) && v41c_res(&ch->xn, r * HD, VKC_DEV) && v41c_res(&ch->mix, r * nm, VKC_DEV) &&
             v41c_res(&ch->hpa, r * hr, VKC_DEV) && v41c_res(&ch->hpf, r * hr, VKC_DEV) && v41c_res(&ch->col, r * D, VKC_DEV) &&
             v41c_res(&ch->nrm, r * D, VKC_DEV) && v41c_res(&ch->qa, r * c->q_lora, VKC_DEV) && v41c_res(&ch->qr, r * c->q_lora, VKC_DEV) &&
             v41c_res(&ch->q, r * nh * hd, VKC_DEV) && v41c_res(&ch->kv, r * hd, VKC_DEV) && v41c_res(&ch->heads, r * nh * hd, VKC_DEV) &&
             v41c_res(&ch->grp, r * c->o_groups * c->o_lora, VKC_DEV) && v41c_res(&ch->br, r * D, VKC_DEV) &&
             v41c_res(&ch->craw, r * hd, VKC_DEV) && v41c_res(&ch->cscr, r * hd, VKC_DEV) &&
             v41c_res(&ch->iq, r * IH * ID, VKC_DEV) && v41c_res(&ch->ihw, r * IH, VKC_DEV) &&
             v41c_res(&ch->isc, r * ch->wcap, VKC_DEV) && v41c_res(&ch->mask, r * ch->wcap, VKC_DEV) &&
             v41c_res(&ch->list, r * (ch->W + ch->K), VKC_DEV) &&
             v41c_res(&ch->cs, (r + ch->rmax) * 2 * c->rope_dim, VKC_DEV) && v41c_res(&ch->erows, r * ew, VKC_DEV) &&
             v41c_res(&ch->ekv, r * (size_t)D * (H + 1), VKC_DEV) && v41c_res(&ch->gs, r * c->moe_inter, VKC_DEV) &&
             v41c_res(&ch->us, r * c->moe_inter, VKC_DEV) && v41c_res(&ch->hs, r * c->moe_inter, VKC_DEV) &&
             v41c_res(&ch->ds, r * D, VKC_DEV) && v41c_res(&ch->h2d, r * D, VKC_DOWN) && v41c_res(&ch->mh, r * T * D, VKC_DOWN) &&
             v41c_res(&ch->xd, r * HD, VKC_DOWN) && v41c_res(&ch->hpd, r * hr, VKC_DOWN) && v41c_res(&ch->routed, r * D, VKC_UP);
    if (ok && ch->ks.on) {     /* the split: a chunk's new compressed rows, the two parts of an attention */
        size_t nc = r / (size_t)ch->rmin + 2;
        if (g_v41c_count >= 0)
            g_v41c_count += (long long)((2 * r * hd * sizeof(float) + ch->rmin - 1) / ch->rmin);
        else ok = v41c_res(&ch->cnew, nc * hd, VKC_DEV) && v41c_res(&ch->cdn, nc * hd, VKC_DOWN);
    }
    if (!ok || g_v41c_count >= 0) return ok;   /* counting: the buffers only */
    if (ch->rows < rows) {
        float *hrt = realloc(ch->host_routed, r * D * sizeof(float));
        if (!hrt) return 0;
        ch->host_routed = hrt; ch->rows = rows;
    }
    return 1;
}

/* ---- one layer's pieces ----------------------------------------------------------------- */
static int v41c_norm(VkcBuf *x, size_t xo, int xrow, size_t wo, VkcBuf *y, size_t yo, int yrow, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, xrow, D, (int)yo, yrow, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, g_v41c->prm, y, &p);
}
/* mHC's entry for a site: the mix, the split into hp, the collapse with `prev`'s pre, the norm into nrm */
static int v41c_pre(V41Chain *ch, Model *m, ColiVkTensor *fn, size_t hco, VkcBuf *hp, VkcBuf *prev, size_t lno, int n) {
    Cfg *c = &m->c; int H = c->hc_mult, D = c->dim, HD = H * D, nm = (2 + H) * H, hr = 2 * H + H * H;
    VkcMhc sp = {n, H, D, c->hc_iters, 0, HD, 0, nm, 0, hr, 0, D, (int)hco, 0, c->norm_eps, c->hc_eps, 0.f};
    VkcMhc co = sp;
    return vkc_matmul(fn, ch->xs, 0, ch->mix, 0, n) && vkc_mhc(VKC_MHC_SPLIT, ch->xs, ch->mix, hp, ch->prm, NULL, &sp) &&
           vkc_mhc(VKC_MHC_COLLAPSE, ch->xs, NULL, prev, NULL, ch->col, &co) &&
           v41c_norm(ch->col, 0, D, lno, ch->nrm, 0, D, n, D, c->norm_eps);
}
/* mHC's exit: xn = comb(xs) + post * br, then the two swap */
static int v41c_post(V41Chain *ch, Model *m, VkcBuf *hp, int n) {
    Cfg *c = &m->c; int H = c->hc_mult, D = c->dim, HD = H * D, hr = 2 * H + H * H;
    VkcMhc po = {n, H, D, c->hc_iters, 0, HD, 0, D, 0, hr, 0, HD, 0, 0, c->norm_eps, c->hc_eps, 0.f};
    if (!vkc_mhc(VKC_MHC_POST, ch->xs, ch->br, hp, NULL, ch->xn, &po)) return 0;
    VkcBuf *t = ch->xs; ch->xs = ch->xn; ch->xn = t;
    return 1;
}
/* RoPE on interleaved pairs from this chunk's tables (ch->cs): nseg segments, per_row a row */
static int v41c_rope(V41Chain *ch, Model *m, VkcBuf *x, int nseg, int per_row, int x_off, int x_row, int x_seg,
                     int cs_off, int cs_row, int inverse) {
    VkcDsRope p = {nseg, per_row, m->c.rope_dim, x_off, x_row, x_seg, cs_off, cs_row, inverse, 0};
    return vkc_dsv4_rope(x, ch->cs, &p);
}

/* The forward being recorded: positions, its rows, and where its pieces go. */
typedef struct {
    int start, n, pb, nr, spec;
    int csB;              /* the compress table's offset in ch->cs: its row 0 is position pb - rmax + 1 */
    int *keyl;            /* per index source layer and row: whose index keys it scores (spec: per row) */
    size_t *praw;         /* per layer: where a speculative step's raw compressor rows go in ch->pull */
} V41Fwd;

/* attention_run for nr rows on the device: the projections, the window ring, the
 * compressor, the indexer, the sparse attention and the output projection, into ch->br */
static int v41c_attention(V41Chain *ch, Model *m, int i, const V41Fwd *f) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    int hd = c->head_dim, nh = c->n_heads, rd = c->rope_dim, QL = c->q_lora, r = c->compress_ratio[i];
    int nr = f->nr, pb = f->pb, qrow = nh * hd, LR = ch->W + ch->K;
    int tcs = r > 0 ? f->csB + (ch->rmax - 1) * rd : 0;            /* rope_for(layer): the row of position pb */
    int ok = vkc_matmul(v41c_w8(&l->wq_a), ch->nrm, 0, ch->qa, 0, nr) &&
             v41c_norm(ch->qa, 0, QL, ch->o_qn[i], ch->qr, 0, QL, nr, QL, c->norm_eps) &&
             vkc_matmul(v41c_w8(&l->wq_b), ch->qr, 0, ch->q, 0, nr) &&
             vkc_matmul(v41c_w8(&l->wkv), ch->nrm, 0, ch->kv, 0, nr) &&
             v41c_norm(ch->kv, 0, hd, ch->o_kn[i], ch->kv, 0, hd, nr, hd, c->norm_eps) &&
             v41c_rope(ch, m, ch->q, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 0) &&
             v41c_rope(ch, m, ch->kv, nr, 1, hd - rd, hd, 0, tcs, rd, 0);
    if (!ok) return 0;
    VkcRegion *rg = malloc((size_t)nr * sizeof *rg);   /* the new rows into the window ring */
    if (!rg) return 0;
    for (int s = 0; s < nr; s++) rg[s] = (VkcRegion){(size_t)((pb + s) % ch->Wd) * hd, (size_t)s * hd, (size_t)hd};
    ok = vkc_copy_regions(ch->win[i], ch->kv, rg, nr);
    free(rg);
    if (ok && r > 0 && c->kv_source[i]) {                           /* the compressor, its latents, the index keys */
        int g0 = pb / r, np = (pb + nr) / r - g0, ihd = c->index_head_dim, sp = ch->ks.on;
        VkcBuf *cb = sp ? ch->cnew : ch->ckv[i];                     /* the split: the new rows through a scratch */
        size_t co = sp ? 0 : (size_t)g0 * hd;
        ok = vkc_matmul(v41c_wb(&l->comp_wkv), ch->nrm, 0, ch->craw, 0, nr);
        if (ok && r == 1) ok = v41c_norm(ch->craw, 0, hd, ch->o_cn[i], cb, co, hd, nr, hd, c->norm_eps);
        else if (ok) {
            VkcDsComp cp = {nr, pb, r, hd, hd, 0, 0, hd, 0, hd, -1, sp ? -g0 * hd : 0, hd, 0};
            ok = vkc_matmul(v41c_wb(&l->comp_wgate), ch->nrm, 0, ch->cscr, 0, nr) &&
                 vkc_dsv4_compress(ch->craw, ch->cscr, ch->ring[i], NULL, cb, &cp) &&
                 (np == 0 || v41c_norm(cb, co, hd, ch->o_cn[i], cb, co, hd, np, hd, c->norm_eps));
            if (ok && f->spec && f->praw)                           /* a verify's raw rows, for its undo rows */
                ok = vkc_copy(ch->pull, f->praw[i], ch->craw, 0, (size_t)nr * hd) &&
                     vkc_copy(ch->pull, f->praw[i] + (size_t)nr * hd, ch->cscr, 0, (size_t)nr * hd);
        }
        if (ok && np > 0) {
            int bc = f->csB + (g0 * r - (pb - ch->rmax + 1)) * rd;   /* a latent rotates at its group's first position */
            ok = vkc_matmul(v41c_wb(&l->idx_wk), cb, co, ch->ikey[i], (size_t)g0 * ihd, np) &&
                 v41c_norm(ch->ikey[i], (size_t)g0 * ihd, ihd, ch->o_ikn[i], ch->ikey[i], (size_t)g0 * ihd, ihd, np, ihd, c->norm_eps) &&
                 v41c_rope(ch, m, ch->ikey[i], np, 1, g0 * ihd + ihd - rd, ihd, 0, bc, r * rd, 0) &&
                 v41c_rope(ch, m, cb, np, 1, (int)co + hd - rd, hd, 0, bc, r * rd, 0);
            if (ok && sp) {   /* into their slots, and down for the host's ckv once the frame is through */
                VkcKvPart pt = {1, hd, l->ckv, 0, ch->ckv[i], 0};
                ok = vkc_kv_store(&ch->ks, ch->sli[i], &pt, cb, 0, (size_t)hd, 0, g0, np) &&
                     vkc_copy(ch->cdn, 0, cb, 0, (size_t)np * hd);
                vkc_kv_done(&ch->ks, ch->sli[i], g0 + np);
            }
        }
    }
    if (ok && r > 0 && c->index_source[i]) {                        /* the indexer: this layer's selection */
        int IH = c->index_n_heads, ID = c->index_head_dim, width = (pb + nr) / r, cs = c->candidate_source;
        int cand = v41c_cand_on(c), mrow = cand && cs < i ? (int)ch->wcap : 0;
        float wscale = (1.0f / sqrtf((float)ID)) * (1.0f / sqrtf((float)IH));
        ok = vkc_matmul(v41c_w8(&l->idx_wq_b), ch->qr, 0, ch->iq, 0, nr) &&
             v41c_rope(ch, m, ch->iq, nr * IH, IH, ID - rd, IH * ID, ID, tcs, rd, 0) &&
             vkc_matmul(v41c_wb(&l->idx_wproj), ch->nrm, 0, ch->ihw, 0, nr);
        const int *kl = f->keyl + (size_t)i * f->n + (pb - f->start);
        int same = 1;
        for (int s = 1; s < nr; s++) same &= kl[s] == kl[0];
        for (int s = 0; ok && s < nr; s += same ? nr : 1) {
            VkcDsScore sp = {same ? nr : 1, pb + s, r, IH, ID, width, s * IH * ID, IH * ID, s * IH, IH, 0, ID, mrow,
                             (int)ch->wcap, wscale, s * (int)ch->wcap, s * mrow};
            ok = vkc_dsv4_score(ch->iq, ch->ihw, ch->ikey[kl[s]], ch->mask, ch->isc, &sp);
        }
        if (ok && cand && cs == i) {
            VkcDsCand cp = {nr, pb, r, width, c->candidate_block_size, c->candidate_topk_blocks, (int)ch->wcap, (int)ch->wcap};
            ok = vkc_dsv4_cand(ch->isc, ch->mask, &cp);
        }
        VkcDsTopk tp = {nr, width, ch->K, (int)ch->wcap, ch->W, LR, ch->Wd, 0};
        ok = ok && vkc_dsv4_topk(ch->isc, ch->list, &tp);
    }
    if (!ok) return 0;
    VkcDsAttn a = {nr, nh, hd, ch->W + (r > 0 ? ch->K : 0), 0, LR, ch->Wd, 0, 0, 0, qrow, 0, qrow, (int)ch->o_sink[i], 0,
                   1.0f / sqrtf((float)hd)};
    int og = c->o_groups, ol = c->o_lora, pg = qrow / og;
    VkcHgemv g = {0, nr, og, ol, ol, 0, 0, qrow, pg, 0, og * ol, ol, 0, 0, 0};
    if (ch->ks.on && r > 0) {   /* stage selected cold rows, preserving the original sparse attention order */
        int own = c->index_owner[i], li = ch->sli[own];
        VkcKvPart pt = {1, hd, m->L[own].ckv, 0, ch->ckv[own], 0};
        VkcKvDs d = {&ch->ks, li, ch->q, ch->win[i], ch->ckv[own], ch->prm, ch->list, ch->heads, nr, nh, hd, a.cnt, 0, LR,
                     ch->Wd, 0, 0, qrow, (int)ch->o_sink[i], 0, 0, qrow, (pb + nr) / r, a.scale, m->L[own].ckv};
        /* the blocks pinned since the chunk began, and the table, before this read */
        ok = vkc_kv_push(&ch->ks, li, &pt, 1, ch->ks.t[li].valid) && vkc_kv_ds(&d);
    } else ok = vkc_dsv4_attn(ch->q, ch->win[i], r > 0 ? ch->ckv[c->index_owner[i]] : NULL, ch->list, ch->prm, ch->heads, &a);
    return ok &&
           v41c_rope(ch, m, ch->heads, nr * nh, nh, hd - rd, qrow, hd, tcs, rd, 1) &&
           vkc_mla_hgemv(v41c_w8(&l->wo_a), ch->heads, ch->grp, NULL, &g) &&
           vkc_matmul(v41c_w8(&l->wo_b), ch->grp, 0, ch->br, 0, nr);
}

/* the shared expert: silu(min(w1 x, lim)) * clamp(w3 x, +-lim), then w2 (swiglu_into) */
static int v41c_shared(V41Chain *ch, Model *m, Layer *l, int nr) {
    Cfg *c = &m->c; int I = c->moe_inter;
    VkcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nr * I, 0.f, 0.f, c->swiglu_limit > 0.0f ? c->swiglu_limit : INFINITY};
    return vkc_matmul(v41c_w8(&l->sh_w1), ch->nrm, 0, ch->gs, 0, nr) && vkc_matmul(v41c_w8(&l->sh_w3), ch->nrm, 0, ch->us, 0, nr) &&
           vkc_mhc(VKC_SWIGLU_CLAMP, ch->gs, ch->us, NULL, NULL, ch->hs, &sw) &&
           vkc_matmul(v41c_w8(&l->sh_w2), ch->hs, 0, ch->ds, 0, nr);
}
/* engram_run on the device: the host's n-gram rows, eng_wkv, the gate into each stream */
static int v41c_engram(V41Chain *ch, Model *m, int i, int nr, int pb, float *rows) {
    Cfg *c = &m->c; Engram *e = &m->engram; Layer *l = &m->L[i];
    int table = l->engram_index, width = e->cols * e->head_dim, H = c->hc_mult, D = c->dim;
    int64_t ids[V41_MAX_NGRAM * V41_MAX_EHEADS];
    double t0 = now_s();
    for (int r = 0; r < nr; r++) {
        engram_hash(e, table, pb + r, ids);
        for (int col = 0; col < e->cols; col++)
            memcpy(rows + (size_t)r * width + (size_t)col * e->head_dim, engram_row(&e->table[table], ids[col], e->head_dim),
                   (size_t)e->head_dim * sizeof(float));
    }
    m->t_engram += now_s() - t0;
    VkcDsEngram p = {nr, H, D, 0, D * (H + 1), 0, H * D, (int)ch->o_eq[i], (int)ch->o_ek[i], c->norm_eps};
    return vkc_write(ch->erows, 0, rows, (size_t)nr * width * sizeof(float)) &&
           vkc_matmul(v41c_w8(&l->eng_wkv), ch->erows, 0, ch->ekv, 0, nr) && vkc_dsv4_engram(ch->ekv, ch->prm, ch->xs, &p);
}

/* Every layer for n rows of streams h (positions start..), the final streams and the last
 * FFN site's mix back into h and pre_mix (every row when all, else the last). 0 = not taken:
 * the CPU runs the layers, h and pre_mix untouched. */
static int v41c_forward(Model *m, float *h, float *pre_mix, int n, int start, int spec, int all) {
    V41Chain *ch = g_v41c;
    if (!g_vk_chain || !ch || !ch->ok || ch->failed || g_trace) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && n <= 2) return 0;
    if (vkc_lost()) { ch->failed = 1; g_vk_chain = 0; return 0; }
    Cfg *c = &m->c;
    int L = c->n_layers, D = c->dim, H = c->hc_mult, HD = H * D, hr = 2 * H + H * H, hd = c->head_dim, rd = c->rope_dim;
    int W = ch->W, LR = W + ch->K, T = m->spec.active ? c->n_spec_targets : 0, E = start + n;
    if (spec && n > W) return 0;                           /* the undo rows: one window at most */
    if (!v41c_plan(ch, m, E)) {
        fprintf(stderr, "[VK] deepseek_v41 chain: device memory for the split's compressed rows refused; the CPU runs the layers\n");
        ch->failed = 1; g_vk_chain = 0;
        return 0;
    }
    int CH = v41c_chunk_rows(ch, m, E), rows;
    { int64_t lim = ((int64_t)32 << 20) / (E > 1 ? E : 1); if (lim < CH) CH = lim < 1 ? 1 : (int)lim; }
    if (ch->ks.on) {   /* a chunk's new compressed rows fit the split's window */
        int mx = ch->rmin == 1 ? ch->ks.chunk : (ch->ks.chunk - 1) * ch->rmin;
        if (CH > mx) CH = mx < 1 ? 1 : mx;
    }
    if (spec && n > CH) return 0;                          /* a verify is one chunk */
    rows = n < CH ? n : CH;
    if (!v41c_grow_win(ch, m, rows)) {
        fprintf(stderr, "[VK] deepseek_v41 chain: device memory for window rings of %d rows refused; the CPU runs the layers\n", rows);
        ch->failed = 1; g_vk_chain = 0;
        return 0;
    }
    V41Fwd f = {start, n, 0, 0, spec, rows * rd, NULL, NULL};
    /* whose index keys each index source reads, per row (indexer_run, published_owner) */
    f.keyl = malloc((size_t)L * n * sizeof(int));
    f.praw = calloc((size_t)L, sizeof(size_t));
    int *need = calloc((size_t)L, sizeof(int));
    float *outs = all ? malloc((size_t)n * HD * sizeof(float)) : malloc((size_t)HD * sizeof(float));
    float *pres = all ? malloc((size_t)n * H * sizeof(float)) : malloc((size_t)H * sizeof(float));
    int *wl = malloc((size_t)rows * LR * sizeof(int));
    float *cs = malloc((size_t)(rows + ch->rmax) * 2 * rd * sizeof(float));
    float *er = m->engram.active ? malloc((size_t)rows * m->engram.cols * m->engram.head_dim * sizeof(float)) : NULL;
    if (!f.keyl || !f.praw || !need || !outs || !pres || !wl || !cs || (m->engram.active && !er)) {
        free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs); free(er);
        return 0;
    }
    static int tidy = -1;
    if (tidy < 0) tidy = getenv("V41_INDEX_OWNER") != NULL;
    /* the rows of each kv_source layer the forward reads that it does not produce first:
     * the complete groups before it, and whatever another layer's selection reaches (the
     * host's rows there, as the CPU reads them) */
    for (int i = 0; i < L; i++) if (c->kv_source[i] && c->compress_ratio[i] > 0) need[i] = start / c->compress_ratio[i];
    int pub = m->published_index_k ? m->published_index_layer : -1, last_pub = -1;
    for (int i = 0; i < L; i++) {
        int r = c->compress_ratio[i];
        if (r <= 0) continue;
        int owner = c->index_owner[i];
        if (c->kv_source[i] && E / r - start / r > 0) { pub = i; last_pub = i; }
        if (E / r > need[owner]) need[owner] = E / r;
        for (int t = 0; t < n; t++) {
            int k = owner;
            if (c->index_source[i] && !tidy) {
                if (m->pub_rows > 0) {
                    int p = published_owner(m, i, start, t);
                    k = p >= 0 ? p : m->pub_before ? m->pub_before_layer : owner;
                } else if (n == 1 && pub >= 0) k = pub;
            }
            f.keyl[(size_t)i * n + t] = k;
            if (c->index_source[i] && (start + t + 1) / r > need[k]) need[k] = (start + t + 1) / r;
        }
    }
    int ok = 1;
    for (int i = 0; i < L && ok; i++) {
        if (!c->kv_source[i]) continue;
        int r = c->compress_ratio[i], host = c->max_positions / r;
        if (need[i] > host) need[i] = host;
        int want = need[i] > E / r ? need[i] : E / r;
        ok = v41c_cache(ch, m, i, want > 0 ? want : 1);
    }
    /* what comes back at the end: the window rows, the compressed rows and keys, the groups,
     * a verify's raw compressor rows */
    size_t pw = (size_t)(n < W ? n : W) * hd, pn = 0, *pc = calloc((size_t)L, sizeof(size_t));
    for (int i = 0; ok && pc && i < L; i++) {
        pc[i] = pn; pn += pw;
        int r = c->compress_ratio[i];
        if (!c->kv_source[i]) continue;
        pn += (size_t)(E / r - start / r) * ((ch->ks.on ? 0 : hd) + c->index_head_dim);   /* split: the rows came back per chunk */
        if (r > 1) { pn += (size_t)2 * r * hd; if (spec) { f.praw[i] = pn; pn += (size_t)2 * n * hd; } }
    }
    ok = ok && pc && v41c_scratch(ch, m, rows, E) && v41c_res(&ch->pull, pn, VKC_DOWN) && (!T || m->main_hidden);
    if (!ok) {
        fprintf(stderr, "[VK] deepseek_v41 chain: device memory for %d rows at %d positions refused; the CPU runs the layers\n", rows, E);
        free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs); free(er); free(pc);
        ch->failed = 1; g_vk_chain = 0;
        return 0;
    }
    vkc_gemm_rows(spec ? 0 : -1);                          /* a verify's rows get a decode step's bits */
    for (int c0 = 0; c0 < n; c0 += rows) {
        int nr = n - c0 < rows ? n - c0 : rows, pb = start + c0, last = c0 + nr == n;
        f.pb = pb; f.nr = nr;
        /* this chunk's RoPE rows (the CPU's own tables): window ones at 0, compress ones at
         * csB from position pb - rmax + 1 */
        for (int s = 0; s < nr; s++) memcpy(cs + (size_t)s * rd, m->rope_window + (size_t)(pb + s) * rd, rd * sizeof(float));
        for (int s = 0; s < nr + ch->rmax - 1; s++) {
            int p = pb - ch->rmax + 1 + s;
            float *dst = cs + f.csB + (size_t)s * rd;
            if (p < 0) memset(dst, 0, rd * sizeof(float));
            else memcpy(dst, m->rope_compress + (size_t)p * rd, rd * sizeof(float));
        }
        /* the window half of each row's list (window_idxs): the positions of the last
         * `window` the row reaches, oldest first, as device ring rows; -1 where the CPU's
         * ring does not hold the position (window_pos). The selection half starts empty. */
        const Layer *l0 = &m->L[0];
        for (int s = 0; s < nr; s++) {
            int p = pb + s;
            for (int k = 0; k < W; k++) {
                int q = p - W + 1 + k, e = -1;
                if (q >= 0 && (q >= start || l0->window_pos[q % W] == q)) e = q % ch->Wd;
                wl[(size_t)s * LR + k] = e;
            }
            for (int k = W; k < LR; k++) wl[(size_t)s * LR + k] = -1;
        }
        if (!vkc_begin() || !vkc_write(ch->xs, 0, h + (size_t)c0 * HD, (size_t)nr * HD * sizeof(float)) ||
            !vkc_write(ch->list, 0, wl, (size_t)nr * LR * sizeof(int)) ||
            !vkc_write(ch->cs, 0, cs, (size_t)(f.csB + (nr + ch->rmax - 1) * rd) * sizeof(float))) goto lost;
        {   /* the first site collapses with [1, 0, ...] (forward_full's pre_mix) */
            float *hp0 = calloc((size_t)nr * hr, sizeof(float));
            if (!hp0) goto lost;
            for (int s = 0; s < nr; s++) hp0[(size_t)s * hr] = 1.0f;
            ok = vkc_write(ch->hpf, 0, hp0, (size_t)nr * hr * sizeof(float));
            free(hp0);
            if (!ok) goto lost;
        }
        if (c0 == 0) {   /* the device's copies made the host's, as the forward needs them */
            for (int q = ch->win_valid > start - W + 1 ? ch->win_valid : start - W + 1; q < start && ok; q++) {
                if (q < 0) continue;
                for (int i = 0; i < L && ok; i++) {
                    const Layer *l = &m->L[i];
                    if (l->window_pos[q % W] != q) continue;
                    ok = vkc_write(ch->win[i], (size_t)(q % ch->Wd) * hd, l->window + (size_t)(q % W) * hd, (size_t)hd * sizeof(float));
                }
            }
            for (int i = 0; i < L && ok; i++) {
                if (!c->kv_source[i]) continue;
                const Layer *l = &m->L[i];
                int r = c->compress_ratio[i], ihd = c->index_head_dim, t0 = ch->kv_valid[i], t1 = need[i];
                if (t1 > t0) ok = (ch->ks.on || vkc_write(ch->ckv[i], (size_t)t0 * hd, l->ckv + (size_t)t0 * hd, (size_t)(t1 - t0) * hd * sizeof(float))) &&
                                  vkc_write(ch->ikey[i], (size_t)t0 * ihd, l->ikey + (size_t)t0 * ihd, (size_t)(t1 - t0) * ihd * sizeof(float));
                if (ok && r > 1) ok = vkc_write(ch->ring[i], 0, l->cstate_kv, (size_t)r * hd * sizeof(float)) &&
                                      vkc_write(ch->ring[i], (size_t)r * hd, l->cstate_score, (size_t)r * hd * sizeof(float));
            }
            if (!ok) goto lost;
        }
        for (int i = 0; ch->ks.on && i < L && ok; i++) {   /* the split: each table's window over this chunk's new rows */
            if (ch->sli[i] < 0) continue;
            int r = c->compress_ratio[i], g0 = pb / r;
            VkcKvPart pt = {1, hd, m->L[i].ckv, 0, ch->ckv[i], 0};
            vkc_kv_place(&ch->ks, ch->sli[i], g0, (pb + nr) / r - g0);
            ok = vkc_kv_push(&ch->ks, ch->sli[i], &pt, 1, g0);
        }
        if (!ok) goto lost;
        int pending = 0;
        for (int i = 0; i < L && ok; i++) {
            Layer *l = &m->L[i];
            if (pending) {                                 /* the FFN branch of the layer before */
                VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
                ok = vkc_ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41c_post(ch, m, ch->hpf, nr);
                pending = 0;
            }
            if (ok && l->engram_index >= 0) ok = v41c_engram(ch, m, i, nr, pb, er);
            for (int k = 0; ok && k < T; k++) {
                if (c->spec_targets[k] != i) continue;
                VkcMhc mean = {nr, H, D, 0, 0, HD, 0, 0, 0, 0, k * D, T * D, 0, 0, 0.f, 0.f, 0.f};
                ok = vkc_mhc(VKC_MHC_MEAN, ch->xs, NULL, NULL, NULL, ch->mh, &mean);
            }
            double ta = now_s();
            ok = ok && v41c_pre(ch, m, ch->fna[i], ch->o_hca[i], ch->hpa, ch->hpf, ch->o_an[i], nr) &&
                 v41c_attention(ch, m, i, &f) && v41c_post(ch, m, ch->hpa, nr) &&
                 v41c_pre(ch, m, ch->fnf[i], ch->o_hcf[i], ch->hpf, ch->hpa, ch->o_fn[i], nr) &&
                 vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)nr * D) && vkc_submit(0);   /* A1 */
            /* while it runs, the tier loads the experts this layer will likely stream (a
             * big prompt chunk only) */
            if (ok) vkt_stream_prefetch(i, nr);
            ok = ok && vkc_finish();
            m->t_attn += now_s() - ta;
            if (!ok) break;
            if (ch->ks.on && ch->sli[i] >= 0) {            /* this chunk's new compressed rows into the host's ckv */
                int r = c->compress_ratio[i], g0 = pb / r, np = (pb + nr) / r - g0;
                if (np > 0) memcpy(l->ckv + (size_t)g0 * hd, vkc_ptr(ch->cdn), (size_t)np * hd * sizeof(float));
            }
            /* A2: the shared expert, while the host computes the routed experts */
            ok = vkc_begin() && v41c_shared(ch, m, l, nr) && vkc_submit(0);
            double t1 = now_s();
            moe_run_at(m, l, &m->cache[i], "layers", i, c->n_routed, c->n_activated, (const float *)vkc_ptr(ch->h2d), nr,
                       ch->host_routed, 0);
            memcpy(vkc_ptr(ch->routed), ch->host_routed, (size_t)nr * D * sizeof(float));
            ch->host_ms += (now_s() - t1) * 1e3;
            ok = ok && vkc_begin();
            pending = 1;
        }
        if (ok && pending) {
            VkcEw add = {VKC_EW_ADD, nr * D, D, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok = vkc_ew(ch->br, ch->routed, ch->ds, NULL, NULL, &add) && v41c_post(ch, m, ch->hpf, nr);
        }
        ok = ok && vkc_copy(ch->xd, 0, ch->xs, 0, (size_t)nr * HD) && vkc_copy(ch->hpd, 0, ch->hpf, 0, (size_t)nr * hr);
        if (ok && last) {                                  /* what the host's state gets back */
            for (int i = 0; i < L && ok; i++) {
                int q0 = E - W > start ? E - W : start, nq = E - q0;
                VkcRegion *rg = malloc((size_t)nq * sizeof *rg);
                if (!rg) { ok = 0; break; }
                for (int q = q0; q < E; q++) rg[q - q0] = (VkcRegion){pc[i] + (size_t)(q - q0) * hd, (size_t)(q % ch->Wd) * hd, (size_t)hd};
                ok = vkc_copy_regions(ch->pull, ch->win[i], rg, nq);
                free(rg);
                if (!ok || !c->kv_source[i]) continue;
                int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
                size_t at = pc[i] + pw;
                ok = (cw == 0 || vkc_copy(ch->pull, at, ch->ckv[i], (size_t)g0 * hd, (size_t)np * hd)) &&
                     vkc_copy(ch->pull, at + (size_t)np * cw, ch->ikey[i], (size_t)g0 * ihd, (size_t)np * ihd) &&
                     (r <= 1 || vkc_copy(ch->pull, at + (size_t)np * (cw + ihd), ch->ring[i], 0, (size_t)2 * r * hd));
            }
        }
        ok = ok && vkc_submit(1);
        if (!ok) goto lost;
        const float *xd = (const float *)vkc_ptr(ch->xd), *hpd = (const float *)vkc_ptr(ch->hpd);
        for (int s = 0; s < nr; s++) {
            int t = c0 + s;
            if (!all && t != n - 1) continue;
            memcpy(outs + (size_t)(all ? t : 0) * HD, xd + (size_t)s * HD, (size_t)HD * sizeof(float));
            memcpy(pres + (size_t)(all ? t : 0) * H, hpd + (size_t)s * hr, (size_t)H * sizeof(float));
        }
        if (T) memcpy(m->main_hidden + (size_t)c0 * T * D, vkc_ptr(ch->mh), (size_t)nr * T * D * sizeof(float));
    }
    /* every chunk is through: the host's state as the CPU would have left it */
    {
        const float *pl = (const float *)vkc_ptr(ch->pull);
        int save = spec && m->rollback_save && n > 1;
        for (int i = 0; i < L; i++) {
            Layer *l = &m->L[i];
            int q0 = E - W > start ? E - W : start;
            for (int t = 0; t < n; t++) {
                int p = start + t, slot = p % W;
                if (save) {
                    memcpy(l->ring_save + (size_t)t * hd, l->window + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    l->ring_save_pos[t] = l->window_pos[slot];
                }
                if (p < q0) continue;                      /* a later row takes its slot */
                memcpy(l->window + (size_t)slot * hd, pl + pc[i] + (size_t)(p - q0) * hd, (size_t)hd * sizeof(float));
                l->window_pos[slot] = p;
            }
            if (!c->kv_source[i]) continue;
            int r = c->compress_ratio[i], g0 = start / r, np = E / r - g0, ihd = c->index_head_dim, cw = ch->ks.on ? 0 : hd;
            const float *src = pl + pc[i] + pw;
            if (cw) memcpy(l->ckv + (size_t)g0 * hd, src, (size_t)np * hd * sizeof(float));   /* split: back per chunk */
            memcpy(l->ikey + (size_t)g0 * ihd, src + (size_t)np * cw, (size_t)np * ihd * sizeof(float));
            if (r <= 1) continue;
            if (save) {   /* the groups' undo rows, as compressor_run's decode path writes them */
                const float *raw = pl + f.praw[i];
                for (int t = 0; t < n; t++) {
                    int slot = (start + t) % r;
                    memcpy(l->cstate_save_kv + (size_t)t * hd, l->cstate_kv + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_save_score + (size_t)t * hd, l->cstate_score + (size_t)slot * hd, (size_t)hd * sizeof(float));
                    l->cstate_save_slot[t] = slot;
                    memcpy(l->cstate_kv + (size_t)slot * hd, raw + (size_t)t * hd, (size_t)hd * sizeof(float));
                    memcpy(l->cstate_score + (size_t)slot * hd, raw + (size_t)(n + t) * hd, (size_t)hd * sizeof(float));
                }
            }
            const float *ring = src + (size_t)np * (cw + ihd);
            memcpy(l->cstate_kv, ring, (size_t)r * hd * sizeof(float));
            memcpy(l->cstate_score, ring + (size_t)r * hd, (size_t)r * hd * sizeof(float));
        }
        if (last_pub >= 0) { m->published_index_k = m->L[last_pub].ikey; m->published_index_layer = last_pub; }
        ch->win_valid = E;
        for (int i = 0; i < L; i++) if (c->kv_source[i]) ch->kv_valid[i] = E / c->compress_ratio[i];
    }
    if (all) { memcpy(h, outs, (size_t)n * HD * sizeof(float)); memcpy(pre_mix, pres, (size_t)n * H * sizeof(float)); }
    else { memcpy(h + (size_t)(n - 1) * HD, outs, (size_t)HD * sizeof(float)); memcpy(pre_mix + (size_t)(n - 1) * H, pres, (size_t)H * sizeof(float)); }
    free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs); free(er); free(pc);
    ch->forwards++;
    return 1;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    free(f.keyl); free(f.praw); free(need); free(outs); free(pres); free(wl); free(cs); free(er); free(pc);
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    g_vk_chain = 0; ch->failed = 1;
    fprintf(stderr, "[VK] deepseek_v41 chain: the device was lost; the CPU runs this forward again and from here on "
                    "(the host's state is current: there is nothing to rebuild)\n");
    return 0;
}

static void v41c_report(void) {
    V41Chain *ch = g_v41c;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] deepseek_v41 chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, ch->host_ms, st.dev_bytes / 1048576.0);
    vkc_kv_report(&ch->ks);
    vkc_prof_print();
}

/* COLI_VK_CHAIN at startup, before the tier sizes itself (the trunk's device copies count
 * as used): the decision, the pipelines, the tensors. */
static int g_v41c_decision = -1;
static int v41c_decide(Model *m) {   /* once: early when the trunk may live on the device only */
    if (g_v41c_decision >= 0) return g_v41c_decision;
    if (!g_vk_ready) return 0;
    int tier_on = vkt_wanted() && m->c.n_routed > 0;
    return g_v41c_decision = coli_vk_chain_decide("deepseek_v41", tier_on, COLI_VK_CHAIN_UNMEASURED);
}
static void v41c_start(Model *m) {
    if (!g_vk_ready) return;
    int on = v41c_decide(m);
    const char *no = NULL;
    if (on && !(g_v41c_inited = vkc_init())) no = "the chain's pipelines did not come up";
    if (on && !no && !(vkc_mla_ready() && vkc_mhc_ready() && vkc_dsv4_ready()))
        no = "the MLA, mHC or DeepSeek shaders are missing (chain_hgemv, chain_mhc, chain_dsv4)";
    if (no) fprintf(stderr, "[VK] deepseek_v41: %s: the dense chain stays off\n", no);
    if (!on || no) return;
    if (!v41c_setup(m)) return;
    g_vk_chain = on;
}
/* After the tier's: at exit the report, then the chain goes, then the device. */
static void v41c_atexit(void) {
    if (!g_v41c_inited) return;
    atexit(vkc_shutdown);
    atexit(v41c_report);
}
