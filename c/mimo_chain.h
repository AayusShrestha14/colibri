/* mimo_chain.h -- MiMo-V2.6's layers as a dense chain on the Vulkan device (vk_chain.h).
 * Included once by mimo.c in a COLI_VULKAN build, after the CPU pieces it reuses (moe()
 * for the routed experts) and before forward(), which calls it; COLI_VK_CHAIN decides
 * (coli_vk_chain_decide, in main).
 *
 * What runs where, per layer, for one block of rows (decode: one row):
 *   device: the routed MoE output of the layer before joins the residual (x += out,
 *     the CPU's add), the input RMSNorm, the fused qkv projection, partial RoPE on q and
 *     k (rotate-half on the first rope_dim dims, the CPU's own cosf/sinf in a host
 *     table, one table per layer kind: the two kinds have their own theta), the value
 *     scale (chain_ew SCALE), the new K/V rows into the device's cache, attention with
 *     the layer's sliding window and sink logit (chain_attn), o_proj, the residual add,
 *     the post-attention RMSNorm; on the dense layer (layer 0 on the release) the MLP
 *     too (gate, up, SwiGLU, down, add), with no host step. A MoE layer's frame ends
 *     with the normalized rows going down and is waited for.
 *   host: moe() as the CPU runs it, on the device's normalized rows: the router
 *     (sigmoid scores, the correction bias picks, top-k renormalized), the routed
 *     experts (the Vulkan expert tier's batch and the CPU's share, joined in each row's
 *     routing order). MiMo has no shared expert, so nothing runs beside it.
 *   last frame: the final norm and lm_head on the last row (every row for a read-out).
 * Crossing per MoE layer: D floats a row down, D floats a row up; per step the new K/V
 * rows of every layer down (a few KB a row) and the logits.
 *
 * The KV caches, and who owns them. The host's stay canonical, in the layout the CPU
 * keeps (mimo.c's kv_alloc): a full-attention layer [ctx][kvh][hd] (V [ctx][kvh][vd]),
 * a sliding-window layer a RING of rows = min(window, ctx) rows, position p in row
 * p % rows. The device mirrors each one in exactly that layout (chain_attn reads
 * position-major rows, and a ring), behind a watermark kv_valid per layer: positions
 * [0, kv_valid) -- for a ring, the last `rows` of them -- equal the host's. A step at
 * pos_base lowers the watermark to pos_base (a reset, a rollback), then uploads the
 * host's rows between it and pos_base (for a ring only those the window still sees); a
 * CPU step lowers it to its pos_base; a photo restored into the host's rings
 * (pin_state_load) drops the rings' watermarks to 0, so the next step uploads them.
 *
 * A windowed layer's attention: one decode row reads the ring in place (its window is
 * exactly the ring's rows); a block of rows cannot (its later rows overwrite slots its
 * earlier rows still see), so it gathers the window's earlier rows from the ring and its
 * own new rows into one position-major scratch, attends there, and then writes its last
 * `rows` rows into the ring, as the CPU's attention() does with its window copy. The two
 * visit the positions in the same order, so a row gets the same bits either way.
 *
 * The new rows reach the host's caches only when the step's last frame has completed,
 * so the host's state always describes whole steps: a device lost in the middle of a
 * step loses nothing the host holds, and the CPU runs the step's remaining rows from
 * the position the host's caches end at (no rebuild, a turn with a picture included).
 * COLI_VK_CHAIN_FAULT=n (vk_chain.c) fakes the loss at the n-th frame, for tests.
 *
 * The chain declines (the per-matrix path or the CPU runs) when a dense matrix did not
 * reach the device, for geometry outside chain_attn (a head dim above 256) and under
 * MIMO_TRACE (the CPU's per-layer dump). */
#include "vk_chain.h"

typedef struct {
    int ok, failed;
    int rows;                                  /* scratch capacity in rows */
    int rd_off[2];                             /* the RoPE table of each layer kind in cs */
    VkcBuf *prm;                               /* the norms and the sink logits */
    size_t *o_ln1, *o_ln2, *o_sink, o_final;
    VkcBuf **kc, **vc;                         /* the device's caches, per layer */
    int *kv_valid;
    size_t *o_kvd;                             /* a layer's new K rows in kvd (V after cap rows) */
    VkcBuf *x, *nrm, *tmp, *qkv, *kn, *vn, *wk, *wv, *ctx, *g, *u, *fin;
    VkcBuf *h2d, *kvd, *outd, *routed, *cs;
    float *host_out;
    unsigned long long forwards, frames;
    double wait_ms, host_ms;
} MimoChain;

static int g_vk_chain = 0;     /* COLI_VK_CHAIN decided on, and the chain's pipelines are up */

static int mc_chunk_rows(void) {
    const char *e = getenv("COLI_VK_CHAIN_ROWS");
    int v = e && *e ? atoi(e) : 512;
    return v < 1 ? 1 : v > 65535 ? 65535 : v;
}

/* the geometry of layer li's attention */
typedef struct { int k, nh, kvh, hd, vd, qd, kd, vdd, rw, half, rows, swa; } McGeo;
static McGeo mc_geo(const Model *m, int li) {
    const Cfg *c = &m->c;
    McGeo g;
    g.k = c->swa[li]; g.swa = g.k;
    g.nh = c->heads[g.k]; g.kvh = c->kv_heads[g.k]; g.hd = c->head_dim[g.k]; g.vd = c->v_dim[g.k];
    g.qd = g.nh * g.hd; g.kd = g.kvh * g.hd; g.vdd = g.kvh * g.vd; g.rw = g.qd + g.kd + g.vdd;
    g.half = c->rope_dim[g.k] / 2;
    g.rows = m->L[li].rows;
    return g;
}

static int mc_ctensor(DW *d) { return dw_upload(d) && d->vk; }

/* The model's parameters on the device, its tensors resolved, its caches allocated;
 * NULL = the chain cannot run. */
static MimoChain *mc_setup(Model *m) {
    MimoChain *ch = (MimoChain *)m->vkchain;
    if (ch) return ch->ok ? ch : NULL;
    ch = (MimoChain *)calloc(1, sizeof *ch);
    if (!ch) return NULL;
    m->vkchain = ch;
    Cfg *c = &m->c; int L = c->n_layers, H = c->hidden;
    for (int k = 0; k < 2; k++)
        if (c->head_dim[k] > 256 || c->v_dim[k] > 256) {
            fprintf(stderr, "[VK] mimo chain: head dims %d/%d are past its attention shader's 256; per-matrix path\n",
                    c->head_dim[k], c->v_dim[k]);
            return NULL;
        }
    ch->o_ln1 = calloc(L, sizeof(size_t)); ch->o_ln2 = calloc(L, sizeof(size_t)); ch->o_sink = calloc(L, sizeof(size_t));
    ch->o_kvd = calloc(L, sizeof(size_t)); ch->kv_valid = calloc(L, sizeof(int));
    ch->kc = calloc(L, sizeof(void *)); ch->vc = calloc(L, sizeof(void *));
    if (!ch->o_ln1 || !ch->o_ln2 || !ch->o_sink || !ch->o_kvd || !ch->kv_valid || !ch->kc || !ch->vc) return NULL;
    size_t n = 0;
    for (int i = 0; i < L; i++) {
        ch->o_ln1[i] = n; n += H; ch->o_ln2[i] = n; n += H;
        if (m->L[i].sink) { ch->o_sink[i] = n; n += c->heads[c->swa[i]]; }
    }
    ch->o_final = n; n += H;
    float *arena = calloc(n, sizeof(float));
    if (!arena) return NULL;
    for (int i = 0; i < L; i++) {
        Layer *l = &m->L[i];
        memcpy(arena + ch->o_ln1[i], l->ln1, H * sizeof(float));
        memcpy(arena + ch->o_ln2[i], l->ln2, H * sizeof(float));
        if (l->sink) memcpy(arena + ch->o_sink[i], l->sink, c->heads[c->swa[i]] * sizeof(float));
    }
    memcpy(arena + ch->o_final, m->norm, H * sizeof(float));
    ch->prm = vkc_buf(n * sizeof(float), VKC_DEV);
    int ok = ch->prm && vkc_begin() && vkc_write(ch->prm, 0, arena, n * sizeof(float)) && vkc_submit(1);
    free(arena);
    if (!ok) return NULL;
    /* the matrices: the device copies the per-matrix path uploads (vk_dense_upload) */
    for (int i = 0; i < L && ok; i++) {
        Layer *l = &m->L[i];
        ok = mc_ctensor(&l->qkv) && mc_ctensor(&l->o) &&
             (c->moe[i] || (mc_ctensor(&l->gate) && mc_ctensor(&l->up) && mc_ctensor(&l->down)));
    }
    ok = ok && mc_ctensor(&m->head);
    if (!ok) { fprintf(stderr, "[VK] mimo chain: a dense matrix did not reach the device; per-matrix path\n"); return NULL; }
    /* the caches, in the host's layout and at its size */
    size_t kv = 0;
    for (int i = 0; i < L; i++) {
        McGeo g = mc_geo(m, i);
        ch->kc[i] = vkc_buf((size_t)g.rows * g.kd * sizeof(float), VKC_DEV);
        ch->vc[i] = vkc_buf((size_t)g.rows * g.vdd * sizeof(float), VKC_DEV);
        if (!ch->kc[i] || !ch->vc[i]) {
            fprintf(stderr, "[VK] mimo chain: device memory for the KV caches refused; per-matrix path\n");
            return NULL;
        }
        kv += (size_t)g.rows * (g.kd + g.vdd);
    }
    ch->ok = 1;
    int swa = 0;
    for (int i = 0; i < L; i++) swa += c->swa[i];
    fprintf(stderr, "[VK] mimo chain: %d layers on the device (%d sliding window), %.1f MiB of KV mirrors\n",
            L, swa, kv * 4 / 1048576.0);
    return ch;
}

static int mc_res(VkcBuf **b, size_t floats, int kind) { return vkc_reserve(b, (floats ? floats : 1) * sizeof(float), kind); }

/* scratch for `rows` rows, `out` rows of logits */
static int mc_scratch(MimoChain *ch, Model *m, int rows, int out) {
    Cfg *c = &m->c; int H = c->hidden, L = c->n_layers;
    size_t r = (size_t)rows, rw = 0, kd = 0, vdd = 0, ctxw = 0, wkd = 0, wvd = 0, kvd = 0;
    int dense = 0, win = 0;
    for (int i = 0; i < L; i++) {
        McGeo g = mc_geo(m, i);
        if ((size_t)g.rw > rw) rw = g.rw;
        if ((size_t)g.kd > kd) kd = g.kd;
        if ((size_t)g.vdd > vdd) vdd = g.vdd;
        if ((size_t)g.nh * g.vd > ctxw) ctxw = (size_t)g.nh * g.vd;
        if (g.swa) {
            if ((size_t)g.kd > wkd) wkd = g.kd;
            if ((size_t)g.vdd > wvd) wvd = g.vdd;
            if (g.rows - 1 > win) win = g.rows - 1;
        }
        int cap = g.swa && g.rows < rows ? g.rows : rows;
        ch->o_kvd[i] = kvd; kvd += (size_t)cap * (g.kd + g.vdd);
        dense |= !c->moe[i];
    }
    ch->rd_off[0] = 0; ch->rd_off[1] = rows * 2 * (c->rope_dim[0] / 2);
    size_t cs = r * 2 * (size_t)(c->rope_dim[0] / 2 + c->rope_dim[1] / 2);
    int ok = mc_res(&ch->x, r * H, VKC_DEV) && mc_res(&ch->nrm, r * H, VKC_DEV) && mc_res(&ch->tmp, r * H, VKC_DEV) &&
             mc_res(&ch->qkv, r * rw, VKC_DEV) && mc_res(&ch->kn, r * kd, VKC_DEV) && mc_res(&ch->vn, r * vdd, VKC_DEV) &&
             mc_res(&ch->ctx, r * ctxw, VKC_DEV) && mc_res(&ch->fin, (size_t)out * H, VKC_DEV) &&
             mc_res(&ch->h2d, r * H, VKC_DOWN) && mc_res(&ch->kvd, kvd, VKC_DOWN) &&
             mc_res(&ch->outd, (size_t)out * c->vocab, VKC_DOWN) && mc_res(&ch->routed, r * H, VKC_UP) &&
             mc_res(&ch->cs, cs, VKC_UP);
    if (ok && wkd) ok = mc_res(&ch->wk, ((size_t)win + r) * wkd, VKC_DEV) && mc_res(&ch->wv, ((size_t)win + r) * wvd, VKC_DEV);
    if (ok && dense) ok = mc_res(&ch->g, r * c->dense_inter, VKC_DEV) && mc_res(&ch->u, r * c->dense_inter, VKC_DEV);
    if (!ok) return 0;
    if (ch->rows < rows) {
        float *ho = realloc(ch->host_out, r * H * sizeof(float));
        if (!ho) return 0;
        ch->host_out = ho;
    }
    ch->rows = rows;   /* o_kvd and rd_off are laid out for exactly this many */
    return 1;
}

/* ---- the KV caches between the host and the device ------------------------------- */
/* A CPU step from pos_base writes the host's rows from there on. */
static void mc_cpu_step(Model *m, int pos_base) {
    MimoChain *ch = (MimoChain *)m->vkchain;
    if (!ch || !ch->ok) return;
    for (int i = 0; i < m->c.n_layers; i++) if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
}
/* The host's rings were replaced (a photo restored): the device's no longer match. */
static void mc_rings_rewritten(Model *m) {
    MimoChain *ch = (MimoChain *)m->vkchain;
    if (!ch || !ch->ok) return;
    for (int i = 0; i < m->c.n_layers; i++) if (m->c.swa[i]) ch->kv_valid[i] = 0;
}
/* host rows [t0, t1) of a ring into the device's ring (at most two runs of slots) */
static int mc_ring_up(VkcBuf *dst, const float *src, int R, int t0, int t1, int w) {
    int ok = 1;
    while (ok && t0 < t1) {
        int slot = t0 % R, run = R - slot < t1 - t0 ? R - slot : t1 - t0;
        ok = vkc_write(dst, (size_t)slot * w, src + (size_t)slot * w, (size_t)run * w * sizeof(float));
        t0 += run;
    }
    return ok;
}
/* device rows of positions [t0, t1) between a ring (rows R) and a linear buffer whose
 * row 0 is position `base`; to_ring 1: linear -> ring, 0: ring -> linear */
static int mc_ring_copy(VkcBuf *ring, VkcBuf *lin, int R, int base, int t0, int t1, int w, int to_ring) {
    int ok = 1;
    while (ok && t0 < t1) {
        int slot = t0 % R, run = R - slot < t1 - t0 ? R - slot : t1 - t0;
        size_t ro = (size_t)slot * w, lo = (size_t)(t0 - base) * w, nf = (size_t)run * w;
        ok = to_ring ? vkc_copy(ring, ro, lin, lo, nf) : vkc_copy(lin, lo, ring, ro, nf);
        t0 += run;
    }
    return ok;
}
/* Record the uploads that make the device's caches the host's, as the step at pos_base
 * reads them. */
static int mc_push_state(MimoChain *ch, Model *m, int pos_base) {
    Cfg *c = &m->c; int ok = 1;
    for (int i = 0; i < c->n_layers && ok; i++) {
        McGeo g = mc_geo(m, i);
        Layer *l = &m->L[i];
        if (ch->kv_valid[i] > pos_base) ch->kv_valid[i] = pos_base;
        if (g.swa) {
            /* the positions the window still sees must be in the host's ring, as the
             * CPU's attention() demands */
            int lo = pos_base - (c->window - 1) > 0 ? pos_base - (c->window - 1) : 0;
            for (int p = lo; p < pos_base; p++)
                if (l->ring_pos[p % g.rows] != p) { fprintf(stderr, "[mimo] window ring lost position %d\n", p); exit(1); }
            int from = ch->kv_valid[i] > pos_base - g.rows ? ch->kv_valid[i] : pos_base - g.rows;
            if (from < 0) from = 0;
            ok = mc_ring_up(ch->kc[i], l->K, g.rows, from, pos_base, g.kd) &&
                 mc_ring_up(ch->vc[i], l->V, g.rows, from, pos_base, g.vdd);
        } else if (ch->kv_valid[i] < pos_base) {
            int t0 = ch->kv_valid[i], n = pos_base - t0;
            ok = vkc_write(ch->kc[i], (size_t)t0 * g.kd, l->K + (size_t)t0 * g.kd, (size_t)n * g.kd * sizeof(float)) &&
                 vkc_write(ch->vc[i], (size_t)t0 * g.vdd, l->V + (size_t)t0 * g.vdd, (size_t)n * g.vdd * sizeof(float));
        }
        ch->kv_valid[i] = pos_base;
    }
    return ok;
}

/* ---- one layer's pieces ------------------------------------------------------------- */
static int mc_norm(VkcBuf *x, size_t xo, VkcBuf *w, size_t wo, VkcBuf *y, size_t yo, int rows, int D, float eps) {
    VkcNorm p = {rows, D, 1, (int)xo, D, D, (int)yo, D, D, (int)wo, 0, 0, eps, 1.f};
    return vkc_norm(x, w, y, &p);
}
static int mc_add(VkcBuf *x, VkcBuf *b, int n) {
    VkcEw p = {VKC_EW_ADD, n, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_ew(x, x, b, NULL, NULL, &p);
}
/* Attention of layer i for n rows at pb, o_proj into tmp; the new rows into the
 * device's cache and into kvd for the host. */
static int mc_attention(MimoChain *ch, Model *m, int i, int n, int pb) {
    Cfg *c = &m->c; Layer *l = &m->L[i];
    McGeo g = mc_geo(m, i);
    int ok = vkc_matmul((ColiVkTensor *)l->qkv.vk, ch->nrm, 0, ch->qkv, 0, n);
    if (ok && g.half) {   /* q's heads and k's heads are contiguous in a qkv row: one rotation */
        VkcRope p = {n * (g.nh + g.kvh), g.nh + g.kvh, 0, g.rw, g.hd, g.half, ch->rd_off[g.k], 2 * g.half};
        ok = vkc_rope(ch->qkv, ch->cs, &p);
    }
    VkcRegion *rg = ok ? malloc(sizeof *rg * (size_t)n) : NULL;
    if (!rg) return 0;
    for (int s = 0; s < n; s++) rg[s] = (VkcRegion){(size_t)s * g.kd, (size_t)s * g.rw + g.qd, (size_t)g.kd};
    ok = vkc_copy_regions(ch->kn, ch->qkv, rg, n);
    for (int s = 0; s < n; s++) rg[s] = (VkcRegion){(size_t)s * g.vdd, (size_t)s * g.rw + g.qd + g.kd, (size_t)g.vdd};
    ok = ok && vkc_copy_regions(ch->vn, ch->qkv, rg, n);
    free(rg);
    if (ok && c->v_scale != 1.0f) {   /* v *= attention_value_scale, as the CPU does before caching */
        VkcEw p = {VKC_EW_SCALE, n * g.vdd, 1, 1, 0, 1, 0, 0, 0, 0, 0, c->v_scale};
        ok = vkc_ew(ch->vn, ch->vn, NULL, NULL, NULL, &p);
    }
    if (!ok) return 0;
    VkcAttnW a;
    memset(&a, 0, sizeof a);
    a.a = (VkcAttn){n, g.nh, g.kvh, g.hd, pb, g.rows, 0, g.rw, g.hd, 0, 0, 0, 0, 0, g.nh * g.vd, 0, 0,
                    1.0f / sqrtf((float)g.hd), 0, 0};
    a.vd = g.vd; a.kv_pm = 1;
    a.sink = l->sink != NULL; a.sink_off = (int)ch->o_sink[i];
    VkcBuf *kc = ch->kc[i], *vc = ch->vc[i];
    if (!g.swa) {                      /* full attention: the rows in at pb, then every position */
        ok = vkc_copy(kc, (size_t)pb * g.kd, ch->kn, 0, (size_t)n * g.kd) &&
             vkc_copy(vc, (size_t)pb * g.vdd, ch->vn, 0, (size_t)n * g.vdd);
    } else if (n == 1) {               /* one row: its window is the ring, read in place */
        int slot = pb % g.rows;
        ok = vkc_copy(kc, (size_t)slot * g.kd, ch->kn, 0, g.kd) && vkc_copy(vc, (size_t)slot * g.vdd, ch->vn, 0, g.vdd);
        a.win = c->window; a.ring = g.rows;
    } else {                           /* a block: the window's earlier rows and the block's own, linear */
        int lo = pb - (c->window - 1) > 0 ? pb - (c->window - 1) : 0, nb = pb - lo;
        ok = mc_ring_copy(kc, ch->wk, g.rows, lo, lo, pb, g.kd, 0) && mc_ring_copy(vc, ch->wv, g.rows, lo, lo, pb, g.vdd, 0) &&
             vkc_copy(ch->wk, (size_t)nb * g.kd, ch->kn, 0, (size_t)n * g.kd) &&
             vkc_copy(ch->wv, (size_t)nb * g.vdd, ch->vn, 0, (size_t)n * g.vdd);
        a.a.pos_base = nb; a.win = c->window;
        kc = ch->wk; vc = ch->wv;
    }
    ok = ok && vkc_attn_w(ch->qkv, kc, vc, ch->ctx, NULL, NULL, ch->prm, &a);
    if (ok && g.swa && n > 1) {        /* the ring keeps the block's last `rows` positions */
        int from = n > g.rows ? n - g.rows : 0;
        ok = mc_ring_copy(ch->kc[i], ch->kn, g.rows, pb, pb + from, pb + n, g.kd, 1) &&
             mc_ring_copy(ch->vc[i], ch->vn, g.rows, pb, pb + from, pb + n, g.vdd, 1);
    }
    /* for the host: every row of a full layer, the ring's share of a windowed one */
    int from = g.swa && n > g.rows ? n - g.rows : 0, cap = g.swa && g.rows < ch->rows ? g.rows : ch->rows;
    ok = ok && vkc_copy(ch->kvd, ch->o_kvd[i], ch->kn, (size_t)from * g.kd, (size_t)(n - from) * g.kd) &&
         vkc_copy(ch->kvd, ch->o_kvd[i] + (size_t)cap * g.kd, ch->vn, (size_t)from * g.vdd, (size_t)(n - from) * g.vdd);
    return ok && vkc_matmul((ColiVkTensor *)l->o.vk, ch->ctx, 0, ch->tmp, 0, n);
}
/* layer 0's dense MLP: tmp = down(silu(gate(nrm)) * up(nrm)) */
static int mc_dense_mlp(MimoChain *ch, Model *m, int i, int n) {
    Layer *l = &m->L[i];
    int F = l->gate.O;
    VkcEw p = {VKC_EW_SWIGLU, n * F, F, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
    return vkc_matmul((ColiVkTensor *)l->gate.vk, ch->nrm, 0, ch->g, 0, n) &&
           vkc_matmul((ColiVkTensor *)l->up.vk, ch->nrm, 0, ch->u, 0, n) &&
           vkc_ew(ch->g, ch->g, ch->u, NULL, NULL, &p) &&
           vkc_matmul((ColiVkTensor *)l->down.vk, ch->g, 0, ch->tmp, 0, n);
}

/* The rows the step at pb wrote, into the host's caches (the CPU's own placement). */
static void mc_commit(MimoChain *ch, Model *m, int n, int pb) {
    const float *kv = (const float *)vkc_ptr(ch->kvd);
    for (int i = 0; i < m->c.n_layers; i++) {
        McGeo g = mc_geo(m, i);
        Layer *l = &m->L[i];
        int from = g.swa && n > g.rows ? n - g.rows : 0, cap = g.swa && g.rows < ch->rows ? g.rows : ch->rows;
        const float *K = kv + ch->o_kvd[i], *V = K + (size_t)cap * g.kd;
        for (int t = from; t < n; t++) {
            int p = pb + t, slot = g.swa ? p % g.rows : p;
            memcpy(l->K + (size_t)slot * g.kd, K + (size_t)(t - from) * g.kd, (size_t)g.kd * sizeof(float));
            memcpy(l->V + (size_t)slot * g.vdd, V + (size_t)(t - from) * g.vdd, (size_t)g.vdd * sizeof(float));
            if (g.swa) l->ring_pos[slot] = p;
        }
        ch->kv_valid[i] = pb + n;
    }
}

/* Every layer for the S rows h (the embedded rows, pictures spliced) at pos_base, the
 * logits as forward() wants them (NULL: none; all_rows: [S][V], else the last row's).
 * Returns how many rows it computed: S, or fewer when the device was lost or the chain
 * declined -- the host's caches then hold every position before pos_base + that many,
 * and the CPU runs the rest. */
static int mc_forward(Model *m, const float *h, int S, int pos_base, float *logits, int all_rows) {
    if (!g_vk_chain) return 0;
    if (g_vk_chain == COLI_VK_CHAIN_PREFILL && S <= 2) return 0;   /* prompts only: decode on the CPU */
    MimoChain *ch = mc_setup(m);
    if (!ch || ch->failed) return 0;
    Cfg *c = &m->c; int H = c->hidden, L = c->n_layers, V = c->vocab;
    int CH = mc_chunk_rows(), rows = S < CH ? S : CH;
    int out = logits && all_rows ? rows : 1;
    if (!mc_scratch(ch, m, rows, out)) {
        fprintf(stderr, "[VK] mimo chain: device memory for %d rows refused; per-matrix path\n", rows);
        ch->failed = 1;
        return 0;
    }
    vkc_gemm_rows(-1);
    int done = 0;
    for (int c0 = 0; c0 < S; c0 += rows) {
        int n = S - c0 < rows ? S - c0 : rows, pb = pos_base + c0, last = c0 + n == S;
        if (!vkc_begin() || !vkc_write(ch->x, 0, h + (size_t)c0 * H, (size_t)n * H * sizeof(float)) ||
            !mc_push_state(ch, m, pb)) goto lost;
        float *cs = (float *)vkc_ptr(ch->cs);   /* the CPU's angles, cosines and sines (rope()) */
        for (int k = 0; k < 2; k++) {
            int rd = c->rope_dim[k], half = rd / 2;
            for (int s = 0; s < n; s++) for (int j = 0; j < half; j++) {
                float inv = 1.0f / powf(c->theta[k], (float)(2 * j) / (float)rd);
                float ang = inv * (float)(pb + s);
                cs[ch->rd_off[k] + (s * half + j) * 2] = cosf(ang);
                cs[ch->rd_off[k] + (s * half + j) * 2 + 1] = sinf(ang);
            }
        }
        int ok = 1, pending = 0;
        for (int i = 0; i < L && ok; i++) {
            if (pending) ok = mc_add(ch->x, ch->routed, n * H);   /* the routed experts join, as h += out */
            pending = 0;
            ok = ok && mc_norm(ch->x, 0, ch->prm, ch->o_ln1[i], ch->nrm, 0, n, H, c->eps) &&
                 mc_attention(ch, m, i, n, pb) && mc_add(ch->x, ch->tmp, n * H) &&
                 mc_norm(ch->x, 0, ch->prm, ch->o_ln2[i], ch->nrm, 0, n, H, c->eps);
            if (ok && !c->moe[i]) ok = mc_dense_mlp(ch, m, i, n) && mc_add(ch->x, ch->tmp, n * H);
            else if (ok) {
                ok = vkc_copy(ch->h2d, 0, ch->nrm, 0, (size_t)n * H);
                double t0 = now_s();
                ok = ok && vkc_submit(1);
                ch->frames++; ch->wait_ms += (now_s() - t0) * 1e3;
                if (!ok) break;
                double t1 = now_s();
                moe(m, i, (const float *)vkc_ptr(ch->h2d), n, ch->host_out);
                memcpy(vkc_ptr(ch->routed), ch->host_out, (size_t)n * H * sizeof(float));
                ch->host_ms += (now_s() - t1) * 1e3;
                ok = vkc_begin();
                pending = 1;
            }
        }
        if (ok && pending) ok = mc_add(ch->x, ch->routed, n * H);
        int ro = logits && all_rows ? n : logits && last ? 1 : 0;
        if (ok && ro) ok = mc_norm(ch->x, (size_t)(n - ro) * H, ch->prm, ch->o_final, ch->fin, 0, ro, H, c->eps) &&
                           vkc_matmul((ColiVkTensor *)m->head.vk, ch->fin, 0, ch->outd, 0, ro);
        double t0 = now_s();
        ok = ok && vkc_submit(1);
        ch->frames++; ch->wait_ms += (now_s() - t0) * 1e3;
        if (!ok) goto lost;
        mc_commit(ch, m, n, pb);
        if (ro) memcpy(all_rows ? logits + (size_t)c0 * V : logits, vkc_ptr(ch->outd), (size_t)ro * V * sizeof(float));
        done = c0 + n;
    }
    ch->forwards++;
    return S;
lost:   /* a frame failed: the device is gone (or would not take a command); the CPU takes over */
    if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost(); }
    g_vk_chain = 0;
    ch->failed = 1;
    fprintf(stderr, "[VK] mimo chain: the device was lost at position %d; the host's caches hold every "
                    "position before it, and the CPU runs from there on\n", pos_base + done);
    return done;
}

static void mc_report(Model *m, const char *what) {
    MimoChain *ch = (MimoChain *)m->vkchain;
    if (!ch || !ch->ok || !ch->forwards) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] mimo chain: %llu forwards, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f ms of routed experts on the host, %.1f MiB on the device (%s)\n",
            ch->forwards, st.frames, st.ops, st.matmuls, st.gemms, ch->wait_ms, ch->host_ms,
            st.dev_bytes / 1048576.0, what);
    vkc_prof_print();
}
