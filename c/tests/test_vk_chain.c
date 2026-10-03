/* The dense chain's ops (vk_chain.c and shaders/chain_*.comp) against CPU references,
 * on any Vulkan device (CI: Lavapipe). Every op runs inside recorded frames the way an
 * engine uses them -- several ops per submission, a submission not waited for, the next
 * one ordered after it -- and its output is compared with a plain C version of what
 * qwen36.c / qwen38_core.h compute, within a float tolerance (the device sums in float
 * and in another order; the references sum in double):
 *   matmul   GEMV (chain_gemv.comp's vectorized one and qmatmul.comp's) and tiled
 *            GEMM, int8 rows, int4-g64, f32, bf16, offsets the device can bind and
 *            offsets it cannot (through the frame's temporaries)
 *   norm     zero-centred, plain, no weight, L2; heads with a gate between them, a
 *            weight slice per stream, in place
 *   rope     rotate-half from a host table, heads at a stride
 *   attn     causal GQA with the output gate over a cache at an offset, prefill rows
 *            after earlier ones, and a selection list
 *   dnconv   both orders of the sum, the ring carried across calls, the snapshot row
 *   dnrec    KD 8 and 128, silu and sigmoid gates, the state carried, the snapshot
 *   ew       every element-wise op
 *   qsa      block keys, and the selection with ties broken as the CPU sorts
 *   ple      the gate and the dilated convolution with its ring and snapshot
 *   frames   vkc_write, vkc_read, a frame left in flight and one ordered after it
 *   mla      the per-head weight blocks in every format, both ways, with a gate; RoPE
 *            interleaved and rotate-half, in place and into a cache row; LayerNorm;
 *            the layer op against colibri.c's absorbed attention over several
 *            geometries (q latent or none, NoPE, kv_b or the split halves, selection
 *            lists with skipped entries, a gate, prefill rows, a long context from
 *            kv_start, latents up to 1024)
 *   dsa      the indexer's top-k selection bit for bit, ties included
 *   kpool    the k-pooled indexer (GLM-5.3): pooled keys, the pools' rank order and
 *            the tail, slot for slot against sparse_index.h
 *   kda      GLM-5.3's KDA layer around delta_attention.h's step: the convolution
 *            and its window, the recurrence and its state over two submissions, the
 *            decay, beta, output norm and gate
 *   mhc      hyper_connections.h's split, collapse and write back, the mean, and the
 *            clamped SwiGLU
 *   dsv4     DeepSeek V4.1 / V4 (deepseek_v41.c): the sparse attention with a sink over
 *            window and compressed rows (V4's bf16 roundings too), interleaved RoPE
 *            both ways, the compressor's ring (V4's overlapping form), the indexer's
 *            scores, candidate blocks and top-k slot for slot, the engram gate
 *   dsv4 rounding  DeepSeek V4's roundings (deepseek_v4.c): bf16, E4M3 and E2M1 per block,
 *            the Hadamard transform, bit for bit; its SwiGLU within one bf16 step
 *
 *   make vk-chain-check VK=1   (VK_ICD_FILENAMES=.../lvp_icd.json for Lavapipe) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"
#include "../delta_attention.h"      /* the KDA step the KDA ops follow */
#include "../hyper_connections.h"    /* the mHC arithmetic */
#include "../sparse_index.h"         /* GLM-5.3's k-pooled indexer */
#include "../sparse_attn.h"          /* DeepSeek V4.1's attention kernel */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static unsigned rng = 12345;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static float frnd(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }
static float *fvec(size_t n, float scale) { float *v = malloc(n * sizeof *v); for (size_t i = 0; i < n; i++) v[i] = frnd() * scale; return v; }
static float sigm(float x) { return 1.f / (1.f + expf(-x)); }

/* max |a-b| / (max |b| + floor) */
static double relerr(const float *a, const float *b, size_t n, double floor) {
    double m = 0, e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > e) e = d; if (fabs(b[i]) > m) m = fabs(b[i]); }
    return e / (m + floor);
}
static int bad(const float *a, size_t n) { for (size_t i = 0; i < n; i++) if (!isfinite(a[i])) return 1; return 0; }

static VkcBuf *up(const float *v, size_t n) {   /* a device buffer holding v */
    VkcBuf *b = vkc_buf(n * 4, VKC_DEV);
    if (!b) return NULL;
    vkc_begin(); vkc_write(b, 0, v, n * 4); vkc_submit(1);
    return b;
}
static float *down(VkcBuf *b, size_t off, size_t n) {
    float *v = malloc(n * sizeof *v);
    if (!vkc_read(b, off, v, n * 4)) { memset(v, 0xff, n * sizeof *v); }
    return v;
}

/* ---- matmul ----------------------------------------------------------------------- */
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static void test_matmul(int fmt, int S, int I, int O, size_t xo, size_t yo) {
    float *x = fvec((size_t)S * I, 1.f), *W = malloc((size_t)O * I * sizeof(float));
    void *codes = NULL; float *sc = NULL; int gs = 0;
    if (fmt == 1) {                      /* int8 rows, one scale per row */
        int8_t *q = malloc((size_t)O * I); sc = malloc(O * sizeof *sc);
        for (int o = 0; o < O; o++) { sc[o] = 0.01f + (rnd() % 100) / 5000.f;
            for (int i = 0; i < I; i++) { q[(size_t)o * I + i] = (int8_t)((int)(rnd() % 255) - 127); W[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[o]; } }
        codes = q;
    } else if (fmt == 4) {               /* int4 (v+8 nibbles, low = even column), one scale per 64 */
        gs = 64; int ng = (I + 63) / 64;
        uint8_t *q = calloc((size_t)O * ((I + 1) / 2), 1); sc = malloc((size_t)O * ng * sizeof *sc);
        for (int o = 0; o < O; o++) for (int g = 0; g < ng; g++) sc[o * ng + g] = 0.02f + (rnd() % 100) / 3000.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int v = (int)(rnd() % 16); q[(size_t)o * ((I + 1) / 2) + i / 2] |= (uint8_t)(v << ((i & 1) * 4));
            W[(size_t)o * I + i] = (v - 8) * sc[o * ng + i / 64];
        }
        codes = q;
    } else if (fmt == 10) {
        float *w = fvec((size_t)O * I, 0.05f); memcpy(W, w, (size_t)O * I * sizeof(float)); codes = w;
    } else {                             /* 11 bf16 */
        uint16_t *w = malloc((size_t)O * I * 2);
        for (size_t i = 0; i < (size_t)O * I; i++) { w[i] = f2bf(frnd() * 0.05f); W[i] = bf2f(w[i]); }
        codes = w;
    }
    ColiVkTensor *t = NULL;
    if (!coli_vk_tensor_ensure(&t, codes, sc, fmt, I, O, gs)) { CHECK(0, "matmul fmt %d: upload", fmt); return; }
    float *ref = malloc((size_t)S * O * sizeof *ref);
    for (int s = 0; s < S; s++) for (int o = 0; o < O; o++) {
        double a = 0; for (int i = 0; i < I; i++) a += (double)x[(size_t)s * I + i] * W[(size_t)o * I + i];
        ref[(size_t)s * O + o] = (float)a;
    }
    VkcBuf *xb = vkc_buf((xo + (size_t)S * I) * 4, VKC_DEV), *yb = vkc_buf((yo + (size_t)S * O) * 4, VKC_DOWN);
    vkc_begin();
    vkc_write(xb, xo, x, (size_t)S * I * 4);
    int ok = vkc_matmul(t, xb, xo, yb, yo, S);
    vkc_submit(1);
    float *y = (float *)vkc_ptr(yb) + yo;
    double e = relerr(y, ref, (size_t)S * O, 1e-3);
    CHECK(ok && e < 2e-4 && !bad(y, (size_t)S * O), "matmul fmt %d S %d I %d O %d xo %zu yo %zu: ok %d err %.2e", fmt, S, I, O, xo, yo, ok, e);
    vkc_free(xb); vkc_free(yb); coli_vk_tensor_free(t);
    free(x); free(W); free(codes); free(sc); free(ref);
}

/* ---- norm ------------------------------------------------------------------------- */
static void test_norm(int flags, int inplace) {
    int rows = 3, per = 4, D = 40, seg = 52, rowst = per * seg + 7;  /* a gap after each head, like q's gate */
    size_t n = (size_t)rows * rowst;
    float *x = fvec(n, 2.f), *w = fvec((size_t)per * D, 0.5f), *ref = malloc(n * sizeof *ref);
    memcpy(ref, x, n * sizeof *ref);
    int wmod = flags & VKC_NORM_L2 ? 0 : per;
    for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
        const float *xs = x + r * rowst + j * seg; float *ys = ref + r * rowst + j * seg;
        double ss = 0; for (int i = 0; i < D; i++) ss += (double)xs[i] * xs[i];
        float rr = flags & VKC_NORM_L2 ? 1.f / sqrtf((float)ss + 1e-6f) : 1.f / sqrtf((float)(ss / D) + 1e-6f);
        const float *ws = w + (wmod ? (j % wmod) * D : 0);
        for (int i = 0; i < D; i++) {
            float wv = flags & VKC_NORM_NOW ? 1.f : flags & VKC_NORM_ADD1 ? 1.f + ws[i] : ws[i];
            ys[i] = xs[i] * rr * wv * 0.5f;
        }
    }
    VkcBuf *xb = up(x, n), *wb = up(w, (size_t)per * D), *yb = inplace ? xb : vkc_buf(n * 4, VKC_DEV);
    if (!inplace) { vkc_begin(); vkc_write(yb, 0, x, n * 4); vkc_submit(1); }
    VkcNorm p = {rows * per, D, per, 0, rowst, seg, 0, rowst, seg, 0, wmod, flags, 1e-6f, 0.5f};
    vkc_begin(); int ok = vkc_norm(xb, wb, yb, &p); vkc_submit(1);
    float *y = down(yb, 0, n);
    double e = relerr(y, ref, n, 1e-3);
    CHECK(ok && e < 1e-5, "norm flags %d inplace %d: err %.2e", flags, inplace, e);
    vkc_free(xb); vkc_free(wb); if (!inplace) vkc_free(yb);
    free(x); free(w); free(ref); free(y);
}

/* ---- rope ------------------------------------------------------------------------- */
static void test_rope(void) {
    int rows = 5, heads = 3, hd = 16, seg = 24, half = 4, rowst = heads * seg;
    size_t n = (size_t)rows * rowst;
    float *x = fvec(n, 1.f), *ref = malloc(n * sizeof *ref), *cs = malloc((size_t)rows * half * 2 * sizeof *cs);
    memcpy(ref, x, n * sizeof *ref);
    for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) {
        float ang = (float)(1000 + 37 * r) * powf(10000.f, -2.f * j / (2 * half));
        cs[(r * half + j) * 2] = cosf(ang); cs[(r * half + j) * 2 + 1] = sinf(ang);
        for (int h = 0; h < heads; h++) {
            float *v = ref + r * rowst + h * seg, a = v[j], b = v[j + half];
            v[j] = a * cs[(r * half + j) * 2] - b * cs[(r * half + j) * 2 + 1];
            v[j + half] = b * cs[(r * half + j) * 2] + a * cs[(r * half + j) * 2 + 1];
        }
    }
    (void)hd;
    VkcBuf *xb = up(x, n), *cb = up(cs, (size_t)rows * half * 2);
    VkcRope p = {rows * heads, heads, 0, rowst, seg, half, 0, half * 2};
    vkc_begin(); int ok = vkc_rope(xb, cb, &p); vkc_submit(1);
    float *y = down(xb, 0, n);
    double e = relerr(y, ref, n, 1e-3);
    CHECK(ok && e < 1e-6, "rope: err %.2e", e);
    vkc_free(xb); vkc_free(cb); free(x); free(ref); free(cs); free(y);
}

/* ---- attention ---------------------------------------------------------------------- */
static void test_attn(int S, int pos_base, int hd, int use_list) {
    int H = 4, KVH = 2, cap = 300, gsz = hd, qseg = hd + gsz, koff = 64;
    int T = pos_base + S;
    size_t kn = (size_t)koff + (size_t)KVH * cap * hd;
    float *q = fvec((size_t)S * H * qseg, 1.f), *kc = fvec(kn, 1.f), *vc = fvec(kn, 1.f);
    float scale = 1.f / sqrtf((float)hd);
    int selrow = 1 + T;
    int *sel = calloc((size_t)S * selrow, sizeof *sel);
    for (int s = 0; s < S; s++) {
        int vis = pos_base + s + 1;
        if (!use_list || s % 2) { sel[s * selrow] = -1; continue; }
        int n = 0;
        for (int t = 0; t < vis; t++) if (t % 3 != 1 || t == vis - 1) sel[s * selrow + 1 + n++] = t;
        sel[s * selrow] = n;
    }
    float *ref = malloc((size_t)S * H * hd * sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), vis = pos_base + s + 1;
        const int *list = sel + s * selrow;
        int n = list[0] >= 0 ? list[0] : vis;
        double *sc = malloc(n * sizeof *sc), mx = -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int t = list[0] >= 0 ? list[1 + j] : j;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[(size_t)s * H * qseg + h * qseg + d] * kc[koff + ((size_t)kvh * cap + t) * hd + d];
            sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
        }
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < hd; d++) {
            double a = 0;
            for (int j = 0; j < n; j++) { int t = list[0] >= 0 ? list[1 + j] : j; a += sc[j] / sum * vc[koff + ((size_t)kvh * cap + t) * hd + d]; }
            ref[((size_t)s * H + h) * hd + d] = (float)a * sigm(q[(size_t)s * H * qseg + h * qseg + hd + d]);
        }
        free(sc);
    }
    VkcBuf *qb = up(q, (size_t)S * H * qseg), *kb = up(kc, kn), *vb = up(vc, kn), *ob = vkc_buf((size_t)S * H * hd * 4, VKC_DOWN);
    VkcBuf *lb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    vkc_begin(); vkc_write(lb, 0, sel, (size_t)S * selrow * 4); vkc_submit(1);
    VkcAttn p = {S, H, KVH, hd, pos_base, cap, 0, H * qseg, qseg, hd, H * qseg, qseg, 1, 0, H * hd, 0, use_list ? selrow : 0, scale, koff, koff};
    vkc_begin(); int ok = vkc_attn(qb, kb, vb, ob, qb, lb, &p); vkc_submit(1);
    float *o = vkc_ptr(ob);
    double e = relerr(o, ref, (size_t)S * H * hd, 1e-3);
    CHECK(ok && e < 2e-5, "attn S %d pos %d hd %d list %d: err %.2e", S, pos_base, hd, use_list, e);
    vkc_free(qb); vkc_free(kb); vkc_free(vb); vkc_free(ob); vkc_free(lb);
    free(q); free(kc); free(vc); free(sel); free(ref);
}

/* ---- DeltaNet: convolution and recurrence ------------------------------------------ */
static void conv_ref(int S, int CD, int CK, const float *in, const float *w, float *ring, float *out, int order, int snap_row, float *snap) {
    for (int s = 0; s < S; s++) for (int c = 0; c < CD; c++) {
        float *rg = ring + c * (CK - 1); const float *wc = w + c * CK; float cur = in[s * CD + c], a;
        if (order == 0) { a = 0; for (int k = 0; k < CK - 1; k++) a += wc[k] * rg[k]; a += wc[CK - 1] * cur; }
        else { a = wc[CK - 1] * cur; for (int k = 0; k < CK - 1; k++) a += wc[k] * rg[k]; }
        out[s * CD + c] = a / (1.f + expf(-a));
        for (int k = 0; k < CK - 2; k++) rg[k] = rg[k + 1];
        rg[CK - 2] = cur;
        if (s == snap_row) memcpy(snap + c * (CK - 1), rg, (CK - 1) * sizeof(float));
    }
}
static void test_dnconv(int order) {
    int CD = 37, CK = 4, S1 = 5, S2 = 1, nh = CK - 1;
    float *w = fvec((size_t)CD * CK, 0.5f), *in = fvec((size_t)(S1 + S2) * CD, 1.f);
    float *ring = calloc((size_t)CD * nh, sizeof *ring), *out = malloc((size_t)(S1 + S2) * CD * sizeof *out);
    float *snap = calloc((size_t)CD * nh, sizeof *snap);
    conv_ref(S1, CD, CK, in, w, ring, out, order, 2, snap);
    float *snap_ref = malloc((size_t)CD * nh * sizeof *snap_ref); memcpy(snap_ref, snap, (size_t)CD * nh * sizeof *snap_ref);
    conv_ref(S2, CD, CK, in + S1 * CD, w, ring, out + S1 * CD, order, -1, NULL);
    VkcBuf *wb = up(w, (size_t)CD * CK), *ib = up(in, (size_t)(S1 + S2) * CD);
    VkcBuf *rb = vkc_buf((size_t)CD * nh * 2 * 4 + 64, VKC_DEV), *ob = vkc_buf((size_t)(S1 + S2) * CD * 4, VKC_DEV);
    int ro = 16, so = 16 + CD * nh;   /* ring and its snapshot in one buffer, at offsets */
    VkcDnConv p1 = {S1, CD, CK, 0, CD, 0, CD, 2, order, 0, ro, so};
    VkcDnConv p2 = {S2, CD, CK, S1 * CD, CD, S1 * CD, CD, -1, order, 0, ro, so};
    vkc_begin(); int ok = vkc_dnconv(ib, wb, rb, ob, rb, &p1); vkc_submit(0);     /* left in flight */
    vkc_begin(); ok &= vkc_dnconv(ib, wb, rb, ob, rb, &p2); vkc_submit(1);
    float *o = down(ob, 0, (size_t)(S1 + S2) * CD), *r = down(rb, ro, (size_t)CD * nh), *sn = down(rb, so, (size_t)CD * nh);
    double e1 = relerr(o, out, (size_t)(S1 + S2) * CD, 1e-3), e2 = relerr(r, ring, (size_t)CD * nh, 1e-3), e3 = relerr(sn, snap_ref, (size_t)CD * nh, 1e-3);
    CHECK(ok && e1 < 1e-6 && e2 == 0 && e3 == 0, "dnconv order %d: out %.2e ring %.2e snap %.2e", order, e1, e2, e3);
    vkc_free(wb); vkc_free(ib); vkc_free(rb); vkc_free(ob);
    free(w); free(in); free(ring); free(out); free(snap); free(snap_ref); free(o); free(r); free(sn);
}

static float softplus(float z) { return z > 20.f ? z : log1pf(expf(z)); }
/* qwen36.c's deltanet recurrence for S rows, from the conv output */
static void rec_ref(int S, int VH, int KH, int KD, int VD, const float *cv, const float *b, const float *a, const float *z,
                    const float *alog, const float *dtb, const float *nw, float *st, float *y, int sig_gate, int snap_row, float *snap) {
    int Kt = KH * KD, rep = VH / KH, CD = 2 * Kt + VH * VD;
    for (int s = 0; s < S; s++) for (int h = 0; h < VH; h++) {
        const float *row = cv + (size_t)s * CD;
        float q[256], k[256], o[128];
        double sq = 1e-6, sk = 1e-6;
        for (int d = 0; d < KD; d++) { q[d] = row[(h / rep) * KD + d]; k[d] = row[Kt + (h / rep) * KD + d]; sq += (double)q[d] * q[d]; sk += (double)k[d] * k[d]; }
        for (int d = 0; d < KD; d++) { q[d] = (float)(q[d] / sqrt(sq) / sqrt((double)KD)); k[d] = (float)(k[d] / sqrt(sk)); }
        float beta = sigm(b[s * VH + h]), decay = expf(-expf(alog[h]) * softplus(a[s * VH + h] + dtb[h]));
        float *Sh = st + (size_t)h * KD * VD;
        for (int t = 0; t < KD * VD; t++) Sh[t] *= decay;
        for (int v = 0; v < VD; v++) {
            float kv = 0; for (int d = 0; d < KD; d++) kv += k[d] * Sh[d * VD + v];
            float delta = (row[2 * Kt + h * VD + v] - kv) * beta;
            for (int d = 0; d < KD; d++) Sh[d * VD + v] += k[d] * delta;
        }
        double ms = 0;
        for (int v = 0; v < VD; v++) { float acc = 0; for (int d = 0; d < KD; d++) acc += q[d] * Sh[d * VD + v]; o[v] = acc; ms += (double)acc * acc; }
        float r = 1.f / sqrtf((float)(ms / VD) + 1e-6f);
        for (int v = 0; v < VD; v++) {
            float zv = z[(size_t)s * VH * VD + h * VD + v], g = sig_gate ? sigm(zv) : zv / (1.f + expf(-zv));
            y[(size_t)s * VH * VD + h * VD + v] = o[v] * r * nw[v] * g;
        }
        if (s == snap_row) memcpy(snap + (size_t)h * KD * VD, Sh, (size_t)KD * VD * sizeof(float));
    }
}
static void test_dnrec(int KD, int VD, int VH, int KH, int sig_gate) {
    int S1 = 3, S2 = 2, S = S1 + S2, Kt = KH * KD, CD = 2 * Kt + VH * VD;
    float *cv = fvec((size_t)S * CD, 1.f), *ab = fvec((size_t)S * VH * 2, 2.f), *z = fvec((size_t)S * VH * VD, 2.f);
    float *prm = fvec((size_t)2 * VH + VD, 1.f);
    float *st = fvec((size_t)VH * KD * VD, 0.1f), *st0 = malloc((size_t)VH * KD * VD * sizeof *st0);
    memcpy(st0, st, (size_t)VH * KD * VD * sizeof *st0);
    float *b = malloc((size_t)S * VH * sizeof *b), *a = malloc((size_t)S * VH * sizeof *a);
    for (int s = 0; s < S; s++) for (int h = 0; h < VH; h++) { b[s * VH + h] = ab[s * 2 * VH + h]; a[s * VH + h] = ab[s * 2 * VH + VH + h]; }
    float *y = malloc((size_t)S * VH * VD * sizeof *y), *snap = malloc((size_t)VH * KD * VD * sizeof *snap);
    rec_ref(S1, VH, KH, KD, VD, cv, b, a, z, prm, prm + VH, prm + 2 * VH, st, y, sig_gate, 1, snap);
    rec_ref(S2, VH, KH, KD, VD, cv + (size_t)S1 * CD, b + S1 * VH, a + S1 * VH, z + (size_t)S1 * VH * VD, prm, prm + VH, prm + 2 * VH,
            st, y + (size_t)S1 * VH * VD, sig_gate, -1, NULL);
    VkcBuf *cb = up(cv, (size_t)S * CD), *abb = up(ab, (size_t)S * VH * 2), *zb = up(z, (size_t)S * VH * VD), *pb = up(prm, (size_t)2 * VH + VD);
    size_t SN = (size_t)VH * KD * VD;
    VkcBuf *sb = vkc_buf(2 * SN * 4, VKC_DEV), *yb = vkc_buf((size_t)S * VH * VD * 4, VKC_DEV);
    vkc_begin(); vkc_write(sb, 0, st0, SN * 4); vkc_submit(1);
    VkcDnRec p1 = {S1, VH, KH, VD, Kt, 0, CD, 0, 2 * VH, VH, 2 * VH, 0, VH * VD, 0, VH * VD, 1, sig_gate, 1e-6f,
                   1.f / sqrtf((float)KD), 0, (int)SN, 0};
    VkcDnRec p2 = p1;
    p2.S = S2; p2.cv_off = S1 * CD; p2.b_off = S1 * 2 * VH; p2.a_off = S1 * 2 * VH + VH; p2.z_off = S1 * VH * VD; p2.y_off = S1 * VH * VD; p2.snap_row = -1;
    vkc_begin(); int ok = vkc_dnrec(KD, cb, abb, zb, sb, pb, yb, sb, &p1); vkc_submit(0);
    vkc_begin(); ok &= vkc_dnrec(KD, cb, abb, zb, sb, pb, yb, sb, &p2); vkc_submit(1);
    float *yd = down(yb, 0, (size_t)S * VH * VD), *sd = down(sb, 0, SN), *nd = down(sb, SN, SN);
    double e1 = relerr(yd, y, (size_t)S * VH * VD, 1e-3), e2 = relerr(sd, st, SN, 1e-3), e3 = relerr(nd, snap, SN, 1e-3);
    CHECK(ok && e1 < 5e-5 && e2 < 5e-6 && e3 < 5e-6, "dnrec KD %d VD %d gate %d: y %.2e state %.2e snap %.2e", KD, VD, sig_gate, e1, e2, e3);
    vkc_free(cb); vkc_free(abb); vkc_free(zb); vkc_free(pb); vkc_free(sb); vkc_free(yb);
    free(cv); free(ab); free(z); free(prm); free(st); free(st0); free(b); free(a); free(y); free(snap); free(yd); free(sd); free(nd);
}

/* ---- element-wise ------------------------------------------------------------------- */
static void test_ew(void) {
    int R = 3, D = 50, C = 4, W = C * D;
    size_t nw = (size_t)R * W;
    float *a = fvec(nw, 2.f), *b = fvec(nw, 2.f), *c = fvec(nw, 2.f), *e = fvec(R, 3.f), *ref = malloc(nw * sizeof *ref);
    VkcBuf *ab = up(a, nw), *bb = up(b, nw), *cb = up(c, nw), *eb = up(e, R), *yb = vkc_buf(nw * 4, VKC_DEV);
    struct { int op, flags, n; const char *name; } ops[] = {
        {VKC_EW_ADD, 0, R * D, "add"}, {VKC_EW_COMBINE, 1 | 2 | 4, R * D, "combine"}, {VKC_EW_COMBINE, 2 | 8, R * D, "combine-noresid"},
        {VKC_EW_COMBINE, 1, R * D, "combine-routed"}, {VKC_EW_SWIGLU, 0, R * D, "swiglu"}, {VKC_EW_HC_LOW, 0, R * D, "hc-low"},
        {VKC_EW_HC_MIX, 0, R * D, "hc-mix"}, {VKC_EW_HC_INJ, 0, R * C, "hc-inj"}, {VKC_EW_HC_APPLY, 0, R * W, "hc-apply"}};
    for (size_t k = 0; k < sizeof ops / sizeof *ops; k++) {
        int n = ops[k].n, op = ops[k].op, f = ops[k].flags;
        for (int i = 0; i < n; i++) {
            int r = i / D, d = i % D;
            switch (op) {
            case VKC_EW_ADD: ref[i] = a[i] + b[i]; break;
            case VKC_EW_COMBINE: { float t = f & 1 ? b[r * D + d] : 0.f;
                if (f & 2) t = t + (f & 4 ? sigm(e[r]) : 1.f) * c[r * D + d];
                ref[i] = f & 8 ? t : a[i] + t; break; }
            case VKC_EW_SWIGLU: ref[i] = a[i] / (1.f + expf(-a[i])) * b[i]; break;
            case VKC_EW_HC_LOW: { float v = a[i] / C; ref[i] = v * sigm(v); break; }
            case VKC_EW_HC_MIX: { float v = 0; for (int s = 0; s < C; s++) v += sigm(a[r * W + s * D + d]) * b[r * W + s * D + d]; ref[i] = v / C; break; }
            case VKC_EW_HC_INJ: ref[i] = 2.f * sigm(a[i] / C); break;
            case VKC_EW_HC_APPLY: { int rr = i / W, rem = i % W, s = rem / D, dd = rem % D;
                ref[i] = c[i] + a[rr * C + s] * b[rr * D + dd]; break; }
            }
        }
        if (op == VKC_EW_HC_APPLY) { vkc_begin(); vkc_write(yb, 0, c, nw * 4); vkc_submit(1); }
        VkcEw p = {op, n, D, C, f, 1, 0, 0, 0, 0, 0, (float)C};
        vkc_begin(); int ok = vkc_ew(yb, ab, bb, cb, eb, &p); vkc_submit(1);
        float *y = down(yb, 0, n);
        double err = relerr(y, ref, n, 1e-3);
        CHECK(ok && err < 1e-6, "ew %s: err %.2e", ops[k].name, err);
        free(y);
    }
    vkc_free(ab); vkc_free(bb); vkc_free(cb); vkc_free(eb); vkc_free(yb);
    free(a); free(b); free(c); free(e); free(ref);
}

/* ---- QSA: block keys and selection ----------------------------------------------------- */
static void test_qsa(void) {
    int ID = 12, R = 2, half = 3, cap = 40, nbmax = cap / R, IQ = 2, budget = 6;   /* take = 3 blocks */
    float *ik = fvec((size_t)cap * ID, 1.f), *w = fvec(ID, 0.3f), *pk = calloc((size_t)nbmax * ID, sizeof *pk);
    int nb = 13;   /* blocks 0..12 */
    float *cs = malloc((size_t)nb * half * 2 * sizeof *cs);
    for (int b = 0; b < nb; b++) for (int j = 0; j < half; j++) {
        float ang = (float)(b * R) * powf(10000.f, -2.f * j / (2 * half));
        cs[(b * half + j) * 2] = cosf(ang); cs[(b * half + j) * 2 + 1] = sinf(ang);
    }
    for (int b = 0; b < nb; b++) {
        float pool[64];
        for (int d = 0; d < ID; d++) { pool[d] = 0; for (int r = 0; r < R; r++) pool[d] += ik[(b * R + r) * ID + d] / R; }
        double ss = 0; for (int d = 0; d < ID; d++) ss += (double)pool[d] * pool[d];
        float rr = 1.f / sqrtf((float)(ss / ID) + 1e-6f);
        for (int d = 0; d < ID; d++) pool[d] = pool[d] * rr * (1.f + w[d]);
        float *o = pk + b * ID;
        for (int d = 0; d < ID; d++) o[d] = pool[d];
        for (int j = 0; j < half; j++) {
            float c = cs[(b * half + j) * 2], s = cs[(b * half + j) * 2 + 1], x0 = pool[j], x1 = pool[j + half];
            o[j] = x0 * c - x1 * s; o[j + half] = x1 * c + x0 * s;
        }
    }
    VkcBuf *ib = up(ik, (size_t)cap * ID), *wb = up(w, ID), *pb = vkc_buf((size_t)nbmax * ID * 4, VKC_DEV), *cb = up(cs, (size_t)nb * half * 2);
    /* blocks in two steps, as two forwards complete them */
    VkcQsa p0 = {0, ID, R, 0, half, 0, 0, 0, 0, 0, 0, 0, 0, 1e-6f, 0, 0, 0, 7};
    VkcQsa p1 = p0; p1.b0 = 7; p1.nb = nb - 7;
    VkcBuf *cb2 = up(cs + 7 * half * 2, (size_t)(nb - 7) * half * 2);
    vkc_begin(); int ok = vkc_qsa(ib, wb, pb, cb, NULL, NULL, &p0); ok &= vkc_qsa(ib, wb, pb, cb2, NULL, NULL, &p1); vkc_submit(1);
    float *pd = down(pb, 0, (size_t)nb * ID);
    double e = relerr(pd, pk, (size_t)nb * ID, 1e-3);
    CHECK(ok && e < 1e-5, "qsa block keys: err %.2e", e);
    /* selection: 4 query rows at positions 20..23; give two blocks the same key to test ties */
    memcpy(pk + 5 * ID, pk + 2 * ID, ID * sizeof(float));
    vkc_begin(); vkc_write(pb, 0, pk, (size_t)nbmax * ID * 4); vkc_submit(1);
    int S = 4, pos_base = 20, selrow = 1 + budget + R - 1;
    float *qi = fvec((size_t)S * IQ * ID, 1.f);
    VkcBuf *qb = up(qi, (size_t)S * IQ * ID), *sb = vkc_buf((size_t)S * 2 * nbmax * 4, VKC_DEV), *lb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    VkcQsa ps = {1, ID, R, 0, 0, S, pos_base, budget, IQ, 0, IQ * ID, nbmax, selrow, 1e-6f, 0, 0, 0, 0};
    vkc_begin(); ok = vkc_qsa(qb, NULL, pb, NULL, sb, lb, &ps); vkc_submit(1);
    int *sel = malloc((size_t)S * selrow * sizeof *sel);
    ok &= vkc_read(lb, 0, sel, (size_t)S * selrow * 4);
    for (int s = 0; s < S; s++) {
        int vis = pos_base + s + 1, blocks = vis / R, take = blocks < budget / R ? blocks : budget / R;
        float score[64]; int sel_ref[64], n = 0, chosen[64] = {0};
        for (int b = 0; b < blocks; b++) {
            float sc = 0; for (int h = 0; h < IQ; h++) { float a = 0; for (int d = 0; d < ID; d++) a += qi[(s * IQ + h) * ID + d] * pk[b * ID + d]; if (a > 0) sc += a; }
            score[b] = sc / sqrtf((float)ID);
        }
        for (int z = 0; z < take; z++) {   /* the CPU's sort: score desc, block asc */
            int best = -1; for (int b = 0; b < blocks; b++) if (!chosen[b] && (best < 0 || score[b] > score[best])) best = b;
            chosen[best] = 1;
        }
        for (int b = 0; b < blocks; b++) if (chosen[b]) for (int r = 0; r < R; r++) sel_ref[n++] = b * R + r;
        for (int t = blocks * R; t < vis; t++) sel_ref[n++] = t;
        int same = sel[s * selrow] == n;
        for (int j = 0; same && j < n; j++) same = sel[s * selrow + 1 + j] == sel_ref[j];
        CHECK(ok && same, "qsa selection row %d: count %d vs %d", s, sel[s * selrow], n);
    }
    vkc_free(ib); vkc_free(wb); vkc_free(pb); vkc_free(cb); vkc_free(cb2); vkc_free(qb); vkc_free(sb); vkc_free(lb);
    free(ik); free(w); free(pk); free(cs); free(pd); free(qi); free(sel);
}

/* ---- PLE ---------------------------------------------------------------------------- */
static void test_ple(void) {
    int S = 4, C = 3, H = 20, W = C * H, CK = 4, NG = 3, SL = (CK - 1) * NG;
    float *keys = fvec((size_t)S * W, 1.f), *hyp = fvec((size_t)S * W, 1.f), *val = fvec((size_t)S * H, 1.f);
    float *prm = fvec((size_t)3 * W, 0.3f), *conv = fvec((size_t)W * CK, 0.5f), *ring = fvec((size_t)W * SL, 0.5f);
    float *ring0 = malloc((size_t)W * SL * sizeof *ring0); memcpy(ring0, ring, (size_t)W * SL * sizeof *ring0);
    float *href = malloc((size_t)S * W * sizeof *href); memcpy(href, hyp, (size_t)S * W * sizeof *href);
    float *snap = malloc((size_t)W * SL * sizeof *snap);
    float *gated = malloc((size_t)S * W * sizeof *gated), *normv = malloc((size_t)S * W * sizeof *normv);
    for (int s = 0; s < S; s++) for (int b = 0; b < C; b++) {
        float kn[64], qn[64]; double sk = 0, sq = 0;
        for (int d = 0; d < H; d++) { sk += (double)keys[s * W + b * H + d] * keys[s * W + b * H + d]; sq += (double)hyp[s * W + b * H + d] * hyp[s * W + b * H + d]; }
        float rk = 1.f / sqrtf((float)(sk / H) + 1e-6f), rq = 1.f / sqrtf((float)(sq / H) + 1e-6f);
        for (int d = 0; d < H; d++) { kn[d] = keys[s * W + b * H + d] * rk * (1.f + prm[b * H + d]); qn[d] = hyp[s * W + b * H + d] * rq * (1.f + prm[W + b * H + d]); }
        float dot = 0; for (int d = 0; d < H; d++) dot += kn[d] * qn[d]; dot /= sqrtf((float)H);
        float g = sigm(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
        double sg = 0;
        for (int d = 0; d < H; d++) { gated[s * W + b * H + d] = g * val[s * H + d]; sg += (double)gated[s * W + b * H + d] * gated[s * W + b * H + d]; }
        float rg = 1.f / sqrtf((float)(sg / H) + 1e-6f);
        for (int d = 0; d < H; d++) normv[s * W + b * H + d] = gated[s * W + b * H + d] * rg * (1.f + prm[2 * W + b * H + d]);
    }
    for (int s = 0; s < S; s++) for (int d = 0; d < W; d++) {
        float a = conv[d * CK + CK - 1] * normv[s * W + d];
        for (int k = 0; k < CK - 1; k++) a += conv[d * CK + k] * ring[d * SL + k * NG];
        href[s * W + d] += gated[s * W + d] + a * sigm(a);
        float *rg = ring + d * SL; for (int k = 0; k < SL - 1; k++) rg[k] = rg[k + 1]; rg[SL - 1] = normv[s * W + d];
        if (s == 1) memcpy(snap + d * SL, rg, SL * sizeof(float));
    }
    VkcBuf *kb = up(keys, (size_t)S * W), *hb = up(hyp, (size_t)S * W), *vb = up(val, (size_t)S * H), *pb = up(prm, (size_t)3 * W);
    VkcBuf *cb = up(conv, (size_t)W * CK), *gb = vkc_buf((size_t)S * W * 4, VKC_DEV), *nb = vkc_buf((size_t)S * W * 4, VKC_DEV);
    VkcBuf *rb = vkc_buf((size_t)2 * W * SL * 4, VKC_DEV);
    vkc_begin(); vkc_write(rb, 0, ring0, (size_t)W * SL * 4); vkc_submit(1);
    VkcPle p0 = {0, S, C, H, CK, NG, 0, 0, 0, -1, 0, 1e-6f, 0, 0, 0};
    VkcPle p1 = {1, S, C, H, CK, NG, 0, 0, 0, 1, W * SL, 1e-6f, 0, 0, 0};
    vkc_begin(); int ok = vkc_ple(kb, hb, vb, pb, gb, nb, NULL, NULL, &p0) && vkc_ple(NULL, hb, NULL, NULL, gb, nb, cb, rb, &p1); vkc_submit(1);
    float *hd = down(hb, 0, (size_t)S * W), *rd = down(rb, 0, (size_t)W * SL), *sd = down(rb, (size_t)W * SL, (size_t)W * SL);
    double e1 = relerr(hd, href, (size_t)S * W, 1e-3), e2 = relerr(rd, ring, (size_t)W * SL, 1e-3), e3 = relerr(sd, snap, (size_t)W * SL, 1e-3);
    CHECK(ok && e1 < 2e-5 && e2 < 2e-5 && e3 < 2e-5, "ple: hyper %.2e ring %.2e snap %.2e", e1, e2, e3);
    vkc_free(kb); vkc_free(hb); vkc_free(vb); vkc_free(pb); vkc_free(cb); vkc_free(gb); vkc_free(nb); vkc_free(rb);
    free(keys); free(hyp); free(val); free(prm); free(conv); free(ring); free(ring0); free(href); free(snap); free(gated); free(normv);
    free(hd); free(rd); free(sd);
}

/* ---- MLA: weight blocks, RoPE and LayerNorm rows, the layer op, the DSA selection ------ */
static float e4m3f(uint8_t b) {   /* the CPU's E4M3_LUT */
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m * 0.001953125f : ldexpf(1.f + m * 0.125f, e - 7);
    return (b & 0x80) ? -v : v;
}
/* A resident [O x I] tensor in fmt with random codes (values about `scale`); W gets the
 * dequantized weights. Groups of 64 for the grouped formats. */
static ColiVkTensor *mk_tensor(int fmt, int O, int I, float scale, float *W) {
    void *codes = NULL; float *sc = NULL; int gs = 0, ng64 = (I + 63) / 64;
    size_t n = (size_t)O * I;
    if (fmt == 1 || fmt == 13) {
        int8_t *q = malloc(n); gs = fmt == 13 ? 64 : 0;
        int ns = fmt == 13 ? O * ng64 : O;
        sc = malloc((size_t)ns * sizeof *sc);
        for (int k = 0; k < ns; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 127.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            q[(size_t)o * I + i] = (int8_t)((int)(rnd() % 255) - 127);
            W[(size_t)o * I + i] = q[(size_t)o * I + i] * sc[fmt == 13 ? o * ng64 + i / 64 : o];
        }
        codes = q;
    } else if (fmt == 2 || fmt == 4) {
        size_t rb = (size_t)(I + 1) / 2; uint8_t *q = calloc((size_t)O * rb, 1); gs = fmt == 4 ? 64 : 0;
        int ns = fmt == 4 ? O * ng64 : O;
        sc = malloc((size_t)ns * sizeof *sc);
        for (int k = 0; k < ns; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 7.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int v = (int)(rnd() % 16);
            q[(size_t)o * rb + i / 2] |= (uint8_t)(v << ((i & 1) * 4));
            W[(size_t)o * I + i] = (v - 8) * sc[fmt == 4 ? o * ng64 + i / 64 : o];
        }
        codes = q;
    } else if (fmt == 5) {               /* int3-g64: 16 bytes of 2-bit lows, 8 of high bits, a scale per 64 */
        size_t rb = (size_t)ng64 * 24; uint8_t *q = calloc((size_t)O * rb, 1);
        sc = malloc((size_t)O * ng64 * sizeof *sc);
        for (int k = 0; k < O * ng64; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 4.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            int u = (int)(rnd() % 8), g = i / 64, j = i % 64;
            uint8_t *gb = q + (size_t)o * rb + (size_t)g * 24;
            gb[j / 4] |= (uint8_t)((u & 3) << ((j & 3) * 2));
            gb[16 + j / 8] |= (uint8_t)(((u >> 2) & 1) << (j & 7));
            W[(size_t)o * I + i] = (u - 4) * sc[o * ng64 + g];
        }
        codes = q;
    } else if (fmt == 12) {              /* fp8 e4m3, a scale per 64 */
        uint8_t *q = malloc(n); gs = 64;
        sc = malloc((size_t)O * ng64 * sizeof *sc);
        for (int k = 0; k < O * ng64; k++) sc[k] = scale * (0.5f + (rnd() % 100) / 100.f) / 8.f;
        for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
            uint8_t b; do b = (uint8_t)rnd(); while ((b & 0x7f) == 0x7f || ((b >> 3) & 15) > 10);
            q[(size_t)o * I + i] = b; W[(size_t)o * I + i] = e4m3f(b) * sc[o * ng64 + i / 64];
        }
        codes = q;
    } else if (fmt == 10) {
        float *w = fvec(n, scale); memcpy(W, w, n * sizeof *W); codes = w;
    } else {                             /* 11 bf16 */
        uint16_t *w = malloc(n * 2);
        for (size_t i = 0; i < n; i++) { w[i] = f2bf(frnd() * scale); W[i] = bf2f(w[i]); }
        codes = w;
    }
    ColiVkTensor *t = NULL;
    int ok = coli_vk_tensor_ensure(&t, codes, sc, fmt, I, O, gs);
    free(codes); free(sc);
    return ok ? t : NULL;
}

/* the per-head weight blocks, both directions, every format, a gate */
static void test_hgemv(int fmt, int trans) {
    int S = 3, H = 3, n = trans ? 24 : 20, I = 128, hstride = n + 9, hoff = 5, O = (H - 1) * hstride + hoff + n + 2;
    float *W = malloc((size_t)O * I * sizeof *W);
    ColiVkTensor *t = mk_tensor(fmt, O, I, 0.1f, W);
    if (!t) { CHECK(0, "hgemv fmt %d: upload", fmt); free(W); return; }
    int xin = trans ? n : I, xseg = xin + 3, xrow = H * xseg + 1, yout = trans ? I : n, yseg = yout + 2, yrow = H * yseg;
    float *x = fvec((size_t)S * xrow + 7, 1.f), *g = fvec((size_t)S * H * n, 2.f);
    float *ref = calloc((size_t)S * yrow, sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const float *xs = x + 7 + (size_t)s * xrow + h * xseg;
        float *ys = ref + (size_t)s * yrow + h * yseg;
        if (trans) for (int i = 0; i < I; i++) {
            double a = 0; for (int d = 0; d < n; d++) a += (double)W[(size_t)(h * hstride + hoff + d) * I + i] * xs[d];
            ys[i] = (float)a;
        } else for (int o = 0; o < n; o++) {
            double a = 0; for (int i = 0; i < I; i++) a += (double)W[(size_t)(h * hstride + hoff + o) * I + i] * xs[i];
            ys[o] = (float)a * sigm(g[(size_t)s * H * n + h * n + o]);
        }
    }
    VkcBuf *xb = up(x, (size_t)S * xrow + 7), *gb = up(g, (size_t)S * H * n), *yb = vkc_buf((size_t)S * yrow * 4, VKC_DEV);
    VkcHgemv p = {trans, S, H, n, hstride, hoff, 7, xrow, xseg, 0, yrow, yseg, 0, H * n, !trans};
    vkc_begin(); int ok = vkc_mla_hgemv(t, xb, yb, trans ? NULL : gb, &p); vkc_submit(1);
    float *y = down(yb, 0, (size_t)S * yrow);
    double e = relerr(y, ref, (size_t)S * yrow, 1e-3);
    CHECK(ok && e < 2e-5, "hgemv fmt %d trans %d: ok %d err %.2e", fmt, trans, ok, e);
    vkc_free(xb); vkc_free(gb); vkc_free(yb); coli_vk_tensor_free(t);
    free(W); free(x); free(g); free(ref); free(y);
}

/* interleaved and rotate-half RoPE, in place and into another place; LayerNorm */
static void test_mla_rows(void) {
    int rows = 4, per = 3, seg = 20, rd = 12, xrow = per * seg + 5, yrow = per * (seg + 2), half = rd / 2;
    float *x = fvec((size_t)rows * xrow, 1.f), *cs = malloc((size_t)rows * rd * sizeof *cs);
    for (int r = 0; r < rows; r++) for (int j = 0; j < half; j++) {
        float ang = (float)(500 + 91 * r) * powf(10000.f, -2.f * j / rd);
        cs[r * rd + 2 * j] = cosf(ang); cs[r * rd + 2 * j + 1] = sinf(ang);
    }
    VkcBuf *cb = up(cs, (size_t)rows * rd);
    for (int style = 0; style < 2; style++) for (int inplace = 0; inplace < 2; inplace++) {
        int yr = inplace ? xrow : yrow, ys = inplace ? seg : seg + 2;
        float *ref = calloc((size_t)rows * yr, sizeof *ref);
        if (inplace) memcpy(ref, x, (size_t)rows * xrow * sizeof *ref);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = x + r * xrow + j * seg; float *o = ref + r * yr + j * ys;
            for (int i = 0; i < half; i++) {
                float a = style ? v[2 * i] : v[i], b = style ? v[2 * i + 1] : v[i + half];
                float c = cs[r * rd + 2 * i], sn = cs[r * rd + 2 * i + 1];
                o[i] = a * c - b * sn; o[i + half] = b * c + a * sn;
            }
            for (int i = rd; i < seg; i++) o[i] = v[i];
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow), *yb = inplace ? xb : vkc_buf((size_t)rows * yr * 4, VKC_DEV);
        VkcMlaRow p = {rows * per, per, rd, seg, style, 0, xrow, seg, 0, yr, ys, 0, rd, 0, 0, 0, 0.f};
        vkc_begin(); int ok = vkc_mla_rope(xb, cb, yb, &p); vkc_submit(1);
        float *y = down(yb, 0, (size_t)rows * yr);
        double e = relerr(y, ref, (size_t)rows * yr, 1e-3);
        CHECK(ok && e < 1e-6, "mla rope style %d in place %d: ok %d err %.2e", style, inplace, ok, e);
        if (!inplace) vkc_free(yb);
        vkc_free(xb); free(ref); free(y);
    }
    /* LayerNorm, with and without the bias, into the next segment's place */
    float *prm = fvec(2 * seg, 1.f);
    VkcBuf *pb = up(prm, 2 * seg);
    for (int hb = 0; hb < 2; hb++) {
        float *ref = calloc((size_t)rows * yrow, sizeof *ref);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            const float *v = x + r * xrow + j * seg; float *o = ref + r * yrow + j * (seg + 2);
            double mu = 0, var = 0; for (int i = 0; i < seg; i++) mu += v[i]; mu /= seg;
            for (int i = 0; i < seg; i++) var += (v[i] - mu) * (v[i] - mu); var /= seg;
            float rr = 1.f / sqrtf((float)var + 1e-6f);
            for (int i = 0; i < seg; i++) o[i] = (float)(v[i] - mu) * rr * prm[i] + (hb ? prm[seg + i] : 0.f);
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow), *yb = vkc_buf((size_t)rows * yrow * 4, VKC_DEV);
        VkcMlaRow p = {rows * per, per, 0, seg, 0, 0, xrow, seg, 0, yrow, seg + 2, 0, 0, 0, seg, hb, 1e-6f};
        vkc_begin(); int ok = vkc_mla_lnorm(xb, pb, yb, &p); vkc_submit(1);
        float *y = down(yb, 0, (size_t)rows * yrow);
        double e = relerr(y, ref, (size_t)rows * yrow, 1e-3);
        CHECK(ok && e < 2e-6, "mla layernorm bias %d: ok %d err %.2e", hb, ok, e);
        vkc_free(xb); vkc_free(yb); free(ref); free(y);
    }
    vkc_free(pb); vkc_free(cb); free(prm); free(x); free(cs);
}

/* The MLA layer op against a double reference of colibri.c's absorbed attention:
 * geometry and formats from the caller; `split` gives the weights as [H*K x Q] and
 * [H*V x K] instead of kv_b; `list` gives every other row a selection list with a
 * skipped entry; `gate` a sigmoid gate on the values. The cache holds random rows
 * before pos_base (from kv_start); the step writes S more. Two frames: the step's
 * projections, then (after the host's selection lists) the attention. */
typedef struct { int H, Q, R, V, K, q_lora, D, S, pos_base, kv_start, style, split, list, gate, fq, fkv, fo; } MlaCase;
static void rmsref(double *y, const double *x, const float *w, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / n + eps); for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
}
static void test_mla(MlaCase c) {
    int H = c.H, Q = c.Q, R = c.R, V = c.V, K = c.K, D = c.D, S = c.S, pb = c.pos_base, QR = Q + R, KR = K + R;
    int cap = pb + S + 3, qin = c.q_lora > 0 ? c.q_lora : D, half = R / 2;
    float eps = 1e-5f, scale = 1.f / sqrtf((float)QR) * 1.1f;
    float *Wqa = c.q_lora ? malloc((size_t)c.q_lora * D * 4) : NULL, *Wqb = malloc((size_t)H * QR * qin * 4);
    float *Wkva = malloc((size_t)KR * D * 4), *Wo = malloc((size_t)D * H * V * 4);
    float *Wkvb = malloc((size_t)H * (Q + V) * K * 4), *Wk = malloc((size_t)H * K * Q * 4), *Wv = malloc((size_t)H * V * K * 4);
    VkcMla m = {H, Q, R, V, K, D, c.q_lora, eps, scale, c.style, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0};
    if (c.q_lora) m.q_a = mk_tensor(c.fq, c.q_lora, D, 1.f / sqrtf((float)D) * 2.f, Wqa);
    m.q_b = mk_tensor(c.fq, H * QR, qin, 1.f / sqrtf((float)qin) * 2.f, Wqb);
    m.kv_a = mk_tensor(c.fkv, KR, D, 1.f / sqrtf((float)D) * 2.f, Wkva);
    if (c.split) { m.k_abs = mk_tensor(c.fkv, H * K, Q, 1.f / sqrtf((float)Q), Wk); m.v_abs = mk_tensor(c.fkv, H * V, K, 1.f / sqrtf((float)K), Wv); }
    else {
        m.kv_b = mk_tensor(c.fkv, H * (Q + V), K, 1.f / sqrtf((float)K), Wkvb);
        for (int h = 0; h < H; h++) {      /* the same weights seen as the two halves */
            for (int d = 0; d < Q; d++) for (int i = 0; i < K; i++) Wk[((size_t)h * K + i) * Q + d] = Wkvb[((size_t)h * (Q + V) + d) * K + i];
            for (int v = 0; v < V; v++) for (int i = 0; i < K; i++) Wv[((size_t)h * V + v) * K + i] = Wkvb[((size_t)h * (Q + V) + Q + v) * K + i];
        }
    }
    m.o = mk_tensor(c.fo, D, H * V, 1.f / sqrtf((float)(H * V)), Wo);
    float *prm = malloc((size_t)(qin + K + 8) * 4);
    for (int i = 0; i < qin + K + 8; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    m.prm = up(prm, (size_t)qin + K + 8); m.q_norm = 3; m.kv_norm = 3 + qin;
    /* the cache: rows before pos_base as the device holds them (normalized, rotated) */
    float *lat = fvec((size_t)cap * K, 1.f), *rope = R ? fvec((size_t)cap * R, 1.f) : NULL;
    float *x = fvec((size_t)S * D, 1.f), *cs = R ? malloc((size_t)S * R * 4) : NULL, *gate = fvec((size_t)S * H * V, 2.f);
    for (int s = 0; s < S; s++) for (int j = 0; j < half; j++) {
        float ang = (float)(pb + s) * powf(10000.f, -2.f * j / R);
        cs[s * R + 2 * j] = cosf(ang); cs[s * R + 2 * j + 1] = sinf(ang);
    }
    int selrow = 2 + cap;
    int *sel = calloc((size_t)S * selrow, sizeof *sel);
    for (int s = 0; s < S; s++) {
        int vis = pb + s + 1, n = 0;
        if (!c.list || s % 2) { sel[s * selrow] = -1; continue; }
        for (int t = vis - 1; t >= c.kv_start; t -= 2) sel[s * selrow + 1 + n++] = t;   /* newest first, every other */
        sel[s * selrow + 1 + n++] = -1;                                              /* a skipped entry */
        sel[s * selrow] = n;
    }
    /* reference */
    double *q = calloc((size_t)S * H * QR, 8), *kvr = calloc((size_t)S * KR, 8), *qa = calloc((size_t)S * qin, 8);
    double *latr = malloc((size_t)cap * K * 8), *roper = malloc((size_t)cap * (R ? R : 1) * 8);
    for (size_t i = 0; i < (size_t)cap * K; i++) latr[i] = lat[i];
    for (size_t i = 0; R && i < (size_t)cap * R; i++) roper[i] = rope[i];
    for (int s = 0; s < S; s++) {
        const float *xs = x + (size_t)s * D;
        if (c.q_lora) {
            double *a = qa + (size_t)s * qin;
            for (int o = 0; o < c.q_lora; o++) { double t = 0; for (int i = 0; i < D; i++) t += (double)Wqa[(size_t)o * D + i] * xs[i]; a[o] = t; }
            rmsref(a, a, prm + 3, c.q_lora, eps);
        } else for (int i = 0; i < D; i++) qa[(size_t)s * qin + i] = xs[i];
        for (int o = 0; o < H * QR; o++) { double t = 0; for (int i = 0; i < qin; i++) t += (double)Wqb[(size_t)o * qin + i] * qa[(size_t)s * qin + i]; q[(size_t)s * H * QR + o] = t; }
        for (int o = 0; o < KR; o++) { double t = 0; for (int i = 0; i < D; i++) t += (double)Wkva[(size_t)o * D + i] * xs[i]; kvr[(size_t)s * KR + o] = t; }
        rmsref(latr + (size_t)(pb + s) * K, kvr + (size_t)s * KR, prm + 3 + qin, K, eps);
        for (int h = 0; h <= H; h++) {      /* the heads' rotated parts, then the shared key's */
            double *v = h < H ? q + (size_t)s * H * QR + h * QR + Q : kvr + (size_t)s * KR + K, in[256];
            double *o = h < H ? v : roper + (size_t)(pb + s) * R;
            for (int i = 0; i < R; i++) in[i] = v[i];
            for (int i = 0; i < half; i++) {
                double a = c.style ? in[2 * i] : in[i], b = c.style ? in[2 * i + 1] : in[i + half];
                double cc = cs[s * R + 2 * i], sn = cs[s * R + 2 * i + 1];
                o[i] = a * cc - b * sn; o[i + half] = b * cc + a * sn;
            }
        }
    }
    float *ref = malloc((size_t)S * D * 4);
    double *ctx = calloc((size_t)H * V, 8);
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) {
            const double *qs = q + (size_t)s * H * QR + h * QR;
            double qabs[1024], clat[1024], *sc = malloc((size_t)cap * 8);
            for (int i = 0; i < K; i++) { double t = 0; for (int d = 0; d < Q; d++) t += (double)Wk[((size_t)h * K + i) * Q + d] * qs[d]; qabs[i] = t; }
            const int *ls = sel + s * selrow;
            int n = ls[0] >= 0 ? ls[0] : pb + s + 1 - c.kv_start, used = 0;
            int *pos = malloc((size_t)(n + 1) * sizeof *pos);
            for (int j = 0; j < n; j++) { int t = ls[0] >= 0 ? ls[1 + j] : c.kv_start + j; if (t >= 0) pos[used++] = t; }
            double mx = -1e300, sum = 0;
            for (int j = 0; j < used; j++) {
                double a = 0; int t = pos[j];
                for (int i = 0; i < K; i++) a += qabs[i] * latr[(size_t)t * K + i];
                for (int i = 0; i < R; i++) a += qs[Q + i] * roper[(size_t)t * R + i];
                sc[j] = a * scale; if (sc[j] > mx) mx = sc[j];
            }
            for (int j = 0; j < used; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
            for (int i = 0; i < K; i++) { double a = 0; for (int j = 0; j < used; j++) a += sc[j] / sum * latr[(size_t)pos[j] * K + i]; clat[i] = a; }
            for (int v = 0; v < V; v++) {
                double a = 0; for (int i = 0; i < K; i++) a += (double)Wv[((size_t)h * V + v) * K + i] * clat[i];
                if (c.gate) a *= 1.0 / (1.0 + exp(-(double)gate[(size_t)s * H * V + h * V + v]));
                ctx[h * V + v] = a;
            }
            free(sc); free(pos);
        }
        for (int o = 0; o < D; o++) { double t = 0; for (int i = 0; i < H * V; i++) t += (double)Wo[(size_t)o * H * V + i] * ctx[i]; ref[(size_t)s * D + o] = (float)t; }
    }
    /* the device */
    VkcMlaCache cc = {up(lat, (size_t)cap * K), R ? up(rope, (size_t)cap * R) : NULL, cap};
    VkcMlaScratch scr = {0};
    VkcBuf *xb = vkc_buf((size_t)(S * D + 16) * 4, VKC_DEV), *csb = R ? up(cs, (size_t)S * R) : NULL;
    VkcBuf *gb = up(gate, (size_t)S * H * V), *ob = vkc_buf((size_t)S * D * 4, VKC_DEV);
    VkcBuf *dn = vkc_buf((size_t)S * KR * 4, VKC_DOWN), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DEV);
    int ok = vkc_mla_scratch(&scr, &m, S) && vkc_begin() && vkc_write(xb, 16, x, (size_t)S * D * 4) &&
             vkc_mla_qkv(&m, &scr, xb, 16, S, pb, csb, &cc, dn, 0) && vkc_submit(0) &&
             vkc_begin() && vkc_write(sb, 0, sel, (size_t)S * selrow * 4) &&
             vkc_mla_attn(&m, &scr, S, pb, c.kv_start, &cc, c.list ? sb : NULL, 0, selrow, c.gate ? gb : NULL, 0, ob, 0) &&
             vkc_submit(1);
    float *y = down(ob, 0, (size_t)S * D);
    double e = relerr(y, ref, (size_t)S * D, 1e-3);
    /* the new cache rows, as the host's copy receives them */
    float *nl = (float *)vkc_ptr(dn);
    double el = 0, er = 0;
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < K; i++) { double d = fabs(nl[(size_t)s * K + i] - latr[(size_t)(pb + s) * K + i]); if (d > el) el = d; }
        for (int i = 0; i < R; i++) { double d = fabs(nl[(size_t)S * K + (size_t)s * R + i] - roper[(size_t)(pb + s) * R + i]); if (d > er) er = d; }
    }
    CHECK(ok && e < 2e-5 && el < 2e-5 && er < 2e-5 && !bad(y, (size_t)S * D),
          "mla H %d Q %d R %d V %d K %d q_lora %d S %d pos %d kv_start %d style %d split %d list %d gate %d fmt %d/%d/%d: "
          "ok %d err %.2e latent %.2e rope %.2e", H, Q, R, V, K, c.q_lora, S, pb, c.kv_start, c.style, c.split, c.list, c.gate,
          c.fq, c.fkv, c.fo, ok, e, el, er);
    if (getenv("VKC_TEST_VERBOSE")) printf("mla case H %d K %d S %d: err %.2e latent %.2e rope %.2e\n", H, K, S, e, el, er);
    vkc_mla_scratch_free(&scr);
    vkc_free(cc.lat); vkc_free(cc.rope); vkc_free(xb); vkc_free(csb); vkc_free(gb); vkc_free(ob); vkc_free(dn); vkc_free(sb);
    vkc_free(m.prm);
    ColiVkTensor *ts[7] = {m.q_a, m.q_b, m.kv_a, m.kv_b, m.k_abs, m.v_abs, m.o};
    for (int i = 0; i < 7; i++) if (ts[i]) coli_vk_tensor_free(ts[i]);
    free(Wqa); free(Wqb); free(Wkva); free(Wo); free(Wkvb); free(Wk); free(Wv); free(prm); free(lat); free(rope);
    free(x); free(cs); free(gate); free(sel); free(q); free(kvr); free(qa); free(latr); free(roper); free(ref); free(ctx); free(y);
}

/* The DSA selection against colibri.c's, bit for bit: quarter-step queries and keys
 * and power-of-two scales make every score exact on both sides, and the many zero
 * scores (every head negative) make ties the selection must break as the CPU does. */
/* negw: every head weight <= 0, so every score is 0 or below and the zeros tie at the
 * threshold far beyond top-k (the second band's order over long chunks) */
static void test_dsa(int S, int pos_base, int IH, int ID, int topk, int force, int negw) {
    int T = pos_base + S, krow = ID + 3, qrow = IH * ID + 1, wrow = IH + 2, selrow = 1 + topk + 2, scrow = T + 5;
    /* the buffers start at offsets 1 (queries), 2 (weights) and 3 (keys) */
    float *q = malloc((size_t)(S * qrow + 1) * 4), *k = malloc((size_t)(T * krow + 3) * 4), *w = malloc((size_t)(S * wrow + 2) * 4);
    for (int i = 0; i < S * qrow + 1; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < T * krow + 3; i++) k[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * wrow + 2; i++) w[i] = ((int)(rnd() % 9) - (negw ? 8 : 4)) * 0.5f;
    for (int t = 0; t < T; t += 5) for (int i = 0; i < ID; i++) k[3 + t * krow + i] = k[3 + (t / 5) * krow + i];   /* equal keys: ties */
    float qs = 0.25f, wsc = 0.5f;
    int *ref = calloc((size_t)S * selrow, sizeof *ref), fails0 = fails;
    for (int s = 0; s < S; s++) {
        int nk = pos_base + s + 1, *dst = ref + s * selrow;
        if (nk <= topk && !force) { dst[0] = -1; continue; }
        int keep = nk < topk ? nk : topk;
        float *isc = malloc((size_t)nk * 4), *tmp = malloc((size_t)nk * 4);
        for (int t = 0; t < nk; t++) {
            float a = 0;
            for (int h = 0; h < IH; h++) {
                float d0 = 0; for (int i = 0; i < ID; i++) d0 += q[1 + s * qrow + h * ID + i] * k[3 + t * krow + i];
                d0 *= qs; if (d0 > 0) a += w[2 + s * wrow + h] * d0;
            }
            isc[t] = a * wsc;
        }
        memcpy(tmp, isc, (size_t)nk * 4);
        for (int a = 0; a < nk; a++) for (int b = a + 1; b < nk; b++) if (tmp[b] > tmp[a]) { float x = tmp[a]; tmp[a] = tmp[b]; tmp[b] = x; }
        float thr = tmp[keep - 1];
        int nd = 0;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] > thr) dst[1 + nd++] = t;
        for (int t = 0; t < nk && nd < keep; t++) if (isc[t] == thr) dst[1 + nd++] = t;
        dst[0] = nd;
        free(isc); free(tmp);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow + 1), *kb = up(k, (size_t)T * krow + 3), *wb = up(w, (size_t)S * wrow + 2);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DOWN);
    VkcDsa p = {S, pos_base, IH, ID, topk, force, 1, qrow, 2, wrow, 3, krow, scrow, selrow, qs, wsc};
    vkc_begin(); int ok = vkc_dsa_select(qb, wb, kb, scb, sb, &p); vkc_submit(1);
    const int *got = (const int *)vkc_ptr(sb);
    CHECK(ok, "dsa S %d pos %d: not recorded", S, pos_base);
    for (int s = 0; ok && s < S && fails == fails0; s++) {
        int n = ref[s * selrow] < 0 ? 0 : ref[s * selrow];
        int same = got[s * selrow] == ref[s * selrow];
        for (int j = 0; same && j < n; j++) same = got[s * selrow + 1 + j] == ref[s * selrow + 1 + j];
        CHECK(same, "dsa S %d pos %d IH %d ID %d topk %d force %d: row %d differs (count %d, expected %d)",
              S, pos_base, IH, ID, topk, force, s, got[s * selrow], ref[s * selrow]);
        if (!same && getenv("DSA_DEBUG")) { for (int j = 0; j < n; j++) printf(" %d/%d", got[s * selrow + 1 + j], ref[s * selrow + 1 + j]); printf("\n"); }
    }
    vkc_free(qb); vkc_free(kb); vkc_free(wb); vkc_free(scb); vkc_free(sb);
    free(q); free(k); free(w); free(ref);
}

/* ---- KDA, mHC, k-pooling ----------------------------------------------------------------- */
/* glm53.c's KDA layer around delta_attention.h's step, S rows over two calls (the state
 * and the window carried on the device between them) */
static void test_kda(int H, int KD, int CK, float xs) {   /* xs: the inputs' scale (small: the l2 eps counts) */
    int VD = KD, P = H * KD, C = 3 * P, S = 5;
    float lb = -5.f, neps = 1e-6f, eps = 1e-5f;
    float *qkv = fvec((size_t)S * C, xs), *f = fvec((size_t)S * P, 1.f), *b = fvec((size_t)S * H, 2.f), *g = fvec((size_t)S * P, 2.f);
    float *taps = fvec((size_t)C * CK, 0.7f), *win = fvec((size_t)C * CK, 1.f), *state = fvec((size_t)H * KD * VD, 0.3f);
    float *prm = malloc((size_t)(H + P + VD) * 4);
    for (int i = 0; i < H; i++) prm[i] = frnd() * 0.5f;
    for (int i = 0; i < P; i++) prm[H + i] = frnd() * 0.5f;
    for (int i = 0; i < VD; i++) prm[H + P + i] = 0.5f + (rnd() % 100) / 100.f;
    /* reference: glm53.c's kda_layer row by row */
    float *rs = malloc((size_t)H * KD * VD * 4), *rw = malloc((size_t)C * CK * 4), *ref = malloc((size_t)S * P * 4);
    memcpy(rs, state, (size_t)H * KD * VD * 4); memcpy(rw, win, (size_t)C * CK * 4);
    float *scratch = malloc((size_t)coli_kda_scratch_floats(H, KD, VD) * 4), *core = malloc((size_t)P * 4);
    float *dk = malloc((size_t)P * 4), *bt = malloc((size_t)H * 4);
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) for (int d = 0; d < KD; d++) {
            int i = h * KD + d; float sv = expf(prm[h]) * (f[s * P + i] + prm[H + i]);
            float sg = sv >= 0 ? 1.f / (1.f + expf(-sv)) : expf(sv) / (1.f + expf(sv));
            dk[i] = lb * sg;
        }
        for (int h = 0; h < H; h++) { float v = b[s * H + h]; bt[h] = v >= 0 ? 1.f / (1.f + expf(-v)) : expf(v) / (1.f + expf(v)); }
        coli_kda_step(core, rs, rw, qkv + (size_t)s * C, taps, dk, bt, H, KD, VD, CK, neps, scratch);
        for (int h = 0; h < H; h++) {
            float sq = 0; for (int d = 0; d < VD; d++) sq += core[h * VD + d] * core[h * VD + d];
            float inv = 1.f / sqrtf(sq / VD + eps);
            for (int d = 0; d < VD; d++) {
                float gv = g[s * P + h * VD + d], sg = gv >= 0 ? 1.f / (1.f + expf(-gv)) : expf(gv) / (1.f + expf(gv));
                ref[s * P + h * VD + d] = core[h * VD + d] * inv * prm[H + P + d] * sg;
            }
        }
    }
    /* the device: q, k, v in three blocks [3][S][P] (in_part = S*P) */
    float *blk = malloc((size_t)3 * S * P * 4);
    for (int s = 0; s < S; s++) for (int part = 0; part < 3; part++)
        memcpy(blk + (size_t)part * S * P + (size_t)s * P, qkv + (size_t)s * C + (size_t)part * P, (size_t)P * 4);
    VkcBuf *ib = up(blk, (size_t)3 * S * P), *tb = up(taps, (size_t)C * CK), *wb = up(win, (size_t)C * CK), *sb = up(state, (size_t)H * KD * VD);
    VkcBuf *fb = up(f, (size_t)S * P), *bb = up(b, (size_t)S * H), *gb = up(g, (size_t)S * P), *pb = up(prm, (size_t)H + P + VD);
    VkcBuf *mb = vkc_buf((size_t)S * C * 4, VKC_DEV), *yb = vkc_buf((size_t)S * P * 4, VKC_DEV);
    int ok = 1, s0 = 0;
    for (int part = 0; part < 2 && ok; part++) {
        int n = part ? S - 3 : 3;
        VkcKdaConv cp = {n, C, CK, P, s0 * P, P, S * P, 0, C, 0, 0};
        VkcKdaRec rp = {n, H, VD, P, 0, C, s0 * P, P, s0 * H, H, s0 * P, P, s0 * P, P, 0, 0, lb, neps, eps};
        ok = vkc_begin() && vkc_kda_conv(ib, tb, wb, mb, &cp) && vkc_kda_rec(KD, mb, fb, bb, gb, pb, sb, yb, &rp) && vkc_submit(part);
        s0 += n;
    }
    float *y = down(yb, 0, (size_t)S * P), *st2 = down(sb, 0, (size_t)H * KD * VD), *w2 = down(wb, 0, (size_t)C * CK);
    double e = relerr(y, ref, (size_t)S * P, 1e-3), es = relerr(st2, rs, (size_t)H * KD * VD, 1e-3), ew = relerr(w2, rw, (size_t)C * CK, 1e-3);
    CHECK(ok && e < 1e-5 && es < 1e-5 && ew == 0 && !bad(y, (size_t)S * P), "kda H %d KD %d K %d: ok %d out %.2e state %.2e window %.2e",
          H, KD, CK, ok, e, es, ew);
    if (getenv("VKC_TEST_VERBOSE")) printf("kda H %d KD %d: out %.2e state %.2e\n", H, KD, e, es);
    vkc_free(ib); vkc_free(tb); vkc_free(wb); vkc_free(sb); vkc_free(fb); vkc_free(bb); vkc_free(gb); vkc_free(pb); vkc_free(mb); vkc_free(yb);
    free(qkv); free(f); free(b); free(g); free(taps); free(win); free(state); free(prm); free(rs); free(rw); free(ref);
    free(scratch); free(core); free(dk); free(bt); free(blk); free(y); free(st2); free(w2);
}

/* hyper_connections.h's pre and post against the split, collapse and write back, the
 * mean, and the clamped SwiGLU */
static void test_mhc(int H, int D, int iters) {
    int S = 3, HD = H * D, nm = (2 + H) * H;
    float eps = 1e-5f, hce = 1e-6f, lim = 1.5f;
    float *x = fvec((size_t)S * HD, 1.f), *fn = fvec((size_t)nm * HD, 0.05f), *br = fvec((size_t)S * D, 1.f);
    float *prm = malloc((size_t)(3 + nm) * 4);
    for (int i = 0; i < 3; i++) prm[i] = 0.5f + (rnd() % 100) / 100.f;
    for (int i = 0; i < nm; i++) prm[3 + i] = frnd() * 0.5f;
    float *rpre = malloc((size_t)S * D * 4), *rpost = malloc((size_t)S * H * 4), *rcomb = malloc((size_t)S * H * H * 4);
    float *rout = malloc((size_t)S * HD * 4), *rmean = malloc((size_t)S * D * 4);
    for (int s = 0; s < S; s++) {
        coli_hc_pre(rpre + s * D, rpost + s * H, rcomb + s * H * H, x + s * HD, fn, prm, prm + 3, H, D, iters, eps, hce);
        coli_hc_post(rout + s * HD, br + s * D, x + s * HD, rpost + s * H, rcomb + s * H * H, H, D);
        for (int d = 0; d < D; d++) { float sm = 0; for (int i = 0; i < H; i++) sm += x[s * HD + i * D + d]; rmean[s * D + d] = sm / H; }
    }
    ColiVkTensor *t = NULL;
    if (!coli_vk_tensor_ensure(&t, fn, NULL, 10, HD, nm, 0)) { CHECK(0, "mhc: upload"); return; }
    VkcBuf *xb = up(x, (size_t)S * HD), *bb = up(br, (size_t)S * D), *pb = up(prm, 3 + (size_t)nm);
    VkcBuf *mb = vkc_buf((size_t)S * nm * 4, VKC_DEV), *hb = vkc_buf((size_t)S * (2 * H + H * H) * 4, VKC_DEV);
    VkcBuf *cb = vkc_buf((size_t)S * D * 4, VKC_DEV), *ob = vkc_buf((size_t)S * HD * 4, VKC_DEV), *eb = vkc_buf((size_t)S * D * 4, VKC_DEV);
    int hrow = 2 * H + H * H;
    VkcMhc sp = {S, H, D, iters, 0, HD, 0, nm, 0, hrow, 0, D, 0, 0, eps, hce, 0.f};
    VkcMhc po = {S, H, D, iters, 0, HD, 0, D, 0, hrow, 0, HD, 0, 0, eps, hce, 0.f};
    int ok = vkc_begin() && vkc_matmul(t, xb, 0, mb, 0, S) && vkc_mhc(VKC_MHC_SPLIT, xb, mb, hb, pb, NULL, &sp) &&
             vkc_mhc(VKC_MHC_COLLAPSE, xb, NULL, hb, NULL, cb, &sp) && vkc_mhc(VKC_MHC_POST, xb, bb, hb, NULL, ob, &po) &&
             vkc_mhc(VKC_MHC_MEAN, xb, NULL, NULL, NULL, eb, &sp) && vkc_submit(1);
    float *hp = down(hb, 0, (size_t)S * hrow), *col = down(cb, 0, (size_t)S * D), *out = down(ob, 0, (size_t)S * HD), *mean = down(eb, 0, (size_t)S * D);
    /* pre, post and comb from coli_hc_split_sinkhorn on the CPU's own mixes */
    float *rp = malloc((size_t)S * hrow * 4), *mixs = malloc((size_t)nm * 4);
    for (int s = 0; s < S; s++) {
        float ms = 0; for (int i = 0; i < HD; i++) ms += x[s * HD + i] * x[s * HD + i];
        float ir = 1.f / sqrtf(ms / HD + eps);
        for (int r = 0; r < nm; r++) { float a = 0; for (int i = 0; i < HD; i++) a += fn[(size_t)r * HD + i] * x[s * HD + i]; mixs[r] = a * ir; }
        coli_hc_split_sinkhorn(rp + s * hrow, rp + s * hrow + H, rp + s * hrow + 2 * H, mixs, prm, prm + 3, H, iters, hce);
    }
    double epre = 0;
    for (int s = 0; s < S; s++) for (int i = 0; i < hrow; i++) { double d = fabs(hp[s * hrow + i] - rp[s * hrow + i]); if (d > epre) epre = d; }
    free(rp); free(mixs);
    double ep = 0, ec = relerr(col, rpre, (size_t)S * D, 1e-3), eo = relerr(out, rout, (size_t)S * HD, 1e-3), em = relerr(mean, rmean, (size_t)S * D, 1e-3);
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < H; i++) { double d = fabs(hp[s * hrow + H + i] - rpost[s * H + i]); if (d > ep) ep = d; }
        for (int i = 0; i < H * H; i++) { double d = fabs(hp[s * hrow + 2 * H + i] - rcomb[s * H * H + i]); if (d > ep) ep = d; }
    }
    CHECK(ok && ep < 2e-5 && epre < 5e-7 && ec < 2e-5 && eo < 2e-5 && em < 1e-6,
          "mhc H %d D %d iters %d: ok %d split %.2e post/comb %.2e collapse %.2e write back %.2e mean %.2e",
          H, D, iters, ok, epre, ep, ec, eo, em);
    if (getenv("VKC_TEST_VERBOSE")) printf("mhc H %d D %d: split %.2e post/comb %.2e collapse %.2e write back %.2e mean %.2e\n", H, D, epre, ep, ec, eo, em);
    /* the clamped SwiGLU, on values both sides of the limit */
    int n = 333;
    float *ga = fvec(n, 3.f), *ua = fvec(n, 3.f), *rr = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) { float gg = ga[i] > lim ? lim : ga[i], uu = ua[i] < -lim ? -lim : (ua[i] > lim ? lim : ua[i]); rr[i] = gg / (1.f + expf(-gg)) * uu; }
    VkcBuf *gab = up(ga, n), *uab = up(ua, n), *yb = vkc_buf((size_t)n * 4, VKC_DEV);
    VkcMhc sw = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, n, 0.f, 0.f, lim};
    ok = vkc_begin() && vkc_mhc(VKC_SWIGLU_CLAMP, gab, uab, NULL, NULL, yb, &sw) && vkc_submit(1);
    float *yy = down(yb, 0, n);
    double es = relerr(yy, rr, n, 1e-3);
    CHECK(ok && es < 1e-6, "swiglu clamp: ok %d err %.2e", ok, es);
    vkc_free(xb); vkc_free(bb); vkc_free(pb); vkc_free(mb); vkc_free(hb); vkc_free(cb); vkc_free(ob); vkc_free(eb);
    vkc_free(gab); vkc_free(uab); vkc_free(yb); coli_vk_tensor_free(t);
    free(x); free(fn); free(br); free(prm); free(rpre); free(rpost); free(rcomb); free(rout); free(rmean);
    free(hp); free(col); free(out); free(mean); free(ga); free(ua); free(rr); free(yy);
}

/* k-pooling against sparse_index.h. Equal gates and no bias make each pooled key the
 * exact mean of its members, so with quarter-step values and power-of-two scales the
 * scores are exact on both sides and the selection must be the CPU's slot for slot;
 * a second pass with random gates and bias checks the pooled keys to rounding. */
static void test_kpool(int S, int pos_base, int IH, int ID, int pool, int topk, int tail) {
    int T = pos_base + S, width = coli_sparse_index_width(topk, pool, tail), selrow = width + 1;
    int npool = T / pool, scrow = npool + 1;
    float *keys = malloc((size_t)T * ID * 4), *gates = calloc((size_t)T * ID, 4), *ape = calloc((size_t)pool * ID, 4);
    float *q = malloc((size_t)S * IH * ID * 4), *hw = malloc((size_t)S * IH * 4);
    for (int i = 0; i < T * ID; i++) keys[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * IH * ID; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * IH; i++) hw[i] = ((int)(rnd() % 9) - 4) * 0.5f * sqrtf((float)IH);
    for (int t = 0; t < T; t += 7) for (int i = 0; i < ID; i++) keys[t * ID + i] = keys[(t % 3) * ID + i];   /* equal pools: ties */
    unsigned char *valid = malloc((size_t)T); memset(valid, 1, (size_t)T);
    int *ref = malloc((size_t)S * width * sizeof(int));
    float *hwd = malloc((size_t)S * IH * 4);
    for (int i = 0; i < S * IH; i++) hwd[i] = hw[i] / sqrtf((float)IH);
    coli_sparse_index_select_range(ref, q, keys, gates, hwd, ape, valid, T, IH, ID, pool, topk, tail, pos_base, T);
    VkcBuf *kb = up(keys, (size_t)T * ID), *gb = up(gates, (size_t)T * ID), *ab = up(ape, (size_t)pool * ID);
    VkcBuf *qb = up(q, (size_t)S * IH * ID), *wb = up(hw, (size_t)S * IH), *pk = vkc_buf((size_t)(npool + 1) * ID * 4, VKC_DEV);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *sb = vkc_buf((size_t)S * selrow * 4, VKC_DOWN);
    VkcDsaPool kp = {npool, pool, 0, ID, 0, ID, 0, 0, 0, ID};
    VkcDsaPick sp = {S, pos_base, IH, ID, topk, pool, 0, IH * ID, 0, IH, 0, tail, scrow, selrow, sqrtf((float)IH), 1.f / sqrtf((float)ID)};
    int ok = vkc_begin() && vkc_dsa_pool_keys(kb, gb, ab, pk, &kp) && vkc_dsa_pool_select(qb, wb, pk, scb, sb, &sp) && vkc_submit(1);
    const int *got = (const int *)vkc_ptr(sb);
    int same = ok;
    for (int s = 0; same && s < S; s++) {
        same = got[s * selrow] == width;
        for (int i = 0; same && i < width; i++) same = got[s * selrow + 1 + i] == ref[s * width + i];
        if (!same && getenv("DSA_DEBUG")) { for (int i = 0; i < width; i++) printf(" %d/%d", got[s * selrow + 1 + i], ref[s * width + i]); printf("\n"); }
    }
    CHECK(same, "kpool S %d pos %d IH %d ID %d pool %d topk %d tail %d: ok %d, the selection differs", S, pos_base, IH, ID, pool, topk, tail, ok);
    /* the pooled keys with real gates and bias, against sparse_index.h's arithmetic */
    for (int i = 0; i < T * ID; i++) gates[i] = frnd() * 2.f;
    for (int i = 0; i < pool * ID; i++) ape[i] = frnd() * 0.5f;
    float *rk = malloc((size_t)npool * ID * 4);
    for (int p = 0; p < npool; p++) for (int d = 0; d < ID; d++) {
        float mx = -FLT_MAX, tot = 0, mix = 0;
        for (int j = 0; j < pool; j++) { float l = gates[(p * pool + j) * ID + d] + ape[j * ID + d]; if (l > mx) mx = l; }
        for (int j = 0; j < pool; j++) tot += expf(gates[(p * pool + j) * ID + d] + ape[j * ID + d] - mx);
        for (int j = 0; j < pool; j++) mix += expf(gates[(p * pool + j) * ID + d] + ape[j * ID + d] - mx) / tot * keys[(p * pool + j) * ID + d];
        rk[p * ID + d] = mix;
    }
    ok = vkc_begin() && vkc_write(gb, 0, gates, (size_t)T * ID * 4) && vkc_write(ab, 0, ape, (size_t)pool * ID * 4) &&
         vkc_dsa_pool_keys(kb, gb, ab, pk, &kp) && vkc_submit(1);
    float *gk = down(pk, 0, (size_t)npool * ID);
    double e = relerr(gk, rk, (size_t)npool * ID, 1e-3);
    CHECK(ok && e < 2e-6, "kpool keys pool %d ID %d: ok %d err %.2e", pool, ID, ok, e);
    vkc_free(kb); vkc_free(gb); vkc_free(ab); vkc_free(qb); vkc_free(wb); vkc_free(pk); vkc_free(scb); vkc_free(sb);
    free(keys); free(gates); free(ape); free(q); free(hw); free(valid); free(ref); free(hwd); free(rk); free(gk);
}

/* ---- DeepSeek V4.1 / V4 attention (chain_dsv4.comp) ------------------------------------- */
static float bf16r(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) != 0x7f800000u) u += 0x7fffu + ((u >> 16) & 1u);
                              u &= 0xffff0000u; memcpy(&f, &u, 4); return f; }
/* sparse_attn.h's scalar kernel (and DeepSeek V4's: the weights and the output to bf16)
 * over lists of window rows (e < nwin), compressed rows (nwin + j) and skipped entries */
static void test_ds_attn(int S, int H, int hd, int nwin, int ncmp, int cnt, int flags) {
    int qrow = H * hd + 3, lrow = cnt + 2, orow = H * hd + 3;
    float *q = fvec((size_t)S * qrow, 1.f), *win = fvec((size_t)(nwin * hd + 5), 1.f), *cmp = fvec((size_t)(ncmp * hd + 7), 1.f);
    float *sink = fvec((size_t)H + 2, 2.f), *ref = calloc((size_t)S * orow, 4), *sc = malloc((size_t)cnt * 4);
    int *list = malloc((size_t)S * lrow * 4);
    float *kvall = malloc((size_t)(nwin + ncmp) * hd * 4);
    memcpy(kvall, win + 5, (size_t)nwin * hd * 4); memcpy(kvall + (size_t)nwin * hd, cmp + 7, (size_t)ncmp * hd * 4);
    for (int s = 0; s < S; s++)
        for (int j = 0; j < lrow; j++) {
            int r = (int)(rnd() % 5);
            list[s * lrow + j] = s == S - 1 && S > 1 ? -1                                     /* a row with nothing to read */
                               : r == 0 ? -1 : r < 3 ? (int)(rnd() % (unsigned)nwin) : nwin + (int)(rnd() % (unsigned)ncmp);
        }
    for (int s = 0; s < S; s++)
        for (int h = 0; h < H; h++) {
            const float *qq = q + (size_t)s * qrow + 1 + h * hd;
            float *o = ref + (size_t)s * orow + 2 + h * hd;
            const int *idx = list + s * lrow + 1;
            if (!(flags & 1)) { coli_sparse_attend_scalar(o, qq, kvall, idx, cnt, hd, sink[1 + h], 1.f / sqrtf((float)hd), sc); continue; }
            float best = -1e30f;   /* DeepSeek V4's reference: w to bf16 before the value sum, the output to bf16 */
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) { sc[j] = -INFINITY; continue; }
                float d = 0; for (int i = 0; i < hd; i++) d += qq[i] * kvall[(size_t)idx[j] * hd + i];
                sc[j] = d * (1.f / sqrtf((float)hd)); if (sc[j] > best) best = sc[j];
            }
            float den = expf(sink[1 + h] - best);
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int j = 0; j < cnt; j++) {
                if (idx[j] < 0) continue;
                float w = expf(sc[j] - best); den += w; w = bf16r(w);
                for (int i = 0; i < hd; i++) o[i] += w * kvall[(size_t)idx[j] * hd + i];
            }
            for (int i = 0; i < hd; i++) o[i] = bf16r(o[i] / den);
        }
    VkcBuf *qb = up(q, (size_t)S * qrow), *wb = up(win, (size_t)nwin * hd + 5), *cb = up(cmp, (size_t)ncmp * hd + 7);
    VkcBuf *sb = up(sink, (size_t)H + 2), *ob = vkc_buf((size_t)S * orow * 4, VKC_DEV), *lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV);
    vkc_begin(); vkc_write(lb, 0, list, (size_t)S * lrow * 4); vkc_submit(1);
    VkcDsAttn p = {S, H, hd, cnt, 1, lrow, nwin, 5, 7, 1, qrow, 2, orow, 1, flags, 1.f / sqrtf((float)hd)};
    vkc_begin(); int ok = vkc_dsv4_attn(qb, wb, cb, lb, sb, ob, &p); vkc_submit(1);
    float *got = down(ob, 0, (size_t)S * orow);
    double e = 0;
    for (int s = 0; s < S; s++) { double r = relerr(got + (size_t)s * orow + 2, ref + (size_t)s * orow + 2, (size_t)H * hd, 1e-3); if (r > e) e = r; }
    double tol = (flags & 1) ? 8e-3 : 2e-5;
    CHECK(ok && e < tol && !bad(got, (size_t)S * orow), "dsv4 attn S %d H %d hd %d cnt %d flags %d: rel err %.2e", S, H, hd, cnt, flags, e);
    if (!(flags & 1) && S > 1) {   /* the empty row is exactly zero */
        int z = 1; for (int i = 0; i < H * hd; i++) z &= got[(size_t)(S - 1) * orow + 2 + i] == 0.f;
        CHECK(z, "dsv4 attn: a row with no entry is not zero");
    }
    printf("  dsv4 attn S %d H %d hd %d cnt %d flags %d: rel err %.2e\n", S, H, hd, cnt, flags, e);
    vkc_free(qb); vkc_free(wb); vkc_free(cb); vkc_free(sb); vkc_free(ob); vkc_free(lb);
    free(q); free(win); free(cmp); free(sink); free(ref); free(sc); free(list); free(kvall); free(got);
}

/* deepseek_v41.c's rope_apply: interleaved pairs, forward and inverse, segments at a stride */
static void test_ds_rope(int flags) {
    int rows = 3, per = 4, seg = 40, rd = 16, xrow = per * seg + 2, csrow = rd + 1;
    float *x = fvec((size_t)rows * xrow + 3, 1.f), *ref = malloc(((size_t)rows * xrow + 3) * 4), *cs = malloc(((size_t)rows * csrow + 2) * 4);
    for (int r = 0; r < rows; r++) for (int i = 0; i < rd / 2; i++) {
        float a = (r * 7 + 3) * powf(10000.f, -2.f * i / rd);
        cs[2 + r * csrow + 2 * i] = cosf(a); cs[2 + r * csrow + 2 * i + 1] = sinf(a);
    }
    for (int inv = 0; inv < 2; inv++) {
        memcpy(ref, x, ((size_t)rows * xrow + 3) * 4);
        for (int r = 0; r < rows; r++) for (int j = 0; j < per; j++) {
            float *v = ref + 3 + r * xrow + j * seg + (seg - rd);
            for (int i = 0; i < rd / 2; i++) {
                float c = cs[2 + r * csrow + 2 * i], sn = cs[2 + r * csrow + 2 * i + 1];
                if (inv) sn = -sn;
                float a = v[2 * i], b = v[2 * i + 1];
                v[2 * i] = a * c - b * sn; v[2 * i + 1] = a * sn + b * c;
                if (flags & 1) { v[2 * i] = bf16r(v[2 * i]); v[2 * i + 1] = bf16r(v[2 * i + 1]); }
            }
        }
        VkcBuf *xb = up(x, (size_t)rows * xrow + 3), *cb = up(cs, (size_t)rows * csrow + 2);
        VkcDsRope p = {rows * per, per, rd, 3 + seg - rd, xrow, seg, 2, csrow, inv, flags};
        vkc_begin(); int ok = vkc_dsv4_rope(xb, cb, &p); vkc_submit(1);
        float *got = down(xb, 0, (size_t)rows * xrow + 3);
        double e = relerr(got, ref, (size_t)rows * xrow + 3, 1e-3);
        CHECK(ok && e < ((flags & 1) ? 8e-3 : 1e-6), "dsv4 rope inverse %d flags %d: rel err %.2e", inv, flags, e);
        vkc_free(xb); vkc_free(cb); free(got);
    }
    free(x); free(ref); free(cs);
}

/* deepseek_v41.c's compressor_run (its decode form; the prefill form gives the same groups)
 * and DeepSeek V4's overlapping one (ratio 4: the previous group's rows in the first half,
 * a channel of its own each, the ape bias), the ring carried over three calls */
static void test_ds_compress(int ratio, int overlap, int ape) {
    int D = 24, P = overlap ? 2 * D : D, rows = overlap ? 2 * ratio : ratio, T = 23, pb0 = 3;
    float *kv = fvec((size_t)T * P, 1.f), *sc = fvec((size_t)T * P, 2.f), *ap = fvec((size_t)ratio * P + 4, 1.f);
    float *rk = calloc((size_t)rows * P, 4), *rs = malloc((size_t)rows * P * 4);
    for (int i = 0; i < rows * P; i++) rs[i] = -INFINITY;
    int G = (pb0 + T) / ratio + 1;
    float *ref = calloc((size_t)G * D, 4);
    for (int t = 0; t < T; t++) {
        int pos = pb0 + t, slot = pos % ratio, sr = overlap ? ratio + slot : slot;
        for (int c = 0; c < P; c++) { rk[sr * P + c] = kv[t * P + c]; rs[sr * P + c] = sc[t * P + c] + (ape ? ap[4 + slot * P + c] : 0.f); }
        if ((pos + 1) % ratio) continue;
        for (int d = 0; d < D; d++) {
            float best = -INFINITY;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; if (rs[k * P + col] > best) best = rs[k * P + col]; }
            float tot = 0, mix = 0;
            for (int k = 0; k < rows; k++) { int col = overlap && k >= ratio ? D + d : d; float w = expf(rs[k * P + col] - best); tot += w; mix += w * rk[k * P + col]; }
            ref[(size_t)(pos / ratio) * D + d] = mix / tot;
        }
        if (overlap) { memcpy(rk, rk + (size_t)ratio * P, (size_t)ratio * P * 4); memcpy(rs, rs + (size_t)ratio * P, (size_t)ratio * P * 4); }
    }
    VkcBuf *kb = up(kv, (size_t)T * P), *sb = up(sc, (size_t)T * P), *ab = up(ap, (size_t)ratio * P + 4);
    VkcBuf *ring = vkc_buf((size_t)(2 * rows * P + 5) * 4, VKC_DEV), *ob = vkc_buf((size_t)G * D * 4, VKC_DEV);
    float *init = malloc((size_t)(2 * rows * P + 5) * 4);
    for (int i = 0; i < 2 * rows * P + 5; i++) init[i] = i >= 5 + rows * P ? -INFINITY : 0.f;
    vkc_begin(); vkc_write(ring, 0, init, (size_t)(2 * rows * P + 5) * 4); vkc_submit(1);
    int cuts[4] = {0, 5, 6, T}, ok = 1;
    for (int k = 0; k < 3; k++) {
        int n = cuts[k + 1] - cuts[k];
        VkcDsComp p = {n, pb0 + cuts[k], ratio, P, D, overlap, cuts[k] * P, P, cuts[k] * P, P, ape ? 4 : -1, 0, D, 5};
        vkc_begin(); ok &= vkc_dsv4_compress(kb, sb, ring, ab, ob, &p); vkc_submit(k == 2);
    }
    float *got = down(ob, 0, (size_t)G * D);
    int g0 = (pb0 + 1 + ratio - 1) / ratio;   /* the first group completed by a row here */
    double e = relerr(got + (size_t)g0 * D, ref + (size_t)g0 * D, (size_t)(G - 1 - g0) * D, 1e-3);
    CHECK(ok && e < 2e-6, "dsv4 compress ratio %d overlap %d ape %d: rel err %.2e", ratio, overlap, ape, e);
    vkc_free(kb); vkc_free(sb); vkc_free(ab); vkc_free(ring); vkc_free(ob);
    free(kv); free(sc); free(ap); free(rk); free(rs); free(ref); free(got); free(init);
}

/* deepseek_v41.c's indexer_run and candidate_blocks: the scores with the reach and the
 * mask, the candidate blocks, and the top-k list slot for slot. Inputs on a grid of 0.25
 * (weights 0.5, a power-of-two scale) so every score is exact in float on both sides,
 * with equal keys for ties. */
static void ds_topk_ref(const float *score, int width, int topk, int base, int order, int *row) {
    int taken = 0;
    for (int k = 0; k < topk; k++) row[k] = -1;
    for (int k = 0; k < topk; k++) {
        int best = -1;
        for (int j = 0; j < width; j++) {
            if (score[j] == -INFINITY) continue;
            int already = 0; for (int u = 0; u < taken; u++) if (row[u] == j) { already = 1; break; }
            if (already) continue;
            if (best < 0 || score[j] > score[best]) best = j;
        }
        if (best < 0) break;
        row[taken++] = best;
    }
    if (order == 0)
        for (int a = 0; a < taken; a++) for (int b = a + 1; b < taken; b++) if (row[b] < row[a]) { int t = row[a]; row[a] = row[b]; row[b] = t; }
    for (int k = 0; k < taken; k++) row[k] += base;
}
static void ds_cand_ref(const float *score, int width, int lens, int block, int topb, int *keep) {
    int blocks = (width + block - 1) / block;
    float *best = malloc((size_t)(blocks + 1) * 4);
    for (int b = 0; b < blocks; b++) { float top = -INFINITY; for (int i = b * block; i < (b + 1) * block && i < width; i++) if (score[i] > top) top = score[i]; best[b] = top; }
    int last = (lens - 1) / block;
    if (last >= 0 && last < blocks) best[last] = INFINITY;
    int wanted = topb < blocks ? topb : blocks;
    for (int i = 0; i < width; i++) keep[i] = 0;
    for (int pick = 0; pick < wanted; pick++) {
        int chosen = -1;
        for (int b = 0; b < blocks; b++) if (best[b] > -INFINITY && (chosen < 0 || best[b] > best[chosen])) chosen = b;
        if (chosen < 0) break;
        for (int i = chosen * block; i < (chosen + 1) * block && i < width; i++) keep[i] = 1;
        best[chosen] = -INFINITY;
    }
    free(best);
}
static void test_ds_index(int S, int pos_base, int ratio, int IH, int ID, int topk, int block, int topb, int order) {
    int width = (pos_base + S) / ratio, krow = ID + 3, qrow = IH * ID + 1, wrow = IH + 2, scrow = width + 3, lrow = topk + 5, base = 11;
    if (width < 1) width = 1;
    float *q = malloc((size_t)(S * qrow + 1) * 4), *k = malloc((size_t)(width * krow + 3) * 4), *w = malloc((size_t)(S * wrow + 2) * 4);
    for (int i = 0; i < S * qrow + 1; i++) q[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < width * krow + 3; i++) k[i] = ((int)(rnd() % 17) - 8) * 0.25f;
    for (int i = 0; i < S * wrow + 2; i++) w[i] = ((int)(rnd() % 9) - 3) * 0.5f;
    for (int t = 0; t < width; t += 4) for (int i = 0; i < ID; i++) k[3 + t * krow + i] = k[3 + (t / 4) * krow + i];   /* ties */
    float wscale = 0.25f;
    float *ref = malloc((size_t)S * scrow * 4), *refm = malloc((size_t)S * scrow * 4);
    int *keep = calloc((size_t)S * scrow, 4), *lst = malloc((size_t)S * lrow * 4), *lstm = malloc((size_t)S * lrow * 4);
    for (int s = 0; s < S; s++) {
        int lens = (pos_base + s + 1) / ratio;
        float wt[64];
        for (int h = 0; h < IH; h++) { wt[h] = w[2 + s * wrow + h]; wt[h] *= wscale; }
        for (int j = 0; j < width; j++) {
            if (j >= lens) { ref[s * scrow + j] = -INFINITY; continue; }
            float total = 0;
            for (int h = 0; h < IH; h++) {
                float dot = 0; for (int i = 0; i < ID; i++) dot += q[1 + s * qrow + h * ID + i] * k[3 + j * krow + i];
                if (dot > 0.f) total += dot * wt[h];
            }
            ref[s * scrow + j] = total;
        }
        ds_cand_ref(ref + s * scrow, width, lens, block, topb, keep + s * scrow);
        for (int j = 0; j < width; j++) refm[s * scrow + j] = keep[s * scrow + j] ? ref[s * scrow + j] : -INFINITY;
        ds_topk_ref(ref + s * scrow, width, topk, base, order, lst + s * lrow + 2);
        ds_topk_ref(refm + s * scrow, width, topk, base, order, lstm + s * lrow + 2);
    }
    VkcBuf *qb = up(q, (size_t)S * qrow + 1), *kb = up(k, (size_t)width * krow + 3), *wb = up(w, (size_t)S * wrow + 2);
    VkcBuf *scb = vkc_buf((size_t)S * scrow * 4, VKC_DEV), *mb = vkc_buf((size_t)S * scrow * 4, VKC_DEV);
    VkcBuf *lb = vkc_buf((size_t)S * lrow * 4, VKC_DEV), *lb2 = vkc_buf((size_t)S * lrow * 4, VKC_DEV), *scb2 = vkc_buf((size_t)S * scrow * 4, VKC_DEV);
    VkcDsScore sp = {S, pos_base, ratio, IH, ID, width, 1, qrow, 2, wrow, 3, krow, 0, scrow, wscale, 0, 0};
    VkcDsCand cp = {S, pos_base, ratio, width, block, topb, scrow, scrow};
    VkcDsTopk tp = {S, width, topk, scrow, 2, lrow, base, order};
    VkcDsScore sm = sp; sm.mask_row = scrow;
    vkc_begin();
    int ok = vkc_dsv4_score(qb, wb, kb, NULL, scb, &sp) && vkc_dsv4_cand(scb, mb, &cp) && vkc_dsv4_topk(scb, lb, &tp);
    if (S > 1)   /* the masked scores one row at a time, at offsets (a speculative step's rows read different keys) */
        for (int s = 0; s < S && ok; s++) {
            VkcDsScore r1 = sm; r1.S = 1; r1.pos_base = pos_base + s; r1.q_off = 1 + s * qrow; r1.w_off = 2 + s * wrow;
            r1.sc_off = s * scrow; r1.mask_off = s * scrow;
            ok = vkc_dsv4_score(qb, wb, kb, mb, scb2, &r1);
        }
    else ok = ok && vkc_dsv4_score(qb, wb, kb, mb, scb2, &sm);
    ok = ok && vkc_dsv4_topk(scb2, lb2, &tp);
    vkc_submit(1);
    float *gs = down(scb, 0, (size_t)S * scrow);
    int *gm = (int *)down(mb, 0, (size_t)S * scrow), *gl = (int *)down(lb, 0, (size_t)S * lrow), *gl2 = (int *)down(lb2, 0, (size_t)S * lrow);
    int same_sc = 1, same_m = 1, same_l = 1, same_l2 = 1;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < width; j++) {
            same_sc &= gs[s * scrow + j] == ref[s * scrow + j];
            same_m &= gm[s * scrow + j] == keep[s * scrow + j];
        }
        for (int t = 0; t < topk; t++) { same_l &= gl[s * lrow + 2 + t] == lst[s * lrow + 2 + t]; same_l2 &= gl2[s * lrow + 2 + t] == lstm[s * lrow + 2 + t]; }
    }
    CHECK(ok && same_sc && same_m && same_l && same_l2, "dsv4 index S %d pos %d ratio %d IH %d ID %d topk %d block %d/%d order %d: "
          "scores %d mask %d list %d masked list %d", S, pos_base, ratio, IH, ID, topk, block, topb, order, same_sc, same_m, same_l, same_l2);
    vkc_free(qb); vkc_free(kb); vkc_free(wb); vkc_free(scb); vkc_free(mb); vkc_free(lb); vkc_free(lb2); vkc_free(scb2);
    free(q); free(k); free(w); free(ref); free(refm); free(keep); free(lst); free(lstm); free(gs); free(gm); free(gl); free(gl2);
}

/* deepseek_v41.c's engram_run gate, in place on the streams */
static void test_ds_engram(int S, int H, int D) {
    int kvrow = (H + 1) * D + 2, xrow = H * D + 3;
    float *kv = fvec((size_t)S * kvrow + 1, 1.f), *x = fvec((size_t)S * xrow + 4, 1.f), *prm = fvec((size_t)2 * H * D + 6, 1.f);
    float *ref = malloc(((size_t)S * xrow + 4) * 4), eps = 1e-6f;
    memcpy(ref, x, ((size_t)S * xrow + 4) * 4);
    for (int s = 0; s < S; s++) {
        const float *k0 = kv + 1 + (size_t)s * kvrow, *value = k0 + (size_t)H * D;
        for (int c = 0; c < H; c++) {
            const float *key = k0 + (size_t)c * D, *qw = prm + 6 + (size_t)c * D, *kw = prm + 6 + (size_t)H * D + (size_t)c * D;
            float *st = ref + 4 + (size_t)s * xrow + (size_t)c * D;
            double ss = 0, ks = 0, dot = 0;
            for (int i = 0; i < D; i++) { ss += (double)st[i] * st[i]; ks += (double)key[i] * key[i]; dot += (double)st[i] * qw[i] * kw[i] * key[i]; }
            float rstd = (1.0f / sqrtf((float)(ss / D) + eps)) * (1.0f / sqrtf((float)(ks / D) + eps));
            float scaled = (float)dot * rstd / sqrtf((float)D);
            float mag = fabsf(scaled); if (mag < 1e-6f) mag = 1e-6f;
            float sr = sqrtf(mag); if (scaled < 0) sr = -sr;
            float g = sigm(sr);
            for (int i = 0; i < D; i++) st[i] += g * value[i];
        }
    }
    VkcBuf *kb = up(kv, (size_t)S * kvrow + 1), *xb = up(x, (size_t)S * xrow + 4), *pb = up(prm, (size_t)2 * H * D + 6);
    VkcDsEngram p = {S, H, D, 1, kvrow, 4, xrow, 6, 6 + H * D, eps};
    vkc_begin(); int ok = vkc_dsv4_engram(kb, pb, xb, &p); vkc_submit(1);
    float *got = down(xb, 0, (size_t)S * xrow + 4);
    double e = relerr(got, ref, (size_t)S * xrow + 4, 1e-3);
    CHECK(ok && e < 2e-6, "dsv4 engram S %d H %d D %d: rel err %.2e", S, H, D, e);
    vkc_free(kb); vkc_free(xb); vkc_free(pb);
    free(kv); free(x); free(prm); free(ref); free(got);
}

/* ---- DeepSeek V4's roundings (chain_dsv4.comp modes 7, 8) ------------------------------
 * References transcribed from deepseek_v4.c (COLI_V4_UNIT_NATIVE_QUANT, the MATH unit's
 * coli_v4_swiglu): the device must give their bits. */
static float ds_e4m3_decode(uint8_t v) {
    int sign = v >> 7, exponent = (v >> 3) & 15, mantissa = v & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float n = !exponent ? ldexpf((float)mantissa, -9) : ldexpf(1.0f + (float)mantissa / 8.0f, exponent - 7);
    return sign ? -n : n;
}
static uint8_t ds_e4m3_encode(float value) {
    if (isnan(value)) return 0x7f;
    int negative = signbit(value) != 0;
    float magnitude = fabsf(value);
    if (!magnitude) return negative ? 0x80 : 0;
    if (magnitude >= 448.0f) return (uint8_t)((negative ? 0x80 : 0) | 0x7e);
    uint8_t best;
    if (magnitude < 0.015625f) {
        float scaled = magnitude * 512.0f;
        uint8_t rounded = (uint8_t)scaled;
        float fraction = scaled - rounded;
        if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) rounded++;
        best = rounded;
    } else {
        uint32_t bits; memcpy(&bits, &magnitude, 4);
        int exponent = (int)((bits >> 23) & 0xff) - 127;
        uint32_t significand = 0x800000u | (bits & 0x7fffffu), rounded = significand >> 20, remainder = significand & 0xfffffu;
        if (remainder > 0x80000u || (remainder == 0x80000u && (rounded & 1u))) rounded++;
        if (rounded == 16u) { rounded = 8u; exponent++; }
        best = (uint8_t)((exponent + 7) * 8 + (int)rounded - 8);
    }
    return (uint8_t)(best | (negative ? 0x80 : 0));
}
static int ds_ceil_log2(float value) { int e; float f = frexpf(value, &e); return f == 0.5f ? e - 1 : e; }
static void ds_fp8_qdq(float *out, const float *in, size_t n, size_t block) {
    for (size_t base = 0; base < n; base += block) {
        size_t count = n - base < block ? n - base : block;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++) maximum = fmaxf(maximum, fabsf(in[base + i]));
        maximum = fmaxf(maximum, 1e-4f);
        int e = ds_ceil_log2(maximum / 448.0f);
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        float scale = ldexpf(1.0f, e);
        for (size_t i = 0; i < count; i++)
            out[base + i] = ds_e4m3_decode(ds_e4m3_encode(fmaxf(-448.0f, fminf(448.0f, in[base + i] / scale)))) * scale;
    }
}
static float ds_e2m1_decode(int n) {
    static const float v[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return v[n & 15];
}
static void ds_fp4_qdq(float *out, const float *in, size_t n, size_t block) {
    for (size_t base = 0; base < n; base += block) {
        size_t count = n - base < block ? n - base : block;
        float maximum = 0.0f;
        for (size_t i = 0; i < count; i++) maximum = fmaxf(maximum, fabsf(in[base + i]));
        maximum = fmaxf(maximum, 6.0f * ldexpf(1.0f, -126));
        int e = ds_ceil_log2(maximum / 6.0f);
        if (e < -127) e = -127;
        if (e > 127) e = 127;
        float scale = ldexpf(1.0f, e);
        for (size_t i = 0; i < count; i++) {
            float value = fmaxf(-6.0f, fminf(6.0f, in[base + i] / scale));
            int best = 0; float distance = fabsf(value - ds_e2m1_decode(0));
            for (int code = 1; code < 16; code++) {
                float c = fabsf(value - ds_e2m1_decode(code));
                if (c < distance) { distance = c; best = code; }
            }
            out[base + i] = ds_e2m1_decode(best) * scale;
        }
    }
}
static void ds_hadamard(float *v, size_t n) {
    for (size_t w = 1; w < n; w *= 2)
        for (size_t base = 0; base < n; base += 2 * w)
            for (size_t i = 0; i < w; i++) { float l = v[base + i], r = v[base + w + i]; v[base + i] = l + r; v[base + w + i] = l - r; }
    float scale = 1.0f / sqrtf((float)n);
    for (size_t i = 0; i < n; i++) v[i] = bf16r(v[i] * scale);
}
static float ds_sigmoid(float v) { if (v >= 0.0f) { float d = expf(-v); return 1.0f / (1.0f + d); } float g = expf(v); return g / (1.0f + g); }

/* A block's values: random magnitudes over 40 binades, and blocks built on purpose:
 * all zero, the amax on the scale's boundary (1.75 * 2^E for E4M3, 1.5 * 2^E for E2M1)
 * and one step above it, exact ties of the target grid (E4M3 mantissa midpoints and
 * subnormal midpoints; E2M1 midpoints), signed zeros, an amax under E4M3's floor. (The
 * bf16 rounding after E4M3 or E2M1 is the CPU's step but changes nothing: both grids
 * are bf16 values.) */
static void ds_fill(float *v, int n, int kind, int pattern) {
    float s = ldexpf(1.0f, (int)(rnd() % 30) - 15);
    for (int i = 0; i < n; i++) v[i] = frnd() * ldexpf(1.0f, (int)(rnd() % 40) - 20);
    if (pattern == 1) for (int i = 0; i < n; i++) v[i] = 0.0f;
    if (pattern == 2 || pattern == 3) {
        float top = (kind == VKC_DS_E2M1 ? 1.5f : 1.75f) * 4.0f * s;
        if (pattern == 3) { uint32_t u; memcpy(&u, &top, 4); u++; memcpy(&top, &u, 4); }
        v[0] = top;
    }
    if (pattern == 4) {
        float amax = (kind == VKC_DS_E2M1 ? 6.0f : 448.0f) * s;
        static const float ties2[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
        v[0] = amax;
        for (int i = 1; i < n; i++) {
            float t;
            if (kind == VKC_DS_E2M1) t = ties2[rnd() % 7];
            else if (rnd() % 3 == 0) t = (float)(2 * (int)(rnd() % 8) + 1) * ldexpf(1.0f, -10);   /* subnormal midpoints */
            else t = (1.0f + (float)(2 * (int)(rnd() % 8) + 1) / 16.0f) * ldexpf(1.0f, (int)(rnd() % 14) - 6);
            v[i] = (rnd() & 1 ? -t : t) * s;
        }
        v[n / 2] = -0.0f; if (n > 3) v[3] = 0.0f;
    }
    if (pattern == 5) for (int i = 0; i < n; i++) v[i] = frnd() * ldexpf(1.0f, -20 - (int)(rnd() % 4));   /* under the floor */
}
/* kind, nseg segments of len (per_row a row, at strides), block, flags; in place or not */
static void test_ds_round(int kind, int nseg, int per_row, int len, int block, int flags, int inplace) {
    int rows = (nseg + per_row - 1) / per_row, xseg = len + 3, xrow = per_row * xseg + 5, yseg = len + 1, yrow = per_row * yseg + 2;
    size_t xn = (size_t)rows * xrow + 7, yn = inplace ? xn : (size_t)rows * yrow + 4;
    int xo = 7, yo = inplace ? 7 : 4, ys = inplace ? xseg : yseg, yr = inplace ? xrow : yrow;
    float *x = fvec(xn, 1.f), *ref = malloc(xn > yn ? xn * 4 : yn * 4), *seg = malloc((size_t)len * 4);
    int bl = kind == VKC_DS_E4M3 || kind == VKC_DS_E2M1 ? block : len, nb = (len + bl - 1) / bl;
    for (int g = 0; g < nseg; g++) {
        float *xs = x + xo + (g / per_row) * xrow + (g % per_row) * xseg;
        for (int b = 0; b < nb; b++) ds_fill(xs + b * bl, len - b * bl < bl ? len - b * bl : bl, kind, (g * nb + b) % 6);
    }
    if (inplace) memcpy(ref, x, xn * 4);
    else for (size_t i = 0; i < yn; i++) ref[i] = -7.0f;     /* what the op must not touch */
    for (int g = 0; g < nseg; g++) {
        const float *xs = x + xo + (g / per_row) * xrow + (g % per_row) * xseg;
        float *ys2 = ref + yo + (g / per_row) * yr + (g % per_row) * ys;
        memcpy(seg, xs, (size_t)len * 4);
        if (kind == VKC_DS_BF16) for (int i = 0; i < len; i++) seg[i] = bf16r(seg[i]);
        else if (kind == VKC_DS_E4M3) ds_fp8_qdq(seg, xs, (size_t)len, (size_t)block);
        else if (kind == VKC_DS_E2M1) ds_fp4_qdq(seg, xs, (size_t)len, (size_t)block);
        else ds_hadamard(seg, (size_t)len);
        if ((flags & 1) && (kind == VKC_DS_E4M3 || kind == VKC_DS_E2M1)) for (int i = 0; i < len; i++) seg[i] = bf16r(seg[i]);
        memcpy(ys2, seg, (size_t)len * 4);
    }
    VkcBuf *xb = up(x, xn), *yb = inplace ? xb : NULL;
    if (!inplace) { float *init = malloc(yn * 4); for (size_t i = 0; i < yn; i++) init[i] = -7.0f; yb = up(init, yn); free(init); }
    VkcDsRound p = {kind, nseg, per_row, len, block, flags, xo, xrow, xseg, yo, yr, ys, 1.0f / sqrtf((float)len)};
    vkc_begin(); int ok = vkc_dsv4_round(xb, yb, &p); vkc_submit(1);
    float *got = down(yb, 0, yn);
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < yn; i++) if (memcmp(&got[i], &ref[i], 4)) { if (!diff) first = i; diff++; }
    CHECK(ok && !diff, "dsv4 round kind %d nseg %d len %d block %d flags %d inplace %d: %zu of %zu values differ (first at %zu: %a, want %a)",
          kind, nseg, len, block, flags, inplace, diff, yn, first, (double)got[first], (double)ref[first]);
    vkc_free(xb); if (!inplace) vkc_free(yb);
    free(x); free(ref); free(seg); free(got);
}
/* The SwiGLU: within one bf16 step of the CPU (the device's exp is not the CPU's expf),
 * and how many values are the CPU's bits */
static void test_ds_swiglu(int n, float lim) {
    float *a = fvec((size_t)n + 3, 14.f), *b = fvec((size_t)n + 5, 14.f), *ref = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float g = bf16r(a[3 + i]), u = bf16r(b[5 + i]);
        if (lim > 0.0f) { g = fminf(g, lim); u = fmaxf(-lim, fminf(u, lim)); }
        ref[i] = bf16r(g * ds_sigmoid(g) * u);
    }
    VkcBuf *ab = up(a, (size_t)n + 3), *bb = up(b, (size_t)n + 5), *yb = vkc_buf((size_t)(n + 2) * 4, VKC_DEV);
    VkcDsSwiglu p = {n, 3, 5, 2, lim};
    vkc_begin(); int ok = vkc_dsv4_swiglu(ab, bb, yb, &p); vkc_submit(1);
    float *got = down(yb, 2, (size_t)n);
    int same = 0, far = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u, v; memcpy(&u, &got[i], 4); memcpy(&v, &ref[i], 4);
        same += u == v;
        long d = (long)(u >> 16) - (long)(v >> 16);
        if ((u & 0xffff) || d > 1 || d < -1) far++;
    }
    CHECK(ok && !far, "dsv4 swiglu n %d lim %g: %d values further than one bf16 step (or not bf16)", n, lim, far);
    printf("  dsv4 swiglu n %d lim %g: %d of %d values the CPU's bits, the rest one bf16 step away\n", n, lim, same, n);
    vkc_free(ab); vkc_free(bb); vkc_free(yb);
    free(a); free(b); free(ref); free(got);
}

/* ---- frames: many ops, frames in flight, ordering ---------------------------------------- */
static void test_frames(void) {
    int n = 1000;
    float *a = fvec(n, 1.f), *acc = calloc(n, sizeof *acc);
    VkcBuf *x = vkc_buf((size_t)n * 4, VKC_DEV), *y = vkc_buf((size_t)n * 4, VKC_DEV);
    int ok = 1;
    for (int f = 0; f < 11; f++) {        /* more frames than the ring: reuse while others are in flight */
        ok &= vkc_begin();
        ok &= vkc_write(y, 0, a, (size_t)n * 4);
        for (int k = 0; k < 3; k++) {
            VkcEw p = {VKC_EW_ADD, n, n, 1, 0, 1, 0, 0, 0, 0, 0, 1.f};
            ok &= vkc_ew(x, x, y, NULL, NULL, &p);
            for (int i = 0; i < n; i++) acc[i] += a[i];
        }
        ok &= vkc_submit(f == 10);
        for (int i = 0; i < n; i++) a[i] = a[i] * 0.5f + 0.25f;
    }
    float *got = down(x, 0, n);
    double e = relerr(got, acc, n, 1e-3);
    VkcStats st; vkc_stats(&st);
    CHECK(ok && e < 1e-5, "frames: err %.2e (%llu frames, %llu barriers)", e, st.frames, st.barriers);
    vkc_free(x); vkc_free(y); free(a); free(acc); free(got);
}

/* COLI_VK_CHAIN_BENCH=1: decode-shaped GEMVs back to back in one frame (the chain's
 * situation: no host gap between matrices), per weight format, in GB/s of weights. */
#include <time.h>
static double bnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void bench_gemv(void) {
    int shapes[3][2] = {{2560, 10240}, {6144, 2560}, {2560, 640}};
    int fmts[3] = {1, 4, 11};
    for (int f = 0; f < 3; f++) for (int k = 0; k < 3; k++) {
        int I = shapes[k][0], O = shapes[k][1], fmt = fmts[f], gs = fmt == 4 ? 64 : 0, nm = 24;
        size_t rb = fmt == 1 ? (size_t)I : fmt == 4 ? (size_t)I / 2 : (size_t)I * 2;
        ColiVkTensor **t = calloc(nm, sizeof *t);
        uint8_t *w = malloc(rb * O); float *sc = malloc((size_t)O * (I / 64 + 1) * sizeof(float));
        for (size_t i = 0; i < rb * O; i++) w[i] = (uint8_t)rnd();
        for (size_t i = 0; i < (size_t)O * (I / 64 + 1); i++) sc[i] = 0.01f;
        for (int j = 0; j < nm; j++) coli_vk_tensor_ensure(&t[j], w, sc, fmt, I, O, gs);
        float *x = fvec(I, 1.f);
        VkcBuf *xb = up(x, I), *yb = vkc_buf((size_t)O * 4 * nm, VKC_DEV);
        double best = 1e9;
        for (int r = 0; r < 4; r++) {
            vkc_begin();
            for (int j = 0; j < nm; j++) vkc_matmul(t[j], xb, 0, yb, (size_t)j * O, 1);
            double t0 = bnow(); vkc_submit(1); double dt = bnow() - t0;
            if (dt < best) best = dt;
        }
        printf("bench fmt %2d [%5d x %5d] x %d: %.3f ms per matrix, %.1f GB/s of weights\n", fmt, O, I, nm,
               best * 1e3 / nm, (double)rb * O * nm / best / 1e9);
        for (int j = 0; j < nm; j++) coli_vk_tensor_free(t[j]);
        vkc_free(xb); vkc_free(yb); free(t); free(w); free(sc); free(x);
    }
}

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    if (!vkc_init()) { printf("FAIL: the chain's pipelines did not come up\n"); return 1; }
    if (getenv("COLI_VK_CHAIN_BENCH")) { bench_gemv(); vkc_shutdown(); coli_vk_shutdown(); return 0; }
    int fmts[4] = {1, 4, 10, 11};
    for (int k = 0; k < 4; k++) {
        test_matmul(fmts[k], 1, 192, 70, 0, 0);
        test_matmul(fmts[k], 3, 128, 96, 64, 128);       /* GEMV rows at bindable offsets */
        test_matmul(fmts[k], 40, 256, 160, 0, 0);        /* the tiled GEMM */
        test_matmul(fmts[k], 2, 64, 33, 3, 5);           /* offsets the device cannot bind */
        test_matmul(fmts[k], 1, 100, 33, 0, 0);          /* rows that are not whole 16-byte steps */
        test_matmul(fmts[k], 2, 3072, 40, 0, 0);         /* a long row (chain_gemv's staging) */
    }
    printf("matmul done\n");
    test_norm(VKC_NORM_ADD1, 0); test_norm(VKC_NORM_ADD1, 1); test_norm(0, 0); test_norm(VKC_NORM_NOW, 1); test_norm(VKC_NORM_L2 | VKC_NORM_NOW, 0);
    printf("norm done\n");
    test_rope();
    test_attn(1, 0, 16, 0); test_attn(1, 140, 32, 0); test_attn(5, 200, 64, 0); test_attn(6, 9, 256, 1); test_attn(130, 3, 24, 0);
    printf("attn done\n");
    test_dnconv(0); test_dnconv(1);
    test_dnrec(8, 8, 8, 4, 0); test_dnrec(4, 4, 4, 2, 1); test_dnrec(128, 128, 4, 2, 0); test_dnrec(32, 100, 6, 3, 1);
    printf("deltanet done\n");
    test_ew();
    test_qsa();
    test_ple();
    test_frames();
    if (!vkc_mla_ready()) { fails++; printf("FAIL: the MLA shaders did not load\n"); }
    else {
        int hf[9] = {1, 2, 4, 5, 7, 10, 11, 12, 13};
        for (int k = 0; k < 9; k++) if (hf[k] != 7) { test_hgemv(hf[k], 0); test_hgemv(hf[k], 1); }
        test_mla_rows();
        printf("mla ops done\n");
        /* H Q R V K q_lora D S pos kv_start style split list gate fq fkv fo */
        MlaCase cases[] = {
            {4, 24, 8, 32, 32, 64, 128, 1, 12, 0, 1, 0, 0, 0, 10, 10, 10},     /* colibri's tiny, decode */
            {4, 24, 8, 32, 32, 64, 128, 5, 7, 0, 1, 0, 0, 0, 10, 10, 10},      /* prefill rows after earlier ones */
            {4, 24, 8, 32, 32, 64, 128, 1, 70, 0, 1, 0, 1, 0, 10, 10, 10},     /* a selection list */
            {3, 16, 16, 20, 48, 0, 96, 3, 9, 0, 0, 1, 0, 1, 1, 1, 1},          /* no q latent, rotate-half, split, gate, int8 */
            {2, 64, 0, 64, 128, 64, 128, 4, 66, 0, 0, 1, 1, 0, 4, 4, 4},       /* NoPE, int4-g64, lists */
            {2, 32, 64, 16, 64, 128, 128, 1, 300, 5, 1, 0, 0, 0, 2, 2, 11},    /* a long context from kv_start, int4 rows, bf16 */
            {2, 192, 64, 256, 512, 128, 256, 2, 70, 0, 1, 0, 0, 0, 4, 4, 4},   /* GLM-5.2's heads */
            {1, 16, 8, 16, 1024, 32, 64, 1, 20, 0, 1, 0, 0, 1, 10, 5, 10},     /* the largest latent, int3-g64 */
        };
        for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) test_mla(cases[k]);
        printf("mla done\n");
        test_dsa(1, 30, 2, 16, 8, 0, 0); test_dsa(4, 40, 4, 32, 16, 0, 0); test_dsa(3, 5, 2, 16, 64, 0, 0);
        test_dsa(3, 5, 2, 16, 64, 1, 0); test_dsa(2, 700, 3, 64, 100, 0, 0); test_dsa(1, 9, 64, 64, 4, 0, 0);
        test_dsa(2, 900, 2, 16, 50, 0, 1);
        printf("dsa done\n");
        test_kpool(3, 20, 4, 16, 2, 8, 1); test_kpool(2, 9, 1, 16, 4, 8, 1); test_kpool(4, 300, 4, 64, 4, 64, 1);
        test_kpool(1, 2, 4, 16, 2, 4, 0); test_kpool(2, 30, 16, 16, 2, 6, 1);
        printf("kpool done\n");
    }
    if (!vkc_kda_ready() || !vkc_mhc_ready()) { fails++; printf("FAIL: the KDA or mHC shaders did not load\n"); }
    else {
        test_kda(3, 16, 4, 1.f); test_kda(2, 32, 4, 1.f); test_kda(4, 128, 3, 1.f); test_kda(1, 64, 2, 1.f);
        test_kda(2, 16, 4, 0.002f);
        printf("kda done\n");
        test_mhc(2, 64, 3); test_mhc(4, 96, 20); test_mhc(3, 40, 1);
        printf("mhc done\n");
    }
    if (!vkc_dsv4_ready()) { fails++; printf("FAIL: the DeepSeek V4 shader did not load\n"); }
    else {
        test_ds_attn(1, 4, 64, 8, 5, 13, 0); test_ds_attn(3, 2, 512, 128, 40, 300, 0); test_ds_attn(2, 3, 100, 6, 9, 20, 0);
        test_ds_attn(2, 2, 64, 8, 12, 2900, 0); test_ds_attn(3, 4, 64, 8, 5, 13, 1);
        test_ds_rope(0); test_ds_rope(1);
        test_ds_compress(2, 0, 0); test_ds_compress(1, 0, 0); test_ds_compress(4, 0, 0); test_ds_compress(4, 1, 1);
        test_ds_compress(3, 0, 1);
        test_ds_index(1, 30, 1, 4, 32, 4, 2, 2, 0); test_ds_index(5, 40, 2, 4, 32, 4, 2, 2, 0); test_ds_index(3, 0, 1, 2, 16, 8, 2, 3, 0);
        test_ds_index(2, 900, 1, 3, 64, 100, 4, 8, 0); test_ds_index(4, 300, 4, 16, 64, 16, 1, 3, 1); test_ds_index(3, 1, 2, 2, 16, 4, 2, 1, 0);
        test_ds_engram(1, 4, 128); test_ds_engram(3, 2, 300);
        printf("dsv4 done\n");
        int in;
        for (in = 0; in < 2; in++) {
            test_ds_round(VKC_DS_BF16, 6, 2, 50, 0, 0, in); test_ds_round(VKC_DS_BF16, 300, 1, 1000, 0, 0, in);
            test_ds_round(VKC_DS_E4M3, 6, 2, 300, 128, 0, in); test_ds_round(VKC_DS_E4M3, 9, 3, 448, 64, 1, in);
            test_ds_round(VKC_DS_E4M3, 40, 1, 4096, 128, 0, in); test_ds_round(VKC_DS_E4M3, 5, 1, 70, 256, 1, in);
            test_ds_round(VKC_DS_E2M1, 6, 2, 128, 32, 1, in); test_ds_round(VKC_DS_E2M1, 10, 5, 100, 32, 0, in);
            test_ds_round(VKC_DS_HADAMARD, 6, 2, 128, 0, 0, in); test_ds_round(VKC_DS_HADAMARD, 3, 1, 4096, 0, 0, in);
            test_ds_round(VKC_DS_HADAMARD, 4, 4, 32, 0, 0, in); test_ds_round(VKC_DS_HADAMARD, 2, 1, 1, 0, 0, in);
        }
        test_ds_swiglu(5000, 10.0f); test_ds_swiglu(3000, 0.0f);
        printf("dsv4 rounding done\n");
    }
    VkcStats st; vkc_stats(&st);
    printf("chain: %llu frames, %llu ops, %llu matmuls (%llu GEMM), %llu barriers\n", st.frames, st.ops, st.matmuls, st.gemms, st.barriers);
    vkc_shutdown();
    coli_vk_shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
