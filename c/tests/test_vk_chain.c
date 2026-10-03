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
 *   sconv    inkling's short convolution (residual inside, the ring carried across
 *            calls) and its scalar multiply and divide
 *   relattn  inkling's attention: the relative-position bias, tau, a sliding window
 *            over a ring the step wraps, the step's rows read from its own K/V, and a
 *            global layer; the shared experts' weighted add (HC_APPLY over one stream)
 *   inkling's hidden size: the GEMV and the norm at D = 6144, the GEMV at I = 24576
 *
 *   make vk-chain-check VK=1   (VK_ICD_FILENAMES=.../lvp_icd.json for Lavapipe) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../backend_vulkan.h"
#include "../vk_chain.h"

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

/* ---- inkling: the short convolution, the scale, the attention ----------------------- */
/* inkling.c's sconv_apply for S rows of C channels, in place, the ring (CK-1 raw inputs) carried */
static void sconv_ref(int S, int C, int CK, float *x, const float *w, float *ring) {
    int P = CK - 1;
    float *col = malloc((size_t)(P + S) * sizeof *col);
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < P; j++) col[j] = ring[c * P + j];
        for (int t = 0; t < S; t++) col[P + t] = x[(size_t)t * C + c];
        for (int t = 0; t < S; t++) {
            float acc = 0.f;
            for (int j = 0; j < CK; j++) acc += w[c * CK + j] * col[t + j];
            x[(size_t)t * C + c] = acc + col[P + t];
        }
        for (int j = 0; j < P; j++) ring[c * P + j] = col[S + j];
    }
    free(col);
}
static void test_sconv(int C, int CK) {
    int S1 = 6, S2 = 1, P = CK - 1, xo = 8, ro = 4;
    size_t nx = (size_t)(S1 + S2) * C;
    float *w = fvec((size_t)C * CK, 0.5f), *x = fvec(nx, 1.f), *ring = fvec((size_t)C * P, 1.f);
    float *xr = malloc(nx * sizeof *xr), *rr = malloc((size_t)C * P * sizeof *rr);
    memcpy(xr, x, nx * sizeof *xr); memcpy(rr, ring, (size_t)C * P * sizeof *rr);
    sconv_ref(S1, C, CK, xr, w, rr);
    sconv_ref(S2, C, CK, xr + (size_t)S1 * C, w, rr);
    VkcBuf *wb = up(w, (size_t)C * CK), *xb = vkc_buf((xo + nx) * 4, VKC_DEV), *rb = vkc_buf((ro + (size_t)C * P) * 4, VKC_DEV);
    vkc_begin(); vkc_write(xb, xo, x, nx * 4); vkc_write(rb, ro, ring, (size_t)C * P * 4); vkc_submit(1);
    VkcSconv p1 = {0, S1, C, CK, xo, C, 0, ro, 0, 1.f}, p2 = {0, S2, C, CK, xo + S1 * C, C, 0, ro, 0, 1.f};
    vkc_begin(); int ok = vkc_sconv(xb, wb, rb, &p1); vkc_submit(0);     /* left in flight */
    vkc_begin(); ok &= vkc_sconv(xb, wb, rb, &p2); vkc_submit(1);
    float *o = down(xb, xo, nx), *r = down(rb, ro, (size_t)C * P);
    double e1 = relerr(o, xr, nx, 1e-3), e2 = relerr(r, rr, (size_t)C * P, 1e-3);
    CHECK(ok && e1 < 1e-6 && e2 == 0, "sconv C %d CK %d: out %.2e ring %.2e", C, CK, e1, e2);
    /* the scalar multiply and divide over n floats at an offset */
    int n = 3 * C + 5; float fc = 24.f;
    for (int mode = 1; mode <= 2; mode++) {
        vkc_begin(); vkc_write(xb, xo, x, (size_t)n * 4); vkc_submit(1);
        VkcSconv ps = {mode, 0, 0, 0, xo, 0, 0, 0, n, fc};
        vkc_begin(); ok = vkc_sconv(xb, NULL, NULL, &ps); vkc_submit(1);
        float *y = down(xb, xo, n), *ref = malloc((size_t)n * sizeof *ref);
        for (int i = 0; i < n; i++) ref[i] = mode == 1 ? x[i] * fc : x[i] / fc;
        double e = relerr(y, ref, n, 1e-6);
        CHECK(ok && e < 1e-6, "sconv mode %d: err %.2e", mode, e);
        free(y); free(ref);
    }
    vkc_free(wb); vkc_free(xb); vkc_free(rb);
    free(w); free(x); free(ring); free(xr); free(rr); free(o); free(r);
}
/* inkling.c's attention() for S rows from pos_base: the K/V of every position in kt/vt
 * ([T][KVH*hd]), r [S][H*dr], relp [dr][ext], tau [S]; the device sees the positions
 * before pos_base through a ring of cap rows (t % cap, the latest such t < pos_base)
 * and the step's own rows in a scratch, k at ks, v at vs, kvd floats a row. */
static void test_relattn(int S, int pos_base, int window, int cap, int hd, int dr, int ext) {
    int H = 4, KVH = 2, kvd = KVH * hd, T = pos_base + S, koff = 32, roff = 16, poff = 24, toff = 8;
    float *q = fvec((size_t)S * H * hd, 1.f), *kt = fvec((size_t)T * kvd, 1.f), *vt = fvec((size_t)T * kvd, 1.f);
    float *r = fvec((size_t)S * H * dr, 0.5f), *relp = fvec((size_t)dr * ext, 0.5f), *tau = malloc(S * sizeof *tau);
    for (int s = 0; s < S; s++) tau[s] = 1.f + (rnd() % 1000) / 2000.f;
    float scale = 1.f / (float)hd;
    float *ref = malloc((size_t)S * H * hd * sizeof *ref);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        int kvh = h / (H / KVH), qpos = pos_base + s, t0 = window > 0 && qpos - window + 1 > 0 ? qpos - window + 1 : 0;
        int n = qpos - t0 + 1;
        double *sc = malloc(n * sizeof *sc), mx = -1e300, sum = 0;
        for (int j = 0; j < n; j++) {
            int t = t0 + j, dist = qpos - t;
            double a = 0; for (int d = 0; d < hd; d++) a += (double)q[((size_t)s * H + h) * hd + d] * kt[(size_t)t * kvd + kvh * hd + d];
            double b = 0; if (dist < ext) for (int k = 0; k < dr; k++) b += (double)r[((size_t)s * H + h) * dr + k] * relp[k * ext + dist];
            sc[j] = tau[s] * (a * scale + b); if (sc[j] > mx) mx = sc[j];
        }
        for (int j = 0; j < n; j++) { sc[j] = exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < hd; d++) {
            double a = 0; for (int j = 0; j < n; j++) a += sc[j] / sum * vt[(size_t)(t0 + j) * kvd + kvh * hd + d];
            ref[((size_t)s * H + h) * hd + d] = (float)a;
        }
        free(sc);
    }
    /* the ring as the host keeps it after the positions before pos_base, and the scratch */
    size_t nr = (size_t)koff + (size_t)KVH * cap * hd;
    float *kr = calloc(nr, sizeof *kr), *vr = calloc(nr, sizeof *vr);
    for (int t = 0; t < pos_base; t++) for (int h = 0; h < KVH; h++) {
        memcpy(kr + koff + ((size_t)h * cap + t % cap) * hd, kt + (size_t)t * kvd + h * hd, hd * sizeof(float));
        memcpy(vr + koff + ((size_t)h * cap + t % cap) * hd, vt + (size_t)t * kvd + h * hd, hd * sizeof(float));
    }
    size_t ks = 4, vs = ks + (size_t)S * kvd, nsc = vs + (size_t)S * kvd;
    float *scr = calloc(nsc, sizeof *scr);
    memcpy(scr + ks, kt + (size_t)pos_base * kvd, (size_t)S * kvd * sizeof(float));
    memcpy(scr + vs, vt + (size_t)pos_base * kvd, (size_t)S * kvd * sizeof(float));
    float *rbuf = calloc(roff + (size_t)S * H * dr, sizeof *rbuf), *pbuf = calloc(poff + (size_t)dr * ext, sizeof *pbuf), *tbuf = calloc(toff + S, sizeof *tbuf);
    memcpy(rbuf + roff, r, (size_t)S * H * dr * sizeof(float)); memcpy(pbuf + poff, relp, (size_t)dr * ext * sizeof(float));
    memcpy(tbuf + toff, tau, S * sizeof(float));
    VkcBuf *qb = up(q, (size_t)S * H * hd), *kb = up(kr, nr), *vb = up(vr, nr), *sb = up(scr, nsc), *rb = up(rbuf, roff + (size_t)S * H * dr);
    VkcBuf *pb = up(pbuf, poff + (size_t)dr * ext), *tb = up(tbuf, toff + S), *ob = vkc_buf((size_t)S * H * hd * 4, VKC_DOWN);
    VkcRelAttn p = {S, H, KVH, hd, pos_base, cap, window, ext, dr, 0, H * hd, 0, H * hd, koff, koff, (int)ks, (int)vs, kvd,
                    roff, H * dr, poff, toff, scale};
    vkc_begin(); int ok = vkc_relattn(qb, kb, vb, ob, sb, rb, pb, tb, &p); vkc_submit(1);
    double e = relerr((float *)vkc_ptr(ob), ref, (size_t)S * H * hd, 1e-3);
    CHECK(ok && e < 2e-5, "relattn S %d pos %d window %d cap %d hd %d d_rel %d ext %d: err %.2e", S, pos_base, window, cap, hd, dr, ext, e);
    vkc_free(qb); vkc_free(kb); vkc_free(vb); vkc_free(sb); vkc_free(rb); vkc_free(pb); vkc_free(tb); vkc_free(ob);
    free(q); free(kt); free(vt); free(r); free(relp); free(tau); free(ref); free(kr); free(vr); free(scr); free(rbuf); free(pbuf); free(tbuf);
}
/* the shared experts joining the routed sum: y = routed; y += w_j[r] * sh_j[r] in order */
static void test_weighted_add(void) {
    int R = 3, D = 70, NS = 2;
    float *routed = fvec((size_t)R * D, 2.f), *sh = fvec((size_t)NS * R * D, 2.f), *w = fvec((size_t)NS * R, 1.f), *ref = malloc((size_t)R * D * sizeof *ref);
    memcpy(ref, routed, (size_t)R * D * sizeof *ref);
    for (int j = 0; j < NS; j++) for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) ref[r * D + d] += w[j * R + r] * sh[((size_t)j * R + r) * D + d];
    VkcBuf *yb = up(routed, (size_t)R * D), *sb = up(sh, (size_t)NS * R * D), *wb = up(w, (size_t)NS * R);
    vkc_begin(); int ok = 1;
    for (int j = 0; j < NS; j++) { VkcEw p = {VKC_EW_HC_APPLY, R * D, D, 1, 0, 1, 0, j * R, j * R * D, 0, 0, 1.f}; ok &= vkc_ew(yb, wb, sb, NULL, NULL, &p); }
    vkc_submit(1);
    float *y = down(yb, 0, (size_t)R * D);
    double e = relerr(y, ref, (size_t)R * D, 1e-3);
    CHECK(ok && e < 1e-6, "weighted add: err %.2e", e);
    vkc_free(yb); vkc_free(sb); vkc_free(wb); free(routed); free(sh); free(w); free(ref); free(y);
}
static void test_inkling(void) {
    test_sconv(37, 4); test_sconv(6144, 4); test_sconv(9, 1); test_sconv(5, 9);
    test_relattn(1, 0, 0, 64, 16, 4, 32);              /* the first position, global */
    test_relattn(5, 200, 0, 300, 32, 16, 64);          /* global, distances past ext (bias 0) */
    test_relattn(1, 40, 16, 16, 32, 4, 16);            /* decode, a sliding window over a wrapped ring */
    test_relattn(20, 30, 16, 16, 16, 4, 16);           /* a step that wraps the ring more than once */
    test_relattn(130, 3, 0, 200, 24, 4, 200);          /* two tiles of positions */
    test_relattn(3, 513, 512, 512, 128, 16, 512);      /* the real window and ring, hd 128 */
    test_relattn(2, 9, 0, 64, 256, 64, 8);             /* the largest head and bank */
    test_weighted_add();
    for (int k = 0; k < 4; k++) { int f[4] = {1, 4, 10, 11}; test_matmul(f[k], 1, 6144, 48, 0, 0); }
    test_matmul(10, 1, 24576, 16, 0, 0);               /* past chain_gemv's staging: qmatmul.comp */
    test_matmul(4, 1, 24576, 16, 0, 0);
    VkcNorm np = {2, 6144, 1, 0, 6144, 6144, 0, 6144, 6144, 0, 0, 0, 1e-6f, 1.f};
    float *x = fvec(2 * 6144, 1.f), *w = fvec(6144, 1.f), *ref = malloc(2 * 6144 * sizeof *ref);
    for (int r = 0; r < 2; r++) { double ms = 0; for (int i = 0; i < 6144; i++) ms += (double)x[r * 6144 + i] * x[r * 6144 + i];
        float rr = 1.f / sqrtf((float)(ms / 6144) + 1e-6f); for (int i = 0; i < 6144; i++) ref[r * 6144 + i] = x[r * 6144 + i] * rr * w[i]; }
    VkcBuf *xb = up(x, 2 * 6144), *wb = up(w, 6144), *yb = vkc_buf(2 * 6144 * 4, VKC_DOWN);
    vkc_begin(); int ok = vkc_norm(xb, wb, yb, &np); vkc_submit(1);
    double e = relerr((float *)vkc_ptr(yb), ref, 2 * 6144, 1e-3);
    CHECK(ok && e < 1e-5, "norm D 6144: err %.2e", e);
    vkc_free(xb); vkc_free(wb); vkc_free(yb); free(x); free(w); free(ref);
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
    if (vkc_sconv_ready() && vkc_relattn_ready()) { test_inkling(); printf("inkling done\n"); }
    else CHECK(0, "inkling's ops: chain_sconv.spv or chain_relattn.spv did not load");
    VkcStats st; vkc_stats(&st);
    printf("chain: %llu frames, %llu ops, %llu matmuls (%llu GEMM), %llu barriers\n", st.frames, st.ops, st.matmuls, st.gemms, st.barriers);
    vkc_shutdown();
    coli_vk_shutdown();
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
