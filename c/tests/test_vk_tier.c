/* The Vulkan routed-expert tier (vk_tier.c) against a CPU reference, on any Vulkan
 * device (CI: Lavapipe). A small synthetic MoE -- L layers of E experts, top-K --
 * whose weights are random in every source format the tier accepts, decoded here
 * on their own, independently of the tier and of the backend. Gates:
 *   formats  every VktSrc kind and both activations: each row the device returns
 *            equals the reference expert output, and the rank-order sum of a mixed
 *            step (device and CPU rows) equals the all-CPU sum, within 2e-3;
 *   warm     a history fills the budget in heat order (vkt_plan / vkt_put);
 *   adapt    a budget of a third of the experts and a hot set that moves halfway:
 *            experts are evicted, the new hot set becomes resident and its steps
 *            go to the device;
 *   partial  a step with more assignments than max_rows: the device takes
 *            max_rows of them, the CPU the rest, the sum is still right;
 *   books    device + CPU = routed, resident <= budget, no failed upload.
 *
 *   make tests/test_vk_tier VK=1 && VK_ICD_FILENAMES=.../lvp_icd.json ./tests/test_vk_tier */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../backend_vulkan.h"
#include "../vk_tier.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

enum { L = 2, E = 10, H = 192, F = 128, K = 3 };
static unsigned rng = 7;
static unsigned rnd(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static float frnd(void) { return (float)((int)(rnd() % 2001) - 1000) / 1000.0f; }

/* ---- one matrix in a source format, and its own decode ------------------------- */
typedef struct { VktFmt f; int I, O; uint8_t *codes; void *scales; float *w; } Mat;
static size_t row_bytes(VktSrc k, int I) {
    switch (k) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return (size_t)I;
    case VKT_SRC_I3_G64: return ((size_t)I + 63) / 64 * 24;
    case VKT_SRC_BF16: return (size_t)I * 2;
    case VKT_SRC_F32: return (size_t)I * 4;
    default: return ((size_t)I + 1) / 2;
    }
}
static float e4m3(uint8_t b) {
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m / 512.0f : ldexpf(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -v : v;
}
static int nib(const uint8_t *row, int i) { return (row[i >> 1] >> ((i & 1) * 4)) & 15; }
/* the weight value (o, i) before scales, as the format defines it */
static float code(const Mat *m, int o, int i) {
    const uint8_t *row = m->codes + (size_t)o * row_bytes(m->f.kind, m->I);
    switch (m->f.kind) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
        return (float)(int8_t)row[i];
    case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4S_PAIRS_GS: { int v = nib(row, i); return (float)(v >= 8 ? v - 16 : v); }
    case VKT_SRC_I4U_PAIRS_ROW: case VKT_SRC_I4U_PAIRS_GS: return (float)(nib(row, i) - 8);
    case VKT_SRC_I4U_PLANAR64: {
        int full = m->I / 64 * 64;
        if (i >= full) return (float)(nib(row + full / 2, i - full) - 8);
        const uint8_t *blk = row + (size_t)(i / 64) * 32; int k = i % 64;
        return (float)((k < 32 ? (blk[k] & 15) : (blk[k - 32] >> 4)) - 8);
    }
    case VKT_SRC_I3_G64: {
        const uint8_t *lo = row + (size_t)(i / 64) * 24, *hi = lo + 16; int j = i % 64;
        return (float)((int)(((lo[j >> 2] >> ((j & 3) * 2)) & 3u) | (((hi[j >> 3] >> (j & 7)) & 1u) << 2)) - 4);
    }
    case VKT_SRC_MXFP4_F32: case VKT_SRC_MXFP4_E8M0: {
        static const float lut[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        int v = nib(row, i); return (v & 8) ? -lut[v & 7] : lut[v & 7];
    }
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return e4m3(row[i]);
    case VKT_SRC_BF16: { uint16_t h; memcpy(&h, row + 2 * i, 2); uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    case VKT_SRC_F32: { float f; memcpy(&f, row + 4 * i, 4); return f; }
    default: return 0;
    }
}
static int per_row(VktSrc k) { return k == VKT_SRC_I8_ROW || k == VKT_SRC_I8_AS_I4_ROW || k == VKT_SRC_I4S_PAIRS_ROW || k == VKT_SRC_I4U_PAIRS_ROW; }
static float scale(const Mat *m, int o, int i) {
    VktSrc k = m->f.kind;
    if (k == VKT_SRC_BF16 || k == VKT_SRC_F32) return 1.0f;
    if (per_row(k)) return ((const float *)m->scales)[o];
    int gs = k == VKT_SRC_I4U_PLANAR64 || k == VKT_SRC_I3_G64 ? 64 : m->f.gs, ng = (m->I + gs - 1) / gs;
    if (k == VKT_SRC_MXFP4_E8M0) return ldexpf(1.0f, ((const uint8_t *)m->scales)[(size_t)o * ng + i / gs] - 127);
    if (k == VKT_SRC_FP8_BLOCK) return ((const float *)m->scales)[(size_t)(o / gs) * ng + i / gs];
    return ((const float *)m->scales)[(size_t)o * ng + i / gs];
}
static void mat_make(Mat *m, VktFmt f, int I, int O) {
    m->f = f; m->I = I; m->O = O;
    size_t rb = row_bytes(f.kind, I);
    m->codes = malloc(rb * O);
    for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)rnd();
    if (f.kind == VKT_SRC_I8_AS_I4_ROW || f.kind == VKT_SRC_I8_AS_I4_GS)   /* an unpacked int4: [-8, 7] */
        for (size_t i = 0; i < rb * O; i++) m->codes[i] = (uint8_t)(int8_t)((int)(m->codes[i] % 16) - 8);
    if (f.kind == VKT_SRC_FP8_GS || f.kind == VKT_SRC_FP8_BLOCK)
        for (size_t i = 0; i < rb * O; i++) if ((m->codes[i] & 0x7f) == 0x7f) m->codes[i] ^= 1;
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32)
        for (int i = 0; i < I * O; i++) {
            float v = frnd() * 0.1f;
            if (f.kind == VKT_SRC_F32) memcpy(m->codes + 4 * (size_t)i, &v, 4);
            else { uint32_t u; memcpy(&u, &v, 4); uint16_t h = (uint16_t)(u >> 16); memcpy(m->codes + 2 * (size_t)i, &h, 2); }
        }
    int gs = f.kind == VKT_SRC_I4U_PLANAR64 || f.kind == VKT_SRC_I3_G64 ? 64 : f.gs, ng = gs ? (I + gs - 1) / gs : 1;
    size_t ns = per_row(f.kind) ? (size_t)O : f.kind == VKT_SRC_FP8_BLOCK ? (size_t)((O + gs - 1) / gs) * ng : (size_t)O * ng;
    if (f.kind == VKT_SRC_MXFP4_E8M0) {
        uint8_t *s = malloc(ns); for (size_t i = 0; i < ns; i++) s[i] = (uint8_t)(120 + rnd() % 4); m->scales = s;
    } else if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) m->scales = NULL;
    else { float *s = malloc(ns * 4); for (size_t i = 0; i < ns; i++) s[i] = 0.004f + (rnd() % 100) / 25000.0f; m->scales = s; }
    m->w = malloc((size_t)I * O * sizeof(float));
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) m->w[(size_t)o * I + i] = code(m, o, i) * scale(m, o, i);
}
static void mat_free(Mat *m) { free(m->codes); free(m->scales); free(m->w); }

/* ---- the synthetic model ---------------------------------------------------------- */
typedef struct { Mat g, u, d; } Ex;
static Ex ex[L][E];
static int g_act; static float g_limit;
static void expert_ref(const Ex *e, const float *x, float *y) {
    float h[F];
    for (int o = 0; o < F; o++) {
        double a = 0, b = 0;
        for (int i = 0; i < H; i++) { a += (double)e->g.w[(size_t)o * H + i] * x[i]; b += (double)e->u.w[(size_t)o * H + i] * x[i]; }
        float g = (float)a, u = (float)b;
        if (g_act == VKT_ACT_SITU) h[o] = 4.f * tanhf(g / 4.f) * (1.f / (1.f + expf(-g))) * (25.f * tanhf(u / 25.f));
        else {
            if (g_limit > 0) { if (g > g_limit) g = g_limit; if (u > g_limit) u = g_limit; if (u < -g_limit) u = -g_limit; }
            h[o] = g / (1.f + expf(-g)) * u;
        }
    }
    for (int o = 0; o < H; o++) { double a = 0; for (int i = 0; i < F; i++) a += (double)e->d.w[(size_t)o * F + i] * h[i]; y[o] = (float)a; }
}
static VktExpertSrc src_of(const Ex *e) {
    return (VktExpertSrc){e->g.codes, e->u.codes, e->d.codes, e->g.scales, e->u.scales, e->d.scales};
}
static void model_make(VktFmt gu, VktFmt dn) {
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        mat_make(&ex[l][e].g, gu, H, F); mat_make(&ex[l][e].u, gu, H, F); mat_make(&ex[l][e].d, dn, F, H);
    }
}
static void model_free(void) {
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) { mat_free(&ex[l][e].g); mat_free(&ex[l][e].u); mat_free(&ex[l][e].d); }
}
static double rel(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return den > 0 ? sqrt(num / den) : sqrt(num);
}

/* One layer step as an engine runs it (vk_tier.h step 3): issue, CPU pairs with a
 * note each, join, rank-order sum. Returns the worst relative error of a device row
 * against the reference and of the sum against the all-CPU sum. */
typedef struct { unsigned long long routed, dev; } Books;
static double step(int layer, int S, const int *idx, const float *w, Books *bk) {
    float *x = calloc((size_t)S * H, sizeof(float)), *cpu = calloc((size_t)S * K * H, sizeof(float));
    float *out = calloc((size_t)S * H, sizeof(float)), *ref = calloc((size_t)S * H, sizeof(float)), *y = malloc(sizeof(float) * H);
    uint8_t *taken = malloc((size_t)S * K); const float **dev = malloc(sizeof(*dev) * S * K);
    for (int i = 0; i < S * H; i++) x[i] = frnd();
    int n = vkt_issue(layer, x, S, K, idx, taken);
    for (int i = 0; i < S * K; i++) {
        if (taken[i]) continue;
        expert_ref(&ex[layer][idx[i]], x + (size_t)(i / K) * H, cpu + (size_t)i * H);
        VktExpertSrc s = src_of(&ex[layer][idx[i]]); vkt_note(layer, idx[i], &s);
    }
    int joined = n ? vkt_join(dev) : 1;
    CHECK(joined, "join failed");
    double worst = 0;
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int i = s * K + k;
            expert_ref(&ex[layer][idx[i]], x + (size_t)s * H, y);
            for (int d = 0; d < H; d++) ref[(size_t)s * H + d] += w[i] * y[d];
            const float *c = taken[i] && joined ? dev[i] : cpu + (size_t)i * H;
            if (taken[i] && joined) { double r = rel(c, y, H); if (r > worst) worst = r; }
            for (int d = 0; d < H; d++) out[(size_t)s * H + d] += w[i] * c[d];
            bk->routed++; bk->dev += taken[i] != 0;
        }
    double r = rel(out, ref, S * H);
    if (r > worst) worst = r;
    free(x); free(cpu); free(out); free(ref); free(y); free(taken); free(dev);
    return worst;
}
/* K distinct experts per row from a skewed draw over [lo, lo+span) */
static void route(int S, int lo, int span, int *idx, float *w) {
    for (int s = 0; s < S; s++)
        for (int k = 0; k < K; k++) {
            int e, dup;
            do { unsigned r = rnd() % 100; e = lo + (r < 60 ? (int)(r % 2) : (int)(r % span)) % span; e %= E;
                 dup = 0; for (int j = 0; j < k; j++) dup |= idx[s * K + j] == e; } while (dup);
            idx[s * K + k] = e; w[s * K + k] = 0.2f + (rnd() % 100) / 200.0f;
        }
}

static VktConfig cfg_of(VktFmt gu, VktFmt dn, int act, float limit) {
    return (VktConfig){.engine = "test", .layers = L, .experts = E, .hidden = H, .inter = F, .topk = K,
                       .gate_up = gu, .down = dn, .act = act, .act_limit = limit, .act_a = 4.f, .act_b = 25.f,
                       .max_rows = 1 << 20};
}
static void set_budget(int experts, VktFmt gu, VktFmt dn) {
    char b[64]; snprintf(b, sizeof b, "%.12f", (experts + 0.5) * vkt_expert_bytes(H, F, gu, dn) / 1073741824.0);
    setenv("COLI_VK_TIER_GB", b, 1);
}

/* every format: the whole model fits, experts are promoted as they pass by */
static void formats(void) {
    struct { VktFmt gu, dn; int act; float limit; const char *name; } cs[] = {
        {{VKT_SRC_I8_ROW, 0}, {VKT_SRC_I8_ROW, 0}, VKT_ACT_SWIGLU, 0, "int8 per row"},
        {{VKT_SRC_I8_GS, 32}, {VKT_SRC_I8_GS, 32}, VKT_ACT_SWIGLU, 0, "int8 gs32"},
        {{VKT_SRC_I8_AS_I4_ROW, 0}, {VKT_SRC_I8_AS_I4_ROW, 0}, VKT_ACT_SWIGLU, 0, "unpacked int4 per row"},
        {{VKT_SRC_I8_AS_I4_GS, 64}, {VKT_SRC_I8_GS, 32}, VKT_ACT_SWIGLU, 0, "unpacked int4 gs64, int8 down"},
        {{VKT_SRC_I4S_PAIRS_ROW, 0}, {VKT_SRC_I4S_PAIRS_ROW, 0}, VKT_ACT_SWIGLU, 0, "signed int4 pairs per row"},
        {{VKT_SRC_I4S_PAIRS_GS, 32}, {VKT_SRC_I4S_PAIRS_GS, 32}, VKT_ACT_SWIGLU, 0, "signed int4 pairs gs32"},
        {{VKT_SRC_I4U_PAIRS_ROW, 0}, {VKT_SRC_I4U_PAIRS_ROW, 0}, VKT_ACT_SWIGLU, 0, "int4 v+8 per row"},
        {{VKT_SRC_I4U_PAIRS_GS, 64}, {VKT_SRC_I4U_PAIRS_GS, 64}, VKT_ACT_SWIGLU, 2.5f, "int4 v+8 gs64, SwiGLU limit"},
        {{VKT_SRC_I4U_PLANAR64, 64}, {VKT_SRC_I4U_PLANAR64, 64}, VKT_ACT_SWIGLU, 0, "planar int4-g64"},
        {{VKT_SRC_I3_G64, 0}, {VKT_SRC_I3_G64, 0}, VKT_ACT_SWIGLU, 0, "int3-g64"},
        {{VKT_SRC_MXFP4_F32, 32}, {VKT_SRC_MXFP4_F32, 32}, VKT_ACT_SWIGLU, 0, "MXFP4 f32 scales"},
        {{VKT_SRC_MXFP4_E8M0, 32}, {VKT_SRC_MXFP4_E8M0, 32}, VKT_ACT_SITU, 0, "MXFP4 ue8m0, SiTU-GLU"},
        {{VKT_SRC_FP8_GS, 32}, {VKT_SRC_FP8_GS, 32}, VKT_ACT_SWIGLU, 0, "fp8 gs32"},
        {{VKT_SRC_FP8_BLOCK, 128}, {VKT_SRC_FP8_BLOCK, 128}, VKT_ACT_SWIGLU, 0, "fp8 128x128 blocks"},
        {{VKT_SRC_BF16, 0}, {VKT_SRC_BF16, 0}, VKT_ACT_SWIGLU, 0, "bf16"},
        {{VKT_SRC_F32, 0}, {VKT_SRC_F32, 0}, VKT_ACT_SWIGLU, 0, "f32"},
    };
    for (size_t c = 0; c < sizeof cs / sizeof *cs; c++) {
        model_make(cs[c].gu, cs[c].dn);
        g_act = cs[c].act; g_limit = cs[c].limit;
        set_budget(L * E, cs[c].gu, cs[c].dn);
        setenv("COLI_VK_TIER_RATE", "64", 1);
        VktConfig vc = cfg_of(cs[c].gu, cs[c].dn, cs[c].act, cs[c].limit);
        int on = vkt_init(&vc, NULL);
        CHECK(on, "%s: the tier did not start", cs[c].name);
        if (!on) { model_free(); continue; }
        Books bk = {0, 0}; double worst = 0;
        static const int Ss[] = {1, 3, 20, 1, 2, 24};
        for (int t = 0; t < 6; t++)
            for (int l = 0; l < L; l++) {
                int idx[24 * K]; float w[24 * K];
                route(Ss[t], 0, E, idx, w);
                double r = step(l, Ss[t], idx, w, &bk);
                if (r > worst) worst = r;
            }
        printf("  %-34s device %3llu of %3llu assignments, worst relative error %.2e\n", cs[c].name, bk.dev, bk.routed, worst);
        CHECK(bk.dev > 0, "%s: nothing ran on the device", cs[c].name);
        CHECK(worst < 2e-3, "%s: device rows off the reference (%.3g)", cs[c].name, worst);
        vkt_shutdown(); model_free();
    }
}

/* a history fills the budget in heat order */
static void warm(void) {
    VktFmt f = {VKT_SRC_I4U_PAIRS_GS, 64};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(7, f, f);
    uint32_t hist[L][E], *rows[L];
    for (int l = 0; l < L; l++) { rows[l] = hist[l]; for (int e = 0; e < E; e++) hist[l][e] = (uint32_t)(1000 - 37 * (l * E + e) % 700); }
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, rows), "warm: the tier did not start");
    int pl[L * E], pe[L * E];
    int n = vkt_plan(pl, pe, L * E);
    CHECK(n == 7, "warm: planned %d experts for a budget of 7", n);
    /* the plan is the 7 hottest, hottest first */
    int ok = 1;
    for (int i = 1; i < n; i++) ok &= hist[pl[i - 1]][pe[i - 1]] >= hist[pl[i]][pe[i]];
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) {
        int planned = 0; for (int i = 0; i < n; i++) planned |= pl[i] == l && pe[i] == e;
        if (!planned) ok &= hist[l][e] <= hist[pl[n - 1]][pe[n - 1]];
    }
    CHECK(ok, "warm: the plan is not the hottest experts in heat order");
    for (int i = 0; i < n; i++) { VktExpertSrc s = src_of(&ex[pl[i]][pe[i]]); vkt_put(pl[i], pe[i], &s); }
    vkt_put_done();
    int res = 0; for (int i = 0; i < n; i++) res += vkt_resident(pl[i], pe[i]);
    CHECK(res == n, "warm: %d of %d planned experts resident", res, n);
    Books bk = {0, 0};
    int idx[K] = {pe[0], (pe[0] + 1) % E, (pe[0] + 2) % E}; float w[K] = {0.5f, 0.3f, 0.2f};
    double r = step(pl[0], 1, idx, w, &bk);
    CHECK(bk.dev >= 1 && r < 2e-3, "warm: the hottest expert is not served from the device (%llu, %.3g)", bk.dev, r);
    printf("  warm start: %d experts planned in heat order and resident, the hottest served by the device\n", n);
    vkt_shutdown(); model_free();
}

/* a third of the experts fit; the hot set moves halfway */
static void adapt(void) {
    VktFmt f = {VKT_SRC_FP8_GS, 32};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E / 3, f, f);
    setenv("COLI_VK_TIER_RATE", "4", 1);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0);
    CHECK(vkt_init(&vc, NULL), "adapt: the tier did not start");
    Books a = {0, 0}, b1 = {0, 0}, b2 = {0, 0}; double worst = 0;
    int idx[4 * K]; float w[4 * K];
    /* LFRU: frequency first, halved every 1024 tokens, so an established hot set
     * yields over a few hundred steps of a new one, not at its first appearance */
    for (int t = 0; t < 100; t++)                     /* phase A: experts 0..4 */
        for (int l = 0; l < L; l++) { route(4, 0, 5, idx, w); double r = step(l, 4, idx, w, &a); if (r > worst) worst = r; }
    int resA = 0; for (int l = 0; l < L; l++) for (int e = 0; e < 5; e++) resA += vkt_resident(l, e);
    for (int t = 0; t < 400; t++)                     /* phase B: experts 5..9 */
        for (int l = 0; l < L; l++) { route(4, 5, 5, idx, w); double r = step(l, 4, idx, w, t < 200 ? &b1 : &b2); if (r > worst) worst = r; }
    int resB = 0, resA2 = 0, res = 0;
    for (int l = 0; l < L; l++) for (int e = 0; e < E; e++) { int v = vkt_resident(l, e); res += v; if (e >= 5) resB += v; else resA2 += v; }
    printf("  adapt: phase A device %.0f%%, resident A %d; phase B device %.0f%% then %.0f%%, resident B %d, A %d, all %d (budget %d)\n",
           100.0 * a.dev / a.routed, resA, 100.0 * b1.dev / b1.routed, 100.0 * b2.dev / b2.routed, resB, resA2, res, L * E / 3);
    vkt_report("test", 0, 0);
    CHECK(resA > 0 && a.dev > 0, "adapt: phase A promoted nothing");
    CHECK(resB > resA2, "adapt: the new hot set did not take the device (B %d, A %d)", resB, resA2);
    CHECK(res <= L * E / 3, "adapt: %d resident over a budget of %d", res, L * E / 3);
    CHECK(b2.dev * b1.routed > b1.dev * b2.routed, "adapt: the device share did not rise in phase B");
    CHECK(worst < 2e-3, "adapt: relative error %.3g", worst);
    vkt_shutdown(); model_free();
}

/* more assignments than max_rows: the rest stays on the CPU */
static void partial(void) {
    VktFmt f = {VKT_SRC_I8_ROW, 0};
    model_make(f, f); g_act = VKT_ACT_SWIGLU; g_limit = 0;
    set_budget(L * E, f, f);
    VktConfig vc = cfg_of(f, f, VKT_ACT_SWIGLU, 0); vc.max_rows = 5;
    setenv("COLI_VK_TIER_RATE", "64", 1);              /* every expert resident after a few steps */
    CHECK(vkt_init(&vc, NULL), "partial: the tier did not start");
    Books bk = {0, 0}; int idx[8 * K]; float w[8 * K];
    for (int t = 0; t < 8; t++) for (int l = 0; l < L; l++) { route(8, 0, E, idx, w); step(l, 8, idx, w, &bk); }
    Books one = {0, 0};
    route(8, 0, E, idx, w);
    double r = step(0, 8, idx, w, &one);
    printf("  partial: a step of %llu assignments, the device took %llu (max_rows 5), relative error %.2e\n", one.routed, one.dev, r);
    CHECK(one.dev > 0 && one.dev <= 5, "partial: the device took %llu of a 5-row cap", one.dev);
    CHECK(r < 2e-3, "partial: relative error %.3g", r);
    vkt_shutdown(); model_free();
}

int main(int argc, char **argv) {
    char buf[1024];
    const char *spv = argc > 1 ? argv[1] : coli_vk_shader_path(buf, sizeof buf);
    if (!coli_vk_init(spv)) { printf("FAIL: no Vulkan device (shaders %s)\n", spv); return 1; }
    setenv("COLI_VK_TIER_RESERVE_GB", "0", 1);
    printf("formats:\n"); formats();
    printf("warm:\n"); warm();
    printf("adapt:\n"); adapt();
    printf("partial:\n"); partial();
    ColiVkPoolStats ps; coli_vk_pool_stats(1, &ps);
    CHECK(ps.live == 0, "%d tier ranges still live after every shutdown", ps.live);
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails != 0;
}
