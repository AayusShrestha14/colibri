/* qi_gemm.h -- the one GEMM of the image engine.
 *
 * Y[M][N] = X[M][K] . W[N][K]^T (+ bias[N]), X and Y f32 row-major, W row-major
 * in one of three storage formats: f32, bf16, or int8 with one f32 scale per
 * row. Every linear layer of the text encoder and the DiT, every convolution
 * of the VAE (through im2col) and both products of the attention go through
 * here, so this is where the time of an image goes.
 *
 * Why not matmul_q_batch: the text engines' kernels are GEMVs that stream the
 * weight row once per output and sweep every activation row under it. That is
 * right for decode (one or two rows) and wrong for a DiT step, where M is the
 * number of image tokens (1296 at 768x432): the activations stop fitting in
 * cache and each output row re-reads all of them. Here W is dequantized into
 * f32 panels of NR rows x KC columns, X is walked in blocks of MC rows that stay
 * in L2, and a register tile of MR x NR accumulates in FMA registers.
 *
 * The weights stay int8 or bf16 in memory; the f32 copy only ever exists as one
 * packed panel per thread. Activations stay f32: no activation quantization, so
 * the result is the f32 product of the dequantized weights up to summation
 * order. */
#ifndef COLIBRI_QI_GEMM_H
#define COLIBRI_QI_GEMM_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

enum { QI_F32 = 0, QI_BF16 = 1, QI_I8 = 2 };

typedef struct {
    int fmt;            /* QI_F32 | QI_BF16 | QI_I8 */
    int N, K;           /* rows (outputs) x columns (inputs) */
    const void *w;      /* [N][K] in fmt */
    const float *sc;    /* QI_I8: one scale per row; NULL otherwise */
    int ld;             /* row stride in elements, 0 = K. Lets attention read one head's
                         * columns of K straight out of the [T][heads*hd] projection. */
} QiMat;

#define QI_MR 6
#define QI_NR 16
#define QI_KC 256
#define QI_MC 192

static inline float qi_bf16(uint16_t h){ union { uint32_t u; float f; } v; v.u = (uint32_t)h << 16; return v.f; }

/* Pack W rows [n0, n0+nr) x columns [k0, k0+kc) as f32, k-major: panel[k*QI_NR + j].
 * Rows past nr are zero so the kernel never branches on the edge. The int8 scale
 * is applied here, so the kernel is the same for every format. */
static void qi_pack_w(float *panel, const QiMat *W, int n0, int nr, int k0, int kc){
    for (int j = 0; j < QI_NR; j++) {
        if (j >= nr) { for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = 0.f; continue; }
        int64_t row = (int64_t)(n0 + j) * (W->ld ? W->ld : W->K) + k0;
        if (W->fmt == QI_F32) {
            const float *s = (const float *)W->w + row;
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = s[k];
        } else if (W->fmt == QI_BF16) {
            const uint16_t *s = (const uint16_t *)W->w + row;
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = qi_bf16(s[k]);
        } else {
            const int8_t *s = (const int8_t *)W->w + row;
            float sc = W->sc[n0 + j];
            for (int k = 0; k < kc; k++) panel[k*QI_NR + j] = (float)s[k] * sc;
        }
    }
}

/* acc[MR][NR] (+)= X[m0.., k0..k0+kc) . panel ; mr <= MR rows valid */
static void qi_kernel(float *Y, int ldy, const float *X, int ldx, const float *panel,
                      int kc, int mr, int nr, int accumulate){
#if defined(__AVX2__) && defined(__FMA__)
    if (mr == QI_MR) {
        __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps(), c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
        __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps(), c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
        __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps(), c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();
        const float *x0 = X, *x1 = X + ldx, *x2 = X + 2*ldx, *x3 = X + 3*ldx, *x4 = X + 4*ldx, *x5 = X + 5*ldx;
        for (int k = 0; k < kc; k++) {
            __m256 b0 = _mm256_loadu_ps(panel + k*QI_NR), b1 = _mm256_loadu_ps(panel + k*QI_NR + 8);
            __m256 a;
            a = _mm256_broadcast_ss(x0 + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
            a = _mm256_broadcast_ss(x1 + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
            a = _mm256_broadcast_ss(x2 + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
            a = _mm256_broadcast_ss(x3 + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
            a = _mm256_broadcast_ss(x4 + k); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
            a = _mm256_broadcast_ss(x5 + k); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        }
        float t[QI_MR][QI_NR];
        _mm256_storeu_ps(t[0], c00); _mm256_storeu_ps(t[0]+8, c01);
        _mm256_storeu_ps(t[1], c10); _mm256_storeu_ps(t[1]+8, c11);
        _mm256_storeu_ps(t[2], c20); _mm256_storeu_ps(t[2]+8, c21);
        _mm256_storeu_ps(t[3], c30); _mm256_storeu_ps(t[3]+8, c31);
        _mm256_storeu_ps(t[4], c40); _mm256_storeu_ps(t[4]+8, c41);
        _mm256_storeu_ps(t[5], c50); _mm256_storeu_ps(t[5]+8, c51);
        for (int i = 0; i < QI_MR; i++)
            for (int j = 0; j < nr; j++)
                Y[(int64_t)i*ldy + j] = accumulate ? Y[(int64_t)i*ldy + j] + t[i][j] : t[i][j];
        return;
    }
#endif
    float t[QI_MR][QI_NR];
    memset(t, 0, sizeof t);
    for (int k = 0; k < kc; k++) {
        const float *b = panel + k*QI_NR;
        for (int i = 0; i < mr; i++) {
            float a = X[(int64_t)i*ldx + k];
            for (int j = 0; j < QI_NR; j++) t[i][j] += a * b[j];
        }
    }
    for (int i = 0; i < mr; i++)
        for (int j = 0; j < nr; j++)
            Y[(int64_t)i*ldy + j] = accumulate ? Y[(int64_t)i*ldy + j] + t[i][j] : t[i][j];
}

/* Y[M][N] = X[M][K] . W^T (+ bias). ldx/ldy are the row strides (K and N for
 * dense rows; larger when X or Y is a column slice of a wider matrix). */
static void qi_gemm_ld(float *Y, int ldy, const float *X, int ldx, int M, const QiMat *W, const float *bias){
    const int N = W->N, K = W->K;
    if (M <= 0 || N <= 0) return;
    const int nblocks = (N + QI_NR - 1) / QI_NR;
    const int mblocks = (M + QI_MC - 1) / QI_MC;
    /* Parallel over (m-block, n-panel) tiles; each tile walks all of K, so the
     * sum for one output is always accumulated in the same order: the result
     * does not depend on the thread count. */
    #pragma omp parallel
    {
        float *panel = (float *)aligned_alloc(64, (size_t)QI_KC * QI_NR * sizeof(float));
        if (!panel) { fprintf(stderr, "OOM qi_gemm panel\n"); exit(1); }
        #pragma omp for schedule(dynamic, 1) collapse(2)
        for (int mb = 0; mb < mblocks; mb++)
            for (int nb = 0; nb < nblocks; nb++) {
                int m0 = mb * QI_MC, mc = M - m0 < QI_MC ? M - m0 : QI_MC;
                int n0 = nb * QI_NR, nr = N - n0 < QI_NR ? N - n0 : QI_NR;
                for (int k0 = 0; k0 < K; k0 += QI_KC) {
                    int kc = K - k0 < QI_KC ? K - k0 : QI_KC;
                    qi_pack_w(panel, W, n0, nr, k0, kc);
                    for (int i = 0; i < mc; i += QI_MR) {
                        int mr = mc - i < QI_MR ? mc - i : QI_MR;
                        qi_kernel(Y + (int64_t)(m0 + i) * ldy + n0, ldy,
                                  X + (int64_t)(m0 + i) * ldx + k0, ldx,
                                  panel, kc, mr, nr, k0 > 0);
                    }
                }
                if (bias)
                    for (int i = 0; i < mc; i++)
                        for (int j = 0; j < nr; j++) Y[(int64_t)(m0 + i) * ldy + n0 + j] += bias[n0 + j];
            }
        free(panel);
    }
}

static inline void qi_gemm(float *Y, const float *X, int M, const QiMat *W, const float *bias){
    qi_gemm_ld(Y, W->N, X, W->K, M, W, bias);
}

/* Per-row int8 quantization of an f32 matrix (max-abs / 127, round to nearest). */
static void qi_quantize_i8(const float *src, int N, int K, int8_t *q, float *sc){
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float *r = src + (int64_t)n * K;
        float am = 0.f;
        for (int k = 0; k < K; k++) { float a = r[k] < 0 ? -r[k] : r[k]; if (a > am) am = a; }
        float s = am > 1e-12f ? am / 127.f : 1.f, inv = 1.f / s;
        sc[n] = s;
        int8_t *d = q + (int64_t)n * K;
        for (int k = 0; k < K; k++) {
            float v = r[k] * inv;
            int iv = (int)(v < 0 ? v - 0.5f : v + 0.5f);
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            d[k] = (int8_t)iv;
        }
    }
}

#endif
