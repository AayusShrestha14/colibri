/* qi_gemm against the naive triple loop, every format, ragged shapes; and a
 * throughput line for the DiT's biggest shape (argv[1] = "bench"). */
#include "../qi_gemm.h"
#include <math.h>
#include <time.h>

static float frand(uint32_t *s){ *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.f) * 2.f - 1.f; }
static uint16_t to_bf16(float f){ union { float f; uint32_t u; } v; v.f = f; return (uint16_t)((v.u + 0x7fff + ((v.u >> 16) & 1)) >> 16); }

static int check(int M, int N, int K, int fmt, int use_bias){
    uint32_t s = 1234u + M * 7 + N * 13 + K * 17 + fmt;
    float *X = malloc(sizeof(float) * M * K), *Wf = malloc(sizeof(float) * N * K);
    float *Y = malloc(sizeof(float) * M * N), *R = malloc(sizeof(float) * M * N), *b = malloc(sizeof(float) * N);
    uint16_t *Wb = malloc(2 * (size_t)N * K); int8_t *Wq = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
    for (int i = 0; i < M * K; i++) X[i] = frand(&s);
    for (int i = 0; i < N * K; i++) Wf[i] = frand(&s);
    for (int i = 0; i < N; i++) b[i] = frand(&s);
    QiMat W = { fmt, N, K, Wf, NULL, 0 };
    if (fmt == QI_BF16) { for (int i = 0; i < N * K; i++) { Wb[i] = to_bf16(Wf[i]); Wf[i] = qi_bf16(Wb[i]); } W.w = Wb; }
    if (fmt == QI_I8) { qi_quantize_i8(Wf, N, K, Wq, sc); for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) Wf[(int64_t)n*K+k] = Wq[(int64_t)n*K+k] * sc[n]; W.w = Wq; W.sc = sc; }
    qi_gemm(Y, X, M, &W, use_bias ? b : NULL);
    double maxerr = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double acc = use_bias ? b[n] : 0;
        for (int k = 0; k < K; k++) acc += (double)X[(int64_t)m*K+k] * Wf[(int64_t)n*K+k];
        double e = fabs(acc - Y[(int64_t)m*N+n]) / (1.0 + fabs(acc));
        if (e > maxerr) maxerr = e;
    }
    int ok = maxerr < 1e-4;
    printf("%s M=%d N=%d K=%d fmt=%d bias=%d maxrel=%.2e\n", ok ? "ok  " : "FAIL", M, N, K, fmt, use_bias, maxerr);
    free(X); free(Wf); free(Y); free(R); free(b); free(Wb); free(Wq); free(sc);
    return ok;
}

int main(int argc, char **argv){
    if (argc > 1 && !strcmp(argv[1], "bench")) {
        int M = argc > 2 ? atoi(argv[2]) : 1296, N = argc > 3 ? atoi(argv[3]) : 12288, K = argc > 4 ? atoi(argv[4]) : 4096;
        float *X = malloc(sizeof(float) * (size_t)M * K), *Y = malloc(sizeof(float) * (size_t)M * N);
        int8_t *q = malloc((size_t)N * K); float *sc = malloc(sizeof(float) * N);
        uint32_t s = 7; for (size_t i = 0; i < (size_t)M * K; i++) X[i] = frand(&s);
        for (size_t i = 0; i < (size_t)N * K; i++) q[i] = (int8_t)(frand(&s) * 127); for (int i = 0; i < N; i++) sc[i] = 0.01f;
        QiMat W = { QI_I8, N, K, q, sc, 0 };
        qi_gemm(Y, X, M, &W, NULL);
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        int reps = 3; for (int r = 0; r < reps; r++) qi_gemm(Y, X, M, &W, NULL);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double t = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) * 1e-9;
        printf("M=%d N=%d K=%d int8: %.1f ms, %.1f GFLOP/s\n", M, N, K, t / reps * 1e3, 2.0 * M * N * K * reps / t * 1e-9);
        return 0;
    }
    int ok = 1;
    int shapes[][3] = { {1,1,1}, {5,17,33}, {6,16,256}, {7,31,257}, {64,48,300}, {193,40,520}, {13,100,64} };
    for (unsigned i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        for (int fmt = 0; fmt < 3; fmt++) ok &= check(shapes[i][0], shapes[i][1], shapes[i][2], fmt, i & 1);
    printf(ok ? "qi_gemm: all ok\n" : "qi_gemm: FAILED\n");
    return !ok;
}
