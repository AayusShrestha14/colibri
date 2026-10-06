/* qwenimage_chain.h -- Qwen-Image's diffusion transformer as a dense chain on the Vulkan
 * device (vk_chain.h). Included once by qwenimage.c in a COLI_VULKAN build, after the CPU
 * DiT it stands in for; COLI_VK_CHAIN decides (coli_vk_chain_decide).
 *
 * A denoising step runs every block on the device, the image rows staying there from
 * img_in to proj_out:
 *   x = img_in(latents)
 *   per block: h = LayerNorm(x) * (1 + scale1); q, k, v of h; an RMSNorm on every head of
 *   q and k; RoPE; the image rows' attention over the prompt's keys and values and their
 *   own; x += tanh(gate1) * o(attention); h = LayerNorm(x) * (1 + scale2);
 *   x += tanh(gate2) * out(silu(gate(h)) * proj(h))
 *   prediction = proj_out(LayerNorm(x) * (1 + outs))
 * The modulation (the timestep's embedding and its two one-row linears) stays on the
 * CPU and goes up as five vectors a step; the prompt's keys and values, which the CPU
 * computes once per prompt (dit_prefix), go up once per prompt. Crossing a step: the
 * latents (in_ch floats a row) up, the prediction down.
 *
 * The rotary pairs: the CPU rotates (2i, 2i+1) in place; the device writes each pair's
 * result into the two halves of the head, (i, i + hd/2) (chain_mla.comp's interleaved
 * style). That permutes q's and k's dimensions alike and leaves every q . k as it was;
 * the prompt's keys go up in that order too, and v is not touched.
 *
 * The device sums in other orders than the CPU (the norms in float trees, the GEMMs by
 * tiles, the softmax online), so a prediction agrees with the CPU's to rounding: the
 * oracle (tests/vulkan_engines.sh, mimo-qwenimage) holds every stage to its tolerance.
 *
 * Declines (the CPU's path runs, its matrices on the device one by one as before):
 * COLI_VK_CHAIN off, a matrix the device does not take, the device's memory refusing a
 * buffer. A device lost in a step: the step runs again on the CPU, and the CPU runs from
 * there (the latents are the host's). */
#include "vk_chain.h"

typedef struct {
    int ok, failed;
    int N, L;                    /* image rows and prompt rows the buffers hold */
    unsigned prefix_gen, step_gen;
    VkcBuf *prm;                 /* the q and k norms, [layers][2][hd] */
    VkcBuf *mod;                 /* VKC_UP, a step's [5][D]: 1+scale1, tanh(gate1), 1+scale2, tanh(gate2), 1+outs */
    VkcBuf *cs;                  /* VKC_UP, the image rows' (cos, sin) pairs, [N][hd] */
    VkcBuf *rng;                 /* VKC_UP, int32 [N][2]: every image row sees rows 0 .. L+N-1 */
    VkcBuf *x, *h, *a, *g, *u;   /* [N][D] and [N][mlp] */
    VkcBuf *qkv;                 /* q [N][D], then k and v [L+N][D] each: the prompt's rows first */
    VkcBuf *lat, *pred, *predd;  /* the latents, the prediction and its read-back, [N][in_ch] */
    VkcBuf **pk, **pv;           /* per block, the prompt's keys (permuted) and values, [L][D] */
    unsigned long long steps;
    double wait_ms;
} QiChain;

static QiChain g_qic;
static int g_qic_on = -1;        /* -1: not decided yet */
static unsigned g_qi_gen;        /* a new prompt prefix or step layout takes the next number */

static ColiVkTensor *qic_tensor(Lin *l) {
    if (l->vk) return (ColiVkTensor *)l->vk;
    if (l->m.ld) return NULL;
    int fmt = l->m.fmt == QI_I8 ? 1 : l->m.fmt == QI_BF16 ? 11 : 10;
    return coli_vk_tensor_ensure((ColiVkTensor **)&l->vk, l->m.w, l->m.fmt == QI_I8 ? l->m.sc : NULL, fmt,
                                 l->m.K, l->m.N, 0) ? (ColiVkTensor *)l->vk : NULL;
}

static void qic_off(const char *why) {
    if (!g_qic.failed) fprintf(stderr, "[VK] qwenimage chain: %s; the matrices run on the device one by one\n", why);
    g_qic.failed = 1;
}

/* The weights and the norms, once: the same device copies the per-matrix path uses. */
static int qic_setup(Dit *d) {
    int H = d->hd, Lr = d->layers;
    for (int l = 0; l < Lr; l++) {
        DitBlock *B = &d->B[l];
        Lin *m[7] = {&B->q, &B->k, &B->v, &B->o, &B->gate, &B->proj, &B->out};
        for (int k = 0; k < 7; k++)
            if (!qic_tensor(m[k])) { qic_off("the device refused a block's matrices"); return 0; }
    }
    if (!qic_tensor(&d->img_in) || !qic_tensor(&d->proj_out)) { qic_off("the device refused img_in or proj_out"); return 0; }
    float *n = malloc((size_t)Lr * 2 * H * sizeof(float));
    if (!n) return 0;
    for (int l = 0; l < Lr; l++) {
        memcpy(n + (size_t)l * 2 * H, d->B[l].nq, (size_t)H * sizeof(float));
        memcpy(n + (size_t)l * 2 * H + H, d->B[l].nk, (size_t)H * sizeof(float));
    }
    g_qic.prm = vkc_buf((size_t)Lr * 2 * H * sizeof(float), VKC_DEV);
    int ok = g_qic.prm && vkc_begin() && vkc_write(g_qic.prm, 0, n, (size_t)Lr * 2 * H * sizeof(float)) && vkc_submit(1);
    free(n);
    g_qic.pk = calloc(Lr, sizeof(VkcBuf *)); g_qic.pv = calloc(Lr, sizeof(VkcBuf *));
    g_qic.mod = vkc_buf(5 * (size_t)d->dim * sizeof(float), VKC_UP);
    if (!ok || !g_qic.pk || !g_qic.pv || !g_qic.mod) { qic_off("the device refused the parameters' buffer"); return 0; }
    g_qic.ok = 1;
    fprintf(stderr, "[VK] qwenimage chain: %d blocks on the device\n", Lr);
    return 1;
}

/* The buffers for N image rows after L prompt rows (grown, never shrunk). */
static int qic_buffers(Dit *d, int N, int L) {
    size_t D = d->dim, M = d->mlp, C = d->in_ch, n = N, t = (size_t)L + N;
    int ok = vkc_reserve(&g_qic.x, n * D * 4, VKC_DEV) && vkc_reserve(&g_qic.h, n * D * 4, VKC_DEV) &&
             vkc_reserve(&g_qic.a, n * D * 4, VKC_DEV) && vkc_reserve(&g_qic.g, n * M * 4, VKC_DEV) &&
             vkc_reserve(&g_qic.u, n * M * 4, VKC_DEV) && vkc_reserve(&g_qic.qkv, (n + 2 * t) * D * 4, VKC_DEV) &&
             vkc_reserve(&g_qic.lat, n * C * 4, VKC_DEV) && vkc_reserve(&g_qic.pred, n * C * 4, VKC_DEV) &&
             vkc_reserve(&g_qic.predd, n * C * 4, VKC_DOWN) && vkc_reserve(&g_qic.cs, n * d->hd * 4, VKC_UP) &&
             vkc_reserve(&g_qic.rng, n * 2 * 4, VKC_UP);
    if (!ok) { qic_off("the device refused a step's buffers"); return 0; }
    return 1;
}

/* The prompt's keys and values, once per prefix: the keys' heads in the device's order. */
static int qic_prefix(Dit *d, const Prefix *p) {
    int L = p->L, D = d->dim, H = d->heads, hd = d->hd, half = hd / 2;
    float *kp = malloc((size_t)L * D * sizeof(float));
    if (!kp) return 0;
    int ok = vkc_begin();
    for (int l = 0; ok && l < d->layers; l++) {
        ok = vkc_reserve(&g_qic.pk[l], (size_t)L * D * 4, VKC_DEV) && vkc_reserve(&g_qic.pv[l], (size_t)L * D * 4, VKC_DEV);
        for (int r = 0; ok && r < L; r++)
            for (int h = 0; h < H; h++) {
                const float *src = p->K[l] + (size_t)r * D + (size_t)h * hd;
                float *dst = kp + (size_t)r * D + (size_t)h * hd;
                for (int j = 0; j < half; j++) { dst[j] = src[2 * j]; dst[j + half] = src[2 * j + 1]; }
            }
        ok = ok && vkc_write(g_qic.pk[l], 0, kp, (size_t)L * D * 4) && vkc_write(g_qic.pv[l], 0, p->V[l], (size_t)L * D * 4) &&
             vkc_submit(1) && (l + 1 == d->layers || vkc_begin());
    }
    free(kp);
    if (!ok) { qic_off(vkc_lost() ? "the device was lost" : "the device refused the prompt's keys and values"); return 0; }
    g_qic.prefix_gen = p->gen;
    return 1;
}

/* One denoising step on the device: out[N][in_ch]. 0: not taken (the CPU runs it). */
static int qic_forward(Dit *d, const Prefix *p, DitStep *s, const float *lat, float t, float *out) {
    if (g_qic_on < 0) g_qic_on = g_vk_ready && vkc_init() &&
                                 coli_vk_chain_decide("qwenimage", 0, COLI_VK_CHAIN_UNMEASURED) == COLI_VK_CHAIN_ON;
    if (!g_qic_on || g_qic.failed || vkc_lost()) return 0;
    if (!g_qic.ok && !qic_setup(d)) return 0;
    int N = s->N, L = p->L, D = d->dim, H = d->heads, hd = d->hd, M = d->mlp, C = d->in_ch;
    if (hd > 128 || D > 4096 || hd % 2) { qic_off("a head or a row its ops do not take"); return 0; }
    if ((N != g_qic.N || L != g_qic.L) && !qic_buffers(d, N, L)) return 0;
    g_qic.N = N; g_qic.L = L;
    if (g_qic.prefix_gen != p->gen && !qic_prefix(d, p)) return 0;
    if (g_qic.step_gen != s->gen) {   /* the image rows' rotations and attention ranges */
        float *cs = vkc_ptr(g_qic.cs); int32_t *rg = vkc_ptr(g_qic.rng);
        for (int r = 0; r < N; r++) {
            for (int j = 0; j < hd / 2; j++) {
                cs[(size_t)r * hd + 2 * j] = s->cs[(size_t)r * (hd / 2) + j];
                cs[(size_t)r * hd + 2 * j + 1] = s->sn[(size_t)r * (hd / 2) + j];
            }
            rg[2 * r] = 0; rg[2 * r + 1] = L + N - 1;
        }
        g_qic.step_gen = s->gen;
    }
    double t0 = g_prof ? now_s() : 0;
    float *mod = fmalloc(4 * (size_t)D), *outs = fmalloc(D), *mv = vkc_ptr(g_qic.mod);
    dit_modulation(d, t, mod, outs);
    for (int i = 0; i < D; i++) {   /* the CPU's own factors: (1 + scale), tanhf(gate) */
        mv[i] = 1.f + mod[i]; mv[D + i] = tanhf(mod[D + i]);
        mv[2 * D + i] = 1.f + mod[2 * D + i]; mv[3 * D + i] = tanhf(mod[3 * D + i]);
        mv[4 * D + i] = 1.f + outs[i];
    }
    free(mod); free(outs);
    size_t KO = (size_t)N * D, VO = KO + (size_t)(L + N) * D;
    VkcEncNorm n1 = {0, N, D, 0, D, 0, 0, 0, D, 0, -1, 0, d->eps}, n2 = n1, no = n1;
    n2.w_off = 2 * D; no.w_off = 4 * D;
    VkcEw g1 = {VKC_EW_GATE_ADD, N * D, D, 1, 0, 0, 0, 0, 0, 0, D, 1.f}, g2 = g1;
    g2.e_off = 3 * D;
    VkcEw sw = {VKC_EW_SWIGLU, N * M, M, 1, 0, 0, 0, 0, 0, 0, 0, 1.f};
    VkcMlaRow rq = {N * H, H, hd, hd, VKC_ROPE_INTERLEAVED, 0, D, hd, 0, D, hd, 0, hd, 0, 0, 0, 0.f}, rk = rq;
    rk.x_off = rk.y_off = (int)(KO + (size_t)L * D);
    VkcEncAttn at = {5, N, H, hd, 0, (int)KO, (int)VO, D, 0, D, 0, 0, 0, 0, 0, 1.f / sqrtf((float)hd), 0};
    int ok = vkc_begin() && vkc_write(g_qic.lat, 0, lat, (size_t)N * C * 4) &&
             vkc_matmul(qic_tensor(&d->img_in), g_qic.lat, 0, g_qic.x, 0, N);
    for (int l = 0; ok && l < d->layers; l++) {
        DitBlock *B = &d->B[l];
        VkcNorm nq = {N * H, hd, H, 0, D, hd, 0, D, hd, l * 2 * hd, 0, 0, d->eps, 1.f}, nk = nq;
        nk.x_off = nk.y_off = (int)(KO + (size_t)L * D); nk.w_off = l * 2 * hd + hd;
        ok = vkc_enc_norm(g_qic.x, NULL, g_qic.mod, g_qic.h, &n1) &&
             vkc_matmul(qic_tensor(&B->q), g_qic.h, 0, g_qic.qkv, 0, N) &&
             vkc_matmul(qic_tensor(&B->k), g_qic.h, 0, g_qic.qkv, KO + (size_t)L * D, N) &&
             vkc_matmul(qic_tensor(&B->v), g_qic.h, 0, g_qic.qkv, VO + (size_t)L * D, N) &&
             vkc_norm(g_qic.qkv, g_qic.prm, g_qic.qkv, &nq) && vkc_norm(g_qic.qkv, g_qic.prm, g_qic.qkv, &nk) &&
             vkc_mla_rope(g_qic.qkv, g_qic.cs, g_qic.qkv, &rq) && vkc_mla_rope(g_qic.qkv, g_qic.cs, g_qic.qkv, &rk) &&
             vkc_copy(g_qic.qkv, KO, g_qic.pk[l], 0, (size_t)L * D) && vkc_copy(g_qic.qkv, VO, g_qic.pv[l], 0, (size_t)L * D) &&
             vkc_enc_attn(g_qic.qkv, NULL, NULL, g_qic.a, g_qic.rng, NULL, NULL, &at) &&
             vkc_matmul(qic_tensor(&B->o), g_qic.a, 0, g_qic.h, 0, N) &&
             vkc_ew(g_qic.x, g_qic.x, g_qic.h, NULL, g_qic.mod, &g1) &&
             vkc_enc_norm(g_qic.x, NULL, g_qic.mod, g_qic.h, &n2) &&
             vkc_matmul(qic_tensor(&B->gate), g_qic.h, 0, g_qic.g, 0, N) &&
             vkc_matmul(qic_tensor(&B->proj), g_qic.h, 0, g_qic.u, 0, N) &&
             vkc_ew(g_qic.g, g_qic.g, g_qic.u, NULL, NULL, &sw) &&
             vkc_matmul(qic_tensor(&B->out), g_qic.g, 0, g_qic.h, 0, N) &&
             vkc_ew(g_qic.x, g_qic.x, g_qic.h, NULL, g_qic.mod, &g2) &&
             vkc_submit(0) && vkc_begin();   /* a submission a block: no single one runs for seconds */
    }
    ok = ok && vkc_enc_norm(g_qic.x, NULL, g_qic.mod, g_qic.h, &no) &&
         vkc_matmul(qic_tensor(&d->proj_out), g_qic.h, 0, g_qic.pred, 0, N) &&
         vkc_copy(g_qic.predd, 0, g_qic.pred, 0, (size_t)N * C) && vkc_submit(1);
    if (!ok) {
        if (!vkc_lost()) { vkc_finish(); coli_vk_mark_lost_dev(0); }
        qic_off("a frame failed (the device was lost); this step runs again on the CPU");
        return 0;
    }
    memcpy(out, vkc_ptr(g_qic.predd), (size_t)N * C * 4);
    g_qic.steps++;
    if (g_prof) g_t_lin += now_s() - t0;   /* the whole step: the device does not split its time */
    return 1;
}

static void qic_report(void) {
    if (!g_qic.ok || !g_qic.steps) return;
    VkcStats st; vkc_stats(&st);
    fprintf(stderr, "[VK] qwenimage chain: %llu steps, %llu frames (%llu ops, %llu matmuls, %llu tiled GEMM), "
                    "%.1f ms waiting for the device, %.1f MiB on the device\n",
            g_qic.steps, st.frames, st.ops, st.matmuls, st.gemms, st.wait_ms, st.dev_bytes / 1048576.0);
    vkc_prof_print();
}
