#ifndef COLIBRI_BACKEND_VULKAN_H
#define COLIBRI_BACKEND_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque persistent device copy of one resident quantized tensor,
 * mirroring backend_cuda.h. On Strix Halo the "upload" writes into
 * HOST_VISIBLE|DEVICE_LOCAL memory — same physical RAM the iGPU reads,
 * so there is no PCIe copy, unlike the discrete-CUDA path. */
typedef struct ColiVkTensor ColiVkTensor;

/* Bring up instance/device/queue/pipeline. Returns 1 on success.
 * spv_path points at the compiled qmatmul.spv. */
int  coli_vk_init(const char *spv_path);
void coli_vk_shutdown(void);
int  coli_vk_available(void);
void coli_vk_mem_info(size_t *used_bytes, size_t *tensor_count);
/* GLM-5.3 SwiGLU clamp for the fused gate_up kernel. 0 disables (GLM-5.2). */
void coli_vk_set_swiglu_limit(float limit);

/* VRAM pressure-proofing (both no-ops when the extension is absent):
 * alloc_priority sets the eviction-priority class of SUBSEQUENT weight uploads
 * (VK_EXT_memory_priority; scratches and the KV mirror pin themselves at 1.0) —
 * the engine brackets the bulk expert-tier fill at 0.4 so an oversubscribed heap
 * evicts cold experts, never the per-token attention working set.
 * mem_budget reports device-local usage/budget in GB (VK_EXT_memory_budget);
 * returns 0 if unavailable. */
void coli_vk_alloc_priority(float p);
int  coli_vk_mem_budget(double *used_gb, double *budget_gb);

/* y[S,O] = (x[S,I] @ dequant(W[O,I])^T) * scale[O].
 * fmt matches QT in glm.c: 1=int8, 2=int4. (0=f32,3=int2 fall back to CPU.)
 * fmt 10 = plain f32 weights and 11 = bf16 weights (low half = even column): no
 * scales, pass NULL. Numbered apart from QT's 0 (f32), which keeps falling back.
 * First call uploads W+scales; later calls reuse the resident copy.
 * Returns 1 on success, 0 if unavailable / unsupported fmt. */
int  coli_vk_matmul(ColiVkTensor **tensor,
                    float *y, const float *x,
                    const void *weights, const float *scales,
                    int fmt, int S, int I, int O, int gs);

/* Fused first half of the expert MLP in ONE dispatch (VK equivalent of
 * grouped_hidden_w4_dual): hidden[s,o] = silu(gate(x)) * up(x), reading x once for both
 * projections. D = input (hidden) dim, I = moe_inter. gate/up upload on first call.
 * Returns 0 if unavailable (no gate_up shader) / unsupported fmt so the caller falls back. */
int  coli_vk_gate_up(ColiVkTensor **gate, ColiVkTensor **up,
                     float *hidden, const float *x,
                     const void *gw, const float *gs,
                     const void *uw, const float *us,
                     int fmt, int S, int D, int I, int grp);

/* Full batched expert MLP for `count` experts in ONE submit, hidden staying on-device:
 * for each c, y_c = down_c(silu(gate_c(x_c)) * up_c(x_c)). x/y packed [sum(rows)*D];
 * experts are resident (gate/up: D->I, down: I->D). Mirrors coli_cuda_expert_group.
 * Returns 0 -> caller falls back to CPU. */
int  coli_vk_expert_group(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                          ColiVkTensor *const *downs, const int *rows, int count,
                          float *y, const float *x);
/* Async form: _issue submits the group and returns immediately (one in flight max);
 * the caller computes its CPU share, then _take joins and reads back the packed y.
 * Both return 0 on failure (caller computes those experts on the CPU instead). */
int  coli_vk_expert_group_issue(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                ColiVkTensor *const *downs, const int *rows, int count,
                                const float *x);
int  coli_vk_expert_group_take(float *y);

/* Upload a resident tensor without computing (expert tier: gate/up/down uploaded once,
 * then driven by coli_vk_expert_group). Returns 0 on failure/unsupported fmt. */
int  coli_vk_tensor_ensure(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);

/* SECOND DEVICE (COLI_VK_DEV2): a self-contained context on another Vulkan GPU that
 * hosts ONLY tier experts and runs ONLY the async expert-group path. devidx: -1 =
 * auto (best real GPU that is not device 0), >=0 = enumeration index (the same
 * physical device is allowed with a warning — pre-hardware test mode). Its group
 * may be in flight simultaneously with device 0's. Tensors remember their device
 * (coli_vk_tensor_dev); free/bytes work on either. */
int  coli_vk_init_dev2(const char *spv_path, int devidx);
int  coli_vk_dev2_available(void);
int  coli_vk_tensor_dev(const ColiVkTensor *t);
int  coli_vk_mem_budget2(double *used_gb, double *budget_gb);
int  coli_vk_tensor_ensure2(ColiVkTensor **tensor, const void *weights, const float *scales, int fmt, int I, int O, int grp);
int  coli_vk_expert_group_issue2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                                 ColiVkTensor *const *downs, const int *rows, int count,
                                 const float *x);
int  coli_vk_expert_group_take2(float *y);
int  coli_vk_expert_group2(ColiVkTensor *const *gates, ColiVkTensor *const *ups,
                           ColiVkTensor *const *downs, const int *rows, int count,
                           float *y, const float *x);

/* MLA absorb attention core (decode). The KV latent/rope caches live in persistent
 * per-layer device buffers: _ensure allocates a layer's cache at max_rows (once; resize
 * via _reset), _row mirrors one host row (absolute position), _reset drops all layers.
 * The caller keeps a valid-watermark and re-mirrors rows after any invalidation.
 * absorb runs S causal query rows over cache rows [st0, T) in one submit:
 * q [S,H*(Q+R)] roped, kv_b [H*(Q+V), K] uploads once (fmt 1=int8/2=int4),
 * ctx out [S,H*V]. Returns 0 -> caller falls back to CPU. */
int  coli_vk_kv_ensure(int layer, int max_rows, int K, int Rd);
int  coli_vk_kv_row(int layer, int pos, const float *L, const float *R);
void coli_vk_kv_reset(void);
int  coli_vk_attention_absorb(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              float *ctx, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale);
/* Two resident matmuls sharing one input x in ONE submit (q_a + kv_a prologue pair).
 * Returns 0 -> caller falls back to single-matmul calls. */
/* q-prep chain: [q_a+kv_a pair] -> rmsnorm(q latent) -> q_b in ONE submit (needs
 * rmsnorm.spv next to the main shader; returns 0 without it -> 3-submit path).
 * lnw = the q-latent RMS-norm weights [Oqa], resident per layer after first call. */
int  coli_vk_attn_qprep(int layer,
                        ColiVkTensor **qa,  const void *wqa,  const float *sqa,  int Oqa,
                        ColiVkTensor **kva, const void *wkva, const float *skva, int Okva,
                        ColiVkTensor **qb,  const void *wqb,  const float *sqb,  int Oqb,
                        int fmt, int grp, const float *lnw, float eps,
                        const float *x, int S, int I, float *q_out, float *kv_out,
                        float *lat_out /* normed q latent [S,Oqa], NULLable — DSA indexer input */);
int  coli_vk_matmul_pair(ColiVkTensor **t1p, float *y1, const void *w1, const float *s1, int O1,
                         ColiVkTensor **t2p, float *y2, const void *w2, const float *s2, int O2,
                         int fmt, const float *x, int S, int I, int grp);

/* Fused variant: absorb + resident o-projection ([Dout, H*V]) in one submit; ctx stays
 * on-device, only out [S,Dout] is read back. Falls back like absorb (returns 0). */
int  coli_vk_attention_absorb_project(ColiVkTensor **kvb, const void *w, const float *sc, int fmt, int grp,
                              ColiVkTensor **ot, const void *ow, const float *osc, int ofmt, int ogrp,
                              float *out, const float *q, int layer, int S, int H,
                              int Q, int R, int V, int K, int st0, int T, float scale, int Dout);

/* Frees the tensor and gives its device memory back to its pool (the next upload
 * reuses it). Safe from any thread; a free while async work is in flight on the
 * tensor's device takes effect when that work has been joined. */
void   coli_vk_tensor_free(ColiVkTensor *t);
size_t coli_vk_tensor_bytes(const ColiVkTensor *t);

/* ---- weight memory -----------------------------------------------------------
 * Resident tensors live in a few big device-memory blocks per pool, handed out by
 * an offset allocator (vk_alloc.h) that takes freed ranges back. Pool 0 holds every
 * engine's resident weights (what coli_vk_mem_info counts), pool 1 the routed-expert
 * tier's experts, pool 2 COLI_VK_DEV2's. */
typedef struct {
    int blocks, live;                 /* device-memory blocks; ranges handed out */
    size_t total, used, free;         /* block bytes; in ranges; between them */
    size_t largest_free, peak_used;
    size_t limit;                     /* the pool's byte limit (0 = none) */
    size_t tensors, payload;          /* live tensors and their row + scale bytes */
    unsigned long long allocs, frees, refusals;   /* refusals: a block over the limit */
    double frag;                      /* 1 - largest free extent / free bytes */
} ColiVkPoolStats;
void coli_vk_pool_stats(int pool, ColiVkPoolStats *st);
/* The expert tier's budget: its pool never holds more block bytes than this. */
void coli_vk_tier_pool_limit(size_t bytes);
/* A tensor in the tier's pool, to fill in place: O rows of coli_vk_tensor_row_bytes
 * at *stride apart (padding zeroed) and coli_vk_tensor_scale_count floats of scales
 * (fmt 10/11: one, set it to 1). Thread-safe. Returns 0 at the budget or when the
 * device is out of memory. */
int    coli_vk_tier_tensor(ColiVkTensor **t, int fmt, int I, int O, int gs,
                           uint8_t **rows, size_t *stride, float **scales);
size_t coli_vk_tensor_row_bytes(int fmt, int I);
size_t coli_vk_tensor_scale_count(int fmt, int I, int O, int gs);

/* ---- async expert batch (the routed-expert tier) ------------------------------
 * One layer step of resident experts in one submit on the tier queue (a second
 * queue when the device has one, so the synchronous dense matmuls do not wait
 * behind it). Every expert: hidden = act(gate(x), up(x)), y = down(hidden), for
 * each of its rows. act: COLI_VK_ACT_SWIGLU (silu(g)*u, limit > 0 clamps the gate
 * from above and up to [-limit, limit]) or COLI_VK_ACT_SITU (a*tanh(g/a)*sigmoid(g) *
 * b*tanh(u/b)). One geometry per process: hidden D, intermediate I.
 * Threading: engine thread only, except where noted. */
#define COLI_VK_ACT_SWIGLU 0
#define COLI_VK_ACT_SITU   1
typedef struct ColiVkExpert ColiVkExpert;
int  coli_vk_xb_init(int D, int I, int act, float limit, float a, float b);   /* again: same D, I, new act */
int  coli_vk_xb_ready(void);
int  coli_vk_xb_queue_shared(void);   /* 1 = the batch shares the main queue */
/* A resident expert from three tensors (gate/up [I x D] in one format, down [D x I]
 * in any): writes its descriptor sets once. NULL when the shapes do not match the
 * geometry. Not while a batch is in flight. */
ColiVkExpert *coli_vk_xb_expert(ColiVkTensor *gate, ColiVkTensor *up, ColiVkTensor *down);
/* Its sets and its three tensors. Not while a batch is in flight. */
void coli_vk_xb_expert_free(ColiVkExpert *e);
/* Submit and return: count experts, rows[c] activation rows each, the rows given as
 * sum(rows) pointers to D floats, expert by expert. 0 = nothing was submitted (the
 * caller computes those experts itself). One batch in flight at a time. */
int  coli_vk_xb_issue(ColiVkExpert *const *ex, const int *rows, int count, const float *const *xrows);
/* Wait for it: yrows[j] points at the D outputs of input row j (same order), valid
 * until the next issue; *device_ms is its device time when timestamps exist (else 0).
 * 0 = the batch failed (device lost): the caller computes those rows itself. */
int  coli_vk_xb_join(const float **yrows, double *device_ms);
typedef struct {
    unsigned long long batches, experts, rows, gemm_experts;
    double device_ms;                 /* summed batch device time (timestamps) */
    int timestamps, queue_shared, gemm_rows;
    size_t scratch_bytes;
} ColiVkXbStats;
void coli_vk_xb_stats(ColiVkXbStats *st);

/* 1 if the selected device is an integrated GPU (shares physical memory with
 * the host), 0 otherwise or when no device is selected. */
int coli_vk_device_integrated(void);
/* 1 for an integrated GPU or a CPU device (Lavapipe): device memory is host RAM. */
int coli_vk_device_shares_ram(void);
/* The largest DEVICE_LOCAL heap in bytes (for a budget without VK_EXT_memory_budget). */
size_t coli_vk_device_local_bytes(void);
const char *coli_vk_device_name(void);

/* For engines: COLI_VULKAN=1 opens the device with the shaders found by
 * coli_vk_shader_path() and prints one line naming the engine; 0 (and one line) when
 * Vulkan is not asked for or no device is usable, so the engine stays on the CPU. */
int coli_vk_init_env(const char *engine);
/* COLI_VK_SHADERS (the .spv or its directory), else shaders/ next to the binary, else
 * shaders/ in the working directory. buf holds the result when it is not a literal. */
const char *coli_vk_shader_path(char *buf, size_t n);
/* How many coli_vk_matmul calls ran on the device: a check that a path is really used. */
unsigned long long coli_vk_matmul_calls(void);

#ifdef __cplusplus
}
#endif

#endif
