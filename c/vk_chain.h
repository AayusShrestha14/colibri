/* vk_chain.h -- a layer's dense chain on the Vulkan device, recorded into one submission.
 *
 * Why: the engines' dense part on Vulkan was one synchronous coli_vk_matmul per matrix
 * (about 726 submits and host round trips per Qwen3.8 decode token), slower on an
 * integrated GPU than the CPU. Here an engine records a whole layer -- norms,
 * projections, RoPE, attention over a KV cache that lives on the device, the Gated
 * DeltaNet recurrence with its state on the device, gates, the router logits, the
 * shared expert, the residual add -- into one command buffer, and the residual stream
 * stays on the device from one layer to the next. Only what the CPU needs crosses:
 * the rows the CPU's routed experts read and the router logits, then the routed sum
 * coming back.
 *
 * The pieces:
 *   - buffers (VkcBuf): VKC_DEV the device's own (state, scratch), VKC_UP written by the
 *     host and read by the device, VKC_DOWN written by the device and read by the host.
 *     They are sub-allocated from a few large memory blocks per kind (vk_alloc.h), so a
 *     submit references a handful of allocations, not hundreds.
 *   - recording: vkc_begin opens a command buffer (a ring of frames, each with its own
 *     fence and descriptor pool); every op records the barrier it needs (a buffer read or
 *     written after an earlier write, or written after a read, since the last barrier);
 *     vkc_submit(wait) sends it. Submissions run in order on the backend's main queue, a
 *     frame's first barrier orders it after everything submitted before it.
 *   - ops: vkc_matmul over the resident tensors the engines already upload
 *     (coli_vk_tensor_ensure), the GEMV per row or, from the backend's threshold, the
 *     fp32 tiled GEMM; and the chain's shaders (shaders/chain_*.comp), each documented
 *     at its top: chain_norm, chain_rope, chain_attn, chain_dnconv, chain_dnrec,
 *     chain_ew, chain_qsa, chain_ple. Offsets and strides are in floats.
 *
 * Threading: the engine thread only (the main queue is the backend's, used from the
 * same thread by coli_vk_matmul; the expert tier submits on its own queue).
 * A failed fence wait marks the device lost (coli_vk_mark_lost): vkc_lost() says so,
 * and every later call returns 0.
 *
 * tests/test_vk_chain.c checks every op against a CPU reference (make vk-chain-check). */
#ifndef COLI_VK_CHAIN_H
#define COLI_VK_CHAIN_H
#include <stddef.h>
#include <stdint.h>
#include "backend_vulkan.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VkcBuf VkcBuf;
#define VKC_DEV  0
#define VKC_UP   1
#define VKC_DOWN 2

int  vkc_init(void);          /* after coli_vk_init; 1 = the chain's pipelines are up */
int  vkc_ready(void);
int  vkc_lost(void);
void vkc_shutdown(void);      /* before coli_vk_shutdown (register it with atexit after it) */

VkcBuf *vkc_buf(size_t bytes, int kind);                 /* zero-filled; NULL when out of memory */
void    vkc_free(VkcBuf *b);                             /* waits for the frames that may read it */
int     vkc_reserve(VkcBuf **b, size_t bytes, int kind); /* at least `bytes`; growing drops the contents */
void   *vkc_ptr(const VkcBuf *b);                        /* host mapping (VKC_UP, VKC_DOWN; VKC_DEV when host-visible) */
size_t  vkc_bytes(const VkcBuf *b);

/* recording */
int  vkc_begin(void);
int  vkc_submit(int wait);
int  vkc_finish(void);        /* wait for every submitted frame */

/* transfers, recorded into the open frame (offsets and counts in floats) */
int  vkc_copy(VkcBuf *dst, size_t doff, VkcBuf *src, size_t soff, size_t n);
int  vkc_zero(VkcBuf *dst, size_t off, size_t n);
/* one copy command over n regions (a KV row per head, say) */
typedef struct { size_t dst, src, n; } VkcRegion;
int  vkc_copy_regions(VkcBuf *dst, VkcBuf *src, const VkcRegion *r, int n);
int  vkc_write(VkcBuf *dst, size_t off, const void *src, size_t bytes);   /* through the frame's staging */
/* synchronous: finishes what is in flight, copies, waits (the open frame, if any, is submitted first) */
int  vkc_read(VkcBuf *src, size_t off, void *dst, size_t bytes);

/* y[S][O] = x[S][I] @ W^T for a resident tensor (its fmt, I, O); x and y at float offsets */
int  vkc_matmul(ColiVkTensor *t, VkcBuf *x, size_t xo, VkcBuf *y, size_t yo, int S);
/* Rows from which vkc_matmul takes the tiled GEMM: -1 = the backend's rule (S >= 2 and
 * S*O >= 4096), 0 = never (an MTP verify, whose rows must get a decode step's bits). */
void vkc_gemm_rows(int rows);

/* chain_norm.comp */
typedef struct { int nseg, D, per_row, x_off, x_row, x_seg, y_off, y_row, y_seg, w_off, w_mod, flags; float eps, post; } VkcNorm;
#define VKC_NORM_ADD1 1
#define VKC_NORM_NOW  2
#define VKC_NORM_L2   4
int  vkc_norm(VkcBuf *x, VkcBuf *w, VkcBuf *y, const VkcNorm *p);
/* chain_rope.comp */
typedef struct { int nseg, per_row, x_off, x_row, x_seg, half_, cs_off, cs_row; } VkcRope;
int  vkc_rope(VkcBuf *x, VkcBuf *cs, const VkcRope *p);
/* chain_attn.comp (k_off/v_off: where the layer's cache starts in kc/vc) */
typedef struct { int S, H, KVH, hd, pos_base, cap, q_off, q_row, q_seg, g_off, g_row, g_seg, has_gate,
                 o_off, o_row, sel_off, sel_row; float scale; int k_off, v_off; } VkcAttn;
int  vkc_attn(VkcBuf *q, VkcBuf *kc, VkcBuf *vc, VkcBuf *o, VkcBuf *gate, VkcBuf *sel, const VkcAttn *p);
/* chain_dnconv.comp */
typedef struct { int S, CD, CK, in_off, in_row, out_off, out_row, snap_row, order, w_off, ring_off, snap_off; } VkcDnConv;
int  vkc_dnconv(VkcBuf *in, VkcBuf *w, VkcBuf *ring, VkcBuf *out, VkcBuf *snap, const VkcDnConv *p);
/* chain_dnrec.comp (KD a specialization constant, VD <= 128) */
typedef struct { int S, VH, KH, VD, Ktot, cv_off, cv_row, b_off, b_row, a_off, a_row, z_off, z_row,
                 y_off, y_row, snap_row, flags; float eps, qscale; int st_off, snap_off, prm_off; } VkcDnRec;
int  vkc_dnrec(int KD, VkcBuf *cv, VkcBuf *ab, VkcBuf *z, VkcBuf *st, VkcBuf *prm, VkcBuf *y, VkcBuf *snap, const VkcDnRec *p);
/* chain_ew.comp */
#define VKC_EW_ADD      0
#define VKC_EW_COMBINE  1
#define VKC_EW_SWIGLU   2
#define VKC_EW_HC_LOW   3
#define VKC_EW_HC_MIX   4
#define VKC_EW_HC_INJ   5
#define VKC_EW_HC_APPLY 6
typedef struct { int op, n, D, C, flags, e_row, y_off, a_off, b_off, c_off, e_off; float fc; } VkcEw;
int  vkc_ew(VkcBuf *y, VkcBuf *a, VkcBuf *b, VkcBuf *c, VkcBuf *e, const VkcEw *p);
/* chain_qsa.comp (mode 0: nb block keys from b0; mode 1: S rows' selections) */
typedef struct { int mode, ID, R, b0, half_, S, pos_base, budget, IQ, q_off, q_row, nbmax, sel_row; float eps;
                 int src_off, w_off, pk_off, nb; } VkcQsa;
int  vkc_qsa(VkcBuf *src, VkcBuf *w, VkcBuf *pk, VkcBuf *cs, VkcBuf *sc, VkcBuf *sel, const VkcQsa *p);
/* chain_ple.comp (mode 0: the gate over S*C (row, stream) pairs; mode 1: the convolution) */
typedef struct { int mode, S, C, H, CK, NG, keys_off, hyp_off, val_off, snap_row, snap_off; float eps;
                 int prm_off, conv_off, ring_off; } VkcPle;
int  vkc_ple(VkcBuf *keys, VkcBuf *hyp, VkcBuf *val, VkcBuf *prm, VkcBuf *gated, VkcBuf *normv,
             VkcBuf *conv, VkcBuf *ring, const VkcPle *p);

/* counters, for the engines' [VK] lines */
typedef struct {
    unsigned long long frames, waits, ops, matmuls, gemms, barriers, bytes_up, bytes_down;
    double wait_ms;               /* host time blocked in fence waits */
    size_t dev_bytes;             /* live chain buffers */
} VkcStats;
void vkc_stats(VkcStats *st);
/* COLI_VK_CHAIN_PROF=1: one stderr line of device time per kind of op */
void vkc_prof_print(void);

#ifdef __cplusplus
}
#endif
#endif
