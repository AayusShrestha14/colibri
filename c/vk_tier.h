/* vk_tier.h -- the Vulkan routed-expert tier shared by the MoE engines.
 *
 * What it does, for a GPU a Vulkan driver can see (COLI_VULKAN=1):
 *   - holds a cache of routed experts in device memory, under a budget taken from
 *     VK_EXT_memory_budget (minus a reserve and the dense trunk the engine will
 *     place there), capped by COLI_VK_TIER_GB;
 *   - fills it at startup from the expert history (.coli_usage, route_trace.h) and
 *     keeps adapting while the user chats: an expert the CPU computes is a candidate,
 *     admitted when there is room or when it is hotter than the coldest resident by
 *     tier.h's LFRU margin, which is then evicted; uploads run on a thread of their
 *     own and never hold up a layer;
 *   - computes, per layer step, every resident expert the routing picked as ONE
 *     asynchronous submit (backend_vulkan.c's expert batch) while the engine computes
 *     the rest on the CPU, then hands back one output row per (row, rank) so the
 *     engine adds them in its own routing order;
 *   - on an integrated GPU, whose device memory is the CPU's RAM, defaults to a
 *     budget that leaves the RAM the engine's expert cache will need;
 *   - accounts device / CPU hits, uploads, evictions and how much device time the
 *     CPU work hid, in one "[VK] tier <engine>" line per run and serve turn.
 * With COLI_VULKAN unset, COLI_VK_TIER=0, a CUDA tier active, or a build without
 * VK=1 (the inline stubs below), nothing here runs and the engine is unchanged.
 *
 * ---- integrating an engine (the phase-2 guide) -------------------------------
 * 0. Open the device saying whether the tier will be tried, so the dense matrices
 *    get their default place (on a device that shares the CPU's RAM they stay on
 *    the CPU while the tier is on; COLI_VK_DENSE decides when set):
 *
 *        ready = coli_vk_init_env_tier("glm53", vkt_wanted() && E > 0 && !cuda_tier);
 *        dense = coli_vk_dense();          // the engine's dense hook checks this
 *        ...                               // step 1
 *        if (ready && !vkt_ready() && !dense)
 *            dense = coli_vk_dense_decide("glm53", 0, 1);   // no tier after all
 *
 * 1. Describe the experts once, after the model and the dense trunk are loaded and
 *    after the engine's history is read (rt_load), before the first token:
 *
 *        VktConfig c = {.engine = "glm53", .layers = L, .experts = E,
 *                       .hidden = H, .inter = F, .topk = K,
 *                       .gate_up = {VKT_SRC_I4U_PAIRS_GS, 64},   // how RAM holds them
 *                       .down    = {VKT_SRC_I4U_PAIRS_GS, 64},
 *                       .act = VKT_ACT_SWIGLU, .act_limit = swiglu_limit,
 *                       .max_rows = prefill_rows * K,
 *                       .ram_reserve = bytes the RAM expert cache may still grow by,
 *                       .dense_bytes = bytes of dense weights the engine will put
 *                                      on the device after this call};
 *        atexit(coli_vk_shutdown);       // runs last: the device goes before the
 *        if (vkt_init(&c, rt_counts_all()))   // drivers unload (a driver can still
 *            atexit(vkt_shutdown);       // be compiling the tier's pipelines), and
 *                                        // whether or not the tier starts: vkt_init
 *                                        // makes the expert batch's pipelines
 *                                        // before it can refuse (no room)
 *
 *    .in_ram (optional) lets the tier hand experts the CPU holds in RAM back to the
 *    CPU when the device is the slower side of a step (an integrated GPU at its
 *    floor clock): give it whenever the engine can answer cheaply.
 *
 *    The heat table is [layers][experts] counts (route_trace.h's rows; NULL rows are
 *    layers without experts). Pick .gate_up/.down from the table below; gate and up
 *    must share one kind. A layer the tier should not serve (an MTP layer in another
 *    format) is simply never passed to vkt_issue: index layers 0..layers-1 only.
 *
 * 2. Warm start (optional, recommended when there is history): the tier plans, the
 *    engine reads each planned expert from disk (any number of threads) and hands
 *    it over, then publishes on the engine thread:
 *
 *        int n = vkt_plan(pl, pe, max);
 *        #pragma omp parallel for
 *        for (i < n) { read expert (pl[i], pe[i]) into a private buffer;
 *                      vkt_put(pl[i], pe[i], &src); }
 *        vkt_put_done();
 *
 * 3. Every MoE layer step, rows x[S][hidden] routed to idx[S][K] (rank order, -1 =
 *    unused slot):
 *
 *        uint8_t taken[S*K];  const float *dev[S*K];
 *        int n = vkt_issue(layer, x, S, K, idx, taken);   // returns at once
 *        for every (s,k) with !taken: compute on the CPU into its own buffer,
 *            and when the expert's bytes are in RAM: vkt_note(layer, e, &src);
 *        shared experts, other CPU work ...
 *        if (n && !vkt_join(dev)) compute the taken (s,k) on the CPU too (device lost);
 *        for s: for k in rank order: out[s] += w[s][k] * (taken ? dev[s*K+k] : cpu[s][k]);
 *
 *    vkt_issue returns how many (s,k) the device took; join only when it took any.
 *
 *    Add in rank order, device or not: then the sum's order never depends on which
 *    experts happened to be resident (it is the CPU run's order, the values differ
 *    by the device's summation order only). Never call the dense Vulkan matmuls
 *    between vkt_issue and vkt_join from a parallel region (backend rule), and
 *    keep one step in flight: join before the next layer's issue.
 *
 * 4. Report: vkt_report("run" or "turn", ram_hits, disk_loads) where the engine
 *    prints its "[VK] <engine>" line; EMAP's tier bits: vkt_resident(l, e) -> 2.
 *
 * Source kinds (VktSrc) and what the device holds -- the conversion runs on the
 * uploader thread, nothing changes in the engine's RAM:
 *   VKT_SRC_I8_ROW        int8, f32 per row                      -> fmt 1
 *   VKT_SRC_I8_GS         int8, f32 per (row, gs)                -> fmt 13
 *   VKT_SRC_I8_AS_I4_ROW  int8 holding int4 values [-8,7], per row -> fmt 2 (repacked: half the bytes)
 *   VKT_SRC_I8_AS_I4_GS   same, f32 per (row, gs)                -> fmt 4
 *   VKT_SRC_I4S_PAIRS_ROW signed int4, byte j = elements 2j (low), 2j+1, per row -> fmt 2
 *   VKT_SRC_I4S_PAIRS_GS  same, per (row, gs)                    -> fmt 4
 *   VKT_SRC_I4U_PAIRS_ROW int4 as v+8, pairs, per row            -> fmt 2
 *   VKT_SRC_I4U_PAIRS_GS  int4 as v+8, pairs, per (row, gs)      -> fmt 4 (GLM, glm53)
 *   VKT_SRC_I4U_PLANAR64  expert_ffn.h planar blocks of 64, v+8, f32 per 64 -> fmt 4 gs 64
 *   VKT_SRC_I3_G64        int3-g64 planes                        -> fmt 5
 *   VKT_SRC_MXFP4_F32     e2m1 pairs, f32 per gs                 -> fmt 7
 *   VKT_SRC_MXFP4_E8M0    e2m1 pairs, ue8m0 byte per gs          -> fmt 7 (kimi_k3, mimo, deepseek_v41)
 *   VKT_SRC_FP8_GS        e4m3, f32 per (row, gs)                -> fmt 12
 *   VKT_SRC_FP8_BLOCK     e4m3, f32 per gs x gs block            -> fmt 12 (qwen38 native)
 *   VKT_SRC_BF16, VKT_SRC_F32                                    -> fmt 11, 10
 * Activations (VktConfig.act): VKT_ACT_SWIGLU silu(g)*u, act_limit > 0 clamps the
 * gate from above and up to [-limit, limit] first (GLM-5.3, DeepSeek V4/V4.1);
 * VKT_ACT_SITU a*tanh(g/a)*sigmoid(g) * b*tanh(u/b) with act_a, act_b (Kimi K3);
 * VKT_ACT_SWIGLU_V4 DeepSeek V4's expert with its CPU kernel's roundings: gate and
 * up to bf16, the clamped SwiGLU, times the route weight (vkt_issue_w) and to bf16,
 * then E4M3 and back per 128 inputs before down. The engine hands in x already
 * rounded to E4M3 per 128 (as its kernel rounds it), adds a device row with no
 * weight of its own (the weight is in it) and rounds the row to bf16 first, as
 * its CPU expert rounds its output. */
#ifndef COLI_VK_TIER_H
#define COLI_VK_TIER_H
#include <stdint.h>
#include <stddef.h>

typedef enum {
    VKT_SRC_NONE = 0,
    VKT_SRC_I8_ROW, VKT_SRC_I8_GS, VKT_SRC_I8_AS_I4_ROW, VKT_SRC_I8_AS_I4_GS,
    VKT_SRC_I4S_PAIRS_ROW, VKT_SRC_I4S_PAIRS_GS, VKT_SRC_I4U_PAIRS_ROW, VKT_SRC_I4U_PAIRS_GS,
    VKT_SRC_I4U_PLANAR64, VKT_SRC_I3_G64, VKT_SRC_MXFP4_F32, VKT_SRC_MXFP4_E8M0,
    VKT_SRC_FP8_GS, VKT_SRC_FP8_BLOCK, VKT_SRC_BF16, VKT_SRC_F32
} VktSrc;
typedef struct { VktSrc kind; int gs; } VktFmt;     /* gs: group (block) size, 0 = per row */

#define VKT_ACT_SWIGLU 0
#define VKT_ACT_SITU   1
#define VKT_ACT_SWIGLU_V4 2

typedef struct {
    const char *engine;            /* names the [VK] tier lines */
    int layers, experts, hidden, inter, topk;
    VktFmt gate_up, down;
    int act; float act_limit, act_a, act_b;
    int max_rows;                  /* most (row, expert) assignments one vkt_issue carries */
    size_t ram_reserve;            /* bytes the engine's RAM caches may still grow by */
    size_t dense_bytes;            /* dense weights the engine puts on the device after vkt_init */
    /* Optional: is this expert in the engine's RAM cache right now? With it the tier
     * balances a step: when the device keeps the CPU waiting, resident experts the
     * CPU also holds in RAM go back to the CPU (never ones it would read from disk),
     * and return to the device when it finishes early. Called on the engine thread. */
    int (*in_ram)(void *ctx, int layer, int eid);
    void *ram_ctx;
    /* Optional: at most this many experts resident whatever the budget holds (0 = no
     * cap); for an engine whose users already size its device tier in experts. */
    int max_experts;
} VktConfig;

/* One expert as it sits in RAM: codes and scales of gate, up, down (float scales,
 * or ue8m0 bytes for VKT_SRC_MXFP4_E8M0; NULL for bf16/f32). Read during the call
 * only: the tier copies what it keeps. */
typedef struct {
    const void *g, *u, *d;
    const void *gs, *us, *ds;
} VktExpertSrc;

#ifdef COLI_VULKAN
/* 1 unless COLI_VK_TIER=0: whether an engine with routed experts will try the tier
 * (for coli_vk_init_env_tier, before the engine knows whether vkt_init succeeds). */
int  vkt_wanted(void);
/* 1 = the tier is on. heat: [layers][experts] routing counts, or NULL. */
int  vkt_init(const VktConfig *cfg, uint32_t *const *heat);
int  vkt_ready(void);
void vkt_shutdown(void);
int  vkt_plan(int *layers, int *eids, int max);
int  vkt_put(int layer, int eid, const VktExpertSrc *src);   /* any thread */
void vkt_put_done(void);
void vkt_note(int layer, int eid, const VktExpertSrc *src);
int  vkt_issue(int layer, const float *x, int S, int K, const int *idx, uint8_t *taken);
/* vkt_issue with the route weights w[S*K] (beside idx), for VKT_ACT_SWIGLU_V4, which
 * applies them on the device; the other activations ignore them. */
int  vkt_issue_w(int layer, const float *x, int S, int K, const int *idx, const float *w, uint8_t *taken);
/* 1 when vkt_note would take this expert now (not on the device or on its way, the
 * promotion rate not spent, room or a colder resident to displace): for an engine
 * whose RAM form must be converted before vkt_note can read it, so it converts only
 * the experts the tier will take. No side effects. */
int  vkt_wants(int layer, int eid);
int  vkt_join(const float **rows);
int  vkt_resident(int layer, int eid);
/* The next vkt_issue starts a forward (promotion rate, decay, eviction candidates).
 * Optional: the tier tells forwards apart by the layer index going back, which a model
 * with a single MoE layer never does; such an engine calls this at its first MoE layer. */
void vkt_begin_forward(void);
void vkt_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads);
/* Sizing helpers for engines: bytes one expert takes on the device in a source format. */
size_t vkt_expert_bytes(int hidden, int inter, VktFmt gate_up, VktFmt down);
#else
static inline int  vkt_wanted(void){return 0;}
static inline int  vkt_init(const VktConfig *c, uint32_t *const *h){(void)c;(void)h;return 0;}
static inline int  vkt_ready(void){return 0;}
static inline void vkt_shutdown(void){}
static inline int  vkt_plan(int *l,int *e,int m){(void)l;(void)e;(void)m;return 0;}
static inline int  vkt_put(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;return 0;}
static inline void vkt_put_done(void){}
static inline void vkt_note(int l,int e,const VktExpertSrc *s){(void)l;(void)e;(void)s;}
static inline int  vkt_issue(int l,const float *x,int S,int K,const int *i,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)t;return 0;}
static inline int  vkt_issue_w(int l,const float *x,int S,int K,const int *i,const float *w,uint8_t *t){(void)l;(void)x;(void)S;(void)K;(void)i;(void)w;(void)t;return 0;}
static inline int  vkt_wants(int l,int e){(void)l;(void)e;return 0;}
static inline int  vkt_join(const float **r){(void)r;return 0;}
static inline int  vkt_resident(int l,int e){(void)l;(void)e;return 0;}
static inline void vkt_begin_forward(void){}
static inline void vkt_report(const char *s,unsigned long long r,unsigned long long d){(void)s;(void)r;(void)d;}
static inline size_t vkt_expert_bytes(int h,int i,VktFmt a,VktFmt b){(void)h;(void)i;(void)a;(void)b;return 0;}
#endif

#endif
