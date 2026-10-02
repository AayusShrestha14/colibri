/* vk_tier.c -- the Vulkan routed-expert tier shared by the MoE engines. See
 * vk_tier.h for what it does and how an engine calls it.
 *
 * Threads. Everything runs on the engine thread except two things: the uploader
 * thread, which turns staged copies of promoted experts into device tensors, and
 * vkt_put, which a warm start may call from many loader threads. Neither touches
 * a slot: they hand finished tensors back through the `done` list, and the engine
 * thread makes them resident at a quiescent point (no batch in flight), where it
 * also frees evicted experts. So the slot table, the counters and every Vulkan
 * call but the pool's allocation are single-threaded; the lock guards the queue,
 * the done list and the room-made signal only. */
#ifdef COLI_VULKAN
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include "vk_tier.h"
#include "backend_vulkan.h"
#include "tier.h"

#define VKT_QCAP 32                 /* promotions staged and waiting for the uploader */
#define VKT_CAND 64                 /* victim candidates kept per refresh */
#define VKT_DECAY_TOKENS 1024u      /* tier.h's decay period, in tokens */

enum { VS_NONE = 0, VS_QUEUED, VS_RESIDENT, VS_EVICT };

typedef struct {
    ColiVkExpert *ex;
    uint32_t heat, last;
    uint8_t state;
} VSlot;
typedef struct { int layer, eid; uint8_t *buf; } VQ;
typedef struct { int layer, eid, ok; ColiVkTensor *g, *u, *d; } VDone;

static struct {
    int on;
    VktConfig c;
    char engine[32];
    int gu_fmt, gu_gs, dn_fmt, dn_gs;
    size_t gu_codes, gu_scales, d_codes, d_scales, stage_bytes;   /* one expert as the engine holds it */
    size_t exp_bytes, budget;
    int max_resident, resident, queued, rate, uma;
    VSlot *s;
    uint32_t tick, decay_at;          /* tokens seen (rows of the forward's first layer) */
    int last_layer, first_layer, promos, promo_cap;
    /* uploader */
    pthread_t th; int th_on, stop;
    pthread_mutex_t mx; pthread_cond_t cv, cv_room;
    VQ q[VKT_QCAP]; int qh, qn;
    VDone *done; int ndone, cdone;
    unsigned long room_gen;
    /* evicted, freed at the next quiescent point */
    int *evict; int nevict, cevict;
    /* the coldest residents, refreshed once per forward */
    int cand[VKT_CAND]; uint64_t cand_score[VKT_CAND]; int ncand;
    /* the step in flight */
    int inflight, S, K;
    int *map; int cmap;
    int *grp; int *touched;              /* expert -> group of this step, and the groups' experts */
    ColiVkExpert **bex; int *brows; int *bfirst; int cb;
    const float **bx, **by; int *rowsrc; int cbx;
    int max_dev_rows;
    double t_issued;
    /* the balance: the share of a step's resident experts the device takes, moved by
     * what each join waited (the device was the slower side) or did not */
    int balance, can_balance; float share; unsigned long long handed;
    /* accounting */
    unsigned long long routed, served, steps, uploads, upload_bytes, evictions, qfull, rated, refused, failed, warm;
    double dev_ms, cpu_ms, wait_ms;
    unsigned long long rep_routed, rep_served;
} T;

/* "2.31 GiB", "35.0 MiB", "9.5 KiB": the budget lines read on a 24 GB card and a tiny fixture alike */
static const char *human(double b, char *buf, size_t n) {
    if (b >= 1073741824.0) snprintf(buf, n, "%.2f GiB", b / 1073741824.0);
    else if (b >= 1048576.0) snprintf(buf, n, "%.1f MiB", b / 1048576.0);
    else snprintf(buf, n, "%.1f KiB", b / 1024.0);
    return buf;
}
static double vkt_now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static VSlot *slot(int layer, int eid) { return &T.s[(size_t)layer * T.c.experts + eid]; }

/* ---- source formats ---------------------------------------------------------- */
static int src_per_row(VktSrc k) {
    return k == VKT_SRC_I8_ROW || k == VKT_SRC_I8_AS_I4_ROW || k == VKT_SRC_I4S_PAIRS_ROW || k == VKT_SRC_I4U_PAIRS_ROW;
}
static size_t src_row_bytes(VktSrc k, int I) {
    switch (k) {
    case VKT_SRC_I8_ROW: case VKT_SRC_I8_GS: case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: return (size_t)I;
    case VKT_SRC_I3_G64: return ((size_t)I + 63) / 64 * 24;
    case VKT_SRC_BF16: return (size_t)I * 2;
    case VKT_SRC_F32: return (size_t)I * 4;
    default: return ((size_t)I + 1) / 2;                 /* the int4 and fp4 kinds */
    }
}
static size_t src_scale_bytes(VktFmt f, int I, int O) {
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) return 0;
    if (src_per_row(f.kind)) return (size_t)O * 4;
    if (f.kind == VKT_SRC_I4U_PLANAR64 || f.kind == VKT_SRC_I3_G64) return (size_t)O * (((size_t)I + 63) / 64) * 4;
    size_t ng = ((size_t)I + f.gs - 1) / f.gs;
    if (f.kind == VKT_SRC_MXFP4_E8M0) return (size_t)O * ng;
    if (f.kind == VKT_SRC_FP8_BLOCK) return (((size_t)O + f.gs - 1) / f.gs) * ng * 4;
    return (size_t)O * ng * 4;
}
/* The backend format a source kind lands in; 0 when the combination is not one. */
static int dev_fmt(VktFmt f, int *fmt, int *gs) {
    int g = f.gs;
    switch (f.kind) {
    case VKT_SRC_I8_ROW: *fmt = 1; *gs = 0; return 1;
    case VKT_SRC_I8_GS: *fmt = 13; *gs = g; return g >= 4 && g % 4 == 0;
    case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4U_PAIRS_ROW: *fmt = 2; *gs = 0; return 1;
    case VKT_SRC_I8_AS_I4_GS: case VKT_SRC_I4S_PAIRS_GS: case VKT_SRC_I4U_PAIRS_GS: *fmt = 4; *gs = g; return g >= 8 && g % 8 == 0;
    case VKT_SRC_I4U_PLANAR64: *fmt = 4; *gs = 64; return 1;
    case VKT_SRC_I3_G64: *fmt = 5; *gs = 0; return 1;
    case VKT_SRC_MXFP4_F32: case VKT_SRC_MXFP4_E8M0: *fmt = 7; *gs = g; return g >= 8 && g % 8 == 0;
    case VKT_SRC_FP8_GS: case VKT_SRC_FP8_BLOCK: *fmt = 12; *gs = g; return g >= 4 && g % 4 == 0;
    case VKT_SRC_BF16: *fmt = 11; *gs = 0; return 1;
    case VKT_SRC_F32: *fmt = 10; *gs = 0; return 1;
    default: return 0;
    }
}

/* One matrix [O x I] from its RAM form into a device tensor's mapping: rows at
 * `stride` bytes (already zeroed past the row), scales as the backend reads them.
 * Pure byte work; the values are never requantized. */
static void convert(VktFmt f, int I, int O, const uint8_t *codes, const void *scales,
                    uint8_t *rows, size_t stride, float *sc) {
    size_t rb = src_row_bytes(f.kind, I);
    switch (f.kind) {
    case VKT_SRC_I8_AS_I4_ROW: case VKT_SRC_I8_AS_I4_GS:   /* int8 in [-8,7] -> v+8 nibble pairs */
        for (int o = 0; o < O; o++) {
            const int8_t *src = (const int8_t *)codes + (size_t)o * I;
            uint8_t *dst = rows + (size_t)o * stride;
            for (int j = 0; j < I / 2; j++)
                dst[j] = (uint8_t)(((src[2 * j] + 8) & 15) | (((src[2 * j + 1] + 8) & 15) << 4));
            if (I & 1) dst[I / 2] = (uint8_t)((src[I - 1] + 8) & 15);
        }
        break;
    case VKT_SRC_I4S_PAIRS_ROW: case VKT_SRC_I4S_PAIRS_GS:  /* two's complement nibble -> v+8: xor 8 */
        for (int o = 0; o < O; o++) {
            const uint8_t *src = codes + (size_t)o * rb;
            uint8_t *dst = rows + (size_t)o * stride;
            for (size_t j = 0; j < rb; j++) dst[j] = src[j] ^ 0x88u;
        }
        break;
    case VKT_SRC_I4U_PLANAR64:   /* planar blocks (byte k = elements k, k+32) -> pairs (2j, 2j+1) */
        for (int o = 0; o < O; o++) {
            const uint8_t *src = codes + (size_t)o * rb;
            uint8_t *dst = rows + (size_t)o * stride;
            int nb = I / 64;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = src + (size_t)b * 32;
                for (int p = 0; p < 32; p++) {
                    int e0 = 2 * p, e1 = 2 * p + 1;
                    unsigned a = e0 < 32 ? (blk[e0] & 15u) : (unsigned)(blk[e0 - 32] >> 4);
                    unsigned c = e1 < 32 ? (blk[e1] & 15u) : (unsigned)(blk[e1 - 32] >> 4);
                    dst[(size_t)b * 32 + p] = (uint8_t)(a | (c << 4));
                }
            }
            memcpy(dst + (size_t)nb * 32, src + (size_t)nb * 32, rb - (size_t)nb * 32);   /* tail: pairs already */
        }
        break;
    default:   /* the same bytes */
        for (int o = 0; o < O; o++) memcpy(rows + (size_t)o * stride, codes + (size_t)o * rb, rb);
        break;
    }
    if (f.kind == VKT_SRC_BF16 || f.kind == VKT_SRC_F32) { sc[0] = 1.0f; return; }
    if (f.kind == VKT_SRC_MXFP4_E8M0) {        /* ue8m0 -> f32 2^(s-127), quant.h's mx4_scale */
        size_t n = (size_t)O * (((size_t)I + f.gs - 1) / f.gs);
        const uint8_t *e = scales;
        for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)e[i] << 23; memcpy(&sc[i], &u, 4); }
        return;
    }
    if (f.kind == VKT_SRC_FP8_BLOCK) {         /* [O/bs][I/bs] blocks -> one row each */
        size_t nb = ((size_t)I + f.gs - 1) / f.gs;
        const float *b = scales;
        for (int o = 0; o < O; o++) memcpy(sc + (size_t)o * nb, b + (size_t)(o / f.gs) * nb, nb * 4);
        return;
    }
    memcpy(sc, scales, src_scale_bytes(f, I, O));
}

/* Upload one expert from its RAM form (any thread). 0 when the pool refused. */
static int upload(const uint8_t *g, const uint8_t *u, const uint8_t *d,
                  const void *gs, const void *us, const void *ds, ColiVkTensor *t[3]) {
    const int H = T.c.hidden, F = T.c.inter;
    const uint8_t *codes[3] = {g, u, d}; const void *sc[3] = {gs, us, ds};
    t[0] = t[1] = t[2] = NULL;
    for (int k = 0; k < 3; k++) {
        VktFmt f = k < 2 ? T.c.gate_up : T.c.down;
        int fmt = k < 2 ? T.gu_fmt : T.dn_fmt, fgs = k < 2 ? T.gu_gs : T.dn_gs;
        int I = k < 2 ? H : F, O = k < 2 ? F : H;
        uint8_t *rows; size_t stride; float *scales;
        if (!coli_vk_tier_tensor(&t[k], fmt, I, O, fgs, &rows, &stride, &scales)) {
            for (int j = 0; j < k; j++) { coli_vk_tensor_free(t[j]); t[j] = NULL; }
            return 0;
        }
        convert(f, I, O, codes[k], sc[k], rows, stride, scales);
    }
    return 1;
}

size_t vkt_expert_bytes(int H, int F, VktFmt gu, VktFmt dn) {
    int gf, gg, df, dg;
    if (!dev_fmt(gu, &gf, &gg) || !dev_fmt(dn, &df, &dg)) return 0;
    size_t b = 0;
    for (int k = 0; k < 3; k++) {
        int fmt = k < 2 ? gf : df, gs = k < 2 ? gg : dg, I = k < 2 ? H : F, O = k < 2 ? F : H;
        size_t rows = (coli_vk_tensor_row_bytes(fmt, I) + 3) / 4 * 4 * (size_t)O;
        size_t sc = coli_vk_tensor_scale_count(fmt, I, O, gs) * 4;
        size_t a = coli_vk_buffer_alignment();   /* two ranges, each aligned */
        b += (rows + a - 1) / a * a + (sc + a - 1) / a * a;
    }
    return b;
}

/* ---- the uploader -------------------------------------------------------------- */
static void push_done(int layer, int eid, int ok, ColiVkTensor *t[3]) {
    pthread_mutex_lock(&T.mx);
    if (T.ndone == T.cdone) {
        int nc = T.cdone ? 2 * T.cdone : 64;
        VDone *n = realloc(T.done, (size_t)nc * sizeof(*n));
        if (!n) {   /* cannot even record it: the slot stays queued and is never used */
            pthread_mutex_unlock(&T.mx);
            for (int k = 0; k < 3; k++) if (t[k]) coli_vk_tensor_free(t[k]);
            return;
        }
        T.done = n; T.cdone = nc;
    }
    T.done[T.ndone++] = (VDone){layer, eid, ok, t[0], t[1], t[2]};
    pthread_mutex_unlock(&T.mx);
}

static void *uploader(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&T.mx);
        while (!T.qn && !T.stop) pthread_cond_wait(&T.cv, &T.mx);
        if (T.stop) { pthread_mutex_unlock(&T.mx); return NULL; }
        VQ e = T.q[T.qh]; T.qh = (T.qh + 1) % VKT_QCAP; T.qn--;
        pthread_mutex_unlock(&T.mx);
        const uint8_t *b = e.buf;
        const uint8_t *g = b, *u = g + T.gu_codes, *d = u + T.gu_codes;
        const uint8_t *gs = d + T.d_codes, *us = gs + T.gu_scales, *ds = us + T.gu_scales;
        ColiVkTensor *t[3];
        int ok = 0;
        /* Refused: the pool is at its budget until the engine thread frees the victim
         * this promotion displaced, at its next quiescent point (a join: every layer
         * in decode, after a chunk's CPU work in prefill). Wait for room to be made
         * and try again; refused again after three frees, or nothing freed for a
         * minute (the engine stopped stepping), the pool is full for real. */
        for (int frees = 0, waited = 0; !ok; ) {
            ok = upload(g, u, d, gs, us, ds, t);
            if (ok || frees >= 3 || waited >= 600) break;
            pthread_mutex_lock(&T.mx);
            unsigned long gen = T.room_gen;
            while (T.room_gen == gen && !T.stop && waited < 600) {
                struct timespec until; clock_gettime(CLOCK_REALTIME, &until);
                until.tv_nsec += 100 * 1000000L; if (until.tv_nsec >= 1000000000L) { until.tv_sec++; until.tv_nsec -= 1000000000L; }
                if (pthread_cond_timedwait(&T.cv_room, &T.mx, &until) == ETIMEDOUT) waited++;
            }
            if (T.room_gen != gen) frees++;
            int stop = T.stop;
            pthread_mutex_unlock(&T.mx);
            if (stop) break;
        }
        free(e.buf);
        if (!ok) t[0] = t[1] = t[2] = NULL;
        push_done(e.layer, e.eid, ok, t);
    }
}

/* ---- quiescent points: nothing in flight ---------------------------------------- */
static void quiesce(void) {
    if (T.inflight) return;
    int freed = 0;
    for (int i = 0; i < T.nevict; i++) {
        VSlot *v = &T.s[T.evict[i]];
        if (v->ex) { coli_vk_xb_expert_free(v->ex); v->ex = NULL; freed = 1; }
        v->state = VS_NONE;
    }
    T.nevict = 0;
    if (freed) {
        pthread_mutex_lock(&T.mx); T.room_gen++; pthread_cond_broadcast(&T.cv_room); pthread_mutex_unlock(&T.mx);
    }
    /* finished uploads become resident */
    pthread_mutex_lock(&T.mx);
    int n = T.ndone; VDone *d = n ? malloc((size_t)n * sizeof(*d)) : NULL;
    if (d) { memcpy(d, T.done, (size_t)n * sizeof(*d)); T.ndone = 0; } else n = 0;
    pthread_mutex_unlock(&T.mx);
    for (int i = 0; i < n; i++) {
        VSlot *v = slot(d[i].layer, d[i].eid);
        T.queued--;
        ColiVkExpert *e = d[i].ok ? coli_vk_xb_expert(d[i].g, d[i].u, d[i].d) : NULL;
        if (e) {
            v->ex = e; v->state = VS_RESIDENT; T.resident++;
            T.uploads++; T.upload_bytes += coli_vk_tensor_bytes(d[i].g) + coli_vk_tensor_bytes(d[i].u) + coli_vk_tensor_bytes(d[i].d);
        } else {
            for (int k = 0; k < 3; k++) { ColiVkTensor *t = k == 0 ? d[i].g : k == 1 ? d[i].u : d[i].d; if (t) coli_vk_tensor_free(t); }
            v->state = VS_NONE; T.failed++;
            /* the budget holds fewer than the count said (fragmentation, or the device
             * ran out first): stop planning past what is there */
            if (T.max_resident > T.resident + T.queued) T.max_resident = T.resident + T.queued;
        }
    }
    free(d);
}

/* ---- placement ------------------------------------------------------------------ */
static uint64_t score(const VSlot *v) { return tier_lfru_score(v->heat, v->last, T.tick); }

/* The coldest residents, ascending, once per forward. */
static void refresh_candidates(void) {
    T.ncand = 0;
    size_t n = (size_t)T.c.layers * T.c.experts;
    for (size_t i = 0; i < n; i++) {
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT) continue;
        uint64_t sc = score(v);
        if (T.ncand == VKT_CAND && sc >= T.cand_score[VKT_CAND - 1]) continue;
        int at = T.ncand < VKT_CAND ? T.ncand++ : VKT_CAND - 1;
        while (at > 0 && T.cand_score[at - 1] > sc) { T.cand[at] = T.cand[at - 1]; T.cand_score[at] = T.cand_score[at - 1]; at--; }
        T.cand[at] = (int)i; T.cand_score[at] = sc;
    }
}
/* The resident a newcomer of score hs would displace, or -1: the coldest candidate
 * still resident, when the newcomer beats it by tier.h's LFRU margin (25% + 4
 * counts). It stays first in the list until drop_victim takes it. */
static void drop_victim(void) {
    memmove(T.cand, T.cand + 1, (size_t)--T.ncand * sizeof(int));
    memmove(T.cand_score, T.cand_score + 1, (size_t)T.ncand * sizeof(uint64_t));
}
static int peek_victim(uint64_t hs) {
    while (T.ncand) {
        int i = T.cand[0];
        VSlot *v = &T.s[i];
        if (v->state != VS_RESIDENT) { drop_victim(); continue; }
        uint64_t cs = score(v);
        return hs > cs + (cs >> 2) + (4u << 8) ? i : -1;
    }
    return -1;
}

/* A forward starts when the layer index goes back (a layer that issues several
 * blocks of rows, or rows one by one, is the same forward). The tick counts tokens:
 * the rows of the forward's first layer, every block of them. */
static void new_forward(void) {
    T.promos = 0; T.promo_cap = 0;
    if (T.tick >= T.decay_at) {
        size_t n = (size_t)T.c.layers * T.c.experts;
        for (size_t i = 0; i < n; i++) T.s[i].heat = tier_decay_value(T.s[i].heat);
        T.decay_at = T.tick + VKT_DECAY_TOKENS;
    }
    refresh_candidates();
}

/* The bytes of an expert the CPU just computed pass by: promote it when there is
 * room, or when it is hot enough to displace the coldest resident. The copy is the
 * only cost on this thread; the uploader does the rest. */
void vkt_note(int layer, int eid, const VktExpertSrc *src) {
    if (!T.on || !src || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return;
    VSlot *v = slot(layer, eid);
    if (v->state != VS_NONE || !src->g || !src->u || !src->d) return;
    if (T.promos >= T.promo_cap) { T.rated++; return; }
    if (T.gu_scales && (!src->gs || !src->us || !src->ds)) return;
    int victim = -1;
    if (T.resident + T.queued >= T.max_resident && (!v->heat || (victim = peek_victim(score(v))) < 0)) return;
    pthread_mutex_lock(&T.mx);
    int qfull = T.qn >= VKT_QCAP;
    pthread_mutex_unlock(&T.mx);
    if (qfull) { T.qfull++; return; }
    uint8_t *buf = malloc(T.stage_bytes);
    if (!buf) return;
    if (victim >= 0) drop_victim();
    uint8_t *p = buf;
    memcpy(p, src->g, T.gu_codes); p += T.gu_codes;
    memcpy(p, src->u, T.gu_codes); p += T.gu_codes;
    memcpy(p, src->d, T.d_codes);  p += T.d_codes;
    if (T.gu_scales) { memcpy(p, src->gs, T.gu_scales); p += T.gu_scales; memcpy(p, src->us, T.gu_scales); p += T.gu_scales; }
    if (T.d_scales) memcpy(p, src->ds, T.d_scales);
    if (victim >= 0) {
        VSlot *w = &T.s[victim];
        w->state = VS_EVICT; T.resident--; T.evictions++;
        if (T.nevict == T.cevict) {
            int nc = T.cevict ? 2 * T.cevict : 64;
            int *n = realloc(T.evict, (size_t)nc * sizeof(int));
            if (!n) { w->state = VS_RESIDENT; T.resident++; T.evictions--; free(buf); return; }
            T.evict = n; T.cevict = nc;
        }
        T.evict[T.nevict++] = victim;
        quiesce();   /* nothing in flight: free it right away */
    }
    v->state = VS_QUEUED; T.queued++; T.promos++;
    pthread_mutex_lock(&T.mx);
    T.q[(T.qh + T.qn) % VKT_QCAP] = (VQ){layer, eid, buf}; T.qn++;
    pthread_cond_signal(&T.cv);
    pthread_mutex_unlock(&T.mx);
}

/* ---- warm start ------------------------------------------------------------------ */
static const uint32_t *const *g_plan_heat;
static int g_plan_E;
static int plan_cmp(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    uint32_t hx = g_plan_heat[x / g_plan_E][x % g_plan_E], hy = g_plan_heat[y / g_plan_E][y % g_plan_E];
    return hx < hy ? 1 : hx > hy ? -1 : (x > y) - (x < y);
}
static uint32_t *const *g_heat_hist;
int vkt_plan(int *layers, int *eids, int max) {
    if (!T.on || !g_heat_hist) return 0;
    int L = T.c.layers, E = T.c.experts, n = 0;
    int *order = malloc((size_t)L * E * sizeof(int));
    if (!order) return 0;
    for (int l = 0; l < L; l++)
        if (g_heat_hist[l]) for (int e = 0; e < E; e++) if (g_heat_hist[l][e]) order[n++] = l * E + e;
    g_plan_heat = (const uint32_t *const *)g_heat_hist; g_plan_E = E;
    qsort(order, (size_t)n, sizeof(int), plan_cmp);
    int k = 0;
    for (int i = 0; i < n && k < max && T.resident + T.queued < T.max_resident; i++) {
        VSlot *v = &T.s[order[i]];
        if (v->state != VS_NONE) continue;
        v->state = VS_QUEUED; T.queued++;
        layers[k] = order[i] / E; eids[k] = order[i] % E; k++;
    }
    free(order);
    return k;
}
int vkt_put(int layer, int eid, const VktExpertSrc *src) {
    if (!T.on || layer < 0 || layer >= T.c.layers || eid < 0 || eid >= T.c.experts) return 0;
    ColiVkTensor *t[3] = {NULL, NULL, NULL};
    int ok = src && src->g && src->u && src->d &&
             upload(src->g, src->u, src->d, src->gs, src->us, src->ds, t);
    push_done(layer, eid, ok, t);
    if (ok) __atomic_add_fetch(&T.warm, 1, __ATOMIC_RELAXED);
    return ok;
}
void vkt_put_done(void) { if (T.on) quiesce(); }

/* ---- one layer step -------------------------------------------------------------- */
/* The per-step arrays: assignments (map), groups (bex, brows, bfirst, touched) and
 * rows (bx, by), each family grown together. */
static int grow_to(int need, int *cap) {
    int nc = *cap ? *cap : 64;
    while (nc < need) nc *= 2;
    return nc;
}
static int grow_map(int need) {
    if (need <= T.cmap) return 1;
    int nc = grow_to(need, &T.cmap);
    int *m = realloc(T.map, (size_t)nc * sizeof(int));
    if (!m) return 0;
    T.map = m; T.cmap = nc;
    return 1;
}
static int grow_groups(int need) {
    if (need <= T.cb) return 1;
    int nc = grow_to(need, &T.cb);
    ColiVkExpert **a = realloc(T.bex, (size_t)nc * sizeof(*a)); if (a) T.bex = a;
    int *b = realloc(T.brows, (size_t)nc * sizeof(int)); if (b) T.brows = b;
    int *c = realloc(T.bfirst, (size_t)nc * sizeof(int)); if (c) T.bfirst = c;
    int *d = realloc(T.touched, (size_t)nc * sizeof(int)); if (d) T.touched = d;
    if (!a || !b || !c || !d) return 0;
    T.cb = nc;
    return 1;
}
static int grow_rows(int need) {
    if (need <= T.cbx) return 1;
    int nc = grow_to(need, &T.cbx);
    const float **a = realloc(T.bx, (size_t)nc * sizeof(*a)); if (a) T.bx = a;
    const float **b = realloc(T.by, (size_t)nc * sizeof(*b)); if (b) T.by = b;
    if (!a || !b) return 0;
    T.cbx = nc;
    return 1;
}

int vkt_issue(int layer, const float *x, int S, int K, const int *idx, uint8_t *taken) {
    if (S > 0 && K > 0) memset(taken, 0, (size_t)S * K);
    if (!T.on || T.inflight || layer < 0 || layer >= T.c.layers || S < 1 || K < 1) return 0;
    if (!coli_vk_xb_ready()) {
        if (T.on) { fprintf(stderr, "[VK] tier %s: the device stopped answering, the experts stay on the CPU\n", T.engine); T.on = 0; }
        return 0;
    }
    if (layer < T.last_layer) { T.first_layer = layer; new_forward(); }
    if (layer == T.first_layer) {   /* COLI_VK_TIER_RATE promotions per token of the forward */
        T.tick += (uint32_t)S;
        long cap = (long)T.promo_cap + (long)T.rate * S;
        T.promo_cap = cap > (1 << 30) ? 1 << 30 : (int)cap;
    }
    T.last_layer = layer;
    quiesce();
    int E = T.c.experts, H = T.c.hidden, n = S * K;
    if (!grow_map(n)) return 0;
    /* heat for every routed expert, groups for the resident ones (first seen first) */
    int ng = 0, total = 0;
    for (int i = 0; i < n; i++) {
        T.map[i] = -1;
        int e = idx[i];
        if (e < 0 || e >= E) continue;
        T.routed++;
        VSlot *v = slot(layer, e);
        if (v->heat < 0xFFFFFFFFu) v->heat++;
        v->last = T.tick;
        if (v->state != VS_RESIDENT) continue;
        int g = T.grp[e];
        if (g < 0) {
            if (total >= T.max_dev_rows || !grow_groups(ng + 1)) continue;   /* the batch is as large as it gets */
            g = ng++; T.grp[e] = g; T.touched[g] = e; T.bex[g] = v->ex; T.brows[g] = 0;
        } else if (total >= T.max_dev_rows) continue;
        T.brows[g]++; total++;
        T.map[i] = g;                                   /* group for now, row index below */
    }
    if (!ng) return 0;
    /* The balance: when the device has been the slower side, the experts the CPU
     * also holds in RAM beyond the device's share of the step's rows go back to the
     * CPU, whole experts (the CPU reads one once for all its rows); the ones only
     * the device holds stay (the CPU would read them from disk). */
    T.can_balance = 0;
    if (T.balance) {
        int cap = (int)(T.share * total + 0.999f), kept = 0, keep_n = 0;
        uint8_t *gone = NULL;
        for (int g = 0; g < ng; g++) {
            if (!T.c.in_ram(T.c.ram_ctx, layer, T.touched[g])) { kept += T.brows[g]; continue; }
            T.can_balance = 1;
            /* the first expert always stays: a step with nothing on the device would
             * measure nothing, and the share could never come back up */
            if (!kept || kept + T.brows[g] <= cap) { kept += T.brows[g]; continue; }
            if (!gone && !(gone = calloc((size_t)ng, 1))) break;
            gone[g] = 1;
        }
        if (gone) {
            int *newg = malloc((size_t)ng * sizeof(int));
            if (newg) {
                for (int g = 0; g < ng; g++) {
                    if (gone[g]) { newg[g] = -1; T.grp[T.touched[g]] = -1; T.handed += (unsigned long long)T.brows[g]; total -= T.brows[g]; continue; }
                    newg[g] = keep_n;
                    T.bex[keep_n] = T.bex[g]; T.brows[keep_n] = T.brows[g]; T.touched[keep_n] = T.touched[g]; keep_n++;
                }
                for (int i = 0; i < n; i++) if (T.map[i] >= 0) T.map[i] = newg[T.map[i]];
                ng = keep_n;
                free(newg);
            }
            free(gone);
        }
        if (!ng) return 0;
    }
    /* rows expert by expert, each expert's in routing order */
    if (!grow_rows(total)) {
        for (int g = 0; g < ng; g++) T.grp[T.touched[g]] = -1;
        for (int i = 0; i < n; i++) T.map[i] = -1;
        return 0;
    }
    int at = 0;
    for (int g = 0; g < ng; g++) { T.bfirst[g] = at; at += T.brows[g]; T.brows[g] = 0; }
    for (int i = 0; i < n; i++) {
        int g = T.map[i];
        if (g < 0) continue;
        int j = T.bfirst[g] + T.brows[g]++;
        T.bx[j] = x + (size_t)(i / K) * H;
        T.map[i] = j;
    }
    for (int g = 0; g < ng; g++) T.grp[T.touched[g]] = -1;
    if (!coli_vk_xb_issue(T.bex, T.brows, ng, T.bx)) {
        for (int i = 0; i < n; i++) T.map[i] = -1;
        return 0;
    }
    for (int i = 0; i < n; i++) if (T.map[i] >= 0) taken[i] = 1;
    T.inflight = 1; T.S = S; T.K = K; T.steps++; T.served += (unsigned long long)total;
    T.t_issued = vkt_now_ms();
    return total;
}

int vkt_join(const float **rows) {
    if (!T.inflight) return 0;
    double t0 = vkt_now_ms(), dms = 0, tc = t0 - T.t_issued;
    T.cpu_ms += tc;
    int ok = coli_vk_xb_join(T.by, &dms);
    double tw = vkt_now_ms() - t0;
    T.wait_ms += tw;
    T.dev_ms += dms;
    /* The balance moves only on steps where it had a choice. Waiting for more than
     * a tenth of the CPU's own time: give the CPU more. The device done well before
     * the CPU (its timestamps say so; without them, nothing waited): take more. */
    if (T.balance && T.can_balance) {
        if (tw > 0.1 * tc + 0.02) T.share *= 0.92f;
        else if (dms > 0 ? dms < 0.8 * tc : tw < 0.01) T.share = T.share * 1.04f + 0.01f;
        if (T.share > 1.f) T.share = 1.f;
        if (T.share < 0.05f) T.share = 0.05f;
    }
    T.inflight = 0;
    int n = T.S * T.K;
    for (int i = 0; i < n; i++) rows[i] = ok && T.map[i] >= 0 ? T.by[T.map[i]] : NULL;
    if (!ok) {
        fprintf(stderr, "[VK] tier %s: a batch failed, its experts are recomputed on the CPU and the tier stops\n", T.engine);
        T.on = 0;
        return 0;
    }
    quiesce();
    return 1;
}

int vkt_resident(int layer, int eid) {
    return T.on && layer >= 0 && layer < T.c.layers && eid >= 0 && eid < T.c.experts &&
           slot(layer, eid)->state == VS_RESIDENT;
}
int vkt_ready(void) { return T.on; }

/* ---- budget ---------------------------------------------------------------------- */
static size_t mem_available(void) {
#ifdef __linux__
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; size_t kb = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}
static double env_gb(const char *name, double def) {
    const char *e = getenv(name);
    return e && *e ? atof(e) : def;
}

int vkt_wanted(void) {
    const char *on = getenv("COLI_VK_TIER");
    return !(on && *on == '0');
}

int vkt_init(const VktConfig *cfg, uint32_t *const *heat) {
    if (T.on || !cfg) return 0;
    if (!vkt_wanted()) return 0;
    if (!coli_vk_available()) return 0;
    const char *eng = cfg->engine ? cfg->engine : "engine";
    memset(&T, 0, sizeof T);
    T.c = *cfg;
    snprintf(T.engine, sizeof T.engine, "%s", eng);
    if (T.c.layers < 1 || T.c.experts < 1 || T.c.hidden < 1 || T.c.inter < 1 || T.c.topk < 1) return 0;
    if (!dev_fmt(T.c.gate_up, &T.gu_fmt, &T.gu_gs) || !dev_fmt(T.c.down, &T.dn_fmt, &T.dn_gs)) {
        fprintf(stderr, "[VK] tier %s: expert format %d/%d (gs %d/%d) has no device form, the experts stay on the CPU\n",
                eng, T.c.gate_up.kind, T.c.down.kind, T.c.gate_up.gs, T.c.down.gs);
        return 0;
    }
    int act = T.c.act == VKT_ACT_SITU ? COLI_VK_ACT_SITU : COLI_VK_ACT_SWIGLU;
    if (!coli_vk_xb_init(T.c.hidden, T.c.inter, act, T.c.act_limit, T.c.act_a, T.c.act_b)) {
        fprintf(stderr, "[VK] tier %s: no expert batch on this device (shaders?), the experts stay on the CPU\n", eng);
        return 0;
    }
    int H = T.c.hidden, F = T.c.inter;
    T.gu_codes = src_row_bytes(T.c.gate_up.kind, H) * (size_t)F;
    T.d_codes = src_row_bytes(T.c.down.kind, F) * (size_t)H;
    T.gu_scales = src_scale_bytes(T.c.gate_up, H, F);
    T.d_scales = src_scale_bytes(T.c.down, F, H);
    T.stage_bytes = 2 * T.gu_codes + T.d_codes + 2 * T.gu_scales + T.d_scales;
    T.exp_bytes = vkt_expert_bytes(H, F, T.c.gate_up, T.c.down);

    /* The budget. Discrete: what VK_EXT_memory_budget says is free on the device,
     * minus a reserve (scratch, KV mirrors, driver) and the dense weights still to
     * come. Integrated (device memory is host RAM): a quarter of what the RAM has
     * left after the engine's expert cache has grown to its size and the dense
     * weights are placed, so the tier never takes what the CPU cache needs.
     * COLI_VK_TIER_GB replaces either, within what the device can hold. */
    const double GiB = 1073741824.0;
    double used = 0, bud = 0;
    int have = coli_vk_mem_budget(&used, &bud);
    double heap_free = have ? (bud - used) * 1e9 : (double)coli_vk_device_local_bytes() * 0.8;
    T.uma = coli_vk_device_shares_ram();
    double reserve = env_gb("COLI_VK_TIER_RESERVE_GB", 1.0) * GiB;
    double room = heap_free - reserve - (double)T.c.dense_bytes;
    double want;
    const char *cap = getenv("COLI_VK_TIER_GB");
    if (cap && *cap) want = atof(cap) * GiB;
    else if (T.uma) {
        double avail = (double)mem_available();
        double ram_room = avail > 0 ? avail - (double)T.c.ram_reserve - (double)T.c.dense_bytes - 2.0 * GiB : room;
        want = ram_room / 4;
    } else want = room;
    if (want > room) want = room;
    if (want < 0) want = 0;
    /* whole experts per pool block, so the count matches what the blocks can hold */
    size_t blk = (size_t)256 << 20;
    if ((size_t)want < blk) blk = (size_t)want;
    long long fit = 0;
    if (T.exp_bytes && blk >= T.exp_bytes) {
        size_t full = (size_t)want / blk, rest = (size_t)want - full * blk;
        fit = (long long)full * (long long)(blk / T.exp_bytes) + (long long)(rest / T.exp_bytes);
    }
    long long all = (long long)T.c.layers * T.c.experts;
    if (fit > all) fit = all;
    if (fit < 2) {
        char hb[32], he[32];
        fprintf(stderr, "[VK] tier %s: no room for experts (budget %s, %s each; COLI_VK_TIER_GB sets it), the experts stay on the CPU\n",
                eng, human(want, hb, sizeof hb), human((double)T.exp_bytes, he, sizeof he));
        return 0;
    }
    T.max_resident = (int)fit;
    T.budget = (size_t)want;
    coli_vk_tier_pool_limit(T.budget);
    const char *bal = getenv("COLI_VK_TIER_BALANCE");
    T.balance = T.c.in_ram != NULL && !(bal && *bal == '0');
    T.share = 1.f;
    const char *r = getenv("COLI_VK_TIER_RATE");
    T.rate = r && *r ? atoi(r) : 16;
    if (T.rate < 0) T.rate = 0;
    T.s = calloc((size_t)all, sizeof(VSlot));
    T.grp = malloc((size_t)T.c.experts * sizeof(int));
    if (!T.s || !T.grp) { free(T.s); free(T.grp); return 0; }
    for (int e = 0; e < T.c.experts; e++) T.grp[e] = -1;
    T.max_dev_rows = T.c.max_rows > 0 ? T.c.max_rows : 1 << 20;
    T.last_layer = 1 << 30; T.first_layer = -1; T.decay_at = VKT_DECAY_TOKENS;
    /* The history, scaled: the hottest expert of a layer starts at 32 and the rest in
     * proportion, so a few minutes of a new workload can displace it (raw counts
     * from a long history would hold the tier for hours). */
    g_heat_hist = heat;
    if (heat)
        for (int l = 0; l < T.c.layers; l++) {
            if (!heat[l]) continue;
            uint32_t mx = 0;
            for (int e = 0; e < T.c.experts; e++) if (heat[l][e] > mx) mx = heat[l][e];
            if (mx) for (int e = 0; e < T.c.experts; e++)
                if (heat[l][e]) T.s[(size_t)l * T.c.experts + e].heat = 1 + (uint32_t)(31.0 * heat[l][e] / mx);
        }
    if (pthread_mutex_init(&T.mx, NULL) || pthread_cond_init(&T.cv, NULL) || pthread_cond_init(&T.cv_room, NULL)) return 0;
    if (pthread_create(&T.th, NULL, uploader, NULL)) return 0;
    T.th_on = 1;
    T.on = 1;
    char hb[32], he[32];
    fprintf(stderr, "[VK] tier %s: on, %s, budget %s = %d experts of %s (fmt %d", eng,
            coli_vk_device_name(), human((double)T.budget, hb, sizeof hb), T.max_resident,
            human((double)T.exp_bytes, he, sizeof he), T.gu_fmt);
    if (T.gu_gs) fprintf(stderr, " gs %d", T.gu_gs);
    fprintf(stderr, ", down fmt %d", T.dn_fmt);
    if (T.dn_gs) fprintf(stderr, " gs %d", T.dn_gs);
    fprintf(stderr, "), %s, %s queue, up to %d promotions per token%s\n",
            T.uma ? (cap && *cap ? "shared RAM (COLI_VK_TIER_GB)" : "shared RAM: a quarter of what the expert cache leaves")
                  : "device memory", coli_vk_xb_queue_shared() ? "shared" : "own", T.rate,
            T.balance ? ", balanced against the CPU" : "");
    return 1;
}

void vkt_report(const char *scope, unsigned long long ram_hits, unsigned long long disk_loads) {
    if (!T.s) return;
    ColiVkPoolStats ps; coli_vk_pool_stats(1, &ps);
    unsigned long long r = T.routed - T.rep_routed, sv = T.served - T.rep_served;
    T.rep_routed = T.routed; T.rep_served = T.served;
    double hidden = T.dev_ms > T.wait_ms ? T.dev_ms - T.wait_ms : 0;
    char hu[32], hb[32], hl[32];
    fprintf(stderr, "[VK] tier %s %s: device %llu of %llu routed experts (%.1f%%; this %s %llu of %llu) | "
            "CPU RAM hits %llu, disk loads %llu | resident %d (budget %d, %s of %s, %d blocks, frag %.2f) | "
            "uploads %llu (%s, %llu warm), evictions %llu, skipped %llu queue + %llu rate, failed %llu | "
            "device %.1f ms, CPU share %.1f ms, waited %.1f ms (%.0f%% of device time hidden)",
            T.engine, scope, T.served, T.routed, T.routed ? 100.0 * T.served / T.routed : 0.0, scope, sv, r,
            ram_hits, disk_loads, T.resident, T.max_resident, human((double)ps.used, hu, sizeof hu),
            human((double)T.budget, hb, sizeof hb), ps.blocks, ps.frag, T.uploads,
            human((double)T.upload_bytes, hl, sizeof hl), T.warm, T.evictions, T.qfull, T.rated, T.failed,
            T.dev_ms, T.cpu_ms, T.wait_ms, T.dev_ms > 0 ? 100.0 * hidden / T.dev_ms : 0.0);
    if (T.balance) fprintf(stderr, " | balance: device share %.2f, %llu rows handed to the CPU", T.share, T.handed);
    fprintf(stderr, "\n");
}

void vkt_shutdown(void) {
    if (!T.s) return;
    if (T.th_on) {
        pthread_mutex_lock(&T.mx); T.stop = 1; pthread_cond_broadcast(&T.cv); pthread_cond_broadcast(&T.cv_room); pthread_mutex_unlock(&T.mx);
        pthread_join(T.th, NULL);
        T.th_on = 0;
    }
    if (T.inflight) { const float *dummy[1]; (void)dummy; coli_vk_xb_join(T.by, NULL); T.inflight = 0; }
    for (int i = 0; i < T.qn; i++) free(T.q[(T.qh + i) % VKT_QCAP].buf);
    T.qn = 0;
    if (coli_vk_available()) {
        quiesce();
        size_t n = (size_t)T.c.layers * T.c.experts;
        for (size_t i = 0; i < n; i++) if (T.s[i].ex) { coli_vk_xb_expert_free(T.s[i].ex); T.s[i].ex = NULL; }
    }
    pthread_mutex_destroy(&T.mx); pthread_cond_destroy(&T.cv); pthread_cond_destroy(&T.cv_room);
    free(T.s); free(T.grp); free(T.map); free(T.touched); free(T.bex); free(T.brows); free(T.bfirst);
    free(T.bx); free(T.by); free(T.evict); free(T.done);
    memset(&T, 0, sizeof T);
}
#endif /* COLI_VULKAN */
