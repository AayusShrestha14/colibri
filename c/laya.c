/* laya.c -- Laya, a decision model by Convai Innovations
 * (convaiinnovations/laya on Hugging Face, Apache-2.0), in C.
 *
 * Laya does not generate. Given a state and typed questions (choice, score,
 * noul) it scores every option of every question in one forward pass and
 * returns calibrated probabilities. It is a bidirectional encoder,
 * ModernBERT-large in the English checkpoint, plus a decision head trained
 * from scratch: two transformer layers, an option scorer read at one [MASK]
 * marker per option, and an act/escalate head.
 *
 * The specification is the model's own Python: the `laya` package (common.py,
 * agent.py) and the ModernBERT modeling code it builds on. Each step below
 * names the function it follows:
 *
 *   sequence     common.build_head / build_sequence / render_options
 *   encoder      ModernBertModel.forward: LayerNorm'd embeddings, 28 pre-norm
 *                layers (no norm before layer 0's attention), RoPE per layer
 *                type (global theta 160000 on every third layer starting at 0,
 *                local theta 10000 elsewhere with a +-64 window), GeGLU MLP,
 *                final LayerNorm; no biases
 *   head         DecisionModel.forward: + type_emb[qtype], two post-attention
 *                nn.TransformerEncoderLayer (norm_first, ReLU, d/64 heads),
 *                scorer LN -> Linear -> GELU -> Linear at each marker,
 *                act head on [h[0], top1, top1-top2, entropy, k/255]
 *   calibration  Agent._decode_answers: one temperature per (type, option
 *                count) bucket, clamped to [0.5, 5] (common.clamp_temperature),
 *                softmax over the question's options
 *
 * Weights are read as stored (F16 in the release) and kept in f32. The
 * forward runs every question of a request as one batch of rows; attention
 * stays per sequence, so a question never sees another's tokens.
 *
 * Two ways to run it:
 *   SERVE=1 SNAP=<dir> laya          the mux serve protocol, DECIDE only
 *                                    (decide_serve.h; what the gateway runs)
 *   laya --model DIR --records F     a JSON array of DECIDE records; one
 *                                    DECISION object per line (tests)
 *   laya --model DIR --tokenize F    a JSON array of strings; their ids
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "compat.h"
#include "json.h"
#include "st.h"
#include "tok.h"
#include "qwen38_nfc.h"
#include "qi_gemm.h"
#include "matmul_f32.h"
#include "serve_codec.h"
#include "decide_serve.h"
#include "omp_tune.h"

/* ------------------------------------------------------------------ utils */

static void die(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[laya] ");
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
    va_end(args);
    exit(1);
}

static void *xmalloc(size_t size)
{
    void *p = malloc(size ? size : 1);
    if (!p) die("out of memory (%zu bytes)", size);
    return p;
}

static void *xcalloc(size_t count, size_t size)
{
    void *p = calloc(count ? count : 1, size ? size : 1);
    if (!p) die("out of memory (%zu x %zu bytes)", count, size);
    return p;
}

static char *read_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 30)) { fclose(f); return NULL; }
    rewind(f);
    char *b = (char *)xmalloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    b[n] = 0;
    fclose(f);
    return b;
}

static jval *read_json(const char *dir, const char *name)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    char *text = read_text(path);
    if (!text) die("cannot read %s", path);
    jval *root = json_parse_checked(text);
    free(text);
    if (!root || root->t != J_OBJ) die("%s is not a JSON object", path);
    return root;
}

static double num_or(jval *o, const char *key, double fallback)
{
    jval *v = json_get(o, key);
    return v && v->t == J_NUM ? v->num : fallback;
}

static int bool_or(jval *o, const char *key, int fallback)
{
    jval *v = json_get(o, key);
    return v && v->t == J_BOOL ? v->boolean : fallback;
}

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* Python's floor division, for the one place the reference divides a value
 * that can be negative (build_head's per-option cap). */
static int floordiv(int a, int b)
{
    int q = a / b;
    if ((a % b) && ((a < 0) != (b < 0))) q--;
    return q;
}

/* ------------------------------------------------------------------ model */

typedef struct {
    float *attn_norm_w, *attn_norm_b;   /* NULL on layer 0: ModernBERT has no norm there */
    QiMat wqkv, wo, wi, wo2;
    float *bqkv, *bo, *bi, *bo2;        /* NULL without attention_bias / mlp_bias */
    float *mlp_norm_w, *mlp_norm_b;
    int global;                          /* full attention; otherwise the local window */
} EncLayer;

typedef struct {
    float *n1w, *n1b, *n2w, *n2b;
    QiMat in_proj, out_proj, lin1, lin2;
    float *in_proj_b, *out_proj_b, *lin1_b, *lin2_b;
} HeadLayer;

typedef struct {
    /* encoder config */
    int d, layers, heads, hd, inter, vocab, max_pos, window;
    float eps, theta_global, theta_local;
    int gelu_tanh;
    float *tok_emb;                      /* [vocab][d] */
    float *emb_norm_w, *emb_norm_b, *final_norm_w, *final_norm_b;
    EncLayer *L;
    float *rope_g, *rope_l;              /* [max_pos][hd/2] cos then [max_pos][hd/2] sin */
    /* decision head */
    int head_layers, head_heads;
    HeadLayer *H;
    float *type_emb;                     /* [3][d] */
    float *sc_ln_w, *sc_ln_b, *sc2_w, sc2_b;
    QiMat sc1; float *sc1_b;
    int act_hidden, n_act;
    QiMat act1; float *act1_b;
    float *act2_w, *act2_b;              /* [n_act][act_hidden] */
    char *action_names[DECIDE_MAX_ACTIONS];
    /* rl_agent_config.json */
    int max_len, head_max_len;
    double temperature[3];
    int n_buckets;
    char *bucket_key[32];
    double bucket_t[32];
    /* tokenizer */
    Tok tok;
    int nfc;
    int cls_id, sep_id, mask_id, pad_id;
    char *mask_str;
    unsigned char added_first[256];      /* first bytes of the added tokens */
    char model_name[128];
} Laya;

/* common.clamp_temperature: a usable temperature, confined to [0.5, 5]; a value
 * that is not a finite number (or is a JSON bool) falls back to 1.0. */
static double clamp_temperature(jval *v)
{
    if (!v || v->t != J_NUM || !isfinite(v->num)) return 1.0;
    double t = v->num;
    return t < 0.5 ? 0.5 : t > 5.0 ? 5.0 : t;
}

static float *load_f32(shards *S, const char *name, int64_t expect)
{
    st_tensor *t = st_find(S, name);
    if (!t) die("model.safetensors has no tensor %s", name);
    if (expect >= 0 && t->numel != expect)
        die("%s has %lld elements, expected %lld", name, (long long)t->numel, (long long)expect);
    float *out = (float *)xmalloc((size_t)t->numel * sizeof(float));
    st_read_f32(S, name, out, 0);
    return out;
}

static float *load_opt(shards *S, const char *name, int64_t expect)
{
    return st_find(S, name) ? load_f32(S, name, expect) : NULL;
}

static QiMat load_mat(shards *S, const char *name, int N, int K)
{
    QiMat m = {QI_F32, N, K, load_f32(S, name, (int64_t)N * K), NULL, 0};
    return m;
}

/* COLI_LAYA_INT8=1: the large matrices as int8 with one scale per row. Opt-in,
 * and only worth it where it was measured to leave the decisions unchanged. */
static void maybe_int8(QiMat *m)
{
    const char *flag = getenv("COLI_LAYA_INT8");
    if (!flag || strcmp(flag, "1")) return;
    int8_t *q = (int8_t *)xmalloc((size_t)m->N * m->K);
    float *sc = (float *)xmalloc((size_t)m->N * sizeof(float));
    qi_quantize_i8((const float *)m->w, m->N, m->K, q, sc);
    free((void *)m->w);
    m->fmt = QI_I8; m->w = q; m->sc = sc;
}

static void rope_table(float *table, int max_pos, int hd, float theta)
{
    /* ModernBertRotaryEmbedding.compute_default_rope_parameters, in float32 like
     * the reference: inv_freq = 1 / theta^(2i/hd), angle = pos * inv_freq. */
    int half = hd / 2;
    float *cosv = table, *sinv = table + (size_t)max_pos * half;
    for (int i = 0; i < half; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)hd);
        for (int p = 0; p < max_pos; p++) {
            float a = (float)p * inv;
            cosv[(size_t)p * half + i] = cosf(a);
            sinv[(size_t)p * half + i] = sinf(a);
        }
    }
}

static const char *special_content(jval *cfg, const char *key)
{
    jval *v = json_get(cfg, key);
    if (v && v->t == J_STR) return v->str;
    if (v && v->t == J_OBJ) { jval *c = json_get(v, "content"); if (c && c->t == J_STR) return c->str; }
    return NULL;
}

static void load_tokenizer(Laya *M, const char *dir)
{
    char path[4096];
    snprintf(path, sizeof(path), "%s/tokenizer/tokenizer.json", dir);
    jval *tj = read_json(dir, "tokenizer/tokenizer.json");
    jval *model = json_get(tj, "model"), *mt = model ? json_get(model, "type") : NULL;
    if (!mt || mt->t != J_STR || strcmp(mt->str, "BPE"))
        die("tokenizer: only a byte-level BPE tokenizer is supported (this is %s)",
            mt && mt->t == J_STR ? mt->str : "unknown");
    jval *pt = json_get(tj, "pre_tokenizer"), *ptt = pt ? json_get(pt, "type") : NULL;
    if (!ptt || ptt->t != J_STR || strcmp(ptt->str, "ByteLevel") || !bool_or(pt, "use_regex", 1) ||
        bool_or(pt, "add_prefix_space", 0))
        die("tokenizer: expected a ByteLevel pre-tokenizer with the GPT-2 regex and no prefix space");
    jval *norm = json_get(tj, "normalizer");
    if (norm && norm->t == J_OBJ) {
        jval *nt = json_get(norm, "type");
        if (nt && nt->t == J_STR && !strcmp(nt->str, "NFC")) M->nfc = 1;
        else die("tokenizer: unsupported normalizer %s", nt && nt->t == J_STR ? nt->str : "?");
    } else if (norm && norm->t != J_NULL) {
        die("tokenizer: unsupported normalizer");
    }
    /* Added tokens: the reference matches them on the raw text (special ones) or
     * on the normalized text (the rest); NFC cannot create or break any of these
     * ASCII strings, so matching once after normalizing is the same split. The
     * one lstrip/rstrip token is [MASK], which never reaches the tokenizer:
     * build_sequence replaces it in every text first. */
    jval *added = json_get(tj, "added_tokens");
    for (int i = 0; added && i < added->len; i++) {
        jval *a = added->kids[i], *c = json_get(a, "content");
        if (!c || c->t != J_STR || !c->str[0]) continue;
        int is_mask = M->mask_str && !strcmp(c->str, M->mask_str);
        if (!is_mask && (bool_or(a, "lstrip", 0) || bool_or(a, "rstrip", 0) || bool_or(a, "single_word", 0)))
            die("tokenizer: added token %s strips whitespace, which this engine does not implement", c->str);
        M->added_first[(unsigned char)c->str[0]] = 1;
    }
    json_free(tj);
    tok_load(&M->tok, path);
    if (!M->tok.gpt2) die("tokenizer: not the GPT-2 byte-level pre-tokenizer");
}

static void load_model(Laya *M, const char *dir)
{
    memset(M, 0, sizeof(*M));
    /* rl_agent_config.json: the head and the calibration */
    jval *rc = read_json(dir, "rl_agent_config.json");
    M->head_layers = (int)num_or(rc, "head_layers", 2);
    M->max_len = (int)num_or(rc, "max_len", 512);
    M->head_max_len = (int)num_or(rc, "head_max_len", 192);
    const char *env;
    if ((env = getenv("COLI_LAYA_MAX_LEN")) && atoi(env) > 0) M->max_len = atoi(env);
    if ((env = getenv("COLI_LAYA_HEAD_MAX_LEN")) && atoi(env) > 0) M->head_max_len = atoi(env);
    jval *name = json_get(rc, "model_name");
    snprintf(M->model_name, sizeof(M->model_name), "%s", name && name->t == J_STR ? name->str : "laya");
    jval *temps = json_get(rc, "temperature");
    for (int i = 0; i < 3; i++)
        M->temperature[i] = temps && temps->t == J_ARR && temps->len == 3 ? clamp_temperature(temps->kids[i]) : 1.0;
    if (temps && (temps->t != J_ARR || temps->len != 3))
        die("rl_agent_config.json: temperature must be a list of 3 numbers");
    jval *tbo = json_get(rc, "temperature_by_options");
    for (int i = 0; tbo && tbo->t == J_OBJ && i < tbo->len && M->n_buckets < 32; i++) {
        M->bucket_key[M->n_buckets] = strdup(tbo->keys[i]);
        M->bucket_t[M->n_buckets++] = clamp_temperature(tbo->kids[i]);
    }
    jval *costs = json_get(rc, "act_costs");
    M->n_act = 1 + (costs && costs->t == J_OBJ ? costs->len : 0);
    if (M->n_act > DECIDE_MAX_ACTIONS) die("rl_agent_config.json: too many act_costs");
    M->action_names[0] = strdup("act");
    for (int i = 1; i < M->n_act; i++) M->action_names[i] = strdup(costs->keys[i - 1]);
    json_free(rc);

    /* encoder/config.json: ModernBERT */
    jval *ec = read_json(dir, "encoder/config.json");
    jval *mtype = json_get(ec, "model_type");
    if (!mtype || mtype->t != J_STR || strcmp(mtype->str, "modernbert"))
        die("encoder/config.json: model_type must be modernbert");
    M->d = (int)num_or(ec, "hidden_size", 0);
    M->layers = (int)num_or(ec, "num_hidden_layers", 0);
    M->heads = (int)num_or(ec, "num_attention_heads", 0);
    M->inter = (int)num_or(ec, "intermediate_size", 0);
    M->vocab = (int)num_or(ec, "vocab_size", 0);
    M->max_pos = (int)num_or(ec, "max_position_embeddings", 8192);
    M->window = (int)num_or(ec, "local_attention", 128) / 2;
    M->eps = (float)num_or(ec, "norm_eps", 1e-5);
    if (M->d < 1 || M->layers < 1 || M->heads < 1 || M->d % M->heads || M->inter < 1 || M->vocab < 1 ||
        M->max_pos < 1 || M->d > 16384 || M->layers > 256 || M->max_pos > 1 << 20)
        die("encoder/config.json: invalid geometry");
    M->hd = M->d / M->heads;
    if (M->hd % 2) die("encoder/config.json: head dimension must be even");
    jval *act = json_get(ec, "hidden_activation");
    if (act && act->t == J_STR) {
        if (!strcmp(act->str, "gelu")) M->gelu_tanh = 0;
        else if (!strcmp(act->str, "gelu_pytorch_tanh") || !strcmp(act->str, "gelu_new")) M->gelu_tanh = 1;
        else die("encoder/config.json: unsupported hidden_activation %s", act->str);
    }
    /* RoPE: transformers 5 writes rope_parameters per layer type; older configs
     * global_rope_theta / local_rope_theta (defaults 160000 / 10000). */
    M->theta_global = (float)num_or(ec, "global_rope_theta", 160000.0);
    M->theta_local = (float)num_or(ec, "local_rope_theta", 10000.0);
    jval *rp = json_get(ec, "rope_parameters");
    if (rp && rp->t == J_OBJ) {
        jval *f = json_get(rp, "full_attention"), *s = json_get(rp, "sliding_attention");
        if (f && f->t == J_OBJ) M->theta_global = (float)num_or(f, "rope_theta", M->theta_global);
        if (s && s->t == J_OBJ) M->theta_local = (float)num_or(s, "rope_theta", M->theta_local);
        jval *ft = f ? json_get(f, "rope_type") : NULL, *stp = s ? json_get(s, "rope_type") : NULL;
        if ((ft && ft->t == J_STR && strcmp(ft->str, "default")) ||
            (stp && stp->t == J_STR && strcmp(stp->str, "default")))
            die("encoder/config.json: only the default RoPE is supported");
    }
    M->L = (EncLayer *)xcalloc((size_t)M->layers, sizeof(EncLayer));
    jval *types = json_get(ec, "layer_types");
    int every = (int)num_or(ec, "global_attn_every_n_layers", 3);
    for (int l = 0; l < M->layers; l++) {
        if (types && types->t == J_ARR && types->len == M->layers && types->kids[l]->t == J_STR)
            M->L[l].global = !strcmp(types->kids[l]->str, "full_attention");
        else
            M->L[l].global = every > 0 ? l % every == 0 : 1;
    }
    json_free(ec);

    /* tokenizer: the special tokens by name, from tokenizer_config.json */
    jval *tc = read_json(dir, "tokenizer/tokenizer_config.json");
    const char *cls = special_content(tc, "cls_token"), *sep = special_content(tc, "sep_token");
    const char *mask = special_content(tc, "mask_token"), *pad = special_content(tc, "pad_token");
    if (!cls || !sep || !mask) die("tokenizer_config.json: needs cls_token, sep_token and mask_token");
    M->mask_str = strdup(mask);
    char *cls_s = strdup(cls), *sep_s = strdup(sep), *pad_s = pad ? strdup(pad) : NULL;
    json_free(tc);
    load_tokenizer(M, dir);
    M->cls_id = tok_id_of(&M->tok, cls_s);
    M->sep_id = tok_id_of(&M->tok, sep_s);
    M->mask_id = tok_id_of(&M->tok, M->mask_str);
    M->pad_id = pad_s ? tok_id_of(&M->tok, pad_s) : -1;
    if (M->cls_id < 0 || M->sep_id < 0 || M->mask_id < 0)
        die("tokenizer: cls/sep/mask are not added tokens of tokenizer.json");
    free(cls_s); free(sep_s); free(pad_s);

    /* weights */
    shards S;
    st_init(&S, dir);
    int d = M->d, I = M->inter;
    M->tok_emb = load_f32(&S, "encoder.embeddings.tok_embeddings.weight", (int64_t)M->vocab * d);
    M->emb_norm_w = load_f32(&S, "encoder.embeddings.norm.weight", d);
    M->emb_norm_b = load_opt(&S, "encoder.embeddings.norm.bias", d);
    M->final_norm_w = load_f32(&S, "encoder.final_norm.weight", d);
    M->final_norm_b = load_opt(&S, "encoder.final_norm.bias", d);
    char n[256];
    for (int l = 0; l < M->layers; l++) {
        EncLayer *E = &M->L[l];
#define NAME(fmt) (snprintf(n, sizeof(n), "encoder.layers.%d." fmt, l), n)
        if (l > 0) {
            E->attn_norm_w = load_f32(&S, NAME("attn_norm.weight"), d);
            E->attn_norm_b = load_opt(&S, NAME("attn_norm.bias"), d);
        }
        E->wqkv = load_mat(&S, NAME("attn.Wqkv.weight"), 3 * d, d);
        E->bqkv = load_opt(&S, NAME("attn.Wqkv.bias"), 3 * d);
        E->wo = load_mat(&S, NAME("attn.Wo.weight"), d, d);
        E->bo = load_opt(&S, NAME("attn.Wo.bias"), d);
        E->mlp_norm_w = load_f32(&S, NAME("mlp_norm.weight"), d);
        E->mlp_norm_b = load_opt(&S, NAME("mlp_norm.bias"), d);
        E->wi = load_mat(&S, NAME("mlp.Wi.weight"), 2 * I, d);
        E->bi = load_opt(&S, NAME("mlp.Wi.bias"), 2 * I);
        E->wo2 = load_mat(&S, NAME("mlp.Wo.weight"), d, I);
        E->bo2 = load_opt(&S, NAME("mlp.Wo.bias"), d);
        maybe_int8(&E->wqkv); maybe_int8(&E->wo); maybe_int8(&E->wi); maybe_int8(&E->wo2);
#undef NAME
    }
    M->head_heads = d / 64 > 1 ? d / 64 : 1;        /* DecisionModel: nhead = max(1, d // 64) */
    if (d % M->head_heads) die("head: hidden size %d does not split into %d heads", d, M->head_heads);
    M->H = (HeadLayer *)xcalloc((size_t)(M->head_layers > 0 ? M->head_layers : 1), sizeof(HeadLayer));
    for (int l = 0; l < M->head_layers; l++) {
        HeadLayer *H = &M->H[l];
#define NAME(fmt) (snprintf(n, sizeof(n), "head.layers.%d." fmt, l), n)
        H->n1w = load_f32(&S, NAME("norm1.weight"), d);
        H->n1b = load_f32(&S, NAME("norm1.bias"), d);
        H->n2w = load_f32(&S, NAME("norm2.weight"), d);
        H->n2b = load_f32(&S, NAME("norm2.bias"), d);
        H->in_proj = load_mat(&S, NAME("self_attn.in_proj_weight"), 3 * d, d);
        H->in_proj_b = load_f32(&S, NAME("self_attn.in_proj_bias"), 3 * d);
        H->out_proj = load_mat(&S, NAME("self_attn.out_proj.weight"), d, d);
        H->out_proj_b = load_f32(&S, NAME("self_attn.out_proj.bias"), d);
        H->lin1 = load_mat(&S, NAME("linear1.weight"), 4 * d, d);
        H->lin1_b = load_f32(&S, NAME("linear1.bias"), 4 * d);
        H->lin2 = load_mat(&S, NAME("linear2.weight"), d, 4 * d);
        H->lin2_b = load_f32(&S, NAME("linear2.bias"), d);
        maybe_int8(&H->in_proj); maybe_int8(&H->out_proj); maybe_int8(&H->lin1); maybe_int8(&H->lin2);
#undef NAME
    }
    M->type_emb = load_f32(&S, "type_emb.weight", 3 * d);
    M->sc_ln_w = load_f32(&S, "scorer.0.weight", d);
    M->sc_ln_b = load_f32(&S, "scorer.0.bias", d);
    M->sc1 = load_mat(&S, "scorer.1.weight", d, d);
    M->sc1_b = load_f32(&S, "scorer.1.bias", d);
    M->sc2_w = load_f32(&S, "scorer.3.weight", d);
    float *b = load_f32(&S, "scorer.3.bias", 1);
    M->sc2_b = b[0];
    free(b);
    st_tensor *a1 = st_find(&S, "act_head.0.weight");
    if (!a1 || a1->rank != 2 || a1->shape[1] != d + 4) die("act_head.0.weight: expected [hidden, %d]", d + 4);
    M->act_hidden = (int)a1->shape[0];
    M->act1 = load_mat(&S, "act_head.0.weight", M->act_hidden, d + 4);
    M->act1_b = load_f32(&S, "act_head.0.bias", M->act_hidden);
    st_tensor *a2 = st_find(&S, "act_head.2.weight");
    if (!a2 || a2->rank != 2 || a2->shape[0] != M->n_act || a2->shape[1] != M->act_hidden)
        die("act_head.2.weight: expected [%d, %d] (one row per action of act_costs)", M->n_act, M->act_hidden);
    M->act2_w = load_f32(&S, "act_head.2.weight", (int64_t)M->n_act * M->act_hidden);
    M->act2_b = load_f32(&S, "act_head.2.bias", M->n_act);
    st_destroy(&S);

    int half = M->hd / 2;
    M->rope_g = (float *)xmalloc((size_t)M->max_pos * half * 2 * sizeof(float));
    M->rope_l = (float *)xmalloc((size_t)M->max_pos * half * 2 * sizeof(float));
    rope_table(M->rope_g, M->max_pos, M->hd, M->theta_global);
    rope_table(M->rope_l, M->max_pos, M->hd, M->theta_local);
    if (M->max_len > M->max_pos) M->max_len = M->max_pos;
}

/* -------------------------------------------------------------- tokenizer */

/* tokenizer.json applied to one text: NFC, the added tokens split out
 * (leftmost, longest), then the GPT-2 pre-tokenizer and BPE on what is left
 * (tok.h). Returns the ids; *count their number. */
static int *laya_encode(Laya *M, const char *text, size_t len, int *count)
{
    char *norm = NULL;
    size_t norm_len = len;
    const char *p = text;
    if (M->nfc) {
        if (q38_nfc_normalize(text, len, &norm, &norm_len) != 0) die("NFC normalization failed");
        p = norm;
    }
    int cap = (int)norm_len + 8, n = 0;
    int *out = (int *)xmalloc((size_t)cap * sizeof(int));
    const unsigned char *u = (const unsigned char *)p;
    int i = 0, L = (int)norm_len;
    Tok *T = &M->tok;
    while (i < L) {
        int hit = -1, hit_len = 0, hit_id = -1;
        for (int j = i; j < L && hit < 0; j++) {
            if (!M->added_first[u[j]]) continue;
            for (int k = 0; k < T->nsp; k++) {
                int sl = T->sp[k].len;
                if (sl > 0 && j + sl <= L && !memcmp(u + j, T->sp[k].str, (size_t)sl)) {
                    hit = j; hit_len = sl; hit_id = T->sp[k].id;
                    break;
                }
            }
        }
        int end = hit < 0 ? L : hit;
        if (end > i) pretok_chunk_gpt2(T, u, i, end, out, &n, cap);
        if (hit < 0) break;
        if (n < cap) out[n++] = hit_id;
        i = hit + hit_len;
    }
    free(norm);
    *count = n;
    return out;
}

/* str.replace(needle, " ") */
static char *replace_all(const char *s, const char *needle)
{
    size_t nl = strlen(needle), sl = strlen(s);
    char *out = (char *)xmalloc(sl + 1);
    size_t o = 0;
    for (size_t i = 0; i < sl;) {
        if (nl && i + nl <= sl && !memcmp(s + i, needle, nl)) { out[o++] = ' '; i += nl; }
        else out[o++] = s[i++];
    }
    out[o] = 0;
    return out;
}

/* ------------------------------------------------------------ the sequence */

typedef struct {
    int *ids, n;
    int *markers, n_markers;
    int qtype;
    int state_tokens, state_used;
} Seq;

static void seq_free(Seq *s) { free(s->ids); free(s->markers); memset(s, 0, sizeof(*s)); }

/* common.render_options: the option texts in label-index order. */
static char *render_option(const DecideQuestion *q, int i)
{
    const DecideOption *o = &q->options[i];
    const char *text = o->text && o->text[0] ? o->text : NULL;
    size_t size = strlen(o->label) + (o->text ? strlen(o->text) : 0) + 64;
    char *out = (char *)xmalloc(size);
    if (q->type == DECIDE_CHOICE) {
        if (text) snprintf(out, size, "%s: %s", o->label, text);
        else snprintf(out, size, "%s", o->label);
    } else if (q->type == DECIDE_SCORE) {
        snprintf(out, size, "level %d: %s", i, o->text ? o->text : "");
    } else {
        /* noul: [false, true], each with its default description */
        const char *fallback = i == 0 ? "no, the statement does not hold" : "yes, the statement holds";
        snprintf(out, size, "%s: %s", o->label, text ? text : fallback);
    }
    return out;
}

/* common.build_head + build_sequence:
 *   [CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
 * Returns 1, or 0 with the reason when not every option marker fits. */
static int build_sequence(Laya *M, const DecideQuestion *q, const int *state, int n_state,
                          int truncate_left, Seq *out, char *err, size_t cap)
{
    memset(out, 0, sizeof(*out));
    out->qtype = q->type;
    const char *tname = decide_type_name(q->type);
    char *ins = replace_all(q->instructions, M->mask_str);
    size_t hl = strlen(tname) + strlen(ins) + 16;
    char *head_text = (char *)xmalloc(hl);
    snprintf(head_text, hl, "%s question: %s", tname, ins);
    int n_head = 0;
    int *head = laya_encode(M, head_text, strlen(head_text), &n_head);
    free(head_text); free(ins);

    int K = q->n_options;
    int **opt = (int **)xcalloc((size_t)K, sizeof(int *));
    int *olen = (int *)xcalloc((size_t)K, sizeof(int));
    int total = 0;
    for (int i = 0; i < K; i++) {
        char *rendered = render_option(q, i);
        char *clean = replace_all(rendered, M->mask_str);
        size_t cl = strlen(clean);
        char *spaced = (char *)xmalloc(cl + 2);
        spaced[0] = ' ';
        memcpy(spaced + 1, clean, cl + 1);
        int n = 0;
        int *ids = laya_encode(M, spaced, cl + 1, &n);
        if (n > 48) n = 48;                        /* truncation=True, max_length=48 */
        opt[i] = (int *)xmalloc((size_t)(n + 1) * sizeof(int));
        opt[i][0] = M->mask_id;
        memcpy(opt[i] + 1, ids, (size_t)n * sizeof(int));
        olen[i] = n + 1;
        total += n + 1;
        free(ids); free(spaced); free(clean); free(rendered);
    }
    int budget = M->head_max_len - total;
    if (budget < 16) {                             /* shrink every option evenly */
        int per = floordiv(M->head_max_len - 16, K > 1 ? K : 1);
        if (per < 4) per = 4;
        total = 0;
        for (int i = 0; i < K; i++) { if (olen[i] > per) olen[i] = per; total += olen[i]; }
        budget = M->head_max_len - total;
    }
    int keep = budget > 8 ? budget : 8;
    if (n_head > keep) n_head = keep;

    int room_cap = 2 + n_head + total + 1 + n_state + 1;
    int *ids = (int *)xmalloc((size_t)room_cap * sizeof(int));
    int *markers = (int *)xmalloc((size_t)K * sizeof(int));
    int n = 0;
    ids[n++] = M->cls_id;
    memcpy(ids + n, head, (size_t)n_head * sizeof(int)); n += n_head;
    ids[n++] = M->sep_id;
    for (int i = 0; i < K; i++) {
        markers[i] = n;
        memcpy(ids + n, opt[i], (size_t)olen[i] * sizeof(int));
        n += olen[i];
    }
    ids[n++] = M->sep_id;
    int room = M->max_len - n - 1;
    if (room < 0) room = 0;
    int take = n_state < room ? n_state : room;
    const int *from = truncate_left ? state + (n_state - take) : state;
    memcpy(ids + n, from, (size_t)take * sizeof(int)); n += take;
    ids[n++] = M->sep_id;
    if (n > M->max_len) n = M->max_len;
    int kept = 0;
    for (int i = 0; i < K; i++) if (markers[i] < M->max_len) markers[kept++] = markers[i];

    for (int i = 0; i < K; i++) free(opt[i]);
    free(opt); free(olen); free(head);
    out->ids = ids; out->n = n; out->markers = markers; out->n_markers = kept;
    out->state_tokens = n_state; out->state_used = take;
    if (kept != K) {
        snprintf(err, cap, "questions.%s: only %d of its %d option markers fit in max_len=%d with "
                 "head_max_len=%d spent on the question; lower head_max_len, raise max_len, or use "
                 "fewer options", q->id, kept, K, M->max_len, M->head_max_len);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------- the forward */

static inline float gelu(float x, int tanh_form)
{
    if (tanh_form) return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

/* LayerNorm over rows of `d`: biased variance, eps inside the root. */
static void layernorm(float *out, const float *x, const float *w, const float *b, int rows, int d, float eps)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * d;
        float *o = out + (size_t)r * d;
        double mean = 0, var = 0;
        for (int i = 0; i < d; i++) mean += xr[i];
        mean /= d;
        for (int i = 0; i < d; i++) { double c = xr[i] - mean; var += c * c; }
        var /= d;
        float rstd = (float)(1.0 / sqrt(var + eps));
        for (int i = 0; i < d; i++) {
            float v = (float)(xr[i] - mean) * rstd * w[i];
            o[i] = b ? v + b[i] : v;
        }
    }
}

static void add_rows(float *x, const float *y, size_t n)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) x[i] += y[i];
}

static void gemm(float *Y, const float *X, int M, const QiMat *W, const float *bias)
{
    qi_gemm(Y, X, M, W, bias);
}

/* Rotate q and k of every row in place: q*cos + rotate_half(q)*sin, positions
 * counted from each sequence's start. */
static void apply_rope(float *qkv, int T, const int *row_pos, int d, int heads, int hd,
                       const float *table, int max_pos)
{
    int half = hd / 2;
    const float *cosv = table, *sinv = table + (size_t)max_pos * half;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < T; r++) {
        const float *c = cosv + (size_t)row_pos[r] * half, *s = sinv + (size_t)row_pos[r] * half;
        for (int part = 0; part < 2; part++) {
            float *base = qkv + (size_t)r * 3 * d + (size_t)part * d;
            for (int h = 0; h < heads; h++) {
                float *v = base + (size_t)h * hd;
                for (int i = 0; i < half; i++) {
                    float a = v[i], b = v[i + half];
                    v[i] = a * c[i] - b * s[i];
                    v[i + half] = b * c[i] + a * s[i];
                }
            }
        }
    }
}

/* Softmax attention inside each sequence. qkv rows are [q | k | v], each d
 * wide with the heads side by side. window < 0 is full attention; otherwise
 * key j is visible to query i when |i - j| <= window. `only` (optional) marks
 * the query rows whose output is needed; the rest are skipped. */
static void attention(const float *qkv, float *out, int T, int d, int heads, int hd,
                      const int *row_seq, const int *seq_off, int window, const unsigned char *only)
{
    const float scale = 1.0f / sqrtf((float)hd);
    #pragma omp parallel
    {
        int cap = 0;
        for (int r = 0; r < T; r++) {
            int L = seq_off[row_seq[r] + 1] - seq_off[row_seq[r]];
            if (L > cap) cap = L;
        }
        float *score = (float *)xmalloc((size_t)(cap > 0 ? cap : 1) * sizeof(float));
        float *acc = (float *)xmalloc((size_t)hd * sizeof(float));
        #pragma omp for schedule(dynamic, 8)
        for (long task = 0; task < (long)T * heads; task++) {
            int r = (int)(task / heads), h = (int)(task % heads);
            if (only && !only[r]) continue;
            int s = row_seq[r], start = seq_off[s], L = seq_off[s + 1] - start, i = r - start;
            int lo = 0, hi = L - 1;
            if (window >= 0) { lo = i - window < 0 ? 0 : i - window; hi = i + window > L - 1 ? L - 1 : i + window; }
            const float *q = qkv + (size_t)r * 3 * d + (size_t)h * hd;
            float mx = -INFINITY;
            for (int j = lo; j <= hi; j++) {
                const float *k = qkv + (size_t)(start + j) * 3 * d + d + (size_t)h * hd;
                float v = dot_f32_lanes(q, k, hd) * scale;
                score[j - lo] = v;
                if (v > mx) mx = v;
            }
            double sum = 0;
            for (int j = lo; j <= hi; j++) { float e = expf(score[j - lo] - mx); score[j - lo] = e; sum += e; }
            float inv = (float)(1.0 / sum);
            for (int t = 0; t < hd; t++) acc[t] = 0.f;
            for (int j = lo; j <= hi; j++) {
                const float *v = qkv + (size_t)(start + j) * 3 * d + 2 * d + (size_t)h * hd;
                float p = score[j - lo] * inv;
                for (int t = 0; t < hd; t++) acc[t] += p * v[t];
            }
            memcpy(out + (size_t)r * d + (size_t)h * hd, acc, (size_t)hd * sizeof(float));
        }
        free(score); free(acc);
    }
}

typedef struct {
    double *logits;      /* [n_markers] */
    double actions[DECIDE_MAX_ACTIONS];
} SeqOut;

/* DecisionModel.forward over a batch of sequences. */
static void forward(Laya *M, Seq *seqs, int S, SeqOut *outs)
{
    int d = M->d, I = M->inter;
    int *seq_off = (int *)xmalloc((size_t)(S + 1) * sizeof(int));
    seq_off[0] = 0;
    for (int s = 0; s < S; s++) seq_off[s + 1] = seq_off[s] + seqs[s].n;
    int T = seq_off[S];
    int *row_seq = (int *)xmalloc((size_t)T * sizeof(int));
    int *row_pos = (int *)xmalloc((size_t)T * sizeof(int));
    for (int s = 0; s < S; s++)
        for (int i = 0; i < seqs[s].n; i++) { row_seq[seq_off[s] + i] = s; row_pos[seq_off[s] + i] = i; }

    float *x = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *h = (float *)xmalloc((size_t)T * d * sizeof(float));
    float *att = (float *)xmalloc((size_t)T * d * sizeof(float));
    int wide = 3 * d;                    /* the widest activation: qkv, the GeGLU input or the head FFN */
    if (2 * I > wide) wide = 2 * I;
    if (4 * d > wide) wide = 4 * d;
    float *big = (float *)xmalloc((size_t)T * wide * sizeof(float));
    float *mid = (float *)xmalloc((size_t)T * (I > d ? I : d) * sizeof(float));

    /* embeddings: tok_embeddings, then LayerNorm */
    for (int r = 0; r < T; r++) {
        int id = seqs[row_seq[r]].ids[row_pos[r]];
        if (id < 0 || id >= M->vocab) id = 0;
        memcpy(h + (size_t)r * d, M->tok_emb + (size_t)id * d, (size_t)d * sizeof(float));
    }
    layernorm(x, h, M->emb_norm_w, M->emb_norm_b, T, d, M->eps);

    for (int l = 0; l < M->layers; l++) {
        EncLayer *E = &M->L[l];
        const float *in = x;
        if (E->attn_norm_w) { layernorm(h, x, E->attn_norm_w, E->attn_norm_b, T, d, M->eps); in = h; }
        gemm(big, in, T, &E->wqkv, E->bqkv);
        apply_rope(big, T, row_pos, d, M->heads, M->hd, E->global ? M->rope_g : M->rope_l, M->max_pos);
        attention(big, att, T, d, M->heads, M->hd, row_seq, seq_off, E->global ? -1 : M->window, NULL);
        gemm(h, att, T, &E->wo, E->bo);
        add_rows(x, h, (size_t)T * d);
        layernorm(h, x, E->mlp_norm_w, E->mlp_norm_b, T, d, M->eps);
        gemm(big, h, T, &E->wi, E->bi);
        #pragma omp parallel for schedule(static)
        for (int r = 0; r < T; r++) {
            const float *u = big + (size_t)r * 2 * I;
            float *g = mid + (size_t)r * I;
            for (int j = 0; j < I; j++) g[j] = gelu(u[j], M->gelu_tanh) * u[I + j];
        }
        gemm(h, mid, T, &E->wo2, E->bo2);
        add_rows(x, h, (size_t)T * d);
    }
    layernorm(h, x, M->final_norm_w, M->final_norm_b, T, d, M->eps);
    float *t = x; x = h; h = t;

    /* the decision head: + type_emb, then norm_first TransformerEncoderLayers */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < T; r++) {
        const float *te = M->type_emb + (size_t)seqs[row_seq[r]].qtype * d;
        float *xr = x + (size_t)r * d;
        for (int i = 0; i < d; i++) xr[i] += te[i];
    }
    int hh = M->head_heads, hhd = d / hh;
    for (int l = 0; l < M->head_layers; l++) {
        HeadLayer *H = &M->H[l];
        layernorm(h, x, H->n1w, H->n1b, T, d, 1e-5f);
        gemm(big, h, T, &H->in_proj, H->in_proj_b);
        attention(big, att, T, d, hh, hhd, row_seq, seq_off, -1, NULL);
        gemm(h, att, T, &H->out_proj, H->out_proj_b);
        add_rows(x, h, (size_t)T * d);
        layernorm(h, x, H->n2w, H->n2b, T, d, 1e-5f);
        gemm(big, h, T, &H->lin1, H->lin1_b);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < (size_t)T * 4 * d; i++) if (big[i] < 0.f) big[i] = 0.f;
        gemm(h, big, T, &H->lin2, H->lin2_b);
        add_rows(x, h, (size_t)T * d);
    }

    /* scorer at the markers: LayerNorm -> Linear -> GELU -> Linear(d, 1) */
    int nm = 0;
    for (int s = 0; s < S; s++) nm += seqs[s].n_markers;
    float *m = (float *)xmalloc((size_t)(nm > 0 ? nm : 1) * d * sizeof(float));
    float *mn = (float *)xmalloc((size_t)(nm > 0 ? nm : 1) * d * sizeof(float));
    float *m1 = (float *)xmalloc((size_t)(nm > 0 ? nm : 1) * d * sizeof(float));
    for (int s = 0, k = 0; s < S; s++)
        for (int i = 0; i < seqs[s].n_markers; i++, k++)
            memcpy(m + (size_t)k * d, x + (size_t)(seq_off[s] + seqs[s].markers[i]) * d, (size_t)d * sizeof(float));
    layernorm(mn, m, M->sc_ln_w, M->sc_ln_b, nm, d, 1e-5f);
    gemm(m1, mn, nm, &M->sc1, M->sc1_b);
    for (int s = 0, k = 0; s < S; s++) {
        int K = seqs[s].n_markers;
        outs[s].logits = (double *)xcalloc((size_t)(K > 0 ? K : 1), sizeof(double));
        for (int i = 0; i < K; i++, k++) {
            float *row = m1 + (size_t)k * d;
            for (int j = 0; j < d; j++) row[j] = gelu(row[j], 0);
            outs[s].logits[i] = (double)(dot_f32_lanes(row, M->sc2_w, d) + M->sc2_b);
        }
    }

    /* act head: [pooled h[0], top1, top1 - top2, normalized entropy, k / 255],
     * the distribution being the raw (untempered) softmax of the logits. */
    float *feat = (float *)xmalloc((size_t)S * (d + 4) * sizeof(float));
    float *hid = (float *)xmalloc((size_t)S * M->act_hidden * sizeof(float));
    for (int s = 0; s < S; s++) {
        int K = seqs[s].n_markers;
        float *f = feat + (size_t)s * (d + 4);
        memcpy(f, x + (size_t)seq_off[s] * d, (size_t)d * sizeof(float));
        float mx = -INFINITY;
        for (int i = 0; i < K; i++) if ((float)outs[s].logits[i] > mx) mx = (float)outs[s].logits[i];
        float *p = (float *)xmalloc((size_t)(K > 0 ? K : 1) * sizeof(float));
        float sum = 0.f;
        for (int i = 0; i < K; i++) { p[i] = expf((float)outs[s].logits[i] - mx); sum += p[i]; }
        float top1 = 0.f, top2 = 0.f, ent = 0.f;
        for (int i = 0; i < K; i++) {
            p[i] /= sum;
            if (p[i] > top1) { top2 = top1; top1 = p[i]; } else if (p[i] > top2) top2 = p[i];
            ent -= p[i] * logf(p[i] > 1e-9f ? p[i] : 1e-9f);
        }
        float kk = (float)(K < 2 ? 2 : K);
        f[d] = top1; f[d + 1] = top1 - top2; f[d + 2] = ent / logf(kk); f[d + 3] = kk / 255.0f;
        free(p);
    }
    gemm(hid, feat, S, &M->act1, M->act1_b);
    for (int s = 0; s < S; s++) {
        float *row = hid + (size_t)s * M->act_hidden;
        for (int j = 0; j < M->act_hidden; j++) row[j] = gelu(row[j], 0);
        double z[DECIDE_MAX_ACTIONS], mx = -INFINITY, sum = 0;
        for (int a = 0; a < M->n_act; a++) {
            z[a] = (double)(dot_f32_lanes(row, M->act2_w + (size_t)a * M->act_hidden, M->act_hidden) + M->act2_b[a]);
            if (z[a] > mx) mx = z[a];
        }
        for (int a = 0; a < M->n_act; a++) { z[a] = exp(z[a] - mx); sum += z[a]; }
        for (int a = 0; a < M->n_act; a++) outs[s].actions[a] = z[a] / sum;
    }

    free(feat); free(hid); free(m); free(mn); free(m1);
    free(x); free(h); free(att); free(big); free(mid);
    free(seq_off); free(row_seq); free(row_pos);
}

/* ------------------------------------------------------------ calibration */

/* common.temp_bucket */
static double temperature_for(Laya *M, int type, int k)
{
    const char *size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
    char key[32];
    snprintf(key, sizeof(key), "%s:%s", decide_type_name(type), size);
    for (int i = 0; i < M->n_buckets; i++) if (!strcmp(M->bucket_key[i], key)) return M->bucket_t[i];
    return M->temperature[type];
}

/* ------------------------------------------------------- the DECIDE call */

typedef struct {
    Laya *M;
    int dump_ids;
    Seq *last;           /* the sequences of the last record, for --records --ids */
    int last_n;
} LayaCtx;

/* Most rows one forward takes before the batch is split: the activations of a
 * batch are about 12 * hidden floats per row. */
#define LAYA_MAX_BATCH_ROWS 16384

static int laya_decide(void *opaque, const DecideRecord *rec, DecideAnswer *answers,
                       int *input_tokens, char *err, size_t cap)
{
    LayaCtx *ctx = (LayaCtx *)opaque;
    Laya *M = ctx->M;
    for (int i = 0; i < ctx->last_n; i++) seq_free(&ctx->last[i]);
    free(ctx->last); ctx->last = NULL; ctx->last_n = 0;

    char *state = replace_all(rec->state, M->mask_str);
    int n_state = 0;
    int *state_ids = laya_encode(M, state, strlen(state), &n_state);
    free(state);
    /* Agent._encode_state: a list (a conversation) keeps its newest turns */
    int truncate_left = rec->state_type == DECIDE_STATE_ARRAY;
    int Q = rec->n_questions;
    Seq *seqs = (Seq *)xcalloc((size_t)Q, sizeof(Seq));
    for (int q = 0; q < Q; q++) {
        if (!build_sequence(M, &rec->questions[q], state_ids, n_state, truncate_left, &seqs[q], err, cap)) {
            for (int i = 0; i <= q; i++) seq_free(&seqs[i]);
            free(seqs); free(state_ids);
            return 0;
        }
    }
    free(state_ids);
    SeqOut *outs = (SeqOut *)xcalloc((size_t)Q, sizeof(SeqOut));
    for (int start = 0; start < Q;) {
        int end = start, rows = 0;
        while (end < Q && (end == start || rows + seqs[end].n <= LAYA_MAX_BATCH_ROWS)) rows += seqs[end++].n;
        forward(M, seqs + start, end - start, outs + start);
        start = end;
    }
    int total = 0;
    for (int q = 0; q < Q; q++) {
        DecideAnswer *a = &answers[q];
        int K = seqs[q].n_markers;
        a->n = K;
        a->logits = (double *)xcalloc((size_t)K, sizeof(double));
        a->probs = (double *)xcalloc((size_t)K, sizeof(double));
        a->temperature = temperature_for(M, seqs[q].qtype, K);
        double mx = -INFINITY, sum = 0;
        for (int i = 0; i < K; i++) {
            a->logits[i] = outs[q].logits[i];
            double z = (double)(float)outs[q].logits[i] / a->temperature;
            a->probs[i] = z;
            if (z > mx) mx = z;
        }
        for (int i = 0; i < K; i++) { a->probs[i] = exp(a->probs[i] - mx); sum += a->probs[i]; }
        for (int i = 0; i < K; i++) a->probs[i] /= sum;
        a->n_actions = M->n_act;
        for (int i = 0; i < M->n_act; i++) { a->action_names[i] = M->action_names[i]; a->actions[i] = outs[q].actions[i]; }
        a->tokens = seqs[q].n;
        a->state_tokens = seqs[q].state_tokens;
        a->state_dropped = seqs[q].state_tokens - seqs[q].state_used;
        total += seqs[q].n;
        free(outs[q].logits);
    }
    free(outs);
    *input_tokens = total;
    if (ctx->dump_ids) { ctx->last = seqs; ctx->last_n = Q; }
    else { for (int q = 0; q < Q; q++) seq_free(&seqs[q]); free(seqs); }
    return 1;
}

/* ------------------------------------------------------------------- main */

static double rss_gb(void)
{
#ifdef __linux__
    FILE *f = fopen("/proc/self/statm", "r");
    long pages = 0, resident = 0;
    if (f) { if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0; fclose(f); }
    return resident * (double)sysconf(_SC_PAGESIZE) / 1e9;
#else
    return 0.0;
#endif
}

static void print_json_string(FILE *out, const char *s)
{
    DecideBuf b = {0};
    decide_buf_string(&b, s);
    fputs(b.data ? b.data : "\"\"", out);
    free(b.data);
}

static int run_records(Laya *M, const char *path, int dump_ids)
{
    char *text = read_text(path);
    if (!text) die("cannot read %s", path);
    jval *root = json_parse_checked(text);
    if (!root || root->t != J_ARR) die("%s: expected a JSON array of records", path);
    LayaCtx ctx = {M, dump_ids, NULL, 0};
    /* Each record is the text of a DECIDE payload and goes through
     * decide_record_parse, the path the serve loop takes. */
    for (int i = 0; i < root->len; i++) {
        jval *r = root->kids[i];
        jval *payload = json_get(r, "payload");
        if (!payload || payload->t != J_STR) die("%s: record %d has no string \"payload\"", path, i);
        DecideRecord rec;
        char reason[1024];
        if (!decide_record_parse(payload->str, &rec, reason, sizeof(reason))) {
            printf("{\"error\":"); print_json_string(stdout, reason); printf("}\n");
            continue;
        }
        DecideAnswer *answers = (DecideAnswer *)xcalloc((size_t)rec.n_questions, sizeof(DecideAnswer));
        int tokens = 0;
        double t0 = now_ms();
        int ok = laya_decide(&ctx, &rec, answers, &tokens, reason, sizeof(reason));
        double ms = now_ms() - t0;
        if (!ok) {
            printf("{\"error\":"); print_json_string(stdout, reason); printf("}\n");
        } else {
            size_t len = 0;
            char *json = decide_answers_json(&rec, answers, tokens, ms, &len);
            if (dump_ids && json && len > 1) {
                json[len - 1] = 0;                 /* reopen the object */
                fputs(json, stdout);
                printf(",\"sequences\":[");
                for (int q = 0; q < ctx.last_n; q++) {
                    printf("%s{\"ids\":[", q ? "," : "");
                    for (int t = 0; t < ctx.last[q].n; t++) printf("%s%d", t ? "," : "", ctx.last[q].ids[t]);
                    printf("],\"markers\":[");
                    for (int t = 0; t < ctx.last[q].n_markers; t++) printf("%s%d", t ? "," : "", ctx.last[q].markers[t]);
                    printf("]}");
                }
                printf("]}\n");
            } else if (json) {
                fputs(json, stdout); fputc('\n', stdout);
            }
            free(json);
        }
        fflush(stdout);
        decide_answers_free(answers, rec.n_questions);
        free(answers);
        decide_record_free(&rec);
    }
    json_free(root);
    free(text);
    return 0;
}

static int run_tokenize(Laya *M, const char *path)
{
    char *text = read_text(path);
    if (!text) die("cannot read %s", path);
    jval *root = json_parse_checked(text);
    if (!root || root->t != J_ARR) die("%s: expected a JSON array of strings", path);
    printf("[");
    for (int i = 0; i < root->len; i++) {
        jval *s = root->kids[i];
        if (s->t != J_STR) die("%s: entry %d is not a string", path, i);
        int n = 0;
        int *ids = laya_encode(M, s->str, strlen(s->str), &n);
        printf("%s[", i ? "," : "");
        for (int t = 0; t < n; t++) printf("%s%d", t ? "," : "", ids[t]);
        printf("]");
        free(ids);
    }
    printf("]\n");
    json_free(root);
    free(text);
    return 0;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *records = NULL, *tokenize = NULL;
    int dump_ids = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--records") && i + 1 < argc) records = argv[++i];
        else if (!strcmp(argv[i], "--tokenize") && i + 1 < argc) tokenize = argv[++i];
        else if (!strcmp(argv[i], "--ids")) dump_ids = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("usage: SERVE=1 SNAP=<dir> laya            serve DECIDE on stdin/stdout\n"
                   "       laya --model DIR --records FILE [--ids]\n"
                   "       laya --model DIR --tokenize FILE\n");
            return 0;
        }
        /* a bare number is the gateway's cache-cap argument: nothing to cache here */
    }
    const char *serve = getenv("SERVE");
    int serving = serve && !strcmp(serve, "1") && !records && !tokenize;
    if (!model) model = getenv("SNAP");
    if (!model) die("no model: pass --model DIR or set SNAP");
    if (serving) coli_serve_stdio_init();
    coli_omp_tune_threads("laya");          /* physical cores: SMT halves the GEMMs here too */
    double t0 = now_ms();
    Laya *M = (Laya *)xcalloc(1, sizeof(Laya));
    load_model(M, model);
    fprintf(stderr, "[laya] %s: ModernBERT %d layers x %d, %d heads; head %d layers; max_len %d "
            "(head %d); loaded in %.1f s%s\n", M->model_name, M->layers, M->d, M->heads,
            M->head_layers, M->max_len, M->head_max_len, (now_ms() - t0) / 1e3,
            getenv("COLI_LAYA_INT8") && !strcmp(getenv("COLI_LAYA_INT8"), "1") ? ", int8 weights" : "");
    if (tokenize) return run_tokenize(M, tokenize);
    if (records) return run_records(M, records, dump_ids);
    if (!serving) die("nothing to do: set SERVE=1, or pass --records / --tokenize");
    int slots = getenv("KV_SLOTS") ? atoi(getenv("KV_SLOTS")) : 1;
    if (slots < 1) slots = 1;
    char caps[256];
    snprintf(caps, sizeof(caps), "decide=1 chat=0 max_len=%d head_max_len=%d", M->max_len, M->head_max_len);
    LayaCtx ctx = {M, 0, NULL, 0};
    return decide_serve(stdin, stdout, laya_decide, &ctx, caps, rss_gb(), slots);
}
