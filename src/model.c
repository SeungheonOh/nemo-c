#include "model.h"
#include "kernel_params.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Profiling aid: NEMO_SKIP bitmask disables kernel categories (output becomes garbage). */
unsigned nemo_skip_mask = 0;
static int skip_init = 0;
static void skip_setup(void) {
    if (skip_init) return;
    skip_init = 1;
    const char *e = getenv("NEMO_SKIP");
    if (e) nemo_skip_mask = (unsigned)strtoul(e, NULL, 0);
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(buf); return NULL; }
    fclose(f);
    buf[n] = 0;
    *len = (size_t)n;
    return buf;
}

/* ---- weight helpers ------------------------------------------------------------- */
static const st_tensor *need(model_t *m, const char *name, size_t numel, char *err, size_t errlen) {
    const st_tensor *t = st_find(&m->st, name);
    if (!t) { snprintf(err, errlen, "missing tensor %s", name); return NULL; }
    if (t->dtype != ST_BF16) { snprintf(err, errlen, "tensor %s is not bf16", name); return NULL; }
    if (numel && t->numel != numel) { snprintf(err, errlen, "tensor %s has %zu elements, expected %zu", name, t->numel, numel); return NULL; }
    return t;
}
static int get_bf16(model_t *m, const char *name, size_t numel, wt_t *out, char *err, size_t errlen) {
    const st_tensor *t = need(m, name, numel, err, errlen);
    if (!t) return -1;
    size_t off = (m->arena_used + 63) & ~(size_t)63;
    if (off + t->nbytes > gpu_buf_len(m->arena)) { snprintf(err, errlen, "arena overflow"); return -1; }
    memcpy((char *)gpu_buf_ptr(m->arena) + off, st_data(&m->st, t), t->nbytes);
    m->arena_used = off + t->nbytes;
    out->buf = m->arena;
    out->off = off;
    return 0;
}
static gpu_buf_t *get_f32(model_t *m, const char *name, size_t numel, char *err, size_t errlen) {
    const st_tensor *t = need(m, name, numel, err, errlen);
    if (!t) return NULL;
    gpu_buf_t *b = gpu_buf_alloc(m->gpu, t->numel * sizeof(float));
    const uint16_t *src = st_data(&m->st, t);
    float *dst = gpu_buf_ptr(b);
    for (size_t i = 0; i < t->numel; ++i) dst[i] = bf16_to_f32(src[i]);
    return b;
}

/* ---- kernel launchers ----------------------------------------------------------- */
void k_gemm(model_t *m, gpu_buf_t *A, size_t a_off, uint32_t lda, wt_t W, uint32_t ldw, gpu_buf_t *bias,
            gpu_buf_t *C, size_t c_off, uint32_t ldc, uint32_t M, uint32_t N, uint32_t K, int act, int accumulate, float alpha) {
    skip_setup();
    if (nemo_skip_mask & 1) return;
    GemmParams p = { M, N, K, lda, ldw, ldc, bias != NULL, (uint32_t)act, (uint32_t)accumulate, alpha };
    gpu_arg_t args[5] = { GPU_BUF(A, a_off), GPU_BUF(W.buf, W.off), GPU_BUF(bias ? bias : A, 0), GPU_BUF(C, c_off), GPU_BYTES(&p) };
    /* Kernel policy from tools/kbench.c against DRAM-resident weights:
         M <= 4          : gemm_spec, 2 columns per SIMD group (pure weight streaming, ~430 GB/s)
         4 < M <= 16     : gemm_mma, split-K over 4 SIMD groups, 8x8 matrix units (1.3-2.5x faster than spec)
         otherwise       : generic gemm_bf16 (subsampling 1x1 convs, position table) */
    const uint32_t nout = act == 3 ? N / 2 : N; /* GLU halves the output width */
    if (M <= 4 || (M <= GEMM_ROWS && (K % 256) != 0)) {
        uint32_t cols = 2;
        uint32_t fc[3] = { M, cols, 2 };
        uint32_t cols_per_tg = act == 3 ? GEMM_SIMDS : GEMM_SIMDS * cols; /* GLU: one output column per SIMD group */
        gpu_dispatch_groups_fc(m->gpu, "gemm_spec", fc, 3, args, 5, (nout + cols_per_tg - 1) / cols_per_tg, 1, 1, GEMM_SIMDS * 32, 1, 1);
    } else if ((K % 256) == 0) {
        /* 2 column tiles x 8 split-K groups x 32-wide K steps: best or within a few percent of best on
           every shape at 7 and 14 rows in tools/kbench.c (min of 5 runs, DRAM-resident weights).
           M > 16 (subsampling 1x1 convs, position table) runs the same kernel over 16-row blocks;
           the caller guarantees A has its rows padded to a multiple of 16. */
        uint32_t ct = 2, split = 8, kstep = 32;
        uint32_t tm = M > 8 ? 2u : 1u;
        uint32_t fc[7] = { M, 0, 0, tm, ct, split, kstep };
        uint32_t cols_per_tg = act == 3 ? 8 : 8 * ct;
        gpu_dispatch_groups_fc(m->gpu, "gemm_mma", fc, 7, args, 5, (nout + cols_per_tg - 1) / cols_per_tg, (M + 8 * tm - 1) / (8 * tm), 1, 32 * split, 1, 1);
    } else {
        gpu_dispatch_groups(m->gpu, "gemm_bf16", args, 5, (N + GEMM_SIMDS - 1) / GEMM_SIMDS, 1, 1, GEMM_SIMDS * 32, 1, 1);
    }
}
void k_layernorm2(model_t *m, gpu_buf_t *X, gpu_buf_t *g1, gpu_buf_t *b1, gpu_buf_t *g2, gpu_buf_t *b2, gpu_buf_t *Y1, gpu_buf_t *Y2, uint32_t M, uint32_t D) {
    if (nemo_skip_mask & 4) return;
    LnParams p = { M, D, m->cfg.ln_eps };
    gpu_arg_t args[8] = { GPU_BUF(X, 0), GPU_BUF(g1, 0), GPU_BUF(b1, 0), GPU_BUF(g2, 0), GPU_BUF(b2, 0), GPU_BUF(Y1, 0), GPU_BUF(Y2, 0), GPU_BYTES(&p) };
    gpu_dispatch_groups(m->gpu, "layernorm2", args, 8, M, 1, 1, LN_THREADS, 1, 1);
}
void k_layernorm(model_t *m, gpu_buf_t *X, size_t x_off, gpu_buf_t *g, gpu_buf_t *b, gpu_buf_t *Y, size_t y_off, uint32_t M, uint32_t D) {
    if (nemo_skip_mask & 4) return;
    LnParams p = { M, D, m->cfg.ln_eps };
    gpu_arg_t args[5] = { GPU_BUF(X, x_off), GPU_BUF(g, 0), GPU_BUF(b, 0), GPU_BUF(Y, y_off), GPU_BYTES(&p) };
    gpu_dispatch_groups(m->gpu, "layernorm", args, 5, M, 1, 1, LN_THREADS, 1, 1);
}
void k_copy(model_t *m, gpu_buf_t *src, size_t s_off, gpu_buf_t *dst, size_t d_off, uint32_t n) {
    if (nemo_skip_mask & 8) return;
    CountParams p = { n };
    gpu_arg_t args[3] = { GPU_BUF(src, s_off), GPU_BUF(dst, d_off), GPU_BYTES(&p) };
    gpu_dispatch(m->gpu, "copy_f32", args, 3, n, 1, 1, n < 256 ? n : 256, 1, 1);
}
void k_fill(model_t *m, gpu_buf_t *dst, size_t d_off, uint32_t n, float v) {
    FillParams p = { n, v };
    gpu_arg_t args[2] = { GPU_BUF(dst, d_off), GPU_BYTES(&p) };
    gpu_dispatch(m->gpu, "fill_f32", args, 2, n, 1, 1, n < 256 ? n : 256, 1, 1);
}

/* ---- config --------------------------------------------------------------------- */
static int parse_config(model_t *m, char *err, size_t errlen) {
    const json_value *c = m->config;
    const json_value *enc = json_get(c, "encoder"), *pre = json_get(c, "preprocessor"), *pr = json_get(c, "prompt");
    const json_value *dec = json_get(c, "decoder"), *jo = json_get(c, "joint");
    if (!enc || !pre || !pr || !dec || !jo) { snprintf(err, errlen, "config.json missing sections"); return -1; }
    model_config_t *g = &m->cfg;
    g->feat_in = (int)json_num(json_get(enc, "feat_in"), 128);
    g->n_layers = (int)json_num(json_get(enc, "n_layers"), 24);
    g->d_model = (int)json_num(json_get(enc, "d_model"), 1024);
    g->n_heads = (int)json_num(json_get(enc, "n_heads"), 8);
    g->head_dim = g->d_model / g->n_heads;
    g->d_ff = g->d_model * (int)json_num(json_get(enc, "ff_expansion_factor"), 4);
    g->subsampling = (int)json_num(json_get(enc, "subsampling_factor"), 8);
    g->sub_channels = (int)json_num(json_get(enc, "subsampling_conv_channels"), 256);
    g->conv_kernel = (int)json_num(json_get(enc, "conv_kernel_size"), 9);
    g->num_prompts = (int)json_num(json_get(pr, "num_prompts"), 128);
    g->prompt_hidden = (int)json_num(json_get(pr, "prompt_hidden"), 2048);
    g->pred_hidden = (int)json_num(json_get(dec, "pred_hidden"), 640);
    g->pred_layers = (int)json_num(json_get(dec, "pred_rnn_layers"), 2);
    g->vocab_size = (int)json_num(json_get(dec, "vocab_size"), 13087);
    g->blank_id = g->vocab_size;
    g->joint_hidden = (int)json_num(json_get(jo, "joint_hidden"), 640);
    g->num_classes = (int)json_num(json_get(jo, "num_classes"), 13087) + 1;
    g->max_symbols = (int)json_num(json_get(c, "max_symbols"), 10);
    g->ln_eps = 1e-5f;
    const json_value *acs = json_get(enc, "att_context_size");
    g->n_trained = 0;
    for (size_t i = 0; i < json_len(acs) && i < 8; ++i) {
        const json_value *pair = json_at(acs, i);
        g->att_left = (int)json_num(json_at(pair, 0), 56);
        g->trained_right[g->n_trained++] = (int)json_num(json_at(pair, 1), 13);
    }
    const json_value *dflt = json_get(c, "default_att_context_size");
    g->att_right_default = (int)json_num(json_at(dflt, 1), 13);
    if (g->pred_layers != 2) { snprintf(err, errlen, "only 2 LSTM layers supported"); return -1; }
    if (g->head_dim != 128 || g->joint_hidden > JOINT_MAX_H || g->conv_kernel != 9 || g->subsampling != 8 || g->d_model % 4) {
        snprintf(err, errlen, "unexpected architecture dims"); return -1;
    }
    if ((int)json_num(json_get(pre, "sample_rate"), 16000) != 16000 || (int)json_num(json_get(pre, "features"), 128) != 128) {
        snprintf(err, errlen, "unexpected preprocessor config"); return -1;
    }
    int maxr = 0;
    for (int i = 0; i < g->n_trained; ++i) if (g->trained_right[i] > maxr) maxr = g->trained_right[i];
    m->Lmax = g->att_left + maxr + 1 + 2; /* +2: a final boundary window can yield up to chunk+2 rows */
    m->prompt_dict = json_get(pr, "prompt_dictionary");
    if (vocab_from_json(json_get(c, "vocabulary"), &m->vocab)) { snprintf(err, errlen, "vocabulary missing"); return -1; }
    return 0;
}
int model_right_context_trained(const model_t *m, int right) {
    for (int i = 0; i < m->cfg.n_trained; ++i) if (m->cfg.trained_right[i] == right) return 1;
    return 0;
}
int model_prompt_index(const model_t *m, const char *lang) {
    const json_value *v = json_get(m->prompt_dict, lang);
    return v ? (int)json_num(v, -1) : -1;
}

/* ---- load ------------------------------------------------------------------------ */
extern const char kernels_metal_src[];

model_t *model_load(const char *dir, char *err, size_t errlen) {
    double t0 = now_ms();
    model_t *m = calloc(1, sizeof *m);
    char path[2048];
    size_t len;
    snprintf(path, sizeof path, "%s/config.json", dir);
    char *cfg = read_file(path, &len);
    if (!cfg) { snprintf(err, errlen, "cannot read %s", path); model_free(m); return NULL; }
    char jerr[256];
    m->config = json_parse(cfg, len, jerr, sizeof jerr);
    free(cfg);
    if (!m->config) { snprintf(err, errlen, "config.json: %s", jerr); model_free(m); return NULL; }
    if (parse_config(m, err, errlen)) { model_free(m); return NULL; }

    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    if (st_open(path, &m->st, err, errlen)) { model_free(m); return NULL; }

    m->gpu = gpu_create(kernels_metal_src, err, errlen);
    if (!m->gpu) { model_free(m); return NULL; }

    size_t arena_bytes = 0;
    for (size_t i = 0; i < m->st.count; ++i) arena_bytes += m->st.tensors[i].nbytes + 64;
    m->arena = gpu_buf_alloc(m->gpu, arena_bytes);
    if (!m->arena) { snprintf(err, errlen, "cannot allocate %zu MB weight arena", arena_bytes >> 20); model_free(m); return NULL; }

    const model_config_t *g = &m->cfg;
    const size_t D = (size_t)g->d_model, CH = (size_t)g->sub_channels;
#define BF(name, numel, dst) if (get_bf16(m, name, numel, &(dst), err, errlen)) { model_free(m); return NULL; }
#define F32(name, numel, dst) if (!((dst) = get_f32(m, name, numel, err, errlen))) { model_free(m); return NULL; }

    /* subsampling: causal dw-striding, 3 stride-2 stages */
    F32("encoder.pre_encode.conv.0.weight", CH * 9, m->conv0_w);
    F32("encoder.pre_encode.conv.0.bias", CH, m->conv0_b);
    F32("encoder.pre_encode.conv.2.weight", CH * 9, m->conv2_w);
    F32("encoder.pre_encode.conv.2.bias", CH, m->conv2_b);
    BF("encoder.pre_encode.conv.3.weight", CH * CH, m->conv3_w);
    F32("encoder.pre_encode.conv.3.bias", CH, m->conv3_b);
    F32("encoder.pre_encode.conv.5.weight", CH * 9, m->conv5_w);
    F32("encoder.pre_encode.conv.5.bias", CH, m->conv5_b);
    BF("encoder.pre_encode.conv.6.weight", CH * CH, m->conv6_w);
    F32("encoder.pre_encode.conv.6.bias", CH, m->conv6_b);
    int freq = g->feat_in;
    for (int i = 0; i < 3; ++i) freq = (freq + 3 - 3) / 2 + 1;
    m->sub_freq_out = freq;
    BF("encoder.pre_encode.out.weight", D * CH * (size_t)freq, m->pre_out_w);
    F32("encoder.pre_encode.out.bias", D, m->pre_out_b);

    m->layers = calloc((size_t)g->n_layers, sizeof *m->layers);
    char n[256];
    for (int l = 0; l < g->n_layers; ++l) {
        layer_w_t *L = &m->layers[l];
#define LN(field, suffix) snprintf(n, sizeof n, "encoder.layers.%d.%s", l, suffix); BF(n, 0, L->field)
#define LF(field, suffix, numel) snprintf(n, sizeof n, "encoder.layers.%d.%s", l, suffix); F32(n, numel, L->field)
        LN(ff1_w1, "feed_forward1.linear1.weight"); LN(ff1_w2, "feed_forward1.linear2.weight");
        LN(ff2_w1, "feed_forward2.linear1.weight"); LN(ff2_w2, "feed_forward2.linear2.weight");
        LN(wq, "self_attn.linear_q.weight"); LN(wk, "self_attn.linear_k.weight"); LN(wv, "self_attn.linear_v.weight");
        if (L->wk.off != L->wq.off + D * D * 2 || L->wv.off != L->wk.off + D * D * 2) { snprintf(err, errlen, "qkv weights not contiguous in arena"); model_free(m); return NULL; }
        L->wqkv = L->wq;
        LN(wo, "self_attn.linear_out.weight"); LN(wpos, "self_attn.linear_pos.weight");
        LN(pw1, "conv.pointwise_conv1.weight"); LN(pw2, "conv.pointwise_conv2.weight");
        LF(ln_ff1_g, "norm_feed_forward1.weight", D); LF(ln_ff1_b, "norm_feed_forward1.bias", D);
        LF(ln_att_g, "norm_self_att.weight", D); LF(ln_att_b, "norm_self_att.bias", D);
        LF(ln_conv_g, "norm_conv.weight", D); LF(ln_conv_b, "norm_conv.bias", D);
        LF(ln_ff2_g, "norm_feed_forward2.weight", D); LF(ln_ff2_b, "norm_feed_forward2.bias", D);
        LF(ln_out_g, "norm_out.weight", D); LF(ln_out_b, "norm_out.bias", D);
        LF(bn_g, "conv.batch_norm.weight", D); LF(bn_b, "conv.batch_norm.bias", D);
        LF(dw_w, "conv.depthwise_conv.weight", D * 9);
        LF(bias_u, "self_attn.pos_bias_u", D); LF(bias_v, "self_attn.pos_bias_v", D);
#undef LN
#undef LF
    }
    /* prompt kernel */
    BF("prompt_kernel.0.weight", (size_t)g->prompt_hidden * (D + (size_t)g->num_prompts), m->prompt_w0);
    BF("prompt_kernel.2.weight", D * (size_t)g->prompt_hidden, m->prompt_w2);
    F32("prompt_kernel.2.bias", D, m->prompt_b2);
    m->prompt_b0_eff = gpu_buf_alloc(m->gpu, (size_t)g->prompt_hidden * sizeof(float));
    m->prompt_index = -1;
    /* decoder */
    const size_t PH = (size_t)g->pred_hidden;
    for (int l = 0; l < 2; ++l) {
        snprintf(n, sizeof n, "decoder.prediction.dec_rnn.lstm.%d.Wx", l); BF(n, 4 * PH * PH, m->lstm_wx[l]);
        snprintf(n, sizeof n, "decoder.prediction.dec_rnn.lstm.%d.Wh", l); BF(n, 4 * PH * PH, m->lstm_wh[l]);
        snprintf(n, sizeof n, "decoder.prediction.dec_rnn.lstm.%d.bias", l); F32(n, 4 * PH, m->lstm_b[l]);
    }
    {
        const st_tensor *t = need(m, "decoder.prediction.embed.weight", (size_t)(g->vocab_size + 1) * PH, err, errlen);
        if (!t) { model_free(m); return NULL; }
        m->embed_bf16 = malloc(t->nbytes);
        memcpy(m->embed_bf16, st_data(&m->st, t), t->nbytes);
        const st_tensor *w0 = need(m, "prompt_kernel.0.weight", 0, err, errlen), *b0 = need(m, "prompt_kernel.0.bias", (size_t)g->prompt_hidden, err, errlen);
        if (!w0 || !b0) { model_free(m); return NULL; }
        m->prompt_w0_cpu = malloc(w0->nbytes); memcpy(m->prompt_w0_cpu, st_data(&m->st, w0), w0->nbytes);
        m->prompt_b0_cpu = malloc(b0->nbytes); memcpy(m->prompt_b0_cpu, st_data(&m->st, b0), b0->nbytes);
    }
    BF("joint.enc.weight", (size_t)g->joint_hidden * D, m->joint_enc_w); F32("joint.enc.bias", (size_t)g->joint_hidden, m->joint_enc_b);
    BF("joint.pred.weight", (size_t)g->joint_hidden * PH, m->joint_pred_w); F32("joint.pred.bias", (size_t)g->joint_hidden, m->joint_pred_b);
    BF("joint.joint_net.2.weight", (size_t)g->num_classes * (size_t)g->joint_hidden, m->joint_out_w);
    F32("joint.joint_net.2.bias", (size_t)g->num_classes, m->joint_out_b);
#undef BF
#undef F32

    /* relative positional table: pe rows for positions (Lmax-1) .. -(Lmax-1), then per-layer linear_pos */
    {
        int P = 2 * m->Lmax - 1;
        gpu_buf_t *pe = gpu_buf_alloc(m->gpu, (size_t)(P + 16) * D * sizeof(float)); /* +16 rows: gemm_mma reads whole row blocks */
        float *pef = gpu_buf_ptr(pe);
        for (int r = 0; r < P; ++r) {
            double pos = (double)(m->Lmax - 1 - r);
            for (size_t i = 0; i < D / 2; ++i) {
                double div = exp((double)(2 * i) * -(log(10000.0) / (double)D));
                pef[(size_t)r * D + 2 * i] = (float)sin(pos * div);
                pef[(size_t)r * D + 2 * i + 1] = (float)cos(pos * div);
            }
        }
        gpu_begin(m->gpu);
        for (int l = 0; l < g->n_layers; ++l) {
            m->layers[l].ptab = gpu_buf_alloc(m->gpu, (size_t)P * D * sizeof(float));
            k_gemm(m, pe, 0, (uint32_t)D, m->layers[l].wpos, (uint32_t)D, NULL, m->layers[l].ptab, 0, (uint32_t)D, (uint32_t)P, (uint32_t)D, (uint32_t)D, 0, 0, 1.0f);
        }
        if (gpu_end(m->gpu, err, errlen)) { gpu_buf_free(pe); model_free(m); return NULL; }
        gpu_buf_free(pe);
    }
    st_close(&m->st); /* everything lives in GPU memory or heap copies now; drop the 1.3 GB mapping */
    m->load_ms = now_ms() - t0;
    return m;
}

int model_set_language(model_t *m, const char *lang, char *err, size_t errlen) {
    int idx = model_prompt_index(m, lang);
    if (idx < 0) { snprintf(err, errlen, "unknown language prompt %s", lang); return -1; }
    /* one-hot prompt folds into the first prompt layer's bias: b_eff = b0 + W0[:, D + idx] */
    const uint16_t *W = m->prompt_w0_cpu, *B = m->prompt_b0_cpu;
    size_t ld = (size_t)m->cfg.d_model + (size_t)m->cfg.num_prompts;
    float *dst = gpu_buf_ptr(m->prompt_b0_eff);
    for (int n = 0; n < m->cfg.prompt_hidden; ++n) dst[n] = bf16_to_f32(B[n]) + bf16_to_f32(W[(size_t)n * ld + (size_t)m->cfg.d_model + (size_t)idx]);
    m->prompt_index = idx;
    return 0;
}

void model_free(model_t *m) {
    if (!m) return;
    if (m->layers) {
        for (int l = 0; l < m->cfg.n_layers; ++l) {
            layer_w_t *L = &m->layers[l];
            gpu_buf_t *bufs[] = { L->ln_ff1_g, L->ln_ff1_b, L->ln_att_g, L->ln_att_b, L->ln_conv_g, L->ln_conv_b, L->ln_ff2_g, L->ln_ff2_b,
                                  L->ln_out_g, L->ln_out_b, L->bn_g, L->bn_b, L->dw_w, L->bias_u, L->bias_v, L->ptab };
            for (size_t i = 0; i < sizeof bufs / sizeof *bufs; ++i) gpu_buf_free(bufs[i]);
        }
        free(m->layers);
    }
    gpu_buf_t *bufs[] = { m->conv0_w, m->conv0_b, m->conv2_w, m->conv2_b, m->conv5_w, m->conv5_b, m->conv3_b, m->conv6_b, m->pre_out_b,
                          m->prompt_b0_eff, m->prompt_b2, m->lstm_b[0], m->lstm_b[1], m->joint_enc_b, m->joint_pred_b, m->joint_out_b, m->arena };
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; ++i) gpu_buf_free(bufs[i]);
    if (m->gpu) gpu_destroy(m->gpu);
    st_close(&m->st);
    free(m->embed_bf16);
    free(m->prompt_w0_cpu);
    free(m->prompt_b0_cpu);
    vocab_free(&m->vocab);
    json_free(m->config);
    free(m);
}
