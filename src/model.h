/* Nemotron 3.5 ASR streaming model: config, weights in GPU memory, common kernel launchers. */
#ifndef NEMO_MODEL_H
#define NEMO_MODEL_H
#include "gpu.h"
#include "json.h"
#include "safetensors.h"
#include "tokenizer.h"
#include <stdint.h>

typedef struct {
    int n_layers, d_model, n_heads, head_dim, d_ff, subsampling, sub_channels, conv_kernel, feat_in;
    int num_prompts, prompt_hidden;
    int pred_hidden, pred_layers, vocab_size, blank_id, joint_hidden, num_classes, max_symbols;
    int att_left, att_right_default;
    int trained_right[8], n_trained;
    float ln_eps;
} model_config_t;

typedef struct { gpu_buf_t *buf; size_t off; } wt_t; /* bf16 weight location inside the arena */

typedef struct {
    wt_t ff1_w1, ff1_w2, ff2_w1, ff2_w2, wq, wk, wv, wo, wpos, pw1, pw2;
    gpu_buf_t *ln_ff1_g, *ln_ff1_b, *ln_att_g, *ln_att_b, *ln_conv_g, *ln_conv_b, *ln_ff2_g, *ln_ff2_b, *ln_out_g, *ln_out_b;
    gpu_buf_t *bn_g, *bn_b;   /* conv module LayerNorm (named batch_norm in the checkpoint) */
    gpu_buf_t *dw_w;          /* f32 [D][9] */
    gpu_buf_t *bias_u, *bias_v;
    gpu_buf_t *ptab;          /* f32 [2*Lmax-1][D]: linear_pos(pos_emb) for the largest window */
} layer_w_t;

typedef struct model {
    model_config_t cfg;
    json_value *config;
    const json_value *prompt_dict;
    vocab_t vocab;
    st_file st;
    gpu_t *gpu;
    gpu_buf_t *arena;
    size_t arena_used;
    /* subsampling */
    gpu_buf_t *conv0_w, *conv0_b, *conv2_w, *conv2_b, *conv5_w, *conv5_b;
    wt_t conv3_w, conv6_w;
    gpu_buf_t *conv3_b, *conv6_b;
    wt_t pre_out_w;
    gpu_buf_t *pre_out_b;
    int sub_freq_out; /* 17 */
    layer_w_t *layers;
    int Lmax;         /* att_left + largest trained chunk */
    /* prompt */
    wt_t prompt_w0, prompt_w2;
    gpu_buf_t *prompt_b0_eff, *prompt_b2;
    int prompt_index;
    /* decoder / joint */
    wt_t lstm_wx[2], lstm_wh[2];
    gpu_buf_t *lstm_b[2];
    uint16_t *embed_bf16;      /* heap copy [vocab_size+1][pred_hidden], read per emitted token */
    uint16_t *prompt_w0_cpu, *prompt_b0_cpu; /* heap copies for model_set_language */
    wt_t joint_enc_w, joint_pred_w, joint_out_w;
    gpu_buf_t *joint_enc_b, *joint_pred_b, *joint_out_b;
    double load_ms;
} model_t;

model_t *model_load(const char *dir, char *err, size_t errlen);
void model_free(model_t *m);
int model_prompt_index(const model_t *m, const char *lang); /* -1 if unknown */
int model_set_language(model_t *m, const char *lang, char *err, size_t errlen);
int model_right_context_trained(const model_t *m, int right);

/* kernel launchers (must be between gpu_begin / gpu_end) */
void k_gemm(model_t *m, gpu_buf_t *A, size_t a_off, uint32_t lda, wt_t W, uint32_t ldw, gpu_buf_t *bias,
            gpu_buf_t *C, size_t c_off, uint32_t ldc, uint32_t M, uint32_t N, uint32_t K, int act, int accumulate, float alpha);
void k_layernorm(model_t *m, gpu_buf_t *X, size_t x_off, gpu_buf_t *g, gpu_buf_t *b, gpu_buf_t *Y, size_t y_off, uint32_t M, uint32_t D);
void k_copy(model_t *m, gpu_buf_t *src, size_t s_off, gpu_buf_t *dst, size_t d_off, uint32_t n);
void k_fill(model_t *m, gpu_buf_t *dst, size_t d_off, uint32_t n, float v);

#endif
