#include "encoder.h"
#include "kernel_params.h"
#include "mel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MEL_CACHE 16

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

struct encoder {
    model_t *m;
    int right, chunk_frames, chunk_mel, cmax, Lmax, left;
    /* pending mel queue */
    float *pending;
    size_t pending_n, pending_cap;
    float mel_cache[MEL_CACHE * MEL_NMEL];
    int mel_cache_n;
    size_t consumed, emitted;
    int closed;
    /* subsampling buffers */
    gpu_buf_t *win, *c0, *c1, *c2, *c3, *c4, *flat, *pre_out;
    /* block buffers */
    gpu_buf_t *h, *tmp, *t2, *ff, *att, *y, *ph, *prompted, *joint_enc;
    /* per layer: kv[l] rows of 3*D floats (q | k | v), din[l] rows of D floats (GLU output);
       both append-only, compacted once per epoch so no per-chunk copies are needed */
    gpu_buf_t **kv, **din;
    int cap_rows, pos, cache_len;
    int conv_left;
};

encoder_t *encoder_create(model_t *m, int right) {
    encoder_t *e = calloc(1, sizeof *e);
    e->m = m;
    e->right = right;
    e->chunk_frames = right + 1;
    e->chunk_mel = e->chunk_frames * m->cfg.subsampling;
    e->cmax = e->chunk_frames + 2; /* a final boundary window can yield up to chunk_frames+2 rows */
    if (e->cmax < MMA_MIN_ROWS) e->cmax = MMA_MIN_ROWS; /* gemm_mma reads 8 or 16 rows of A */
    e->left = m->cfg.att_left;
    e->Lmax = m->Lmax;
    e->conv_left = m->cfg.conv_kernel - 1;
    const size_t D = (size_t)m->cfg.d_model, CH = (size_t)m->cfg.sub_channels, F = (size_t)m->cfg.feat_in;
    size_t nwin = MEL_CACHE + (size_t)e->chunk_mel + 1;
    size_t T1 = nwin / 2 + 1, F1 = F / 2 + 1, T2 = T1 / 2 + 1, F2 = F1 / 2 + 1, T3 = T2 / 2 + 1, F3 = F2 / 2 + 1;
    /* +16 rows of slack: gemm_mma reads whole 16-row blocks of its input */
    e->win = gpu_buf_alloc(m->gpu, nwin * F * sizeof(float));
    e->c0 = gpu_buf_alloc(m->gpu, T1 * F1 * CH * sizeof(float));
    e->c1 = gpu_buf_alloc(m->gpu, (T2 * F2 + 16) * CH * sizeof(float));
    e->c2 = gpu_buf_alloc(m->gpu, (T2 * F2 + 16) * CH * sizeof(float));
    e->c3 = gpu_buf_alloc(m->gpu, (T3 * F3 + 16) * CH * sizeof(float));
    e->c4 = gpu_buf_alloc(m->gpu, (T3 * F3 + 16) * CH * sizeof(float));
    e->flat = gpu_buf_alloc(m->gpu, (T3 + 16) * F3 * CH * sizeof(float));
    e->pre_out = gpu_buf_alloc(m->gpu, (T3 + 16) * D * sizeof(float));
    const size_t cm = (size_t)e->cmax;
    e->h = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->tmp = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->ff = gpu_buf_alloc(m->gpu, cm * (size_t)m->cfg.d_ff * sizeof(float));
    e->att = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->t2 = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->y = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->ph = gpu_buf_alloc(m->gpu, cm * (size_t)m->cfg.prompt_hidden * sizeof(float));
    e->prompted = gpu_buf_alloc(m->gpu, cm * D * sizeof(float));
    e->joint_enc = gpu_buf_alloc(m->gpu, cm * (size_t)m->cfg.joint_hidden * sizeof(float));
    int nl = m->cfg.n_layers;
    e->cap_rows = e->left + CACHE_EPOCH_CHUNKS * e->cmax;
    e->pos = e->conv_left; /* rows [0, conv_left) stay zero: the empty conv cache */
    e->kv = calloc((size_t)nl, sizeof *e->kv);
    e->din = calloc((size_t)nl, sizeof *e->din);
    for (int l = 0; l < nl; ++l) {
        e->kv[l] = gpu_buf_alloc(m->gpu, (size_t)e->cap_rows * 3 * D * sizeof(float));
        e->din[l] = gpu_buf_alloc(m->gpu, (size_t)e->cap_rows * D * sizeof(float));
    }
    return e;
}
void encoder_destroy(encoder_t *e) {
    if (!e) return;
    gpu_buf_t *bufs[] = { e->win, e->c0, e->c1, e->c2, e->c3, e->c4, e->flat, e->pre_out, e->h, e->tmp, e->t2, e->ff, e->att, e->y, e->ph, e->prompted, e->joint_enc };
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; ++i) gpu_buf_free(bufs[i]);
    for (int l = 0; l < e->m->cfg.n_layers; ++l) { gpu_buf_free(e->kv[l]); gpu_buf_free(e->din[l]); }
    free(e->kv); free(e->din);
    free(e->pending);
    free(e);
}
int encoder_chunk_frames(const encoder_t *e) { return e->chunk_frames; }
int encoder_chunk_mel(const encoder_t *e) { return e->chunk_mel; }

void encoder_push(encoder_t *e, const float *mel, size_t nframes) {
    if (!nframes) return;
    if (e->pending_n + nframes > e->pending_cap) {
        e->pending_cap = (e->pending_n + nframes) * 2;
        e->pending = realloc(e->pending, e->pending_cap * MEL_NMEL * sizeof(float));
    }
    memcpy(e->pending + e->pending_n * MEL_NMEL, mel, nframes * MEL_NMEL * sizeof(float));
    e->pending_n += nframes;
}

static int sub_len(int n) { return n / 2 + 1; }

/* subsampling of the current window (nwin mel rows) into pre_out; returns T3 */
static int run_pre_encode(encoder_t *e, int nwin) {
    model_t *m = e->m;
    const uint32_t CH = (uint32_t)m->cfg.sub_channels, F = (uint32_t)m->cfg.feat_in;
    uint32_t T1 = (uint32_t)sub_len(nwin), F1 = F / 2 + 1, T2 = T1 / 2 + 1, F2 = F1 / 2 + 1, T3 = T2 / 2 + 1, F3 = F2 / 2 + 1;
    Conv2dParams p0 = { (uint32_t)nwin, F, T1, F1, CH };
    gpu_arg_t a0[5] = { GPU_BUF(e->win, 0), GPU_BUF(m->conv0_w, 0), GPU_BUF(m->conv0_b, 0), GPU_BUF(e->c0, 0), GPU_BYTES(&p0) };
    gpu_dispatch(m->gpu, "conv2d_c1_s2_relu", a0, 5, CH, F1, T1, CH, 1, 1);
    Conv2dParams p1 = { T1, F1, T2, F2, CH };
    gpu_arg_t a1[5] = { GPU_BUF(e->c0, 0), GPU_BUF(m->conv2_w, 0), GPU_BUF(m->conv2_b, 0), GPU_BUF(e->c1, 0), GPU_BYTES(&p1) };
    gpu_dispatch(m->gpu, "dwconv2d_s2", a1, 5, CH, F2, T2, CH, 1, 1);
    k_gemm(m, e->c1, 0, CH, m->conv3_w, CH, m->conv3_b, e->c2, 0, CH, T2 * F2, CH, CH, 2, 0, 1.0f);
    Conv2dParams p2 = { T2, F2, T3, F3, CH };
    gpu_arg_t a2[5] = { GPU_BUF(e->c2, 0), GPU_BUF(m->conv5_w, 0), GPU_BUF(m->conv5_b, 0), GPU_BUF(e->c3, 0), GPU_BYTES(&p2) };
    gpu_dispatch(m->gpu, "dwconv2d_s2", a2, 5, CH, F3, T3, CH, 1, 1);
    k_gemm(m, e->c3, 0, CH, m->conv6_w, CH, m->conv6_b, e->c4, 0, CH, T3 * F3, CH, CH, 2, 0, 1.0f);
    FlatParams fp = { T3, F3, CH };
    gpu_arg_t a3[3] = { GPU_BUF(e->c4, 0), GPU_BUF(e->flat, 0), GPU_BYTES(&fp) };
    gpu_dispatch(m->gpu, "flatten_cf", a3, 3, CH, F3, T3, CH, 1, 1);
    const uint32_t D = (uint32_t)m->cfg.d_model, K = CH * F3;
    k_gemm(m, e->flat, 0, K, m->pre_out_w, K, m->pre_out_b, e->pre_out, 0, D, T3, D, K, 0, 0, 1.0f);
    return (int)T3;
}

/* the 24 conformer blocks + prompt + joint projection for c rows already in e->h */
static void run_blocks(encoder_t *e, int c) {
    model_t *m = e->m;
    const uint32_t D = (uint32_t)m->cfg.d_model, FF = (uint32_t)m->cfg.d_ff, H = (uint32_t)m->cfg.n_heads, dh = (uint32_t)m->cfg.head_dim;
    const size_t rowb = (size_t)D * sizeof(float), kvrowb = 3 * rowb;
    const uint32_t L = (uint32_t)(e->cache_len + c);
    const int pos = e->pos;
    gpu_buf_t *h = e->h, *t = e->tmp, *t2 = e->t2;
    for (int l = 0; l < m->cfg.n_layers; ++l) {
        layer_w_t *W = &m->layers[l];
        /* FF1 (half-step); for l > 0 the norm was fused into the previous block's output norm (t2) */
        if (l == 0) k_layernorm(m, h, 0, W->ln_ff1_g, W->ln_ff1_b, t2, 0, (uint32_t)c, D);
        k_gemm(m, t2, 0, D, W->ff1_w1, D, NULL, e->ff, 0, FF, (uint32_t)c, FF, D, 1, 0, 1.0f);
        k_gemm(m, e->ff, 0, FF, W->ff1_w2, FF, NULL, h, 0, D, (uint32_t)c, D, FF, 0, 1, 0.5f);
        /* attention: fused q|k|v projection appended to the per-layer cache rows [pos, pos+c) */
        k_layernorm(m, h, 0, W->ln_att_g, W->ln_att_b, t, 0, (uint32_t)c, D);
        k_gemm(m, t, 0, D, W->wqkv, D, NULL, e->kv[l], (size_t)pos * kvrowb, 3 * D, (uint32_t)c, 3 * D, D, 0, 0, 1.0f);
        AttnParams ap = { (uint32_t)c, L, (uint32_t)e->Lmax, H, dh, D, 3 * D, 3 * D, 1.0f / sqrtf((float)dh) };
        const size_t kv0 = (size_t)(pos - e->cache_len) * kvrowb;
        gpu_arg_t aa[8] = { GPU_BUF(e->kv[l], (size_t)pos * kvrowb), GPU_BUF(e->kv[l], kv0 + rowb), GPU_BUF(e->kv[l], kv0 + 2 * rowb), GPU_BUF(W->ptab, 0),
                            GPU_BUF(W->bias_u, 0), GPU_BUF(W->bias_v, 0), GPU_BUF(e->att, 0), GPU_BYTES(&ap) };
        if (!(nemo_skip_mask & 2)) gpu_dispatch_groups(m->gpu, "attention_rel", aa, 8, H, (uint32_t)c, 1, ATT_DH, 1, 1);
        k_gemm(m, e->att, 0, D, W->wo, D, NULL, h, 0, D, (uint32_t)c, D, D, 0, 1, 1.0f);
        /* convolution module: GLU output appended at din rows [pos, pos+c); the causal conv reads
           the previous conv_left rows as history */
        k_layernorm(m, h, 0, W->ln_conv_g, W->ln_conv_b, t, 0, (uint32_t)c, D);
        /* pointwise conv 1 with the GLU fused into the GEMM epilogue, written straight into the conv history */
        k_gemm(m, t, 0, D, W->pw1, D, NULL, e->din[l], (size_t)pos * rowb, D, (uint32_t)c, 2 * D, D, 3, 0, 1.0f);
        DwParams dp = { (uint32_t)c, D, (uint32_t)m->cfg.conv_kernel, m->cfg.ln_eps };
        gpu_arg_t ad[6] = { GPU_BUF(e->din[l], (size_t)(pos - e->conv_left) * rowb), GPU_BUF(W->dw_w, 0), GPU_BUF(W->bn_g, 0), GPU_BUF(W->bn_b, 0), GPU_BUF(e->y, 0), GPU_BYTES(&dp) };
        if (!(nemo_skip_mask & 16)) gpu_dispatch_groups(m->gpu, "dwconv_ln_silu", ad, 6, (uint32_t)c, 1, 1, LN_THREADS, 1, 1);
        k_gemm(m, e->y, 0, D, W->pw2, D, NULL, h, 0, D, (uint32_t)c, D, D, 0, 1, 1.0f);
        /* FF2 (half-step) + output norm */
        k_layernorm(m, h, 0, W->ln_ff2_g, W->ln_ff2_b, t, 0, (uint32_t)c, D);
        k_gemm(m, t, 0, D, W->ff2_w1, D, NULL, e->ff, 0, FF, (uint32_t)c, FF, D, 1, 0, 1.0f);
        k_gemm(m, e->ff, 0, FF, W->ff2_w2, FF, NULL, h, 0, D, (uint32_t)c, D, FF, 0, 1, 0.5f);
        if (l + 1 < m->cfg.n_layers) {
            /* output norm of this block and the FF1 norm of the next in one dispatch */
            layer_w_t *Wn = &m->layers[l + 1];
            k_layernorm2(m, h, W->ln_out_g, W->ln_out_b, Wn->ln_ff1_g, Wn->ln_ff1_b, t, t2, (uint32_t)c, D);
        } else {
            k_layernorm(m, h, 0, W->ln_out_g, W->ln_out_b, t, 0, (uint32_t)c, D);
        }
        gpu_buf_t *sw = h; h = t; t = sw;
    }
    /* language prompt (one-hot folded into the bias) and joint encoder projection */
    const uint32_t PH = (uint32_t)m->cfg.prompt_hidden, NP = (uint32_t)m->cfg.num_prompts;
    k_gemm(m, h, 0, D, m->prompt_w0, D + NP, m->prompt_b0_eff, e->ph, 0, PH, (uint32_t)c, PH, D, 2, 0, 1.0f);
    k_gemm(m, e->ph, 0, PH, m->prompt_w2, PH, m->prompt_b2, e->prompted, 0, D, (uint32_t)c, D, PH, 0, 0, 1.0f);
    const uint32_t JH = (uint32_t)m->cfg.joint_hidden;
    k_gemm(m, e->prompted, 0, D, m->joint_enc_w, D, m->joint_enc_b, e->joint_enc, 0, JH, (uint32_t)c, JH, D, 0, 0, 1.0f);
    e->pos += c;
    e->cache_len = (int)L < e->left ? (int)L : e->left;
}

/* Once per epoch: move the live history (cache_len attention rows, conv_left conv rows) to the
   front of the append-only buffers. Regions never overlap because cap_rows >= 2*left + cmax. */
static void compact_caches(encoder_t *e) {
    model_t *m = e->m;
    const uint32_t D = (uint32_t)m->cfg.d_model;
    const size_t rowb = (size_t)D * sizeof(float);
    const int src_kv = e->pos - e->cache_len, src_din = e->pos - e->conv_left, dst_din = e->cache_len - e->conv_left;
    for (int l = 0; l < m->cfg.n_layers; ++l) {
        if (e->cache_len > 0) k_copy(m, e->kv[l], (size_t)src_kv * 3 * rowb, e->kv[l], 0, (uint32_t)e->cache_len * 3 * D);
        k_copy(m, e->din[l], (size_t)src_din * rowb, e->din[l], (size_t)dst_din * rowb, (uint32_t)e->conv_left * D);
    }
    e->pos = e->cache_len;
}

int encoder_step(encoder_t *e, int final, enc_out_t *out, char *err, size_t errlen) {
    memset(out, 0, sizeof *out);
    if (e->closed) return 0;
    if (e->pending_n == 0) { if (final) e->closed = 1; return 0; }
    if ((int)e->pending_n < e->chunk_mel && !final) return 0;
    size_t take = (size_t)e->chunk_mel < e->pending_n ? (size_t)e->chunk_mel : e->pending_n;
    if (final && e->pending_n <= (size_t)e->chunk_mel) take = e->pending_n;
    int include_boundary = final && e->pending_n == take;

    double t0 = now_ms();
    /* window = mel cache ++ m */
    int nwin = e->mel_cache_n + (int)take;
    float *w = gpu_buf_ptr(e->win);
    memcpy(w, e->mel_cache, (size_t)e->mel_cache_n * MEL_NMEL * sizeof(float));
    memcpy(w + (size_t)e->mel_cache_n * MEL_NMEL, e->pending, take * MEL_NMEL * sizeof(float));
    /* bookkeeping identical to ConformerStreamingState._encode_mel_chunk */
    size_t end = e->consumed + take;
    size_t base = (e->consumed - (size_t)e->mel_cache_n) / (size_t)e->m->cfg.subsampling;
    size_t lo = e->emitted - base;
    int T3 = sub_len(sub_len(sub_len(nwin)));
    size_t hi = include_boundary ? (size_t)T3 : end / (size_t)e->m->cfg.subsampling - base;
    e->consumed = end;
    /* new mel cache = last 16 rows of the window */
    int keep = nwin < MEL_CACHE ? nwin : MEL_CACHE;
    memcpy(e->mel_cache, w + (size_t)(nwin - keep) * MEL_NMEL, (size_t)keep * MEL_NMEL * sizeof(float));
    e->mel_cache_n = keep;
    /* pop consumed pending rows */
    memmove(e->pending, e->pending + take * MEL_NMEL, (e->pending_n - take) * MEL_NMEL * sizeof(float));
    e->pending_n -= take;
    if (final && e->pending_n == 0) e->closed = 1;

    if (hi <= lo) {
        e->emitted = base + (lo > hi ? lo : hi);
        out->wall_ms = now_ms() - t0;
        return 1;
    }
    e->emitted = base + hi;
    int c = (int)(hi - lo);
    if (c > e->cmax) { snprintf(err, errlen, "chunk of %d rows exceeds buffer %d", c, e->cmax); return -1; }

    model_t *m = e->m;
    const uint32_t D = (uint32_t)m->cfg.d_model;
    gpu_begin(m->gpu);
    if (e->pos + c > e->cap_rows) compact_caches(e);
    if (!(nemo_skip_mask & 32)) run_pre_encode(e, nwin);
    k_copy(m, e->pre_out, lo * D * sizeof(float), e->h, 0, (uint32_t)c * D);
    run_blocks(e, c);
    if (gpu_end(m->gpu, err, errlen)) return -1;
    out->frames = c;
    out->prompted = e->prompted;
    out->joint_enc = e->joint_enc;
    out->gpu_ms = gpu_last_ms(m->gpu);
    out->wall_ms = now_ms() - t0;
    return 1;
}
