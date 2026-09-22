#include "decoder.h"
#include "kernel_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

struct decoder {
    model_t *m;
    gpu_buf_t *x, *gates, *pred_out, *argmax, *part_v, *part_i;
    gpu_buf_t *h[2], *c[2];   /* committed LSTM state */
    gpu_buf_t *hp[2], *cp[2]; /* provisional state after feeding the last token */
    int last_token;
    int pred_valid;
    double last_ms;
    int last_steps;
};

decoder_t *decoder_create(model_t *m) {
    decoder_t *d = calloc(1, sizeof *d);
    d->m = m;
    size_t PH = (size_t)m->cfg.pred_hidden * sizeof(float);
    d->x = gpu_buf_alloc(m->gpu, PH);
    d->gates = gpu_buf_alloc(m->gpu, 4 * PH);
    d->pred_out = gpu_buf_alloc(m->gpu, (size_t)m->cfg.joint_hidden * sizeof(float));
    d->argmax = gpu_buf_alloc(m->gpu, (size_t)DEC_MAX_BATCH * 2 * sizeof(int));
    d->part_v = gpu_buf_alloc(m->gpu, (size_t)DEC_MAX_BATCH * JOINT_GROUPS * GEMM_SIMDS * sizeof(float));
    d->part_i = gpu_buf_alloc(m->gpu, (size_t)DEC_MAX_BATCH * JOINT_GROUPS * GEMM_SIMDS * sizeof(int));
    for (int l = 0; l < 2; ++l) {
        d->h[l] = gpu_buf_alloc(m->gpu, PH); d->c[l] = gpu_buf_alloc(m->gpu, PH);
        d->hp[l] = gpu_buf_alloc(m->gpu, PH); d->cp[l] = gpu_buf_alloc(m->gpu, PH);
    }
    decoder_reset(d);
    return d;
}
void decoder_destroy(decoder_t *d) {
    if (!d) return;
    gpu_buf_t *bufs[] = { d->x, d->gates, d->pred_out, d->argmax, d->part_v, d->part_i, d->h[0], d->h[1], d->c[0], d->c[1], d->hp[0], d->hp[1], d->cp[0], d->cp[1] };
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; ++i) gpu_buf_free(bufs[i]);
    free(d);
}
void decoder_reset(decoder_t *d) {
    size_t PH = (size_t)d->m->cfg.pred_hidden * sizeof(float);
    for (int l = 0; l < 2; ++l) { memset(gpu_buf_ptr(d->h[l]), 0, PH); memset(gpu_buf_ptr(d->c[l]), 0, PH); }
    d->last_token = d->m->cfg.blank_id;
    d->pred_valid = 0;
}
double decoder_last_ms(const decoder_t *d) { return d->last_ms; }
int decoder_last_steps(const decoder_t *d) { return d->last_steps; }

/* encode: prediction network from the committed state with input = embed(last_token) (zeros for blank) */
static void encode_pred(decoder_t *d) {
    model_t *m = d->m;
    const uint32_t PH = (uint32_t)m->cfg.pred_hidden, JH = (uint32_t)m->cfg.joint_hidden;
    float *x = gpu_buf_ptr(d->x);
    if (d->last_token == m->cfg.blank_id) memset(x, 0, PH * sizeof(float));
    else {
        const uint16_t *row = m->embed_bf16 + (size_t)d->last_token * PH;
        for (uint32_t i = 0; i < PH; ++i) x[i] = bf16_to_f32(row[i]);
    }
    gpu_buf_t *in = d->x;
    for (int l = 0; l < 2; ++l) {
        k_gemm(m, in, 0, PH, m->lstm_wx[l], PH, m->lstm_b[l], d->gates, 0, 4 * PH, 1, 4 * PH, PH, 0, 0, 1.0f);
        k_gemm(m, d->h[l], 0, PH, m->lstm_wh[l], PH, NULL, d->gates, 0, 4 * PH, 1, 4 * PH, PH, 0, 1, 1.0f);
        LstmParams lp = { PH };
        gpu_arg_t args[5] = { GPU_BUF(d->gates, 0), GPU_BUF(d->c[l], 0), GPU_BUF(d->cp[l], 0), GPU_BUF(d->hp[l], 0), GPU_BYTES(&lp) };
        gpu_dispatch(m->gpu, "lstm_cell", args, 5, PH, 1, 1, 64, 1, 1);
        in = d->hp[l];
    }
    k_gemm(m, d->hp[1], 0, PH, m->joint_pred_w, PH, m->joint_pred_b, d->pred_out, 0, JH, 1, JH, PH, 0, 0, 1.0f);
}

/* joint + argmax for `rows` consecutive encoder frames starting at `row`, all with the current
   prediction vector (speculating that they decode to blank); results land in argmax[2*i] */
static void encode_argmax(decoder_t *d, gpu_buf_t *joint_enc, int row, int rows) {
    model_t *m = d->m;
    JointParams jp = { (uint32_t)m->cfg.num_classes, (uint32_t)m->cfg.joint_hidden, (uint32_t)row, (uint32_t)m->cfg.joint_hidden };
    gpu_arg_t a1[7] = { GPU_BUF(joint_enc, 0), GPU_BUF(d->pred_out, 0), GPU_BUF(m->joint_out_w.buf, m->joint_out_w.off), GPU_BUF(m->joint_out_b, 0),
                        GPU_BUF(d->part_v, 0), GPU_BUF(d->part_i, 0), GPU_BYTES(&jp) };
    gpu_dispatch_groups(m->gpu, "joint_partial", a1, 7, JOINT_GROUPS, (uint32_t)rows, 1, GEMM_SIMDS * 32, 1, 1);
    CountParams cp = { JOINT_GROUPS * GEMM_SIMDS };
    gpu_arg_t a2[4] = { GPU_BUF(d->part_v, 0), GPU_BUF(d->part_i, 0), GPU_BUF(d->argmax, 0), GPU_BYTES(&cp) };
    gpu_dispatch_groups(m->gpu, "argmax_reduce", a2, 4, (uint32_t)rows, 1, 1, 256, 1, 1);
}

int decoder_run(decoder_t *d, gpu_buf_t *joint_enc, int frames, int *out, int max_out, char *err, size_t errlen) {
    model_t *m = d->m;
    double t0 = now_ms();
    int n = 0, steps = 0;
    int t = 0, symbols = 0;
    while (t < frames) {
        int rows = frames - t;
        if (rows > DEC_MAX_BATCH) rows = DEC_MAX_BATCH;
        gpu_begin(m->gpu);
        if (!d->pred_valid) encode_pred(d);
        encode_argmax(d, joint_enc, t, rows);
        if (gpu_end(m->gpu, err, errlen)) return -1;
        steps++;
        d->pred_valid = 1;
        const int *res = gpu_buf_ptr(d->argmax);
        int i = 0;
        for (; i < rows; ++i) {
            int pred = res[2 * i];
            if (pred == m->cfg.blank_id) { symbols = 0; continue; } /* blank: frame t+i is done, next frame keeps the same prediction state */
            if (n < max_out) out[n++] = pred;
            d->last_token = pred;
            for (int l = 0; l < 2; ++l) { /* commit the provisional state by swapping buffers */
                gpu_buf_t *s = d->h[l]; d->h[l] = d->hp[l]; d->hp[l] = s;
                s = d->c[l]; d->c[l] = d->cp[l]; d->cp[l] = s;
            }
            d->pred_valid = 0;
            symbols++;
            if (symbols >= m->cfg.max_symbols) { symbols = 0; i++; } /* frame exhausted: move on */
            break; /* later frames were scored with the old prediction state: redo them */
        }
        t += i;
    }
    d->last_ms = now_ms() - t0;
    d->last_steps = steps;
    return n;
}
