#include "warmup.h"
#include "decoder.h"
#include "encoder.h"
#include "mel.h"
#include <string.h>

/* Compile every GEMM specialisation the stream can hit (row counts 1..16, plain and GLU) so no
   chunk pays a pipeline build mid-stream; the final boundary chunk can have any row count. */
static void precompile_gemms(model_t *m, char *err, size_t errlen) {
    const uint32_t D = (uint32_t)m->cfg.d_model;
    gpu_buf_t *a = gpu_buf_alloc(m->gpu, 16 * D * sizeof(float));
    gpu_buf_t *c = gpu_buf_alloc(m->gpu, 16 * 2 * D * sizeof(float));
    gpu_begin(m->gpu);
    for (uint32_t M = 1; M <= 16; ++M) {
        k_gemm(m, a, 0, D, m->layers[0].wq, D, NULL, c, 0, D, M, D, D, 0, 0, 1.0f);
        k_gemm(m, a, 0, D, m->layers[0].pw1, D, NULL, c, 0, D, M, 2 * D, D, 3, 0, 1.0f);
    }
    gpu_end(m->gpu, err, errlen);
    gpu_buf_free(a);
    gpu_buf_free(c);
}

void asr_warmup(model_t *m, int right, char *err, size_t errlen) {
    precompile_gemms(m, err, errlen);
    encoder_t *e = encoder_create(m, right);
    decoder_t *d = decoder_create(m);
    mel_state_t *ms = mel_create();
    float zeros[1600] = {0};
    const float *frames;
    size_t nf;
    for (int i = 0; i < 15; ++i) {
        mel_push(ms, zeros, 1600, i == 14, &frames, &nf);
        encoder_push(e, frames, nf);
        enc_out_t out;
        int tok[64];
        while (encoder_step(e, i == 14, &out, err, errlen) == 1)
            if (out.frames > 0) decoder_run(d, out.joint_enc, out.frames, tok, 64, err, errlen);
    }
    mel_destroy(ms);
    decoder_destroy(d);
    encoder_destroy(e);
}
