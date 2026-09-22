/* Cache-aware streaming FastConformer encoder + language prompt + joint encoder projection. */
#ifndef NEMO_ENCODER_H
#define NEMO_ENCODER_H
#include "model.h"

typedef struct encoder encoder_t;

typedef struct {
    int frames;            /* encoder frames produced by this step (may be 0) */
    gpu_buf_t *prompted;   /* [frames][d_model] f32 post-prompt encoder output */
    gpu_buf_t *joint_enc;  /* [frames][joint_hidden] f32 joint.enc projection */
    double gpu_ms, wall_ms;
} enc_out_t;

encoder_t *encoder_create(model_t *m, int right_context);
void encoder_destroy(encoder_t *e);
int encoder_chunk_frames(const encoder_t *e);
int encoder_chunk_mel(const encoder_t *e);
/* queue mel frames (MEL_NMEL floats each) */
void encoder_push(encoder_t *e, const float *mel, size_t nframes);
/* Process one pending piece (a full chunk, or the remainder when final). Returns 1 if a piece
   was consumed (check out->frames), 0 if nothing to do, -1 on error (message in err). */
int encoder_step(encoder_t *e, int final, enc_out_t *out, char *err, size_t errlen);

#endif
