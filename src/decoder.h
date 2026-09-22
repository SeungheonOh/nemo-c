/* Greedy RNN-T decoding (2-layer LSTM prediction network + joint), one encoder chunk at a time. */
#ifndef NEMO_DECODER_H
#define NEMO_DECODER_H
#include "model.h"

typedef struct decoder decoder_t;

decoder_t *decoder_create(model_t *m);
void decoder_destroy(decoder_t *d);
void decoder_reset(decoder_t *d);
/* Decode `frames` rows of joint_enc; appends token ids to out. Returns count or -1 on error. */
int decoder_run(decoder_t *d, gpu_buf_t *joint_enc, int frames, int *out, int max_out, char *err, size_t errlen);
double decoder_last_ms(const decoder_t *d);
int decoder_last_steps(const decoder_t *d); /* joint evaluations in the last run */

#endif
