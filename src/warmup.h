/* GPU warm-up shared by the CLI and the library: compile every kernel pipeline the stream can hit. */
#ifndef NEMO_WARMUP_H
#define NEMO_WARMUP_H
#include "model.h"
void asr_warmup(model_t *m, int right_context, char *err, size_t errlen);
#endif
