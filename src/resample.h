/* Windowed-sinc sample-rate converter (any ratio), streaming. */
#ifndef NEMO_RESAMPLE_H
#define NEMO_RESAMPLE_H
#include <stddef.h>
typedef struct resampler resampler_t;
resampler_t *resampler_create(int in_rate, int out_rate);
void resampler_destroy(resampler_t *r);
/* Push input; returns how many output samples were written (<= out_max; remaining output stays queued for later). */
size_t resampler_process(resampler_t *r, const float *in, size_t n, float *out, size_t out_max);
/* Drain with zero padding at end of stream. */
size_t resampler_flush(resampler_t *r, float *out, size_t out_max);
#endif
