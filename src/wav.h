#ifndef NEMO_WAV_H
#define NEMO_WAV_H
#include <stddef.h>
/* Reads a RIFF/WAVE file (PCM 16/24/32-bit or float32, any channel count, downmixed to mono).
   Returns malloc'd float samples in [-1, 1]; *rate receives the sample rate. NULL on error. */
float *wav_read_mono(const char *path, int *rate, size_t *nsamples, char *err, size_t errlen);
#endif
