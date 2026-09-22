/* Streaming log-mel frontend matching NeMo/mlx-audio AudioToMelSpectrogramPreprocessor:
   16 kHz, preemphasis 0.97, 25 ms symmetric Hann (400) centered in a 512 FFT, 10 ms hop,
   centered frames with reflect padding, power spectrum, 128 Slaney mel bins, log(x + 2^-24). */
#ifndef NEMO_MEL_H
#define NEMO_MEL_H
#include <stddef.h>

#define MEL_SR 16000
#define MEL_NFFT 512
#define MEL_WIN 400
#define MEL_HOP 160
#define MEL_NMEL 128
#define MEL_NFREQ (MEL_NFFT / 2 + 1)

typedef struct mel_state mel_state_t;

mel_state_t *mel_create(void);
void mel_destroy(mel_state_t *m);
/* Feed samples (may be 0 with final=1 to flush). Newly available frames are returned through
   *frames (MEL_NMEL floats each, internal buffer valid until the next call). */
void mel_push(mel_state_t *m, const float *samples, size_t n, int final, const float **frames, size_t *nframes);
size_t mel_total_samples(const mel_state_t *m);
size_t mel_emitted_frames(const mel_state_t *m);

#endif
