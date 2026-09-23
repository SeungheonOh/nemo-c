/* nemoasr: NVIDIA Nemotron 3.5 ASR streaming, from scratch in C + Metal. Library interface.

   Usage: nemoasr_open() once per session (about 300 ms: model load + GPU warm-up), then call
   nemoasr_feed() with mono float audio as it arrives; every call may return newly decoded text.
   Finish with nemoasr_feed(..., final=1) and nemoasr_close(). Not thread-safe: call all
   functions on one thread (or serialise them). */
#ifndef NEMOASR_H
#define NEMOASR_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nemoasr nemoasr_t;

typedef struct {
    double audio_seconds;
    size_t chunks, tokens;
    double encoder_ms, decoder_ms;
} nemoasr_stats_t;

/* Default model directory: the mlx-community/nemotron-3.5-asr-streaming-0.6b snapshot in the
   Hugging Face cache (downloaded by the Python project). Returns 0 and fills `out` if found. */
int nemoasr_default_model_dir(char *out, size_t n);

/* Load the model and prepare a streaming session.
   language: prompt key such as "en-US", "ko-KR" or "auto".
   latency_ms: 80, 320, 560 or 1120 (encoder chunk look-ahead).
   input_rate: sample rate of the audio you will feed; it is resampled to 16 kHz internally. */
nemoasr_t *nemoasr_open(const char *model_dir, const char *language, int latency_ms, int input_rate, char *err, size_t errlen);
void nemoasr_close(nemoasr_t *s);

/* Feed `n` mono float samples at input_rate (n may be 0). Returns newly decoded text as a
   malloc'd UTF-8 string (free with nemoasr_free) or NULL when nothing new. With final=1 the
   remaining audio is flushed and the session is finished. On failure returns NULL with a
   non-empty message in err. */
char *nemoasr_feed(nemoasr_t *s, const float *samples, size_t n, int final, char *err, size_t errlen);
void nemoasr_free(char *text);

/* Start a fresh stream on the same loaded model: drops all streaming state (encoder caches,
   decoder state, buffered audio, pending text, the finished flag) and switches the language prompt
   and chunk latency. Audio not yet decoded is discarded; call nemoasr_feed(..., final=1) first to
   flush it. Takes microseconds unless the latency's kernels were never compiled (see below). */
int nemoasr_reset(nemoasr_t *s, const char *language, int latency_ms, char *err, size_t errlen);

/* Compile and warm the kernels for another latency ahead of time, so a later nemoasr_reset to it
   does not pause the stream. */
int nemoasr_prepare_latency(nemoasr_t *s, int latency_ms, char *err, size_t errlen);

double nemoasr_load_ms(const nemoasr_t *s);           /* model load + warm-up */
const char *nemoasr_gpu_name(const nemoasr_t *s);
int nemoasr_latency_ms(const nemoasr_t *s);
void nemoasr_stats(const nemoasr_t *s, nemoasr_stats_t *out);

#ifdef __cplusplus
}
#endif
#endif
