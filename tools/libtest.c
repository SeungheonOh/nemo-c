/* Exercise the library API the way an app would: open, feed audio in 100 ms blocks, print text. */
#include "nemoasr.h"
#include "wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: libtest file.wav [language] [latency_ms]\n"); return 2; }
    const char *lang = argc > 2 ? argv[2] : "en-US";
    int latency = argc > 3 ? atoi(argv[3]) : 560;
    char err[512], dir[1200];
    if (nemoasr_default_model_dir(dir, sizeof dir)) { fprintf(stderr, "model not found\n"); return 1; }
    int rate; size_t ns;
    float *audio = wav_read_mono(argv[1], &rate, &ns, err, sizeof err);
    if (!audio) { fprintf(stderr, "%s\n", err); return 1; }
    nemoasr_t *s = nemoasr_open(dir, lang, latency, rate, err, sizeof err);
    if (!s) { fprintf(stderr, "open: %s\n", err); return 1; }
    fprintf(stderr, "[lib] ready in %.0f ms on %s, input %d Hz, %d ms chunks\n", nemoasr_load_ms(s), nemoasr_gpu_name(s), rate, nemoasr_latency_ms(s));
    size_t block = (size_t)rate / 10;
    for (size_t pos = 0; pos < ns; pos += block) {
        size_t n = pos + block <= ns ? block : ns - pos;
        char *t = nemoasr_feed(s, audio + pos, n, pos + n >= ns, err, sizeof err);
        if (err[0]) { fprintf(stderr, "feed: %s\n", err); return 1; }
        if (t) { fputs(t, stdout); fflush(stdout); nemoasr_free(t); }
    }
    fputc('\n', stdout);
    nemoasr_stats_t st; nemoasr_stats(s, &st);
    fprintf(stderr, "[lib] %.2fs audio, %zu chunks, %zu tokens, encoder %.1f ms, decoder %.1f ms\n", st.audio_seconds, st.chunks, st.tokens, st.encoder_ms, st.decoder_ms);
    nemoasr_close(s);
    free(audio);
    return 0;
}
