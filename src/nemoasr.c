#include "nemoasr.h"
#include "decoder.h"
#include "encoder.h"
#include "mel.h"
#include "model.h"
#include "resample.h"
#include "warmup.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

struct nemoasr {
    model_t *m;
    encoder_t *enc;
    decoder_t *dec;
    mel_state_t *mel;
    resampler_t *rs;
    int input_rate, latency_ms, finished, printed_any;
    unsigned warmed; /* bit r set: kernels for right context r are compiled */
    float *tmp;
    size_t tmp_cap;
    char *text;
    size_t text_len, text_cap;
    int tokens[4096];
    double load_ms;
    nemoasr_stats_t stats;
};

int nemoasr_default_model_dir(char *out, size_t n) {
    const char *home = getenv("HOME");
    char snaps[1024];
    snprintf(snaps, sizeof snaps, "%s/.cache/huggingface/hub/models--mlx-community--nemotron-3.5-asr-streaming-0.6b/snapshots", home ? home : ".");
    DIR *d = opendir(snaps);
    if (!d) return -1;
    struct dirent *ent;
    int found = 0;
    while ((ent = readdir(d))) {
        if (ent->d_name[0] == '.') continue;
        snprintf(out, n, "%s/%s", snaps, ent->d_name);
        char probe[1200];
        snprintf(probe, sizeof probe, "%s/model.safetensors", out);
        struct stat sb;
        if (stat(probe, &sb) == 0) { found = 1; break; }
    }
    closedir(d);
    return found ? 0 : -1;
}

static const int LAT_MS[] = { 80, 160, 320, 560, 1120 };
static const int LAT_R[] = { 0, 1, 3, 6, 13 };

static int right_for(int latency_ms) {
    for (size_t i = 0; i < sizeof LAT_MS / sizeof *LAT_MS; ++i) if (LAT_MS[i] == latency_ms) return LAT_R[i];
    return -1;
}

nemoasr_t *nemoasr_open(const char *model_dir, const char *language, int latency_ms, int input_rate, char *err, size_t errlen) {
    if (errlen) err[0] = 0;
    int right = right_for(latency_ms);
    if (right < 0) { snprintf(err, errlen, "latency must be one of 80, 320, 560, 1120 ms"); return NULL; }
    if (input_rate <= 0) { snprintf(err, errlen, "input_rate must be positive"); return NULL; }
    double t0 = now_ms();
    nemoasr_t *s = calloc(1, sizeof *s);
    s->m = model_load(model_dir, err, errlen);
    if (!s->m) { free(s); return NULL; }
    if (!model_right_context_trained(s->m, right)) { snprintf(err, errlen, "latency %d ms not trained for this checkpoint", latency_ms); nemoasr_close(s); return NULL; }
    if (model_set_language(s->m, language ? language : "auto", err, errlen)) { nemoasr_close(s); return NULL; }
    asr_warmup(s->m, right, err, errlen);
    s->warmed = 1u << right;
    s->enc = encoder_create(s->m, right);
    s->dec = decoder_create(s->m);
    s->mel = mel_create();
    s->input_rate = input_rate;
    s->latency_ms = latency_ms;
    if (input_rate != MEL_SR) s->rs = resampler_create(input_rate, MEL_SR);
    s->load_ms = now_ms() - t0;
    return s;
}

void nemoasr_close(nemoasr_t *s) {
    if (!s) return;
    if (s->mel) mel_destroy(s->mel);
    if (s->dec) decoder_destroy(s->dec);
    if (s->enc) encoder_destroy(s->enc);
    if (s->rs) resampler_destroy(s->rs);
    if (s->m) model_free(s->m);
    free(s->tmp);
    free(s->text);
    free(s);
}

static void append_text(nemoasr_t *s, const char *t) {
    size_t n = strlen(t);
    if (!n) return;
    if (s->text_len + n + 1 > s->text_cap) {
        s->text_cap = (s->text_len + n + 1) * 2;
        s->text = realloc(s->text, s->text_cap);
    }
    memcpy(s->text + s->text_len, t, n + 1);
    s->text_len += n;
}

static int run_pending(nemoasr_t *s, int final, char *err, size_t errlen) {
    enc_out_t out;
    int rc;
    while ((rc = encoder_step(s->enc, final, &out, err, errlen)) == 1) {
        if (out.frames <= 0) continue;
        s->stats.chunks++;
        s->stats.encoder_ms += out.wall_ms;
        int n = decoder_run(s->dec, out.joint_enc, out.frames, s->tokens, 4096, err, errlen);
        if (n < 0) return -1;
        s->stats.decoder_ms += decoder_last_ms(s->dec);
        s->stats.tokens += (size_t)n;
        for (int i = 0; i < n; ++i) {
            char *txt = tok_text(&s->m->vocab, s->tokens[i]);
            const char *p = txt;
            if (!s->printed_any) while (*p == ' ') p++;
            if (*p) { append_text(s, p); s->printed_any = 1; }
            free(txt);
        }
    }
    return rc;
}

char *nemoasr_feed(nemoasr_t *s, const float *samples, size_t n, int final, char *err, size_t errlen) {
    if (errlen) err[0] = 0;
    if (s->finished) return NULL;
    const float *in = samples;
    size_t count = n;
    if (s->rs) {
        size_t need = (size_t)((double)n * MEL_SR / s->input_rate) + 4096;
        if (need > s->tmp_cap) { s->tmp_cap = need * 2; s->tmp = realloc(s->tmp, s->tmp_cap * sizeof(float)); }
        count = n ? resampler_process(s->rs, samples, n, s->tmp, s->tmp_cap) : 0;
        if (final) count += resampler_flush(s->rs, s->tmp + count, s->tmp_cap - count);
        in = s->tmp;
    }
    s->stats.audio_seconds += (double)count / MEL_SR;
    if (!(final && count == 0 && mel_total_samples(s->mel) == 0)) {
        const float *frames;
        size_t nf;
        mel_push(s->mel, in, count, final, &frames, &nf);
        if (nf) encoder_push(s->enc, frames, nf);
        if (run_pending(s, final, err, errlen) < 0) return NULL;
    }
    if (final) s->finished = 1;
    if (!s->text_len) return NULL;
    char *out = s->text;
    s->text = NULL;
    s->text_len = s->text_cap = 0;
    return out;
}

int nemoasr_prepare_latency(nemoasr_t *s, int latency_ms, char *err, size_t errlen) {
    if (errlen) err[0] = 0;
    int right = right_for(latency_ms);
    if (right < 0) { snprintf(err, errlen, "latency must be one of 80, 320, 560, 1120 ms"); return -1; }
    if (!model_right_context_trained(s->m, right)) { snprintf(err, errlen, "latency %d ms not trained for this checkpoint", latency_ms); return -1; }
    if (!(s->warmed & (1u << right))) {
        asr_warmup(s->m, right, err, errlen);
        s->warmed |= 1u << right;
    }
    return 0;
}

int nemoasr_reset(nemoasr_t *s, const char *language, int latency_ms, char *err, size_t errlen) {
    if (errlen) err[0] = 0;
    if (nemoasr_prepare_latency(s, latency_ms, err, errlen)) return -1;
    if (language && model_set_language(s->m, language, err, errlen)) return -1;
    int right = right_for(latency_ms);
    if (encoder_right(s->enc) == right) {
        encoder_reset(s->enc);   /* same chunking: keep the GPU buffers, just empty the caches */
    } else {
        encoder_destroy(s->enc);
        s->enc = encoder_create(s->m, right);
    }
    mel_destroy(s->mel);
    s->mel = mel_create();
    decoder_reset(s->dec);
    if (s->rs) { resampler_destroy(s->rs); s->rs = resampler_create(s->input_rate, MEL_SR); }
    free(s->text);
    s->text = NULL;
    s->text_len = s->text_cap = 0;
    s->finished = 0;
    s->printed_any = 0;
    s->latency_ms = latency_ms;
    return 0;
}

void nemoasr_free(char *text) { free(text); }
double nemoasr_load_ms(const nemoasr_t *s) { return s->load_ms; }
const char *nemoasr_gpu_name(const nemoasr_t *s) { return gpu_device_name(s->m->gpu); }
int nemoasr_latency_ms(const nemoasr_t *s) { return s->latency_ms; }
void nemoasr_stats(const nemoasr_t *s, nemoasr_stats_t *out) { *out = s->stats; }
