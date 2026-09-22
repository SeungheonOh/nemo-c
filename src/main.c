/* nemoasr-c: NVIDIA Nemotron 3.5 ASR Streaming 0.6B, from scratch in C + Metal.
   Live microphone or WAV file -> streaming log-mel -> cache-aware FastConformer -> RNNT greedy -> text. */
#include "decoder.h"
#include "encoder.h"
#include "mel.h"
#include "mic.h"
#include "model.h"
#include "resample.h"
#include "wav.h"
#include <dirent.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static double warmup_ms_global = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    const char *model_dir, *file, *language, *device, *dump_mel, *dump_enc, *dump_tokens;
    int latency, realtime, list_devices, list_languages, verbose, no_warmup, block_ms, info, gpu_warm;
} opts_t;

static const int LAT_MS[] = { 80, 160, 320, 560, 1120 };
static const int LAT_R[] = { 0, 1, 3, 6, 13 };

static void usage(void) {
    fprintf(stderr,
        "usage: nemoasr-c [options]\n"
        "  --model-dir DIR      directory with config.json + model.safetensors (default: HF cache snapshot)\n"
        "  --file WAV           transcribe a 16 kHz WAV instead of the microphone\n"
        "  --realtime           with --file: pace at real time\n"
        "  --language KEY       prompt key, e.g. en-US, ko-KR, auto (default en-US)\n"
        "  --latency MS         80 | 160 | 320 | 560 | 1120 (default 560; must be trained for the checkpoint)\n"
        "  --device SPEC        input device index or name substring\n"
        "  --list-devices | --list-languages | --info\n"
        "  --dump-mel F | --dump-enc F | --dump-tokens F   write parity artifacts\n"
        "  --gpu-warm           keep the GPU clocked up between chunks (halves live chunk latency, costs power)\n"
        "  --no-warmup | --verbose\n");
}

static int find_default_model_dir(char *out, size_t n) {
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

/* ---- pipeline state ---------------------------------------------------------------- */
typedef struct {
    model_t *m;
    mel_state_t *mel;
    encoder_t *enc;
    decoder_t *dec;
    FILE *f_mel, *f_enc, *f_tok;
    int tokens[4096];
    size_t total_tokens, chunks, dec_steps;
    double enc_gpu_ms, enc_wall_ms, enc_wall_max, dec_ms, dec_max, first_token_ms;
    double t_start;
    int printed_any;
    int verbose;
} pipeline_t;

static int run_pending(pipeline_t *p, int final, char *err, size_t errlen) {
    enc_out_t out;
    int rc;
    while ((rc = encoder_step(p->enc, final, &out, err, errlen)) == 1) {
        if (out.frames <= 0) continue;
        p->chunks++;
        p->enc_gpu_ms += out.gpu_ms;
        p->enc_wall_ms += out.wall_ms;
        if (out.wall_ms > p->enc_wall_max) p->enc_wall_max = out.wall_ms;
        if (p->f_enc) fwrite(gpu_buf_ptr(out.prompted), sizeof(float), (size_t)out.frames * (size_t)p->m->cfg.d_model, p->f_enc);
        int n = decoder_run(p->dec, out.joint_enc, out.frames, p->tokens, 4096, err, errlen);
        if (n < 0) return -1;
        p->dec_ms += decoder_last_ms(p->dec);
        if (decoder_last_ms(p->dec) > p->dec_max) p->dec_max = decoder_last_ms(p->dec);
        p->dec_steps += (size_t)decoder_last_steps(p->dec);
        for (int i = 0; i < n; ++i) {
            if (!p->printed_any) p->first_token_ms = now_ms() - p->t_start;
            char *txt = tok_text(&p->m->vocab, p->tokens[i]);
            const char *s = txt;
            if (!p->printed_any) while (*s == ' ') s++;
            if (*s) { fputs(s, stdout); p->printed_any = 1; }
            free(txt);
            if (p->f_tok) fprintf(p->f_tok, "%d\n", p->tokens[i]);
        }
        p->total_tokens += (size_t)n;
        fflush(stdout);
        if (p->verbose) fprintf(stderr, "\n[chunk %zu] %d frames, enc %.1f ms gpu / %.1f ms wall, dec %.1f ms (%d steps), %d tokens\n",
                                p->chunks, out.frames, out.gpu_ms, out.wall_ms, decoder_last_ms(p->dec), decoder_last_steps(p->dec), n);
    }
    return rc;
}

static int feed(pipeline_t *p, const float *samples, size_t n, int final, char *err, size_t errlen) {
    const float *frames;
    size_t nf;
    if (final && n == 0 && mel_total_samples(p->mel) == 0) return 0;
    mel_push(p->mel, samples, n, final, &frames, &nf);
    if (nf) {
        if (p->f_mel) fwrite(frames, sizeof(float), nf * MEL_NMEL, p->f_mel);
        encoder_push(p->enc, frames, nf);
    }
    return run_pending(p, final, err, errlen);
}

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

/* Between chunks the GPU idles and clocks down, so the next chunk starts slow (about 2x the
   fast-path time). Optional: keep it busy with small dispatches until `until_ms`. Measured on
   an M4 Max: live 560 ms chunks 14 -> 6 ms; short bursts just before a chunk do not help,
   the clock governor needs sustained activity. Costs GPU power, hence opt-in. */
static gpu_buf_t *g_warmbuf = NULL;
static void gpu_keep_warm(model_t *m, double until_ms) {
    char err[256];
    if (!g_warmbuf) g_warmbuf = gpu_buf_alloc(m->gpu, 8u << 20);
    while (now_ms() < until_ms - 0.3) {
        gpu_begin(m->gpu);
        for (int i = 0; i < 4; ++i) k_fill(m, g_warmbuf, 0, 2u << 20, 0.0f);
        gpu_end(m->gpu, err, sizeof err);
    }
}

static void warmup(model_t *m, int right, char *err, size_t errlen) {
    precompile_gemms(m, err, errlen);
    encoder_t *e = encoder_create(m, right);
    decoder_t *d = decoder_create(m);
    mel_state_t *ms = mel_create();
    float zeros[1600] = {0};
    const float *frames; size_t nf;
    for (int i = 0; i < 15; ++i) {
        mel_push(ms, zeros, 1600, i == 14, &frames, &nf);
        encoder_push(e, frames, nf);
        enc_out_t out;
        int tok[64];
        while (encoder_step(e, i == 14, &out, err, errlen) == 1)
            if (out.frames > 0) decoder_run(d, out.joint_enc, out.frames, tok, 64, err, errlen);
    }
    mel_destroy(ms); decoder_destroy(d); encoder_destroy(e);
}

int main(int argc, char **argv) {
    opts_t o = { .language = "en-US", .latency = 560, .block_ms = 100 };
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
#define ARG(name, field) if (!strcmp(a, name)) { if (!v) { usage(); return 2; } o.field = v; i++; continue; }
        ARG("--model-dir", model_dir) ARG("--file", file) ARG("--language", language) ARG("--device", device)
        ARG("--dump-mel", dump_mel) ARG("--dump-enc", dump_enc) ARG("--dump-tokens", dump_tokens)
#undef ARG
        if (!strcmp(a, "--latency")) { if (!v) { usage(); return 2; } o.latency = atoi(v); i++; continue; }
        if (!strcmp(a, "--block-ms")) { if (!v) { usage(); return 2; } o.block_ms = atoi(v); i++; continue; }
        if (!strcmp(a, "--realtime")) { o.realtime = 1; continue; }
        if (!strcmp(a, "--list-devices")) { o.list_devices = 1; continue; }
        if (!strcmp(a, "--list-languages")) { o.list_languages = 1; continue; }
        if (!strcmp(a, "--info")) { o.info = 1; continue; }
        if (!strcmp(a, "--verbose")) { o.verbose = 1; continue; }
        if (!strcmp(a, "--no-warmup")) { o.no_warmup = 1; continue; }
        if (!strcmp(a, "--gpu-warm")) { o.gpu_warm = 1; continue; }
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        fprintf(stderr, "unknown option %s\n", a); usage(); return 2;
    }
    if (o.list_devices) return mic_list_devices();

    int right = -1;
    for (size_t i = 0; i < sizeof LAT_MS / sizeof *LAT_MS; ++i) if (LAT_MS[i] == o.latency) right = LAT_R[i];
    if (right < 0) { fprintf(stderr, "latency must be one of 80 160 320 560 1120 ms\n"); return 2; }

    char dir[1200], err[1024];
    if (o.model_dir) snprintf(dir, sizeof dir, "%s", o.model_dir);
    else if (find_default_model_dir(dir, sizeof dir)) { fprintf(stderr, "model not found in the Hugging Face cache; pass --model-dir\n"); return 1; }

    model_t *m = model_load(dir, err, sizeof err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err); return 1; }
    fprintf(stderr, "[model] %s loaded in %.0f ms on %s (%d layers, d=%d, vocab %d)\n", dir, m->load_ms, gpu_device_name(m->gpu), m->cfg.n_layers, m->cfg.d_model, m->cfg.vocab_size);

    if (o.list_languages) {
        for (size_t i = 0; i < json_len(m->prompt_dict); ++i) printf("%s\n", m->prompt_dict->keys[i]);
        model_free(m); return 0;
    }
    if (o.info) {
        printf("trained latencies (ms):");
        for (int i = 0; i < m->cfg.n_trained; ++i) printf(" %d", (m->cfg.trained_right[i] + 1) * 80);
        printf("\nlanguages: %zu\nblank id: %d\nmax symbols: %d\n", json_len(m->prompt_dict), m->cfg.blank_id, m->cfg.max_symbols);
        model_free(m); return 0;
    }
    if (!model_right_context_trained(m, right)) { fprintf(stderr, "latency %d ms not trained for this checkpoint\n", o.latency); model_free(m); return 2; }
    if (model_set_language(m, o.language, err, sizeof err)) { fprintf(stderr, "%s (see --list-languages)\n", err); model_free(m); return 2; }
    fprintf(stderr, "[model] att_context_size=[%d, %d] (%d ms chunks) | language=%s (prompt %d)\n", m->cfg.att_left, right, o.latency, o.language, m->prompt_index);

    if (!o.no_warmup) {
        double t = now_ms();
        warmup(m, right, err, sizeof err);
        warmup_ms_global = now_ms() - t;
        fprintf(stderr, "[model] warm-up done in %.0f ms\n", warmup_ms_global);
    }

    pipeline_t p = { .m = m, .verbose = o.verbose };
    p.mel = mel_create();
    p.enc = encoder_create(m, right);
    p.dec = decoder_create(m);
    if (o.dump_mel) p.f_mel = fopen(o.dump_mel, "wb");
    if (o.dump_enc) p.f_enc = fopen(o.dump_enc, "wb");
    if (o.dump_tokens) p.f_tok = fopen(o.dump_tokens, "w");
    signal(SIGINT, on_sigint);

    double audio_seconds = 0, wall_s = 0, warmup_ms = 0;
    int rc = 0;
    if (o.file) {
        int rate; size_t ns;
        float *audio = wav_read_mono(o.file, &rate, &ns, err, sizeof err);
        if (!audio) { fprintf(stderr, "%s\n", err); rc = 1; goto done; }
        if (rate != MEL_SR) {
            resampler_t *rs = resampler_create(rate, MEL_SR);
            size_t cap = (size_t)((double)ns * MEL_SR / rate) + 1024;
            float *res = malloc(cap * sizeof(float));
            size_t n = resampler_process(rs, audio, ns, res, cap);
            n += resampler_flush(rs, res + n, cap - n);
            resampler_destroy(rs);
            free(audio);
            audio = res;
            fprintf(stderr, "[file] resampled %d Hz -> %d Hz (%zu -> %zu samples)\n", rate, MEL_SR, ns, n);
            ns = n;
        }
        audio_seconds = (double)ns / MEL_SR;
        fprintf(stderr, "[file] %s (%.2fs) | %s\n", o.file, audio_seconds, o.realtime ? "real-time paced" : "as fast as possible");
        size_t piece = o.realtime ? (size_t)MEL_SR * (size_t)o.block_ms / 1000 : (size_t)encoder_chunk_mel(p.enc) * MEL_HOP;
        p.t_start = now_ms();
        for (size_t pos = 0; pos < ns && !g_stop; pos += piece) {
            size_t n = pos + piece <= ns ? piece : ns - pos;
            int final = pos + n >= ns;
            if (o.realtime) {
                double target = p.t_start + (double)(pos + n) / MEL_SR * 1000.0;
                if (o.gpu_warm) { gpu_keep_warm(m, target); }
                double now = now_ms();
                if (target > now) usleep((useconds_t)((target - now) * 1000.0));
            }
            if (feed(&p, audio + pos, n, final, err, sizeof err) < 0) { fprintf(stderr, "\n%s\n", err); rc = 1; break; }
        }
        wall_s = (now_ms() - p.t_start) / 1000.0;
        free(audio);
        fputc('\n', stdout);
        fprintf(stderr, "[file] done in %.2fs (%.1fx real-time)\n", wall_s, audio_seconds / wall_s);
    } else {
        mic_t *mic = mic_open(o.device, MEL_SR, MEL_SR * o.block_ms / 1000, err, sizeof err);
        if (!mic) { fprintf(stderr, "%s\n", err); rc = 1; goto done; }
        if (mic_start(mic)) { fprintf(stderr, "cannot start audio queue\n"); mic_close(mic); rc = 1; goto done; }
        fprintf(stderr, "[mic] %s @ %d Hz native -> %d Hz, %d ms blocks | Ctrl+C to stop\n\n", mic_device_name(mic), mic_native_rate(mic), MEL_SR, o.block_ms);
        float buf[16000];
        size_t zero_run = 0, total = 0;
        int warned = 0, signal_seen = 0;
        p.t_start = now_ms();
        while (!g_stop) {
            size_t n = mic_read(mic, buf, sizeof buf / sizeof *buf);
            if (!n) { if (o.gpu_warm) gpu_keep_warm(m, now_ms() + 2.0); else usleep(2000); continue; }
            total += n;
            if (!signal_seen) {
                float peak = 0; for (size_t i = 0; i < n; ++i) peak = fmaxf(peak, fabsf(buf[i]));
                if (peak > 0) { signal_seen = 1; fprintf(stderr, "[mic] audio detected (peak %.4f); listening…\n", peak); }
                else { zero_run += n; if (zero_run >= 2 * MEL_SR && !warned) { warned = 1; fprintf(stderr, "[mic] WARNING: pure digital silence; macOS is probably blocking microphone access for this terminal (System Settings > Privacy & Security > Microphone)\n"); } }
            }
            if (feed(&p, buf, n, 0, err, sizeof err) < 0) { fprintf(stderr, "\n%s\n", err); rc = 1; break; }
        }
        mic_stop(mic);
        fprintf(stderr, "\n[mic] stopping, flushing decoder…\n");
        if (rc == 0 && feed(&p, NULL, 0, 1, err, sizeof err) < 0) { fprintf(stderr, "%s\n", err); rc = 1; }
        audio_seconds = (double)total / MEL_SR;
        fputc('\n', stdout);
        if (mic_dropped(mic)) fprintf(stderr, "[mic] warning: dropped %zu samples\n", mic_dropped(mic));
        mic_close(mic);
    }
    fprintf(stderr, "[stats] %.2fs audio, %zu chunks, %zu tokens | encoder %.1f ms gpu / %.1f ms wall total, %.1f ms mean, %.1f ms max per chunk | decoder %.1f ms total, %.2f ms per joint step (%zu steps), %.1f ms max per chunk",
            audio_seconds, p.chunks, p.total_tokens, p.enc_gpu_ms, p.enc_wall_ms, p.chunks ? p.enc_wall_ms / p.chunks : 0, p.enc_wall_max,
            p.dec_ms, p.dec_steps ? p.dec_ms / p.dec_steps : 0, p.dec_steps, p.dec_max);
    if (p.first_token_ms > 0) fprintf(stderr, " | first token at %.0f ms", p.first_token_ms);
    if (audio_seconds > 0) fprintf(stderr, " | compute RTF %.3f", (p.enc_wall_ms + p.dec_ms) / 1000.0 / audio_seconds);
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, " | max RSS %.0f MB\n", ru.ru_maxrss / 1e6);
    warmup_ms = warmup_ms_global;
    fprintf(stderr, "[json] {\"impl\": \"c\", \"audio_s\": %.4f, \"wall_s\": %.4f, \"chunks\": %zu, \"tokens\": %zu, \"enc_ms\": %.3f, \"enc_gpu_ms\": %.3f, \"enc_mean_ms\": %.3f, \"enc_max_ms\": %.3f, "
                    "\"dec_ms\": %.3f, \"dec_per_step_ms\": %.4f, \"dec_max_ms\": %.3f, \"steps\": %zu, \"first_token_ms\": %.1f, \"load_ms\": %.1f, \"warmup_ms\": %.1f, \"max_rss_mb\": %.1f, \"rtf\": %.4f, \"speed_x\": %.2f}\n",
            audio_seconds, wall_s, p.chunks, p.total_tokens, p.enc_wall_ms, p.enc_gpu_ms, p.chunks ? p.enc_wall_ms / p.chunks : 0.0, p.enc_wall_max,
            p.dec_ms, p.dec_steps ? p.dec_ms / p.dec_steps : 0.0, p.dec_max, p.dec_steps, p.first_token_ms, m->load_ms, warmup_ms, ru.ru_maxrss / 1e6,
            audio_seconds > 0 ? (p.enc_wall_ms + p.dec_ms) / 1000.0 / audio_seconds : 0.0, wall_s > 0 ? audio_seconds / wall_s : 0.0);
done:
    if (p.f_mel) fclose(p.f_mel);
    if (p.f_enc) fclose(p.f_enc);
    if (p.f_tok) fclose(p.f_tok);
    mel_destroy(p.mel); decoder_destroy(p.dec); encoder_destroy(p.enc);
    model_free(m);
    return rc;
}
