#include "mel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PREEMPH 0.97f
#define LOG_GUARD 5.960464477539063e-08f /* 2^-24 */
#define LOOKAHEAD (MEL_NFFT / 2)

struct mel_state {
    float *y;          /* preemphasized samples, y[i] is global sample (y_start + i) */
    size_t y_len, y_cap, y_start;
    float last_x;
    int have_last;
    size_t total, next_frame;
    int closed;
    float window[MEL_NFFT];
    float fb[MEL_NMEL * MEL_NFREQ];
    double cos_t[MEL_NFFT / 2], sin_t[MEL_NFFT / 2];
    int bitrev[MEL_NFFT];
    float *out;
    size_t out_cap;
};

static double hz_to_mel_slaney(double f) {
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    double m = f / f_sp;
    if (f >= min_log_hz) m = min_log_mel + log(f / min_log_hz) / logstep;
    return m;
}
static double mel_to_hz_slaney(double m) {
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = log(6.4) / 27.0;
    if (m >= min_log_mel) return min_log_hz * exp(logstep * (m - min_log_mel));
    return f_sp * m;
}

static void build_filterbank(float *fb) {
    double all_freqs[MEL_NFREQ], f_pts[MEL_NMEL + 2];
    for (int i = 0; i < MEL_NFREQ; ++i) all_freqs[i] = (MEL_SR / 2) * (double)i / (MEL_NFREQ - 1);
    double m_min = hz_to_mel_slaney(0.0), m_max = hz_to_mel_slaney(MEL_SR / 2.0);
    for (int i = 0; i < MEL_NMEL + 2; ++i) {
        double m = m_min + (m_max - m_min) * (double)i / (MEL_NMEL + 1);
        f_pts[i] = mel_to_hz_slaney(m);
    }
    for (int m = 0; m < MEL_NMEL; ++m) {
        double enorm = 2.0 / (f_pts[m + 2] - f_pts[m]);
        for (int f = 0; f < MEL_NFREQ; ++f) {
            double down = (all_freqs[f] - f_pts[m]) / (f_pts[m + 1] - f_pts[m]);
            double up = (f_pts[m + 2] - all_freqs[f]) / (f_pts[m + 2] - f_pts[m + 1]);
            double v = fmin(down, up);
            if (v < 0) v = 0;
            fb[m * MEL_NFREQ + f] = (float)(v * enorm);
        }
    }
}

mel_state_t *mel_create(void) {
    mel_state_t *m = calloc(1, sizeof *m);
    m->y_cap = MEL_SR * 4;
    m->y = malloc(m->y_cap * sizeof(float));
    /* symmetric Hann(400) centered in 512 */
    int left = (MEL_NFFT - MEL_WIN) / 2;
    for (int n = 0; n < MEL_WIN; ++n) m->window[left + n] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * n / (MEL_WIN - 1))));
    build_filterbank(m->fb);
    for (int i = 0; i < MEL_NFFT / 2; ++i) { m->cos_t[i] = cos(-2.0 * M_PI * i / MEL_NFFT); m->sin_t[i] = sin(-2.0 * M_PI * i / MEL_NFFT); }
    int bits = 9;
    for (int i = 0; i < MEL_NFFT; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        m->bitrev[i] = r;
    }
    return m;
}
void mel_destroy(mel_state_t *m) {
    if (!m) return;
    free(m->y);
    free(m->out);
    free(m);
}
size_t mel_total_samples(const mel_state_t *m) { return m->total; }
size_t mel_emitted_frames(const mel_state_t *m) { return m->next_frame; }

/* in-place iterative radix-2 complex FFT, size 512 */
static void fft512(const mel_state_t *m, double *re, double *im) {
    for (int i = 0; i < MEL_NFFT; ++i) {
        int j = m->bitrev[i];
        if (j > i) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= MEL_NFFT; len <<= 1) {
        int half = len >> 1, step = MEL_NFFT / len;
        for (int i = 0; i < MEL_NFFT; i += len) {
            for (int k = 0; k < half; ++k) {
                double wr = m->cos_t[k * step], wi = m->sin_t[k * step];
                double ur = re[i + k], ui = im[i + k];
                double vr = re[i + k + half] * wr - im[i + k + half] * wi;
                double vi = re[i + k + half] * wi + im[i + k + half] * wr;
                re[i + k] = ur + vr; im[i + k] = ui + vi;
                re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
            }
        }
    }
}

static float sample_at(const mel_state_t *m, long g) {
    /* reflect padding without edge repeat: y[-k] = y[k]; y[N-1+k] = y[N-1-k] */
    long N = (long)m->total;
    if (g < 0) g = -g;
    if (g >= N) g = 2 * N - 2 - g;
    if (g < 0 || g >= N) return 0.0f;
    long i = g - (long)m->y_start;
    if (i < 0 || i >= (long)m->y_len) return 0.0f;
    return m->y[(size_t)i];
}

void mel_push(mel_state_t *m, const float *samples, size_t n, int final, const float **frames, size_t *nframes) {
    *frames = NULL;
    *nframes = 0;
    if (m->closed) return;
    if (m->y_len + n > m->y_cap) {
        while (m->y_len + n > m->y_cap) m->y_cap *= 2;
        m->y = realloc(m->y, m->y_cap * sizeof(float));
    }
    for (size_t i = 0; i < n; ++i) {
        float x = samples[i];
        m->y[m->y_len++] = m->have_last ? x - PREEMPH * m->last_x : x;
        m->last_x = x;
        m->have_last = 1;
    }
    m->total += n;

    size_t frame_end;
    if (final) frame_end = m->total / MEL_HOP + 1;
    else frame_end = m->total >= LOOKAHEAD ? (m->total - LOOKAHEAD) / MEL_HOP + 1 : 0;
    if (final) m->closed = 1;
    if (frame_end <= m->next_frame) return;

    size_t count = frame_end - m->next_frame;
    if (count * MEL_NMEL > m->out_cap) {
        m->out_cap = count * MEL_NMEL;
        m->out = realloc(m->out, m->out_cap * sizeof(float));
    }
    double re[MEL_NFFT], im[MEL_NFFT];
    float power[MEL_NFREQ];
    for (size_t t = m->next_frame; t < frame_end; ++t) {
        long start = (long)t * MEL_HOP - LOOKAHEAD;
        for (int k = 0; k < MEL_NFFT; ++k) {
            re[k] = (double)(sample_at(m, start + k) * m->window[k]);
            im[k] = 0.0;
        }
        fft512(m, re, im);
        for (int k = 0; k < MEL_NFREQ; ++k) power[k] = (float)(re[k] * re[k] + im[k] * im[k]);
        float *row = m->out + (t - m->next_frame) * MEL_NMEL;
        for (int b = 0; b < MEL_NMEL; ++b) {
            const float *w = m->fb + b * MEL_NFREQ;
            float acc = 0.0f;
            for (int k = 0; k < MEL_NFREQ; ++k) acc += w[k] * power[k];
            row[b] = logf(acc + LOG_GUARD);
        }
    }
    *frames = m->out;
    *nframes = count;
    m->next_frame = frame_end;

    /* drop history we can no longer need (keep a full window before the next frame center) */
    long keep_from = (long)m->next_frame * MEL_HOP - MEL_NFFT - 2;
    if (keep_from > (long)m->y_start) {
        size_t drop = (size_t)keep_from - m->y_start;
        if (drop > m->y_len) drop = m->y_len;
        memmove(m->y, m->y + drop, (m->y_len - drop) * sizeof(float));
        m->y_len -= drop;
        m->y_start += drop;
    }
}
