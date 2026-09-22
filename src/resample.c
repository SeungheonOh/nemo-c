#include "resample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ZERO_CROSSINGS 16

struct resampler {
    double ratio;     /* input samples per output sample */
    double half;      /* kernel half width in input samples */
    double cutoff2;   /* 2 * cutoff frequency (cycles per input sample) */
    float *buf;       /* input FIFO */
    size_t len, cap;
    unsigned long long start;   /* absolute input index of buf[0] */
    unsigned long long in_count, out_count;
    int passthrough;
};

resampler_t *resampler_create(int in_rate, int out_rate) {
    resampler_t *r = calloc(1, sizeof *r);
    r->passthrough = in_rate == out_rate;
    r->ratio = (double)in_rate / (double)out_rate;
    r->half = ZERO_CROSSINGS * fmax(r->ratio, 1.0);
    r->cutoff2 = fmin(1.0, 1.0 / r->ratio); /* 2*fc: full input band when upsampling, output Nyquist when downsampling */
    r->cap = 8192;
    r->buf = malloc(r->cap * sizeof(float));
    return r;
}
void resampler_destroy(resampler_t *r) {
    if (!r) return;
    free(r->buf);
    free(r);
}
static double kernel(const resampler_t *r, double u) {
    /* 2fc * sinc(2fc u) * Blackman(u / half) */
    double x = u / r->half;
    if (x <= -1.0 || x >= 1.0) return 0.0;
    double w = 0.42 + 0.5 * cos(M_PI * x) + 0.08 * cos(2.0 * M_PI * x);
    double a = M_PI * r->cutoff2 * u;
    double s = fabs(a) < 1e-9 ? 1.0 : sin(a) / a;
    return r->cutoff2 * s * w;
}
static void append(resampler_t *r, const float *in, size_t n) {
    if (r->len + n > r->cap) {
        while (r->len + n > r->cap) r->cap *= 2;
        r->buf = realloc(r->buf, r->cap * sizeof(float));
    }
    memcpy(r->buf + r->len, in, n * sizeof(float));
    r->len += n;
    r->in_count += n;
}
static size_t produce(resampler_t *r, float *out, size_t out_max, int final) {
    size_t produced = 0;
    while (produced < out_max) {
        double t = (double)r->out_count * r->ratio;           /* center in input samples */
        long k0 = (long)ceil(t - r->half), k1 = (long)floor(t + r->half);
        if (!final && (unsigned long long)(k1 + 1) > r->in_count) break; /* need more input */
        if (final && (unsigned long long)k0 >= r->in_count) break;       /* nothing left to convert */
        double acc = 0.0, wsum = 0.0;
        for (long k = k0; k <= k1; ++k) {
            double h = kernel(r, (double)k - t);
            wsum += h;
            if (k < 0 || (unsigned long long)k >= r->in_count) continue; /* zero outside the stream */
            long i = k - (long)r->start;
            if (i < 0 || (size_t)i >= r->len) continue;
            acc += h * r->buf[i];
        }
        out[produced++] = (float)(wsum > 0 ? acc / wsum : 0.0);
        r->out_count++;
    }
    /* drop input no longer reachable by any future kernel */
    double tn = (double)r->out_count * r->ratio;
    long keep_from = (long)floor(tn - r->half) - 2;
    if (keep_from > (long)r->start) {
        size_t drop = (size_t)keep_from - (size_t)r->start;
        if (drop > r->len) drop = r->len;
        memmove(r->buf, r->buf + drop, (r->len - drop) * sizeof(float));
        r->len -= drop;
        r->start += drop;
    }
    return produced;
}
size_t resampler_process(resampler_t *r, const float *in, size_t n, float *out, size_t out_max) {
    if (r->passthrough) {
        size_t m = n < out_max ? n : out_max;
        memcpy(out, in, m * sizeof(float));
        return m;
    }
    append(r, in, n);
    return produce(r, out, out_max, 0);
}
size_t resampler_flush(resampler_t *r, float *out, size_t out_max) {
    if (r->passthrough) return 0;
    return produce(r, out, out_max, 1);
}
