#include "wav.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

float *wav_read_mono(const char *path, int *rate, size_t *nsamples, char *err, size_t errlen) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size < 12) { fclose(f); snprintf(err, errlen, "not a WAV file"); return NULL; }
    unsigned char *buf = malloc((size_t)size);
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) { fclose(f); free(buf); snprintf(err, errlen, "short read"); return NULL; }
    fclose(f);
    if (memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) { free(buf); snprintf(err, errlen, "not a RIFF/WAVE file"); return NULL; }

    int fmt = 0, channels = 0, bits = 0, sr = 0;
    const unsigned char *data = NULL;
    size_t data_len = 0;
    size_t pos = 12;
    while (pos + 8 <= (size_t)size) {
        uint32_t clen = rd32(buf + pos + 4);
        const unsigned char *body = buf + pos + 8;
        if (!memcmp(buf + pos, "fmt ", 4)) {
            fmt = rd16(body);
            channels = rd16(body + 2);
            sr = (int)rd32(body + 4);
            bits = rd16(body + 14);
            if (fmt == 0xFFFE && clen >= 26) fmt = rd16(body + 24); /* WAVE_FORMAT_EXTENSIBLE: sub-format */
        } else if (!memcmp(buf + pos, "data", 4)) {
            data = body;
            data_len = clen;
            if (pos + 8 + data_len > (size_t)size) data_len = (size_t)size - pos - 8;
        }
        pos += 8 + clen + (clen & 1);
    }
    if (!data || channels <= 0 || bits <= 0) { free(buf); snprintf(err, errlen, "missing fmt/data chunk"); return NULL; }
    if (!((fmt == 1 && (bits == 16 || bits == 24 || bits == 32)) || (fmt == 3 && bits == 32))) {
        free(buf); snprintf(err, errlen, "unsupported WAV format %d / %d bits", fmt, bits); return NULL;
    }
    size_t bytes_per = (size_t)bits / 8;
    size_t frames = data_len / (bytes_per * (size_t)channels);
    float *out = malloc(frames * sizeof(float));
    for (size_t i = 0; i < frames; ++i) {
        double acc = 0;
        for (int c = 0; c < channels; ++c) {
            const unsigned char *s = data + (i * (size_t)channels + (size_t)c) * bytes_per;
            double v;
            if (fmt == 3) { float fv; memcpy(&fv, s, 4); v = fv; }
            else if (bits == 16) v = (int16_t)rd16(s) / 32768.0;
            else if (bits == 24) { int32_t x = (s[0] << 8) | (s[1] << 16) | ((int32_t)s[2] << 24); v = (x >> 8) / 8388608.0; }
            else v = (int32_t)rd32(s) / 2147483648.0;
            acc += v;
        }
        out[i] = (float)(acc / channels);
    }
    free(buf);
    *rate = sr;
    *nsamples = frames;
    return out;
}
