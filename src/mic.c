#include "mic.h"
#include "resample.h"
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NBUF 4

struct mic {
    AudioQueueRef q;
    AudioQueueBufferRef bufs[NBUF];
    float *ring;
    size_t cap, head, tail, count, dropped;
    pthread_mutex_t mu;
    char name[256];
    int running;
    int native_rate, out_rate;
    resampler_t *rs;
    float *tmp;
    size_t tmp_cap;
};

static void cfstr_to_utf8(CFStringRef s, char *out, size_t n) {
    if (!s || !CFStringGetCString(s, out, (CFIndex)n, kCFStringEncodingUTF8)) snprintf(out, n, "?");
}

static int device_input_channels(AudioDeviceID dev) {
    AudioObjectPropertyAddress addr = { kAudioDevicePropertyStreamConfiguration, kAudioDevicePropertyScopeInput, kAudioObjectPropertyElementMain };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(dev, &addr, 0, NULL, &size) != noErr || !size) return 0;
    AudioBufferList *bl = malloc(size);
    int ch = 0;
    if (AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, bl) == noErr)
        for (UInt32 i = 0; i < bl->mNumberBuffers; ++i) ch += (int)bl->mBuffers[i].mNumberChannels;
    free(bl);
    return ch;
}
static CFStringRef device_string(AudioDeviceID dev, AudioObjectPropertySelector sel) {
    AudioObjectPropertyAddress addr = { sel, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    CFStringRef s = NULL;
    UInt32 size = sizeof s;
    if (AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &s) != noErr) return NULL;
    return s;
}
static int list_devices(AudioDeviceID **out) {
    AudioObjectPropertyAddress addr = { kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &addr, 0, NULL, &size) != noErr) return 0;
    *out = malloc(size);
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, *out) != noErr) { free(*out); *out = NULL; return 0; }
    return (int)(size / sizeof(AudioDeviceID));
}
static double device_nominal_rate(AudioDeviceID dev) {
    AudioObjectPropertyAddress addr = { kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    Float64 rate = 0;
    UInt32 size = sizeof rate;
    if (AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &rate) != noErr) return 0;
    return rate;
}
static AudioDeviceID default_input(void) {
    AudioObjectPropertyAddress addr = { kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain };
    AudioDeviceID dev = kAudioObjectUnknown;
    UInt32 size = sizeof dev;
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, &dev);
    return dev;
}

int mic_list_devices(void) {
    AudioDeviceID *devs = NULL;
    int n = list_devices(&devs), idx = 0;
    AudioDeviceID dflt = default_input();
    for (int i = 0; i < n; ++i) {
        int ch = device_input_channels(devs[i]);
        if (!ch) continue;
        char name[256];
        CFStringRef s = device_string(devs[i], kAudioObjectPropertyName);
        cfstr_to_utf8(s, name, sizeof name);
        if (s) CFRelease(s);
        printf("%s %2d  %s (%d in, %.0f Hz)\n", devs[i] == dflt ? ">" : " ", idx, name, ch, device_nominal_rate(devs[i]));
        idx++;
    }
    free(devs);
    return 0;
}

static void input_cb(void *user, AudioQueueRef q, AudioQueueBufferRef buf, const AudioTimeStamp *ts, UInt32 npackets, const AudioStreamPacketDescription *desc) {
    mic_t *m = user;
    const float *src = buf->mAudioData;
    size_t n = buf->mAudioDataByteSize / sizeof(float);
    if (m->rs) {
        size_t need = (size_t)((double)n * m->out_rate / m->native_rate) + 64;
        if (need > m->tmp_cap) { m->tmp_cap = need * 2; m->tmp = realloc(m->tmp, m->tmp_cap * sizeof(float)); }
        n = resampler_process(m->rs, src, n, m->tmp, m->tmp_cap);
        src = m->tmp;
    }
    pthread_mutex_lock(&m->mu);
    for (size_t i = 0; i < n; ++i) {
        if (m->count == m->cap) { m->tail = (m->tail + 1) % m->cap; m->count--; m->dropped++; }
        m->ring[m->head] = src[i];
        m->head = (m->head + 1) % m->cap;
        m->count++;
    }
    pthread_mutex_unlock(&m->mu);
    if (m->running) AudioQueueEnqueueBuffer(q, buf, 0, NULL);
}

mic_t *mic_open(const char *spec, int sample_rate, int block_frames, char *err, size_t errlen) {
    mic_t *m = calloc(1, sizeof *m);
    pthread_mutex_init(&m->mu, NULL);
    m->cap = (size_t)sample_rate * 30;
    m->ring = malloc(m->cap * sizeof(float));

    /* device selection: index from --list-devices or a name substring */
    AudioDeviceID chosen = default_input();
    if (spec) {
        AudioDeviceID *devs = NULL;
        int n = list_devices(&devs), idx = 0, found = 0;
        char *endp;
        long want = strtol(spec, &endp, 10);
        int numeric = *endp == 0 && *spec != 0;
        for (int i = 0; i < n && !found; ++i) {
            if (!device_input_channels(devs[i])) continue;
            char name[256];
            CFStringRef s = device_string(devs[i], kAudioObjectPropertyName);
            cfstr_to_utf8(s, name, sizeof name);
            if (s) CFRelease(s);
            if ((numeric && idx == want) || (!numeric && strcasestr(name, spec))) { chosen = devs[i]; found = 1; }
            idx++;
        }
        free(devs);
        if (!found) { snprintf(err, errlen, "no input device matching '%s' (see --list-devices)", spec); mic_close(m); return NULL; }
    }
    CFStringRef nm = device_string(chosen, kAudioObjectPropertyName);
    cfstr_to_utf8(nm, m->name, sizeof m->name);
    if (nm) CFRelease(nm);

    /* Capture at the device's own rate (no CoreAudio converter in the path) and resample ourselves. */
    double native = device_nominal_rate(chosen);
    m->native_rate = native > 0 ? (int)native : sample_rate;
    m->out_rate = sample_rate;
    if (m->native_rate != sample_rate) m->rs = resampler_create(m->native_rate, sample_rate);

    AudioStreamBasicDescription fmt = {0};
    fmt.mSampleRate = m->native_rate;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    fmt.mBytesPerPacket = 4; fmt.mFramesPerPacket = 1; fmt.mBytesPerFrame = 4;
    fmt.mChannelsPerFrame = 1; fmt.mBitsPerChannel = 32;
    if (AudioQueueNewInput(&fmt, input_cb, m, NULL, NULL, 0, &m->q) != noErr) { snprintf(err, errlen, "AudioQueueNewInput failed"); mic_close(m); return NULL; }
    CFStringRef uid = device_string(chosen, kAudioDevicePropertyDeviceUID);
    if (uid) {
        AudioQueueSetProperty(m->q, kAudioQueueProperty_CurrentDevice, &uid, sizeof uid);
        CFRelease(uid);
    }
    block_frames = (int)((long)block_frames * m->native_rate / sample_rate);

    for (int i = 0; i < NBUF; ++i) {
        if (AudioQueueAllocateBuffer(m->q, (UInt32)block_frames * 4, &m->bufs[i]) != noErr) { snprintf(err, errlen, "AudioQueueAllocateBuffer failed"); mic_close(m); return NULL; }
        AudioQueueEnqueueBuffer(m->q, m->bufs[i], 0, NULL);
    }
    return m;
}
int mic_start(mic_t *m) {
    m->running = 1;
    return AudioQueueStart(m->q, NULL) == noErr ? 0 : -1;
}
void mic_stop(mic_t *m) {
    m->running = 0;
    AudioQueueStop(m->q, true);
}
void mic_close(mic_t *m) {
    if (!m) return;
    if (m->q) AudioQueueDispose(m->q, true);
    resampler_destroy(m->rs);
    free(m->tmp);
    free(m->ring);
    pthread_mutex_destroy(&m->mu);
    free(m);
}
size_t mic_read(mic_t *m, float *out, size_t max) {
    pthread_mutex_lock(&m->mu);
    size_t n = m->count < max ? m->count : max;
    for (size_t i = 0; i < n; ++i) { out[i] = m->ring[m->tail]; m->tail = (m->tail + 1) % m->cap; }
    m->count -= n;
    pthread_mutex_unlock(&m->mu);
    return n;
}
size_t mic_dropped(mic_t *m) { return m->dropped; }
const char *mic_device_name(mic_t *m) { return m->name; }
int mic_native_rate(mic_t *m) { return m->native_rate; }
