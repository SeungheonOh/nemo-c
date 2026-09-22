/* CoreAudio (AudioQueue) microphone capture at 16 kHz mono float32 into a ring buffer. */
#ifndef NEMO_MIC_H
#define NEMO_MIC_H
#include <stddef.h>
typedef struct mic mic_t;
int mic_list_devices(void);
mic_t *mic_open(const char *device_spec, int sample_rate, int block_frames, char *err, size_t errlen);
int mic_start(mic_t *m);
void mic_stop(mic_t *m);
void mic_close(mic_t *m);
size_t mic_read(mic_t *m, float *out, size_t max);   /* non-blocking */
size_t mic_dropped(mic_t *m);
const char *mic_device_name(mic_t *m);
int mic_native_rate(mic_t *m);
#endif
