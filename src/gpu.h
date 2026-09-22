/* Thin C interface over Metal: buffers, kernel dispatch batched into one command buffer. */
#ifndef NEMO_GPU_H
#define NEMO_GPU_H
#include <stddef.h>
#include <stdint.h>

typedef struct gpu gpu_t;
typedef struct gpu_buf gpu_buf_t;

typedef struct {
    gpu_buf_t *buf;     /* buffer argument, or NULL for inline bytes */
    size_t offset;      /* byte offset into buf */
    const void *bytes;  /* inline constant bytes (setBytes) */
    size_t size;
} gpu_arg_t;

#define GPU_BUF(b, off) ((gpu_arg_t){ (b), (size_t)(off), NULL, 0 })
#define GPU_BYTES(ptr) ((gpu_arg_t){ NULL, 0, (ptr), sizeof *(ptr) })

gpu_t *gpu_create(const char *metal_source, char *err, size_t errlen);
void gpu_destroy(gpu_t *g);
const char *gpu_device_name(gpu_t *g);

gpu_buf_t *gpu_buf_alloc(gpu_t *g, size_t bytes);         /* shared storage, zero-filled */
void gpu_buf_free(gpu_buf_t *b);
void *gpu_buf_ptr(gpu_buf_t *b);
size_t gpu_buf_len(gpu_buf_t *b);

/* Encode kernels between begin/end; end commits and waits. Returns 0 on success. */
int gpu_begin(gpu_t *g);
int gpu_dispatch(gpu_t *g, const char *kernel, const gpu_arg_t *args, int nargs,
                 uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz); /* total threads */
int gpu_dispatch_groups(gpu_t *g, const char *kernel, const gpu_arg_t *args, int nargs,
                        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                        uint32_t tx, uint32_t ty, uint32_t tz);                                 /* threadgroups */
/* Same, but the pipeline is specialised with uint function constants 0..nfc-1 (cached per value set). */
int gpu_dispatch_groups_fc(gpu_t *g, const char *kernel, const uint32_t *fc, int nfc, const gpu_arg_t *args, int nargs,
                           uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, uint32_t tx, uint32_t ty, uint32_t tz);
int gpu_end(gpu_t *g, char *err, size_t errlen);
double gpu_last_ms(gpu_t *g); /* GPU execution time of the last completed command buffer */
const char *gpu_last_error(gpu_t *g);

#endif
