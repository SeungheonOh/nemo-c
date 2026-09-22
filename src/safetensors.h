/* Read-only safetensors file: mmap + header index. */
#ifndef NEMO_SAFETENSORS_H
#define NEMO_SAFETENSORS_H
#include <stddef.h>
#include <stdint.h>

typedef enum { ST_BF16, ST_F16, ST_F32, ST_OTHER } st_dtype;

typedef struct {
    char *name;
    st_dtype dtype;
    int ndim;
    int64_t shape[8];
    size_t offset; /* byte offset of data within the file */
    size_t nbytes;
    size_t numel;
} st_tensor;

typedef struct {
    void *map;
    size_t map_len;
    size_t data_start;
    st_tensor *tensors;
    size_t count;
} st_file;

int st_open(const char *path, st_file *out, char *err, size_t errlen);
void st_close(st_file *f);
const st_tensor *st_find(const st_file *f, const char *name);
const void *st_data(const st_file *f, const st_tensor *t);

static inline float bf16_to_f32(uint16_t v) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)v << 16;
    return c.f;
}

#endif
