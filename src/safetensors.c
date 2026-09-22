#include "safetensors.h"
#include "json.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int st_open(const char *path, st_file *out, char *err, size_t errlen) {
    memset(out, 0, sizeof *out);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, errlen, "cannot open %s", path); return -1; }
    struct stat sb;
    if (fstat(fd, &sb) != 0) { close(fd); snprintf(err, errlen, "fstat failed"); return -1; }
    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { snprintf(err, errlen, "mmap failed"); return -1; }
    out->map = map;
    out->map_len = (size_t)sb.st_size;
    if (out->map_len < 8) { snprintf(err, errlen, "file too small"); st_close(out); return -1; }
    uint64_t hlen;
    memcpy(&hlen, map, 8);
    if (8 + hlen > out->map_len) { snprintf(err, errlen, "bad header length"); st_close(out); return -1; }
    out->data_start = 8 + (size_t)hlen;

    char jerr[256];
    json_value *hdr = json_parse((const char *)map + 8, (size_t)hlen, jerr, sizeof jerr);
    if (!hdr || hdr->type != JSON_OBJECT) { snprintf(err, errlen, "header parse: %s", jerr); json_free(hdr); st_close(out); return -1; }

    out->tensors = calloc(hdr->count, sizeof *out->tensors);
    for (size_t i = 0; i < hdr->count; ++i) {
        if (!strcmp(hdr->keys[i], "__metadata__")) continue;
        const json_value *t = hdr->items[i];
        st_tensor *T = &out->tensors[out->count];
        T->name = strdup(hdr->keys[i]);
        const char *dt = json_str(json_get(t, "dtype"), "");
        T->dtype = !strcmp(dt, "BF16") ? ST_BF16 : !strcmp(dt, "F16") ? ST_F16 : !strcmp(dt, "F32") ? ST_F32 : ST_OTHER;
        const json_value *shape = json_get(t, "shape");
        T->ndim = (int)json_len(shape);
        T->numel = 1;
        for (int d = 0; d < T->ndim && d < 8; ++d) {
            T->shape[d] = (int64_t)json_num(json_at(shape, (size_t)d), 0);
            T->numel *= (size_t)T->shape[d];
        }
        const json_value *off = json_get(t, "data_offsets");
        size_t a = (size_t)json_num(json_at(off, 0), 0), b = (size_t)json_num(json_at(off, 1), 0);
        T->offset = out->data_start + a;
        T->nbytes = b - a;
        if (T->offset + T->nbytes > out->map_len) { snprintf(err, errlen, "tensor %s out of file bounds", T->name); json_free(hdr); st_close(out); return -1; }
        out->count++;
    }
    json_free(hdr);
    return 0;
}

void st_close(st_file *f) {
    if (f->map) munmap(f->map, f->map_len);
    for (size_t i = 0; i < f->count; ++i) free(f->tensors[i].name);
    free(f->tensors);
    memset(f, 0, sizeof *f);
}

const st_tensor *st_find(const st_file *f, const char *name) {
    for (size_t i = 0; i < f->count; ++i)
        if (!strcmp(f->tensors[i].name, name)) return &f->tensors[i];
    return NULL;
}

const void *st_data(const st_file *f, const st_tensor *t) { return (const char *)f->map + t->offset; }
