#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu.h"
#include "kernels_metal.h"
#include <string.h>

struct gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    double last_ms;
    char err[512];
    char name[128];
};
struct gpu_buf {
    id<MTLBuffer> buf;
    size_t len;
};

gpu_t *gpu_create(const char *metal_source, char *err, size_t errlen) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { snprintf(err, errlen, "no Metal device"); return NULL; }
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.fastMathEnabled = NO; /* keep exp/tanh/division precise for parity with the reference */
        NSError *nserr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:metal_source] options:opts error:&nserr];
        if (!lib) {
            snprintf(err, errlen, "metal compile: %s", nserr ? nserr.localizedDescription.UTF8String : "unknown");
            return NULL;
        }
        gpu_t *g = calloc(1, sizeof *g);
        g->dev = dev;
        g->queue = [dev newCommandQueue];
        g->lib = lib;
        g->pipelines = [NSMutableDictionary new];
        snprintf(g->name, sizeof g->name, "%s", dev.name.UTF8String);
        return g;
    }
}
void gpu_destroy(gpu_t *g) {
    if (!g) return;
    g->dev = nil; g->queue = nil; g->lib = nil; g->pipelines = nil; g->cb = nil; g->enc = nil;
    free(g);
}
const char *gpu_device_name(gpu_t *g) { return g->name; }

gpu_buf_t *gpu_buf_alloc(gpu_t *g, size_t bytes) {
    gpu_buf_t *b = calloc(1, sizeof *b);
    b->len = bytes ? bytes : 16;
    b->buf = [g->dev newBufferWithLength:b->len options:MTLResourceStorageModeShared];
    if (!b->buf) { free(b); return NULL; }
    memset(b->buf.contents, 0, b->len);
    return b;
}
void gpu_buf_free(gpu_buf_t *b) { if (b) { b->buf = nil; free(b); } }
void *gpu_buf_ptr(gpu_buf_t *b) { return b->buf.contents; }
size_t gpu_buf_len(gpu_buf_t *b) { return b->len; }

static id<MTLComputePipelineState> pipeline_fc(gpu_t *g, const char *kernel, const uint32_t *fc, int nfc) {
    char keybuf[256];
    int off = snprintf(keybuf, sizeof keybuf, "%s", kernel);
    for (int i = 0; i < nfc && off < (int)sizeof keybuf - 12; ++i) off += snprintf(keybuf + off, sizeof keybuf - (size_t)off, "|%u", fc[i]);
    NSString *key = [NSString stringWithUTF8String:keybuf];
    id<MTLComputePipelineState> ps = g->pipelines[key];
    if (ps) return ps;
    id<MTLFunction> fn;
    if (nfc) {
        MTLFunctionConstantValues *vals = [MTLFunctionConstantValues new];
        for (int i = 0; i < nfc; ++i) [vals setConstantValue:&fc[i] type:MTLDataTypeUInt atIndex:(NSUInteger)i];
        NSError *ferr = nil;
        fn = [g->lib newFunctionWithName:[NSString stringWithUTF8String:kernel] constantValues:vals error:&ferr];
        if (!fn) { snprintf(g->err, sizeof g->err, "specialise %s: %s", kernel, ferr.localizedDescription.UTF8String); return nil; }
    } else {
        fn = [g->lib newFunctionWithName:[NSString stringWithUTF8String:kernel]];
    }
    if (!fn) { snprintf(g->err, sizeof g->err, "no kernel %s", kernel); return nil; }
    NSError *nserr = nil;
    ps = [g->dev newComputePipelineStateWithFunction:fn error:&nserr];
    if (!ps) { snprintf(g->err, sizeof g->err, "pipeline %s: %s", kernel, nserr.localizedDescription.UTF8String); return nil; }
    g->pipelines[key] = ps;
    return ps;
}
static id<MTLComputePipelineState> pipeline(gpu_t *g, const char *kernel) { return pipeline_fc(g, kernel, NULL, 0); }

int gpu_begin(gpu_t *g) {
    g->cb = [g->queue commandBuffer];
    g->enc = [g->cb computeCommandEncoder];
    return (g->cb && g->enc) ? 0 : -1;
}
static int encode_fc(gpu_t *g, const char *kernel, const uint32_t *fc, int nfc, const gpu_arg_t *args, int nargs) {
    id<MTLComputePipelineState> ps = pipeline_fc(g, kernel, fc, nfc);
    if (!ps) return -1;
    [g->enc setComputePipelineState:ps];
    for (int i = 0; i < nargs; ++i) {
        if (args[i].buf) [g->enc setBuffer:args[i].buf->buf offset:args[i].offset atIndex:(NSUInteger)i];
        else [g->enc setBytes:args[i].bytes length:args[i].size atIndex:(NSUInteger)i];
    }
    return 0;
}
static int encode(gpu_t *g, const char *kernel, const gpu_arg_t *args, int nargs) { return encode_fc(g, kernel, NULL, 0, args, nargs); }
int gpu_dispatch_groups_fc(gpu_t *g, const char *kernel, const uint32_t *fc, int nfc, const gpu_arg_t *args, int nargs,
                           uint32_t groups_x, uint32_t groups_y, uint32_t groups_z, uint32_t tx, uint32_t ty, uint32_t tz) {
    if (encode_fc(g, kernel, fc, nfc, args, nargs)) return -1;
    [g->enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    return 0;
}
int gpu_dispatch(gpu_t *g, const char *kernel, const gpu_arg_t *args, int nargs,
                 uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
    if (encode(g, kernel, args, nargs)) return -1;
    [g->enc dispatchThreads:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    return 0;
}
int gpu_dispatch_groups(gpu_t *g, const char *kernel, const gpu_arg_t *args, int nargs,
                        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                        uint32_t tx, uint32_t ty, uint32_t tz) {
    if (encode(g, kernel, args, nargs)) return -1;
    [g->enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, groups_z) threadsPerThreadgroup:MTLSizeMake(tx, ty, tz)];
    return 0;
}
int gpu_end(gpu_t *g, char *err, size_t errlen) {
    @autoreleasepool {
        [g->enc endEncoding];
        [g->cb commit];
        [g->cb waitUntilCompleted];
        g->last_ms = (g->cb.GPUEndTime - g->cb.GPUStartTime) * 1000.0;
        int rc = 0;
        if (g->cb.error) {
            snprintf(err, errlen, "gpu: %s", g->cb.error.localizedDescription.UTF8String);
            rc = -1;
        }
        g->cb = nil;
        g->enc = nil;
        return rc;
    }
}
double gpu_last_ms(gpu_t *g) { return g->last_ms; }
const char *gpu_last_error(gpu_t *g) { return g->err; }
