/* Kernel micro-benchmarks: dispatch overhead and gemm variants at the model's shapes. */
#include "gpu.h"
#include "kernel_params.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern const char kernels_metal_src[];

static const char *extra_src =
"#define GEMM_V3(NAME, COLS) \\\n"
"kernel void NAME(device const float *A [[buffer(0)]], device const ushort *W [[buffer(1)]], device const float *bias [[buffer(2)]], \\\n"
"                 device float *C [[buffer(3)]], constant GemmParams &p [[buffer(4)]], uint g [[threadgroup_position_in_grid]], \\\n"
"                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \\\n"
"    const uint n0 = (g * GEMM_SIMDS + sg) * COLS; \\\n"
"    if (n0 >= p.N) return; \\\n"
"    const uint K4 = p.K >> 2; \\\n"
"    device const ushort4 *w4[COLS]; \\\n"
"    for (uint cc = 0; cc < COLS; ++cc) w4[cc] = (device const ushort4 *)(W + (ulong)min(n0 + cc, p.N - 1) * p.ldw); \\\n"
"    for (uint m0 = 0; m0 < p.M; m0 += GEMM_ROWS) { \\\n"
"        const uint rows = min((uint)GEMM_ROWS, p.M - m0); \\\n"
"        float acc[GEMM_ROWS][COLS]; \\\n"
"        for (uint r = 0; r < GEMM_ROWS; ++r) for (uint cc = 0; cc < COLS; ++cc) acc[r][cc] = 0.0f; \\\n"
"        for (uint k4 = lane; k4 < K4; k4 += 32) { \\\n"
"            float4 wf[COLS]; \\\n"
"            for (uint cc = 0; cc < COLS; ++cc) wf[cc] = bf2f4(w4[cc][k4]); \\\n"
"            for (uint r = 0; r < GEMM_ROWS; ++r) { \\\n"
"                if (r < rows) { const float4 a = ((device const float4 *)(A + (ulong)(m0 + r) * p.lda))[k4]; \\\n"
"                    for (uint cc = 0; cc < COLS; ++cc) acc[r][cc] += dot(a, wf[cc]); } \\\n"
"            } \\\n"
"        } \\\n"
"        for (uint r = 0; r < GEMM_ROWS; ++r) { if (r >= rows) break; \\\n"
"            for (uint cc = 0; cc < COLS; ++cc) { float v = simd_sum(acc[r][cc]); \\\n"
"                if (lane == 0 && n0 + cc < p.N) { v += p.has_bias ? bias[n0 + cc] : 0.0f; \\\n"
"                    if (p.act == 1) v = silu_f(v); else if (p.act == 2) v = max(v, 0.0f); v *= p.alpha; \\\n"
"                    const ulong ci = (ulong)(m0 + r) * p.ldc + n0 + cc; C[ci] = p.accumulate ? (C[ci] + v) : v; } } } \\\n"
"    } }\n"
"GEMM_V3(gemm_c1, 1)\nGEMM_V3(gemm_c2, 2)\nGEMM_V3(gemm_c4, 4)\n"
/* unrolled-by-2 single column: two independent accumulator sets */
"kernel void gemm_u2(device const float *A [[buffer(0)]], device const ushort *W [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
"                    device float *C [[buffer(3)]], constant GemmParams &p [[buffer(4)]], uint g [[threadgroup_position_in_grid]],\n"
"                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    const uint n = g * GEMM_SIMDS + sg; if (n >= p.N) return;\n"
"    device const ushort4 *w4 = (device const ushort4 *)(W + (ulong)n * p.ldw); const uint K4 = p.K >> 2;\n"
"    for (uint m0 = 0; m0 < p.M; m0 += GEMM_ROWS) { const uint rows = min((uint)GEMM_ROWS, p.M - m0);\n"
"        float acc[GEMM_ROWS], acc2[GEMM_ROWS]; for (uint r = 0; r < GEMM_ROWS; ++r) { acc[r] = 0.0f; acc2[r] = 0.0f; }\n"
"        uint k4 = lane;\n"
"        for (; k4 + 32 < K4; k4 += 64) { const float4 wf = bf2f4(w4[k4]); const float4 wg = bf2f4(w4[k4 + 32]);\n"
"            for (uint r = 0; r < GEMM_ROWS; ++r) if (r < rows) { device const float4 *a4 = (device const float4 *)(A + (ulong)(m0 + r) * p.lda); acc[r] += dot(a4[k4], wf); acc2[r] += dot(a4[k4 + 32], wg); } }\n"
"        if (k4 < K4) { const float4 wf = bf2f4(w4[k4]); for (uint r = 0; r < GEMM_ROWS; ++r) if (r < rows) acc[r] += dot(((device const float4 *)(A + (ulong)(m0 + r) * p.lda))[k4], wf); }\n"
"        for (uint r = 0; r < GEMM_ROWS; ++r) { if (r >= rows) break; float v = simd_sum(acc[r] + acc2[r]);\n"
"            if (lane == 0) { v += p.has_bias ? bias[n] : 0.0f; if (p.act == 1) v = silu_f(v); else if (p.act == 2) v = max(v, 0.0f); v *= p.alpha;\n"
"                const ulong ci = (ulong)(m0 + r) * p.ldc + n; C[ci] = p.accumulate ? (C[ci] + v) : v; } } } }\n"
"constant uint FC_TM [[function_constant(3)]];\n"
"constant uint FC_CT [[function_constant(4)]];\n"
"#define MMA_SIMDS 2\n#define MMA_KSTEP 32\n#define MMA_MAXCT 4\n"
"kernel void gemm_mma(device const float *A [[buffer(0)]], device const ushort *W [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
"                     device float *C [[buffer(3)]], constant GemmParams &p [[buffer(4)]], uint g [[threadgroup_position_in_grid]],\n"
"                     uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float wtile[MMA_SIMDS][MMA_MAXCT][8][MMA_KSTEP];\n"
"    threadgroup float ctile[MMA_SIMDS][8][8];\n"
"    const uint n0 = (g * MMA_SIMDS + sg) * 8 * FC_CT;\n"
"    if (n0 >= p.N) return;\n"
"    simdgroup_float8x8 acc[2][MMA_MAXCT];\n"
"    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) acc[tm][ct] = simdgroup_float8x8(0.0f);\n"
"    const uint r = lane >> 2, kk = (lane & 3) * 8;\n"
"    for (uint k0 = 0; k0 < p.K; k0 += MMA_KSTEP) {\n"
"        for (uint ct = 0; ct < FC_CT; ++ct) {\n"
"            device const ushort4 *wp = (device const ushort4 *)(W + (ulong)min(n0 + ct * 8 + r, p.N - 1) * p.ldw + k0 + kk);\n"
"            threadgroup float4 *dst = (threadgroup float4 *)&wtile[sg][ct][r][kk];\n"
"            dst[0] = bf2f4(wp[0]); dst[1] = bf2f4(wp[1]);\n"
"        }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint kt = 0; kt < MMA_KSTEP / 8; ++kt) {\n"
"            simdgroup_float8x8 a[2];\n"
"            for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_load(a[tm], A + (ulong)tm * 8 * p.lda + k0 + kt * 8, p.lda);\n"
"            for (uint ct = 0; ct < FC_CT; ++ct) {\n"
"                simdgroup_float8x8 b;\n"
"                simdgroup_load(b, &wtile[sg][ct][0][kt * 8], MMA_KSTEP, ulong2(0, 0), true);\n"
"                for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_multiply_accumulate(acc[tm][ct], a[tm], b, acc[tm][ct]);\n"
"            }\n"
"        }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) {\n"
"        simdgroup_store(acc[tm][ct], &ctile[sg][0][0], 8);\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint e = lane; e < 64; e += 32) {\n"
"            const uint rr = e >> 3, cc = e & 7, row = tm * 8 + rr, col = n0 + ct * 8 + cc;\n"
"            if (row < p.M && col < p.N) {\n"
"                float v = ctile[sg][rr][cc] + (p.has_bias ? bias[col] : 0.0f);\n"
"                if (p.act == 1) v = silu_f(v); else if (p.act == 2) v = max(v, 0.0f);\n"
"                v *= p.alpha; const ulong ci = (ulong)row * p.ldc + col; C[ci] = p.accumulate ? (C[ci] + v) : v;\n"
"            }\n"
"        }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"}\n"
"constant uint FC_SPLIT [[function_constant(5)]];\n"
"kernel void gemm_mma2(device const float *A [[buffer(0)]], device const ushort *W [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
"                      device float *C [[buffer(3)]], constant GemmParams &p [[buffer(4)]], uint g [[threadgroup_position_in_grid]],\n"
"                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {\n"
"    threadgroup float wtile[4][MMA_MAXCT][8][MMA_KSTEP];\n"
"    threadgroup float red[4][2][MMA_MAXCT][8][8];\n"
"    const uint n0 = g * 8 * FC_CT;\n"
"    if (n0 >= p.N) return;\n"
"    simdgroup_float8x8 acc[2][MMA_MAXCT];\n"
"    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) acc[tm][ct] = simdgroup_float8x8(0.0f);\n"
"    const uint r = lane >> 2, kk = (lane & 3) * 8;\n"
"    const uint kper = p.K / FC_SPLIT, kbeg = sg * kper, kend = kbeg + kper;\n"
"    for (uint k0 = kbeg; k0 < kend; k0 += MMA_KSTEP) {\n"
"        for (uint ct = 0; ct < FC_CT; ++ct) {\n"
"            device const ushort4 *wp = (device const ushort4 *)(W + (ulong)min(n0 + ct * 8 + r, p.N - 1) * p.ldw + k0 + kk);\n"
"            threadgroup float4 *dst = (threadgroup float4 *)&wtile[sg][ct][r][kk];\n"
"            dst[0] = bf2f4(wp[0]); dst[1] = bf2f4(wp[1]);\n"
"        }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"        for (uint kt = 0; kt < MMA_KSTEP / 8; ++kt) {\n"
"            simdgroup_float8x8 a[2];\n"
"            for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_load(a[tm], A + (ulong)tm * 8 * p.lda + k0 + kt * 8, p.lda);\n"
"            for (uint ct = 0; ct < FC_CT; ++ct) {\n"
"                simdgroup_float8x8 b;\n"
"                simdgroup_load(b, &wtile[sg][ct][0][kt * 8], MMA_KSTEP, ulong2(0, 0), true);\n"
"                for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_multiply_accumulate(acc[tm][ct], a[tm], b, acc[tm][ct]);\n"
"            }\n"
"        }\n"
"        simdgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) simdgroup_store(acc[tm][ct], &red[sg][tm][ct][0][0], 8);\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (sg == 0) {\n"
"        for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) for (uint e = lane; e < 64; e += 32) {\n"
"            const uint rr = e >> 3, cc = e & 7, row = tm * 8 + rr, col = n0 + ct * 8 + cc;\n"
"            if (row < p.M && col < p.N) {\n"
"                float v = 0.0f; for (uint s2 = 0; s2 < FC_SPLIT; ++s2) v += red[s2][tm][ct][rr][cc];\n"
"                v += p.has_bias ? bias[col] : 0.0f;\n"
"                if (p.act == 1) v = silu_f(v); else if (p.act == 2) v = max(v, 0.0f);\n"
"                v *= p.alpha; const ulong ci = (ulong)row * p.ldc + col; C[ci] = p.accumulate ? (C[ci] + v) : v;\n"
"            }\n"
"        }\n"
"    }\n"
"}\n";


static gpu_t *G;
static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static double time_gemm(const char *kernel, int cols, uint32_t M, uint32_t N, uint32_t K, gpu_buf_t *A, gpu_buf_t *W, gpu_buf_t *C, int reps) {
    GemmParams p = { M, N, K, K, K, N, 0, 0, 0, 1.0f };
    gpu_arg_t args[5] = { GPU_BUF(A, 0), GPU_BUF(W, 0), GPU_BUF(A, 0), GPU_BUF(C, 0), GPU_BYTES(&p) };
    uint32_t groups = (N + GEMM_SIMDS * cols - 1) / (GEMM_SIMDS * cols);
    char err[256];
    gpu_begin(G);
    for (int i = 0; i < reps; ++i) gpu_dispatch_groups(G, kernel, args, 5, groups, 1, 1, GEMM_SIMDS * 32, 1, 1);
    if (gpu_end(G, err, sizeof err)) die(err);
    return gpu_last_ms(G) / reps;
}

int main(void) {
    char *src = malloc(strlen(kernels_metal_src) + strlen(extra_src) + 1);
    strcpy(src, kernels_metal_src);
    strcat(src, extra_src);
    char err[1024];
    G = gpu_create(src, err, sizeof err);
    if (!G) die(err);
    printf("device: %s\n", gpu_device_name(G));

    /* dispatch overhead: 576 trivial kernels in one command buffer */
    gpu_buf_t *tiny = gpu_buf_alloc(G, 4096);
    FillParams fp = { 1, 0.0f };
    gpu_arg_t fa[2] = { GPU_BUF(tiny, 0), GPU_BYTES(&fp) };
    for (int rep = 0; rep < 2; ++rep) {
        gpu_begin(G);
        for (int i = 0; i < 576; ++i) gpu_dispatch(G, "fill_f32", fa, 2, 1, 1, 1, 1, 1, 1);
        gpu_end(G, err, sizeof err);
    }
    printf("576 trivial dispatches: %.2f ms GPU (%.1f us each)\n", gpu_last_ms(G), gpu_last_ms(G) * 1000.0 / 576);

    const uint32_t Ms[] = { 1, 7, 14 };
    struct { uint32_t N, K; const char *what; } shapes[] = { { 4096, 1024, "ff_up   " }, { 1024, 4096, "ff_down " }, { 1024, 1024, "proj    " }, { 2048, 1024, "pw1     " } };
    gpu_buf_t *A = gpu_buf_alloc(G, 16 * 4096 * sizeof(float));
    float *af = gpu_buf_ptr(A);
    for (size_t i = 0; i < 16 * 4096; ++i) af[i] = (float)((i * 7919) % 1000) / 1000.0f - 0.5f;
    gpu_buf_t *W = gpu_buf_alloc(G, 4096 * 4096 * 2);
    uint16_t *wf = gpu_buf_ptr(W);
    for (size_t i = 0; i < 4096 * 4096; ++i) wf[i] = (uint16_t)(0x3c00 + (i % 512)); /* small bf16 values */
    gpu_buf_t *C = gpu_buf_alloc(G, 16 * 4096 * sizeof(float));
    const char *variants[] = { "gemm_bf16", "gemm_u2", "gemm_c2", "gemm_c4" };
    const int vcols[] = { 1, 1, 2, 4 };
    printf("\n%-9s %3s %-10s %8s %10s\n", "shape", "M", "kernel", "ms/call", "GB/s(W)");
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
        for (size_t mi = 0; mi < 3; ++mi) {
            for (size_t v = 0; v < 4; ++v) {
                time_gemm(variants[v], vcols[v], Ms[mi], shapes[s].N, shapes[s].K, A, W, C, 4); /* warm */
                double ms = time_gemm(variants[v], vcols[v], Ms[mi], shapes[s].N, shapes[s].K, A, W, C, 24);
                double gbs = (double)shapes[s].N * shapes[s].K * 2 / (ms / 1000.0) / 1e9;
                printf("%-9s %3u %-10s %8.4f %10.0f\n", shapes[s].what, Ms[mi], variants[v], ms, gbs);
            }
        }
    }
    /* DRAM-realistic: 24 distinct weight matrices (like 24 layers), so nothing is served from the cache */
    enum { NW = 24 };
    gpu_buf_t *Ws[NW];
    for (int i = 0; i < NW; ++i) { Ws[i] = gpu_buf_alloc(G, 4096 * 1024 * 2); memcpy(gpu_buf_ptr(Ws[i]), wf, 4096 * 1024 * 2); }
    gpu_buf_t *Abig = gpu_buf_alloc(G, 16 * 4096 * sizeof(float)); memcpy(gpu_buf_ptr(Abig), af, 16 * 4096 * sizeof(float));
    printf("\n--- DRAM-realistic (24 distinct weight matrices per timing), ms per gemm ---\n");
    printf("%-9s %3s | %-8s | %-8s %-8s %-8s | %-8s %-8s %-8s\n", "shape", "M", "spec c2", "ct1 s1", "ct1 s2", "ct1 s4", "ct2 s2", "ct2 s4", "ct4 s4");
    const uint32_t cfg[6][2] = { {1,1}, {1,2}, {1,4}, {2,2}, {2,4}, {4,4} };
    for (size_t sh = 0; sh < sizeof shapes / sizeof *shapes; ++sh) {
        const uint32_t N = shapes[sh].N, K = shapes[sh].K;
        for (size_t mi = 0; mi < 3; ++mi) {
            const uint32_t M = Ms[mi];
            GemmParams p = { M, N, K, K, K, N, 0, 0, 0, 1.0f };
            double t[7] = {0};
            for (int variant = 0; variant < 7; ++variant) {
                for (int rep = 0; rep < 2; ++rep) {
                    gpu_begin(G);
                    for (int i = 0; i < NW; ++i) {
                        gpu_arg_t args[5] = { GPU_BUF(Abig, 0), GPU_BUF(Ws[i], 0), GPU_BUF(Abig, 0), GPU_BUF(C, 0), GPU_BYTES(&p) };
                        if (variant == 0) { uint32_t fc[3] = { M, 2, 2 }; gpu_dispatch_groups_fc(G, "gemm_spec", fc, 3, args, 5, (N + GEMM_SIMDS * 2 - 1) / (GEMM_SIMDS * 2), 1, 1, 256, 1, 1); }
                        else { uint32_t ct = cfg[variant - 1][0], sp = cfg[variant - 1][1]; uint32_t fc[6] = { M, 1, 1, M > 8 ? 2u : 1u, ct, sp };
                               gpu_dispatch_groups_fc(G, "gemm_mma2", fc, 6, args, 5, (N + 8 * ct - 1) / (8 * ct), 1, 1, 32 * sp, 1, 1); }
                    }
                    if (gpu_end(G, err, sizeof err)) die(err);
                }
                t[variant] = gpu_last_ms(G) / NW;
            }
            double best = t[0]; for (int v = 1; v < 7; ++v) best = fmin(best, t[v]);
            printf("%-9s %3u | %8.4f | %8.4f %8.4f %8.4f | %8.4f %8.4f %8.4f   (best %.0f GB/s)\n", shapes[sh].what, M, t[0], t[1], t[2], t[3], t[4], t[5], t[6], N * K * 2 / (best / 1000.0) / 1e9);
        }
    }
    /* mma2 correctness vs gemm_bf16 */
    for (uint32_t M = 1; M <= 14; M += 13) {
        GemmParams p = { M, 1024, 4096, 4096, 4096, 1024, 0, 0, 0, 1.0f };
        gpu_arg_t args[5] = { GPU_BUF(Abig, 0), GPU_BUF(Ws[0], 0), GPU_BUF(Abig, 0), GPU_BUF(C, 0), GPU_BYTES(&p) };
        gpu_begin(G); gpu_dispatch_groups(G, "gemm_bf16", args, 5, 1024 / 8, 1, 1, 256, 1, 1); gpu_end(G, err, sizeof err);
        float *ref = malloc(M * 1024 * sizeof(float)); memcpy(ref, gpu_buf_ptr(C), M * 1024 * sizeof(float));
        uint32_t fc[6] = { M, 1, 1, M > 8 ? 2u : 1u, 2, 4 };
        gpu_begin(G); gpu_dispatch_groups_fc(G, "gemm_mma2", fc, 6, args, 5, 1024 / 16, 1, 1, 128, 1, 1); gpu_end(G, err, sizeof err);
        float *got = gpu_buf_ptr(C); double md = 0, mref = 0;
        for (size_t i = 0; i < M * 1024; ++i) { md = fmax(md, fabs(got[i] - ref[i])); mref = fmax(mref, fabs(ref[i])); }
        printf("gemm_mma2 (ct2,s4) M=%u max diff vs gemm_bf16: %.3e (ref max %.3e)\n", M, md, mref);
        free(ref);
    }
    /* mma correctness vs gemm_bf16 */
    for (uint32_t M = 1; M <= 14; M += 13) {
        GemmParams p = { M, 1024, 4096, 4096, 4096, 1024, 0, 0, 0, 1.0f };
        gpu_arg_t args[5] = { GPU_BUF(Abig, 0), GPU_BUF(Ws[0], 0), GPU_BUF(Abig, 0), GPU_BUF(C, 0), GPU_BYTES(&p) };
        gpu_begin(G); gpu_dispatch_groups(G, "gemm_bf16", args, 5, 1024 / 8, 1, 1, 256, 1, 1); gpu_end(G, err, sizeof err);
        float *ref = malloc(M * 1024 * sizeof(float)); memcpy(ref, gpu_buf_ptr(C), M * 1024 * sizeof(float));
        uint32_t fc[5] = { M, 1, 1, M > 8 ? 2u : 1u, 2 };
        gpu_begin(G); gpu_dispatch_groups_fc(G, "gemm_mma", fc, 5, args, 5, 1024 / 32, 1, 1, 64, 1, 1); gpu_end(G, err, sizeof err);
        float *got = gpu_buf_ptr(C); double md = 0, mref = 0;
        for (size_t i = 0; i < M * 1024; ++i) { md = fmax(md, fabs(got[i] - ref[i])); mref = fmax(mref, fabs(ref[i])); }
        printf("gemm_mma M=%u max diff vs gemm_bf16: %.3e (ref max %.3e)\n", M, md, mref);
        free(ref);
    }
    /* correctness of gemm_spec vs gemm_bf16 for M=14, cols=4 */
    {
        time_gemm("gemm_bf16", 1, 14, 1024, 4096, A, W, C, 1);
        float *ref = malloc(14 * 1024 * sizeof(float)); memcpy(ref, gpu_buf_ptr(C), 14 * 1024 * sizeof(float));
        uint32_t fc[3] = { 14, 4, 2 };
        GemmParams p = { 14, 1024, 4096, 4096, 4096, 1024, 0, 0, 0, 1.0f };
        gpu_arg_t args[5] = { GPU_BUF(A, 0), GPU_BUF(W, 0), GPU_BUF(A, 0), GPU_BUF(C, 0), GPU_BYTES(&p) };
        gpu_begin(G); gpu_dispatch_groups_fc(G, "gemm_spec", fc, 3, args, 5, 1024 / 32, 1, 1, 256, 1, 1); gpu_end(G, err, sizeof err);
        float *got = gpu_buf_ptr(C); double md = 0;
        for (size_t i = 0; i < 14 * 1024; ++i) md = fmax(md, fabs(got[i] - ref[i]));
        printf("gemm_spec(14,4,2) max diff vs gemm_bf16: %.3e\n", md);
        free(ref);
    }
    /* correctness cross-check of variants vs gemm_bf16 for M=14 */
    for (size_t v = 1; v < 4; ++v) {
        time_gemm("gemm_bf16", 1, 14, 1024, 4096, A, W, C, 1);
        float *ref = malloc(14 * 1024 * sizeof(float)); memcpy(ref, gpu_buf_ptr(C), 14 * 1024 * sizeof(float));
        time_gemm(variants[v], vcols[v], 14, 1024, 4096, A, W, C, 1);
        float *got = gpu_buf_ptr(C); double md = 0;
        for (size_t i = 0; i < 14 * 1024; ++i) md = fmax(md, fabs(got[i] - ref[i]));
        printf("variant %s max diff vs gemm_bf16: %.3e\n", variants[v], md);
        free(ref);
    }
    return 0;
}
