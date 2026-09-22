/* Shared between C and Metal: kernel parameter blocks (passed with setBytes). */
#ifndef NEMO_KERNEL_PARAMS_H
#define NEMO_KERNEL_PARAMS_H
#ifdef __METAL_VERSION__
typedef uint u32;
#else
#include <stdint.h>
typedef uint32_t u32;
#endif

#define GEMM_ROWS 16     /* rows of A accumulated per pass in gemm_bf16 */
#define GEMM_SIMDS 8     /* SIMD groups (output columns) per gemm threadgroup: 256 threads */
#define ATT_DH 128       /* head dim = attention threadgroup size */
#define JOINT_GROUPS 64  /* threadgroups for joint_partial */
#define ATT_MAX_L 80     /* max attention window (56 cache + 14 chunk, rounded) */
#define LN_THREADS 256
#define JOINT_MAX_H 640

typedef struct { u32 M, N, K, lda, ldw, ldc, has_bias, act, accumulate; float alpha; } GemmParams; /* act: 0 none, 1 silu, 2 relu */
typedef struct { u32 M, D; float eps; } LnParams;
typedef struct { u32 c, L, Lmax, H, dh, D; float scale; } AttnParams;
typedef struct { u32 M, D; } GluParams;
typedef struct { u32 c, D, taps; float eps; } DwParams;
typedef struct { u32 T, F, T1, F1, C; } Conv2dParams;
typedef struct { u32 T, F, C; } FlatParams;
typedef struct { u32 n; } CountParams;
typedef struct { u32 n; float value; } FillParams;
typedef struct { u32 H; } LstmParams;
typedef struct { u32 V, H, enc_row, enc_ld; } JointParams;

#endif
