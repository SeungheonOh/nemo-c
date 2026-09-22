#include <metal_stdlib>
using namespace metal;

inline float bf2f(ushort v) { return as_type<float>(((uint)v) << 16); }
inline float4 bf2f4(ushort4 v) { return float4(bf2f(v.x), bf2f(v.y), bf2f(v.z), bf2f(v.w)); }
inline float sigmoid_f(float v) { return 1.0f / (1.0f + exp(-v)); }
inline float silu_f(float v) { return v / (1.0f + exp(-v)); }

/* C[m][n] (= or +=) alpha * act( sum_k A[m][k] * W[n][k] + bias[n] ), W stored bf16 row-major [N][ldw].
   One SIMD group (32 lanes) per output column n: lanes stream the weight row cooperatively
   (coalesced), accumulate partial dots for up to GEMM_ROWS rows of A, then simd_sum. */
kernel void gemm_bf16(device const float *A [[buffer(0)]],
                      device const ushort *W [[buffer(1)]],
                      device const float *bias [[buffer(2)]],
                      device float *C [[buffer(3)]],
                      constant GemmParams &p [[buffer(4)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint n = g * GEMM_SIMDS + sg;
    if (n >= p.N) return; /* uniform per SIMD group */
    device const ushort4 *w4 = (device const ushort4 *)(W + (ulong)n * p.ldw);
    const uint K4 = p.K >> 2;
    const float b = p.has_bias ? bias[n] : 0.0f;
    for (uint m0 = 0; m0 < p.M; m0 += GEMM_ROWS) {
        const uint rows = min((uint)GEMM_ROWS, p.M - m0);
        float acc[GEMM_ROWS];
        for (uint r = 0; r < GEMM_ROWS; ++r) acc[r] = 0.0f;
        for (uint k4 = lane; k4 < K4; k4 += 32) {
            const float4 wf = bf2f4(w4[k4]);
            for (uint r = 0; r < GEMM_ROWS; ++r) {
                if (r < rows) acc[r] += dot(((device const float4 *)(A + (ulong)(m0 + r) * p.lda))[k4], wf);
            }
        }
        for (uint r = 0; r < GEMM_ROWS; ++r) {
            if (r >= rows) break;
            float v = simd_sum(acc[r]);
            if (lane == 0) {
                v += b;
                if (p.act == 1) v = silu_f(v);
                else if (p.act == 2) v = max(v, 0.0f);
                v *= p.alpha;
                const ulong ci = (ulong)(m0 + r) * p.ldc + n;
                C[ci] = p.accumulate ? (C[ci] + v) : v;
            }
        }
    }
}


/* Specialised GEMM: exact row count, columns per SIMD group and unroll factor are function
   constants, so every loop below is fully unrolled at pipeline creation. Same math as gemm_bf16. */
constant uint FC_ROWS [[function_constant(0)]];
constant uint FC_COLS [[function_constant(1)]];
constant uint FC_UNROLL [[function_constant(2)]];
#define GEMM_MAXCOLS 8

kernel void gemm_spec(device const float *A [[buffer(0)]],
                      device const ushort *W [[buffer(1)]],
                      device const float *bias [[buffer(2)]],
                      device float *C [[buffer(3)]],
                      constant GemmParams &p [[buffer(4)]],
                      uint g [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint n0 = (g * GEMM_SIMDS + sg) * FC_COLS;
    if (n0 >= p.N) return;
    const uint K4 = p.K >> 2;
    device const ushort4 *w4[GEMM_MAXCOLS];
    for (uint cc = 0; cc < FC_COLS; ++cc) w4[cc] = (device const ushort4 *)(W + (ulong)min(n0 + cc, p.N - 1) * p.ldw);
    device const float4 *a4[GEMM_ROWS];
    for (uint r = 0; r < FC_ROWS; ++r) a4[r] = (device const float4 *)(A + (ulong)r * p.lda);
    float acc[GEMM_ROWS][GEMM_MAXCOLS];
    for (uint r = 0; r < FC_ROWS; ++r) for (uint cc = 0; cc < FC_COLS; ++cc) acc[r][cc] = 0.0f;
    uint k4 = lane;
    if (FC_UNROLL == 2) {
        for (; k4 + 32 < K4; k4 += 64) {
            float4 wf[GEMM_MAXCOLS], wg[GEMM_MAXCOLS];
            for (uint cc = 0; cc < FC_COLS; ++cc) { wf[cc] = bf2f4(w4[cc][k4]); wg[cc] = bf2f4(w4[cc][k4 + 32]); }
            for (uint r = 0; r < FC_ROWS; ++r) {
                const float4 a = a4[r][k4], b = a4[r][k4 + 32];
                for (uint cc = 0; cc < FC_COLS; ++cc) acc[r][cc] += dot(a, wf[cc]) + dot(b, wg[cc]);
            }
        }
    }
    for (; k4 < K4; k4 += 32) {
        float4 wf[GEMM_MAXCOLS];
        for (uint cc = 0; cc < FC_COLS; ++cc) wf[cc] = bf2f4(w4[cc][k4]);
        for (uint r = 0; r < FC_ROWS; ++r) {
            const float4 a = a4[r][k4];
            for (uint cc = 0; cc < FC_COLS; ++cc) acc[r][cc] += dot(a, wf[cc]);
        }
    }
    for (uint r = 0; r < FC_ROWS; ++r) {
        for (uint cc = 0; cc < FC_COLS; ++cc) {
            float v = simd_sum(acc[r][cc]);
            if (lane == 0 && n0 + cc < p.N) {
                v += p.has_bias ? bias[n0 + cc] : 0.0f;
                if (p.act == 1) v = silu_f(v);
                else if (p.act == 2) v = max(v, 0.0f);
                v *= p.alpha;
                const ulong ci = (ulong)r * p.ldc + n0 + cc;
                C[ci] = p.accumulate ? (C[ci] + v) : v;
            }
        }
    }
}

/* Split-K MMA GEMM for M <= 16 rows: one threadgroup per block of 8*FC_CT output columns,
   FC_SPLIT SIMD groups each streaming a K slice through the 8x8 matrix units (bf16 weight
   tiles converted to float in threadgroup memory), then a threadgroup reduction. A must have
   at least 8*FC_TM rows allocated (rows >= M are read but never stored). */
constant uint FC_TM [[function_constant(3)]];
constant uint FC_CT [[function_constant(4)]];
constant uint FC_SPLIT [[function_constant(5)]];
#define MMA_KSTEP 32
#define MMA_MAXCT 2
#define MMA_MAXSPLIT 4

kernel void gemm_mma(device const float *A [[buffer(0)]],
                     device const ushort *W [[buffer(1)]],
                     device const float *bias [[buffer(2)]],
                     device float *C [[buffer(3)]],
                     constant GemmParams &p [[buffer(4)]],
                     uint g [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wtile[MMA_MAXSPLIT][MMA_MAXCT][8][MMA_KSTEP];
    threadgroup float red[MMA_MAXSPLIT][2][MMA_MAXCT][8][8];
    const uint n0 = g * 8 * FC_CT;
    if (n0 >= p.N) return;
    simdgroup_float8x8 acc[2][MMA_MAXCT];
    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) acc[tm][ct] = simdgroup_float8x8(0.0f);
    const uint r = lane >> 2, kk = (lane & 3) * 8;
    const uint kper = p.K / FC_SPLIT, kbeg = sg * kper, kend = kbeg + kper;
    for (uint k0 = kbeg; k0 < kend; k0 += MMA_KSTEP) {
        for (uint ct = 0; ct < FC_CT; ++ct) {
            device const ushort4 *wp = (device const ushort4 *)(W + (ulong)min(n0 + ct * 8 + r, p.N - 1) * p.ldw + k0 + kk);
            threadgroup float4 *dst = (threadgroup float4 *)&wtile[sg][ct][r][kk];
            dst[0] = bf2f4(wp[0]);
            dst[1] = bf2f4(wp[1]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kt = 0; kt < MMA_KSTEP / 8; ++kt) {
            simdgroup_float8x8 a[2];
            for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_load(a[tm], A + (ulong)tm * 8 * p.lda + k0 + kt * 8, p.lda);
            for (uint ct = 0; ct < FC_CT; ++ct) {
                simdgroup_float8x8 b;
                simdgroup_load(b, &wtile[sg][ct][0][kt * 8], MMA_KSTEP, ulong2(0, 0), true);
                for (uint tm = 0; tm < FC_TM; ++tm) simdgroup_multiply_accumulate(acc[tm][ct], a[tm], b, acc[tm][ct]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) simdgroup_store(acc[tm][ct], &red[sg][tm][ct][0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint tm = 0; tm < FC_TM; ++tm) for (uint ct = 0; ct < FC_CT; ++ct) for (uint e = lane; e < 64; e += 32) {
            const uint rr = e >> 3, cc = e & 7, row = tm * 8 + rr, col = n0 + ct * 8 + cc;
            if (row < p.M && col < p.N) {
                float v = 0.0f;
                for (uint s2 = 0; s2 < FC_SPLIT; ++s2) v += red[s2][tm][ct][rr][cc];
                v += p.has_bias ? bias[col] : 0.0f;
                if (p.act == 1) v = silu_f(v);
                else if (p.act == 2) v = max(v, 0.0f);
                v *= p.alpha;
                const ulong ci = (ulong)row * p.ldc + col;
                C[ci] = p.accumulate ? (C[ci] + v) : v;
            }
        }
    }
}

/* One threadgroup (LN_THREADS) per row. */
kernel void layernorm(device const float *X [[buffer(0)]],
                      device const float *gamma [[buffer(1)]],
                      device const float *beta [[buffer(2)]],
                      device float *Y [[buffer(3)]],
                      constant LnParams &p [[buffer(4)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[LN_THREADS / 32];
    device const float *x = X + (ulong)row * p.D;
    float s = 0.0f;
    for (uint i = tid; i < p.D; i += LN_THREADS) s += x[i];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < LN_THREADS / 32; ++i) tot += red[i];
    const float mean = tot / (float)p.D;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0.0f;
    for (uint i = tid; i < p.D; i += LN_THREADS) { const float d = x[i] - mean; v += d * d; }
    v = simd_sum(v);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vt = 0.0f;
    for (uint i = 0; i < LN_THREADS / 32; ++i) vt += red[i];
    const float inv = 1.0f / sqrt(vt / (float)p.D + p.eps);
    device float *y = Y + (ulong)row * p.D;
    for (uint i = tid; i < p.D; i += LN_THREADS) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}

/* Relative-position attention: one threadgroup (dh = 128 threads) per (head, query).
   Q: [c][D], K,V: [L][D], P: [2*Lmax-1][D] position projections (row r <-> relative position (Lmax-1)-r).
   Thread j < L computes score j; thread d computes output feature d. */
kernel void attention_rel(device const float *Q [[buffer(0)]],
                          device const float *K [[buffer(1)]],
                          device const float *V [[buffer(2)]],
                          device const float *P [[buffer(3)]],
                          device const float *bias_u [[buffer(4)]],
                          device const float *bias_v [[buffer(5)]],
                          device float *O [[buffer(6)]],
                          constant AttnParams &p [[buffer(7)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float qu[ATT_DH], qv[ATT_DH], s[ATT_MAX_L];
    const uint h = tg.x, i = tg.y, dh = p.dh, D = p.D, L = p.L, off = h * dh;
    {
        const float q = Q[(ulong)i * D + off + tid];
        qu[tid] = q + bias_u[off + tid];
        qv[tid] = q + bias_v[off + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < L) {
        const uint j = tid;
        /* relative position = (L - c + i) - j  ->  row in P (window L) = c-1-i+j, shifted by Lmax-L */
        const uint prow = (p.Lmax - L) + (p.c - 1 - i + j);
        device const float4 *k4 = (device const float4 *)(K + (ulong)j * D + off);
        device const float4 *p4 = (device const float4 *)(P + (ulong)prow * D + off);
        float a = 0.0f, b = 0.0f;
        for (uint d4 = 0; d4 < dh / 4; ++d4) {
            a += dot(float4(qu[4 * d4], qu[4 * d4 + 1], qu[4 * d4 + 2], qu[4 * d4 + 3]), k4[d4]);
            b += dot(float4(qv[4 * d4], qv[4 * d4 + 1], qv[4 * d4 + 2], qv[4 * d4 + 3]), p4[d4]);
        }
        s[j] = (a + b) * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint j = 0; j < L; ++j) mx = max(mx, s[j]);
        float sum = 0.0f;
        for (uint j = 0; j < L; ++j) { s[j] = exp(s[j] - mx); sum += s[j]; }
        const float inv = 1.0f / sum;
        for (uint j = 0; j < L; ++j) s[j] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < L; ++j) acc += s[j] * V[(ulong)j * D + off + tid];
    O[(ulong)i * D + off + tid] = acc;
}

/* Y[m][d] = X[m][d] * sigmoid(X[m][D+d]) */
kernel void glu(device const float *X [[buffer(0)]], device float *Y [[buffer(1)]],
                constant GluParams &p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x, m = gid.y;
    if (d >= p.D || m >= p.M) return;
    const float a = X[(ulong)m * 2 * p.D + d], b = X[(ulong)m * 2 * p.D + p.D + d];
    Y[(ulong)m * p.D + d] = a * sigmoid_f(b);
}

/* Causal depthwise conv (valid, taps rows of history already in din) + LayerNorm + SiLU.
   din: [(taps-1) + c][D]; W: [D][taps]; one threadgroup (LN_THREADS) per output row t. */
kernel void dwconv_ln_silu(device const float *din [[buffer(0)]],
                           device const float *W [[buffer(1)]],
                           device const float *gamma [[buffer(2)]],
                           device const float *beta [[buffer(3)]],
                           device float *Y [[buffer(4)]],
                           constant DwParams &p [[buffer(5)]],
                           uint t [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[LN_THREADS / 32];
    const uint per = (p.D + LN_THREADS - 1) / LN_THREADS; /* 4 for D=1024 */
    float v[8];
    float s = 0.0f;
    for (uint k = 0; k < per; ++k) {
        const uint ch = tid + k * LN_THREADS;
        float acc = 0.0f;
        if (ch < p.D) {
            for (uint tap = 0; tap < p.taps; ++tap) acc += W[ch * p.taps + tap] * din[(ulong)(t + tap) * p.D + ch];
        }
        v[k] = acc;
        s += acc;
    }
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < LN_THREADS / 32; ++i) tot += red[i];
    const float mean = tot / (float)p.D;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f;
    for (uint k = 0; k < per; ++k) {
        const uint ch = tid + k * LN_THREADS;
        if (ch < p.D) { const float d = v[k] - mean; var += d * d; }
    }
    var = simd_sum(var);
    if (lane == 0) red[sg] = var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float vt = 0.0f;
    for (uint i = 0; i < LN_THREADS / 32; ++i) vt += red[i];
    const float inv = 1.0f / sqrt(vt / (float)p.D + p.eps);
    for (uint k = 0; k < per; ++k) {
        const uint ch = tid + k * LN_THREADS;
        if (ch < p.D) {
            const float y = (v[k] - mean) * inv * gamma[ch] + beta[ch];
            Y[(ulong)t * p.D + ch] = silu_f(y);
        }
    }
}

/* Subsampling conv 0: 1 input channel, 3x3, stride 2, causal pad (left 2, right 1) on time and freq, + ReLU.
   X: [T][F], W: [C][3][3], Y: [T1][F1][C]. grid = (C, F1, T1). */
kernel void conv2d_c1_s2_relu(device const float *X [[buffer(0)]],
                              device const float *W [[buffer(1)]],
                              device const float *bias [[buffer(2)]],
                              device float *Y [[buffer(3)]],
                              constant Conv2dParams &p [[buffer(4)]],
                              uint3 gid [[thread_position_in_grid]]) {
    const uint o = gid.x, f1 = gid.y, t1 = gid.z;
    if (o >= p.C || f1 >= p.F1 || t1 >= p.T1) return;
    float acc = bias[o];
    for (uint kh = 0; kh < 3; ++kh) {
        const int ti = (int)(2 * t1 + kh) - 2;
        if (ti < 0 || ti >= (int)p.T) continue;
        for (uint kw = 0; kw < 3; ++kw) {
            const int fi = (int)(2 * f1 + kw) - 2;
            if (fi < 0 || fi >= (int)p.F) continue;
            acc += W[o * 9 + kh * 3 + kw] * X[(ulong)ti * p.F + (ulong)fi];
        }
    }
    Y[((ulong)t1 * p.F1 + f1) * p.C + o] = max(acc, 0.0f);
}

/* Depthwise 3x3 stride-2 conv with the same causal padding, no activation. X: [T][F][C] -> Y: [T1][F1][C]. */
kernel void dwconv2d_s2(device const float *X [[buffer(0)]],
                        device const float *W [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device float *Y [[buffer(3)]],
                        constant Conv2dParams &p [[buffer(4)]],
                        uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, f1 = gid.y, t1 = gid.z;
    if (c >= p.C || f1 >= p.F1 || t1 >= p.T1) return;
    float acc = bias[c];
    for (uint kh = 0; kh < 3; ++kh) {
        const int ti = (int)(2 * t1 + kh) - 2;
        if (ti < 0 || ti >= (int)p.T) continue;
        for (uint kw = 0; kw < 3; ++kw) {
            const int fi = (int)(2 * f1 + kw) - 2;
            if (fi < 0 || fi >= (int)p.F) continue;
            acc += W[c * 9 + kh * 3 + kw] * X[((ulong)ti * p.F + (ulong)fi) * p.C + c];
        }
    }
    Y[((ulong)t1 * p.F1 + f1) * p.C + c] = acc;
}

/* [T][F][C] -> [T][C*F] (channel-major flatten, as NeMo). grid = (C, F, T) */
kernel void flatten_cf(device const float *X [[buffer(0)]], device float *Y [[buffer(1)]],
                       constant FlatParams &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    const uint c = gid.x, f = gid.y, t = gid.z;
    if (c >= p.C || f >= p.F || t >= p.T) return;
    Y[(ulong)t * p.C * p.F + c * p.F + f] = X[((ulong)t * p.F + f) * p.C + c];
}

kernel void copy_f32(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                     constant CountParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = src[i];
}

kernel void fill_f32(device float *dst [[buffer(0)]], constant FillParams &p [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < p.n) dst[i] = p.value;
}

/* gates: [4H] in i,f,g,o order (already Wx x + Wh h + b). */
kernel void lstm_cell(device const float *gates [[buffer(0)]], device const float *c_in [[buffer(1)]],
                      device float *c_out [[buffer(2)]], device float *h_out [[buffer(3)]],
                      constant LstmParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.H) return;
    const float ig = sigmoid_f(gates[i]);
    const float fg = sigmoid_f(gates[p.H + i]);
    const float gg = tanh(gates[2 * p.H + i]);
    const float og = sigmoid_f(gates[3 * p.H + i]);
    const float c = fg * c_in[i] + ig * gg;
    c_out[i] = c;
    h_out[i] = og * tanh(c);
}

/* RNNT joint, stage 1: each SIMD group scores rows n = g*8+sg, g*8+sg+G*8, ... of
   W . relu(enc + pred) + b and keeps its best (lowest index on ties). */
kernel void joint_partial(device const float *enc [[buffer(0)]],
                          device const float *pred [[buffer(1)]],
                          device const ushort *W [[buffer(2)]],
                          device const float *bias [[buffer(3)]],
                          device float *part_v [[buffer(4)]],
                          device int *part_i [[buffer(5)]],
                          constant JointParams &p [[buffer(6)]],
                          uint g [[threadgroup_position_in_grid]],
                          uint G [[threadgroups_per_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float4 r4[JOINT_MAX_H / 4];
    threadgroup float *r = (threadgroup float *)r4;
    device const float *e = enc + (ulong)p.enc_row * p.enc_ld;
    for (uint i = tid; i < p.H; i += GEMM_SIMDS * 32) r[i] = max(e[i] + pred[i], 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint H4 = p.H / 4;
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint n = g * GEMM_SIMDS + sg; n < p.V; n += G * GEMM_SIMDS) {
        device const ushort4 *w4 = (device const ushort4 *)(W + (ulong)n * p.H);
        float acc = 0.0f;
        for (uint k4 = lane; k4 < H4; k4 += 32) acc += dot(r4[k4], bf2f4(w4[k4]));
        acc = simd_sum(acc) + bias[n];
        if (acc > bv || (acc == bv && (int)n < bi)) { bv = acc; bi = (int)n; }
    }
    if (lane == 0) { part_v[g * GEMM_SIMDS + sg] = bv; part_i[g * GEMM_SIMDS + sg] = bi; }
}

/* RNNT joint, stage 2: reduce the partial bests. One threadgroup of 256. out[0] = index, out[1] = value bits. */
kernel void argmax_reduce(device const float *part_v [[buffer(0)]],
                          device const int *part_i [[buffer(1)]],
                          device int *out [[buffer(2)]],
                          constant CountParams &p [[buffer(3)]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float bestv[256];
    threadgroup int besti[256];
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (uint i = tid; i < p.n; i += 256) {
        if (part_v[i] > bv || (part_v[i] == bv && part_i[i] < bi)) { bv = part_v[i]; bi = part_i[i]; }
    }
    bestv[tid] = bv;
    besti[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            const float ov = bestv[tid + st];
            const int oi = besti[tid + st];
            if (ov > bestv[tid] || (ov == bestv[tid] && oi < besti[tid])) { bestv[tid] = ov; besti[tid] = oi; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { out[0] = besti[0]; out[1] = as_type<int>(bestv[0]); }
}
