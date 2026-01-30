// FP8 (8-bit Floating Point) Kernels for Apple Silicon
//
// Two formats supported:
// - E4M3: 1 sign, 4 exponent, 3 mantissa (better precision, used for weights)
// - E5M2: 1 sign, 5 exponent, 2 mantissa (better range, used for gradients)
//
// This bypasses PyTorch's lack of FP8 support on MPS by implementing
// custom quantization/dequantization in Metal.
//
// Note: M3/M4 chips have native FP8 in Neural Engine, but we use software
// emulation here for broader compatibility (M1/M2 support).

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// FP8 E4M3 Format
// Range: ±448, smallest normal: 2^-6 = 0.015625
// Special: No infinity, NaN represented as 0x7F and 0xFF
// =============================================================================

// Convert FP8 E4M3 (stored as uchar) to float
inline float fp8_e4m3_to_float(uchar fp8) {
    // Extract components
    uint sign = (fp8 >> 7) & 0x1;
    uint exp = (fp8 >> 3) & 0xF;   // 4 bits
    uint mant = fp8 & 0x7;          // 3 bits

    float result;

    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            result = 0.0f;
        } else {
            // Subnormal: value = (-1)^s * 2^(-6) * (0.mant)
            result = ldexp(float(mant) / 8.0f, -6);
        }
    } else if (exp == 15) {
        // E4M3 uses exp=15, mant=7 as NaN, no infinity
        if (mant == 7) {
            result = NAN;
        } else {
            // Normal number at max exponent
            result = ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
        }
    } else {
        // Normal: value = (-1)^s * 2^(exp-7) * (1.mant)
        result = ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
    }

    return sign ? -result : result;
}

// Convert float to FP8 E4M3
inline uchar float_to_fp8_e4m3(float val) {
    if (isnan(val)) return 0x7F;  // NaN

    uint sign = val < 0 ? 1 : 0;
    val = abs(val);

    // Clamp to FP8 E4M3 max (448)
    val = min(val, 448.0f);

    if (val == 0.0f) return sign << 7;

    // Get exponent and mantissa
    int exp;
    float mant = frexp(val, &exp);  // val = mant * 2^exp, 0.5 <= mant < 1
    mant *= 2.0f;  // Now 1 <= mant < 2
    exp -= 1;      // Adjust for the *2

    // Bias exponent (bias = 7 for E4M3)
    int biased_exp = exp + 7;

    if (biased_exp <= 0) {
        // Subnormal
        int shift = 1 - biased_exp;
        if (shift > 3) return sign << 7;  // Underflow to zero
        uint mant_bits = uint((mant - 1.0f) * 8.0f + 0.5f) >> shift;
        mant_bits = min(mant_bits, 7u);
        return (sign << 7) | mant_bits;
    } else if (biased_exp >= 15) {
        // Overflow - clamp to max normal
        return (sign << 7) | (14 << 3) | 7;
    } else {
        // Normal
        uint mant_bits = uint((mant - 1.0f) * 8.0f + 0.5f);
        mant_bits = min(mant_bits, 7u);
        return (sign << 7) | (biased_exp << 3) | mant_bits;
    }
}

// =============================================================================
// FP8 E5M2 Format
// Range: ±57344, smallest normal: 2^-14
// Has infinity and NaN like FP16
// =============================================================================

inline float fp8_e5m2_to_float(uchar fp8) {
    uint sign = (fp8 >> 7) & 0x1;
    uint exp = (fp8 >> 2) & 0x1F;  // 5 bits
    uint mant = fp8 & 0x3;          // 2 bits

    float result;

    if (exp == 0) {
        if (mant == 0) {
            result = 0.0f;
        } else {
            // Subnormal
            result = ldexp(float(mant) / 4.0f, -14);
        }
    } else if (exp == 31) {
        // Infinity or NaN
        result = (mant == 0) ? INFINITY : NAN;
    } else {
        // Normal
        result = ldexp(1.0f + float(mant) / 4.0f, int(exp) - 15);
    }

    return sign ? -result : result;
}

inline uchar float_to_fp8_e5m2(float val) {
    if (isnan(val)) return 0x7F;  // NaN
    if (isinf(val)) return (val < 0 ? 0xFF : 0x7C);  // ±Inf

    uint sign = val < 0 ? 1 : 0;
    val = abs(val);

    if (val == 0.0f) return sign << 7;

    // Clamp to max normal
    val = min(val, 57344.0f);

    int exp;
    float mant = frexp(val, &exp);
    mant *= 2.0f;
    exp -= 1;

    int biased_exp = exp + 15;

    if (biased_exp <= 0) {
        int shift = 1 - biased_exp;
        if (shift > 2) return sign << 7;
        uint mant_bits = uint((mant - 1.0f) * 4.0f + 0.5f) >> shift;
        mant_bits = min(mant_bits, 3u);
        return (sign << 7) | mant_bits;
    } else if (biased_exp >= 31) {
        return (sign << 7) | (30 << 2) | 3;  // Max normal
    } else {
        uint mant_bits = uint((mant - 1.0f) * 4.0f + 0.5f);
        mant_bits = min(mant_bits, 3u);
        return (sign << 7) | (biased_exp << 2) | mant_bits;
    }
}

// =============================================================================
// FP8 Quantization Kernels (Row-wise scaling like INT8)
// =============================================================================

kernel void fp8_e4m3_quantize(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= rows) return;

    uint thread_id = tid.x;
    uint threads_per_row = 256;

    device const float* row_data = input + row * cols;
    device uchar* row_output = output + row * cols;

    // Find row max for scaling
    float local_max = 0.0f;
    for (uint i = thread_id; i < cols; i += threads_per_row) {
        local_max = max(local_max, abs(row_data[i]));
    }

    local_max = simd_max(local_max);

    threadgroup float shared_max[8];
    if (simd_lane == 0) shared_max[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_id == 0) {
        float final_max = 0.0f;
        for (uint i = 0; i < 8; i++) final_max = max(final_max, shared_max[i]);
        // Scale to fit in E4M3 range (max ~448)
        scales[row] = max(final_max / 448.0f, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = scales[row];

    // Quantize
    for (uint i = thread_id; i < cols; i += threads_per_row) {
        float val = row_data[i] / scale;
        row_output[i] = float_to_fp8_e4m3(val);
    }
}

kernel void fp8_e5m2_quantize(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= rows) return;

    uint thread_id = tid.x;
    uint threads_per_row = 256;

    device const float* row_data = input + row * cols;
    device uchar* row_output = output + row * cols;

    float local_max = 0.0f;
    for (uint i = thread_id; i < cols; i += threads_per_row) {
        local_max = max(local_max, abs(row_data[i]));
    }

    local_max = simd_max(local_max);

    threadgroup float shared_max[8];
    if (simd_lane == 0) shared_max[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_id == 0) {
        float final_max = 0.0f;
        for (uint i = 0; i < 8; i++) final_max = max(final_max, shared_max[i]);
        scales[row] = max(final_max / 57344.0f, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = scales[row];

    for (uint i = thread_id; i < cols; i += threads_per_row) {
        float val = row_data[i] / scale;
        row_output[i] = float_to_fp8_e5m2(val);
    }
}

// =============================================================================
// FP8 Dequantization Kernels
// =============================================================================

kernel void fp8_e4m3_dequantize(
    device const uchar* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;

    uchar fp8_val = input[row * cols + col];
    float scale = scales[row];
    float val = fp8_e4m3_to_float(fp8_val) * scale;
    output[row * cols + col] = half(val);
}

kernel void fp8_e5m2_dequantize(
    device const uchar* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;

    uchar fp8_val = input[row * cols + col];
    float scale = scales[row];
    float val = fp8_e5m2_to_float(fp8_val) * scale;
    output[row * cols + col] = half(val);
}

// =============================================================================
// FP8 E4M3 MatMul with Fused Dequantization
// =============================================================================

constant uint FP8_TILE_M = 32;
constant uint FP8_TILE_N = 32;
constant uint FP8_TILE_K = 32;
constant uint FP8_THREAD_M = 4;
constant uint FP8_THREAD_N = 4;

kernel void fp8_e4m3_matmul_fused(
    device const uchar* A [[buffer(0)]],      // [M, K] FP8
    device const uchar* B [[buffer(1)]],      // [K, N] FP8
    device half* C [[buffer(2)]],             // [M, N] FP16 output
    device const float* A_scales [[buffer(3)]], // [M]
    device const float* B_scales [[buffer(4)]], // [N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup float As[FP8_TILE_M][FP8_TILE_K + 1];
    threadgroup float Bs[FP8_TILE_K][FP8_TILE_N + 1];

    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint thread_id = ty * 8 + tx;
    const uint row_base = tgid.y * FP8_TILE_M + ty * FP8_THREAD_M;
    const uint col_base = tgid.x * FP8_TILE_N + tx * FP8_THREAD_N;

    float acc[FP8_THREAD_M][FP8_THREAD_N] = {{0.0f}};

    for (uint t = 0; t < K; t += FP8_TILE_K) {
        // Load A tile with dequantization
        for (uint i = 0; i < (FP8_TILE_M * FP8_TILE_K) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_row = flat_idx / FP8_TILE_K;
            uint load_col = flat_idx % FP8_TILE_K;
            uint global_row = tgid.y * FP8_TILE_M + load_row;
            uint global_col = t + load_col;

            if (global_row < M && global_col < K) {
                uchar fp8_val = A[global_row * K + global_col];
                float scale = A_scales[global_row];
                As[load_row][load_col] = fp8_e4m3_to_float(fp8_val) * scale;
            } else {
                As[load_row][load_col] = 0.0f;
            }
        }

        // Load B tile with dequantization
        for (uint i = 0; i < (FP8_TILE_K * FP8_TILE_N) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_row = flat_idx / FP8_TILE_N;
            uint load_col = flat_idx % FP8_TILE_N;
            uint global_row = t + load_row;
            uint global_col = tgid.x * FP8_TILE_N + load_col;

            if (global_row < K && global_col < N) {
                uchar fp8_val = B[global_row * N + global_col];
                float scale = B_scales[global_col];
                Bs[load_row][load_col] = fp8_e4m3_to_float(fp8_val) * scale;
            } else {
                Bs[load_row][load_col] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < FP8_TILE_K; k++) {
            float a_vals[FP8_THREAD_M];
            float b_vals[FP8_THREAD_N];

            for (uint m = 0; m < FP8_THREAD_M; m++)
                a_vals[m] = As[ty * FP8_THREAD_M + m][k];
            for (uint n = 0; n < FP8_THREAD_N; n++)
                b_vals[n] = Bs[k][tx * FP8_THREAD_N + n];

            for (uint m = 0; m < FP8_THREAD_M; m++)
                for (uint n = 0; n < FP8_THREAD_N; n++)
                    acc[m][n] += a_vals[m] * b_vals[n];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint m = 0; m < FP8_THREAD_M; m++) {
        uint out_row = row_base + m;
        if (out_row >= M) continue;
        for (uint n = 0; n < FP8_THREAD_N; n++) {
            uint out_col = col_base + n;
            if (out_col >= N) continue;
            C[out_row * N + out_col] = half(acc[m][n]);
        }
    }
}

// =============================================================================
// FP8 Linear Layer (input FP16, weights FP8)
// For inference: input in FP16, weights stored as FP8
// =============================================================================

kernel void fp8_e4m3_linear(
    device const half* input [[buffer(0)]],       // [M, K] FP16
    device const uchar* weight [[buffer(1)]],     // [N, K] FP8 (transposed)
    device const float* weight_scales [[buffer(2)]], // [N]
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n = gid.x;
    if (m >= M || n >= N) return;

    float scale = weight_scales[n];
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        uchar fp8_w = weight[n * K + k];
        float w_val = fp8_e4m3_to_float(fp8_w) * scale;
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    output[m * N + n] = half(acc);
}

// Tiled version for larger matrices
kernel void fp8_e4m3_linear_tiled(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half As[FP8_TILE_M][FP8_TILE_K + 1];
    threadgroup float Bs[FP8_TILE_K][FP8_TILE_N + 1];

    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint thread_id = ty * 8 + tx;
    const uint row_base = tgid.y * FP8_TILE_M + ty * FP8_THREAD_M;
    const uint col_base = tgid.x * FP8_TILE_N + tx * FP8_THREAD_N;

    float acc[FP8_THREAD_M][FP8_THREAD_N] = {{0.0f}};

    for (uint t = 0; t < K; t += FP8_TILE_K) {
        // Load input tile (FP16)
        for (uint i = 0; i < (FP8_TILE_M * FP8_TILE_K) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_row = flat_idx / FP8_TILE_K;
            uint load_col = flat_idx % FP8_TILE_K;
            uint global_row = tgid.y * FP8_TILE_M + load_row;
            uint global_col = t + load_col;

            As[load_row][load_col] = (global_row < M && global_col < K)
                ? input[global_row * K + global_col] : half(0.0f);
        }

        // Load weight tile (FP8) with dequantization
        for (uint i = 0; i < (FP8_TILE_K * FP8_TILE_N) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_k = flat_idx / FP8_TILE_N;
            uint load_n = flat_idx % FP8_TILE_N;
            uint global_k = t + load_k;
            uint global_n = tgid.x * FP8_TILE_N + load_n;

            if (global_k < K && global_n < N) {
                uchar fp8_w = weight[global_n * K + global_k];
                float scale = weight_scales[global_n];
                Bs[load_k][load_n] = fp8_e4m3_to_float(fp8_w) * scale;
            } else {
                Bs[load_k][load_n] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < FP8_TILE_K; k++) {
            half a_vals[FP8_THREAD_M];
            float b_vals[FP8_THREAD_N];

            for (uint m = 0; m < FP8_THREAD_M; m++)
                a_vals[m] = As[ty * FP8_THREAD_M + m][k];
            for (uint n = 0; n < FP8_THREAD_N; n++)
                b_vals[n] = Bs[k][tx * FP8_THREAD_N + n];

            for (uint m = 0; m < FP8_THREAD_M; m++)
                for (uint n = 0; n < FP8_THREAD_N; n++)
                    acc[m][n] += float(a_vals[m]) * b_vals[n];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint m = 0; m < FP8_THREAD_M; m++) {
        uint out_row = row_base + m;
        if (out_row >= M) continue;
        for (uint n = 0; n < FP8_THREAD_N; n++) {
            uint out_col = col_base + n;
            if (out_col >= N) continue;
            float result = acc[m][n];
            if (has_bias) result += float(bias[out_col]);
            output[out_row * N + out_col] = half(result);
        }
    }
}
