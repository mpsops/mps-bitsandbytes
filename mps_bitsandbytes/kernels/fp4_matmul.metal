// FP4 (4-bit Floating Point) Quantization Kernels for Apple Silicon
//
// FP4 uses E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// This gives 16 possible values with floating-point-like distribution.
//
// Compared to NF4:
// - NF4: Codebook optimized for normal distribution (better for typical weights)
// - FP4: True floating point distribution (better dynamic range)

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// FP4 E2M1 Codebook
// Format: SEEM (Sign, Exp, Exp, Mantissa)
// Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} normalized
// =============================================================================

// FP4 E2M1 values (unsigned, will apply sign separately)
// Index 0-7: positive values, Index 8-15: negative values
constant float FP4_CODEBOOK[16] = {
    0.0f,       // 0000: +0
    0.001953125f, // 0001: smallest subnormal (acts as near-zero positive)
    0.25f,      // 0010: 2^-2
    0.5f,       // 0011: 2^-1
    1.0f,       // 0100: 2^0
    1.5f,       // 0101: 1.5 * 2^0
    2.0f,       // 0110: 2^1
    3.0f,       // 0111: 1.5 * 2^1
    -0.0f,      // 1000: -0 (treated same as +0)
    -0.001953125f, // 1001: smallest subnormal negative
    -0.25f,     // 1010: -2^-2
    -0.5f,      // 1011: -2^-1
    -1.0f,      // 1100: -2^0
    -1.5f,      // 1101: -1.5 * 2^0
    -2.0f,      // 1110: -2^1
    -3.0f,      // 1111: -1.5 * 2^1
};

// Alternative normalized FP4 codebook (scaled to [-1, 1] like NF4)
constant float FP4_CODEBOOK_NORMALIZED[16] = {
    0.0f,
    0.0625f,
    0.125f,
    0.25f,
    0.375f,
    0.5f,
    0.75f,
    1.0f,
    -0.0f,
    -0.0625f,
    -0.125f,
    -0.25f,
    -0.375f,
    -0.5f,
    -0.75f,
    -1.0f,
};

// =============================================================================
// Helper Functions
// =============================================================================

inline uchar extract_fp4_index(uchar packed, uint position) {
    return (position == 0) ? (packed & 0x0F) : (packed >> 4);
}

inline float dequantize_fp4_value(uchar packed, uint position, float absmax) {
    uchar idx = extract_fp4_index(packed, position);
    return FP4_CODEBOOK_NORMALIZED[idx] * absmax;
}

// =============================================================================
// FP4 Quantization Kernel
// =============================================================================

kernel void fp4_quantize(
    device const float* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    device float* absmax [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= rows) return;

    uint num_blocks = (cols + block_size - 1) / block_size;
    uint threads_per_row = 256;
    uint thread_id = tid.x;

    device const float* row_data = input + row * cols;
    device uchar* row_output = output + row * (cols / 2);
    device float* row_absmax = absmax + row * num_blocks;

    for (uint block = 0; block < num_blocks; block++) {
        uint block_start = block * block_size;
        uint block_end = min(block_start + block_size, cols);
        uint block_len = block_end - block_start;

        // Compute absmax
        float local_max = 0.0f;
        for (uint i = thread_id; i < block_len; i += threads_per_row) {
            float val = row_data[block_start + i];
            local_max = max(local_max, abs(val));
        }

        local_max = simd_max(local_max);

        threadgroup float shared_max[8];
        if (simd_lane == 0) shared_max[simd_group] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float final_max = 0.0f;
            for (uint i = 0; i < 8; i++) final_max = max(final_max, shared_max[i]);
            row_absmax[block] = max(final_max, 1e-8f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_absmax = row_absmax[block];

        // Quantize to FP4
        for (uint i = thread_id * 2; i < block_len; i += threads_per_row * 2) {
            uint idx0 = block_start + i;
            uint idx1 = block_start + i + 1;

            float v0 = (idx0 < cols) ? row_data[idx0] / block_absmax : 0.0f;
            float v1 = (idx1 < cols) ? row_data[idx1] / block_absmax : 0.0f;

            // Find nearest FP4 codebook value
            uchar q0 = 0, q1 = 0;
            float best_dist0 = INFINITY, best_dist1 = INFINITY;
            for (uchar c = 0; c < 16; c++) {
                float dist0 = abs(v0 - FP4_CODEBOOK_NORMALIZED[c]);
                float dist1 = abs(v1 - FP4_CODEBOOK_NORMALIZED[c]);
                if (dist0 < best_dist0) { best_dist0 = dist0; q0 = c; }
                if (dist1 < best_dist1) { best_dist1 = dist1; q1 = c; }
            }

            uchar packed = q0 | (q1 << 4);
            row_output[(idx0 / 2)] = packed;
        }
    }
}

// =============================================================================
// FP4 Dequantization Kernel
// =============================================================================

kernel void fp4_dequantize(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= rows || col >= cols) return;

    uint num_blocks = (cols + block_size - 1) / block_size;
    uint block_idx = col / block_size;
    uint packed_idx = row * (cols / 2) + (col / 2);
    uint position = col % 2;

    uchar packed_val = packed[packed_idx];
    float scale = absmax[row * num_blocks + block_idx];
    float dequant = dequantize_fp4_value(packed_val, position, scale);
    output[row * cols + col] = half(dequant);
}

// =============================================================================
// FP4 MatMul Fused Kernel
// =============================================================================

constant uint FP4_TILE_M = 32;
constant uint FP4_TILE_N = 32;
constant uint FP4_TILE_K = 64;
constant uint FP4_THREAD_M = 4;
constant uint FP4_THREAD_N = 4;

kernel void fp4_matmul_fused(
    device const half* input [[buffer(0)]],
    device const uchar* weight_packed [[buffer(1)]],
    device const float* weight_absmax [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& block_size [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half As[FP4_TILE_M][FP4_TILE_K + 1];
    threadgroup half Bs[FP4_TILE_K][FP4_TILE_N + 1];

    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint thread_id = ty * 8 + tx;
    const uint row_base = tgid.y * FP4_TILE_M + ty * FP4_THREAD_M;
    const uint col_base = tgid.x * FP4_TILE_N + tx * FP4_THREAD_N;

    float acc[FP4_THREAD_M][FP4_THREAD_N] = {{0.0f}};
    const uint num_blocks = (K + block_size - 1) / block_size;

    for (uint t = 0; t < K; t += FP4_TILE_K) {
        // Load input tile
        for (uint i = 0; i < (FP4_TILE_M * FP4_TILE_K) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_row = flat_idx / FP4_TILE_K;
            uint load_col = flat_idx % FP4_TILE_K;
            uint global_row = tgid.y * FP4_TILE_M + load_row;
            uint global_col = t + load_col;
            As[load_row][load_col] = (global_row < M && global_col < K)
                ? input[global_row * K + global_col] : half(0.0f);
        }

        // Load + dequantize weight tile
        for (uint i = 0; i < (FP4_TILE_K * FP4_TILE_N) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_k = flat_idx / FP4_TILE_N;
            uint load_n = flat_idx % FP4_TILE_N;
            uint global_k = t + load_k;
            uint global_n = tgid.x * FP4_TILE_N + load_n;

            if (global_k < K && global_n < N) {
                uint packed_idx = global_n * (K / 2) + (global_k / 2);
                uchar packed = weight_packed[packed_idx];
                uint position = global_k % 2;
                uint blk_idx = global_k / block_size;
                float scale = weight_absmax[global_n * num_blocks + blk_idx];
                Bs[load_k][load_n] = half(dequantize_fp4_value(packed, position, scale));
            } else {
                Bs[load_k][load_n] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < FP4_TILE_K; k++) {
            half a_vals[FP4_THREAD_M];
            half b_vals[FP4_THREAD_N];
            for (uint m = 0; m < FP4_THREAD_M; m++)
                a_vals[m] = As[ty * FP4_THREAD_M + m][k];
            for (uint n = 0; n < FP4_THREAD_N; n++)
                b_vals[n] = Bs[k][tx * FP4_THREAD_N + n];
            for (uint m = 0; m < FP4_THREAD_M; m++)
                for (uint n = 0; n < FP4_THREAD_N; n++)
                    acc[m][n] += float(a_vals[m]) * float(b_vals[n]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint m = 0; m < FP4_THREAD_M; m++) {
        uint out_row = row_base + m;
        if (out_row >= M) continue;
        for (uint n = 0; n < FP4_THREAD_N; n++) {
            uint out_col = col_base + n;
            if (out_col >= N) continue;
            float result = acc[m][n];
            if (has_bias) result += float(bias[out_col]);
            output[out_row * N + out_col] = half(result);
        }
    }
}

// Simple FP4 linear for small matrices
kernel void fp4_linear_simple(
    device const half* input [[buffer(0)]],
    device const uchar* weight_packed [[buffer(1)]],
    device const float* weight_absmax [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& block_size [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y;
    uint n = gid.x;
    if (m >= M || n >= N) return;

    uint num_blocks = (K + block_size - 1) / block_size;
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        uint packed_idx = n * (K / 2) + (k / 2);
        uchar packed = weight_packed[packed_idx];
        uint position = k % 2;
        uint block_idx = k / block_size;
        float scale = weight_absmax[n * num_blocks + block_idx];
        float w_val = dequantize_fp4_value(packed, position, scale);
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    output[m * N + n] = half(acc);
}
