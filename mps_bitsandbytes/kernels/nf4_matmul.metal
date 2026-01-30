// NF4 (4-bit NormalFloat) Quantization Kernels for Apple Silicon
//
// NF4 is a 4-bit data type optimized for normally distributed weights.
// Used by QLoRA for 4x memory reduction with minimal accuracy loss.
//
// Storage format:
//   - weight_packed: [out_features, in_features // 2] uint8 (two 4-bit indices per byte)
//   - weight_absmax: [out_features, num_blocks] float32 (one scale per block)
//   - block_size: typically 64 (elements per block)

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// NF4 Codebook - 16 values optimized for normal distribution
// =============================================================================

// NF4 codebook: 16 values optimized for normal distribution (from bitsandbytes)
constant float NF4_CODEBOOK[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

// =============================================================================
// Helper: Extract 4-bit index from packed byte
// =============================================================================

inline uchar extract_nf4_index(uchar packed, uint position) {
    // position 0 = low nibble, position 1 = high nibble
    return (position == 0) ? (packed & 0x0F) : (packed >> 4);
}

inline float dequantize_nf4_value(uchar packed, uint position, float absmax) {
    uchar idx = extract_nf4_index(packed, position);
    return NF4_CODEBOOK[idx] * absmax;
}

// =============================================================================
// NF4 MatMul Fused Kernel
// Computes: output = input @ dequantize(weight_packed, weight_absmax).T + bias
//
// input:         [M, K] float16
// weight_packed: [N, K/2] uint8 (packed NF4)
// weight_absmax: [N, num_blocks] float32 where num_blocks = ceil(K / block_size)
// bias:          [N] float16 (optional, can be null)
// output:        [M, N] float16
// =============================================================================

constant uint NF4_TILE_M = 32;
constant uint NF4_TILE_N = 32;
constant uint NF4_TILE_K = 64;  // Match block_size for efficient absmax lookup
constant uint NF4_THREAD_M = 4;
constant uint NF4_THREAD_N = 4;

kernel void nf4_matmul_fused(
    device const half* input [[buffer(0)]],           // [M, K]
    device const uchar* weight_packed [[buffer(1)]],  // [N, K/2]
    device const float* weight_absmax [[buffer(2)]],  // [N, num_blocks]
    device const half* bias [[buffer(3)]],            // [N] or nullptr
    device half* output [[buffer(4)]],                // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& block_size [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory for tiles
    threadgroup half As[NF4_TILE_M][NF4_TILE_K + 1];     // Input tile
    threadgroup half Bs[NF4_TILE_K][NF4_TILE_N + 1];     // Dequantized weight tile

    const uint tx = tid.x;  // 0-7
    const uint ty = tid.y;  // 0-7
    const uint thread_id = ty * 8 + tx;

    const uint row_base = tgid.y * NF4_TILE_M + ty * NF4_THREAD_M;
    const uint col_base = tgid.x * NF4_TILE_N + tx * NF4_THREAD_N;

    // Accumulator registers
    float acc[NF4_THREAD_M][NF4_THREAD_N] = {{0.0f}};

    const uint num_blocks = (K + block_size - 1) / block_size;

    // Loop over K tiles
    for (uint t = 0; t < K; t += NF4_TILE_K) {
        // Cooperative load of input tile A [TILE_M x TILE_K]
        // 64 threads load 32x64 = 2048 elements = 32 per thread
        for (uint i = 0; i < (NF4_TILE_M * NF4_TILE_K) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_row = flat_idx / NF4_TILE_K;
            uint load_col = flat_idx % NF4_TILE_K;
            uint global_row = tgid.y * NF4_TILE_M + load_row;
            uint global_col = t + load_col;

            if (global_row < M && global_col < K) {
                As[load_row][load_col] = input[global_row * K + global_col];
            } else {
                As[load_row][load_col] = half(0.0f);
            }
        }

        // Cooperative load + dequantize of weight tile B [TILE_K x TILE_N]
        // Weight is stored as [N, K/2], we need [K, N] view
        // Each thread handles 32 elements
        for (uint i = 0; i < (NF4_TILE_K * NF4_TILE_N) / 64; i++) {
            uint flat_idx = thread_id + i * 64;
            uint load_k = flat_idx / NF4_TILE_N;      // K dimension (row in tile)
            uint load_n = flat_idx % NF4_TILE_N;      // N dimension (col in tile)
            uint global_k = t + load_k;
            uint global_n = tgid.x * NF4_TILE_N + load_n;

            if (global_k < K && global_n < N) {
                // Weight is [N, K/2] packed, index as weight[n, k/2]
                uint packed_idx = global_n * (K / 2) + (global_k / 2);
                uchar packed = weight_packed[packed_idx];
                uint position = global_k % 2;

                // Get absmax for this block
                uint block_idx = global_k / block_size;
                float absmax = weight_absmax[global_n * num_blocks + block_idx];

                // Dequantize
                float dequant = dequantize_nf4_value(packed, position, absmax);
                Bs[load_k][load_n] = half(dequant);
            } else {
                Bs[load_k][load_n] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 4x4 output block
        for (uint k = 0; k < NF4_TILE_K; k++) {
            half a_vals[NF4_THREAD_M];
            half b_vals[NF4_THREAD_N];

            for (uint m = 0; m < NF4_THREAD_M; m++) {
                a_vals[m] = As[ty * NF4_THREAD_M + m][k];
            }
            for (uint n = 0; n < NF4_THREAD_N; n++) {
                b_vals[n] = Bs[k][tx * NF4_THREAD_N + n];
            }

            for (uint m = 0; m < NF4_THREAD_M; m++) {
                for (uint n = 0; n < NF4_THREAD_N; n++) {
                    acc[m][n] += float(a_vals[m]) * float(b_vals[n]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output with optional bias
    for (uint m = 0; m < NF4_THREAD_M; m++) {
        uint out_row = row_base + m;
        if (out_row >= M) continue;

        for (uint n = 0; n < NF4_THREAD_N; n++) {
            uint out_col = col_base + n;
            if (out_col >= N) continue;

            float result = acc[m][n];
            if (has_bias) {
                result += float(bias[out_col]);
            }
            output[out_row * N + out_col] = half(result);
        }
    }
}

// =============================================================================
// NF4 Quantization Kernel
// Quantizes a float tensor to NF4 format
//
// input:   [rows, cols] float16/float32
// output:  [rows, cols/2] uint8 (packed)
// absmax:  [rows, num_blocks] float32
// =============================================================================

kernel void nf4_quantize(
    device const float* input [[buffer(0)]],      // [rows, cols] as float
    device uchar* output [[buffer(1)]],           // [rows, cols/2] packed
    device float* absmax [[buffer(2)]],           // [rows, num_blocks]
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    // Each threadgroup handles one row
    uint row = tgid.y;
    if (row >= rows) return;

    uint num_blocks = (cols + block_size - 1) / block_size;
    uint threads_per_row = 256;  // Threadgroup width
    uint thread_id = tid.x;

    device const float* row_data = input + row * cols;
    device uchar* row_output = output + row * (cols / 2);
    device float* row_absmax = absmax + row * num_blocks;

    // Process each block
    for (uint block = 0; block < num_blocks; block++) {
        uint block_start = block * block_size;
        uint block_end = min(block_start + block_size, cols);
        uint block_len = block_end - block_start;

        // Step 1: Compute absmax for this block (reduction)
        float local_max = 0.0f;
        for (uint i = thread_id; i < block_len; i += threads_per_row) {
            float val = row_data[block_start + i];
            local_max = max(local_max, abs(val));
        }

        // SIMD reduction
        local_max = simd_max(local_max);

        // Threadgroup reduction using shared memory
        threadgroup float shared_max[8];  // 8 simd groups
        if (simd_lane == 0) {
            shared_max[simd_group] = local_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float final_max = 0.0f;
            for (uint i = 0; i < 8; i++) {
                final_max = max(final_max, shared_max[i]);
            }
            // Clamp to avoid division by zero
            row_absmax[block] = max(final_max, 1e-8f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_absmax = row_absmax[block];

        // Step 2: Quantize values in this block
        // Each thread handles pairs of values (to pack into one byte)
        for (uint i = thread_id * 2; i < block_len; i += threads_per_row * 2) {
            uint idx0 = block_start + i;
            uint idx1 = block_start + i + 1;

            // Normalize to [-1, 1]
            float v0 = (idx0 < cols) ? row_data[idx0] / block_absmax : 0.0f;
            float v1 = (idx1 < cols) ? row_data[idx1] / block_absmax : 0.0f;

            // Find nearest codebook index (binary search)
            uchar q0 = 0, q1 = 0;

            // Brute force nearest neighbor (faster for 16 values)
            float best_dist0 = INFINITY, best_dist1 = INFINITY;
            for (uchar c = 0; c < 16; c++) {
                float dist0 = abs(v0 - NF4_CODEBOOK[c]);
                float dist1 = abs(v1 - NF4_CODEBOOK[c]);
                if (dist0 < best_dist0) { best_dist0 = dist0; q0 = c; }
                if (dist1 < best_dist1) { best_dist1 = dist1; q1 = c; }
            }

            // Pack: low nibble = first value, high nibble = second value
            uchar packed = q0 | (q1 << 4);
            row_output[(idx0 / 2)] = packed;
        }
    }
}

// =============================================================================
// NF4 Dequantization Kernel (standalone)
// Useful for debugging or when full dequantization is needed
// =============================================================================

kernel void nf4_dequantize(
    device const uchar* packed [[buffer(0)]],     // [rows, cols/2]
    device const float* absmax [[buffer(1)]],     // [rows, num_blocks]
    device half* output [[buffer(2)]],            // [rows, cols]
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

    float dequant = dequantize_nf4_value(packed_val, position, scale);
    output[row * cols + col] = half(dequant);
}

// =============================================================================
// Simple NF4 Linear (non-tiled, for small matrices or reference)
// =============================================================================

kernel void nf4_linear_simple(
    device const half* input [[buffer(0)]],           // [M, K]
    device const uchar* weight_packed [[buffer(1)]],  // [N, K/2]
    device const float* weight_absmax [[buffer(2)]],  // [N, num_blocks]
    device const half* bias [[buffer(3)]],            // [N]
    device half* output [[buffer(4)]],                // [M, N]
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
        // Get input value
        float in_val = float(input[m * K + k]);

        // Dequantize weight
        uint packed_idx = n * (K / 2) + (k / 2);
        uchar packed = weight_packed[packed_idx];
        uint position = k % 2;
        uint block_idx = k / block_size;
        float scale = weight_absmax[n * num_blocks + block_idx];
        float w_val = dequantize_nf4_value(packed, position, scale);

        acc += in_val * w_val;
    }

    if (has_bias) {
        acc += float(bias[n]);
    }

    output[m * N + n] = half(acc);
}
