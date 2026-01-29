// Int8 Matrix Multiplication Kernels for Apple Silicon
// Port of bitsandbytes int8 matmul for MPS
//
// Optimized with:
// - SIMD operations for vectorized int8 loads
// - Larger tiles with register blocking
// - Coalesced memory access patterns
// - Fused dequantization

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Optimized Int8 MatMul - 32x32 tiles with 4x4 register blocking
// Each thread computes a 4x4 output block
// =============================================================================

constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 32;
constant uint THREAD_M = 4;  // Each thread computes 4 rows
constant uint THREAD_N = 4;  // Each thread computes 4 cols

kernel void int8_matmul_tiled(
    device const char* A [[buffer(0)]],      // [M, K] int8
    device const char* B [[buffer(1)]],      // [K, N] int8
    device int* C [[buffer(2)]],             // [M, N] int32 output
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Shared memory for tiles
    threadgroup char As[TILE_M][TILE_K + 1];  // +1 to avoid bank conflicts
    threadgroup char Bs[TILE_K][TILE_N + 1];

    // Thread indices within 8x8 threadgroup
    const uint tx = tid.x;  // 0-7
    const uint ty = tid.y;  // 0-7

    // Global output position for this thread's 4x4 block
    const uint row_base = tgid.y * TILE_M + ty * THREAD_M;
    const uint col_base = tgid.x * TILE_N + tx * THREAD_N;

    // Accumulator registers - 4x4 block per thread
    int acc[THREAD_M][THREAD_N] = {{0}};

    // Loop over K tiles
    for (uint t = 0; t < K; t += TILE_K) {
        // Cooperative loading of A tile [TILE_M x TILE_K]
        // 64 threads load 32x32 = 1024 elements, so 16 elements per thread
        for (uint i = 0; i < (TILE_M * TILE_K) / 64; i++) {
            uint flat_idx = (ty * 8 + tx) + i * 64;
            uint load_row = flat_idx / TILE_K;
            uint load_col = flat_idx % TILE_K;
            uint global_row = tgid.y * TILE_M + load_row;
            uint global_col = t + load_col;

            if (global_row < M && global_col < K) {
                As[load_row][load_col] = A[global_row * K + global_col];
            } else {
                As[load_row][load_col] = 0;
            }
        }

        // Cooperative loading of B tile [TILE_K x TILE_N]
        for (uint i = 0; i < (TILE_K * TILE_N) / 64; i++) {
            uint flat_idx = (ty * 8 + tx) + i * 64;
            uint load_row = flat_idx / TILE_N;
            uint load_col = flat_idx % TILE_N;
            uint global_row = t + load_row;
            uint global_col = tgid.x * TILE_N + load_col;

            if (global_row < K && global_col < N) {
                Bs[load_row][load_col] = B[global_row * N + global_col];
            } else {
                Bs[load_row][load_col] = 0;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute 4x4 output block
        for (uint k = 0; k < TILE_K; k++) {
            // Load A values for this thread's rows
            char a_vals[THREAD_M];
            for (uint m = 0; m < THREAD_M; m++) {
                a_vals[m] = As[ty * THREAD_M + m][k];
            }

            // Load B values for this thread's cols
            char b_vals[THREAD_N];
            for (uint n = 0; n < THREAD_N; n++) {
                b_vals[n] = Bs[k][tx * THREAD_N + n];
            }

            // Accumulate
            for (uint m = 0; m < THREAD_M; m++) {
                for (uint n = 0; n < THREAD_N; n++) {
                    acc[m][n] += int(a_vals[m]) * int(b_vals[n]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write output
    for (uint m = 0; m < THREAD_M; m++) {
        for (uint n = 0; n < THREAD_N; n++) {
            uint out_row = row_base + m;
            uint out_col = col_base + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = acc[m][n];
            }
        }
    }
}

// =============================================================================
// Fused Int8 MatMul + Dequantize - outputs directly to fp16
// =============================================================================

kernel void int8_matmul_dequant(
    device const char* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device half* C [[buffer(2)]],              // Direct fp16 output
    device const float* A_scales [[buffer(3)]], // [M]
    device const float* B_scales [[buffer(4)]], // [N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup char As[TILE_M][TILE_K + 1];
    threadgroup char Bs[TILE_K][TILE_N + 1];

    const uint tx = tid.x;
    const uint ty = tid.y;
    const uint row_base = tgid.y * TILE_M + ty * THREAD_M;
    const uint col_base = tgid.x * TILE_N + tx * THREAD_N;

    int acc[THREAD_M][THREAD_N] = {{0}};

    for (uint t = 0; t < K; t += TILE_K) {
        for (uint i = 0; i < (TILE_M * TILE_K) / 64; i++) {
            uint flat_idx = (ty * 8 + tx) + i * 64;
            uint load_row = flat_idx / TILE_K;
            uint load_col = flat_idx % TILE_K;
            uint global_row = tgid.y * TILE_M + load_row;
            uint global_col = t + load_col;

            As[load_row][load_col] = (global_row < M && global_col < K)
                ? A[global_row * K + global_col] : 0;
        }

        for (uint i = 0; i < (TILE_K * TILE_N) / 64; i++) {
            uint flat_idx = (ty * 8 + tx) + i * 64;
            uint load_row = flat_idx / TILE_N;
            uint load_col = flat_idx % TILE_N;
            uint global_row = t + load_row;
            uint global_col = tgid.x * TILE_N + load_col;

            Bs[load_row][load_col] = (global_row < K && global_col < N)
                ? B[global_row * N + global_col] : 0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            char a_vals[THREAD_M];
            char b_vals[THREAD_N];

            for (uint m = 0; m < THREAD_M; m++) {
                a_vals[m] = As[ty * THREAD_M + m][k];
            }
            for (uint n = 0; n < THREAD_N; n++) {
                b_vals[n] = Bs[k][tx * THREAD_N + n];
            }

            for (uint m = 0; m < THREAD_M; m++) {
                for (uint n = 0; n < THREAD_N; n++) {
                    acc[m][n] += int(a_vals[m]) * int(b_vals[n]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Dequantize and write output
    const float inv_127_sq = 1.0f / (127.0f * 127.0f);

    for (uint m = 0; m < THREAD_M; m++) {
        uint out_row = row_base + m;
        if (out_row >= M) continue;

        float a_scale = A_scales[out_row];

        for (uint n = 0; n < THREAD_N; n++) {
            uint out_col = col_base + n;
            if (out_col >= N) continue;

            float b_scale = B_scales[out_col];
            float scale = a_scale * b_scale * inv_127_sq;
            float result = float(acc[m][n]) * scale;

            C[out_row * N + out_col] = half(result);
        }
    }
}

// =============================================================================
// Dequantization: int32 -> fp16 (standalone version)
// =============================================================================

kernel void dequantize_int32_to_fp16(
    device const int* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const float* row_scales [[buffer(2)]],
    device const float* col_scales [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    uint idx = row * N + col;
    int int_val = input[idx];

    float scale = (row_scales[row] * col_scales[col]) / (127.0f * 127.0f);
    output[idx] = half(float(int_val) * scale);
}
