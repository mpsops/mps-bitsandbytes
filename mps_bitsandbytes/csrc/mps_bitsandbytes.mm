/**
 * MPS BitsAndBytes - PyTorch C++ Extension
 *
 * Quantization kernels for Apple Silicon:
 * - INT8: 8-bit integer quantization
 * - NF4: 4-bit NormalFloat (QLoRA)
 * - FP4: 4-bit floating point
 * - FP8: 8-bit floating point (E4M3/E5M2)
 * - Double quantization support
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <mutex>
#include <atomic>

// =============================================================================
// Metal Device and Libraries (Thread-Safe)
// =============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

// Thread-safety primitives
static std::mutex g_init_mutex;
static std::atomic<bool> g_device_initialized{false};

static void ensure_device() {
    // Fast path: already initialized
    if (g_device_initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Slow path: acquire lock and initialize
    std::lock_guard<std::mutex> lock(g_init_mutex);

    // Double-check after acquiring lock
    if (g_device_initialized.load(std::memory_order_relaxed)) {
        return;
    }

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        throw std::runtime_error("Failed to create Metal device");
    }

    g_device_initialized.store(true, std::memory_order_release);
}

// =============================================================================
// Embedded Shader Source (all kernels in one)
// =============================================================================

static const char* KERNELS_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// NF4 Codebook
// =============================================================================

constant float NF4_CODEBOOK[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// =============================================================================
// FP4 Codebook (normalized to [-1, 1])
// =============================================================================

constant float FP4_CODEBOOK[16] = {
    0.0f, 0.0625f, 0.125f, 0.25f, 0.375f, 0.5f, 0.75f, 1.0f,
    -0.0f, -0.0625f, -0.125f, -0.25f, -0.375f, -0.5f, -0.75f, -1.0f
};

// Half (fp16) max representable value - values beyond this become Inf
constant float FP16_MAX_VAL = 65504.0f;

// =============================================================================
// FP8 Conversion Functions
// =============================================================================

inline float fp8_e4m3_to_float(uchar fp8) {
    uint sign = (fp8 >> 7) & 0x1;
    uint exp = (fp8 >> 3) & 0xF;
    uint mant = fp8 & 0x7;

    float result;
    if (exp == 0) {
        result = (mant == 0) ? 0.0f : ldexp(float(mant) / 8.0f, -6);
    } else if (exp == 15 && mant == 7) {
        result = NAN;
    } else {
        result = ldexp(1.0f + float(mant) / 8.0f, int(exp) - 7);
    }
    return sign ? -result : result;
}

inline uchar float_to_fp8_e4m3(float val) {
    if (isnan(val)) return 0x7F;
    uint sign = val < 0 ? 1 : 0;
    val = abs(val);
    val = min(val, 448.0f);
    if (val == 0.0f) return sign << 7;

    int exp;
    float mant = frexp(val, exp);  // Metal uses reference, not pointer
    mant *= 2.0f; exp -= 1;  // frexp returns [0.5, 1), we want [1, 2)
    int biased_exp = exp + 7;

    if (biased_exp <= 0) {
        // Subnormal
        int shift = 1 - biased_exp;
        if (shift > 3) return sign << 7;
        uint mant_bits = uint((mant - 1.0f) * 8.0f + 0.5f) >> shift;
        return (sign << 7) | min(mant_bits, 7u);
    } else if (biased_exp >= 15) {
        // Overflow to max
        return (sign << 7) | (14 << 3) | 7;
    } else {
        uint mant_bits = uint((mant - 1.0f) * 8.0f + 0.5f);
        return (sign << 7) | (biased_exp << 3) | min(mant_bits, 7u);
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

inline float dequant_nf4(uchar packed, uint pos, float scale) {
    uchar idx = (pos == 0) ? (packed & 0x0F) : (packed >> 4);
    return NF4_CODEBOOK[idx] * scale;
}

inline float dequant_fp4(uchar packed, uint pos, float scale) {
    uchar idx = (pos == 0) ? (packed & 0x0F) : (packed >> 4);
    return FP4_CODEBOOK[idx] * scale;
}

// =============================================================================
// INT8 MatMul
// =============================================================================

kernel void int8_matmul_dequant(
    device const char* A [[buffer(0)]],
    device const char* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const float* A_scales [[buffer(3)]],
    device const float* B_scales [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint TILE = 16;
    threadgroup char As[16][16];
    threadgroup char Bs[16][16];

    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;
    int acc = 0;

    for (uint t = 0; t < K; t += TILE) {
        uint a_col = t + tid.x;
        uint b_row = t + tid.y;
        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            acc += int(As[tid.y][k]) * int(Bs[k][tid.x]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        float scale = (A_scales[row] * B_scales[col]) / (127.0f * 127.0f);
        float result = float(acc) * scale;
        // Clamp to half range to prevent Inf
        result = clamp(result, -FP16_MAX_VAL, FP16_MAX_VAL);
        C[row * N + col] = half(result);
    }
}

// =============================================================================
// INT8 MatMul with SIMD Group Matrix Operations (Apple Silicon optimized)
// Weight layout: [N, K] with row-wise scales[N]
// =============================================================================

kernel void int8_matmul_simd(
    device const half* input [[buffer(0)]],       // [M, K] half
    device const char* weight [[buffer(1)]],      // [N, K] int8
    device const float* weight_scales [[buffer(2)]], // [N] per-row scales
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint out_row_base = tgid.y * 8;
    uint out_col_base = tgid.x * 8;

    threadgroup half A_tile[8][8];
    threadgroup half B_tile[8][8];

    simdgroup_matrix<float, 8, 8> C;
    C = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += 8) {
        // Load A tile
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint r0 = idx0 / 8, c0 = idx0 % 8;
            uint r1 = idx1 / 8, c1 = idx1 % 8;

            uint gr0 = out_row_base + r0, gc0 = k_base + c0;
            uint gr1 = out_row_base + r1, gc1 = k_base + c1;

            A_tile[r0][c0] = (gr0 < M && gc0 < K) ? input[gr0 * K + gc0] : half(0);
            A_tile[r1][c1] = (gr1 < M && gc1 < K) ? input[gr1 * K + gc1] : half(0);
        }

        // Load + dequantize B tile (INT8)
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint k0 = idx0 / 8, n0 = idx0 % 8;
            uint k1 = idx1 / 8, n1 = idx1 % 8;

            uint gk0 = k_base + k0, gn0 = out_col_base + n0;
            uint gk1 = k_base + k1, gn1 = out_col_base + n1;

            if (gk0 < K && gn0 < N) {
                float scale = weight_scales[gn0];
                B_tile[k0][n0] = half(float(weight[gn0 * K + gk0]) * scale / 127.0f);
            } else {
                B_tile[k0][n0] = half(0);
            }

            if (gk1 < K && gn1 < N) {
                float scale = weight_scales[gn1];
                B_tile[k1][n1] = half(float(weight[gn1 * K + gk1]) * scale / 127.0f);
            } else {
                B_tile[k1][n1] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> A_mat, B_mat;
        simdgroup_load(A_mat, &A_tile[0][0], 8);
        simdgroup_load(B_mat, &B_tile[0][0], 8);
        simdgroup_multiply_accumulate(C, A_mat, B_mat, C);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    threadgroup float C_tile[8][8];
    simdgroup_store(C, &C_tile[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id * 2; idx < 64; idx += 64) {
        uint r0 = idx / 8, c0 = idx % 8;
        uint r1 = (idx + 1) / 8, c1 = (idx + 1) % 8;

        uint gr0 = out_row_base + r0, gc0 = out_col_base + c0;
        uint gr1 = out_row_base + r1, gc1 = out_col_base + c1;

        if (gr0 < M && gc0 < N) {
            float v = C_tile[r0][c0];
            if (has_bias) v += float(bias[gc0]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr0 * N + gc0] = half(v);
        }
        if (gr1 < M && gc1 < N) {
            float v = C_tile[r1][c1];
            if (has_bias) v += float(bias[gc1]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr1 * N + gc1] = half(v);
        }
    }
}

// =============================================================================
// NF4 Kernels
// =============================================================================

kernel void nf4_quantize(
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
    uint thread_id = tid.x;

    device const float* row_data = input + row * cols;
    device uchar* row_out = output + row * (cols / 2);
    device float* row_absmax = absmax + row * num_blocks;

    for (uint block = 0; block < num_blocks; block++) {
        uint start = block * block_size;
        uint end = min(start + block_size, cols);
        uint len = end - start;

        float local_max = 0.0f;
        for (uint i = thread_id; i < len; i += 256) {
            local_max = max(local_max, abs(row_data[start + i]));
        }
        local_max = simd_max(local_max);

        threadgroup float shared[8];
        if (simd_lane == 0) shared[simd_group] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float m = 0.0f;
            for (uint i = 0; i < 8; i++) m = max(m, shared[i]);
            row_absmax[block] = max(m, 1e-8f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float bmax = row_absmax[block];

        for (uint i = thread_id * 2; i < len; i += 512) {
            uint i0 = start + i, i1 = start + i + 1;
            float v0 = (i0 < cols) ? row_data[i0] / bmax : 0.0f;
            float v1 = (i1 < cols) ? row_data[i1] / bmax : 0.0f;

            uchar q0 = 0, q1 = 0;
            float d0 = INFINITY, d1 = INFINITY;
            for (uchar c = 0; c < 16; c++) {
                float dist0 = abs(v0 - NF4_CODEBOOK[c]);
                float dist1 = abs(v1 - NF4_CODEBOOK[c]);
                if (dist0 < d0) { d0 = dist0; q0 = c; }
                if (dist1 < d1) { d1 = dist1; q1 = c; }
            }
            row_out[i0 / 2] = q0 | (q1 << 4);
        }
    }
}

kernel void nf4_dequantize(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y, col = gid.x;
    if (row >= rows || col >= cols) return;

    uint num_blocks = (cols + block_size - 1) / block_size;
    uint blk = col / block_size;
    float scale = absmax[row * num_blocks + blk];
    output[row * cols + col] = half(dequant_nf4(packed[row * (cols/2) + col/2], col % 2, scale));
}

kernel void nf4_linear_simple(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& K_weight [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y, n = gid.x;
    if (m >= M || n >= N) return;

    uint num_blocks = K_weight / block_size;
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        float scale = absmax[n * num_blocks + k / block_size];
        float w_val = dequant_nf4(weight[n * (K_weight/2) + k/2], k % 2, scale);
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    // Clamp to half range to prevent Inf
    acc = clamp(acc, -FP16_MAX_VAL, FP16_MAX_VAL);
    output[m * N + n] = half(acc);
}

constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 64;

// =============================================================================
// NF4 MatMul with SIMD Group Matrix Operations (Apple Silicon optimized)
// Uses hardware 8x8 matrix multiply for ~2-4x speedup
//
// Computes: C[M,N] = A[M,K] @ B[K,N] where B is dequantized from NF4
// Weight is stored as [N, K] so B[k,n] = dequant(weight[n,k])
// =============================================================================

// Simple version: one simdgroup handles one 8x8 output tile
// K = input dimension (original), K_weight = weight dimension (may be padded)
kernel void nf4_matmul_simd(
    device const half* input [[buffer(0)]],       // [M, K]
    device const uchar* weight [[buffer(1)]],     // [N, K_weight/2] packed
    device const float* absmax [[buffer(2)]],     // [N, num_blocks]
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],               // Input K (original)
    constant uint& K_weight [[buffer(8)]],        // Weight K (padded)
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Each threadgroup is 32 threads (1 simdgroup)
    // Each threadgroup handles one 8x8 output tile
    // tgid.x = output column block (0..N/8)
    // tgid.y = output row block (0..M/8)

    uint out_row_base = tgid.y * 8;
    uint out_col_base = tgid.x * 8;

    // Shared memory for tiles
    threadgroup half A_tile[8][8];   // [8 rows, 8 cols of K]
    threadgroup half B_tile[8][8];   // [8 rows of K, 8 cols]

    // Accumulator
    simdgroup_matrix<float, 8, 8> C;
    C = simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_blocks = K_weight / block_size;  // Use padded K for weight blocks

    // Loop over K in chunks of 8
    for (uint k_base = 0; k_base < K; k_base += 8) {
        // Load A tile [8 x 8] - 64 elements, 32 threads = 2 per thread
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint r0 = idx0 / 8, c0 = idx0 % 8;
            uint r1 = idx1 / 8, c1 = idx1 % 8;

            uint gr0 = out_row_base + r0, gc0 = k_base + c0;
            uint gr1 = out_row_base + r1, gc1 = k_base + c1;

            A_tile[r0][c0] = (gr0 < M && gc0 < K) ? input[gr0 * K + gc0] : half(0);
            A_tile[r1][c1] = (gr1 < M && gc1 < K) ? input[gr1 * K + gc1] : half(0);
        }

        // Load + dequantize B tile [8 x 8]
        // B[k,n] = dequant(weight[n,k])
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint k0 = idx0 / 8, n0 = idx0 % 8;
            uint k1 = idx1 / 8, n1 = idx1 % 8;

            uint gk0 = k_base + k0, gn0 = out_col_base + n0;
            uint gk1 = k_base + k1, gn1 = out_col_base + n1;

            if (gk0 < K && gn0 < N) {
                float scale = absmax[gn0 * num_blocks + gk0 / block_size];
                uchar packed = weight[gn0 * (K_weight/2) + gk0/2];
                B_tile[k0][n0] = half(dequant_nf4(packed, gk0 % 2, scale));
            } else {
                B_tile[k0][n0] = half(0);
            }

            if (gk1 < K && gn1 < N) {
                float scale = absmax[gn1 * num_blocks + gk1 / block_size];
                uchar packed = weight[gn1 * (K_weight/2) + gk1/2];
                B_tile[k1][n1] = half(dequant_nf4(packed, gk1 % 2, scale));
            } else {
                B_tile[k1][n1] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load into simdgroup matrices and multiply
        simdgroup_matrix<half, 8, 8> A_mat, B_mat;
        simdgroup_load(A_mat, &A_tile[0][0], 8);
        simdgroup_load(B_mat, &B_tile[0][0], 8);

        simdgroup_multiply_accumulate(C, A_mat, B_mat, C);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    threadgroup float C_tile[8][8];
    simdgroup_store(C, &C_tile[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane stores 2 elements
    for (uint idx = simd_lane_id * 2; idx < 64; idx += 64) {
        uint r0 = idx / 8, c0 = idx % 8;
        uint r1 = (idx + 1) / 8, c1 = (idx + 1) % 8;

        uint gr0 = out_row_base + r0, gc0 = out_col_base + c0;
        uint gr1 = out_row_base + r1, gc1 = out_col_base + c1;

        if (gr0 < M && gc0 < N) {
            float v = C_tile[r0][c0];
            if (has_bias) v += float(bias[gc0]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr0 * N + gc0] = half(v);
        }
        if (gr1 < M && gc1 < N) {
            float v = C_tile[r1][c1];
            if (has_bias) v += float(bias[gc1]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr1 * N + gc1] = half(v);
        }
    }
}

// =============================================================================
// NF4 MatMul Large Tile - 64x64 output with 8 simdgroups using simdgroup_matrix
// Best for large M (>128) - reduces dispatch overhead, maximizes ALU utilization
// =============================================================================

kernel void nf4_matmul_large(
    device const half* input [[buffer(0)]],       // [M, K]
    device const uchar* weight [[buffer(1)]],     // [N, K_weight/2] packed
    device const float* absmax [[buffer(2)]],     // [N, num_blocks]
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& K_weight [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // 256 threads = 8 simdgroups
    // Each threadgroup handles 64x64 output
    // Each simdgroup handles 8x64 output (8 rows, 64 cols = 8 x 8x8 matrices)
    // simd_group_id: 0-7 = rows 0-7, 8-15, ..., 56-63

    const uint TILE_OUT = 64;
    const uint TILE_K = 32;  // K tile size

    uint out_row_base = tgid.y * TILE_OUT + simd_group_id * 8;
    uint out_col_base = tgid.x * TILE_OUT;

    // Each simdgroup accumulates 8x64 = 1x8 grid of 8x8 simdgroup_matrices
    simdgroup_matrix<float, 8, 8> C[8];
    for (int j = 0; j < 8; j++)
        C[j] = simdgroup_matrix<float, 8, 8>(0.0f);

    // Shared memory for tiles
    threadgroup half A_shared[64][TILE_K + 1];  // [64 rows, 32 K]
    threadgroup half B_shared[TILE_K][64 + 1];  // [32 K, 64 N]

    uint num_blocks = K_weight / block_size;

    // Loop over K dimension in chunks of TILE_K
    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // === Cooperative load of A tile [64 x 32] ===
        // 256 threads, 2048 elements = 8 per thread
        {
            uint thread_id = simd_group_id * 32 + simd_lane_id;
            for (uint i = 0; i < 8; i++) {
                uint flat_idx = thread_id + i * 256;
                uint local_row = flat_idx / TILE_K;
                uint local_col = flat_idx % TILE_K;
                uint global_row = tgid.y * TILE_OUT + local_row;
                uint global_col = k_base + local_col;

                half val = half(0);
                if (global_row < M && global_col < K) {
                    val = input[global_row * K + global_col];
                }
                A_shared[local_row][local_col] = val;
            }
        }

        // === Cooperative load + dequantize B tile [32 x 64] ===
        // B[k,n] = dequant(weight[n,k])
        {
            uint thread_id = simd_group_id * 32 + simd_lane_id;
            for (uint i = 0; i < 8; i++) {
                uint flat_idx = thread_id + i * 256;
                uint local_k = flat_idx / 64;
                uint local_n = flat_idx % 64;
                uint global_k = k_base + local_k;
                uint global_n = tgid.x * TILE_OUT + local_n;

                half val = half(0);
                if (global_k < K && global_n < N) {
                    float scale = absmax[global_n * num_blocks + global_k / block_size];
                    uchar packed = weight[global_n * (K_weight/2) + global_k/2];
                    val = half(dequant_nf4(packed, global_k % 2, scale));
                }
                B_shared[local_k][local_n] = val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Each simdgroup computes its 8x64 output ===
        // Loop over K in chunks of 8 for simdgroup_matrix
        for (uint k_inner = 0; k_inner < TILE_K; k_inner += 8) {
            // Load A matrix for this simdgroup's rows [8x8]
            simdgroup_matrix<half, 8, 8> A_mat;
            simdgroup_load(A_mat, &A_shared[simd_group_id * 8][k_inner], TILE_K + 1);

            // Load B matrices for all 8 columns [8x8 each]
            simdgroup_matrix<half, 8, 8> B_mat[8];
            for (int ni = 0; ni < 8; ni++) {
                simdgroup_load(B_mat[ni], &B_shared[k_inner][ni * 8], 65);
            }

            // Multiply-accumulate: C[ni] += A_mat * B_mat[ni]
            for (int ni = 0; ni < 8; ni++) {
                simdgroup_multiply_accumulate(C[ni], A_mat, B_mat[ni], C[ni]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store results directly to global memory ===
    // Each simdgroup stores its 8x64 (8 x 8x8 matrices)
    // Use separate shared memory per simdgroup to avoid conflicts
    threadgroup float C_all[8][8][65];  // [simdgroup][row][col]

    for (int ni = 0; ni < 8; ni++) {
        simdgroup_store(C[ni], &C_all[simd_group_id][0][ni * 8], 65);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane writes 16 elements (512 / 32 = 16)
    for (uint i = 0; i < 16; i++) {
        uint flat_idx = simd_lane_id + i * 32;
        uint local_row = flat_idx / 64;
        uint local_col = flat_idx % 64;
        uint global_row = out_row_base + local_row;
        uint global_col = out_col_base + local_col;

        if (global_row < M && global_col < N) {
            float v = C_all[simd_group_id][local_row][local_col];
            if (has_bias) v += float(bias[global_col]);
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[global_row * N + global_col] = half(v);
        }
    }
}

kernel void nf4_matmul_fused(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& K_weight [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half As[TILE_M][TILE_K + 1];
    threadgroup half Bs[TILE_K][TILE_N + 1];

    uint tx = tid.x, ty = tid.y;
    uint thread_id = ty * 8 + tx;
    uint row_base = tgid.y * TILE_M + ty * 4;
    uint col_base = tgid.x * TILE_N + tx * 4;

    float acc[4][4] = {{0.0f}};
    uint num_blocks = K_weight / block_size;

    for (uint t = 0; t < K; t += TILE_K) {
        for (uint i = 0; i < (TILE_M * TILE_K) / 64; i++) {
            uint fi = thread_id + i * 64;
            uint lr = fi / TILE_K, lc = fi % TILE_K;
            uint gr = tgid.y * TILE_M + lr, gc = t + lc;
            As[lr][lc] = (gr < M && gc < K) ? input[gr * K + gc] : half(0);
        }

        for (uint i = 0; i < (TILE_K * TILE_N) / 64; i++) {
            uint fi = thread_id + i * 64;
            uint lk = fi / TILE_N, ln = fi % TILE_N;
            uint gk = t + lk, gn = tgid.x * TILE_N + ln;
            if (gk < K && gn < N) {
                float scale = absmax[gn * num_blocks + gk / block_size];
                Bs[lk][ln] = half(dequant_nf4(weight[gn * (K_weight/2) + gk/2], gk % 2, scale));
            } else {
                Bs[lk][ln] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_K; k++) {
            half av[4], bv[4];
            for (uint m = 0; m < 4; m++) av[m] = As[ty * 4 + m][k];
            for (uint n = 0; n < 4; n++) bv[n] = Bs[k][tx * 4 + n];
            for (uint m = 0; m < 4; m++)
                for (uint n = 0; n < 4; n++)
                    acc[m][n] += float(av[m]) * float(bv[n]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint m = 0; m < 4; m++) {
        uint or_ = row_base + m;
        if (or_ >= M) continue;
        for (uint n = 0; n < 4; n++) {
            uint oc = col_base + n;
            if (oc >= N) continue;
            float r = acc[m][n];
            if (has_bias) r += float(bias[oc]);
            // Clamp to half range to prevent Inf
            r = clamp(r, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[or_ * N + oc] = half(r);
        }
    }
}

// =============================================================================
// FP4 Kernels
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
    uint thread_id = tid.x;

    device const float* row_data = input + row * cols;
    device uchar* row_out = output + row * (cols / 2);
    device float* row_absmax = absmax + row * num_blocks;

    for (uint block = 0; block < num_blocks; block++) {
        uint start = block * block_size;
        uint end = min(start + block_size, cols);
        uint len = end - start;

        float local_max = 0.0f;
        for (uint i = thread_id; i < len; i += 256) {
            local_max = max(local_max, abs(row_data[start + i]));
        }
        local_max = simd_max(local_max);

        threadgroup float shared[8];
        if (simd_lane == 0) shared[simd_group] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float m = 0.0f;
            for (uint i = 0; i < 8; i++) m = max(m, shared[i]);
            row_absmax[block] = max(m, 1e-8f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float bmax = row_absmax[block];

        for (uint i = thread_id * 2; i < len; i += 512) {
            uint i0 = start + i, i1 = start + i + 1;
            float v0 = (i0 < cols) ? row_data[i0] / bmax : 0.0f;
            float v1 = (i1 < cols) ? row_data[i1] / bmax : 0.0f;

            uchar q0 = 0, q1 = 0;
            float d0 = INFINITY, d1 = INFINITY;
            for (uchar c = 0; c < 16; c++) {
                float dist0 = abs(v0 - FP4_CODEBOOK[c]);
                float dist1 = abs(v1 - FP4_CODEBOOK[c]);
                if (dist0 < d0) { d0 = dist0; q0 = c; }
                if (dist1 < d1) { d1 = dist1; q1 = c; }
            }
            row_out[i0 / 2] = q0 | (q1 << 4);
        }
    }
}

kernel void fp4_dequantize(
    device const uchar* packed [[buffer(0)]],
    device const float* absmax [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y, col = gid.x;
    if (row >= rows || col >= cols) return;

    uint num_blocks = (cols + block_size - 1) / block_size;
    uint blk = col / block_size;
    float scale = absmax[row * num_blocks + blk];
    output[row * cols + col] = half(dequant_fp4(packed[row * (cols/2) + col/2], col % 2, scale));
}

kernel void fp4_linear_simple(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& K_weight [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y, n = gid.x;
    if (m >= M || n >= N) return;

    uint num_blocks = K_weight / block_size;
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        float scale = absmax[n * num_blocks + k / block_size];
        float w_val = dequant_fp4(weight[n * (K_weight/2) + k/2], k % 2, scale);
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    // Clamp to half range to prevent Inf
    acc = clamp(acc, -FP16_MAX_VAL, FP16_MAX_VAL);
    output[m * N + n] = half(acc);
}

// =============================================================================
// FP4 MatMul with SIMD Group Matrix Operations (Apple Silicon optimized)
// =============================================================================

kernel void fp4_matmul_simd(
    device const half* input [[buffer(0)]],       // [M, K]
    device const uchar* weight [[buffer(1)]],     // [N, K_weight/2] packed
    device const float* absmax [[buffer(2)]],     // [N, num_blocks]
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& K_weight [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    constant uint& has_bias [[buffer(10)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint out_row_base = tgid.y * 8;
    uint out_col_base = tgid.x * 8;

    threadgroup half A_tile[8][8];
    threadgroup half B_tile[8][8];

    simdgroup_matrix<float, 8, 8> C;
    C = simdgroup_matrix<float, 8, 8>(0.0f);

    uint num_blocks = K_weight / block_size;

    for (uint k_base = 0; k_base < K; k_base += 8) {
        // Load A tile
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint r0 = idx0 / 8, c0 = idx0 % 8;
            uint r1 = idx1 / 8, c1 = idx1 % 8;

            uint gr0 = out_row_base + r0, gc0 = k_base + c0;
            uint gr1 = out_row_base + r1, gc1 = k_base + c1;

            A_tile[r0][c0] = (gr0 < M && gc0 < K) ? input[gr0 * K + gc0] : half(0);
            A_tile[r1][c1] = (gr1 < M && gc1 < K) ? input[gr1 * K + gc1] : half(0);
        }

        // Load + dequantize B tile (FP4)
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint k0 = idx0 / 8, n0 = idx0 % 8;
            uint k1 = idx1 / 8, n1 = idx1 % 8;

            uint gk0 = k_base + k0, gn0 = out_col_base + n0;
            uint gk1 = k_base + k1, gn1 = out_col_base + n1;

            if (gk0 < K && gn0 < N) {
                float scale = absmax[gn0 * num_blocks + gk0 / block_size];
                uchar packed = weight[gn0 * (K_weight/2) + gk0/2];
                B_tile[k0][n0] = half(dequant_fp4(packed, gk0 % 2, scale));
            } else {
                B_tile[k0][n0] = half(0);
            }

            if (gk1 < K && gn1 < N) {
                float scale = absmax[gn1 * num_blocks + gk1 / block_size];
                uchar packed = weight[gn1 * (K_weight/2) + gk1/2];
                B_tile[k1][n1] = half(dequant_fp4(packed, gk1 % 2, scale));
            } else {
                B_tile[k1][n1] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> A_mat, B_mat;
        simdgroup_load(A_mat, &A_tile[0][0], 8);
        simdgroup_load(B_mat, &B_tile[0][0], 8);
        simdgroup_multiply_accumulate(C, A_mat, B_mat, C);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    threadgroup float C_tile[8][8];
    simdgroup_store(C, &C_tile[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id * 2; idx < 64; idx += 64) {
        uint r0 = idx / 8, c0 = idx % 8;
        uint r1 = (idx + 1) / 8, c1 = (idx + 1) % 8;

        uint gr0 = out_row_base + r0, gc0 = out_col_base + c0;
        uint gr1 = out_row_base + r1, gc1 = out_col_base + c1;

        if (gr0 < M && gc0 < N) {
            float v = C_tile[r0][c0];
            if (has_bias) v += float(bias[gc0]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr0 * N + gc0] = half(v);
        }
        if (gr1 < M && gc1 < N) {
            float v = C_tile[r1][c1];
            if (has_bias) v += float(bias[gc1]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr1 * N + gc1] = half(v);
        }
    }
}

// =============================================================================
// FP8 Kernels
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
    device const float* row_data = input + row * cols;
    device uchar* row_out = output + row * cols;

    float local_max = 0.0f;
    for (uint i = thread_id; i < cols; i += 256) {
        local_max = max(local_max, abs(row_data[i]));
    }
    local_max = simd_max(local_max);

    threadgroup float shared[8];
    if (simd_lane == 0) shared[simd_group] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_id == 0) {
        float m = 0.0f;
        for (uint i = 0; i < 8; i++) m = max(m, shared[i]);
        scales[row] = max(m / 448.0f, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = scales[row];
    for (uint i = thread_id; i < cols; i += 256) {
        row_out[i] = float_to_fp8_e4m3(row_data[i] / scale);
    }
}

kernel void fp8_e4m3_dequantize(
    device const uchar* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y, col = gid.x;
    if (row >= rows || col >= cols) return;

    float scale = scales[row];
    float val = fp8_e4m3_to_float(input[row * cols + col]) * scale;
    output[row * cols + col] = half(val);
}

kernel void fp8_e4m3_linear(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y, n = gid.x;
    if (m >= M || n >= N) return;

    float scale = scales[n];
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        float w_val = fp8_e4m3_to_float(weight[n * K + k]) * scale;
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    // Clamp to half range to prevent Inf
    acc = clamp(acc, -FP16_MAX_VAL, FP16_MAX_VAL);
    output[m * N + n] = half(acc);
}

// =============================================================================
// FP8 MatMul with SIMD Group Matrix Operations (Apple Silicon optimized)
// =============================================================================

kernel void fp8_matmul_simd(
    device const half* input [[buffer(0)]],       // [M, K]
    device const uchar* weight [[buffer(1)]],     // [N, K] FP8 packed
    device const float* scales [[buffer(2)]],     // [N] per-row scales
    device const half* bias [[buffer(3)]],        // [N]
    device half* output [[buffer(4)]],            // [M, N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& has_bias [[buffer(8)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint out_row_base = tgid.y * 8;
    uint out_col_base = tgid.x * 8;

    threadgroup half A_tile[8][8];
    threadgroup half B_tile[8][8];

    simdgroup_matrix<float, 8, 8> C;
    C = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_base = 0; k_base < K; k_base += 8) {
        // Load A tile
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint r0 = idx0 / 8, c0 = idx0 % 8;
            uint r1 = idx1 / 8, c1 = idx1 % 8;

            uint gr0 = out_row_base + r0, gc0 = k_base + c0;
            uint gr1 = out_row_base + r1, gc1 = k_base + c1;

            A_tile[r0][c0] = (gr0 < M && gc0 < K) ? input[gr0 * K + gc0] : half(0);
            A_tile[r1][c1] = (gr1 < M && gc1 < K) ? input[gr1 * K + gc1] : half(0);
        }

        // Load + dequantize B tile (FP8)
        {
            uint idx0 = simd_lane_id * 2;
            uint idx1 = simd_lane_id * 2 + 1;

            uint k0 = idx0 / 8, n0 = idx0 % 8;
            uint k1 = idx1 / 8, n1 = idx1 % 8;

            uint gk0 = k_base + k0, gn0 = out_col_base + n0;
            uint gk1 = k_base + k1, gn1 = out_col_base + n1;

            if (gk0 < K && gn0 < N) {
                float scale = scales[gn0];
                B_tile[k0][n0] = half(fp8_e4m3_to_float(weight[gn0 * K + gk0]) * scale);
            } else {
                B_tile[k0][n0] = half(0);
            }

            if (gk1 < K && gn1 < N) {
                float scale = scales[gn1];
                B_tile[k1][n1] = half(fp8_e4m3_to_float(weight[gn1 * K + gk1]) * scale);
            } else {
                B_tile[k1][n1] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> A_mat, B_mat;
        simdgroup_load(A_mat, &A_tile[0][0], 8);
        simdgroup_load(B_mat, &B_tile[0][0], 8);
        simdgroup_multiply_accumulate(C, A_mat, B_mat, C);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result
    threadgroup float C_tile[8][8];
    simdgroup_store(C, &C_tile[0][0], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lane_id * 2; idx < 64; idx += 64) {
        uint r0 = idx / 8, c0 = idx % 8;
        uint r1 = (idx + 1) / 8, c1 = (idx + 1) % 8;

        uint gr0 = out_row_base + r0, gc0 = out_col_base + c0;
        uint gr1 = out_row_base + r1, gc1 = out_col_base + c1;

        if (gr0 < M && gc0 < N) {
            float v = C_tile[r0][c0];
            if (has_bias) v += float(bias[gc0]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr0 * N + gc0] = half(v);
        }
        if (gr1 < M && gc1 < N) {
            float v = C_tile[r1][c1];
            if (has_bias) v += float(bias[gc1]);
            // Clamp to half range to prevent Inf
            v = clamp(v, -FP16_MAX_VAL, FP16_MAX_VAL);
            output[gr1 * N + gc1] = half(v);
        }
    }
}

// =============================================================================
// Double Quantization (quantize absmax with INT8)
// =============================================================================

// =============================================================================
// Embedding Kernels
// =============================================================================

kernel void embedding_4bit_nf4(
    device const uint* indices [[buffer(0)]],
    device const uchar* weight_packed [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& num_indices [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    constant uint& num_blocks [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint idx_pos = tgid.y;
    if (idx_pos >= num_indices) return;

    uint emb_idx = indices[idx_pos];
    uint thread_id = tid.x;
    uint packed_dim = embedding_dim / 2;

    device const uchar* row_packed = weight_packed + emb_idx * packed_dim;
    device const float* row_absmax = absmax + emb_idx * num_blocks;
    device half* out_row = output + idx_pos * embedding_dim;

    for (uint col = thread_id; col < embedding_dim; col += 256) {
        uint blk = col / block_size;
        float scale = row_absmax[blk];
        uchar packed = row_packed[col / 2];
        float val = dequant_nf4(packed, col % 2, scale);
        out_row[col] = half(val);
    }
}

kernel void embedding_4bit_fp4(
    device const uint* indices [[buffer(0)]],
    device const uchar* weight_packed [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& num_indices [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    constant uint& block_size [[buffer(6)]],
    constant uint& num_blocks [[buffer(7)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    uint idx_pos = tgid.y;
    if (idx_pos >= num_indices) return;

    uint emb_idx = indices[idx_pos];
    uint thread_id = tid.x;
    uint packed_dim = embedding_dim / 2;

    device const uchar* row_packed = weight_packed + emb_idx * packed_dim;
    device const float* row_absmax = absmax + emb_idx * num_blocks;
    device half* out_row = output + idx_pos * embedding_dim;

    for (uint col = thread_id; col < embedding_dim; col += 256) {
        uint blk = col / block_size;
        float scale = row_absmax[blk];
        uchar packed = row_packed[col / 2];
        float val = dequant_fp4(packed, col % 2, scale);
        out_row[col] = half(val);
    }
}

kernel void embedding_8bit(
    device const uint* indices [[buffer(0)]],
    device const char* weight_int8 [[buffer(1)]],
    device const float* weight_scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& num_indices [[buffer(4)]],
    constant uint& embedding_dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint idx_pos = gid.y;
    if (col >= embedding_dim || idx_pos >= num_indices) return;

    uint emb_idx = indices[idx_pos];
    float scale = weight_scales[emb_idx];
    float val = float(weight_int8[emb_idx * embedding_dim + col]) * scale / 127.0f;
    output[idx_pos * embedding_dim + col] = half(val);
}

// =============================================================================
// 8-bit Optimizer Kernels
// =============================================================================

kernel void adam8bit_step(
    device half* param [[buffer(0)]],
    device const half* grad [[buffer(1)]],
    device char* exp_avg_int8 [[buffer(2)]],
    device float* exp_avg_absmax [[buffer(3)]],
    device uchar* exp_avg_sq_u8 [[buffer(4)]],
    device float* exp_avg_sq_max [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& lr [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    constant uint& step [[buffer(11)]],
    constant uint& N [[buffer(12)]],
    constant uint& block_size [[buffer(13)]],
    constant uint& is_adamw [[buffer(14)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint block_idx = tgid;
    uint block_start = block_idx * block_size;
    if (block_start >= N) return;
    uint block_end = min(block_start + block_size, N);
    uint block_len = block_end - block_start;

    // Dequantize exp_avg (signed int8)
    float m_absmax = exp_avg_absmax[block_idx];
    // Dequantize exp_avg_sq (unsigned uint8 with sqrt compression)
    float v_max = exp_avg_sq_max[block_idx];

    // Bias correction
    float bc1 = 1.0f - pow(beta1, float(step));
    float bc2 = 1.0f - pow(beta2, float(step));
    float step_size = lr / bc1;

    // Process elements in this block
    // Phase 1: Update moments and param, track new absmax
    float new_m_max = 0.0f;
    float new_v_max = 0.0f;

    // We process multiple elements per thread
    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;

        // Load param and grad
        float p = float(param[global_idx]);
        float g = float(grad[global_idx]);

        // Dequantize m
        float m = float(exp_avg_int8[global_idx]) * (m_absmax / 127.0f);
        // Dequantize v (reverse sqrt compression)
        float v_sqrt = float(exp_avg_sq_u8[global_idx]) / 255.0f;
        float v = v_sqrt * v_sqrt * v_max;

        // Weight decay
        if (is_adamw != 0) {
            // AdamW: decoupled weight decay
            p = p * (1.0f - lr * weight_decay);
        } else {
            // Adam: L2 regularization on gradient
            g = g + weight_decay * p;
        }

        // Update moments
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        // Update param
        float v_hat = v / bc2;
        float update = m * step_size / (sqrt(v_hat) + eps);
        p = p - update;
        param[global_idx] = half(p);

        // Track absmax for requantization
        new_m_max = max(new_m_max, abs(m));
        new_v_max = max(new_v_max, v);

        // Store float values temporarily in the int8 buffers
        // We'll requantize after the reduction
        // Use a two-pass approach: store to threadgroup memory
    }

    // SIMD reduction for new_m_max
    new_m_max = simd_max(new_m_max);
    new_v_max = simd_max(new_v_max);

    threadgroup float shared_m[8];
    threadgroup float shared_v[8];
    if (simd_lane == 0) {
        shared_m[simd_group] = new_m_max;
        shared_v[simd_group] = new_v_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float m_max = 0.0f, v_mx = 0.0f;
        for (uint i = 0; i < 8; i++) {
            m_max = max(m_max, shared_m[i]);
            v_mx = max(v_mx, shared_v[i]);
        }
        exp_avg_absmax[block_idx] = max(m_max, 1e-8f);
        exp_avg_sq_max[block_idx] = max(v_mx, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Requantize with new absmax
    float final_m_max = exp_avg_absmax[block_idx];
    float final_v_max = exp_avg_sq_max[block_idx];

    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;

        // Re-compute moments (same math as above)
        float p_orig = float(param[global_idx]);
        // We need the updated m and v values - recompute from original
        float g = float(grad[global_idx]);
        float m = float(exp_avg_int8[global_idx]) * (m_absmax / 127.0f);
        float v_sqrt = float(exp_avg_sq_u8[global_idx]) / 255.0f;
        float v = v_sqrt * v_sqrt * v_max;

        if (is_adamw == 0) {
            // Need to reconstruct the adjusted gradient for L2
            // But param already updated, so we just recompute moments
            float p_before_update = p_orig; // already updated, get original
            g = g + weight_decay * float(grad[global_idx]); // approx
        }

        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;

        // Requantize m -> int8
        exp_avg_int8[global_idx] = char(clamp(round(m / final_m_max * 127.0f), -127.0f, 127.0f));
        // Requantize v -> uint8 with sqrt compression
        float v_norm = v / final_v_max;
        exp_avg_sq_u8[global_idx] = uchar(clamp(round(sqrt(v_norm) * 255.0f), 0.0f, 255.0f));
    }
}

kernel void lion8bit_step(
    device half* param [[buffer(0)]],
    device const half* grad [[buffer(1)]],
    device char* exp_avg_int8 [[buffer(2)]],
    device float* exp_avg_absmax [[buffer(3)]],
    constant float& beta1 [[buffer(4)]],
    constant float& beta2 [[buffer(5)]],
    constant float& lr [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    constant uint& N [[buffer(8)]],
    constant uint& block_size [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint block_idx = tgid;
    uint block_start = block_idx * block_size;
    if (block_start >= N) return;
    uint block_end = min(block_start + block_size, N);
    uint block_len = block_end - block_start;

    float m_absmax = exp_avg_absmax[block_idx];
    float new_m_max = 0.0f;

    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;

        float p = float(param[global_idx]);
        float g = float(grad[global_idx]);

        // Dequantize m
        float m = float(exp_avg_int8[global_idx]) * (m_absmax / 127.0f);

        // Weight decay (decoupled)
        p = p * (1.0f - lr * weight_decay);

        // Lion update: sign of interpolation
        float u = beta1 * m + (1.0f - beta1) * g;
        p = p - lr * sign(u);
        param[global_idx] = half(p);

        // Update momentum for next step
        m = beta2 * m + (1.0f - beta2) * g;
        new_m_max = max(new_m_max, abs(m));

        // Store m temporarily (will requantize after reduction)
        // We use a float reinterpret trick - store as int bits
        // Actually, we need a second pass. Store m in shared or recompute.
    }

    // SIMD reduction for new_m_max
    new_m_max = simd_max(new_m_max);
    threadgroup float shared[8];
    if (simd_lane == 0) shared[simd_group] = new_m_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float mx = 0.0f;
        for (uint i = 0; i < 8; i++) mx = max(mx, shared[i]);
        exp_avg_absmax[block_idx] = max(mx, 1e-8f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Requantize
    float final_m_max = exp_avg_absmax[block_idx];
    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;
        float g = float(grad[global_idx]);
        float m = float(exp_avg_int8[global_idx]) * (m_absmax / 127.0f);
        m = beta2 * m + (1.0f - beta2) * g;
        exp_avg_int8[global_idx] = char(clamp(round(m / final_m_max * 127.0f), -127.0f, 127.0f));
    }
}

kernel void sgd8bit_step(
    device half* param [[buffer(0)]],
    device const half* grad [[buffer(1)]],
    device char* momentum_int8 [[buffer(2)]],
    device float* momentum_absmax [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& momentum_val [[buffer(5)]],
    constant float& dampening [[buffer(6)]],
    constant float& weight_decay [[buffer(7)]],
    constant uint& nesterov [[buffer(8)]],
    constant uint& N [[buffer(9)]],
    constant uint& block_size [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint block_idx = tgid;
    uint block_start = block_idx * block_size;
    if (block_start >= N) return;
    uint block_end = min(block_start + block_size, N);
    uint block_len = block_end - block_start;

    float buf_absmax = momentum_absmax[block_idx];
    float new_buf_max = 0.0f;

    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;

        float p = float(param[global_idx]);
        float g = float(grad[global_idx]);

        // Weight decay (L2)
        g = g + weight_decay * p;

        // Dequantize momentum buffer
        float buf = float(momentum_int8[global_idx]) * (buf_absmax / 127.0f);

        // Update momentum buffer
        buf = momentum_val * buf + (1.0f - dampening) * g;
        new_buf_max = max(new_buf_max, abs(buf));

        // Apply update
        float update;
        if (nesterov != 0) {
            update = g + momentum_val * buf;
        } else {
            update = buf;
        }
        p = p - lr * update;
        param[global_idx] = half(p);
    }

    // SIMD reduction
    new_buf_max = simd_max(new_buf_max);
    threadgroup float shared[8];
    if (simd_lane == 0) shared[simd_group] = new_buf_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float mx = 0.0f;
        for (uint i = 0; i < 8; i++) mx = max(mx, shared[i]);
        momentum_absmax[block_idx] = max(mx, 1e-8f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Requantize
    float final_buf_max = momentum_absmax[block_idx];
    for (uint i = tid; i < block_len; i += 256) {
        uint global_idx = block_start + i;
        float p = float(param[global_idx]);
        float g = float(grad[global_idx]);
        g = g + weight_decay * p;
        float buf = float(momentum_int8[global_idx]) * (buf_absmax / 127.0f);
        buf = momentum_val * buf + (1.0f - dampening) * g;
        momentum_int8[global_idx] = char(clamp(round(buf / final_buf_max * 127.0f), -127.0f, 127.0f));
    }
}

// =============================================================================
// Sparse MatMul Kernels (CSR format)
// =============================================================================

kernel void spmm_csr(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const float* values [[buffer(2)]],
    device const half* dense [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= M || col >= N) return;

    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];

    float acc = 0.0f;
    for (uint i = start; i < end; i++) {
        uint k = col_indices[i];
        acc += values[i] * float(dense[k * N + col]);
    }
    output[row * N + col] = half(acc);
}

kernel void spmm_csr_int8(
    device const uint* row_ptr [[buffer(0)]],
    device const uint* col_indices [[buffer(1)]],
    device const char* values_int8 [[buffer(2)]],
    device const float& values_scale [[buffer(3)]],
    device const half* dense [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant uint& M [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    constant uint& K [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;
    if (row >= M || col >= N) return;

    uint start = row_ptr[row];
    uint end = row_ptr[row + 1];

    float acc = 0.0f;
    for (uint i = start; i < end; i++) {
        uint k = col_indices[i];
        float val = float(values_int8[i]) * values_scale;
        acc += val * float(dense[k * N + col]);
    }
    output[row * N + col] = half(acc);
}

kernel void double_quant_absmax(
    device const float* absmax [[buffer(0)]],
    device uchar* absmax_quant [[buffer(1)]],
    device float* absmax_scales [[buffer(2)]],
    constant uint& num_rows [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    constant uint& double_quant_block [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.y;
    if (row >= num_rows) return;

    uint thread_id = tid.x;
    uint total_blocks = blocks_per_row;
    uint dq_blocks = (total_blocks + double_quant_block - 1) / double_quant_block;

    device const float* row_absmax = absmax + row * blocks_per_row;
    device uchar* row_quant = absmax_quant + row * blocks_per_row;
    device float* row_scales = absmax_scales + row * dq_blocks;

    for (uint dqb = 0; dqb < dq_blocks; dqb++) {
        uint start = dqb * double_quant_block;
        uint end = min(start + double_quant_block, total_blocks);
        uint len = end - start;

        float local_max = 0.0f;
        for (uint i = thread_id; i < len; i += 256) {
            local_max = max(local_max, abs(row_absmax[start + i]));
        }
        local_max = simd_max(local_max);

        threadgroup float shared[8];
        if (simd_lane == 0) shared[simd_group] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (thread_id == 0) {
            float m = 0.0f;
            for (uint i = 0; i < 8; i++) m = max(m, shared[i]);
            row_scales[dqb] = max(m / 127.0f, 1e-12f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scale = row_scales[dqb];
        for (uint i = thread_id; i < len; i += 256) {
            float val = row_absmax[start + i] / scale;
            row_quant[start + i] = uchar(clamp(round(val), 0.0f, 255.0f));
        }
    }
}
)";

// =============================================================================
// Initialize Metal Library (Thread-Safe)
// =============================================================================

static std::atomic<bool> g_library_initialized{false};
static std::mutex g_library_mutex;
static std::mutex g_pipeline_mutex;

static void init_library() {
    // Fast path: already initialized
    if (g_library_initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Slow path: acquire lock and initialize
    std::lock_guard<std::mutex> lock(g_library_mutex);

    // Double-check after acquiring lock
    if (g_library_initialized.load(std::memory_order_relaxed)) {
        return;
    }

    ensure_device();

    @autoreleasepool {
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        NSString* source = [NSString stringWithUTF8String:KERNELS_SOURCE];
        g_library = [g_device newLibraryWithSource:source options:options error:&error];

        if (!g_library) {
            throw std::runtime_error("Failed to compile Metal library: " +
                std::string([[error localizedDescription] UTF8String]));
        }
    }

    g_library_initialized.store(true, std::memory_order_release);
}

static id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    init_library();

    // Check cache with lock (pipelines map is not thread-safe)
    {
        std::lock_guard<std::mutex> lock(g_pipeline_mutex);
        auto it = g_pipelines.find(name);
        if (it != g_pipelines.end()) {
            return it->second;
        }
    }

    // Create pipeline (can be done outside lock)
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> fn = [g_library newFunctionWithName:
            [NSString stringWithUTF8String:name.c_str()]];
        if (!fn) {
            throw std::runtime_error("Function not found: " + name);
        }

        pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            throw std::runtime_error("Failed to create pipeline: " + name);
        }
    }

    // Store in cache with lock
    {
        std::lock_guard<std::mutex> lock(g_pipeline_mutex);
        // Another thread may have added it, but that's OK - overwrite is safe
        g_pipelines[name] = pipeline;
    }

    return pipeline;
}

// =============================================================================
// INT8 Operations
// =============================================================================

at::Tensor matmul_int8_mps(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& A_scales,
    const at::Tensor& B_scales,
    at::ScalarType out_dtype
) {
    TORCH_CHECK(A.device().is_mps() && B.device().is_mps(), "Inputs must be on MPS");
    TORCH_CHECK(A.dtype() == at::kChar && B.dtype() == at::kChar, "Inputs must be int8");

    const int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto A_s = A_scales.to(at::kFloat).contiguous();
    auto B_s = B_scales.to(at::kFloat).contiguous();
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    // Kernels always output fp16, we convert to requested dtype at the end
    auto output_fp16 = at::empty({M, N}, A.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("int8_matmul_dequant");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(A_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(B_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output_fp16) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(A_s) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(B_s) offset:0 atIndex:4];

        uint32_t dims[3] = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
        [encoder setBytes:&dims[0] length:4 atIndex:5];
        [encoder setBytes:&dims[1] length:4 atIndex:6];
        [encoder setBytes:&dims[2] length:4 atIndex:7];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    // Convert to requested output dtype
    return (out_dtype == at::kHalf) ? output_fp16 : output_fp16.to(out_dtype);
}

// INT8 linear layer: half input @ int8 weight (with row-wise scales)
at::Tensor linear_int8_mps(const at::Tensor& input, const at::Tensor& weight,
                           const at::Tensor& weight_scales, const std::optional<at::Tensor>& bias,
                           at::ScalarType dtype) {
    bool is_1d = input.dim() == 1;
    auto input_2d = is_1d ? input.unsqueeze(0) : input;

    const int64_t M = input_2d.size(0), K = input_2d.size(1), N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "Weight K mismatch");

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight.contiguous();
    auto scales_c = weight_scales.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    // Kernels always output fp16, we convert to requested dtype at the end
    auto output_fp16 = at::empty({M, N}, input.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();

        // Use SIMD kernel for small batches (M <= 128)
        bool use_simd = (M <= 128 && N >= 8 && K >= 8);
        auto pipeline = get_pipeline(use_simd ? "int8_matmul_simd" : "int8_matmul_dequant");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output_fp16) offset:0 atIndex:4];

        uint32_t params[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, has_bias ? 1u : 0u};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        if (use_simd) {
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize ntg = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else {
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        }
    }

    // Convert to requested output dtype
    auto output = (dtype == at::kHalf) ? output_fp16 : output_fp16.to(dtype);
    return is_1d ? output.squeeze(0) : output;
}

// =============================================================================
// NF4 Operations
// =============================================================================

std::tuple<at::Tensor, at::Tensor> quantize_nf4_mps(const at::Tensor& input, int64_t block_size) {
    TORCH_CHECK(input.device().is_mps(), "Input must be on MPS");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(input.size(1) % 2 == 0, "Cols must be even");

    const int64_t rows = input.size(0), cols = input.size(1);
    const int64_t num_blocks = (cols + block_size - 1) / block_size;

    auto input_f32 = input.to(at::kFloat).contiguous();
    auto packed = at::empty({rows, cols / 2}, input.options().dtype(at::kByte));
    auto absmax = at::empty({rows, num_blocks}, input.options().dtype(at::kFloat));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("nf4_quantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_f32) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(packed) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax) offset:0 atIndex:2];

        uint32_t params[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)block_size};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];
        [encoder setBytes:&params[2] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(256, rows, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return std::make_tuple(packed, absmax);
}

at::Tensor dequantize_nf4_mps(const at::Tensor& packed, const at::Tensor& absmax, int64_t block_size, at::ScalarType dtype) {
    TORCH_CHECK(packed.device().is_mps() && absmax.device().is_mps(), "Inputs must be on MPS");

    const int64_t rows = packed.size(0), cols = packed.size(1) * 2;
    auto output = at::empty({rows, cols}, packed.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("nf4_dequantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(packed) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax.to(at::kFloat).contiguous()) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:2];

        uint32_t params[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)block_size};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];
        [encoder setBytes:&params[2] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((cols + 15) / 16 * 16, (rows + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor matmul_nf4_mps(const at::Tensor& input, const at::Tensor& weight_packed,
                          const at::Tensor& weight_absmax, const std::optional<at::Tensor>& bias,
                          int64_t block_size, at::ScalarType dtype) {
    TORCH_CHECK(input.device().is_mps(), "Input must be on MPS");

    bool is_1d = input.dim() == 1;
    auto input_2d = is_1d ? input.unsqueeze(0) : input;

    const int64_t M = input_2d.size(0), K = input_2d.size(1), N = weight_packed.size(0);
    // K_weight is the padded K dimension used for weight storage
    const int64_t K_weight = weight_packed.size(1) * 2;

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = weight_absmax.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    // Kernels always output fp16, we convert to requested dtype at the end
    auto output_fp16 = at::empty({M, N}, input.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();

        // Choose kernel based on matrix size:
        // - simd: M <= 128, N >= 8, K >= 8 (8x8 tiles, 1 simdgroup - up to 15x faster)
        // - large: 128 < M <= 512, N >= 64, K >= 32 (64x64 tiles, 8 simdgroups - 1.1-1.9x faster)
        // - For M > 512: fall back to dequant+matmul (PyTorch's GEMM is faster)
        // - tiled: fallback for edge cases
        // - simple: fallback for very small matrices
        bool use_simd = (M <= 128 && N >= 8 && K >= 8);
        bool use_large = (M > 128 && M <= 512 && N >= 64 && K >= 32);
        bool use_tiled = !use_simd && !use_large && (M >= 32 && N >= 32 && K >= 64);

        const char* kernel_name = use_simd ? "nf4_matmul_simd" :
                                  use_large ? "nf4_matmul_large" :
                                  use_tiled ? "nf4_matmul_fused" : "nf4_linear_simple";
        auto pipeline = get_pipeline(kernel_name);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output_fp16) offset:0 atIndex:4];

        uint32_t params[6] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)K_weight, (uint32_t)block_size, has_bias ? 1u : 0u};
        for (int i = 0; i < 6; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        if (use_simd) {
            // SIMD kernel: 32 threads (1 simdgroup) per threadgroup
            // Each threadgroup handles 8x8 output
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize ntg = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else if (use_large) {
            // Large kernel: 256 threads (8 simdgroups) per threadgroup
            // Each threadgroup handles 64x64 output
            MTLSize tg = MTLSizeMake(256, 1, 1);
            MTLSize ntg = MTLSizeMake((N + 63) / 64, (M + 63) / 64, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else if (use_tiled) {
            MTLSize tg = MTLSizeMake(8, 8, 1);
            MTLSize ntg = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else {
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        }
    }

    // Convert to requested output dtype
    auto output = (dtype == at::kHalf) ? output_fp16 : output_fp16.to(dtype);
    return is_1d ? output.squeeze(0) : output;
}

// =============================================================================
// FP4 Operations
// =============================================================================

std::tuple<at::Tensor, at::Tensor> quantize_fp4_mps(const at::Tensor& input, int64_t block_size) {
    TORCH_CHECK(input.device().is_mps(), "Input must be on MPS");
    TORCH_CHECK(input.dim() == 2 && input.size(1) % 2 == 0, "Invalid input shape");

    const int64_t rows = input.size(0), cols = input.size(1);
    const int64_t num_blocks = (cols + block_size - 1) / block_size;

    auto input_f32 = input.to(at::kFloat).contiguous();
    auto packed = at::empty({rows, cols / 2}, input.options().dtype(at::kByte));
    auto absmax = at::empty({rows, num_blocks}, input.options().dtype(at::kFloat));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp4_quantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_f32) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(packed) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax) offset:0 atIndex:2];

        uint32_t params[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)block_size};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];
        [encoder setBytes:&params[2] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(256, rows, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return std::make_tuple(packed, absmax);
}

at::Tensor dequantize_fp4_mps(const at::Tensor& packed, const at::Tensor& absmax, int64_t block_size, at::ScalarType dtype) {
    const int64_t rows = packed.size(0), cols = packed.size(1) * 2;
    auto output = at::empty({rows, cols}, packed.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp4_dequantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(packed) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax.to(at::kFloat).contiguous()) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:2];

        uint32_t params[3] = {(uint32_t)rows, (uint32_t)cols, (uint32_t)block_size};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];
        [encoder setBytes:&params[2] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((cols + 15) / 16 * 16, (rows + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor matmul_fp4_mps(const at::Tensor& input, const at::Tensor& weight_packed,
                          const at::Tensor& weight_absmax, const std::optional<at::Tensor>& bias,
                          int64_t block_size, at::ScalarType dtype) {
    bool is_1d = input.dim() == 1;
    auto input_2d = is_1d ? input.unsqueeze(0) : input;

    const int64_t M = input_2d.size(0), K = input_2d.size(1), N = weight_packed.size(0);
    // K_weight is the padded K dimension used for weight storage
    const int64_t K_weight = weight_packed.size(1) * 2;

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = weight_absmax.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    // Kernels always output fp16, we convert to requested dtype at the end
    auto output_fp16 = at::empty({M, N}, input.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();

        // Use SIMD kernel for small batches (M <= 128), fallback to simple for larger
        bool use_simd = (M <= 128 && N >= 8 && K >= 8);
        const char* kernel_name = use_simd ? "fp4_matmul_simd" : "fp4_linear_simple";
        auto pipeline = get_pipeline(kernel_name);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output_fp16) offset:0 atIndex:4];

        uint32_t params[6] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)K_weight, (uint32_t)block_size, has_bias ? 1u : 0u};
        for (int i = 0; i < 6; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        if (use_simd) {
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize ntg = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else {
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        }
    }

    // Convert to requested output dtype
    auto output = (dtype == at::kHalf) ? output_fp16 : output_fp16.to(dtype);
    return is_1d ? output.squeeze(0) : output;
}

// =============================================================================
// FP8 Operations
// =============================================================================

std::tuple<at::Tensor, at::Tensor> quantize_fp8_e4m3_mps(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_mps(), "Input must be on MPS");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    const int64_t rows = input.size(0), cols = input.size(1);

    auto input_f32 = input.to(at::kFloat).contiguous();
    auto output = at::empty({rows, cols}, input.options().dtype(at::kByte));
    auto scales = at::empty({rows}, input.options().dtype(at::kFloat));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp8_e4m3_quantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_f32) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales) offset:0 atIndex:2];

        uint32_t params[2] = {(uint32_t)rows, (uint32_t)cols};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(256, rows, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return std::make_tuple(output, scales);
}

at::Tensor dequantize_fp8_e4m3_mps(const at::Tensor& input, const at::Tensor& scales, at::ScalarType dtype) {
    const int64_t rows = input.size(0), cols = input.size(1);
    auto output = at::empty({rows, cols}, input.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp8_e4m3_dequantize");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales.to(at::kFloat).contiguous()) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:2];

        uint32_t params[2] = {(uint32_t)rows, (uint32_t)cols};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((cols + 15) / 16 * 16, (rows + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor matmul_fp8_e4m3_mps(const at::Tensor& input, const at::Tensor& weight,
                               const at::Tensor& weight_scales, const std::optional<at::Tensor>& bias,
                               at::ScalarType dtype) {
    bool is_1d = input.dim() == 1;
    auto input_2d = is_1d ? input.unsqueeze(0) : input;

    const int64_t M = input_2d.size(0), K = input_2d.size(1), N = weight.size(0);

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight.contiguous();
    auto scales_c = weight_scales.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    // Kernels always output fp16, we convert to requested dtype at the end
    auto output_fp16 = at::empty({M, N}, input.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();

        // Use SIMD kernel for small batches (M <= 128), fallback to simple for larger
        bool use_simd = (M <= 128 && N >= 8 && K >= 8);
        const char* kernel_name = use_simd ? "fp8_matmul_simd" : "fp8_e4m3_linear";
        auto pipeline = get_pipeline(kernel_name);

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output_fp16) offset:0 atIndex:4];

        uint32_t params[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, has_bias ? 1u : 0u};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        if (use_simd) {
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize ntg = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else {
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        }
    }

    // Convert to requested output dtype
    auto output = (dtype == at::kHalf) ? output_fp16 : output_fp16.to(dtype);
    return is_1d ? output.squeeze(0) : output;
}

// =============================================================================
// Double Quantization
// =============================================================================

std::tuple<at::Tensor, at::Tensor> double_quant_mps(const at::Tensor& absmax, int64_t double_quant_block) {
    TORCH_CHECK(absmax.device().is_mps(), "Input must be on MPS");
    TORCH_CHECK(absmax.dim() == 2, "Input must be 2D");

    const int64_t rows = absmax.size(0), blocks_per_row = absmax.size(1);
    const int64_t dq_blocks = (blocks_per_row + double_quant_block - 1) / double_quant_block;

    auto absmax_f32 = absmax.to(at::kFloat).contiguous();
    auto absmax_quant = at::empty({rows, blocks_per_row}, absmax.options().dtype(at::kByte));
    auto absmax_scales = at::empty({rows, dq_blocks}, absmax.options().dtype(at::kFloat));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("double_quant_absmax");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_f32) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_quant) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_scales) offset:0 atIndex:2];

        uint32_t params[3] = {(uint32_t)rows, (uint32_t)blocks_per_row, (uint32_t)double_quant_block};
        [encoder setBytes:&params[0] length:4 atIndex:3];
        [encoder setBytes:&params[1] length:4 atIndex:4];
        [encoder setBytes:&params[2] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize grid = MTLSizeMake(256, rows, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return std::make_tuple(absmax_quant, absmax_scales);
}

// =============================================================================
// Embedding Operations
// =============================================================================

at::Tensor embedding_4bit_nf4_mps(
    const at::Tensor& indices,
    const at::Tensor& weight_packed,
    const at::Tensor& absmax,
    int64_t embedding_dim,
    int64_t block_size
) {
    TORCH_CHECK(indices.device().is_mps(), "indices must be on MPS");
    TORCH_CHECK(weight_packed.device().is_mps(), "weight must be on MPS");

    auto indices_c = indices.to(at::kInt).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = absmax.to(at::kFloat).contiguous();

    int64_t num_indices = indices_c.numel();
    int64_t num_blocks = (embedding_dim + block_size - 1) / block_size;

    auto output = at::empty({num_indices, embedding_dim}, weight_packed.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("embedding_4bit_nf4");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(indices_c) offset:indices_c.storage_offset()*4 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:weight_c.storage_offset() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:absmax_c.storage_offset()*4 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:3];

        uint32_t params[4] = {(uint32_t)num_indices, (uint32_t)embedding_dim, (uint32_t)block_size, (uint32_t)num_blocks};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params[i] length:4 atIndex:4 + i];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize ntg = MTLSizeMake(1, num_indices, 1);
        [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor embedding_4bit_fp4_mps(
    const at::Tensor& indices,
    const at::Tensor& weight_packed,
    const at::Tensor& absmax,
    int64_t embedding_dim,
    int64_t block_size
) {
    TORCH_CHECK(indices.device().is_mps(), "indices must be on MPS");

    auto indices_c = indices.to(at::kInt).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = absmax.to(at::kFloat).contiguous();

    int64_t num_indices = indices_c.numel();
    int64_t num_blocks = (embedding_dim + block_size - 1) / block_size;

    auto output = at::empty({num_indices, embedding_dim}, weight_packed.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("embedding_4bit_fp4");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(indices_c) offset:indices_c.storage_offset()*4 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:weight_c.storage_offset() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:absmax_c.storage_offset()*4 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:3];

        uint32_t params[4] = {(uint32_t)num_indices, (uint32_t)embedding_dim, (uint32_t)block_size, (uint32_t)num_blocks};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params[i] length:4 atIndex:4 + i];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize ntg = MTLSizeMake(1, num_indices, 1);
        [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor embedding_8bit_mps(
    const at::Tensor& indices,
    const at::Tensor& weight_int8,
    const at::Tensor& weight_scales
) {
    TORCH_CHECK(indices.device().is_mps(), "indices must be on MPS");

    auto indices_c = indices.to(at::kInt).contiguous();
    auto weight_c = weight_int8.contiguous();
    auto scales_c = weight_scales.to(at::kFloat).contiguous();

    int64_t num_indices = indices_c.numel();
    int64_t embedding_dim = weight_int8.size(1);

    auto output = at::empty({num_indices, embedding_dim}, weight_int8.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("embedding_8bit");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(indices_c) offset:indices_c.storage_offset()*4 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:weight_c.storage_offset() atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales_c) offset:scales_c.storage_offset()*4 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:3];

        uint32_t params[2] = {(uint32_t)num_indices, (uint32_t)embedding_dim};
        [encoder setBytes:&params[0] length:4 atIndex:4];
        [encoder setBytes:&params[1] length:4 atIndex:5];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((embedding_dim + 15) / 16) * 16, ((num_indices + 15) / 16) * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

// =============================================================================
// 8-bit Optimizer Operations
// =============================================================================

void adam8bit_step_mps(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& exp_avg_int8,
    at::Tensor& exp_avg_absmax,
    at::Tensor& exp_avg_sq_u8,
    at::Tensor& exp_avg_sq_max,
    double beta1, double beta2, double eps, double lr,
    double weight_decay, int64_t step, int64_t block_size,
    bool is_adamw
) {
    TORCH_CHECK(param.device().is_mps(), "param must be on MPS");

    auto param_c = param.contiguous();
    auto grad_c = grad.to(at::kHalf).contiguous();
    auto ea_int8 = exp_avg_int8.contiguous();
    auto ea_absmax = exp_avg_absmax.to(at::kFloat).contiguous();
    auto ea_sq_u8 = exp_avg_sq_u8.contiguous();
    auto ea_sq_max = exp_avg_sq_max.to(at::kFloat).contiguous();

    int64_t N = param_c.numel();
    int64_t num_blocks = (N + block_size - 1) / block_size;

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("adam8bit_step");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(param_c) offset:param_c.storage_offset()*2 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_c) offset:grad_c.storage_offset()*2 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_int8) offset:ea_int8.storage_offset() atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_absmax) offset:ea_absmax.storage_offset()*4 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_sq_u8) offset:ea_sq_u8.storage_offset() atIndex:4];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_sq_max) offset:ea_sq_max.storage_offset()*4 atIndex:5];

        float params_f[5] = {(float)beta1, (float)beta2, (float)eps, (float)lr, (float)weight_decay};
        for (int i = 0; i < 5; i++) [encoder setBytes:&params_f[i] length:4 atIndex:6 + i];

        uint32_t params_u[4] = {(uint32_t)step, (uint32_t)N, (uint32_t)block_size, is_adamw ? 1u : 0u};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params_u[i] length:4 atIndex:11 + i];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize ntg = MTLSizeMake(num_blocks, 1, 1);
        [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
    }

    // Copy results back if param was not contiguous
    if (!param.is_contiguous()) {
        param.copy_(param_c);
    }
    exp_avg_int8.copy_(ea_int8);
    exp_avg_absmax.copy_(ea_absmax);
    exp_avg_sq_u8.copy_(ea_sq_u8);
    exp_avg_sq_max.copy_(ea_sq_max);
}

void lion8bit_step_mps(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& exp_avg_int8,
    at::Tensor& exp_avg_absmax,
    double beta1, double beta2, double lr,
    double weight_decay, int64_t block_size
) {
    TORCH_CHECK(param.device().is_mps(), "param must be on MPS");

    auto param_c = param.contiguous();
    auto grad_c = grad.to(at::kHalf).contiguous();
    auto ea_int8 = exp_avg_int8.contiguous();
    auto ea_absmax = exp_avg_absmax.to(at::kFloat).contiguous();

    int64_t N = param_c.numel();
    int64_t num_blocks = (N + block_size - 1) / block_size;

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("lion8bit_step");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(param_c) offset:param_c.storage_offset()*2 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_c) offset:grad_c.storage_offset()*2 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_int8) offset:ea_int8.storage_offset() atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(ea_absmax) offset:ea_absmax.storage_offset()*4 atIndex:3];

        float params_f[4] = {(float)beta1, (float)beta2, (float)lr, (float)weight_decay};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params_f[i] length:4 atIndex:4 + i];

        uint32_t params_u[2] = {(uint32_t)N, (uint32_t)block_size};
        [encoder setBytes:&params_u[0] length:4 atIndex:8];
        [encoder setBytes:&params_u[1] length:4 atIndex:9];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize ntg = MTLSizeMake(num_blocks, 1, 1);
        [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
    }

    if (!param.is_contiguous()) param.copy_(param_c);
    exp_avg_int8.copy_(ea_int8);
    exp_avg_absmax.copy_(ea_absmax);
}

void sgd8bit_step_mps(
    at::Tensor& param,
    const at::Tensor& grad,
    at::Tensor& momentum_int8,
    at::Tensor& momentum_absmax,
    double lr, double momentum, double dampening,
    double weight_decay, bool nesterov, int64_t block_size
) {
    TORCH_CHECK(param.device().is_mps(), "param must be on MPS");

    auto param_c = param.contiguous();
    auto grad_c = grad.to(at::kHalf).contiguous();
    auto m_int8 = momentum_int8.contiguous();
    auto m_absmax = momentum_absmax.to(at::kFloat).contiguous();

    int64_t N = param_c.numel();
    int64_t num_blocks = (N + block_size - 1) / block_size;

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("sgd8bit_step");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(param_c) offset:param_c.storage_offset()*2 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(grad_c) offset:grad_c.storage_offset()*2 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(m_int8) offset:m_int8.storage_offset() atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(m_absmax) offset:m_absmax.storage_offset()*4 atIndex:3];

        float params_f[4] = {(float)lr, (float)momentum, (float)dampening, (float)weight_decay};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params_f[i] length:4 atIndex:4 + i];

        uint32_t params_u[3] = {nesterov ? 1u : 0u, (uint32_t)N, (uint32_t)block_size};
        for (int i = 0; i < 3; i++) [encoder setBytes:&params_u[i] length:4 atIndex:8 + i];

        MTLSize tg = MTLSizeMake(256, 1, 1);
        MTLSize ntg = MTLSizeMake(num_blocks, 1, 1);
        [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
    }

    if (!param.is_contiguous()) param.copy_(param_c);
    momentum_int8.copy_(m_int8);
    momentum_absmax.copy_(m_absmax);
}

// =============================================================================
// Sparse MatMul Operations
// =============================================================================

at::Tensor spmm_coo_mps(
    const at::Tensor& row_indices,
    const at::Tensor& col_indices,
    const at::Tensor& values,
    const at::Tensor& dense,
    int64_t sparse_rows,
    int64_t sparse_cols
) {
    TORCH_CHECK(dense.device().is_mps(), "dense must be on MPS");

    int64_t M = sparse_rows;
    int64_t N = dense.size(1);
    int64_t K = sparse_cols;
    int64_t nnz = values.numel();

    auto values_c = values.to(at::kFloat).contiguous();
    auto dense_c = dense.to(at::kHalf).contiguous();

    // Convert COO to CSR on CPU (cheap)
    auto row_idx_cpu = row_indices.to(at::kLong).cpu();
    auto col_idx_cpu = col_indices.to(at::kInt).cpu();

    // Sort by row
    auto sort_indices = row_idx_cpu.argsort();
    auto sorted_rows = row_idx_cpu.index_select(0, sort_indices);
    auto sorted_cols = col_idx_cpu.index_select(0, sort_indices);
    auto sorted_vals_cpu = values_c.cpu().index_select(0, sort_indices.to(values_c.device()));

    // Build row_ptr
    auto row_ptr_cpu = at::zeros({M + 1}, at::kInt);
    auto row_ptr_acc = row_ptr_cpu.accessor<int, 1>();
    auto sorted_rows_acc = sorted_rows.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nnz; i++) {
        row_ptr_acc[sorted_rows_acc[i] + 1]++;
    }
    for (int64_t i = 1; i <= M; i++) {
        row_ptr_acc[i] += row_ptr_acc[i - 1];
    }

    // Move to MPS
    auto row_ptr_mps = row_ptr_cpu.to(at::kInt).to(dense.device());
    auto col_idx_mps = sorted_cols.to(at::kInt).to(dense.device());
    auto vals_mps = sorted_vals_cpu.to(at::kFloat).to(dense.device());

    auto output = at::zeros({M, N}, dense.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("spmm_csr");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(row_ptr_mps) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(col_idx_mps) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(vals_mps) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(dense_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:4];

        uint32_t params[3] = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
        for (int i = 0; i < 3; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((N + 15) / 16) * 16, ((M + 15) / 16) * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

at::Tensor spmm_coo_int8_mps(
    const at::Tensor& row_indices,
    const at::Tensor& col_indices,
    const at::Tensor& values_int8,
    const at::Tensor& values_scale,
    const at::Tensor& dense,
    int64_t sparse_rows,
    int64_t sparse_cols
) {
    TORCH_CHECK(dense.device().is_mps(), "dense must be on MPS");

    int64_t M = sparse_rows;
    int64_t N = dense.size(1);
    int64_t K = sparse_cols;
    int64_t nnz = values_int8.numel();

    auto dense_c = dense.to(at::kHalf).contiguous();
    auto scale_c = values_scale.to(at::kFloat).contiguous();

    // Convert COO to CSR on CPU
    auto row_idx_cpu = row_indices.to(at::kLong).cpu();
    auto col_idx_cpu = col_indices.to(at::kInt).cpu();

    auto sort_indices = row_idx_cpu.argsort();
    auto sorted_rows = row_idx_cpu.index_select(0, sort_indices);
    auto sorted_cols = col_idx_cpu.index_select(0, sort_indices);
    auto sorted_vals_cpu = values_int8.cpu().index_select(0, sort_indices.to(values_int8.device()));

    auto row_ptr_cpu = at::zeros({M + 1}, at::kInt);
    auto row_ptr_acc = row_ptr_cpu.accessor<int, 1>();
    auto sorted_rows_acc = sorted_rows.accessor<int64_t, 1>();
    for (int64_t i = 0; i < nnz; i++) {
        row_ptr_acc[sorted_rows_acc[i] + 1]++;
    }
    for (int64_t i = 1; i <= M; i++) {
        row_ptr_acc[i] += row_ptr_acc[i - 1];
    }

    auto row_ptr_mps = row_ptr_cpu.to(at::kInt).to(dense.device());
    auto col_idx_mps = sorted_cols.to(at::kInt).to(dense.device());
    auto vals_mps = sorted_vals_cpu.to(at::kChar).to(dense.device());

    auto output = at::zeros({M, N}, dense.options().dtype(at::kHalf));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("spmm_csr_int8");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(row_ptr_mps) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(col_idx_mps) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(vals_mps) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scale_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(dense_c) offset:0 atIndex:4];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:5];

        uint32_t params[3] = {(uint32_t)M, (uint32_t)N, (uint32_t)K};
        for (int i = 0; i < 3; i++) [encoder setBytes:&params[i] length:4 atIndex:6 + i];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake(((N + 15) / 16) * 16, ((M + 15) / 16) * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

    return output;
}

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "MPS BitsAndBytes - INT8, NF4, FP4, FP8 quantization for Apple Silicon";

    // INT8
    m.def("matmul_int8", &matmul_int8_mps, "INT8 matmul",
          py::arg("A"), py::arg("B"), py::arg("A_scales"), py::arg("B_scales"),
          py::arg("out_dtype") = at::kHalf);
    m.def("linear_int8", &linear_int8_mps, "INT8 linear layer (half input, int8 weight)",
          py::arg("input"), py::arg("weight"), py::arg("weight_scales"),
          py::arg("bias") = py::none(), py::arg("out_dtype") = at::kHalf);

    // NF4
    m.def("quantize_nf4", &quantize_nf4_mps, py::arg("input"), py::arg("block_size") = 64);
    m.def("dequantize_nf4", &dequantize_nf4_mps,
          py::arg("packed"), py::arg("absmax"), py::arg("block_size") = 64, py::arg("out_dtype") = at::kHalf);
    m.def("matmul_nf4", &matmul_nf4_mps,
          py::arg("input"), py::arg("weight_packed"), py::arg("weight_absmax"),
          py::arg("bias") = py::none(), py::arg("block_size") = 64, py::arg("out_dtype") = at::kHalf);

    // FP4
    m.def("quantize_fp4", &quantize_fp4_mps, py::arg("input"), py::arg("block_size") = 64);
    m.def("dequantize_fp4", &dequantize_fp4_mps,
          py::arg("packed"), py::arg("absmax"), py::arg("block_size") = 64, py::arg("out_dtype") = at::kHalf);
    m.def("matmul_fp4", &matmul_fp4_mps,
          py::arg("input"), py::arg("weight_packed"), py::arg("weight_absmax"),
          py::arg("bias") = py::none(), py::arg("block_size") = 64, py::arg("out_dtype") = at::kHalf);

    // FP8
    m.def("quantize_fp8_e4m3", &quantize_fp8_e4m3_mps, py::arg("input"));
    m.def("dequantize_fp8_e4m3", &dequantize_fp8_e4m3_mps,
          py::arg("input"), py::arg("scales"), py::arg("out_dtype") = at::kHalf);
    m.def("matmul_fp8_e4m3", &matmul_fp8_e4m3_mps,
          py::arg("input"), py::arg("weight"), py::arg("weight_scales"),
          py::arg("bias") = py::none(), py::arg("out_dtype") = at::kHalf);

    // Double Quantization
    m.def("double_quant", &double_quant_mps, py::arg("absmax"), py::arg("double_quant_block") = 256);

    // Embedding
    m.def("embedding_4bit_nf4", &embedding_4bit_nf4_mps,
          py::arg("indices"), py::arg("weight_packed"), py::arg("absmax"),
          py::arg("embedding_dim"), py::arg("block_size"));
    m.def("embedding_4bit_fp4", &embedding_4bit_fp4_mps,
          py::arg("indices"), py::arg("weight_packed"), py::arg("absmax"),
          py::arg("embedding_dim"), py::arg("block_size"));
    m.def("embedding_8bit", &embedding_8bit_mps,
          py::arg("indices"), py::arg("weight_int8"), py::arg("weight_scales"));

    // 8-bit Optimizers
    m.def("adam8bit_step", &adam8bit_step_mps,
          py::arg("param"), py::arg("grad"),
          py::arg("exp_avg_int8"), py::arg("exp_avg_absmax"),
          py::arg("exp_avg_sq_u8"), py::arg("exp_avg_sq_max"),
          py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("lr"),
          py::arg("weight_decay"), py::arg("step"), py::arg("block_size"),
          py::arg("is_adamw"));
    m.def("lion8bit_step", &lion8bit_step_mps,
          py::arg("param"), py::arg("grad"),
          py::arg("exp_avg_int8"), py::arg("exp_avg_absmax"),
          py::arg("beta1"), py::arg("beta2"), py::arg("lr"),
          py::arg("weight_decay"), py::arg("block_size"));
    m.def("sgd8bit_step", &sgd8bit_step_mps,
          py::arg("param"), py::arg("grad"),
          py::arg("momentum_int8"), py::arg("momentum_absmax"),
          py::arg("lr"), py::arg("momentum"), py::arg("dampening"),
          py::arg("weight_decay"), py::arg("nesterov"), py::arg("block_size"));

    // Sparse MatMul
    m.def("spmm_coo", &spmm_coo_mps,
          py::arg("row_indices"), py::arg("col_indices"), py::arg("values"),
          py::arg("dense"), py::arg("sparse_rows"), py::arg("sparse_cols"));
    m.def("spmm_coo_int8", &spmm_coo_int8_mps,
          py::arg("row_indices"), py::arg("col_indices"), py::arg("values_int8"),
          py::arg("values_scale"), py::arg("dense"), py::arg("sparse_rows"), py::arg("sparse_cols"));
}
