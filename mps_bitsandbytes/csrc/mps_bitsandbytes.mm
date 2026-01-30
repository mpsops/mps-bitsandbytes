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

// =============================================================================
// Metal Device and Libraries
// =============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

static void ensure_device() {
    if (!g_device) {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("Failed to create Metal device");
        }
    }
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
        C[row * N + col] = half(float(acc) * scale);
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
    constant uint& block_size [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y, n = gid.x;
    if (m >= M || n >= N) return;

    uint num_blocks = (K + block_size - 1) / block_size;
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        float scale = absmax[n * num_blocks + k / block_size];
        float w_val = dequant_nf4(weight[n * (K/2) + k/2], k % 2, scale);
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    output[m * N + n] = half(acc);
}

constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint TILE_K = 64;

kernel void nf4_matmul_fused(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const float* absmax [[buffer(2)]],
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
    threadgroup half As[TILE_M][TILE_K + 1];
    threadgroup half Bs[TILE_K][TILE_N + 1];

    uint tx = tid.x, ty = tid.y;
    uint thread_id = ty * 8 + tx;
    uint row_base = tgid.y * TILE_M + ty * 4;
    uint col_base = tgid.x * TILE_N + tx * 4;

    float acc[4][4] = {{0.0f}};
    uint num_blocks = (K + block_size - 1) / block_size;

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
                Bs[lk][ln] = half(dequant_nf4(weight[gn * (K/2) + gk/2], gk % 2, scale));
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
    constant uint& block_size [[buffer(8)]],
    constant uint& has_bias [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint m = gid.y, n = gid.x;
    if (m >= M || n >= N) return;

    uint num_blocks = (K + block_size - 1) / block_size;
    float acc = 0.0f;

    for (uint k = 0; k < K; k++) {
        float in_val = float(input[m * K + k]);
        float scale = absmax[n * num_blocks + k / block_size];
        float w_val = dequant_fp4(weight[n * (K/2) + k/2], k % 2, scale);
        acc += in_val * w_val;
    }

    if (has_bias) acc += float(bias[n]);
    output[m * N + n] = half(acc);
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
    output[m * N + n] = half(acc);
}

// =============================================================================
// Double Quantization (quantize absmax with INT8)
// =============================================================================

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
// Initialize Metal Library
// =============================================================================

static void init_library() {
    if (g_library) return;

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
}

static id<MTLComputePipelineState> get_pipeline(const std::string& name) {
    init_library();

    auto it = g_pipelines.find(name);
    if (it != g_pipelines.end()) {
        return it->second;
    }

    @autoreleasepool {
        NSError* error = nil;
        id<MTLFunction> fn = [g_library newFunctionWithName:
            [NSString stringWithUTF8String:name.c_str()]];
        if (!fn) {
            throw std::runtime_error("Function not found: " + name);
        }

        id<MTLComputePipelineState> pipeline =
            [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            throw std::runtime_error("Failed to create pipeline: " + name);
        }

        g_pipelines[name] = pipeline;
        return pipeline;
    }
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
    auto output = at::empty({M, N}, A.options().dtype(out_dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("int8_matmul_dequant");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(A_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(B_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:2];
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

    return output;
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
    TORCH_CHECK(weight_packed.size(1) * 2 == K, "Weight K mismatch");

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = weight_absmax.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    auto output = at::empty({M, N}, input.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();

        bool use_tiled = (M >= 32 && N >= 32 && K >= 64);
        auto pipeline = get_pipeline(use_tiled ? "nf4_matmul_fused" : "nf4_linear_simple");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:4];

        uint32_t params[5] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)block_size, has_bias ? 1u : 0u};
        for (int i = 0; i < 5; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        if (use_tiled) {
            MTLSize tg = MTLSizeMake(8, 8, 1);
            MTLSize ntg = MTLSizeMake((N + 31) / 32, (M + 31) / 32, 1);
            [encoder dispatchThreadgroups:ntg threadsPerThreadgroup:tg];
        } else {
            MTLSize tg = MTLSizeMake(16, 16, 1);
            MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        }
    }

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

    auto input_c = input_2d.to(at::kHalf).contiguous();
    auto weight_c = weight_packed.contiguous();
    auto absmax_c = weight_absmax.to(at::kFloat).contiguous();

    bool has_bias = bias.has_value();
    auto bias_c = has_bias ? bias.value().to(at::kHalf).contiguous() : at::empty({1}, input.options().dtype(at::kHalf));
    auto output = at::empty({M, N}, input.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp4_linear_simple");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(absmax_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:4];

        uint32_t params[5] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, (uint32_t)block_size, has_bias ? 1u : 0u};
        for (int i = 0; i < 5; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

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
    auto output = at::empty({M, N}, input.options().dtype(dtype));

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto encoder = stream->commandEncoder();
        auto pipeline = get_pipeline("fp8_e4m3_linear");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(input_c) offset:0 atIndex:0];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(weight_c) offset:0 atIndex:1];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(scales_c) offset:0 atIndex:2];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(bias_c) offset:0 atIndex:3];
        [encoder setBuffer:at::native::mps::getMTLBufferStorage(output) offset:0 atIndex:4];

        uint32_t params[4] = {(uint32_t)M, (uint32_t)N, (uint32_t)K, has_bias ? 1u : 0u};
        for (int i = 0; i < 4; i++) [encoder setBytes:&params[i] length:4 atIndex:5 + i];

        MTLSize tg = MTLSizeMake(16, 16, 1);
        MTLSize grid = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    }

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
// Python Bindings
// =============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "MPS BitsAndBytes - INT8, NF4, FP4, FP8 quantization for Apple Silicon";

    // INT8
    m.def("matmul_int8", &matmul_int8_mps, "INT8 matmul",
          py::arg("A"), py::arg("B"), py::arg("A_scales"), py::arg("B_scales"),
          py::arg("out_dtype") = at::kHalf);

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
}
