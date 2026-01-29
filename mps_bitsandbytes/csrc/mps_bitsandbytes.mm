/**
 * MPS BitsAndBytes - PyTorch C++ Extension
 *
 * Int8 quantization and matmul kernels for Apple Silicon.
 *
 * NOTE: Int8 matmul on MPS is primarily for MEMORY savings, not speed.
 * Apple's fp16 matmul uses the AMX coprocessor which is extremely fast.
 * The benefit of int8 is storing weights at 1/2 the size.
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// =============================================================================
// Metal Kernel Cache
// =============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_matmul_dequant_pipeline = nil;

static bool init_metal() {
    if (g_device) return true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("Failed to create Metal device");
        }

        // Simple but correct int8 matmul kernel
        // 16x16 tiles, direct computation
        NSString* source = @R"(
            #include <metal_stdlib>
            using namespace metal;

            constant uint TILE_SIZE = 16;

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
                threadgroup char As[TILE_SIZE][TILE_SIZE];
                threadgroup char Bs[TILE_SIZE][TILE_SIZE];

                uint row = tgid.y * TILE_SIZE + tid.y;
                uint col = tgid.x * TILE_SIZE + tid.x;

                int acc = 0;

                for (uint t = 0; t < K; t += TILE_SIZE) {
                    uint a_col = t + tid.x;
                    uint b_row = t + tid.y;

                    As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0;
                    Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0;

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (uint k = 0; k < TILE_SIZE; k++) {
                        acc += int(As[tid.y][k]) * int(Bs[k][tid.x]);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                if (row < M && col < N) {
                    float a_scale = A_scales[row];
                    float b_scale = B_scales[col];
                    float scale = (a_scale * b_scale) / (127.0f * 127.0f);
                    C[row * N + col] = half(float(acc) * scale);
                }
            }
        )";

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        g_library = [g_device newLibraryWithSource:source options:options error:&error];

        if (!g_library) {
            throw std::runtime_error("Failed to create Metal library: " +
                                     std::string([[error localizedDescription] UTF8String]));
        }

        id<MTLFunction> fn = [g_library newFunctionWithName:@"int8_matmul_dequant"];
        if (!fn) {
            throw std::runtime_error("Failed to find int8_matmul_dequant function");
        }

        g_matmul_dequant_pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
        if (!g_matmul_dequant_pipeline) {
            throw std::runtime_error("Failed to create pipeline: " +
                                     std::string([[error localizedDescription] UTF8String]));
        }
    }

    return true;
}

// =============================================================================
// Int8 MatMul with Fused Dequantization
// =============================================================================

at::Tensor matmul_int8_mps(
    const at::Tensor& A,        // [M, K] int8
    const at::Tensor& B,        // [K, N] int8
    const at::Tensor& A_scales, // [M] float
    const at::Tensor& B_scales, // [N] float
    at::ScalarType out_dtype
) {
    TORCH_CHECK(A.device().is_mps(), "A must be on MPS");
    TORCH_CHECK(B.device().is_mps(), "B must be on MPS");
    TORCH_CHECK(A.dtype() == at::kChar, "A must be int8");
    TORCH_CHECK(B.dtype() == at::kChar, "B must be int8");

    init_metal();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);

    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A is [M,K], B must be [K,N]");

    // Ensure scales are fp32 and contiguous
    auto A_scales_f32 = A_scales.to(at::kFloat).contiguous();
    auto B_scales_f32 = B_scales.to(at::kFloat).contiguous();

    // Make inputs contiguous
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    // Allocate output
    auto output = at::empty({M, N}, A.options().dtype(out_dtype));

    // Use MPS stream with PyTorch's shared encoder (zero-sync)
    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // Get buffers
        id<MTLBuffer> A_buf = at::native::mps::getMTLBufferStorage(A_contig);
        id<MTLBuffer> B_buf = at::native::mps::getMTLBufferStorage(B_contig);
        id<MTLBuffer> out_buf = at::native::mps::getMTLBufferStorage(output);
        id<MTLBuffer> A_scales_buf = at::native::mps::getMTLBufferStorage(A_scales_f32);
        id<MTLBuffer> B_scales_buf = at::native::mps::getMTLBufferStorage(B_scales_f32);

        // Set pipeline and buffers
        [encoder setComputePipelineState:g_matmul_dequant_pipeline];
        [encoder setBuffer:A_buf offset:0 atIndex:0];
        [encoder setBuffer:B_buf offset:0 atIndex:1];
        [encoder setBuffer:out_buf offset:0 atIndex:2];
        [encoder setBuffer:A_scales_buf offset:0 atIndex:3];
        [encoder setBuffer:B_scales_buf offset:0 atIndex:4];

        uint32_t M_val = static_cast<uint32_t>(M);
        uint32_t N_val = static_cast<uint32_t>(N);
        uint32_t K_val = static_cast<uint32_t>(K);
        [encoder setBytes:&M_val length:sizeof(uint32_t) atIndex:5];
        [encoder setBytes:&N_val length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&K_val length:sizeof(uint32_t) atIndex:7];

        // Dispatch
        const uint32_t TILE_SIZE = 16;
        MTLSize threadgroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
        MTLSize gridSize = MTLSizeMake(
            (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            (M + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
            1
        );
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

        // Don't endEncoding/commit - PyTorch manages encoder lifecycle
    }

    return output;
}

// =============================================================================
// Python Bindings
// =============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "MPS BitsAndBytes - 8-bit quantization for Apple Silicon";

    m.def("matmul_int8", &matmul_int8_mps,
          "Int8 matrix multiplication with fused dequantization",
          py::arg("A"),
          py::arg("B"),
          py::arg("A_scales"),
          py::arg("B_scales"),
          py::arg("out_dtype") = at::kHalf);
}
