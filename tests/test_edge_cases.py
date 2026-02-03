"""
Edge case tests for mps-bitsandbytes.

These tests expose potential bugs similar to those found in mps-flash-attention:
1. Bias dtype mismatches
2. Division by zero in scale calculations
3. Buffer bounds with unusual shapes
4. Stress tests for numerical stability
"""

import pytest
import torch
import math

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


@pytest.fixture(scope="module")
def bnb():
    """Import mps_bitsandbytes module."""
    import mps_bitsandbytes as bnb
    return bnb


# =============================================================================
# Bias Dtype Mismatch Tests
# =============================================================================

class TestBiasDtype:
    """Test bias handling with different dtypes.

    The Metal kernels declare bias as `half*` but callers might pass float32.
    This mirrors the FP16/FP32 mismatch bug found in mps-flash-attention.
    """

    def test_matmul_bias_fp32(self, bnb):
        """Test matmul_4bit with FP32 bias - potential dtype mismatch."""
        M, N, K = 32, 64, 128

        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)
        bias = torch.randn(N, device='mps', dtype=torch.float32)  # FP32 bias!

        weight_packed, weight_state = bnb.functional.quantize_4bit(weight, blocksize=64)

        # This might produce wrong results if Metal kernel expects half
        output = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=bias)

        assert not torch.isnan(output).any(), "FP32 bias produced NaN"
        assert not torch.isinf(output).any(), "FP32 bias produced Inf"

        # Verify bias actually affected output (not ignored)
        output_no_bias = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=None)
        diff = (output - output_no_bias).abs().max().item()

        # Bias should make a difference
        assert diff > 0.01, f"FP32 bias might be ignored (diff={diff})"

    def test_matmul_bias_bf16(self, bnb):
        """Test matmul_4bit with BF16 bias - potential dtype mismatch."""
        M, N, K = 32, 64, 128

        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)
        bias = torch.randn(N, device='mps', dtype=torch.bfloat16)  # BF16 bias!

        weight_packed, weight_state = bnb.functional.quantize_4bit(weight, blocksize=64)

        output = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=bias)

        assert not torch.isnan(output).any(), "BF16 bias produced NaN"
        assert not torch.isinf(output).any(), "BF16 bias produced Inf"

    def test_matmul_bias_not_ignored(self, bnb):
        """Critical test: Ensure bias is actually applied, not silently ignored."""
        M, N, K = 32, 64, 128

        torch.manual_seed(42)
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)

        weight_packed, weight_state = bnb.functional.quantize_4bit(weight, blocksize=64)

        # Zero bias should match no-bias
        zero_bias = torch.zeros(N, device='mps', dtype=torch.float16)
        output_zero_bias = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=zero_bias)
        output_no_bias = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=None)

        zero_diff = (output_zero_bias - output_no_bias).abs().max().item()
        assert zero_diff < 0.01, f"Zero bias should match no-bias, got diff {zero_diff}"

        # Large bias MUST produce different output
        large_bias = torch.ones(N, device='mps', dtype=torch.float16) * 10.0
        output_large_bias = bnb.functional.matmul_4bit(input_tensor, weight_packed, weight_state, bias=large_bias)

        large_diff = (output_large_bias - output_no_bias).abs().max().item()
        assert large_diff > 1.0, f"Bias appears to be ignored! Diff with large bias: {large_diff}"


# =============================================================================
# Division by Zero / Zero Input Tests
# =============================================================================

class TestZeroInputs:
    """Test handling of zero/near-zero inputs.

    The scale calculation `127.0 / absmax` can produce INF if absmax is 0.
    The code uses clamp(min=1e-8) but we test edge cases.
    """

    def test_quantize_all_zeros(self, bnb):
        """Quantizing all-zero tensor should not produce NaN/Inf."""
        x = torch.zeros(256, 256, device='mps', dtype=torch.float16)

        quantized, state = bnb.functional.quantize_4bit(x)

        assert not torch.isnan(quantized).any(), "Zero input produced NaN in quantized"
        assert not torch.isinf(state.absmax).any(), "Zero input produced Inf in absmax"

        # Dequantize should give back zeros (or near-zero)
        dequantized = bnb.functional.dequantize_4bit(quantized, state)
        assert dequantized.abs().max() < 0.1, "Zero input didn't dequantize to near-zero"

    def test_quantize_single_nonzero(self, bnb):
        """Tensor with single non-zero value per block."""
        blocksize = 64
        x = torch.zeros(256, 256, device='mps', dtype=torch.float16)
        # Put single non-zero in each block
        x[::blocksize, 0] = 1.0

        quantized, state = bnb.functional.quantize_4bit(x, blocksize=blocksize)

        assert not torch.isnan(quantized).any(), "Single nonzero produced NaN"
        assert not torch.isinf(state.absmax).any(), "Single nonzero produced Inf in absmax"

    def test_quantize_near_zero(self, bnb):
        """Tensor with very small values (denormals)."""
        x = torch.full((128, 128), 1e-38, device='mps', dtype=torch.float32)

        quantized, state = bnb.functional.quantize_4bit(x)

        assert not torch.isnan(quantized).any(), "Denormal input produced NaN"
        assert not torch.isinf(state.absmax).any(), "Denormal input produced Inf"

    def test_int8_quantize_zeros(self, bnb):
        """INT8 quantization with zero input."""
        x = torch.zeros(128, 128, device='mps', dtype=torch.float16)

        quantized, state = bnb.functional.quantize_blockwise(x, blocksize=64)

        assert not torch.isnan(quantized.float()).any(), "INT8 zero input produced NaN"


# =============================================================================
# Large Value / Overflow Tests
# =============================================================================

class TestLargeValues:
    """Test handling of very large values that might overflow."""

    def test_quantize_fp16_max(self, bnb):
        """Quantizing FP16 max values."""
        x = torch.full((64, 64), 65504.0, device='mps', dtype=torch.float16)  # FP16 max

        quantized, state = bnb.functional.quantize_4bit(x)

        assert not torch.isnan(quantized).any(), "FP16 max produced NaN"
        assert not torch.isinf(state.absmax).any(), "FP16 max produced Inf in absmax"

    def test_quantize_mixed_extreme(self, bnb):
        """Mix of very large and very small values."""
        x = torch.zeros(128, 128, device='mps', dtype=torch.float16)
        x[0, 0] = 65504.0   # FP16 max
        x[1, 1] = 1e-4      # Very small
        x[2, 2] = -65504.0  # FP16 min

        quantized, state = bnb.functional.quantize_4bit(x)

        assert not torch.isnan(quantized).any(), "Mixed extreme produced NaN"

        dequantized = bnb.functional.dequantize_4bit(quantized, state)
        assert not torch.isnan(dequantized).any(), "Mixed extreme dequant produced NaN"


# =============================================================================
# Unusual Shape Tests
# =============================================================================

class TestUnusualShapes:
    """Test with shapes that might trigger edge cases in padding/blocking."""

    @pytest.mark.parametrize("shape", [
        (1, 1),        # Minimum
        (1, 63),       # Not divisible by 64
        (1, 65),       # Just over 64
        (7, 13),       # Prime dimensions
        (128, 127),    # Off-by-one from power of 2
        (1, 1024),     # Very wide
        (1024, 1),     # Very tall
        (3, 17),       # Small primes
    ])
    def test_quantize_unusual_shapes(self, bnb, shape):
        """Test quantization with unusual shapes."""
        x = torch.randn(*shape, device='mps', dtype=torch.float16)

        quantized, state = bnb.functional.quantize_4bit(x, blocksize=64)

        assert not torch.isnan(quantized).any(), f"Shape {shape} produced NaN"

        dequantized = bnb.functional.dequantize_4bit(quantized, state)
        assert dequantized.shape == shape, f"Shape mismatch: {dequantized.shape} vs {shape}"

    @pytest.mark.parametrize("blocksize", [32, 64, 128, 256, 512, 1024])
    def test_various_blocksizes(self, bnb, blocksize):
        """Test different block sizes."""
        # Shape smaller than blocksize
        x = torch.randn(16, 16, device='mps', dtype=torch.float16)

        quantized, state = bnb.functional.quantize_4bit(x, blocksize=blocksize)

        assert not torch.isnan(quantized).any(), f"Blocksize {blocksize} produced NaN"


# =============================================================================
# Matmul Stress Tests
# =============================================================================

class TestMatmulStress:
    """Stress tests for matmul operations."""

    def test_matmul_repeated_no_nan(self, bnb):
        """Repeated matmul should not accumulate errors into NaN."""
        M, N, K = 32, 128, 256

        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        bias = torch.randn(N, device='mps', dtype=torch.float16)
        weight_packed, weight_state = bnb.functional.quantize_4bit(weight, blocksize=64)

        nan_count = 0
        for i in range(50):
            torch.manual_seed(i)
            x = torch.randn(M, K, device='mps', dtype=torch.float16)

            output = bnb.functional.matmul_4bit(x, weight_packed, weight_state, bias=bias)

            if torch.isnan(output).any():
                nan_count += 1

        assert nan_count == 0, f"Matmul stress test: {nan_count}/50 had NaN"

    def test_matmul_various_sizes_no_nan(self, bnb):
        """Test matmul with various sizes that might trigger edge cases."""
        test_cases = [
            (1, 64, 128),     # Single row
            (7, 64, 128),     # Prime M
            (32, 63, 127),    # Non-power-of-2 N, K
            (32, 64, 65),     # K just over 64
            (128, 256, 512),  # Large
        ]

        nan_count = 0
        for M, N, K in test_cases:
            weight = torch.randn(N, K, device='mps', dtype=torch.float16)
            x = torch.randn(M, K, device='mps', dtype=torch.float16)
            bias = torch.randn(N, device='mps', dtype=torch.float16)

            weight_packed, weight_state = bnb.functional.quantize_4bit(weight, blocksize=64)
            output = bnb.functional.matmul_4bit(x, weight_packed, weight_state, bias=bias)

            if torch.isnan(output).any():
                nan_count += 1
                print(f"NaN at shape M={M}, N={N}, K={K}")

        assert nan_count == 0, f"Various sizes test: {nan_count}/{len(test_cases)} had NaN"


# =============================================================================
# Scale Index Bounds Tests
# =============================================================================

class TestScaleBounds:
    """Test that scale indexing doesn't go out of bounds."""

    def test_mismatched_weight_absmax_shape(self, bnb):
        """Test behavior when weight and absmax shapes might mismatch."""
        in_features, out_features = 128, 64
        blocksize = 64

        # Manually create a weight
        weight = torch.randn(out_features, in_features, device='mps', dtype=torch.float16)

        # Quantize
        quantized, state = bnb.functional.quantize_4bit(weight, blocksize=blocksize)

        # Verify shapes are consistent
        K_padded = ((in_features + blocksize - 1) // blocksize) * blocksize
        if K_padded % 2 != 0:
            K_padded += blocksize

        num_blocks = out_features * (K_padded // blocksize)

        assert state.absmax.numel() == num_blocks, \
            f"Absmax size mismatch: {state.absmax.numel()} vs expected {num_blocks}"


# =============================================================================
# Memory Corruption Detection Tests
# =============================================================================

class TestMemoryCorruption:
    """Tests that might detect memory corruption from buffer overflows."""

    def test_quantize_dequantize_roundtrip(self, bnb):
        """Quantize then dequantize should give reasonable results."""
        x = torch.randn(256, 256, device='mps', dtype=torch.float16)

        quantized, state = bnb.functional.quantize_4bit(x)
        dequantized = bnb.functional.dequantize_4bit(quantized, state)

        # NF4 is lossy, but shouldn't be wildly off
        max_diff = (x - dequantized).abs().max().item()
        mean_diff = (x - dequantized).abs().mean().item()

        assert max_diff < 2.0, f"Roundtrip max diff {max_diff} too large (memory corruption?)"
        assert mean_diff < 0.5, f"Roundtrip mean diff {mean_diff} too large"

    def test_adjacent_buffer_corruption(self, bnb):
        """Check if quantization corrupts adjacent memory."""
        # Create adjacent tensors
        a = torch.randn(64, 64, device='mps', dtype=torch.float16)
        sentinel = torch.full((64, 64), 42.0, device='mps', dtype=torch.float16)
        b = torch.randn(64, 64, device='mps', dtype=torch.float16)

        sentinel_copy = sentinel.clone()

        # Quantize a and b (but not sentinel)
        qa, sa = bnb.functional.quantize_4bit(a)
        qb, sb = bnb.functional.quantize_4bit(b)

        # Sentinel should be unchanged
        assert torch.equal(sentinel, sentinel_copy), "Adjacent memory was corrupted!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
