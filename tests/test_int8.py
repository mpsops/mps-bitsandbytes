"""
Tests for INT8 (8-bit) quantization

Run with: pytest tests/test_int8.py -v
"""

import pytest
import torch

# Skip all tests if not on macOS with MPS
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


class TestInt8Quantization:
    """Tests for INT8 quantize/dequantize operations."""

    def test_quantize_rowwise(self):
        """Test row-wise INT8 quantization."""
        from mps_bitsandbytes import quantize_rowwise

        tensor = torch.randn(64, 128, device='mps', dtype=torch.float16)
        quantized, scales = quantize_rowwise(tensor)

        assert quantized.shape == tensor.shape
        assert quantized.dtype == torch.int8
        assert scales.shape == (64,)

    def test_dequantize_rowwise(self):
        """Test row-wise INT8 dequantization."""
        from mps_bitsandbytes import quantize_rowwise, dequantize_rowwise

        tensor = torch.randn(32, 64, device='mps', dtype=torch.float16)
        quantized, scales = quantize_rowwise(tensor)
        reconstructed = dequantize_rowwise(quantized, scales, dtype=torch.float16)

        assert reconstructed.shape == tensor.shape
        assert reconstructed.dtype == torch.float16

        # Check reconstruction error (should be small for int8)
        error = (tensor.float() - reconstructed.float()).abs()
        relative_error = error / (tensor.float().abs() + 1e-8)
        mean_relative_error = relative_error.mean().item()

        print(f"INT8 mean relative error: {mean_relative_error:.4f}")
        assert mean_relative_error < 0.05, f"INT8 error too high: {mean_relative_error}"

    def test_quantize_preserves_sign(self):
        """Test that quantization preserves sign."""
        from mps_bitsandbytes import quantize_rowwise, dequantize_rowwise

        tensor = torch.tensor([[-1.0, 0.0, 1.0], [-0.5, 0.5, 0.0]], device='mps', dtype=torch.float16)
        quantized, scales = quantize_rowwise(tensor)
        reconstructed = dequantize_rowwise(quantized, scales)

        # Signs should match
        original_signs = torch.sign(tensor)
        reconstructed_signs = torch.sign(reconstructed)

        # Zero might have different sign representation, so only check non-zeros
        non_zero = tensor.abs() > 0.1
        assert torch.all(original_signs[non_zero] == reconstructed_signs[non_zero])


class TestLinear8bit:
    """Tests for Linear8bit module."""

    def test_linear8bit_creation(self):
        """Test Linear8bit creation."""
        from mps_bitsandbytes import Linear8bit

        layer = Linear8bit(128, 64, device='mps')

        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.weight_int8.shape == (64, 128)
        assert layer.weight_int8.dtype == torch.int8

    def test_linear8bit_from_linear(self):
        """Test converting nn.Linear to Linear8bit."""
        from mps_bitsandbytes import Linear8bit

        linear = torch.nn.Linear(256, 128).half().to('mps')
        linear_8bit = Linear8bit.from_linear(linear)

        assert linear_8bit.in_features == 256
        assert linear_8bit.out_features == 128

    def test_linear8bit_forward(self):
        """Test Linear8bit forward pass."""
        from mps_bitsandbytes import Linear8bit

        linear = torch.nn.Linear(128, 64).half().to('mps')
        linear_8bit = Linear8bit.from_linear(linear)

        x = torch.randn(16, 128, device='mps', dtype=torch.float16)
        output = linear_8bit(x)

        assert output.shape == (16, 64)

    def test_linear8bit_accuracy(self):
        """Test Linear8bit accuracy vs nn.Linear."""
        from mps_bitsandbytes import Linear8bit

        linear = torch.nn.Linear(256, 128).half().to('mps')
        linear_8bit = Linear8bit.from_linear(linear)

        x = torch.randn(32, 256, device='mps', dtype=torch.float16)

        reference = linear(x)
        quantized = linear_8bit(x)

        error = (reference.float() - quantized.float()).abs()
        relative_error = error / (reference.float().abs() + 1e-8)
        mean_relative_error = relative_error.mean().item()

        print(f"Linear8bit mean relative error: {mean_relative_error:.4f}")
        assert mean_relative_error < 0.1  # INT8 is accurate but not perfect

    def test_linear8bit_cache(self):
        """Test Linear8bit weight caching."""
        from mps_bitsandbytes import Linear8bit

        linear = torch.nn.Linear(128, 64).half().to('mps')
        linear_8bit = Linear8bit.from_linear(linear, use_cache=True)

        x = torch.randn(16, 128, device='mps', dtype=torch.float16)

        # First forward (populates cache)
        output1 = linear_8bit(x)

        # Check cache exists
        assert linear_8bit._weight_cache is not None

        # Second forward (uses cache)
        output2 = linear_8bit(x)

        assert torch.allclose(output1, output2)

        # Clear cache
        linear_8bit.clear_cache()
        assert linear_8bit._weight_cache is None


class TestInt8Matmul:
    """Tests for INT8 matrix multiplication."""

    def test_matmul_int8(self):
        """Test INT8 matmul."""
        from mps_bitsandbytes import quantize_rowwise, matmul_int8

        M, K, N = 32, 64, 48

        A = torch.randn(M, K, device='mps', dtype=torch.float16)
        B = torch.randn(K, N, device='mps', dtype=torch.float16)

        A_int8, A_scales = quantize_rowwise(A)
        B_int8, B_scales = quantize_rowwise(B.T)  # Quantize transposed
        B_int8 = B_int8.T.contiguous()  # Restore layout

        output = matmul_int8(A_int8, B_int8, A_scales, B_scales)

        assert output.shape == (M, N)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
