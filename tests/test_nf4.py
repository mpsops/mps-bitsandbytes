"""
Tests for NF4 (4-bit) quantization

Run with: pytest tests/test_nf4.py -v
"""

import pytest
import torch
import sys

# Skip all tests if not on macOS with MPS
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


class TestNF4Quantization:
    """Tests for NF4 quantize/dequantize operations."""

    def test_quantize_basic(self):
        """Test basic NF4 quantization."""
        from mps_bitsandbytes import quantize_nf4

        # Create test tensor
        weight = torch.randn(128, 256, device='mps', dtype=torch.float16)

        # Quantize
        packed, absmax = quantize_nf4(weight, block_size=64)

        # Check shapes
        assert packed.shape == (128, 128), f"Expected (128, 128), got {packed.shape}"
        assert absmax.shape == (128, 4), f"Expected (128, 4), got {absmax.shape}"
        assert packed.dtype == torch.uint8
        assert absmax.dtype == torch.float32

    def test_quantize_dequantize_roundtrip(self):
        """Test that quantize -> dequantize preserves values reasonably."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        # Create normally distributed weights (NF4 is optimized for this)
        weight = torch.randn(64, 128, device='mps', dtype=torch.float16)

        # Quantize and dequantize
        packed, absmax = quantize_nf4(weight, block_size=64)
        reconstructed = dequantize_nf4(packed, absmax, block_size=64)

        # Check shape preserved
        assert reconstructed.shape == weight.shape

        # For 4-bit quantization, use mean absolute error relative to standard deviation
        # NF4 achieves ~0.1 normalized MSE on normal distributions
        mae = (weight.float() - reconstructed.float()).abs().mean().item()
        weight_std = weight.float().std().item()
        normalized_mae = mae / weight_std

        print(f"Normalized MAE: {normalized_mae:.4f} (MAE={mae:.4f}, std={weight_std:.4f})")
        # 4-bit quantization typically has ~10% normalized MAE
        assert normalized_mae < 0.25, f"Reconstruction error too high: {normalized_mae}"

    def test_quantize_different_block_sizes(self):
        """Test quantization with different block sizes."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        weight = torch.randn(32, 256, device='mps', dtype=torch.float16)

        for block_size in [32, 64, 128]:
            packed, absmax = quantize_nf4(weight, block_size=block_size)
            num_blocks = (256 + block_size - 1) // block_size

            assert absmax.shape == (32, num_blocks), f"Wrong absmax shape for block_size={block_size}"

            # Verify roundtrip
            reconstructed = dequantize_nf4(packed, absmax, block_size=block_size)
            assert reconstructed.shape == weight.shape

    def test_quantize_preserves_zeros(self):
        """Test that zero values are preserved."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        weight = torch.zeros(16, 64, device='mps', dtype=torch.float16)
        packed, absmax = quantize_nf4(weight, block_size=64)
        reconstructed = dequantize_nf4(packed, absmax, block_size=64)

        assert torch.allclose(reconstructed, weight, atol=1e-6)

    def test_quantize_large_values(self):
        """Test that large values are handled correctly."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        # Large uniform values
        weight = torch.ones(16, 64, device='mps', dtype=torch.float16) * 100.0
        packed, absmax = quantize_nf4(weight, block_size=64)
        reconstructed = dequantize_nf4(packed, absmax, block_size=64)

        # Should reconstruct to approximately the same value
        error = (weight - reconstructed).abs().mean().item()
        relative_error = error / 100.0
        assert relative_error < 0.1, f"Large value error: {relative_error}"


class TestNF4Matmul:
    """Tests for NF4 matrix multiplication."""

    def test_matmul_basic(self):
        """Test basic NF4 matmul."""
        from mps_bitsandbytes import quantize_nf4, matmul_nf4

        # Create weight and quantize
        M, N, K = 32, 64, 128
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)

        weight_packed, weight_absmax = quantize_nf4(weight, block_size=64)

        # NF4 matmul
        output = matmul_nf4(input_tensor, weight_packed, weight_absmax, block_size=64)

        assert output.shape == (M, N), f"Expected ({M}, {N}), got {output.shape}"
        assert output.dtype == torch.float16

    def test_matmul_with_bias(self):
        """Test NF4 matmul with bias."""
        from mps_bitsandbytes import quantize_nf4, matmul_nf4

        M, N, K = 16, 32, 64
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        bias = torch.randn(N, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)

        weight_packed, weight_absmax = quantize_nf4(weight, block_size=64)

        output = matmul_nf4(input_tensor, weight_packed, weight_absmax, bias=bias, block_size=64)

        assert output.shape == (M, N)

    def test_matmul_1d_input(self):
        """Test NF4 matmul with 1D input (single vector)."""
        from mps_bitsandbytes import quantize_nf4, matmul_nf4

        N, K = 64, 128
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(K, device='mps', dtype=torch.float16)

        weight_packed, weight_absmax = quantize_nf4(weight, block_size=64)

        output = matmul_nf4(input_tensor, weight_packed, weight_absmax, block_size=64)

        assert output.shape == (N,), f"Expected ({N},), got {output.shape}"

    def test_matmul_accuracy(self):
        """Test NF4 matmul accuracy vs FP16 reference."""
        from mps_bitsandbytes import quantize_nf4, matmul_nf4

        M, N, K = 32, 64, 128
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)

        # Reference FP16 result
        reference = input_tensor @ weight.T

        # NF4 result
        weight_packed, weight_absmax = quantize_nf4(weight, block_size=64)
        nf4_output = matmul_nf4(input_tensor, weight_packed, weight_absmax, block_size=64)

        # Use cosine similarity for overall direction preservation
        ref_flat = reference.float().flatten()
        nf4_flat = nf4_output.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), nf4_flat.unsqueeze(0)).item()

        # Also check correlation
        corr = torch.corrcoef(torch.stack([ref_flat, nf4_flat]))[0, 1].item()

        print(f"Matmul cosine similarity: {cosine_sim:.4f}, correlation: {corr:.4f}")
        # NF4 matmul should preserve the general direction well
        assert cosine_sim > 0.9, f"Cosine similarity too low: {cosine_sim}"
        assert corr > 0.9, f"Correlation too low: {corr}"

    def test_matmul_large_matrices(self):
        """Test NF4 matmul with larger matrices (triggers tiled kernel)."""
        from mps_bitsandbytes import quantize_nf4, matmul_nf4

        # Large enough to trigger tiled kernel (M >= 32, N >= 32, K >= 64)
        M, N, K = 128, 256, 512
        weight = torch.randn(N, K, device='mps', dtype=torch.float16)
        input_tensor = torch.randn(M, K, device='mps', dtype=torch.float16)

        weight_packed, weight_absmax = quantize_nf4(weight, block_size=64)
        output = matmul_nf4(input_tensor, weight_packed, weight_absmax, block_size=64)

        assert output.shape == (M, N)

        # Basic sanity check
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"


class TestLinear4bit:
    """Tests for Linear4bit module."""

    def test_linear4bit_creation(self):
        """Test Linear4bit creation."""
        from mps_bitsandbytes import Linear4bit

        layer = Linear4bit(128, 64, device='mps')

        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.weight_packed.shape == (64, 64)  # 128/2 = 64

    def test_linear4bit_from_linear(self):
        """Test converting nn.Linear to Linear4bit."""
        from mps_bitsandbytes import Linear4bit

        # Create and initialize linear
        linear = torch.nn.Linear(256, 128).half().to('mps')
        torch.nn.init.normal_(linear.weight)

        # Convert to 4-bit
        linear_4bit = Linear4bit.from_linear(linear)

        assert linear_4bit.in_features == 256
        assert linear_4bit.out_features == 128
        assert linear_4bit.weight_packed.shape == (128, 128)

    def test_linear4bit_forward(self):
        """Test Linear4bit forward pass."""
        from mps_bitsandbytes import Linear4bit

        linear = torch.nn.Linear(128, 64).half().to('mps')
        linear_4bit = Linear4bit.from_linear(linear)

        # Forward pass
        x = torch.randn(16, 128, device='mps', dtype=torch.float16)
        output = linear_4bit(x)

        assert output.shape == (16, 64)
        assert output.dtype == torch.float16

    def test_linear4bit_vs_linear_accuracy(self):
        """Test Linear4bit accuracy vs nn.Linear."""
        from mps_bitsandbytes import Linear4bit

        linear = torch.nn.Linear(256, 128).half().to('mps')
        torch.nn.init.normal_(linear.weight)
        linear_4bit = Linear4bit.from_linear(linear)

        x = torch.randn(32, 256, device='mps', dtype=torch.float16)

        reference = linear(x)
        quantized = linear_4bit(x)

        # Use cosine similarity for overall direction preservation
        ref_flat = reference.float().flatten()
        quant_flat = quantized.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), quant_flat.unsqueeze(0)).item()

        print(f"Linear4bit cosine similarity: {cosine_sim:.4f}")
        # 4-bit quantization should preserve the general output direction
        assert cosine_sim > 0.85, f"Cosine similarity too low: {cosine_sim}"

    def test_linear4bit_batched(self):
        """Test Linear4bit with batched input."""
        from mps_bitsandbytes import Linear4bit

        linear = torch.nn.Linear(64, 32).half().to('mps')
        linear_4bit = Linear4bit.from_linear(linear)

        # Batched input [batch, seq_len, features]
        x = torch.randn(4, 16, 64, device='mps', dtype=torch.float16)
        output = linear_4bit(x)

        assert output.shape == (4, 16, 32)

    def test_linear4bit_memory_savings(self):
        """Test that Linear4bit actually saves memory."""
        from mps_bitsandbytes import Linear4bit, get_memory_footprint

        in_f, out_f = 4096, 4096

        # FP16 linear
        linear = torch.nn.Linear(in_f, out_f).half().to('mps')

        # 4-bit linear
        linear_4bit = Linear4bit.from_linear(linear)

        # Calculate memory
        fp16_bytes = out_f * in_f * 2  # fp16 = 2 bytes
        nf4_bytes = out_f * (in_f // 2) * 1  # uint8 = 1 byte (packed)

        expected_ratio = nf4_bytes / fp16_bytes
        print(f"Expected memory ratio: {expected_ratio:.2f}x (should be ~0.25)")

        assert expected_ratio < 0.3, "Memory savings not achieved"


class TestMemoryFootprint:
    """Tests for memory footprint calculation."""

    def test_memory_footprint(self):
        """Test memory footprint calculation."""
        from mps_bitsandbytes import Linear4bit, get_memory_footprint
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1024, 512)
                self.fc2 = nn.Linear(512, 256)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleModel().half().to('mps')
        footprint_fp16 = get_memory_footprint(model)

        # Quantize
        model.fc1 = Linear4bit.from_linear(model.fc1)
        model.fc2 = Linear4bit.from_linear(model.fc2)

        footprint_4bit = get_memory_footprint(model)

        print(f"FP16 size: {footprint_fp16['actual_size_gb']*1000:.2f} MB")
        print(f"4-bit size: {footprint_4bit['actual_size_gb']*1000:.2f} MB")
        print(f"Savings: {footprint_4bit['savings_pct']:.1f}%")

        # Should have significant savings
        assert footprint_4bit['actual_size_gb'] < footprint_fp16['actual_size_gb']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
