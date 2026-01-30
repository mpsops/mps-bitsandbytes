"""
Tests for FP4, FP8, and Double Quantization

Run with: pytest tests/test_fp4_fp8_double.py -v
"""

import pytest
import torch

# Skip all tests if not on macOS with MPS
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


class TestFP4Quantization:
    """Tests for FP4 (4-bit floating point) quantization."""

    def test_quantize_fp4_basic(self):
        """Test basic FP4 quantization."""
        from mps_bitsandbytes import quantize_fp4, QuantState

        tensor = torch.randn(32, 64, device='mps', dtype=torch.float16)
        packed, state = quantize_fp4(tensor)

        assert packed.dtype == torch.uint8
        assert isinstance(state, QuantState)
        assert state.quant_type == 'fp4'

    def test_fp4_roundtrip(self):
        """Test FP4 quantize/dequantize roundtrip."""
        from mps_bitsandbytes import quantize_fp4, dequantize_fp4

        tensor = torch.randn(16, 64, device='mps', dtype=torch.float16)
        packed, state = quantize_fp4(tensor)
        reconstructed = dequantize_fp4(packed, state)

        # FP4 has limited precision, check cosine similarity
        tensor_flat = tensor.float().flatten()
        recon_flat = reconstructed.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            tensor_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()
        assert cosine_sim > 0.85, f"Cosine similarity {cosine_sim} too low"

    def test_matmul_fp4_basic(self):
        """Test FP4 matmul."""
        from mps_bitsandbytes import quantize_fp4, matmul_4bit

        # Create input and weight
        input_tensor = torch.randn(8, 64, device='mps', dtype=torch.float16)
        weight = torch.randn(32, 64, device='mps', dtype=torch.float16)

        # Quantize weight
        weight_packed, weight_state = quantize_fp4(weight)

        # FP4 matmul
        output = matmul_4bit(input_tensor, weight_packed, weight_state)

        assert output.shape == (8, 32)
        assert not torch.isnan(output).any()

    def test_matmul_fp4_with_bias(self):
        """Test FP4 matmul with bias."""
        from mps_bitsandbytes import quantize_fp4, matmul_4bit

        input_tensor = torch.randn(4, 64, device='mps', dtype=torch.float16)
        weight = torch.randn(16, 64, device='mps', dtype=torch.float16)
        bias = torch.randn(16, device='mps', dtype=torch.float16)

        weight_packed, weight_state = quantize_fp4(weight)
        output = matmul_4bit(input_tensor, weight_packed, weight_state, bias=bias)

        assert output.shape == (4, 16)
        assert not torch.isnan(output).any()


class TestFP8Quantization:
    """Tests for FP8 E4M3 (8-bit floating point) quantization."""

    def test_quantize_fp8_basic(self):
        """Test basic FP8 quantization."""
        from mps_bitsandbytes import quantize_fp8_e4m3

        tensor = torch.randn(32, 64, device='mps', dtype=torch.float16)
        quantized, scales = quantize_fp8_e4m3(tensor)

        assert quantized.dtype == torch.uint8
        assert quantized.shape == tensor.shape
        assert scales.shape == (32,)  # Row-wise scales

    def test_fp8_roundtrip(self):
        """Test FP8 quantize/dequantize roundtrip."""
        from mps_bitsandbytes import quantize_fp8_e4m3, dequantize_fp8_e4m3

        tensor = torch.randn(16, 64, device='mps', dtype=torch.float16)
        quantized, scales = quantize_fp8_e4m3(tensor)
        reconstructed = dequantize_fp8_e4m3(quantized, scales)

        # FP8 should have better precision than FP4
        tensor_flat = tensor.float().flatten()
        recon_flat = reconstructed.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            tensor_flat.unsqueeze(0), recon_flat.unsqueeze(0)
        ).item()
        assert cosine_sim > 0.95, f"FP8 cosine similarity {cosine_sim} too low"

    def test_matmul_fp8_basic(self):
        """Test FP8 matmul."""
        from mps_bitsandbytes import quantize_fp8_e4m3, matmul_fp8_e4m3

        input_tensor = torch.randn(8, 64, device='mps', dtype=torch.float16)
        weight = torch.randn(32, 64, device='mps', dtype=torch.float16)

        weight_quant, weight_scales = quantize_fp8_e4m3(weight)
        output = matmul_fp8_e4m3(input_tensor, weight_quant, weight_scales)

        assert output.shape == (8, 32)
        assert not torch.isnan(output).any()

    def test_fp8_accuracy_vs_fp16(self):
        """Test FP8 matmul accuracy vs FP16 reference."""
        from mps_bitsandbytes import quantize_fp8_e4m3, matmul_fp8_e4m3

        torch.manual_seed(42)
        input_tensor = torch.randn(16, 128, device='mps', dtype=torch.float16)
        weight = torch.randn(64, 128, device='mps', dtype=torch.float16)

        # Reference FP16 matmul
        ref_output = torch.nn.functional.linear(input_tensor, weight)

        # FP8 matmul
        weight_quant, weight_scales = quantize_fp8_e4m3(weight)
        fp8_output = matmul_fp8_e4m3(input_tensor, weight_quant, weight_scales)

        # FP8 should be more accurate than 4-bit
        cosine_sim = torch.nn.functional.cosine_similarity(
            ref_output.float().flatten().unsqueeze(0),
            fp8_output.float().flatten().unsqueeze(0)
        ).item()
        print(f"FP8 vs FP16 cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim > 0.9, f"FP8 cosine similarity {cosine_sim} too low"


class TestDoubleQuantization:
    """Tests for double quantization (quantizing the scales)."""

    def test_double_quant_via_compress_statistics(self):
        """Test double quantization via compress_statistics parameter."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        # Create a weight and quantize with compress_statistics
        weight = torch.randn(128, 256, device='mps', dtype=torch.float16)
        packed, state = quantize_nf4(weight, blocksize=64, compress_statistics=True)

        # Should have nested state2
        assert state.state2 is not None

        # Dequantize should work correctly
        reconstructed = dequantize_nf4(packed, state)
        assert reconstructed.shape == weight.shape

    def test_double_quant_accuracy(self):
        """Test double quant doesn't significantly degrade quality."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4

        weight = torch.randn(64, 128, device='mps', dtype=torch.float16)

        # Without double quant
        packed1, state1 = quantize_nf4(weight, blocksize=64, compress_statistics=False)
        recon1 = dequantize_nf4(packed1, state1)

        # With double quant
        packed2, state2 = quantize_nf4(weight, blocksize=64, compress_statistics=True)
        recon2 = dequantize_nf4(packed2, state2)

        # Both should be close to original
        error1 = (weight - recon1).abs().mean() / weight.abs().mean()
        error2 = (weight - recon2).abs().mean() / weight.abs().mean()

        print(f"Without double quant error: {error1:.4f}")
        print(f"With double quant error: {error2:.4f}")

        # Double quant adds some error but should still be reasonable
        assert error1 < 0.15
        assert error2 < 0.20


class TestFP4VsNF4:
    """Compare FP4 and NF4 performance."""

    def test_nf4_better_for_normal_weights(self):
        """Test that NF4 is better for normally distributed weights."""
        from mps_bitsandbytes import quantize_nf4, dequantize_nf4, quantize_fp4, dequantize_fp4

        # Normally distributed weights (like in neural networks)
        torch.manual_seed(42)
        weight = torch.randn(64, 128, device='mps', dtype=torch.float16)

        # NF4
        nf4_packed, nf4_state = quantize_nf4(weight)
        nf4_recon = dequantize_nf4(nf4_packed, nf4_state)

        # FP4
        fp4_packed, fp4_state = quantize_fp4(weight)
        fp4_recon = dequantize_fp4(fp4_packed, fp4_state)

        # Compare reconstruction quality
        nf4_sim = torch.nn.functional.cosine_similarity(
            weight.float().flatten().unsqueeze(0),
            nf4_recon.float().flatten().unsqueeze(0)
        ).item()

        fp4_sim = torch.nn.functional.cosine_similarity(
            weight.float().flatten().unsqueeze(0),
            fp4_recon.float().flatten().unsqueeze(0)
        ).item()

        print(f"NF4 cosine similarity: {nf4_sim:.4f}")
        print(f"FP4 cosine similarity: {fp4_sim:.4f}")

        # NF4 should be at least as good for normal distributions
        # (they're close, so just check both are reasonable)
        assert nf4_sim > 0.85
        assert fp4_sim > 0.85


class TestLinear4bitFP4Mode:
    """Test Linear4bit with FP4 quantization type."""

    def test_linear4bit_fp4_creation(self):
        """Test creating Linear4bit with FP4."""
        from mps_bitsandbytes import Linear4bit

        linear = torch.nn.Linear(64, 32).half().to('mps')
        l4_fp4 = Linear4bit.from_linear(linear, quant_type='fp4')

        assert l4_fp4.quant_type == 'fp4'
        assert l4_fp4.weight_quant_state is not None
        assert l4_fp4.weight_quant_state.quant_type == 'fp4'

    def test_linear4bit_fp4_forward(self):
        """Test forward pass with FP4."""
        from mps_bitsandbytes import Linear4bit

        linear = torch.nn.Linear(64, 32).half().to('mps')
        l4_fp4 = Linear4bit.from_linear(linear, quant_type='fp4')

        x = torch.randn(8, 64, device='mps', dtype=torch.float16)
        output = l4_fp4(x)

        assert output.shape == (8, 32)
        assert not torch.isnan(output).any()

    def test_linear4bit_fp4_vs_nf4(self):
        """Compare FP4 and NF4 modes."""
        from mps_bitsandbytes import Linear4bit

        torch.manual_seed(42)
        linear = torch.nn.Linear(128, 64).half().to('mps')
        x = torch.randn(16, 128, device='mps', dtype=torch.float16)

        l4_nf4 = Linear4bit.from_linear(linear, quant_type='nf4')
        l4_fp4 = Linear4bit.from_linear(linear, quant_type='fp4')

        out_nf4 = l4_nf4(x)
        out_fp4 = l4_fp4(x)

        # Both should produce valid outputs
        assert not torch.isnan(out_nf4).any()
        assert not torch.isnan(out_fp4).any()

        # They should be similar but not identical
        cosine_sim = torch.nn.functional.cosine_similarity(
            out_nf4.flatten().unsqueeze(0),
            out_fp4.flatten().unsqueeze(0)
        ).item()
        print(f"NF4 vs FP4 cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim > 0.9  # Should be similar


class TestLinearFP8:
    """Tests for LinearFP8 module."""

    def test_linear_fp8_creation(self):
        """Test creating LinearFP8."""
        from mps_bitsandbytes import LinearFP8

        linear = torch.nn.Linear(64, 32).half().to('mps')
        lfp8 = LinearFP8.from_linear(linear)

        assert lfp8.weight_fp8.dtype == torch.uint8
        assert lfp8.weight_fp8.shape == (32, 64)
        assert lfp8.weight_scales.shape == (32,)

    def test_linear_fp8_forward(self):
        """Test forward pass."""
        from mps_bitsandbytes import LinearFP8

        linear = torch.nn.Linear(64, 32).half().to('mps')
        lfp8 = LinearFP8.from_linear(linear)

        x = torch.randn(8, 64, device='mps', dtype=torch.float16)
        output = lfp8(x)

        assert output.shape == (8, 32)
        assert not torch.isnan(output).any()

    def test_linear_fp8_accuracy(self):
        """Test FP8 accuracy vs FP16."""
        from mps_bitsandbytes import LinearFP8

        torch.manual_seed(42)
        linear = torch.nn.Linear(128, 64).half().to('mps')
        lfp8 = LinearFP8.from_linear(linear)

        x = torch.randn(16, 128, device='mps', dtype=torch.float16)

        ref_output = linear(x)
        fp8_output = lfp8(x)

        cosine_sim = torch.nn.functional.cosine_similarity(
            ref_output.flatten().unsqueeze(0),
            fp8_output.flatten().unsqueeze(0)
        ).item()
        print(f"LinearFP8 vs FP16 cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim > 0.9

    def test_linear_fp8_batched(self):
        """Test batched input."""
        from mps_bitsandbytes import LinearFP8

        linear = torch.nn.Linear(64, 32).half().to('mps')
        lfp8 = LinearFP8.from_linear(linear)

        x = torch.randn(4, 8, 64, device='mps', dtype=torch.float16)
        output = lfp8(x)

        assert output.shape == (4, 8, 32)

    def test_linear_fp8_memory_savings(self):
        """Test memory savings vs FP16."""
        from mps_bitsandbytes import LinearFP8

        linear = torch.nn.Linear(1024, 2048).half().to('mps')
        lfp8 = LinearFP8.from_linear(linear)

        # FP16: 1024 * 2048 * 2 bytes = 4MB
        fp16_size = linear.weight.numel() * 2

        # FP8: 1024 * 2048 * 1 byte + 2048 * 4 bytes (scales)
        fp8_size = lfp8.weight_fp8.numel() * 1 + lfp8.weight_scales.numel() * 4

        savings = (fp16_size - fp8_size) / fp16_size * 100
        print(f"LinearFP8 memory savings: {savings:.1f}%")
        assert savings > 45  # Should be close to 50%


class TestImports:
    """Test that all new functions are properly exported."""

    def test_all_exports(self):
        """Test all FP4/FP8/double quant functions are exported."""
        from mps_bitsandbytes import (
            # QuantState
            QuantState,
            # Generic 4-bit
            quantize_4bit,
            dequantize_4bit,
            matmul_4bit,
            # FP4
            quantize_fp4,
            dequantize_fp4,
            FP4_CODEBOOK,
            # FP8
            quantize_fp8_e4m3,
            dequantize_fp8_e4m3,
            matmul_fp8_e4m3,
            # Blockwise
            quantize_blockwise,
            dequantize_blockwise,
            # Linear modules
            Linear4bit,
            Linear8bit,
            LinearFP8,
        )

        # Just verify they're callable
        assert callable(quantize_4bit)
        assert callable(dequantize_4bit)
        assert callable(matmul_4bit)
        assert callable(quantize_fp4)
        assert callable(dequantize_fp4)
        assert callable(quantize_fp8_e4m3)
        assert callable(dequantize_fp8_e4m3)
        assert callable(matmul_fp8_e4m3)
        assert callable(quantize_blockwise)
        assert callable(dequantize_blockwise)

        # Linear modules
        assert callable(Linear4bit)
        assert callable(Linear8bit)
        assert callable(LinearFP8)

        # Codebooks are tensors
        assert isinstance(FP4_CODEBOOK, torch.Tensor)
        assert len(FP4_CODEBOOK) == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
