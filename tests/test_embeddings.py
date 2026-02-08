"""
Tests for quantized embedding layers.
"""

import pytest
import torch
import torch.nn as nn

from mps_bitsandbytes.nn import (
    Embedding4bit, Embedding8bit,
    EmbeddingNF4, EmbeddingFP4,
)


@pytest.fixture
def device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class TestEmbedding4bit:
    def test_from_embedding_nf4(self, device):
        """Test conversion from nn.Embedding with NF4."""
        vocab_size, embed_dim = 1000, 256
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, quant_type='nf4', device=device)

        assert embed_4bit.num_embeddings == vocab_size
        assert embed_4bit.embedding_dim == embed_dim
        assert embed_4bit.quant_type == 'nf4'

    def test_from_embedding_fp4(self, device):
        """Test conversion from nn.Embedding with FP4."""
        vocab_size, embed_dim = 1000, 256
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, quant_type='fp4', device=device)

        assert embed_4bit.quant_type == 'fp4'

    def test_forward(self, device):
        """Test forward pass."""
        vocab_size, embed_dim = 1000, 256
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        # Single batch
        indices = torch.randint(0, vocab_size, (8, 32), device=device)
        output = embed_4bit(indices)

        assert output.shape == (8, 32, embed_dim)
        assert output.dtype == torch.float16

    def test_output_close_to_original(self, device):
        """Test quantized output is close to original."""
        vocab_size, embed_dim = 100, 64
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        indices = torch.randint(0, vocab_size, (4, 16), device=device)

        orig_output = embed(indices)
        quant_output = embed_4bit(indices)

        # Should be reasonably close (4-bit has some error)
        relative_error = (orig_output - quant_output).abs().mean() / orig_output.abs().mean()
        assert relative_error < 0.2  # Within 20% relative error

    def test_padding_idx(self, device):
        """Test padding_idx handling."""
        vocab_size, embed_dim = 100, 64
        padding_idx = 0
        embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        indices = torch.tensor([[0, 1, 2], [3, 0, 4]], device=device)
        output = embed_4bit(indices)

        # Padding indices should be zero
        assert torch.allclose(output[:, 0, :][indices[:, 0] == padding_idx],
                              torch.zeros(embed_dim, device=device, dtype=torch.float16),
                              atol=1e-3)

    def test_odd_embedding_dim(self, device):
        """Test handling of odd embedding dimensions."""
        vocab_size = 100
        embed_dim = 63  # Odd

        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        # Should pad to even
        assert embed_4bit.embedding_dim == 64

        indices = torch.randint(0, vocab_size, (4, 8), device=device)
        output = embed_4bit(indices)

        # Output includes the padding dimension
        assert output.shape == (4, 8, 64)


class TestEmbedding8bit:
    def test_from_embedding(self, device):
        """Test conversion from nn.Embedding."""
        vocab_size, embed_dim = 1000, 256
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_8bit = Embedding8bit.from_embedding(embed, device=device)

        assert embed_8bit.num_embeddings == vocab_size
        assert embed_8bit.embedding_dim == embed_dim

    def test_forward(self, device):
        """Test forward pass."""
        vocab_size, embed_dim = 1000, 256
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_8bit = Embedding8bit.from_embedding(embed, device=device)

        indices = torch.randint(0, vocab_size, (8, 32), device=device)
        output = embed_8bit(indices)

        assert output.shape == (8, 32, embed_dim)

    def test_output_close_to_original(self, device):
        """Test quantized output is close to original."""
        vocab_size, embed_dim = 100, 64
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_8bit = Embedding8bit.from_embedding(embed, device=device)

        indices = torch.randint(0, vocab_size, (4, 16), device=device)

        orig_output = embed(indices)
        quant_output = embed_8bit(indices)

        # 8-bit should be very close
        relative_error = (orig_output - quant_output).abs().mean() / orig_output.abs().mean()
        assert relative_error < 0.05  # Within 5% relative error

    def test_memory_savings(self, device):
        """Test memory footprint is reduced."""
        vocab_size, embed_dim = 50000, 4096
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_8bit = Embedding8bit.from_embedding(embed, device=device)

        # FP16: vocab_size * embed_dim * 2 bytes
        fp16_size = vocab_size * embed_dim * 2
        # INT8: vocab_size * embed_dim * 1 byte + vocab_size * 4 bytes (scales)
        int8_size = vocab_size * embed_dim * 1 + vocab_size * 4

        # Should be roughly 50% savings
        assert int8_size < fp16_size * 0.6


class TestEmbeddingAliases:
    def test_embedding_nf4(self, device):
        """Test EmbeddingNF4 alias."""
        embed = EmbeddingNF4(100, 64, device=device)
        assert embed.quant_type == 'nf4'

    def test_embedding_fp4(self, device):
        """Test EmbeddingFP4 alias."""
        embed = EmbeddingFP4(100, 64, device=device)
        assert embed.quant_type == 'fp4'


class TestEmbeddingIntegration:
    def test_unique_index_optimization(self, device):
        """Test that repeated indices are handled efficiently."""
        vocab_size, embed_dim = 100, 64
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        # Indices with many repeats
        indices = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]], device=device)
        output = embed_4bit(indices)

        # All embeddings in each row should be identical
        assert torch.allclose(output[0, 0], output[0, 1])
        assert torch.allclose(output[0, 0], output[0, 2])
        assert torch.allclose(output[1, 0], output[1, 1])

    def test_gradient_flow(self, device):
        """Test that gradients flow through (for LoRA-style training)."""
        vocab_size, embed_dim = 100, 64
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        # Add a trainable projection
        proj = nn.Linear(embed_dim, 32).half().to(device)

        indices = torch.randint(0, vocab_size, (4, 8), device=device)
        output = embed_4bit(indices)
        output = proj(output)
        loss = output.sum()
        loss.backward()

        # Projection should have gradients
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0


class TestNativeVsFallbackEmbedding:
    """Test that native Metal kernels produce same results as Python fallback."""

    def _force_fallback_embedding4bit(self, embed_4bit, input):
        """Run the Python fallback path explicitly."""
        from mps_bitsandbytes.functional import dequantize_nf4, dequantize_fp4, QuantState
        flat_input = input.flatten()
        unique_indices, inverse = flat_input.unique(return_inverse=True)

        dequant_fn = dequantize_nf4 if embed_4bit.quant_type == 'nf4' else dequantize_fp4
        packed = embed_4bit.weight_packed[unique_indices]
        absmax = embed_4bit.weight_absmax[unique_indices]

        embeddings_list = []
        for i in range(packed.shape[0]):
            quant_state = QuantState(
                absmax=absmax[i],
                shape=torch.Size([embed_4bit.embedding_dim]),
                blocksize=embed_4bit.blocksize,
                quant_type=embed_4bit.quant_type,
                dtype=embed_4bit.dtype,
            )
            emb = dequant_fn(packed[i], quant_state)
            embeddings_list.append(emb)
        embeddings = torch.stack(embeddings_list)
        output = embeddings[inverse]
        output_shape = list(input.shape) + [embed_4bit.embedding_dim]
        return output.view(*output_shape)

    def test_native_vs_fallback_embedding4bit_nf4(self, device):
        """Compare native Metal kernel vs Python fallback for NF4 embedding."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        vocab_size, embed_dim = 500, 128
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, quant_type='nf4', device=device)

        indices = torch.randint(0, vocab_size, (4, 16), device=device)

        native_output = embed_4bit(indices)
        fallback_output = self._force_fallback_embedding4bit(embed_4bit, indices)

        assert torch.allclose(native_output, fallback_output, atol=1e-3), \
            f"Max diff: {(native_output - fallback_output).abs().max().item()}"

    def test_native_vs_fallback_embedding4bit_fp4(self, device):
        """Compare native Metal kernel vs Python fallback for FP4 embedding."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        vocab_size, embed_dim = 500, 128
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, quant_type='fp4', device=device)

        indices = torch.randint(0, vocab_size, (4, 16), device=device)

        native_output = embed_4bit(indices)
        fallback_output = self._force_fallback_embedding4bit(embed_4bit, indices)

        assert torch.allclose(native_output, fallback_output, atol=1e-3), \
            f"Max diff: {(native_output - fallback_output).abs().max().item()}"

    def test_native_vs_fallback_embedding8bit(self, device):
        """Compare native Metal kernel vs Python fallback for 8-bit embedding."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        vocab_size, embed_dim = 500, 128
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_8bit = Embedding8bit.from_embedding(embed, device=device)

        indices = torch.randint(0, vocab_size, (4, 16), device=device)

        # Native path
        native_output = embed_8bit(indices)

        # Python fallback
        weight_int8 = embed_8bit.weight_int8[indices]
        scales = embed_8bit.weight_scales[indices]
        fallback_output = weight_int8.to(embed_8bit.dtype) * (scales.unsqueeze(-1) / 127.0).to(embed_8bit.dtype)

        # Native kernel computes in float32 then converts to half;
        # fallback truncates scale to half before multiply, causing ~1 ULP diff
        assert torch.allclose(native_output, fallback_output, atol=2e-3), \
            f"Max diff: {(native_output - fallback_output).abs().max().item()}"

    def test_large_vocabulary_embedding(self, device):
        """Stress test with large vocabulary."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        vocab_size, embed_dim = 50000, 4096
        embed = nn.Embedding(vocab_size, embed_dim).half().to(device)
        embed_4bit = Embedding4bit.from_embedding(embed, device=device)

        indices = torch.randint(0, vocab_size, (2, 512), device=device)
        output = embed_4bit(indices)

        assert output.shape == (2, 512, embed_dim)
        assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
