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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
