"""
Quantized Embedding layers for MPS

4-bit and 8-bit embeddings for memory-efficient LLM inference.
Embeddings often account for 10-30% of model size in LLMs.
"""

import torch
from torch import nn, Tensor
from typing import Optional

from ..functional import (
    quantize_nf4, dequantize_nf4,
    quantize_fp4, dequantize_fp4,
    quantize_rowwise, dequantize_rowwise,
    QuantState,
)


class Embedding4bit(nn.Module):
    """
    4-bit quantized embedding layer.

    Stores embedding weights in NF4 or FP4 format, reducing memory
    by ~75% compared to FP16 embeddings.

    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of embeddings
        padding_idx: If specified, entries at this index are zeros
        quant_type: Quantization type ('nf4' or 'fp4')
        blocksize: Block size for quantization (default: 64)

    Example:
        >>> embed = Embedding4bit(50000, 4096)
        >>> embed_4bit = Embedding4bit.from_embedding(embed)
        >>> output = embed_4bit(input_ids)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        quant_type: str = 'nf4',
        blocksize: int = 64,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__()

        if quant_type not in ('nf4', 'fp4'):
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type}")

        # Ensure embedding_dim is even for packing
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.quant_type = quant_type
        self.blocksize = blocksize
        self.dtype = dtype

        num_blocks = (embedding_dim + blocksize - 1) // blocksize

        # Quantized weight storage
        self.register_buffer(
            'weight_packed',
            torch.zeros(num_embeddings, embedding_dim // 2, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            'weight_absmax',
            torch.ones(num_embeddings, num_blocks, dtype=torch.float32, device=device)
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Look up embeddings for input indices.

        Args:
            input: Tensor of indices [*, seq_len]

        Returns:
            Embedded tensor [*, seq_len, embedding_dim]
        """
        # Get unique indices for efficient dequantization
        flat_input = input.flatten()
        unique_indices, inverse = flat_input.unique(return_inverse=True)

        # Dequantize only unique embeddings
        dequant_fn = dequantize_nf4 if self.quant_type == 'nf4' else dequantize_fp4

        # Gather packed weights for unique indices
        packed = self.weight_packed[unique_indices]
        absmax = self.weight_absmax[unique_indices]

        # Create QuantState for each row and dequantize
        embeddings_list = []
        for i in range(packed.shape[0]):
            quant_state = QuantState(
                absmax=absmax[i],
                shape=torch.Size([self.embedding_dim]),
                blocksize=self.blocksize,
                quant_type=self.quant_type,
                dtype=self.dtype,
            )
            emb = dequant_fn(packed[i], quant_state)
            embeddings_list.append(emb)
        embeddings = torch.stack(embeddings_list)

        # Gather using inverse indices
        output = embeddings[inverse]

        # Reshape to original input shape + embedding_dim
        output_shape = list(input.shape) + [self.embedding_dim]
        output = output.view(*output_shape)

        # Handle padding_idx
        if self.padding_idx is not None:
            mask = (input == self.padding_idx).unsqueeze(-1)
            output = output.masked_fill(mask, 0.0)

        return output

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        quant_type: str = 'nf4',
        blocksize: int = 64,
        device=None,
    ) -> 'Embedding4bit':
        """
        Convert a regular nn.Embedding to 4-bit.

        Args:
            embedding: Source nn.Embedding layer
            quant_type: Quantization type ('nf4' or 'fp4')
            blocksize: Block size for quantization

        Returns:
            Embedding4bit layer with quantized weights
        """
        if device is None:
            device = embedding.weight.device

        dtype = embedding.weight.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float16

        # Handle odd embedding_dim
        embedding_dim = embedding.embedding_dim
        weight = embedding.weight.data
        if embedding_dim % 2 != 0:
            embedding_dim = embedding_dim + 1
            weight = torch.nn.functional.pad(weight, (0, 1))

        layer = cls(
            embedding.num_embeddings,
            embedding_dim,
            padding_idx=embedding.padding_idx,
            quant_type=quant_type,
            blocksize=blocksize,
            device=device,
            dtype=dtype,
        )

        # Quantize weights row by row
        quantize_fn = quantize_nf4 if quant_type == 'nf4' else quantize_fp4
        weight_device = weight.to(device)
        packed_list = []
        absmax_list = []
        for i in range(weight_device.shape[0]):
            packed, state = quantize_fn(weight_device[i], blocksize=blocksize)
            packed_list.append(packed)
            absmax_list.append(state.absmax)

        layer.weight_packed.copy_(torch.stack(packed_list))
        layer.weight_absmax.copy_(torch.stack(absmax_list))

        return layer

    def extra_repr(self) -> str:
        return (
            f'{self.num_embeddings}, {self.embedding_dim}, '
            f'padding_idx={self.padding_idx}, quant_type={self.quant_type}, '
            f'blocksize={self.blocksize}'
        )


class Embedding8bit(nn.Module):
    """
    8-bit quantized embedding layer.

    Stores embedding weights in int8 format with row-wise scaling,
    reducing memory by ~50% compared to FP16.

    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of embeddings
        padding_idx: If specified, entries at this index are zeros
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.dtype = dtype

        self.register_buffer(
            'weight_int8',
            torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scales',
            torch.ones(num_embeddings, dtype=torch.float32, device=device)
        )

    def forward(self, input: Tensor) -> Tensor:
        """Look up embeddings for input indices."""
        # Gather int8 weights
        weight_int8 = self.weight_int8[input]  # [*, seq_len, embed_dim]
        scales = self.weight_scales[input]  # [*, seq_len]

        # Dequantize
        output = weight_int8.to(self.dtype) * (scales.unsqueeze(-1) / 127.0).to(self.dtype)

        # Handle padding_idx
        if self.padding_idx is not None:
            mask = (input == self.padding_idx).unsqueeze(-1)
            output = output.masked_fill(mask, 0.0)

        return output

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        device=None,
    ) -> 'Embedding8bit':
        """Convert nn.Embedding to 8-bit."""
        if device is None:
            device = embedding.weight.device

        dtype = embedding.weight.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float16

        layer = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            device=device,
            dtype=dtype,
        )

        # Quantize weights
        weight_int8, weight_scales = quantize_rowwise(embedding.weight.data.to(device))
        layer.weight_int8.copy_(weight_int8)
        layer.weight_scales.copy_(weight_scales)

        return layer

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}, padding_idx={self.padding_idx}'


# Aliases for specific quant types
class EmbeddingNF4(Embedding4bit):
    """4-bit NF4 embedding (alias for Embedding4bit with quant_type='nf4')."""

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        kwargs['quant_type'] = 'nf4'
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    @classmethod
    def from_embedding(cls, embedding, blocksize: int = 64, device=None) -> 'EmbeddingNF4':
        return super().from_embedding(embedding, quant_type='nf4', blocksize=blocksize, device=device)


class EmbeddingFP4(Embedding4bit):
    """4-bit FP4 embedding (alias for Embedding4bit with quant_type='fp4')."""

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):
        kwargs['quant_type'] = 'fp4'
        super().__init__(num_embeddings, embedding_dim, **kwargs)

    @classmethod
    def from_embedding(cls, embedding, blocksize: int = 64, device=None) -> 'EmbeddingFP4':
        return super().from_embedding(embedding, quant_type='fp4', blocksize=blocksize, device=device)
