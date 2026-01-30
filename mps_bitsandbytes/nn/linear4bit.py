"""
Linear4bit - 4-bit NF4/FP4 quantized linear layer for MPS

Provides ~4x memory reduction compared to FP16 weights with minimal accuracy loss.
Compatible with HuggingFace transformers and QLoRA training.
"""

import torch
from torch import nn, Tensor
from typing import Optional

from ..functional import (
    quantize_nf4, dequantize_nf4, matmul_nf4,
    quantize_fp4, dequantize_fp4, matmul_fp4,
)


class Linear4bit(nn.Module):
    """
    4-bit quantized linear layer using NF4 (NormalFloat4) quantization.

    NF4 is optimized for normally distributed weights, achieving ~4x memory
    reduction compared to FP16 with minimal accuracy loss. This is the same
    quantization used by QLoRA.

    Storage format:
    - weight_packed: [out_features, in_features // 2] uint8
    - weight_absmax: [out_features, num_blocks] float32

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: True)
        device: Target device
        compute_dtype: Dtype for computation (torch.float16 or torch.bfloat16)
        quant_type: Quantization type ('nf4' or 'fp4')
        block_size: Block size for quantization (default: 64)

    Example:
        >>> linear = Linear4bit(1024, 4096)
        >>> linear.load_from_linear(pretrained_linear)
        >>> output = linear(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = 'nf4',
        block_size: int = 64,
    ):
        super().__init__()

        if quant_type not in ('nf4', 'fp4'):
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type}")

        if in_features % 2 != 0:
            raise ValueError(f"in_features must be even for 4-bit packing, got {in_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.block_size = block_size

        num_blocks = (in_features + block_size - 1) // block_size

        # Quantized weight storage
        self.register_buffer(
            'weight_packed',
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device)
        )
        self.register_buffer(
            'weight_absmax',
            torch.ones(out_features, num_blocks, dtype=torch.float32, device=device)
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)

        # Metadata for HuggingFace compatibility
        self.quant_state = {
            'quant_type': quant_type,
            'block_size': block_size,
            'compute_dtype': compute_dtype,
        }

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with fused 4-bit dequantization and matmul.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Handle batched input
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)

        # Select matmul function based on quant_type
        matmul_fn = matmul_nf4 if self.quant_type == 'nf4' else matmul_fp4

        # Fused 4-bit matmul (dequantize + matmul in Metal kernel)
        output = matmul_fn(
            x,
            self.weight_packed,
            self.weight_absmax,
            self.bias,
            self.block_size,
            self.compute_dtype
        )

        # Restore batch dimensions
        if len(orig_shape) > 2:
            output = output.view(*orig_shape[:-1], self.out_features)

        return output

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        device=None,
        compute_dtype: Optional[torch.dtype] = None,
        quant_type: str = 'nf4',
        block_size: int = 64,
    ) -> 'Linear4bit':
        """
        Convert a regular nn.Linear to 4-bit quantized.

        Args:
            linear: Source nn.Linear layer
            device: Target device (default: same as source)
            compute_dtype: Dtype for computation (default: infer from source)
            quant_type: Quantization type ('nf4' or 'fp4')
            block_size: Block size for quantization

        Returns:
            Linear4bit layer with quantized weights

        Example:
            >>> linear_fp16 = nn.Linear(1024, 4096).half().to('mps')
            >>> linear_4bit = Linear4bit.from_linear(linear_fp16)
        """
        if device is None:
            device = linear.weight.device

        if compute_dtype is None:
            if linear.weight.dtype == torch.bfloat16:
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16

        # Handle odd in_features by padding
        in_features = linear.in_features
        if in_features % 2 != 0:
            in_features = in_features + 1
            # Pad weight with zeros
            weight = torch.nn.functional.pad(linear.weight.data, (0, 1))
        else:
            weight = linear.weight.data

        layer = cls(
            in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=device,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            block_size=block_size,
        )

        # Store original in_features for forward if padded
        layer._original_in_features = linear.in_features

        # Quantize weights (select function based on quant_type)
        quantize_fn = quantize_nf4 if quant_type == 'nf4' else quantize_fp4
        weight_packed, weight_absmax = quantize_fn(
            weight.to(device),
            block_size=block_size
        )
        layer.weight_packed.copy_(weight_packed)
        layer.weight_absmax.copy_(weight_absmax)

        # Copy bias
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.to(compute_dtype).to(device))

        return layer

    def dequantize(self) -> Tensor:
        """
        Dequantize weights back to floating point.

        Useful for debugging or when full precision is needed temporarily.

        Returns:
            Dequantized weight tensor [out_features, in_features]
        """
        dequant_fn = dequantize_nf4 if self.quant_type == 'nf4' else dequantize_fp4
        return dequant_fn(
            self.weight_packed,
            self.weight_absmax,
            self.block_size,
            self.compute_dtype
        )

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, quant_type={self.quant_type}, '
            f'block_size={self.block_size}'
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading from state dict, including conversion from FP16/FP32."""
        # Check if we're loading from a non-quantized state dict
        weight_key = prefix + 'weight'
        packed_key = prefix + 'weight_packed'

        if weight_key in state_dict and packed_key not in state_dict:
            # Convert from FP16/FP32 weight on the fly
            weight = state_dict.pop(weight_key)
            quantize_fn = quantize_nf4 if self.quant_type == 'nf4' else quantize_fp4
            weight_packed, weight_absmax = quantize_fn(weight, self.block_size)
            state_dict[packed_key] = weight_packed
            state_dict[prefix + 'weight_absmax'] = weight_absmax

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


class Params4bit(nn.Parameter):
    """
    Parameter wrapper for 4-bit quantized tensors.

    This is used for compatibility with HuggingFace transformers which
    expect weight parameters to have certain attributes.
    """

    def __new__(cls, data=None, requires_grad=False, quant_state=None):
        if data is None:
            data = torch.empty(0)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.quant_state = quant_state
        return instance

    @property
    def shape(self):
        # Return logical shape (unpacked)
        if hasattr(self, 'quant_state') and self.quant_state is not None:
            return (self.quant_state.get('shape', super().shape))
        return super().shape
