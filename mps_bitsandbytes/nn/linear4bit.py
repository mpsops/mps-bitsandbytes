"""
Linear4bit - 4-bit NF4/FP4 quantized linear layer for MPS

Provides ~4x memory reduction compared to FP16 weights with minimal accuracy loss.
Compatible with HuggingFace transformers and QLoRA training.
"""

import torch
from torch import nn, Tensor
from typing import Optional

from ..functional import (
    quantize_4bit, dequantize_4bit, matmul_4bit,
    QuantState,
)


class Linear4bit(nn.Module):
    """
    4-bit quantized linear layer using NF4 (NormalFloat4) quantization.

    NF4 is optimized for normally distributed weights, achieving ~4x memory
    reduction compared to FP16 with minimal accuracy loss. This is the same
    quantization used by QLoRA.

    Storage format:
    - weight: packed uint8 tensor
    - weight.quant_state: QuantState with absmax, shape, etc.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: True)
        device: Target device
        compute_dtype: Dtype for computation (torch.float16 or torch.bfloat16)
        quant_type: Quantization type ('nf4' or 'fp4')
        blocksize: Block size for quantization (default: 64)
        compress_statistics: Whether to apply double quantization

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
        blocksize: int = 64,
        compress_statistics: bool = False,
    ):
        super().__init__()

        if quant_type not in ('nf4', 'fp4'):
            raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type}")

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.quant_type = quant_type
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics

        # Calculate packed size
        numel = out_features * in_features
        padded_numel = ((numel + blocksize - 1) // blocksize) * blocksize
        if padded_numel % 2 != 0:
            padded_numel += blocksize
        packed_size = padded_numel // 2

        # Quantized weight storage
        self.register_buffer(
            'weight',
            torch.zeros(packed_size, dtype=torch.uint8, device=device)
        )

        # QuantState will be set during quantization
        self.weight_quant_state: Optional[QuantState] = None

        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with fused 4-bit dequantization and matmul.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        if self.weight_quant_state is None:
            raise RuntimeError("Weight not quantized. Call from_linear() or load weights first.")

        # Handle batched input
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)

        # Fused 4-bit matmul (dequantize + matmul)
        output = matmul_4bit(x, self.weight, self.weight_quant_state, self.bias)

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
        blocksize: int = 64,
        compress_statistics: bool = False,
    ) -> 'Linear4bit':
        """
        Convert a regular nn.Linear to 4-bit quantized.

        Args:
            linear: Source nn.Linear layer
            device: Target device (default: same as source)
            compute_dtype: Dtype for computation (default: infer from source)
            quant_type: Quantization type ('nf4' or 'fp4')
            blocksize: Block size for quantization
            compress_statistics: Apply double quantization to absmax

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

        in_features = linear.in_features
        out_features = linear.out_features

        layer = cls(
            in_features,
            out_features,
            bias=linear.bias is not None,
            device=device,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            blocksize=blocksize,
            compress_statistics=compress_statistics,
        )

        # Quantize weights using bitsandbytes-compatible API
        weight = linear.weight.data.to(device)
        weight_packed, quant_state = quantize_4bit(
            weight,
            blocksize=blocksize,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )

        # Copy quantized weight
        layer.weight = layer.weight.new_zeros(weight_packed.numel())
        layer.weight.copy_(weight_packed)
        layer.weight_quant_state = quant_state

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
        if self.weight_quant_state is None:
            raise RuntimeError("Weight not quantized")

        return dequantize_4bit(self.weight, self.weight_quant_state)

    @property
    def quant_state(self):
        """Compatibility property for HuggingFace."""
        return self.weight_quant_state

    @property
    def device(self) -> torch.device:
        """Return the device of the quantized weights.

        This is a convenience property for LoRA adapters and other code
        that needs to create tensors on the same device as this layer.
        Standard PyTorch uses weight.device, but since our weight is a
        buffer (not a parameter), next(module.parameters()) may fail.
        """
        return self.weight.device

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, quant_type={self.quant_type}, '
            f'blocksize={self.blocksize}'
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save quantization state along with weights."""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.weight_quant_state is not None:
            destination[prefix + 'weight_quant_state'] = self.weight_quant_state.as_dict()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading from state dict, including conversion from FP16/FP32."""
        # Check for quant_state
        quant_state_key = prefix + 'weight_quant_state'
        if quant_state_key in state_dict:
            self.weight_quant_state = QuantState.from_dict(
                state_dict.pop(quant_state_key),
                device=self.weight.device
            )

        # Check if we're loading from a non-quantized state dict
        weight_key = prefix + 'weight'
        if weight_key in state_dict:
            weight_data = state_dict[weight_key]
            # If it's a full-precision weight, quantize it
            if weight_data.dtype in (torch.float16, torch.float32, torch.bfloat16):
                weight_packed, quant_state = quantize_4bit(
                    weight_data,
                    blocksize=self.blocksize,
                    compress_statistics=self.compress_statistics,
                    quant_type=self.quant_type,
                )
                state_dict[weight_key] = weight_packed
                self.weight_quant_state = quant_state

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
            if isinstance(self.quant_state, QuantState):
                return self.quant_state.shape
            return self.quant_state.get('shape', super().shape)
        return super().shape
