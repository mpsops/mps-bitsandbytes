"""
Linear8bit - 8-bit quantized linear layer for MPS

Provides ~2x memory reduction compared to FP16 weights.
Compatible with QLoRA training on Apple Silicon.
"""

import torch
from torch import nn, Tensor
from typing import Optional

from ..functional import quantize_rowwise, dequantize_rowwise


class Linear8bit(nn.Module):
    """
    8-bit quantized linear layer for memory-efficient inference and QLoRA training.

    Stores weights in int8 (50% memory savings), dequantizes to fp16/bf16 for
    fast AMX-accelerated matmul on Apple Silicon.

    For QLoRA: Add LoRA adapters on top of this layer. The int8 weights
    stay frozen while LoRA trains in fp16/bf16.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        device: Target device
        use_cache: If True, cache dequantized weights (faster but uses more memory)
        compute_dtype: Dtype for dequantized weights (torch.float16 or torch.bfloat16)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        use_cache: bool = True,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cache = use_cache
        self.compute_dtype = compute_dtype

        # Quantized weight storage (int8)
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # Optional bias (stored in compute_dtype)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)

        # Cache for dequantized weights
        self._weight_cache: Optional[Tensor] = None

    def _get_weight(self) -> Tensor:
        """Get weight in compute_dtype, using cache if enabled."""
        if self.use_cache and self._weight_cache is not None:
            return self._weight_cache

        # Dequantize to compute_dtype
        weight = dequantize_rowwise(
            self.weight_int8,
            self.weight_scales,
            dtype=self.compute_dtype
        )

        if self.use_cache:
            self._weight_cache = weight

        return weight

    def clear_cache(self):
        """Clear the weight cache to free memory."""
        self._weight_cache = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with dequantized weights.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        weight = self._get_weight()
        return torch.nn.functional.linear(x, weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        device=None,
        use_cache: bool = True,
        compute_dtype: Optional[torch.dtype] = None
    ) -> 'Linear8bit':
        """
        Convert a regular linear layer to 8-bit.

        Args:
            linear: Source nn.Linear layer
            device: Target device (default: same as source)
            use_cache: Cache dequantized weights for speed
            compute_dtype: Dtype for dequantized weights

        Returns:
            Linear8bit layer with quantized weights
        """
        if device is None:
            device = linear.weight.device

        if compute_dtype is None:
            if linear.weight.dtype == torch.bfloat16:
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16

        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=device,
            use_cache=use_cache,
            compute_dtype=compute_dtype,
        )

        # Quantize weights
        weight_int8, weight_scales = quantize_rowwise(linear.weight.data.to(device))
        layer.weight_int8.copy_(weight_int8)
        layer.weight_scales.copy_(weight_scales.to(torch.float32))

        # Copy bias
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.to(compute_dtype).to(device))

        return layer

    @property
    def device(self) -> torch.device:
        """Return the device of the quantized weights.

        Convenience property for LoRA adapters and other code that needs
        to create tensors on the same device as this layer.
        """
        return self.weight_int8.device

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}'
        )
