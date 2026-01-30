"""
LinearFP8 - 8-bit floating point quantized linear layer for MPS

FP8 E4M3 provides better precision than INT8 with the same memory footprint.
Range: ±448, suitable for neural network weights.
"""

import torch
from torch import nn, Tensor
from typing import Optional

from ..functional import quantize_fp8_e4m3, dequantize_fp8_e4m3, matmul_fp8_e4m3


class LinearFP8(nn.Module):
    """
    8-bit floating point (FP8 E4M3) quantized linear layer.

    FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    Range: ±448, better suited for the distribution of neural network weights
    compared to INT8 which has uniform quantization.

    Provides ~2x memory reduction compared to FP16 with better precision
    characteristics than INT8 for typical weight distributions.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: True)
        device: Target device
        compute_dtype: Dtype for computation (torch.float16 or torch.bfloat16)

    Example:
        >>> linear = LinearFP8(1024, 4096)
        >>> linear_fp8 = LinearFP8.from_linear(pretrained_linear)
        >>> output = linear_fp8(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        compute_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # FP8 quantized weight storage
        self.register_buffer(
            'weight_fp8',
            torch.zeros(out_features, in_features, dtype=torch.uint8, device=device)
        )
        # Row-wise scales for FP8
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with fused FP8 dequantization and matmul.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Handle batched input
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)

        # Fused FP8 matmul
        output = matmul_fp8_e4m3(
            x,
            self.weight_fp8,
            self.weight_scales,
            self.bias,
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
    ) -> 'LinearFP8':
        """
        Convert a regular nn.Linear to FP8 quantized.

        Args:
            linear: Source nn.Linear layer
            device: Target device (default: same as source)
            compute_dtype: Dtype for computation (default: infer from source)

        Returns:
            LinearFP8 layer with quantized weights

        Example:
            >>> linear_fp16 = nn.Linear(1024, 4096).half().to('mps')
            >>> linear_fp8 = LinearFP8.from_linear(linear_fp16)
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
            compute_dtype=compute_dtype,
        )

        # Quantize weights to FP8
        weight_fp8, weight_scales = quantize_fp8_e4m3(linear.weight.data.to(device))
        layer.weight_fp8.copy_(weight_fp8)
        layer.weight_scales.copy_(weight_scales)

        # Copy bias
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.to(compute_dtype).to(device))

        return layer

    def dequantize(self) -> Tensor:
        """
        Dequantize weights back to floating point.

        Returns:
            Dequantized weight tensor [out_features, in_features]
        """
        return dequantize_fp8_e4m3(
            self.weight_fp8,
            self.weight_scales,
            self.compute_dtype
        )

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, quant_type=fp8_e4m3'
        )
