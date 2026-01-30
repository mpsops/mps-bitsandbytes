"""
Outlier-aware INT8 Linear layer (LLM.int8())

Implements the mixed-precision decomposition from the LLM.int8() paper:
https://arxiv.org/abs/2208.07339

Key insight: Large magnitude "outlier" features break INT8 quantization.
Solution: Identify outliers (>threshold) and compute them in FP16, rest in INT8.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Tuple

from ..functional import quantize_rowwise, dequantize_rowwise


class OutlierAwareLinear(nn.Module):
    """
    INT8 Linear with outlier-aware mixed-precision decomposition.

    For inputs with outlier features (magnitude > threshold), those features
    are computed in FP16 while the rest use INT8. This enables INT8 inference
    for large language models without quality degradation.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include a bias term
        threshold: Outlier threshold (default: 6.0, from LLM.int8 paper)
        compute_dtype: Dtype for FP16 computations

    Example:
        >>> linear = nn.Linear(4096, 4096).half().to('mps')
        >>> int8_linear = OutlierAwareLinear.from_linear(linear)
        >>> output = int8_linear(input)  # Automatic outlier handling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 6.0,
        compute_dtype=torch.float16,
        device=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.compute_dtype = compute_dtype

        # INT8 quantized weights
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # FP16 weights for outlier columns (stored sparsely)
        # These are populated during from_linear() based on weight analysis
        self.register_buffer(
            'outlier_indices',
            torch.tensor([], dtype=torch.long, device=device)
        )
        self.register_buffer(
            'outlier_weights',
            torch.zeros(out_features, 0, dtype=compute_dtype, device=device)
        )

        if bias:
            self.register_buffer(
                'bias',
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_buffer('bias', None)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with outlier-aware mixed precision.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        original_shape = x.shape[:-1]
        x = x.view(-1, self.in_features)

        if len(self.outlier_indices) > 0:
            # Mixed precision path
            output = self._forward_mixed(x)
        else:
            # Pure INT8 path (no outliers detected)
            output = self._forward_int8(x)

        # Reshape and add bias
        output = output.view(*original_shape, self.out_features)

        if self.bias is not None:
            output = output + self.bias

        return output

    def _forward_int8(self, x: Tensor) -> Tensor:
        """Pure INT8 forward pass."""
        # Quantize input
        x_int8, x_scales = quantize_rowwise(x)

        # INT8 matmul (dequantized for MPS compatibility)
        weight_fp = self.weight_int8.to(self.compute_dtype) * (self.weight_scales.unsqueeze(1) / 127.0).to(self.compute_dtype)
        x_fp = x_int8.to(self.compute_dtype) * (x_scales.unsqueeze(1) / 127.0).to(self.compute_dtype)

        return torch.mm(x_fp, weight_fp.t())

    def _forward_mixed(self, x: Tensor) -> Tensor:
        """Mixed precision forward with outlier decomposition."""
        # Create mask for non-outlier columns
        non_outlier_mask = torch.ones(self.in_features, dtype=torch.bool, device=x.device)
        non_outlier_mask[self.outlier_indices] = False

        # Split input into outlier and non-outlier parts
        x_main = x[:, non_outlier_mask]
        x_outlier = x[:, self.outlier_indices]

        # INT8 computation for main features
        x_main_int8, x_main_scales = quantize_rowwise(x_main)

        # Get non-outlier weights
        weight_main_int8 = self.weight_int8[:, non_outlier_mask]
        weight_main_fp = weight_main_int8.to(self.compute_dtype) * (self.weight_scales.unsqueeze(1) / 127.0).to(self.compute_dtype)
        x_main_fp = x_main_int8.to(self.compute_dtype) * (x_main_scales.unsqueeze(1) / 127.0).to(self.compute_dtype)

        output_main = torch.mm(x_main_fp, weight_main_fp.t())

        # FP16 computation for outlier features
        output_outlier = torch.mm(x_outlier.to(self.compute_dtype), self.outlier_weights.t())

        return output_main + output_outlier

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        threshold: float = 6.0,
        device=None,
    ) -> 'OutlierAwareLinear':
        """
        Convert nn.Linear to outlier-aware INT8.

        Analyzes weights to identify outlier columns and stores them in FP16.

        Args:
            linear: Source nn.Linear layer
            threshold: Outlier threshold (features with magnitude > threshold)
            device: Target device

        Returns:
            OutlierAwareLinear with mixed-precision weights
        """
        if device is None:
            device = linear.weight.device

        dtype = linear.weight.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            dtype = torch.float16

        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            compute_dtype=dtype,
            device=device,
        )

        weight = linear.weight.data.to(device)

        # Identify outlier columns based on weight magnitude
        # Outliers are columns where any weight exceeds threshold * mean_abs
        col_max = weight.abs().max(dim=0).values
        mean_abs = weight.abs().mean()
        outlier_mask = col_max > (threshold * mean_abs)
        outlier_indices = torch.where(outlier_mask)[0]

        if len(outlier_indices) > 0:
            # Store outlier weights in FP16
            layer.outlier_indices = outlier_indices
            layer.outlier_weights = weight[:, outlier_indices].to(dtype)

            # Zero out outlier columns before INT8 quantization
            weight_for_int8 = weight.clone()
            weight_for_int8[:, outlier_indices] = 0
        else:
            weight_for_int8 = weight

        # Quantize remaining weights to INT8
        weight_int8, weight_scales = quantize_rowwise(weight_for_int8)
        layer.weight_int8.copy_(weight_int8)
        layer.weight_scales.copy_(weight_scales)

        if linear.bias is not None:
            layer.bias.copy_(linear.bias.data.to(dtype))

        return layer

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, threshold={self.threshold}, '
            f'outliers={len(self.outlier_indices)}'
        )
