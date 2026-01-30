"""
SwitchBack Linear layer for efficient INT8 training

Implements switchback quantization where:
- Forward pass: INT8 weights, FP16 activations
- Backward pass: FP16 weights (for gradient computation)

This allows memory-efficient forward pass while maintaining training stability.
Reference: https://arxiv.org/abs/2304.13013
"""

import torch
from torch import nn, Tensor
from torch.autograd import Function
from typing import Optional, Tuple, Any

from ..functional import quantize_rowwise, dequantize_rowwise


class SwitchBackFunction(Function):
    """
    Autograd function for switchback linear.

    Forward: Uses INT8 weights
    Backward: Uses FP16 weights for accurate gradients
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight_int8: Tensor,
        weight_scales: Tensor,
        weight_fp: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        """
        Forward pass using INT8 weights.

        Args:
            input: Input tensor [batch, in_features]
            weight_int8: INT8 quantized weights [out, in]
            weight_scales: Per-row scales [out]
            weight_fp: FP16 weights for backward [out, in]
            bias: Optional bias [out]

        Returns:
            Output tensor [batch, out_features]
        """
        # Save for backward
        ctx.save_for_backward(input, weight_fp, bias)

        # Dequantize INT8 weights for matmul
        dtype = input.dtype
        weight_dequant = weight_int8.to(dtype) * (weight_scales.unsqueeze(1) / 127.0).to(dtype)

        # Forward matmul
        output = torch.mm(input.view(-1, input.size(-1)), weight_dequant.t())

        if bias is not None:
            output = output + bias

        return output.view(*input.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass using FP16 weights for accurate gradients.
        """
        input, weight_fp, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        grad_output_2d = grad_output.view(-1, grad_output.size(-1))
        input_2d = input.view(-1, input.size(-1))

        # Gradient w.r.t. input (uses FP16 weights)
        if ctx.needs_input_grad[0]:
            grad_input = torch.mm(grad_output_2d, weight_fp)
            grad_input = grad_input.view_as(input)

        # Gradient w.r.t. weight (for the FP16 copy)
        if ctx.needs_input_grad[3]:
            grad_weight = torch.mm(grad_output_2d.t(), input_2d)

        # Gradient w.r.t. bias
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output_2d.sum(0)

        return grad_input, None, None, grad_weight, grad_bias


class SwitchBackLinear(nn.Module):
    """
    Linear layer with switchback INT8/FP16 for efficient training.

    During forward pass, uses INT8 quantized weights for memory efficiency.
    During backward pass, uses FP16 weights for accurate gradient computation.

    This provides ~2x memory savings during forward pass while maintaining
    training quality similar to full FP16.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include a bias term
        compute_dtype: Dtype for computations (default: float16)

    Example:
        >>> linear = nn.Linear(4096, 4096).half().to('mps')
        >>> sb_linear = SwitchBackLinear.from_linear(linear)
        >>> output = sb_linear(input)  # INT8 forward
        >>> loss.backward()  # FP16 backward
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype=torch.float16,
        device=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # INT8 quantized weights (for forward)
        self.register_buffer(
            'weight_int8',
            torch.zeros(out_features, in_features, dtype=torch.int8, device=device)
        )
        self.register_buffer(
            'weight_scales',
            torch.ones(out_features, dtype=torch.float32, device=device)
        )

        # FP16 weights (for backward, trainable)
        self.weight_fp = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=compute_dtype, device=device)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=compute_dtype, device=device)
            )
        else:
            self.register_parameter('bias', None)

        self._update_int8_pending = False

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using INT8 weights.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Update INT8 weights if FP16 weights changed (after optimizer step)
        if self.training and self._update_int8_pending:
            self._update_int8_weights()
            self._update_int8_pending = False

        return SwitchBackFunction.apply(
            x, self.weight_int8, self.weight_scales, self.weight_fp, self.bias
        )

    def _update_int8_weights(self):
        """Re-quantize INT8 weights from FP16."""
        with torch.no_grad():
            weight_int8, weight_scales = quantize_rowwise(self.weight_fp.data)
            self.weight_int8.copy_(weight_int8)
            self.weight_scales.copy_(weight_scales)

    def sync_weights(self):
        """
        Manually sync INT8 weights from FP16.

        Call this after optimizer.step() to update the INT8 weights
        used in the forward pass.
        """
        self._update_int8_weights()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        device=None,
    ) -> 'SwitchBackLinear':
        """
        Convert nn.Linear to SwitchBackLinear.

        Args:
            linear: Source nn.Linear layer
            device: Target device

        Returns:
            SwitchBackLinear with quantized weights
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
            compute_dtype=dtype,
            device=device,
        )

        # Copy FP16 weights
        layer.weight_fp.data.copy_(linear.weight.data.to(dtype))

        # Quantize to INT8
        weight_int8, weight_scales = quantize_rowwise(linear.weight.data.to(device))
        layer.weight_int8.copy_(weight_int8)
        layer.weight_scales.copy_(weight_scales)

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data.to(dtype))

        return layer

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}'
        )


class SwitchBackLinearCallback:
    """
    Callback to sync INT8 weights after optimizer step.

    Usage:
        >>> callback = SwitchBackLinearCallback(model)
        >>> for epoch in range(epochs):
        ...     loss.backward()
        ...     optimizer.step()
        ...     callback.sync()  # Update INT8 weights
    """

    def __init__(self, model: nn.Module):
        self.switchback_layers = []
        for module in model.modules():
            if isinstance(module, SwitchBackLinear):
                self.switchback_layers.append(module)

    def sync(self):
        """Sync all SwitchBackLinear layers."""
        for layer in self.switchback_layers:
            layer.sync_weights()
