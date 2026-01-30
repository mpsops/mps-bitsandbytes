"""
MPS BitsAndBytes - 8-bit quantization for PyTorch on Apple Silicon

Enables QLoRA training and memory-efficient inference on MPS devices.

Key insight: Store weights in int8 (50% memory savings), dequantize to fp16
for fast AMX-accelerated matmul. Best of both worlds!
"""

import torch
from torch import nn
from typing import Optional, Tuple
import os

__version__ = "0.1.2"


def is_available() -> bool:
    """Check if MPS is available."""
    return torch.backends.mps.is_available()


def quantize_rowwise(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to int8 using row-wise absmax scaling.

    Args:
        tensor: Input tensor of shape [..., K]

    Returns:
        quantized: Int8 tensor of same shape
        scales: Float tensor of shape [...] containing absmax per row
    """
    # Flatten to 2D for quantization
    orig_shape = tensor.shape
    tensor_2d = tensor.view(-1, tensor.shape[-1])

    # Compute row-wise absmax
    absmax = tensor_2d.abs().max(dim=-1).values
    scales = absmax.clamp(min=1e-8)  # Avoid division by zero

    # Quantize
    quantized = torch.clamp(
        torch.round(tensor_2d * (127.0 / scales.unsqueeze(-1))),
        -127, 127
    ).to(torch.int8)

    return quantized.view(orig_shape), scales


def dequantize_rowwise(quantized: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    Dequantize an int8 tensor back to float.

    Args:
        quantized: Int8 tensor
        scales: Absmax scales from quantization
        dtype: Output dtype (torch.float16 or torch.bfloat16)

    Returns:
        Dequantized tensor in specified dtype
    """
    orig_shape = quantized.shape
    quantized_2d = quantized.view(-1, quantized.shape[-1])
    scales_2d = scales.view(-1)

    dequantized = (quantized_2d.to(dtype) * (scales_2d.unsqueeze(-1) / 127.0).to(dtype))
    return dequantized.view(orig_shape)


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
            torch.ones(out_features, dtype=torch.float32, device=device)  # Store scales in fp32 for precision
        )

        # Optional bias (stored in compute_dtype)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype, device=device))
        else:
            self.register_parameter('bias', None)

        # Cache for dequantized weights
        self._weight_cache: Optional[torch.Tensor] = None

    def _get_weight(self) -> torch.Tensor:
        """Get weight in compute_dtype, using cache if enabled."""
        if self.use_cache and self._weight_cache is not None:
            return self._weight_cache

        # Dequantize to compute_dtype
        weight = dequantize_rowwise(self.weight_int8, self.weight_scales, dtype=self.compute_dtype)

        if self.use_cache:
            self._weight_cache = weight

        return weight

    def clear_cache(self):
        """Clear the weight cache to free memory."""
        self._weight_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight in compute_dtype (cached or dequantized)
        weight = self._get_weight()

        # Fast matmul (AMX accelerated on Apple Silicon)
        return torch.nn.functional.linear(x, weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, device=None, use_cache: bool = True, compute_dtype: torch.dtype = None) -> 'Linear8bit':
        """
        Convert a regular linear layer to 8-bit.

        Args:
            linear: Source nn.Linear layer
            device: Target device (default: same as source)
            use_cache: Cache dequantized weights for speed
            compute_dtype: Dtype for dequantized weights (default: infer from input, fallback to fp16)

        Returns:
            Linear8bit layer with quantized weights
        """
        if device is None:
            device = linear.weight.device

        # Infer compute_dtype from source if not specified
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

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def quantize_model(model: nn.Module, device='mps', compute_dtype: torch.dtype = None) -> nn.Module:
    """
    Convert all Linear layers in a model to 8-bit.

    Args:
        model: PyTorch model
        device: Target device
        compute_dtype: Dtype for dequantized weights (torch.float16 or torch.bfloat16)
                       If None, infers from each layer's weight dtype

    Returns:
        Model with Linear layers replaced by Linear8bit
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with 8-bit version
            setattr(model, name, Linear8bit.from_linear(module, device=device, compute_dtype=compute_dtype))
        else:
            # Recurse
            quantize_model(module, device=device, compute_dtype=compute_dtype)

    return model


def get_memory_footprint(model: nn.Module) -> dict:
    """
    Calculate memory footprint of a model.

    Returns dict with:
        - total_params: Total number of parameters
        - fp16_size_gb: Size if all params were fp16
        - actual_size_gb: Actual size with mixed precision
        - savings_gb: Memory saved by quantization
    """
    total_bytes = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    for name, buf in model.named_buffers():
        total_params += buf.numel()
        total_bytes += buf.numel() * buf.element_size()

    fp16_size = total_params * 2 / 1e9  # If all were fp16
    actual_size = total_bytes / 1e9

    return {
        'total_params': total_params,
        'fp16_size_gb': fp16_size,
        'actual_size_gb': actual_size,
        'savings_gb': fp16_size - actual_size,
        'savings_pct': (1 - actual_size / fp16_size) * 100 if fp16_size > 0 else 0,
    }


# For compatibility with bitsandbytes API
Linear8bitLt = Linear8bit
