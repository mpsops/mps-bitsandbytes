"""
MPS BitsAndBytes - 4-bit and 8-bit quantization for PyTorch on Apple Silicon

Enables QLoRA training and memory-efficient inference on MPS devices with
real NF4/FP4 4-bit and FP8/INT8 8-bit quantization using Metal GPU acceleration.

Features:
- NF4 4-bit quantization (~4x memory reduction, optimal for neural network weights)
- FP4 4-bit quantization (alternative with better dynamic range)
- FP8 E4M3 8-bit quantization (better precision than INT8)
- INT8 8-bit quantization (~2x memory reduction)
- Double quantization support (~10% extra savings)
- Metal GPU kernels for fused dequant+matmul
- HuggingFace transformers compatible API
- QLoRA training support

Example:
    >>> import torch
    >>> from mps_bitsandbytes import Linear4bit, quantize_nf4
    >>>
    >>> # Quantize a linear layer
    >>> linear = torch.nn.Linear(4096, 4096).half().to('mps')
    >>> linear_4bit = Linear4bit.from_linear(linear)
    >>>
    >>> # Or use the functional API
    >>> weight = torch.randn(4096, 4096).half().to('mps')
    >>> packed, absmax = quantize_nf4(weight)

HuggingFace Integration:
    >>> from mps_bitsandbytes import BitsAndBytesConfig, quantize_model
    >>> from transformers import AutoModelForCausalLM
    >>>
    >>> config = BitsAndBytesConfig(
    ...     load_in_4bit=True,
    ...     bnb_4bit_quant_type="nf4",
    ...     bnb_4bit_compute_dtype=torch.float16,
    ... )
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("model_name", torch_dtype=torch.float16)
    >>> model = quantize_model(model, quantization_config=config, device='mps')
"""

import torch as _torch

__version__ = "0.2.0"

# Core functional API
from .functional import (
    # NF4 (4-bit NormalFloat - best for neural network weights)
    quantize_nf4,
    dequantize_nf4,
    matmul_nf4,
    NF4_CODEBOOK,
    # FP4 (4-bit Floating Point - better dynamic range)
    quantize_fp4,
    dequantize_fp4,
    matmul_fp4,
    FP4_CODEBOOK,
    # FP8 E4M3 (8-bit - better precision than INT8)
    quantize_fp8_e4m3,
    dequantize_fp8_e4m3,
    matmul_fp8_e4m3,
    # INT8 (8-bit integer)
    quantize_rowwise,
    dequantize_rowwise,
    matmul_int8,
    # Double quantization (extra ~10% savings)
    double_quant,
    dequant_absmax,
)

# Neural network modules
from .nn import Linear4bit, Linear8bit, LinearFP8

# HuggingFace integration
from .integration import (
    BitsAndBytesConfig,
    quantize_model,
    replace_linear_with_4bit,
    replace_linear_with_8bit,
    get_memory_footprint,
)


def is_available() -> bool:
    """Check if MPS is available for quantized operations."""
    return _torch.backends.mps.is_available()


def has_native_kernels() -> bool:
    """Check if native Metal kernels are available."""
    try:
        from . import _C
        return True
    except ImportError:
        return False




# All public exports
__all__ = [
    # Version
    '__version__',

    # Availability checks
    'is_available',
    'has_native_kernels',

    # NF4 functional API (4-bit, optimal for NN weights)
    'quantize_nf4',
    'dequantize_nf4',
    'matmul_nf4',
    'NF4_CODEBOOK',

    # FP4 functional API (4-bit, better dynamic range)
    'quantize_fp4',
    'dequantize_fp4',
    'matmul_fp4',
    'FP4_CODEBOOK',

    # FP8 functional API (8-bit floating point)
    'quantize_fp8_e4m3',
    'dequantize_fp8_e4m3',
    'matmul_fp8_e4m3',

    # INT8 functional API
    'quantize_rowwise',
    'dequantize_rowwise',
    'matmul_int8',

    # Double quantization
    'double_quant',
    'dequant_absmax',

    # Neural network modules
    'Linear4bit',
    'Linear8bit',
    'LinearFP8',

    # HuggingFace integration
    'BitsAndBytesConfig',
    'quantize_model',
    'replace_linear_with_4bit',
    'replace_linear_with_8bit',
    'get_memory_footprint',
]
