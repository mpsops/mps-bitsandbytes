"""
MPS BitsAndBytes - Functional API

Core functions for quantizing, dequantizing, and computing with quantized tensors.
Supports: INT8, NF4, FP4, FP8 (E4M3), and Double Quantization.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional

# =============================================================================
# Codebooks
# =============================================================================

# NF4: 16 values optimized for normal distribution
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)

# FP4: Normalized floating point distribution
FP4_CODEBOOK = torch.tensor([
    0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0,
    -0.0, -0.0625, -0.125, -0.25, -0.375, -0.5, -0.75, -1.0
], dtype=torch.float32)


def _try_load_native():
    """Try to load native C++ extension."""
    try:
        from . import _C
        return _C
    except ImportError:
        return None


# =============================================================================
# NF4 (4-bit NormalFloat) - Best for neural network weights
# =============================================================================

def quantize_nf4(tensor: Tensor, block_size: int = 64) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to NF4 (4-bit NormalFloat) format.

    NF4 is optimized for normally distributed weights (like neural networks).
    Achieves ~4x memory reduction with minimal accuracy loss.

    Args:
        tensor: Input [rows, cols], cols must be even
        block_size: Elements per quantization block (default: 64)

    Returns:
        packed: [rows, cols//2] uint8 (two 4-bit values per byte)
        absmax: [rows, num_blocks] float32 scales
    """
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.dim()}D")
    if tensor.size(1) % 2 != 0:
        raise ValueError(f"cols must be even, got {tensor.size(1)}")

    _C = _try_load_native()
    if _C is not None and tensor.device.type == 'mps':
        return _C.quantize_nf4(tensor, block_size)
    return _quantize_4bit_python(tensor, block_size, NF4_CODEBOOK)


def dequantize_nf4(packed: Tensor, absmax: Tensor, block_size: int = 64,
                   dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize NF4 tensor back to floating point."""
    _C = _try_load_native()
    if _C is not None and packed.device.type == 'mps':
        return _C.dequantize_nf4(packed, absmax, block_size, dtype)
    return _dequantize_4bit_python(packed, absmax, block_size, NF4_CODEBOOK, dtype)


def matmul_nf4(input: Tensor, weight_packed: Tensor, weight_absmax: Tensor,
               bias: Optional[Tensor] = None, block_size: int = 64,
               dtype: torch.dtype = torch.float16) -> Tensor:
    """Matrix multiplication with NF4-quantized weights."""
    _C = _try_load_native()
    if _C is not None and input.device.type == 'mps':
        return _C.matmul_nf4(input, weight_packed, weight_absmax, bias, block_size, dtype)
    return _matmul_4bit_python(input, weight_packed, weight_absmax, bias, block_size, NF4_CODEBOOK, dtype)


# =============================================================================
# FP4 (4-bit Floating Point)
# =============================================================================

def quantize_fp4(tensor: Tensor, block_size: int = 64) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to FP4 (4-bit floating point) format.

    FP4 has better dynamic range than NF4 but may be less optimal
    for normally distributed weights.

    Args:
        tensor: Input [rows, cols], cols must be even
        block_size: Elements per quantization block (default: 64)

    Returns:
        packed: [rows, cols//2] uint8
        absmax: [rows, num_blocks] float32 scales
    """
    if tensor.dim() != 2 or tensor.size(1) % 2 != 0:
        raise ValueError("Input must be 2D with even cols")

    _C = _try_load_native()
    if _C is not None and tensor.device.type == 'mps':
        return _C.quantize_fp4(tensor, block_size)
    return _quantize_4bit_python(tensor, block_size, FP4_CODEBOOK)


def dequantize_fp4(packed: Tensor, absmax: Tensor, block_size: int = 64,
                   dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize FP4 tensor back to floating point."""
    _C = _try_load_native()
    if _C is not None and packed.device.type == 'mps':
        return _C.dequantize_fp4(packed, absmax, block_size, dtype)
    return _dequantize_4bit_python(packed, absmax, block_size, FP4_CODEBOOK, dtype)


def matmul_fp4(input: Tensor, weight_packed: Tensor, weight_absmax: Tensor,
               bias: Optional[Tensor] = None, block_size: int = 64,
               dtype: torch.dtype = torch.float16) -> Tensor:
    """Matrix multiplication with FP4-quantized weights."""
    _C = _try_load_native()
    if _C is not None and input.device.type == 'mps':
        return _C.matmul_fp4(input, weight_packed, weight_absmax, bias, block_size, dtype)
    return _matmul_4bit_python(input, weight_packed, weight_absmax, bias, block_size, FP4_CODEBOOK, dtype)


# =============================================================================
# FP8 (8-bit Floating Point) - E4M3 format
# =============================================================================

def quantize_fp8_e4m3(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to FP8 E4M3 format.

    FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) offers better precision
    than INT8 with similar memory footprint. Range: ±448.

    Args:
        tensor: Input [rows, cols]

    Returns:
        quantized: [rows, cols] uint8 (FP8 encoded)
        scales: [rows] float32 row-wise scales
    """
    if tensor.dim() != 2:
        raise ValueError("Input must be 2D")

    _C = _try_load_native()
    if _C is not None and tensor.device.type == 'mps':
        return _C.quantize_fp8_e4m3(tensor)
    return _quantize_fp8_e4m3_python(tensor)


def dequantize_fp8_e4m3(quantized: Tensor, scales: Tensor,
                        dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize FP8 E4M3 tensor."""
    _C = _try_load_native()
    if _C is not None and quantized.device.type == 'mps':
        return _C.dequantize_fp8_e4m3(quantized, scales, dtype)
    return _dequantize_fp8_e4m3_python(quantized, scales, dtype)


def matmul_fp8_e4m3(input: Tensor, weight: Tensor, weight_scales: Tensor,
                    bias: Optional[Tensor] = None,
                    dtype: torch.dtype = torch.float16) -> Tensor:
    """Matrix multiplication with FP8 E4M3 weights."""
    _C = _try_load_native()
    if _C is not None and input.device.type == 'mps':
        return _C.matmul_fp8_e4m3(input, weight, weight_scales, bias, dtype)
    return _matmul_fp8_python(input, weight, weight_scales, bias, dtype)


# =============================================================================
# INT8 (8-bit Integer)
# =============================================================================

def quantize_rowwise(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to INT8 using row-wise absmax scaling.

    Args:
        tensor: Input [..., K]

    Returns:
        quantized: INT8 tensor, same shape
        scales: Float scales [...] per row
    """
    orig_shape = tensor.shape
    tensor_2d = tensor.view(-1, tensor.shape[-1])

    absmax = tensor_2d.abs().max(dim=-1).values
    scales = absmax.clamp(min=1e-8)

    quantized = torch.clamp(
        torch.round(tensor_2d * (127.0 / scales.unsqueeze(-1))),
        -127, 127
    ).to(torch.int8)

    return quantized.view(orig_shape), scales


def dequantize_rowwise(quantized: Tensor, scales: Tensor,
                       dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize INT8 tensor."""
    orig_shape = quantized.shape
    quantized_2d = quantized.view(-1, quantized.shape[-1])
    scales_2d = scales.view(-1)

    dequantized = (quantized_2d.to(dtype) * (scales_2d.unsqueeze(-1) / 127.0).to(dtype))
    return dequantized.view(orig_shape)


def matmul_int8(A: Tensor, B: Tensor, A_scales: Tensor, B_scales: Tensor,
                dtype: torch.dtype = torch.float16) -> Tensor:
    """INT8 matmul with fused dequantization."""
    _C = _try_load_native()
    if _C is not None and A.device.type == 'mps':
        return _C.matmul_int8(A, B, A_scales, B_scales, dtype)

    A_dequant = dequantize_rowwise(A, A_scales, dtype)
    B_dequant = dequantize_rowwise(B.T, B_scales, dtype).T
    return torch.matmul(A_dequant, B_dequant)


# =============================================================================
# Double Quantization
# =============================================================================

def double_quant(absmax: Tensor, double_quant_block: int = 256) -> Tuple[Tensor, Tensor]:
    """
    Apply double quantization to absmax scales.

    Quantizes the absmax values themselves with INT8, providing
    additional ~10% memory savings on top of 4-bit quantization.

    Args:
        absmax: [rows, num_blocks] float32 absmax values
        double_quant_block: Block size for double quantization

    Returns:
        absmax_quant: [rows, num_blocks] uint8 quantized absmax
        absmax_scales: [rows, dq_blocks] float32 scales for absmax
    """
    if absmax.dim() != 2:
        raise ValueError("absmax must be 2D")

    _C = _try_load_native()
    if _C is not None and absmax.device.type == 'mps':
        return _C.double_quant(absmax, double_quant_block)
    return _double_quant_python(absmax, double_quant_block)


def dequant_absmax(absmax_quant: Tensor, absmax_scales: Tensor,
                   double_quant_block: int = 256) -> Tensor:
    """Dequantize double-quantized absmax values."""
    rows, num_blocks = absmax_quant.shape
    dq_blocks = absmax_scales.size(1)

    absmax = torch.zeros(rows, num_blocks, dtype=torch.float32, device=absmax_quant.device)

    for dqb in range(dq_blocks):
        start = dqb * double_quant_block
        end = min(start + double_quant_block, num_blocks)
        scale = absmax_scales[:, dqb:dqb+1]
        absmax[:, start:end] = absmax_quant[:, start:end].float() * scale

    return absmax


# =============================================================================
# Python Fallback Implementations
# =============================================================================

def _quantize_4bit_python(tensor: Tensor, block_size: int, codebook: Tensor) -> Tuple[Tensor, Tensor]:
    """Generic 4-bit quantization with given codebook."""
    device = tensor.device
    rows, cols = tensor.shape
    num_blocks = (cols + block_size - 1) // block_size
    tensor_f32 = tensor.float()
    cb = codebook.to(device)

    # Compute absmax per block
    absmax_list = []
    for block in range(num_blocks):
        start, end = block * block_size, min((block + 1) * block_size, cols)
        block_absmax = tensor_f32[:, start:end].abs().max(dim=1).values.clamp(min=1e-8)
        absmax_list.append(block_absmax)
    absmax = torch.stack(absmax_list, dim=1)

    # Quantize
    packed = torch.zeros(rows, cols // 2, dtype=torch.uint8, device=device)
    for block in range(num_blocks):
        start, end = block * block_size, min((block + 1) * block_size, cols)
        block_scale = absmax[:, block:block+1]

        for i in range(start, end, 2):
            if i >= cols:
                break
            v0 = tensor_f32[:, i] / block_scale.squeeze(1)
            v1 = tensor_f32[:, i + 1] / block_scale.squeeze(1) if i + 1 < cols else torch.zeros_like(v0)

            idx0 = (v0.unsqueeze(1) - cb.unsqueeze(0)).abs().argmin(dim=1)
            idx1 = (v1.unsqueeze(1) - cb.unsqueeze(0)).abs().argmin(dim=1)
            packed[:, i // 2] = (idx0 | (idx1 << 4)).to(torch.uint8)

    return packed, absmax


def _dequantize_4bit_python(packed: Tensor, absmax: Tensor, block_size: int,
                             codebook: Tensor, dtype: torch.dtype) -> Tensor:
    """Generic 4-bit dequantization."""
    device = packed.device
    rows, cols = packed.size(0), packed.size(1) * 2
    cb = codebook.to(device)
    output = torch.zeros(rows, cols, dtype=dtype, device=device)

    for col in range(cols):
        packed_idx = col // 2
        position = col % 2
        block_idx = col // block_size

        packed_val = packed[:, packed_idx]
        idx = (packed_val & 0x0F) if position == 0 else ((packed_val >> 4) & 0x0F)
        scale = absmax[:, block_idx]
        output[:, col] = (cb[idx.long()] * scale).to(dtype)

    return output


def _matmul_4bit_python(input: Tensor, weight_packed: Tensor, weight_absmax: Tensor,
                         bias: Optional[Tensor], block_size: int,
                         codebook: Tensor, dtype: torch.dtype) -> Tensor:
    """Generic 4-bit matmul fallback."""
    weight = _dequantize_4bit_python(weight_packed, weight_absmax, block_size, codebook, dtype)

    is_1d = input.dim() == 1
    if is_1d:
        input = input.unsqueeze(0)

    output = torch.nn.functional.linear(input.to(dtype), weight, bias)
    return output.squeeze(0) if is_1d else output


def _fp8_e4m3_to_float(fp8: int) -> float:
    """Convert single FP8 E4M3 value to float."""
    sign = (fp8 >> 7) & 0x1
    exp = (fp8 >> 3) & 0xF
    mant = fp8 & 0x7

    if exp == 0:
        result = 0.0 if mant == 0 else (mant / 8.0) * (2 ** -6)
    elif exp == 15 and mant == 7:
        result = float('nan')
    else:
        result = (1.0 + mant / 8.0) * (2 ** (exp - 7))

    return -result if sign else result


def _float_to_fp8_e4m3(val: float) -> int:
    """Convert float to FP8 E4M3."""
    import math
    if math.isnan(val):
        return 0x7F

    sign = 1 if val < 0 else 0
    val = abs(val)
    val = min(val, 448.0)

    if val == 0:
        return sign << 7

    exp = int(math.floor(math.log2(val)))
    mant = val / (2 ** exp) - 1.0
    biased_exp = exp + 7

    if biased_exp <= 0:
        return sign << 7
    elif biased_exp >= 15:
        return (sign << 7) | (14 << 3) | 7
    else:
        mant_bits = min(int(mant * 8 + 0.5), 7)
        return (sign << 7) | (biased_exp << 3) | mant_bits


def _quantize_fp8_e4m3_python(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """Python FP8 E4M3 quantization."""
    rows, cols = tensor.shape
    tensor_f32 = tensor.float()

    # Row-wise scaling
    absmax = tensor_f32.abs().max(dim=1).values
    scales = (absmax / 448.0).clamp(min=1e-12)

    # Quantize
    output = torch.zeros(rows, cols, dtype=torch.uint8, device=tensor.device)
    for r in range(rows):
        for c in range(cols):
            val = tensor_f32[r, c].item() / scales[r].item()
            output[r, c] = _float_to_fp8_e4m3(val)

    return output, scales


def _dequantize_fp8_e4m3_python(quantized: Tensor, scales: Tensor,
                                 dtype: torch.dtype) -> Tensor:
    """Python FP8 E4M3 dequantization."""
    rows, cols = quantized.shape
    output = torch.zeros(rows, cols, dtype=dtype, device=quantized.device)

    for r in range(rows):
        scale = scales[r].item()
        for c in range(cols):
            fp8_val = quantized[r, c].item()
            output[r, c] = _fp8_e4m3_to_float(fp8_val) * scale

    return output


def _matmul_fp8_python(input: Tensor, weight: Tensor, weight_scales: Tensor,
                       bias: Optional[Tensor], dtype: torch.dtype) -> Tensor:
    """Python FP8 matmul fallback."""
    weight_dequant = _dequantize_fp8_e4m3_python(weight, weight_scales, dtype)

    is_1d = input.dim() == 1
    if is_1d:
        input = input.unsqueeze(0)

    output = torch.nn.functional.linear(input.to(dtype), weight_dequant, bias)
    return output.squeeze(0) if is_1d else output


def _double_quant_python(absmax: Tensor, double_quant_block: int) -> Tuple[Tensor, Tensor]:
    """Python double quantization."""
    rows, num_blocks = absmax.shape
    dq_blocks = (num_blocks + double_quant_block - 1) // double_quant_block

    absmax_quant = torch.zeros(rows, num_blocks, dtype=torch.uint8, device=absmax.device)
    absmax_scales = torch.zeros(rows, dq_blocks, dtype=torch.float32, device=absmax.device)

    for dqb in range(dq_blocks):
        start = dqb * double_quant_block
        end = min(start + double_quant_block, num_blocks)

        block_max = absmax[:, start:end].abs().max(dim=1).values
        scale = (block_max / 127.0).clamp(min=1e-12)
        absmax_scales[:, dqb] = scale

        for i in range(start, end):
            absmax_quant[:, i] = (absmax[:, i] / scale).clamp(0, 255).round().to(torch.uint8)

    return absmax_quant, absmax_scales
