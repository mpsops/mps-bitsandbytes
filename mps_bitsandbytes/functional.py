"""
MPS BitsAndBytes - Functional API

Core functions for quantizing, dequantizing, and computing with quantized tensors.
API compatible with bitsandbytes for drop-in replacement.

Supports: INT8, NF4, FP4, FP8 (E4M3), and Double Quantization.
"""

import torch
from torch import Tensor
from typing import Tuple, Optional
from dataclasses import dataclass

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


def create_normal_map(offset=0.9677083, use_extra_value=True):
    """Create NF4 codebook (for compatibility with bitsandbytes)."""
    return NF4_CODEBOOK.clone()


def create_fp4_map(signed=True):
    """Create FP4 codebook (for compatibility with bitsandbytes)."""
    return FP4_CODEBOOK.clone()


def _try_load_native():
    """Try to load native C++ extension."""
    try:
        from . import _C
        return _C
    except ImportError:
        return None


# =============================================================================
# QuantState - Matches bitsandbytes API
# =============================================================================

@dataclass
class QuantState:
    """
    Quantization state for dequantization.

    Matches bitsandbytes QuantState for API compatibility.

    Attributes:
        absmax: Absolute maximum values per block
        shape: Original tensor shape
        code: Quantization codebook (NF4 or FP4)
        blocksize: Block size used for quantization
        quant_type: 'nf4' or 'fp4'
        dtype: Original tensor dtype
        offset: Optional offset tensor
        state2: Nested QuantState for double quantization
    """
    absmax: Tensor
    shape: torch.Size
    code: Optional[Tensor] = None
    blocksize: int = 64
    quant_type: str = "nf4"
    dtype: torch.dtype = torch.float16
    offset: Optional[Tensor] = None
    state2: Optional['QuantState'] = None

    def __post_init__(self):
        if self.code is None:
            self.code = NF4_CODEBOOK if self.quant_type == "nf4" else FP4_CODEBOOK

    def to(self, device):
        """Move state to device."""
        self.absmax = self.absmax.to(device)
        if self.code is not None:
            self.code = self.code.to(device)
        if self.offset is not None:
            self.offset = self.offset.to(device)
        if self.state2 is not None:
            self.state2 = self.state2.to(device)
        return self

    def as_dict(self, packed=False):
        """Convert to dictionary for serialization."""
        return {
            'absmax': self.absmax,
            'shape': self.shape,
            'blocksize': self.blocksize,
            'quant_type': self.quant_type,
            'dtype': self.dtype,
            'state2': self.state2.as_dict() if self.state2 else None,
        }

    @classmethod
    def from_dict(cls, state_dict, device='cpu'):
        """Create from dictionary."""
        state2 = None
        if state_dict.get('state2') is not None:
            state2 = cls.from_dict(state_dict['state2'], device)

        return cls(
            absmax=state_dict['absmax'].to(device),
            shape=state_dict['shape'],
            blocksize=state_dict.get('blocksize', 64),
            quant_type=state_dict.get('quant_type', 'nf4'),
            dtype=state_dict.get('dtype', torch.float16),
            state2=state2,
        )


# =============================================================================
# 4-bit Quantization - bitsandbytes compatible API
# =============================================================================

def quantize_4bit(
    A: Tensor,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
    compress_statistics: bool = False,
    quant_type: str = "nf4",
    quant_storage: torch.dtype = torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor to 4-bit NF4 or FP4 format.

    Args:
        A: Input tensor (any shape, will be flattened internally)
        absmax: Optional pre-allocated absmax tensor
        out: Optional pre-allocated output tensor
        blocksize: Elements per quantization block (default: 64)
        compress_statistics: If True, apply double quantization to absmax
        quant_type: 'nf4' or 'fp4'
        quant_storage: dtype for storing quantized values (default: uint8)

    Returns:
        Tuple of (quantized tensor, QuantState)
    """
    if quant_type not in ("nf4", "fp4"):
        raise ValueError(f"quant_type must be 'nf4' or 'fp4', got {quant_type}")

    orig_shape = A.shape
    orig_dtype = A.dtype
    A = A.contiguous()

    # Flatten to 2D for processing
    numel = A.numel()
    # Pad to be divisible by blocksize * 2 (for packing)
    padded_numel = ((numel + blocksize - 1) // blocksize) * blocksize
    if padded_numel % 2 != 0:
        padded_numel += blocksize

    A_flat = torch.zeros(padded_numel, dtype=A.dtype, device=A.device)
    A_flat[:numel] = A.flatten()

    # Reshape to [num_blocks, blocksize]
    num_blocks = padded_numel // blocksize
    A_blocked = A_flat.view(num_blocks, blocksize)

    # Compute absmax per block
    if absmax is None:
        absmax = A_blocked.float().abs().max(dim=1).values.clamp(min=1e-8)

    # Normalize and quantize
    codebook = NF4_CODEBOOK if quant_type == "nf4" else FP4_CODEBOOK
    codebook = codebook.to(A.device)

    A_norm = A_blocked.float() / absmax.unsqueeze(1)

    # Find nearest codebook entry
    # [num_blocks, blocksize, 1] vs [1, 1, 16]
    diffs = (A_norm.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)  # [num_blocks, blocksize]

    # Pack two 4-bit values per byte
    indices_flat = indices.flatten()
    packed_numel = padded_numel // 2
    if out is None:
        out = torch.zeros(packed_numel, dtype=quant_storage, device=A.device)

    out[:] = (indices_flat[0::2] | (indices_flat[1::2] << 4)).to(quant_storage)

    # Double quantization
    state2 = None
    if compress_statistics:
        absmax_quant, state2 = quantize_blockwise(absmax, blocksize=256)
        absmax = absmax_quant

    quant_state = QuantState(
        absmax=absmax,
        shape=orig_shape,
        blocksize=blocksize,
        quant_type=quant_type,
        dtype=orig_dtype,
        state2=state2,
    )

    return out, quant_state


def dequantize_4bit(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
    quant_type: str = "nf4",
) -> Tensor:
    """
    Dequantize 4-bit tensor back to floating point.

    Args:
        A: Quantized tensor (packed uint8)
        quant_state: QuantState from quantize_4bit
        absmax: Optional absmax (if quant_state not provided)
        out: Optional pre-allocated output tensor
        blocksize: Block size (if quant_state not provided)
        quant_type: 'nf4' or 'fp4' (if quant_state not provided)

    Returns:
        Dequantized tensor with original shape
    """
    if quant_state is not None:
        absmax = quant_state.absmax
        blocksize = quant_state.blocksize
        quant_type = quant_state.quant_type
        shape = quant_state.shape
        dtype = quant_state.dtype

        # Handle double quantization
        if quant_state.state2 is not None:
            absmax = dequantize_blockwise(absmax, quant_state.state2)
    else:
        if absmax is None:
            raise ValueError("Either quant_state or absmax must be provided")
        shape = None
        dtype = torch.float16

    codebook = NF4_CODEBOOK if quant_type == "nf4" else FP4_CODEBOOK
    codebook = codebook.to(A.device)

    # Unpack
    low = (A & 0x0F).long()
    high = ((A >> 4) & 0x0F).long()

    # Interleave
    unpacked = torch.zeros(A.numel() * 2, dtype=torch.long, device=A.device)
    unpacked[0::2] = low.flatten()
    unpacked[1::2] = high.flatten()

    # Reshape to blocks
    num_blocks = absmax.numel()
    padded_numel = num_blocks * blocksize
    unpacked = unpacked[:padded_numel].view(num_blocks, blocksize)

    # Dequantize
    values = codebook[unpacked]  # [num_blocks, blocksize]
    values = values * absmax.view(-1, 1)

    # Reshape to original
    if out is None:
        if shape is not None:
            out = values.flatten()[:torch.Size(shape).numel()].view(shape).to(dtype)
        else:
            out = values.flatten().to(dtype)
    else:
        out[:] = values.flatten()[:out.numel()].view(out.shape).to(dtype)

    return out


def quantize_nf4(
    A: Tensor,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
    compress_statistics: bool = False,
    quant_storage: torch.dtype = torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """Quantize to NF4 format. Alias for quantize_4bit with quant_type='nf4'."""
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "nf4", quant_storage)


def dequantize_nf4(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
) -> Tensor:
    """Dequantize NF4 tensor. Alias for dequantize_4bit with quant_type='nf4'."""
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "nf4")


def quantize_fp4(
    A: Tensor,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
    compress_statistics: bool = False,
    quant_storage: torch.dtype = torch.uint8,
) -> Tuple[Tensor, QuantState]:
    """Quantize to FP4 format. Alias for quantize_4bit with quant_type='fp4'."""
    return quantize_4bit(A, absmax, out, blocksize, compress_statistics, "fp4", quant_storage)


def dequantize_fp4(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 64,
) -> Tensor:
    """Dequantize FP4 tensor. Alias for dequantize_4bit with quant_type='fp4'."""
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, "fp4")


# =============================================================================
# Blockwise Quantization (INT8)
# =============================================================================

def quantize_blockwise(
    A: Tensor,
    code: Optional[Tensor] = None,
    absmax: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> Tuple[Tensor, QuantState]:
    """
    Quantize tensor to INT8 using blockwise absmax scaling.

    Args:
        A: Input tensor
        code: Optional codebook (unused, for API compatibility)
        absmax: Optional pre-allocated absmax tensor
        out: Optional pre-allocated output tensor
        blocksize: Elements per block
        nested: If True, apply nested quantization

    Returns:
        Tuple of (quantized tensor, QuantState)
    """
    orig_shape = A.shape
    orig_dtype = A.dtype
    A = A.contiguous().flatten()
    numel = A.numel()

    # Pad to blocksize
    padded_numel = ((numel + blocksize - 1) // blocksize) * blocksize
    A_padded = torch.zeros(padded_numel, dtype=A.dtype, device=A.device)
    A_padded[:numel] = A

    num_blocks = padded_numel // blocksize
    A_blocked = A_padded.view(num_blocks, blocksize)

    # Compute absmax per block
    if absmax is None:
        absmax = A_blocked.float().abs().max(dim=1).values.clamp(min=1e-8)

    # Quantize
    scale = 127.0 / absmax.unsqueeze(1)
    if out is None:
        out = torch.clamp(torch.round(A_blocked.float() * scale), -127, 127).to(torch.int8)
    else:
        out[:] = torch.clamp(torch.round(A_blocked.float() * scale), -127, 127).to(torch.int8)

    out = out.flatten()[:numel].view(orig_shape)

    state2 = None
    if nested:
        absmax, state2 = quantize_blockwise(absmax, blocksize=256)

    quant_state = QuantState(
        absmax=absmax,
        shape=orig_shape,
        blocksize=blocksize,
        quant_type="int8",
        dtype=orig_dtype,
        state2=state2,
    )

    return out, quant_state


def dequantize_blockwise(
    A: Tensor,
    quant_state: Optional[QuantState] = None,
    absmax: Optional[Tensor] = None,
    code: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
    blocksize: int = 4096,
    nested: bool = False,
) -> Tensor:
    """
    Dequantize blockwise INT8 tensor.

    Args:
        A: Quantized int8 tensor
        quant_state: QuantState from quantize_blockwise
        absmax: Optional absmax (if quant_state not provided)
        code: Unused, for API compatibility
        out: Optional pre-allocated output tensor
        blocksize: Block size (if quant_state not provided)
        nested: If True, dequantize nested state first

    Returns:
        Dequantized tensor
    """
    if quant_state is not None:
        absmax = quant_state.absmax
        blocksize = quant_state.blocksize
        shape = quant_state.shape
        dtype = quant_state.dtype

        if quant_state.state2 is not None:
            absmax = dequantize_blockwise(absmax, quant_state.state2)
    else:
        if absmax is None:
            raise ValueError("Either quant_state or absmax must be provided")
        shape = A.shape
        dtype = torch.float16

    A_flat = A.flatten().float()
    numel = A_flat.numel()

    # Pad
    padded_numel = ((numel + blocksize - 1) // blocksize) * blocksize
    A_padded = torch.zeros(padded_numel, dtype=torch.float32, device=A.device)
    A_padded[:numel] = A_flat

    num_blocks = padded_numel // blocksize
    A_blocked = A_padded.view(num_blocks, blocksize)

    # Dequantize
    scale = absmax.view(-1, 1) / 127.0
    dequant = A_blocked * scale
    dequant = dequant.flatten()[:numel].view(shape).to(dtype)

    if out is not None:
        out[:] = dequant
        return out

    return dequant


# =============================================================================
# Convenience aliases for old API (backward compatibility)
# =============================================================================

def quantize_rowwise(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to INT8 using row-wise absmax scaling.

    Returns:
        Tuple of (quantized int8 tensor, scales)
    """
    orig_shape = tensor.shape
    tensor_2d = tensor.view(-1, tensor.shape[-1])

    absmax = tensor_2d.float().abs().max(dim=-1).values
    scales = absmax.clamp(min=1e-8)

    quantized = torch.clamp(
        torch.round(tensor_2d.float() * (127.0 / scales.unsqueeze(-1))),
        -127, 127
    ).to(torch.int8)

    return quantized.view(orig_shape), scales


def dequantize_rowwise(quantized: Tensor, scales: Tensor,
                       dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize INT8 tensor with row-wise scales."""
    orig_shape = quantized.shape
    quantized_2d = quantized.view(-1, quantized.shape[-1])
    scales_2d = scales.view(-1)

    dequantized = (quantized_2d.float() * (scales_2d.unsqueeze(-1) / 127.0)).to(dtype)
    return dequantized.view(orig_shape)


# =============================================================================
# FP8 (8-bit Floating Point) - E4M3 format
# =============================================================================

def quantize_fp8_e4m3(tensor: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Quantize tensor to FP8 E4M3 format.

    FP8 E4M3 (1 sign, 4 exponent, 3 mantissa) offers better precision
    than INT8 with similar memory footprint. Range: +/-448.

    Args:
        tensor: Input tensor

    Returns:
        Tuple of (quantized uint8 tensor, row-wise scales)
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


# =============================================================================
# Matrix multiplication with quantized weights
# =============================================================================

def matmul_4bit(
    A: Tensor,
    B: Tensor,
    quant_state: QuantState,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Matrix multiplication with 4-bit quantized weights.

    Args:
        A: Input tensor [..., in_features]
        B: Quantized weight tensor (packed uint8)
        quant_state: QuantState from quantize_4bit
        bias: Optional bias tensor

    Returns:
        Output tensor [..., out_features]
    """
    # Dequantize weights
    weight = dequantize_4bit(B, quant_state)

    # Reshape for matmul if needed
    orig_shape = A.shape
    if A.dim() > 2:
        A = A.view(-1, A.shape[-1])

    output = torch.nn.functional.linear(A.to(weight.dtype), weight, bias)

    if len(orig_shape) > 2:
        output = output.view(*orig_shape[:-1], -1)

    return output


def matmul_nf4(input: Tensor, weight_packed: Tensor, weight_state: QuantState,
               bias: Optional[Tensor] = None) -> Tensor:
    """Matrix multiplication with NF4-quantized weights."""
    return matmul_4bit(input, weight_packed, weight_state, bias)


def matmul_fp4(input: Tensor, weight_packed: Tensor, weight_state: QuantState,
               bias: Optional[Tensor] = None) -> Tensor:
    """Matrix multiplication with FP4-quantized weights."""
    return matmul_4bit(input, weight_packed, weight_state, bias)


def matmul_int8(A: Tensor, B: Tensor, A_scales: Tensor, B_scales: Tensor,
                dtype: torch.dtype = torch.float16) -> Tensor:
    """INT8 matmul with fused dequantization."""
    A_dequant = dequantize_rowwise(A, A_scales, dtype)
    B_dequant = dequantize_rowwise(B.T, B_scales, dtype).T
    return torch.matmul(A_dequant, B_dequant)


def matmul_fp8_e4m3(input: Tensor, weight: Tensor, weight_scales: Tensor,
                    bias: Optional[Tensor] = None,
                    dtype: torch.dtype = torch.float16) -> Tensor:
    """Matrix multiplication with FP8 E4M3 weights."""
    weight_dequant = dequantize_fp8_e4m3(weight, weight_scales, dtype)

    is_1d = input.dim() == 1
    if is_1d:
        input = input.unsqueeze(0)

    output = torch.nn.functional.linear(input.to(dtype), weight_dequant, bias)
    return output.squeeze(0) if is_1d else output


# =============================================================================
# Double Quantization
# =============================================================================

def double_quant(
    A: Tensor,
    col_stats: Optional[Tensor] = None,
    row_stats: Optional[Tensor] = None,
    out_col: Optional[Tensor] = None,
    out_row: Optional[Tensor] = None,
    threshold: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Apply double quantization with row and column statistics.

    This is the bitsandbytes-style double_quant that computes both
    row and column statistics for LLM.int8() style quantization.

    Args:
        A: Input tensor [rows, cols]
        col_stats: Optional pre-computed column statistics
        row_stats: Optional pre-computed row statistics
        out_col: Optional output for column-quantized
        out_row: Optional output for row-quantized
        threshold: Outlier threshold (unused in this impl)

    Returns:
        Tuple of (col_quantized, row_quantized, col_stats, row_stats, outliers)
    """
    if A.dim() != 2:
        raise ValueError("Input must be 2D")

    A_f32 = A.float()

    if row_stats is None:
        row_stats = A_f32.abs().max(dim=1).values.clamp(min=1e-8)
    if col_stats is None:
        col_stats = A_f32.abs().max(dim=0).values.clamp(min=1e-8)

    # Row-wise quantization
    if out_row is None:
        out_row = torch.clamp(
            torch.round(A_f32 * (127.0 / row_stats.unsqueeze(1))),
            -127, 127
        ).to(torch.int8)

    # Column-wise quantization
    if out_col is None:
        out_col = torch.clamp(
            torch.round(A_f32 * (127.0 / col_stats.unsqueeze(0))),
            -127, 127
        ).to(torch.int8)

    return out_col, out_row, col_stats, row_stats, None


def dequant_absmax(absmax_quant: Tensor, absmax_scales: Tensor,
                   blocksize: int = 256) -> Tensor:
    """Dequantize double-quantized absmax values."""
    # Handle QuantState
    if isinstance(absmax_scales, QuantState):
        return dequantize_blockwise(absmax_quant, absmax_scales)

    rows = absmax_quant.shape[0] if absmax_quant.dim() > 1 else 1
    num_blocks = absmax_quant.numel() // rows if rows > 0 else absmax_quant.numel()
    dq_blocks = absmax_scales.numel() // rows if rows > 0 else absmax_scales.numel()

    if absmax_quant.dim() == 1:
        absmax_quant = absmax_quant.unsqueeze(0)
        absmax_scales = absmax_scales.unsqueeze(0)

    absmax = torch.zeros_like(absmax_quant, dtype=torch.float32)

    for dqb in range(dq_blocks):
        start = dqb * blocksize
        end = min(start + blocksize, num_blocks)
        scale = absmax_scales[:, dqb:dqb+1]
        absmax[:, start:end] = absmax_quant[:, start:end].float() * scale

    return absmax.squeeze(0) if rows == 1 else absmax


# =============================================================================
# INT8 with Column + Row Statistics (LLM.int8 style)
# =============================================================================

def quantize_colrow(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Quantize matrix using both column-wise and row-wise statistics.

    Uses the geometric mean of row and column absmax for better accuracy.
    This is the approach used in LLM.int8() for handling varying magnitudes.

    Args:
        tensor: Input [rows, cols] tensor

    Returns:
        Tuple of (quantized int8, row_scales, col_scales)
    """
    if tensor.dim() != 2:
        raise ValueError("Input must be 2D")

    tensor_f32 = tensor.float()

    row_absmax = tensor_f32.abs().max(dim=1).values.clamp(min=1e-8)
    col_absmax = tensor_f32.abs().max(dim=0).values.clamp(min=1e-8)

    scale_matrix = torch.sqrt(row_absmax.unsqueeze(1) * col_absmax.unsqueeze(0))

    quantized = torch.clamp(
        torch.round(tensor_f32 * (127.0 / scale_matrix)),
        -127, 127
    ).to(torch.int8)

    return quantized, row_absmax, col_absmax


def dequantize_colrow(quantized: Tensor, row_scales: Tensor, col_scales: Tensor,
                      dtype: torch.dtype = torch.float16) -> Tensor:
    """Dequantize INT8 tensor with column+row statistics."""
    scale_matrix = torch.sqrt(row_scales.unsqueeze(1) * col_scales.unsqueeze(0))
    dequant = quantized.float() * (scale_matrix / 127.0)
    return dequant.to(dtype)


def matmul_colrow(input: Tensor, weight_int8: Tensor,
                  weight_row_scales: Tensor, weight_col_scales: Tensor,
                  bias: Optional[Tensor] = None,
                  dtype: torch.dtype = torch.float16) -> Tensor:
    """Matrix multiplication with column+row quantized weights."""
    weight_dequant = dequantize_colrow(weight_int8, weight_row_scales, weight_col_scales, dtype)
    output = torch.nn.functional.linear(input.to(dtype), weight_dequant, bias)
    return output


# =============================================================================
# Sparse Matrix Multiplication (COO format)
# =============================================================================

def spmm_coo(
    row_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
    dense: Tensor,
    sparse_rows: int,
    sparse_cols: int,
) -> Tensor:
    """
    Sparse-dense matrix multiplication in COO format.

    Computes: sparse @ dense where sparse is in COO format.
    """
    indices = torch.stack([row_indices, col_indices], dim=0)
    sparse = torch.sparse_coo_tensor(
        indices, values,
        size=(sparse_rows, sparse_cols),
        device=values.device,
        dtype=values.dtype
    )
    return torch.sparse.mm(sparse, dense)


def spmm_coo_int8(
    row_indices: Tensor,
    col_indices: Tensor,
    values_int8: Tensor,
    values_scale: Tensor,
    dense: Tensor,
    sparse_rows: int,
    sparse_cols: int,
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Sparse-dense matrix multiplication with INT8 sparse values."""
    scale = values_scale.item() if values_scale.numel() == 1 else values_scale.to(dtype)
    values = values_int8.to(dtype) * scale
    return spmm_coo(row_indices, col_indices, values, dense.to(dtype), sparse_rows, sparse_cols)


def sparse_coo_from_dense(tensor: Tensor, threshold: float = 0.0) -> Tuple[Tensor, Tensor, Tensor, int, int]:
    """Convert dense tensor to COO sparse format."""
    rows, cols = tensor.shape

    if threshold > 0:
        mask = tensor.abs() >= threshold
        sparse = tensor * mask
    else:
        sparse = tensor

    nonzero = sparse.nonzero()
    row_indices = nonzero[:, 0]
    col_indices = nonzero[:, 1]
    values = sparse[row_indices, col_indices]

    return row_indices, col_indices, values, rows, cols


def quantize_sparse_coo(
    row_indices: Tensor,
    col_indices: Tensor,
    values: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Quantize sparse COO values to INT8."""
    absmax = values.float().abs().max().clamp(min=1e-8)
    scale = absmax / 127.0

    values_int8 = torch.clamp(
        torch.round(values.float() / scale),
        -127, 127
    ).to(torch.int8)

    return row_indices, col_indices, values_int8, scale.view(1)


# =============================================================================
# Python Fallback Implementations
# =============================================================================

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

    absmax = tensor_f32.abs().max(dim=1).values
    scales = (absmax / 448.0).clamp(min=1e-12)

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
