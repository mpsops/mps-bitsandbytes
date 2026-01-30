"""
MPS BitsAndBytes - 8-bit and Paged Optimizers

Memory-efficient optimizers for training on Apple Silicon.

8-bit optimizers store momentum/variance in int8, saving ~75% memory.
Paged optimizers offload states to CPU, enabling larger model training.
"""

from .adam8bit import (
    Adam8bit, AdamW8bit,
    quantize_state, dequantize_state,
    quantize_state_unsigned, dequantize_state_unsigned,
)
from .lion8bit import Lion8bit
from .sgd8bit import SGD8bit
from .paged import PagedAdam, PagedAdamW, PagedLion

__all__ = [
    # 8-bit optimizers
    'Adam8bit',
    'AdamW8bit',
    'Lion8bit',
    'SGD8bit',
    # Paged optimizers
    'PagedAdam',
    'PagedAdamW',
    'PagedLion',
    # Utilities
    'quantize_state',
    'dequantize_state',
]
