"""
MPS BitsAndBytes - Neural Network Modules

Quantized linear and embedding layers for memory-efficient inference and training.
"""

from .linear4bit import Linear4bit
from .linear8bit import Linear8bit
from .linear_fp8 import LinearFP8
from .embedding import Embedding4bit, Embedding8bit, EmbeddingNF4, EmbeddingFP4
from .outlier_aware import OutlierAwareLinear
from .switchback import SwitchBackLinear, SwitchBackLinearCallback

__all__ = [
    # Linear layers
    'Linear4bit',
    'Linear8bit',
    'LinearFP8',
    # Advanced linear layers
    'OutlierAwareLinear',
    'SwitchBackLinear',
    'SwitchBackLinearCallback',
    # Embedding layers
    'Embedding4bit',
    'Embedding8bit',
    'EmbeddingNF4',
    'EmbeddingFP4',
]
