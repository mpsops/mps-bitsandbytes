"""
HuggingFace Transformers Integration

Provides BitsAndBytesConfig compatible API for loading quantized models
on Apple Silicon MPS devices.
"""

import torch
from torch import nn
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

from .nn import Linear4bit, Linear8bit


@dataclass
class BitsAndBytesConfig:
    """
    Configuration for loading models with bitsandbytes quantization on MPS.

    This class mimics the transformers BitsAndBytesConfig API for compatibility.

    Args:
        load_in_8bit: Load model in 8-bit precision
        load_in_4bit: Load model in 4-bit precision (NF4)
        llm_int8_threshold: Threshold for outlier detection (not used on MPS)
        llm_int8_skip_modules: Modules to skip for 8-bit quantization
        llm_int8_enable_fp32_cpu_offload: Enable CPU offload (not used on MPS)
        llm_int8_has_fp16_weight: Has FP16 weight (not used on MPS)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit (torch.float16 or torch.bfloat16)
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
        bnb_4bit_use_double_quant: Use double quantization (not implemented yet)
        bnb_4bit_quant_storage: Storage type for quantized weights

    Example:
        >>> from mps_bitsandbytes import BitsAndBytesConfig
        >>> config = BitsAndBytesConfig(
        ...     load_in_4bit=True,
        ...     bnb_4bit_quant_type="nf4",
        ...     bnb_4bit_compute_dtype=torch.float16,
        ... )
    """

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[list] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_storage: torch.dtype = torch.uint8

    def __post_init__(self):
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot load in both 4-bit and 8-bit")

        if self.bnb_4bit_quant_type not in ('nf4', 'fp4'):
            raise ValueError(f"bnb_4bit_quant_type must be 'nf4' or 'fp4', got {self.bnb_4bit_quant_type}")

        if self.llm_int8_skip_modules is None:
            self.llm_int8_skip_modules = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'load_in_8bit': self.load_in_8bit,
            'load_in_4bit': self.load_in_4bit,
            'llm_int8_threshold': self.llm_int8_threshold,
            'llm_int8_skip_modules': self.llm_int8_skip_modules,
            'bnb_4bit_compute_dtype': str(self.bnb_4bit_compute_dtype),
            'bnb_4bit_quant_type': self.bnb_4bit_quant_type,
            'bnb_4bit_use_double_quant': self.bnb_4bit_use_double_quant,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BitsAndBytesConfig':
        """Create config from dictionary."""
        # Handle dtype conversion
        if 'bnb_4bit_compute_dtype' in config_dict:
            dtype_str = config_dict['bnb_4bit_compute_dtype']
            if isinstance(dtype_str, str):
                if 'float16' in dtype_str:
                    config_dict['bnb_4bit_compute_dtype'] = torch.float16
                elif 'bfloat16' in dtype_str:
                    config_dict['bnb_4bit_compute_dtype'] = torch.bfloat16
                else:
                    config_dict['bnb_4bit_compute_dtype'] = torch.float16

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @property
    def is_quantizable(self) -> bool:
        """Check if config specifies quantization."""
        return self.load_in_4bit or self.load_in_8bit

    @property
    def quantization_method(self) -> str:
        """Get quantization method string."""
        if self.load_in_4bit:
            return 'bitsandbytes_4bit'
        elif self.load_in_8bit:
            return 'bitsandbytes_8bit'
        return 'none'


def replace_linear_with_4bit(
    model: nn.Module,
    quantization_config: BitsAndBytesConfig,
    modules_to_not_convert: Optional[list] = None,
    current_key_name: Optional[str] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with Linear4bit.

    Args:
        model: Model to quantize
        quantization_config: Quantization configuration
        modules_to_not_convert: List of module names to skip
        current_key_name: Current module path (for recursion)

    Returns:
        Model with quantized linear layers
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, module in model.named_children():
        full_name = f"{current_key_name}.{name}" if current_key_name else name

        if isinstance(module, nn.Linear):
            # Check if we should skip this module
            should_skip = any(skip in full_name for skip in modules_to_not_convert)
            if should_skip:
                continue

            # Replace with 4-bit version
            new_module = Linear4bit.from_linear(
                module,
                compute_dtype=quantization_config.bnb_4bit_compute_dtype,
                quant_type=quantization_config.bnb_4bit_quant_type,
            )
            setattr(model, name, new_module)
        else:
            # Recurse
            replace_linear_with_4bit(
                module,
                quantization_config,
                modules_to_not_convert,
                full_name,
            )

    return model


def replace_linear_with_8bit(
    model: nn.Module,
    quantization_config: BitsAndBytesConfig,
    modules_to_not_convert: Optional[list] = None,
    current_key_name: Optional[str] = None,
) -> nn.Module:
    """
    Replace all nn.Linear layers with Linear8bit.

    Args:
        model: Model to quantize
        quantization_config: Quantization configuration
        modules_to_not_convert: List of module names to skip
        current_key_name: Current module path (for recursion)

    Returns:
        Model with quantized linear layers
    """
    if modules_to_not_convert is None:
        modules_to_not_convert = quantization_config.llm_int8_skip_modules or []

    for name, module in model.named_children():
        full_name = f"{current_key_name}.{name}" if current_key_name else name

        if isinstance(module, nn.Linear):
            should_skip = any(skip in full_name for skip in modules_to_not_convert)
            if should_skip:
                continue

            new_module = Linear8bit.from_linear(module)
            setattr(model, name, new_module)
        else:
            replace_linear_with_8bit(
                module,
                quantization_config,
                modules_to_not_convert,
                full_name,
            )

    return model


def quantize_model(
    model: nn.Module,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = 'mps',
    compute_dtype: torch.dtype = torch.float16,
    modules_to_not_convert: Optional[list] = None,
) -> nn.Module:
    """
    Quantize a model for memory-efficient inference.

    This is the main entry point for quantizing models on MPS.

    Args:
        model: Model to quantize
        quantization_config: Optional BitsAndBytesConfig
        load_in_4bit: Load in 4-bit (if no config provided)
        load_in_8bit: Load in 8-bit (if no config provided)
        device: Target device
        compute_dtype: Compute dtype
        modules_to_not_convert: Modules to skip

    Returns:
        Quantized model

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from mps_bitsandbytes import quantize_model, BitsAndBytesConfig
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> config = BitsAndBytesConfig(load_in_4bit=True)
        >>> model = quantize_model(model, quantization_config=config)
    """
    # Create config if not provided
    if quantization_config is None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    # Quantize on current device first (avoids loading full fp16 to GPU)
    # Then move to target device
    if quantization_config.load_in_4bit:
        model = replace_linear_with_4bit(model, quantization_config, modules_to_not_convert)
    elif quantization_config.load_in_8bit:
        model = replace_linear_with_8bit(model, quantization_config, modules_to_not_convert)

    # Move quantized model to target device
    model = model.to(device)

    return model


def get_memory_footprint(model: nn.Module) -> Dict[str, Any]:
    """
    Calculate memory footprint of a model.

    Returns:
        Dictionary with memory statistics
    """
    total_bytes = 0
    total_params = 0
    quantized_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        total_bytes += param.numel() * param.element_size()

    for name, buf in model.named_buffers():
        total_params += buf.numel()
        total_bytes += buf.numel() * buf.element_size()

        # Track quantized parameters
        if 'weight_packed' in name or 'weight_int8' in name:
            quantized_params += buf.numel()

    fp16_size = total_params * 2 / 1e9  # If all were fp16
    actual_size = total_bytes / 1e9

    return {
        'total_params': total_params,
        'quantized_params': quantized_params,
        'fp16_size_gb': fp16_size,
        'actual_size_gb': actual_size,
        'savings_gb': fp16_size - actual_size,
        'savings_pct': (1 - actual_size / fp16_size) * 100 if fp16_size > 0 else 0,
    }


# Monkey-patch hook for transformers integration
def _patch_transformers():
    """
    Attempt to patch transformers to use our quantization on MPS.

    This is called automatically on import if transformers is available.
    """
    try:
        import transformers
        from transformers import modeling_utils

        # Store original
        _original_load_pretrained = modeling_utils.PreTrainedModel.from_pretrained.__func__

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            # Check if MPS quantization is requested
            quantization_config = kwargs.get('quantization_config')
            device_map = kwargs.get('device_map')

            if (quantization_config is not None and
                isinstance(quantization_config, BitsAndBytesConfig) and
                (device_map == 'mps' or device_map == {'': 'mps'})):

                # Remove quantization_config to load in FP16 first
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('quantization_config', None)
                kwargs_copy['device_map'] = None
                kwargs_copy['torch_dtype'] = quantization_config.bnb_4bit_compute_dtype

                # Load model
                model = _original_load_pretrained(cls, *args, **kwargs_copy)

                # Apply our quantization
                model = quantize_model(model, quantization_config, device='mps')

                return model

            return _original_load_pretrained(cls, *args, **kwargs)

        # Apply patch
        # modeling_utils.PreTrainedModel.from_pretrained = _patched_from_pretrained

    except ImportError:
        pass  # transformers not installed
