# MPS BitsAndBytes

**Real 4-bit and 8-bit quantization for PyTorch on Apple Silicon (M1/M2/M3/M4).**

Proper NF4/FP4/FP8/INT8 quantization with Metal GPU kernels for running large models on your Mac.

## Features

| Format | Bits | Memory Savings | Best For |
|--------|------|----------------|----------|
| **NF4** | 4-bit | ~75% | LLM weights (normally distributed) |
| **FP4** | 4-bit | ~75% | Alternative with better dynamic range |
| **FP8 E4M3** | 8-bit | ~50% | Better precision than INT8 |
| **INT8** | 8-bit | ~50% | General purpose |

Plus:
- **Metal GPU kernels** - Fused dequant+matmul, no Python overhead
- **Double quantization** - Extra ~10% savings on scales
- **HuggingFace compatible** - `BitsAndBytesConfig` API works out of the box
- **QLoRA training** - Freeze quantized weights, train LoRA adapters

## Installation

```bash
pip install mps-bitsandbytes
```

Or from source:

```bash
git clone https://github.com/mpsops/mps-bitsandbytes
cd mps-bitsandbytes
pip install -e .
python setup.py build_ext --inplace  # Build Metal kernels
```

## Quick Start

### 4-bit Quantization (NF4 - Recommended for LLMs)

```python
import torch
from mps_bitsandbytes import Linear4bit, BitsAndBytesConfig, quantize_model

# Convert a single layer
linear = torch.nn.Linear(4096, 4096).half().to('mps')
linear_4bit = Linear4bit.from_linear(linear)  # NF4 by default

# Or use FP4
linear_fp4 = Linear4bit.from_linear(linear, quant_type='fp4')

# Quantize entire model
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = quantize_model(your_model, quantization_config=config, device='mps')
```

### 8-bit Quantization (FP8 or INT8)

```python
from mps_bitsandbytes import Linear8bit, LinearFP8

# INT8 (traditional)
linear_int8 = Linear8bit.from_linear(linear)

# FP8 E4M3 (better precision)
linear_fp8 = LinearFP8.from_linear(linear)
```

### Functional API

```python
from mps_bitsandbytes import (
    quantize_nf4, matmul_nf4,
    quantize_fp4, matmul_fp4,
    quantize_fp8_e4m3, matmul_fp8_e4m3,
    double_quant,
)

# NF4
weight = torch.randn(4096, 4096, device='mps', dtype=torch.float16)
packed, absmax = quantize_nf4(weight, block_size=64)
output = matmul_nf4(input, packed, absmax)

# Double quantization (quantize the scales too)
absmax_quant, absmax_scales = double_quant(absmax)
```

## Memory Savings

| Model | FP16 | INT8/FP8 | NF4/FP4 |
|-------|------|----------|---------|
| 7B params | 14 GB | 7 GB | **3.5 GB** |
| 13B params | 26 GB | 13 GB | **6.5 GB** |
| 70B params | 140 GB | 70 GB | **35 GB** |

## HuggingFace Integration

```python
from transformers import AutoModelForCausalLM
from mps_bitsandbytes import BitsAndBytesConfig, quantize_model

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
)

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = quantize_model(model, quantization_config=config, device='mps')
```

## QLoRA Training

```python
from mps_bitsandbytes import BitsAndBytesConfig, quantize_model
from peft import get_peft_model, LoraConfig

# Load in 4-bit
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("model_name", torch_dtype=torch.float16)
model = quantize_model(model, quantization_config=config, device='mps')

# Add LoRA adapters (train in fp16 while base stays quantized)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
trainer.train()
```

## API Reference

### Linear Modules

| Class | Format | Use Case |
|-------|--------|----------|
| `Linear4bit` | NF4 or FP4 | LLM inference, QLoRA |
| `Linear8bit` | INT8 | General quantization |
| `LinearFP8` | FP8 E4M3 | Better precision 8-bit |

### Functional API

**4-bit (NF4/FP4):**
- `quantize_nf4(tensor, block_size=64)` / `quantize_fp4(...)`
- `dequantize_nf4(packed, absmax, ...)` / `dequantize_fp4(...)`
- `matmul_nf4(input, weight_packed, weight_absmax, bias=None)` / `matmul_fp4(...)`

**8-bit:**
- `quantize_fp8_e4m3(tensor)` - FP8 quantization
- `quantize_rowwise(tensor)` - INT8 quantization
- `matmul_fp8_e4m3(...)` / `matmul_int8(...)`

**Double Quantization:**
- `double_quant(absmax, double_quant_block=256)` - Quantize scales
- `dequant_absmax(absmax_quant, absmax_scales)` - Restore scales

**Utilities:**
- `is_available()` - Check MPS availability
- `has_native_kernels()` - Check Metal kernels loaded
- `get_memory_footprint(model)` - Calculate memory usage

## How It Works

### NF4 Quantization

NF4 (4-bit NormalFloat) uses a 16-value codebook optimized for Gaussian distributions:

1. Divide weights into blocks (default: 64 elements)
2. Compute absmax per block for scaling
3. Normalize to [-1, 1] and map to nearest codebook value
4. Pack two 4-bit indices per byte

### Metal Kernels

Fused kernels that dequantize on-the-fly during matmul:
- Tiled algorithms for cache efficiency
- Avoids memory bandwidth bottleneck of separate dequant+matmul
- FP32 accumulation for precision, FP16 output

## Comparison with bitsandbytes

| Feature | bitsandbytes (CUDA) | mps-bitsandbytes |
|---------|---------------------|------------------|
| NF4 | CUDA | Metal |
| FP4 | CUDA | Metal |
| INT8 | CUDA | Metal |
| FP8 | CUDA | Metal |
| Double quant | CUDA | Metal |
| Platform | NVIDIA | Apple Silicon |

## Demo

```bash
# Chat with a quantized LLM
python demo/chat.py
```

## License

MIT

## Credits

- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) - Original CUDA implementation
- [QLoRA](https://arxiv.org/abs/2305.14314) - NF4 quantization paper
