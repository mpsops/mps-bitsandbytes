# MPS BitsAndBytes

**Real 4-bit and 8-bit quantization for PyTorch on Apple Silicon (M1/M2/M3/M4).**

Full bitsandbytes-compatible API with Metal GPU acceleration for running large models on your Mac.

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
- **8-bit Optimizers** - Adam8bit, AdamW8bit, Lion8bit, SGD8bit
- **Paged Optimizers** - CPU offloading for larger models
- **Quantized Embeddings** - Embedding4bit, Embedding8bit
- **Sparse Operations** - spmm_coo, spmm_coo_int8
- **LLM.int8** - OutlierAwareLinear with col+row quantization
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

### 8-bit Optimizers

Memory-efficient optimizers that store momentum/variance in 8-bit:

```python
from mps_bitsandbytes import Adam8bit, AdamW8bit, Lion8bit, SGD8bit

# Drop-in replacement for torch optimizers
optimizer = Adam8bit(model.parameters(), lr=1e-3)
optimizer = AdamW8bit(model.parameters(), lr=1e-3, weight_decay=0.01)
optimizer = Lion8bit(model.parameters(), lr=1e-4)
optimizer = SGD8bit(model.parameters(), lr=0.1, momentum=0.9)
```

### Paged Optimizers

Offload optimizer states to CPU for training larger models:

```python
from mps_bitsandbytes import PagedAdam, PagedAdamW, PagedLion

# States are stored on CPU, copied to GPU during step()
optimizer = PagedAdamW(model.parameters(), lr=1e-3, page_to_cpu=True)
```

### Quantized Embeddings

Reduce embedding table memory by 50-75%:

```python
from mps_bitsandbytes import Embedding4bit, Embedding8bit, EmbeddingNF4, EmbeddingFP4

# Convert existing embedding
embed = torch.nn.Embedding(50000, 4096).half().to('mps')
embed_4bit = Embedding4bit.from_embedding(embed)  # NF4 by default
embed_fp4 = EmbeddingFP4.from_embedding(embed)    # FP4
embed_8bit = Embedding8bit.from_embedding(embed)  # INT8
```

### Functional API

```python
from mps_bitsandbytes import (
    # 4-bit
    quantize_nf4, dequantize_nf4, matmul_nf4,
    quantize_fp4, dequantize_fp4, matmul_fp4,
    # 8-bit
    quantize_fp8_e4m3, dequantize_fp8_e4m3, matmul_fp8_e4m3,
    quantize_rowwise, dequantize_rowwise, matmul_int8,
    # Col+Row INT8 (LLM.int8 style)
    quantize_colrow, dequantize_colrow, matmul_colrow,
    # Double quantization
    double_quant, dequant_absmax,
    # Sparse
    spmm_coo, spmm_coo_int8, sparse_coo_from_dense, quantize_sparse_coo,
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
from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, Adam8bit
from peft import get_peft_model, LoraConfig

# Load in 4-bit
config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained("model_name", torch_dtype=torch.float16)
model = quantize_model(model, quantization_config=config, device='mps')

# Add LoRA adapters (train in fp16 while base stays quantized)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# Use 8-bit optimizer for extra memory savings
optimizer = Adam8bit(model.parameters(), lr=1e-4)
trainer.train()
```

## API Reference

### Linear Modules

| Class | Format | Use Case |
|-------|--------|----------|
| `Linear4bit` | NF4 or FP4 | LLM inference, QLoRA |
| `Linear8bit` | INT8 | General quantization |
| `LinearFP8` | FP8 E4M3 | Better precision 8-bit |
| `OutlierAwareLinear` | INT8 + FP16 | LLM.int8 mixed precision |
| `SwitchBackLinear` | INT8 | Training with quantized forward |

### Embedding Modules

| Class | Format | Memory Savings |
|-------|--------|----------------|
| `Embedding4bit` | NF4 (default) | ~75% |
| `EmbeddingNF4` | NF4 | ~75% |
| `EmbeddingFP4` | FP4 | ~75% |
| `Embedding8bit` | INT8 | ~50% |

### Optimizers

| Class | Description |
|-------|-------------|
| `Adam8bit` | Adam with 8-bit states |
| `AdamW8bit` | AdamW with 8-bit states |
| `Lion8bit` | Lion optimizer with 8-bit momentum |
| `SGD8bit` | SGD with 8-bit momentum |
| `PagedAdam` | Adam with CPU offloading |
| `PagedAdamW` | AdamW with CPU offloading |
| `PagedLion` | Lion with CPU offloading |

### Functional API

**4-bit (NF4/FP4):**
- `quantize_nf4(tensor, block_size=64)` / `quantize_fp4(...)`
- `dequantize_nf4(packed, absmax, ...)` / `dequantize_fp4(...)`
- `matmul_nf4(input, weight_packed, weight_absmax, bias=None)` / `matmul_fp4(...)`

**8-bit:**
- `quantize_fp8_e4m3(tensor)` - FP8 quantization
- `quantize_rowwise(tensor)` - INT8 row-wise quantization
- `quantize_colrow(tensor)` - INT8 col+row quantization (LLM.int8)
- `matmul_fp8_e4m3(...)` / `matmul_int8(...)` / `matmul_colrow(...)`

**Double Quantization:**
- `double_quant(absmax, double_quant_block=256)` - Quantize scales
- `dequant_absmax(absmax_quant, absmax_scales)` - Restore scales

**Sparse Operations:**
- `sparse_coo_from_dense(tensor)` - Convert to COO format
- `spmm_coo(row_idx, col_idx, values, dense, rows, cols)` - Sparse matmul
- `spmm_coo_int8(...)` - INT8 sparse matmul
- `quantize_sparse_coo(row_idx, col_idx, values)` - Quantize sparse values

**Utilities:**
- `is_available()` - Check MPS availability
- `has_native_kernels()` - Check Metal kernels loaded
- `get_memory_footprint(model)` - Calculate memory usage

## Comparison with bitsandbytes

| Feature | bitsandbytes (CUDA) | mps-bitsandbytes |
|---------|---------------------|------------------|
| NF4/FP4 | CUDA | Metal |
| INT8/FP8 | CUDA | Metal |
| Double quant | CUDA | Metal |
| 8-bit Optimizers | CUDA | Pure PyTorch |
| Paged Optimizers | CUDA | Pure PyTorch |
| Quantized Embeddings | CUDA | Pure PyTorch |
| Sparse matmul | CUDA | Pure PyTorch |
| LLM.int8 (col+row) | CUDA | Pure PyTorch |
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
