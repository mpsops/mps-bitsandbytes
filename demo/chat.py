#!/usr/bin/env python3
"""Demo: Run a quantized LLM on Apple Silicon with mps-bitsandbytes"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, get_memory_footprint

MODEL = "Qwen/Qwen2.5-0.5B"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)

fp16_size = get_memory_footprint(model)["actual_size_gb"] * 1000
print(f"FP16 size: {fp16_size:.0f} MB")

print("Quantizing to NF4 4-bit...")
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = quantize_model(model, quantization_config=config, device="mps")

nf4_size = get_memory_footprint(model)["actual_size_gb"] * 1000
print(f"NF4 size: {nf4_size:.0f} MB (saved {100*(1-nf4_size/fp16_size):.0f}%)\n")

print("=" * 50)
print("Chat with the model (Ctrl+C to exit)")
print("=" * 50 + "\n")

while True:
    try:
        prompt = input("You: ")
        if not prompt.strip():
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to("mps")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        print(f"AI: {response}\n")

    except KeyboardInterrupt:
        print("\nBye!")
        break
