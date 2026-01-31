"""Test fused NF4 matmul kernel."""
import torch
import time

def test_fused_nf4_matmul():
    """Test that fused NF4 matmul works and is faster than Python fallback."""
    from mps_bitsandbytes.functional import quantize_4bit, matmul_4bit, dequantize_4bit

    device = "mps"
    M, K, N = 128, 4096, 4096  # Typical LLM dimensions

    # Create random weights and input
    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    input_tensor = torch.randn(M, K, dtype=torch.float16, device=device)

    # Quantize weights
    weight_packed, quant_state = quantize_4bit(weight, blocksize=64, quant_type="nf4")

    # Test correctness: compare fused vs dequantize+matmul
    # Fused path
    output_fused = matmul_4bit(input_tensor, weight_packed, quant_state)

    # Reference: manual dequantize + matmul
    weight_dequant = dequantize_4bit(weight_packed, quant_state)
    output_ref = torch.nn.functional.linear(input_tensor.to(weight_dequant.dtype), weight_dequant)

    # Check correctness
    torch.mps.synchronize()
    max_diff = (output_fused - output_ref).abs().max().item()
    print(f"Max diff (fused vs reference): {max_diff:.6f}")
    assert max_diff < 0.1, f"Fused kernel output differs too much: {max_diff}"

    # Benchmark
    warmup = 5
    iters = 20

    # Warmup
    for _ in range(warmup):
        _ = matmul_4bit(input_tensor, weight_packed, quant_state)
    torch.mps.synchronize()

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(iters):
        _ = matmul_4bit(input_tensor, weight_packed, quant_state)
    torch.mps.synchronize()
    fused_time = (time.perf_counter() - start) / iters * 1000

    # Benchmark dequantize + matmul (Python fallback style)
    for _ in range(warmup):
        w = dequantize_4bit(weight_packed, quant_state)
        _ = torch.nn.functional.linear(input_tensor.to(w.dtype), w)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        w = dequantize_4bit(weight_packed, quant_state)
        _ = torch.nn.functional.linear(input_tensor.to(w.dtype), w)
    torch.mps.synchronize()
    unfused_time = (time.perf_counter() - start) / iters * 1000

    print(f"\n=== NF4 MatMul Benchmark (M={M}, K={K}, N={N}) ===")
    print(f"Fused kernel:    {fused_time:.2f} ms")
    print(f"Dequant+matmul:  {unfused_time:.2f} ms")
    print(f"Speedup:         {unfused_time/fused_time:.2f}x")

    print("\n✓ Fused NF4 matmul test passed!")


def test_fused_fp4_matmul():
    """Test FP4 fused matmul."""
    from mps_bitsandbytes.functional import quantize_4bit, matmul_4bit

    device = "mps"
    M, K, N = 64, 2048, 2048

    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    input_tensor = torch.randn(M, K, dtype=torch.float16, device=device)

    weight_packed, quant_state = quantize_4bit(weight, blocksize=64, quant_type="fp4")
    output = matmul_4bit(input_tensor, weight_packed, quant_state)

    assert output.shape == (M, N)
    print(f"✓ FP4 fused matmul: output shape {output.shape}")


def test_batched_input():
    """Test fused kernel with batched input."""
    from mps_bitsandbytes.functional import quantize_4bit, matmul_4bit

    device = "mps"
    B, S, K, N = 2, 128, 1024, 1024

    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    input_tensor = torch.randn(B, S, K, dtype=torch.float16, device=device)

    weight_packed, quant_state = quantize_4bit(weight, blocksize=64, quant_type="nf4")
    output = matmul_4bit(input_tensor, weight_packed, quant_state)

    assert output.shape == (B, S, N)
    print(f"✓ Batched input: {input_tensor.shape} -> {output.shape}")


def test_with_bias():
    """Test fused kernel with bias."""
    from mps_bitsandbytes.functional import quantize_4bit, matmul_4bit

    device = "mps"
    M, K, N = 32, 512, 512

    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    bias = torch.randn(N, dtype=torch.float16, device=device)
    input_tensor = torch.randn(M, K, dtype=torch.float16, device=device)

    weight_packed, quant_state = quantize_4bit(weight, blocksize=64, quant_type="nf4")
    output = matmul_4bit(input_tensor, weight_packed, quant_state, bias=bias)

    assert output.shape == (M, N)
    print(f"✓ With bias: output shape {output.shape}")


if __name__ == "__main__":
    test_fused_nf4_matmul()
    test_fused_fp4_matmul()
    test_batched_input()
    test_with_bias()
    print("\n=== All fused NF4 tests passed! ===")
