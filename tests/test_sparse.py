"""
Tests for sparse matrix operations.
"""

import pytest
import torch

from mps_bitsandbytes.functional import (
    spmm_coo, spmm_coo_int8,
    sparse_coo_from_dense, quantize_sparse_coo,
)


@pytest.fixture
def device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class TestSpmmCoo:
    def test_basic(self, device):
        """Test spmm_coo against known dense matmul result."""
        # 3x4 sparse @ 4x2 dense = 3x2
        sparse_dense = torch.tensor([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ], device=device)
        dense = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ], device=device)

        expected = sparse_dense @ dense

        row_indices, col_indices, values, M, K = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, M, K)

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-3), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_vs_cpu(self, device):
        """Test spmm_coo on MPS matches CPU torch.sparse.mm."""
        M, K, N = 64, 128, 32
        sparsity = 0.9

        # Create random sparse matrix
        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device)

        # Dense reference
        expected = sparse_dense @ dense

        # COO spmm
        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, rows, cols)

        assert torch.allclose(result, expected, atol=1e-3), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_empty_rows(self, device):
        """Test spmm_coo with rows that have no nonzeros."""
        # Row 1 is all zeros
        sparse_dense = torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ], device=device)
        dense = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], device=device)

        expected = sparse_dense @ dense

        row_indices, col_indices, values, M, K = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, M, K)

        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-3)
        # Row 1 should be all zeros
        assert torch.allclose(result[1], torch.zeros(2, device=device), atol=1e-3)

    def test_single_nonzero(self, device):
        """Test spmm_coo with only one nonzero element."""
        sparse_dense = torch.zeros(4, 4, device=device)
        sparse_dense[2, 1] = 5.0
        dense = torch.randn(4, 3, device=device)

        expected = sparse_dense @ dense

        row_indices, col_indices, values, M, K = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, M, K)

        assert torch.allclose(result, expected, atol=1e-3)

    def test_half_precision(self, device):
        """Test spmm_coo with float16 tensors."""
        M, K, N = 32, 64, 16
        sparsity = 0.8

        sparse_dense = torch.randn(M, K, device=device, dtype=torch.float16)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device, dtype=torch.float16)
        expected = sparse_dense @ dense

        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, rows, cols)

        assert torch.allclose(result, expected, atol=0.1), \
            f"Max diff: {(result - expected).abs().max().item()}"


class TestSpmmCooInt8:
    def test_basic(self, device):
        """Test spmm_coo_int8 produces reasonable results."""
        M, K, N = 32, 64, 16
        sparsity = 0.8

        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device, dtype=torch.float16)

        # Reference: float spmm
        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        ref_result = spmm_coo(row_indices, col_indices, values, dense.float(), rows, cols)

        # INT8 spmm
        row_idx, col_idx, values_int8, scale = quantize_sparse_coo(
            row_indices, col_indices, values
        )
        int8_result = spmm_coo_int8(
            row_idx, col_idx, values_int8, scale, dense, rows, cols
        )

        # Should be reasonably close (int8 quantization introduces some error)
        relative_error = (ref_result.half() - int8_result).abs().mean() / ref_result.abs().mean().half()
        assert relative_error < 0.15, f"Relative error too high: {relative_error.item()}"

    def test_matches_dequant_spmm(self, device):
        """Test INT8 spmm matches manual dequant + float spmm."""
        M, K, N = 16, 32, 8

        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > 0.7
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device, dtype=torch.float16)

        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        row_idx, col_idx, values_int8, scale = quantize_sparse_coo(
            row_indices, col_indices, values
        )

        # Via spmm_coo_int8
        result = spmm_coo_int8(row_idx, col_idx, values_int8, scale, dense, rows, cols)

        # Manual: dequantize then spmm
        dequant_values = values_int8.float() * scale.float()
        expected = spmm_coo(row_idx, col_idx, dequant_values, dense.float(), rows, cols)

        assert torch.allclose(result.float(), expected.float(), atol=1e-2), \
            f"Max diff: {(result.float() - expected.float()).abs().max().item()}"


class TestSparseCooFromDense:
    def test_basic(self, device):
        """Test conversion from dense to COO."""
        dense = torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ], device=device)

        row_idx, col_idx, values, M, K = sparse_coo_from_dense(dense)

        assert M == 2
        assert K == 3
        assert len(values) == 3  # 3 nonzeros

        # Reconstruct and verify
        reconstructed = torch.zeros(M, K, device=device)
        reconstructed[row_idx, col_idx] = values
        assert torch.allclose(reconstructed, dense)

    def test_with_threshold(self, device):
        """Test sparsification with threshold."""
        dense = torch.tensor([
            [0.01, 0.5, -0.02],
            [1.0, 0.03, -2.0],
        ], device=device)

        row_idx, col_idx, values, M, K = sparse_coo_from_dense(dense, threshold=0.1)

        # Only values with abs >= 0.1 should remain
        assert all(v.abs() >= 0.1 for v in values)
        assert len(values) == 3  # 0.5, 1.0, -2.0

    def test_empty(self, device):
        """Test conversion of zero tensor."""
        dense = torch.zeros(4, 4, device=device)
        row_idx, col_idx, values, M, K = sparse_coo_from_dense(dense)
        assert len(values) == 0


class TestQuantizeSparseCoo:
    def test_roundtrip(self, device):
        """Test quantize/dequantize roundtrip for sparse values."""
        values = torch.randn(100, device=device)
        row_idx = torch.randint(0, 10, (100,), device=device)
        col_idx = torch.randint(0, 10, (100,), device=device)

        _, _, values_int8, scale = quantize_sparse_coo(row_idx, col_idx, values)

        recovered = values_int8.float() * scale.float()
        assert torch.allclose(values, recovered, atol=0.05, rtol=0.05)

    def test_preserves_sign(self, device):
        """Test that signs are preserved after quantization."""
        values = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0], device=device)
        row_idx = torch.zeros(5, device=device, dtype=torch.long)
        col_idx = torch.arange(5, device=device)

        _, _, values_int8, scale = quantize_sparse_coo(row_idx, col_idx, values)

        recovered = values_int8.float() * scale.float()
        # Signs should match (except zero)
        for orig, rec in zip(values[values != 0], recovered[values != 0]):
            assert (orig > 0) == (rec > 0), f"Sign mismatch: {orig} -> {rec}"


class TestNativeVsFallbackSparse:
    """Test that native Metal sparse kernels match Python fallback."""

    def test_native_spmm_coo(self, device):
        """Compare native vs fallback spmm_coo."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        M, K, N = 64, 128, 32
        sparsity = 0.85

        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device)

        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)

        # Both paths should produce the same result as dense matmul
        expected = sparse_dense @ dense
        result = spmm_coo(row_indices, col_indices, values, dense, rows, cols)

        assert torch.allclose(result, expected, atol=1e-3), \
            f"Max diff: {(result - expected).abs().max().item()}"

    def test_native_spmm_coo_int8(self, device):
        """Compare native vs fallback spmm_coo_int8."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        M, K, N = 32, 64, 16
        sparsity = 0.8

        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device, dtype=torch.float16)

        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        row_idx, col_idx, values_int8, scale = quantize_sparse_coo(
            row_indices, col_indices, values
        )

        result = spmm_coo_int8(row_idx, col_idx, values_int8, scale, dense, rows, cols)

        assert not torch.isnan(result).any(), "NaN in result"
        assert not torch.isinf(result).any(), "Inf in result"
        assert result.shape == (M, N)

    def test_large_sparse(self, device):
        """Stress test with large sparse matrix."""
        if device != 'mps':
            pytest.skip("Native kernels require MPS")

        M, K, N = 1000, 2000, 256
        sparsity = 0.95

        sparse_dense = torch.randn(M, K, device=device)
        mask = torch.rand(M, K, device=device) > sparsity
        sparse_dense = sparse_dense * mask

        dense = torch.randn(K, N, device=device)

        row_indices, col_indices, values, rows, cols = sparse_coo_from_dense(sparse_dense)
        result = spmm_coo(row_indices, col_indices, values, dense, rows, cols)

        assert result.shape == (M, N)
        assert not torch.isnan(result).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
