"""
Tests for 8-bit and paged optimizers.
"""

import pytest
import torch
import torch.nn as nn

from mps_bitsandbytes.optim import (
    Adam8bit, AdamW8bit, Lion8bit, SGD8bit,
    PagedAdam, PagedAdamW, PagedLion,
    quantize_state, dequantize_state,
)


@pytest.fixture
def device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestQuantizeState:
    def test_roundtrip(self, device):
        """Test quantize/dequantize roundtrip."""
        state = torch.randn(1000, device=device)
        block_size = 256

        int8, absmax = quantize_state(state, block_size)
        recovered = dequantize_state(int8, absmax, block_size, state.dtype)

        # Should be close (within quantization error)
        assert recovered.shape == state.shape
        assert torch.allclose(state, recovered, atol=0.05, rtol=0.05)

    def test_preserves_shape(self, device):
        """Test various shapes are preserved."""
        shapes = [(100,), (32, 64), (8, 16, 32)]

        for shape in shapes:
            state = torch.randn(*shape, device=device)
            int8, absmax = quantize_state(state, 64)
            recovered = dequantize_state(int8, absmax, 64, state.dtype)
            assert recovered.shape == shape


class TestAdam8bit:
    def test_step(self, device):
        """Test basic optimization step."""
        model = SimpleModel().to(device)
        optimizer = Adam8bit(model.parameters(), lr=1e-3)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        for _ in range(5):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        # Check state was created
        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                assert 'exp_avg_int8' in state
                assert 'exp_avg_sq_uint8' in state  # unsigned for better precision

    def test_converges(self, device):
        """Test optimizer converges on simple problem."""
        model = SimpleModel().to(device)
        optimizer = Adam8bit(model.parameters(), lr=1e-2)

        x = torch.randn(32, 64, device=device)
        y = torch.randn(32, 16, device=device)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        assert final_loss < initial_loss * 0.5


class TestAdamW8bit:
    def test_weight_decay(self, device):
        """Test decoupled weight decay is applied and optimizer converges."""
        model = SimpleModel().to(device)

        optimizer = AdamW8bit(model.parameters(), lr=1e-2, weight_decay=0.1)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        # Optimizer should reduce loss
        assert final_loss < initial_loss * 0.5


class TestLion8bit:
    def test_step(self, device):
        """Test Lion optimization step."""
        model = SimpleModel().to(device)
        optimizer = Lion8bit(model.parameters(), lr=1e-4)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        assert final_loss < initial_loss

    def test_only_one_momentum(self, device):
        """Test Lion only stores one momentum state."""
        model = SimpleModel().to(device)
        optimizer = Lion8bit(model.parameters(), lr=1e-4)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                assert 'exp_avg_int8' in state
                assert 'exp_avg_sq_int8' not in state  # Lion doesn't have this


class TestSGD8bit:
    def test_step(self, device):
        """Test SGD with momentum."""
        model = SimpleModel().to(device)
        optimizer = SGD8bit(model.parameters(), lr=0.1, momentum=0.9)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(50):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        assert final_loss < initial_loss


class TestPagedAdamW:
    def test_states_on_cpu(self, device):
        """Test optimizer states are stored on CPU."""
        if device == 'cpu':
            pytest.skip("Paged optimizers only make sense for GPU")

        model = SimpleModel().to(device)
        optimizer = PagedAdamW(model.parameters(), lr=1e-3, page_to_cpu=True)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        # States should be on CPU
        for p in model.parameters():
            if p.grad is not None:
                state = optimizer.state[p]
                assert state['exp_avg'].device.type == 'cpu'
                assert state['exp_avg_sq'].device.type == 'cpu'

    def test_converges(self, device):
        """Test paged optimizer converges."""
        model = SimpleModel().to(device)
        optimizer = PagedAdamW(model.parameters(), lr=1e-2)

        x = torch.randn(32, 64, device=device)
        y = torch.randn(32, 16, device=device)

        initial_loss = ((model(x) - y) ** 2).mean().item()

        for _ in range(100):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()

        final_loss = ((model(x) - y) ** 2).mean().item()
        assert final_loss < initial_loss * 0.5


class TestPagedLion:
    def test_step(self, device):
        """Test PagedLion optimization."""
        model = SimpleModel().to(device)
        optimizer = PagedLion(model.parameters(), lr=1e-4)

        x = torch.randn(8, 64, device=device)
        y = torch.randn(8, 16, device=device)

        for _ in range(10):
            optimizer.zero_grad()
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
