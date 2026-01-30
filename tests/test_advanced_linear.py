"""
Tests for advanced linear layers (OutlierAwareLinear, SwitchBackLinear).
"""

import pytest
import torch
import torch.nn as nn

from mps_bitsandbytes.nn import (
    OutlierAwareLinear,
    SwitchBackLinear, SwitchBackLinearCallback,
)


@pytest.fixture
def device():
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class TestOutlierAwareLinear:
    def test_from_linear(self, device):
        """Test conversion from nn.Linear."""
        linear = nn.Linear(256, 128).half().to(device)
        oa_linear = OutlierAwareLinear.from_linear(linear, device=device)

        assert oa_linear.in_features == 256
        assert oa_linear.out_features == 128

    def test_forward_shape(self, device):
        """Test forward pass produces correct shape."""
        linear = nn.Linear(256, 128).half().to(device)
        oa_linear = OutlierAwareLinear.from_linear(linear, device=device)

        x = torch.randn(8, 32, 256, device=device, dtype=torch.float16)
        output = oa_linear(x)

        assert output.shape == (8, 32, 128)

    def test_output_close_to_original(self, device):
        """Test output is close to FP16 reference."""
        linear = nn.Linear(128, 64, bias=True).half().to(device)
        oa_linear = OutlierAwareLinear.from_linear(linear, device=device)

        x = torch.randn(4, 16, 128, device=device, dtype=torch.float16)

        orig_output = linear(x)
        quant_output = oa_linear(x)

        # Should be reasonably close
        relative_error = (orig_output - quant_output).abs().mean() / orig_output.abs().mean()
        assert relative_error < 0.15

    def test_outlier_detection(self, device):
        """Test outlier columns are detected."""
        # Create weight with obvious outliers
        linear = nn.Linear(64, 32, bias=False).half().to(device)
        with torch.no_grad():
            linear.weight.fill_(0.1)
            # Add large outliers in specific columns
            linear.weight[:, 5] = 10.0
            linear.weight[:, 20] = 10.0

        oa_linear = OutlierAwareLinear.from_linear(linear, threshold=3.0, device=device)

        # Should detect the outlier columns
        assert len(oa_linear.outlier_indices) > 0

    def test_no_outliers(self, device):
        """Test layer works when no outliers detected."""
        linear = nn.Linear(64, 32).half().to(device)
        # Normal initialization shouldn't have outliers
        oa_linear = OutlierAwareLinear.from_linear(linear, threshold=10.0, device=device)

        x = torch.randn(4, 64, device=device, dtype=torch.float16)
        output = oa_linear(x)

        assert output.shape == (4, 32)

    def test_bias(self, device):
        """Test bias handling."""
        # With bias
        linear_bias = nn.Linear(64, 32, bias=True).half().to(device)
        oa_bias = OutlierAwareLinear.from_linear(linear_bias, device=device)
        assert oa_bias.bias is not None

        # Without bias
        linear_no_bias = nn.Linear(64, 32, bias=False).half().to(device)
        oa_no_bias = OutlierAwareLinear.from_linear(linear_no_bias, device=device)
        assert oa_no_bias.bias is None


class TestSwitchBackLinear:
    def test_from_linear(self, device):
        """Test conversion from nn.Linear."""
        linear = nn.Linear(256, 128).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)

        assert sb_linear.in_features == 256
        assert sb_linear.out_features == 128

    def test_forward_shape(self, device):
        """Test forward pass produces correct shape."""
        linear = nn.Linear(256, 128).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)

        x = torch.randn(8, 32, 256, device=device, dtype=torch.float16)
        output = sb_linear(x)

        assert output.shape == (8, 32, 128)

    def test_backward_pass(self, device):
        """Test backward pass computes gradients."""
        linear = nn.Linear(64, 32).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)

        x = torch.randn(4, 16, 64, device=device, dtype=torch.float16)
        output = sb_linear(x)
        loss = output.sum()
        loss.backward()

        # FP16 weights should have gradients
        assert sb_linear.weight_fp.grad is not None
        assert sb_linear.weight_fp.grad.abs().sum() > 0

    def test_output_close_to_original(self, device):
        """Test output is close to FP16 reference."""
        linear = nn.Linear(128, 64, bias=True).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)

        x = torch.randn(4, 16, 128, device=device, dtype=torch.float16)

        orig_output = linear(x)
        sb_output = sb_linear(x)

        # Should be very close (uses dequantized INT8)
        relative_error = (orig_output - sb_output).abs().mean() / orig_output.abs().mean()
        assert relative_error < 0.1

    def test_weight_sync(self, device):
        """Test INT8 weights can be synced from FP16."""
        linear = nn.Linear(64, 32).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)

        # Modify FP16 weights
        with torch.no_grad():
            sb_linear.weight_fp.fill_(0.5)

        # Sync should update INT8
        sb_linear.sync_weights()

        # INT8 weights should reflect the change
        expected_int8 = torch.full((32, 64), 127, dtype=torch.int8, device=device)
        assert torch.allclose(sb_linear.weight_int8.float(), expected_int8.float(), atol=1)

    def test_training_loop(self, device):
        """Test SwitchBackLinear in a training loop."""
        linear = nn.Linear(64, 32).half().to(device)
        sb_linear = SwitchBackLinear.from_linear(linear, device=device)
        sb_linear.train()

        optimizer = torch.optim.AdamW([sb_linear.weight_fp], lr=1e-2)

        x = torch.randn(8, 64, device=device, dtype=torch.float16)
        target = torch.randn(8, 32, device=device, dtype=torch.float16)

        initial_loss = ((sb_linear(x) - target) ** 2).mean().item()

        for _ in range(20):
            optimizer.zero_grad()
            loss = ((sb_linear(x) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            sb_linear.sync_weights()

        final_loss = ((sb_linear(x) - target) ** 2).mean().item()
        assert final_loss < initial_loss


class TestSwitchBackLinearCallback:
    def test_callback_syncs_all_layers(self, device):
        """Test callback syncs all SwitchBackLinear layers."""
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = SwitchBackLinear.from_linear(
                    nn.Linear(64, 32).half().to(device), device=device
                )
                self.fc2 = SwitchBackLinear.from_linear(
                    nn.Linear(32, 16).half().to(device), device=device
                )

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = Model()
        callback = SwitchBackLinearCallback(model)

        # Modify FP16 weights
        with torch.no_grad():
            model.fc1.weight_fp.fill_(0.1)
            model.fc2.weight_fp.fill_(0.2)

        # Sync all
        callback.sync()

        # Both layers should be updated
        assert model.fc1.weight_int8.float().mean().item() == pytest.approx(12.7, abs=1)  # 0.1 * 127
        assert model.fc2.weight_int8.float().mean().item() == pytest.approx(25.4, abs=1)  # 0.2 * 127


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
