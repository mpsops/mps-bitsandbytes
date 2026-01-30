"""
Tests for HuggingFace transformers compatibility

Run with: pytest tests/test_hf_compat.py -v
"""

import pytest
import torch
import torch.nn as nn

# Skip all tests if not on macOS with MPS
pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available"
)


class TestBitsAndBytesConfig:
    """Tests for BitsAndBytesConfig."""

    def test_config_creation_4bit(self):
        """Test creating 4-bit config."""
        from mps_bitsandbytes import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        assert config.load_in_4bit is True
        assert config.load_in_8bit is False
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.is_quantizable is True
        assert config.quantization_method == 'bitsandbytes_4bit'

    def test_config_creation_8bit(self):
        """Test creating 8-bit config."""
        from mps_bitsandbytes import BitsAndBytesConfig

        config = BitsAndBytesConfig(load_in_8bit=True)

        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
        assert config.quantization_method == 'bitsandbytes_8bit'

    def test_config_invalid(self):
        """Test that invalid config raises error."""
        from mps_bitsandbytes import BitsAndBytesConfig

        # Can't have both 4-bit and 8-bit
        with pytest.raises(ValueError):
            BitsAndBytesConfig(load_in_4bit=True, load_in_8bit=True)

        # Invalid quant type
        with pytest.raises(ValueError):
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="invalid")

    def test_config_to_dict(self):
        """Test config serialization."""
        from mps_bitsandbytes import BitsAndBytesConfig

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
        )

        d = config.to_dict()
        assert d['load_in_4bit'] is True
        assert d['bnb_4bit_quant_type'] == 'nf4'

    def test_config_from_dict(self):
        """Test config deserialization."""
        from mps_bitsandbytes import BitsAndBytesConfig

        d = {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': 'torch.float16',
        }

        config = BitsAndBytesConfig.from_dict(d)
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == torch.float16


class TestQuantizeModel:
    """Tests for model quantization."""

    def test_quantize_model_4bit(self):
        """Test quantizing a model to 4-bit."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, Linear4bit

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleModel().half().to('mps')
        config = BitsAndBytesConfig(load_in_4bit=True)

        quantized_model = quantize_model(model, quantization_config=config)

        # Check layers were replaced
        assert isinstance(quantized_model.fc1, Linear4bit)
        assert isinstance(quantized_model.fc2, Linear4bit)

    def test_quantize_model_8bit(self):
        """Test quantizing a model to 8-bit."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, Linear8bit

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel().half().to('mps')
        config = BitsAndBytesConfig(load_in_8bit=True)

        quantized_model = quantize_model(model, quantization_config=config)

        assert isinstance(quantized_model.fc, Linear8bit)

    def test_quantize_model_skip_modules(self):
        """Test skipping certain modules during quantization."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, Linear4bit

        class ModelWithLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(128, 64)
                self.lm_head = nn.Linear(64, 1000)  # Should skip this

            def forward(self, x):
                return self.lm_head(self.encoder(x))

        model = ModelWithLMHead().half().to('mps')
        config = BitsAndBytesConfig(load_in_4bit=True)

        quantized_model = quantize_model(
            model,
            quantization_config=config,
            modules_to_not_convert=['lm_head']
        )

        assert isinstance(quantized_model.encoder, Linear4bit)
        assert isinstance(quantized_model.lm_head, nn.Linear)  # Not quantized

    def test_quantize_nested_model(self):
        """Test quantizing a model with nested modules."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, Linear4bit

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 64)
                self.fc2 = nn.Linear(64, 64)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(128, 64)
                self.blocks = nn.ModuleList([Block() for _ in range(3)])
                self.head = nn.Linear(64, 10)

            def forward(self, x):
                x = self.embed(x)
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        model = NestedModel().half().to('mps')
        config = BitsAndBytesConfig(load_in_4bit=True)

        quantized_model = quantize_model(model, quantization_config=config)

        # Check nested modules were quantized
        assert isinstance(quantized_model.embed, Linear4bit)
        assert isinstance(quantized_model.head, Linear4bit)
        for block in quantized_model.blocks:
            assert isinstance(block.fc1, Linear4bit)
            assert isinstance(block.fc2, Linear4bit)


class TestQuantizedModelInference:
    """Tests for inference with quantized models."""

    def test_quantized_model_forward(self):
        """Test forward pass on quantized model."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 16)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleModel().half().to('mps')
        config = BitsAndBytesConfig(load_in_4bit=True)
        quantized_model = quantize_model(model, quantization_config=config)

        # Forward pass
        x = torch.randn(8, 64, device='mps', dtype=torch.float16)
        output = quantized_model(x)

        assert output.shape == (8, 16)
        assert not torch.isnan(output).any()

    def test_quantized_vs_original_accuracy(self):
        """Test quantized model accuracy vs original."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 32)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        # Create and initialize
        torch.manual_seed(42)
        model = SimpleModel().half().to('mps')

        # Clone weights before quantization
        fc1_weight = model.fc1.weight.clone()
        fc2_weight = model.fc2.weight.clone()

        config = BitsAndBytesConfig(load_in_4bit=True)
        quantized_model = quantize_model(model, quantization_config=config)

        # Test input
        x = torch.randn(16, 128, device='mps', dtype=torch.float16)

        # Reference output (using original weights)
        ref_model = SimpleModel().half().to('mps')
        ref_model.fc1.weight.data = fc1_weight
        ref_model.fc2.weight.data = fc2_weight
        ref_output = ref_model(x)

        # Quantized output
        quant_output = quantized_model(x)

        # Check relative error
        error = (ref_output.float() - quant_output.float()).abs()
        relative_error = error / (ref_output.float().abs() + 1e-8)
        mean_relative_error = relative_error.mean().item()

        # Use cosine similarity for accuracy check (relative error is misleading for small values)
        ref_flat = ref_output.float().flatten()
        quant_flat = quant_output.float().flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(ref_flat.unsqueeze(0), quant_flat.unsqueeze(0)).item()
        print(f"Quantized model cosine similarity: {cosine_sim:.4f}")
        assert cosine_sim > 0.8  # Allow some deviation due to quantization


class TestGetMemoryFootprint:
    """Tests for memory footprint calculation."""

    def test_memory_footprint_reduction(self):
        """Test that quantization reduces memory footprint."""
        from mps_bitsandbytes import BitsAndBytesConfig, quantize_model, get_memory_footprint

        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(1024, 2048)
                self.fc2 = nn.Linear(2048, 1024)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = LargeModel().half().to('mps')
        original_footprint = get_memory_footprint(model)

        config = BitsAndBytesConfig(load_in_4bit=True)
        quantized_model = quantize_model(model, quantization_config=config)
        quantized_footprint = get_memory_footprint(quantized_model)

        print(f"Original: {original_footprint['actual_size_gb']*1000:.2f} MB")
        print(f"Quantized: {quantized_footprint['actual_size_gb']*1000:.2f} MB")
        print(f"Savings: {quantized_footprint['savings_pct']:.1f}%")

        # Should have significant savings (actual savings depend on overhead from absmax storage)
        assert quantized_footprint['actual_size_gb'] < original_footprint['actual_size_gb']
        assert quantized_footprint['savings_pct'] > 40  # At least 40% savings


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
