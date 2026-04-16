"""Tests for steering service."""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from chuk_lazarus.introspection.steering.service import (
    DirectionExtractionResult,
    SteeringComparisonResult,
    SteeringGenerationResult,
    SteeringService,
    SteeringServiceConfig,
)


class TestSteeringServiceConfig:
    """Tests for SteeringServiceConfig."""

    def test_create_config(self):
        """Test creating steering service config."""
        config = SteeringServiceConfig(
            model="/path/to/model",
            layer=10,
            coefficient=0.5,
            max_tokens=50,
            temperature=0.7,
        )

        assert config.model == "/path/to/model"
        assert config.layer == 10
        assert config.coefficient == 0.5
        assert config.max_tokens == 50
        assert config.temperature == 0.7

    def test_default_values(self):
        """Test default values."""
        config = SteeringServiceConfig(model="/path/to/model")

        assert config.layer is None
        assert config.coefficient == 1.0
        assert config.max_tokens == 100
        assert config.temperature == 0.0

    def test_frozen_config(self):
        """Test that config is frozen (immutable)."""
        from pydantic import ValidationError

        config = SteeringServiceConfig(model="/path/to/model")

        with pytest.raises(ValidationError):
            config.model = "new_model"


class TestDirectionExtractionResult:
    """Tests for DirectionExtractionResult."""

    def test_create_result(self):
        """Test creating direction extraction result."""
        direction = np.random.randn(128).astype(np.float32)

        result = DirectionExtractionResult(
            direction=direction,
            layer=5,
            norm=1.5,
            cosine_similarity=0.3,
            separation=0.7,
            positive_prompt="Be helpful",
            negative_prompt="Be harmful",
        )

        assert result.layer == 5
        assert result.norm == 1.5
        assert result.cosine_similarity == 0.3
        assert result.separation == 0.7
        assert result.positive_prompt == "Be helpful"
        assert result.negative_prompt == "Be harmful"


class TestSteeringGenerationResult:
    """Tests for SteeringGenerationResult."""

    def test_create_result(self):
        """Test creating steering generation result."""
        result = SteeringGenerationResult(
            prompt="Hello",
            output="Hi there!",
            layer=10,
            coefficient=1.5,
        )

        assert result.prompt == "Hello"
        assert result.output == "Hi there!"
        assert result.layer == 10
        assert result.coefficient == 1.5


class TestSteeringComparisonResult:
    """Tests for SteeringComparisonResult."""

    def test_create_result(self):
        """Test creating steering comparison result."""
        result = SteeringComparisonResult(
            prompt="Hello",
            results={0.0: "Baseline", 1.0: "Steered", -1.0: "Reverse"},
        )

        assert result.prompt == "Hello"
        assert result.results[0.0] == "Baseline"
        assert result.results[1.0] == "Steered"
        assert result.results[-1.0] == "Reverse"


class TestSteeringServiceSaveDirection:
    """Tests for SteeringService.save_direction."""

    def test_save_direction(self):
        """Test saving direction to file."""
        direction = np.random.randn(128).astype(np.float32)
        result = DirectionExtractionResult(
            direction=direction,
            layer=5,
            norm=1.5,
            cosine_similarity=0.3,
            separation=0.7,
            positive_prompt="positive",
            negative_prompt="negative",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            SteeringService.save_direction(result, path, "model-id")

            assert path.exists()

            # Verify contents
            loaded = np.load(path, allow_pickle=True)
            assert "direction" in loaded
            assert "layer" in loaded
            assert "positive_prompt" in loaded
            assert "negative_prompt" in loaded


class TestSteeringServiceLoadDirection:
    """Tests for SteeringService.load_direction."""

    def test_load_direction_npz(self):
        """Test loading direction from npz file."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            np.savez(
                path,
                direction=direction,
                layer=10,
                positive_prompt="pos",
                negative_prompt="neg",
                norm=1.5,
                cosine_similarity=0.3,
            )

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer == 10
            assert metadata["positive_prompt"] == "pos"
            assert metadata["negative_prompt"] == "neg"
            assert metadata["norm"] == 1.5
            assert metadata["cosine_similarity"] == 0.3

    def test_load_direction_npz_minimal(self):
        """Test loading direction from npz with minimal fields."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.npz"

            np.savez(path, direction=direction)

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer is None
            assert metadata == {}

    def test_load_direction_json(self):
        """Test loading direction from json file."""
        direction = np.random.randn(128).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.json"

            with open(path, "w") as f:
                json.dump(
                    {
                        "direction": direction.tolist(),
                        "layer": 5,
                        "extra_field": "test",
                    },
                    f,
                )

            loaded_direction, layer, metadata = SteeringService.load_direction(path)

            assert np.allclose(loaded_direction, direction)
            assert layer == 5
            assert metadata["extra_field"] == "test"

    def test_load_direction_unsupported_format(self):
        """Test loading direction from unsupported format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "direction.txt"
            path.write_text("test")

            with pytest.raises(ValueError, match="Unsupported direction format"):
                SteeringService.load_direction(path)


class TestSteeringServiceCreateNeuronDirection:
    """Tests for SteeringService.create_neuron_direction."""

    def test_create_neuron_direction(self):
        """Test creating one-hot neuron direction."""
        direction = SteeringService.create_neuron_direction(hidden_size=128, neuron_idx=50)

        assert direction.shape == (128,)
        assert direction.dtype == np.float32
        assert direction[50] == 1.0
        assert np.sum(direction) == 1.0  # Only one non-zero element

    def test_create_neuron_direction_first(self):
        """Test creating direction for first neuron."""
        direction = SteeringService.create_neuron_direction(hidden_size=64, neuron_idx=0)

        assert direction[0] == 1.0
        assert np.sum(direction) == 1.0

    def test_create_neuron_direction_last(self):
        """Test creating direction for last neuron."""
        direction = SteeringService.create_neuron_direction(hidden_size=64, neuron_idx=63)

        assert direction[63] == 1.0
        assert np.sum(direction) == 1.0


class TestSteeringServiceConfigAlias:
    """Tests for Config class alias."""

    def test_config_alias(self):
        """Test that SteeringService.Config is SteeringServiceConfig."""
        assert SteeringService.Config == SteeringServiceConfig


torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.inference.generation import GenerationResult, GenerationStats, StopReason


class FakeTorchRuntime:
    """Runtime stub for backend-aware steering service tests."""

    backend = LazarusBackend.CUDA

    def __init__(self):
        self.generated = []

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        if "positive" in prompt:
            vector = [3.0, 0.0, 0.0]
        elif "negative" in prompt:
            vector = [-1.0, 0.0, 0.0]
        else:
            vector = [1.0, 0.5, -0.5]

        tensor = torch.tensor([vector], dtype=torch.float32)
        return ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=layer_index,
            tensor=tensor,
            sequence_length=1,
            hidden_size=3,
            dtype="float32",
            device="cpu",
        )

    def generate_with_residual(self, prompt: str, residual_state: ResidualState, config) -> GenerationResult:
        self.generated.append((prompt, residual_state, config))
        return GenerationResult(
            text=f"steered:{prompt}",
            stats=GenerationStats(
                input_tokens=1,
                output_tokens=1,
                total_time_seconds=0.01,
                tokens_per_second=100.0,
            ),
            stop_reason=StopReason.MAX_TOKENS,
        )


def _fake_pipeline():
    return SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        config=SimpleNamespace(num_hidden_layers=8, hidden_size=3),
        runtime=FakeTorchRuntime(),
    )


class TestSteeringServiceTorchRuntime:
    """Torch/runtime coverage for steering service."""

    @pytest.mark.asyncio
    async def test_extract_direction_uses_unified_pipeline_backend_overrides(self):
        """Test direction extraction through the runtime seam."""
        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=_fake_pipeline(),
        ) as mock_from_pretrained:
            result = await SteeringService.extract_direction(
                model="fake/model",
                positive_prompt="positive prompt",
                negative_prompt="negative prompt",
                backend="torch",
                device="cuda:1",
            )

        call = mock_from_pretrained.call_args
        assert call.kwargs["pipeline_config"].backend_name == "torch"
        assert call.kwargs["pipeline_config"].device == "cuda:1"
        assert result.layer == 4
        assert np.allclose(result.direction, np.array([4.0, 0.0, 0.0], dtype=np.float32))

    @pytest.mark.asyncio
    async def test_generate_with_steering_uses_torch_residual_injection(self):
        """Test torch steering generation goes through runtime.generate_with_residual."""
        pipeline = _fake_pipeline()
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ):
            result = await SteeringService.generate_with_steering(
                model="fake/model",
                prompts=["hello"],
                direction=direction,
                layer=2,
                coefficient=1.5,
                backend="torch",
                device="cuda:0",
            )

        assert result[0].output == "steered:hello"
        assert len(pipeline.runtime.generated) == 1
        residual_state = pipeline.runtime.generated[0][1]
        assert type(residual_state.tensor).__module__.startswith("torch")

    @pytest.mark.asyncio
    async def test_compare_coefficients_uses_torch_runtime_generation(self):
        """Test compare mode uses runtime residual injection for each coefficient."""
        pipeline = _fake_pipeline()
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ):
            result = await SteeringService.compare_coefficients(
                model="fake/model",
                prompt="hello",
                direction=direction,
                layer=1,
                coefficients=[-1.0, 0.0, 1.0],
                backend="torch",
                device="cuda:0",
            )

        assert sorted(result.results) == [-1.0, 0.0, 1.0]
        assert len(pipeline.runtime.generated) == 3
