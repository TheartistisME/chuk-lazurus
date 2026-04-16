"""Tests for neuron analysis service."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from chuk_lazarus.introspection.steering.neuron_service import (
    DiscoveredNeuron,
    NeuronActivationResult,
    NeuronAnalysisService,
    NeuronAnalysisServiceConfig,
)


class TestNeuronActivationResult:
    """Tests for NeuronActivationResult."""

    def test_basic_result(self):
        """Test basic neuron activation result."""
        result = NeuronActivationResult(
            neuron_idx=42,
            min_val=-1.0,
            max_val=1.0,
            mean_val=0.0,
            std_val=0.5,
        )
        assert result.neuron_idx == 42
        assert result.min_val == -1.0
        assert result.max_val == 1.0
        assert result.mean_val == 0.0
        assert result.std_val == 0.5
        assert result.weight is None
        assert result.separation is None

    def test_result_with_weight(self):
        """Test result with weight."""
        result = NeuronActivationResult(
            neuron_idx=10,
            min_val=-0.5,
            max_val=0.5,
            mean_val=0.1,
            std_val=0.2,
            weight=0.95,
        )
        assert result.weight == 0.95

    def test_result_with_separation(self):
        """Test result with separation score."""
        result = NeuronActivationResult(
            neuron_idx=5,
            min_val=-2.0,
            max_val=2.0,
            mean_val=0.0,
            std_val=1.0,
            separation=3.5,
        )
        assert result.separation == 3.5


class TestDiscoveredNeuron:
    """Tests for DiscoveredNeuron."""

    def test_basic_discovered_neuron(self):
        """Test basic discovered neuron."""
        neuron = DiscoveredNeuron(
            idx=100,
            separation=5.0,
            overall_std=0.8,
            mean_range=2.5,
        )
        assert neuron.idx == 100
        assert neuron.separation == 5.0
        assert neuron.overall_std == 0.8
        assert neuron.mean_range == 2.5
        assert neuron.best_pair is None
        assert neuron.group_means == {}

    def test_discovered_neuron_with_all_fields(self):
        """Test discovered neuron with all fields."""
        neuron = DiscoveredNeuron(
            idx=50,
            separation=8.0,
            best_pair=("positive", "negative"),
            overall_std=1.2,
            mean_range=4.0,
            group_means={"positive": 2.0, "negative": -2.0},
        )
        assert neuron.best_pair == ("positive", "negative")
        assert neuron.group_means["positive"] == 2.0
        assert neuron.group_means["negative"] == -2.0


class TestNeuronAnalysisServiceConfig:
    """Tests for NeuronAnalysisServiceConfig."""

    def test_basic_config(self):
        """Test basic config."""
        config = NeuronAnalysisServiceConfig(
            model="test-model",
            layers=[10, 11, 12],
        )
        assert config.model == "test-model"
        assert config.layers == [10, 11, 12]
        assert config.neurons is None
        assert config.top_k == 10

    def test_config_with_neurons(self):
        """Test config with specific neurons."""
        config = NeuronAnalysisServiceConfig(
            model="test-model",
            layers=[5],
            neurons=[10, 20, 30],
            top_k=5,
        )
        assert config.neurons == [10, 20, 30]
        assert config.top_k == 5


class TestNeuronAnalysisService:
    """Tests for NeuronAnalysisService."""

    def test_load_neurons_from_direction(self):
        """Test loading neurons from direction file."""
        # Create a mock direction file
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.array([0.1, -0.5, 0.3, -0.8, 0.2])
            np.savez(
                f.name,
                direction=direction,
                label_positive="happy",
                label_negative="sad",
            )

            neurons, weights, metadata = NeuronAnalysisService.load_neurons_from_direction(
                f.name, top_k=3
            )

        # Should return top 3 by absolute weight
        assert len(neurons) == 3
        assert 3 in neurons  # -0.8 has highest abs weight
        assert 1 in neurons  # -0.5 has second highest
        assert 2 in neurons or 4 in neurons  # 0.3 or 0.2

        # Check weights
        assert weights[3] == -0.8
        assert weights[1] == -0.5

        # Check metadata
        assert metadata["positive_label"] == "happy"
        assert metadata["negative_label"] == "sad"

        Path(f.name).unlink()

    def test_load_neurons_without_labels(self):
        """Test loading neurons without labels in file."""
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            direction = np.array([0.5, -0.5, 0.3])
            np.savez(f.name, direction=direction)

            neurons, weights, metadata = NeuronAnalysisService.load_neurons_from_direction(
                f.name, top_k=2
            )

        assert len(neurons) == 2
        assert metadata == {}

        Path(f.name).unlink()


torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState


class FakeRuntime:
    """Runtime stub for neuron-service extraction tests."""

    backend = LazarusBackend.CUDA

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        if prompt.startswith("easy"):
            vector = [3.0 + layer_index, 0.2, 0.0]
        elif prompt.startswith("hard"):
            vector = [-3.0 - layer_index, -0.2, 0.0]
        else:
            vector = [1.0 + layer_index, 0.0, 0.0]

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


def _fake_pipeline():
    return SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        config=SimpleNamespace(num_hidden_layers=6, hidden_size=3),
        runtime=FakeRuntime(),
    )


class AddLayer(torch.nn.Module):
    """Tiny layer that adds a fixed delta."""

    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta, dtype=torch.float32))

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return hidden_states + self.delta.view(1, 1, -1)


class ToyTorchModel(torch.nn.Module):
    """Tiny model for steering-aware capture tests."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList(
                [
                    AddLayer([1.0, 0.0]),
                    AddLayer([0.0, 1.0]),
                ]
            )
        )

    def forward(self, input_ids, use_cache: bool = False):
        del use_cache
        hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2), dtype=torch.float32)
        hidden[..., 0] = input_ids.to(torch.float32)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class ToyTorchRuntime:
    """Torch runtime stub with layer resolution and tokenization helpers."""

    backend = LazarusBackend.CUDA
    _device = torch.device("cpu")

    def __init__(self, model):
        self._model = model

    def _resolve_layers(self):
        return self._model.model.layers

    def _tokenize_prompt(self, prompt: str):
        token = 1 if prompt == "a" else 2
        return {"input_ids": torch.tensor([[token]], dtype=torch.long)}

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        captured = {}

        def hook(_module, _args, output):
            captured["tensor"] = output[:, -1, :].detach().to("cpu")

        handle = self._resolve_layers()[layer_index].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                self._model(**self._tokenize_prompt(prompt), use_cache=False)
        finally:
            handle.remove()

        return ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=layer_index,
            tensor=captured["tensor"],
            sequence_length=1,
            hidden_size=int(captured["tensor"].shape[-1]),
            dtype="float32",
            device="cpu",
        )


class TestNeuronAnalysisServiceTorchRuntime:
    """Torch/runtime coverage for neuron-service paths."""

    @pytest.mark.asyncio
    async def test_auto_discover_neurons_uses_runtime_extraction(self):
        """Test auto-discover uses UnifiedPipeline residual extraction."""
        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=_fake_pipeline(),
        ) as mock_from_pretrained:
            result = await NeuronAnalysisService.auto_discover_neurons(
                model="fake/model",
                prompts=["easy-1", "easy-2", "hard-1", "hard-2"],
                labels=["easy", "easy", "hard", "hard"],
                layer=2,
                top_k=2,
                backend="torch",
                device="cuda:0",
            )

        call = mock_from_pretrained.call_args
        assert call.kwargs["pipeline_config"].backend_name == "torch"
        assert call.kwargs["pipeline_config"].device == "cuda:0"
        assert len(result) == 2
        assert result[0].separation >= result[1].separation

    @pytest.mark.asyncio
    async def test_analyze_neurons_uses_runtime_extraction_without_steering(self):
        """Test neuron analysis without steering uses runtime residual capture."""
        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=_fake_pipeline(),
        ):
            result = await NeuronAnalysisService.analyze_neurons(
                model="fake/model",
                prompts=["easy-1", "hard-1"],
                neurons=[0],
                layers=[1],
                backend="torch",
                device="cuda:1",
            )

        assert 1 in result
        assert result[1][0].neuron_idx == 0
        assert result[1][0].max_val > result[1][0].min_val

    @pytest.mark.asyncio
    async def test_analyze_neurons_with_steering_uses_torch_capture_path(self):
        """Test steering-aware neuron analysis captures steered torch activations."""
        model = ToyTorchModel()
        pipeline = SimpleNamespace(
            model=model,
            tokenizer=SimpleNamespace(),
            config=SimpleNamespace(num_hidden_layers=2, hidden_size=2),
            runtime=ToyTorchRuntime(model),
        )

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ):
            baseline = await NeuronAnalysisService.analyze_neurons(
                model="fake/model",
                prompts=["a", "b"],
                neurons=[0],
                layers=[1],
                backend="torch",
                device="cuda:0",
            )
            steered = await NeuronAnalysisService.analyze_neurons(
                model="fake/model",
                prompts=["a", "b"],
                neurons=[0],
                layers=[1],
                steer_config={
                    "direction": np.array([1.0, 0.0], dtype=np.float32),
                    "layer": 0,
                    "coefficient": 1.0,
                },
                backend="torch",
                device="cuda:0",
            )

        assert steered[1][0].mean_val > baseline[1][0].mean_val
