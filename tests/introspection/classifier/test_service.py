"""Backend-aware classifier service tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.introspection.classifier.service import ClassifierConfig, ClassifierService


class FakeTorchRuntime:
    """Runtime stub for classifier-service torch coverage."""

    backend = LazarusBackend.CUDA

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        if prompt.startswith("add"):
            vector = [3.0 + layer_index, 0.0, 0.0]
        elif prompt.startswith("mult"):
            vector = [0.0, 3.0 + layer_index, 0.0]
        else:
            vector = [0.0, 0.0, 3.0 + layer_index]

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
        config=SimpleNamespace(num_hidden_layers=6),
        runtime=FakeTorchRuntime(),
    )


class TestClassifierService:
    """Tests for classifier service."""

    def test_import(self):
        """Test classifier service can be imported."""
        assert ClassifierService is not None

    @pytest.mark.asyncio
    async def test_train_and_evaluate_uses_runtime_hidden_states(self):
        """Test classifier training runs through the backend-aware runtime path."""
        categories = {
            "add": [f"add-{idx}" for idx in range(5)],
            "mult": [f"mult-{idx}" for idx in range(5)],
            "sub": [f"sub-{idx}" for idx in range(5)],
        }

        with patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=_fake_pipeline(),
        ) as mock_from_pretrained:
            result = await ClassifierService.train_and_evaluate(
                ClassifierConfig(
                    model="fake/model",
                    categories=categories,
                    layers=[1],
                    backend="torch",
                    device="cuda:3",
                )
            )

        call = mock_from_pretrained.call_args
        assert call.kwargs["pipeline_config"].backend_name == "torch"
        assert call.kwargs["pipeline_config"].device == "cuda:3"
        assert result.best_layer == 1
        assert result.best_accuracy >= 1.0
        assert result.categories == ["add", "mult", "sub"]
