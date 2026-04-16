"""Backend-aware clustering service tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.introspection.clustering.service import ClusteringConfig, ClusteringService


class FakeTorchRuntime:
    """Runtime stub that exposes layer residual vectors."""

    backend = LazarusBackend.CUDA

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        base = 2.0 if "+" in prompt else -2.0
        tensor = torch.tensor([[base + layer_index, base, 0.0]], dtype=torch.float32)
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
        config=SimpleNamespace(num_hidden_layers=8),
        runtime=FakeTorchRuntime(),
    )


@pytest.mark.asyncio
async def test_clustering_service_uses_runtime_extraction_with_backend_overrides() -> None:
    with patch(
        "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
        return_value=_fake_pipeline(),
    ) as mock_from_pretrained:
        result = await ClusteringService.analyze(
            ClusteringConfig(
                model="fake/model",
                prompts=["2+2=", "3+3=", "8*8=", "9*9="],
                labels=["easy", "easy", "hard", "hard"],
                target_layers=[3],
                backend="torch",
                device="cuda:0",
            )
        )

    call = mock_from_pretrained.call_args
    assert call.args == ("fake/model",)
    assert call.kwargs["pipeline_config"].backend_name == "torch"
    assert call.kwargs["pipeline_config"].device == "cuda:0"
    assert result.model_id == "fake/model"
    assert result.unique_labels == ["easy", "hard"]
    assert result.layer_results[0]["layer"] == 3
    assert "ascii_grid" in result.layer_results[0]
