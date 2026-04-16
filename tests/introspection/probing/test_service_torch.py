"""Torch-style runtime tests for probing services."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.introspection.probing.service import (
    ProbeConfig,
    ProbeService,
    UncertaintyConfig,
    UncertaintyService,
)


class FakeTorchRuntime:
    """Runtime stub that emits prompt-dependent residual vectors."""

    backend = LazarusBackend.CUDA

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        if prompt.startswith("working"):
            vector = [3.0 + layer_index, 0.5]
        elif prompt.startswith("broken"):
            vector = [-3.0 - layer_index, -0.5]
        elif prompt.startswith("positive"):
            vector = [2.0 + layer_index, 1.0]
        elif prompt.startswith("negative"):
            vector = [-2.0 - layer_index, -1.0]
        else:
            vector = [0.0, 0.0]

        tensor = torch.tensor([vector], dtype=torch.float32)
        return ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=layer_index,
            tensor=tensor,
            sequence_length=1,
            hidden_size=2,
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


@pytest.mark.asyncio
async def test_uncertainty_service_uses_runtime_hidden_states() -> None:
    with patch(
        "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
        return_value=_fake_pipeline(),
    ) as mock_from_pretrained:
        result = await UncertaintyService.analyze(
            UncertaintyConfig(
                model="fake/model",
                prompts=["working-test", "broken-test"],
                working_prompts=["working-1", "working-2"],
                broken_prompts=["broken-1", "broken-2"],
                layer=2,
                backend="torch",
                device="cuda:1",
            )
        )

    call = mock_from_pretrained.call_args
    assert call.kwargs["pipeline_config"].backend_name == "torch"
    assert call.kwargs["pipeline_config"].device == "cuda:1"
    assert result.detection_layer == 2
    assert result.confident_count == 1
    assert result.uncertain_count == 1
    assert [entry["prediction"] for entry in result.results] == ["CONFIDENT", "UNCERTAIN"]


@pytest.mark.asyncio
async def test_probe_service_trains_on_runtime_hidden_states() -> None:
    with patch(
        "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
        return_value=_fake_pipeline(),
    ) as mock_from_pretrained:
        result = await ProbeService.train_and_evaluate(
            ProbeConfig(
                model="fake/model",
                positive_prompts=[f"positive-{idx}" for idx in range(5)],
                negative_prompts=[f"negative-{idx}" for idx in range(5)],
                positive_label="math",
                negative_label="other",
                layers=[0, 2],
                cross_val_folds=2,
                backend="torch",
                device="cuda:0",
            )
        )

    call = mock_from_pretrained.call_args
    assert call.kwargs["pipeline_config"].backend_name == "torch"
    assert call.kwargs["pipeline_config"].device == "cuda:0"
    assert [entry["layer"] for entry in result.layer_results] == [0, 2]
    assert result.best_layer in {0, 2}
    assert result.best_accuracy >= 1.0
