"""Torch-runtime coverage for virtual expert service entrypoints."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.inference.generation import GenerationResult, GenerationStats, StopReason
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
from chuk_lazarus.inference.virtual_experts.registry import reset_default_registry
from chuk_lazarus.introspection.virtual_expert import VirtualExpertConfig, VirtualExpertService


class FakeTokenizer:
    """Minimal tokenizer stub for service-level tests."""

    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, list):
            return "".join(str(token) for token in ids)
        return str(ids)


class FakeDenseModel:
    """Dense model stub with config only."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=4, embedding_scale=None)


class FakeTorchRuntime:
    """Runtime stub for service-level torch path tests."""

    backend = LazarusBackend.CUDA

    def __init__(self):
        self.generated_prompts: list[str] = []
        self._layers = [SimpleNamespace() for _ in range(4)]

    def _resolve_layers(self):
        return self._layers

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        del layer_index
        prompt_lower = prompt.lower()
        is_math = any(op in prompt_lower for op in ("+", "-", "*", "/")) or '"expert": "math"' in prompt
        value = 4.0 if is_math else -4.0
        tensor = torch.tensor([[value, 0.0, 0.0, 0.0]], dtype=torch.float32)
        return ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=0,
            tensor=tensor,
            sequence_length=1,
            hidden_size=4,
            dtype="float32",
            device="cpu",
        )

    def generate(self, prompt: str, config) -> GenerationResult:
        del config
        self.generated_prompts.append(prompt)
        if "Action:" in prompt:
            text = (
                '{"expert":"math","operation":"evaluate","parameters":{"expression":"2 + 2"},'
                '"confidence":1.0,"reasoning":"math"}'
            )
        else:
            text = "model-output"

        return GenerationResult(
            text=text,
            stats=GenerationStats(
                input_tokens=1,
                output_tokens=1,
                total_time_seconds=0.01,
                tokens_per_second=100.0,
            ),
            stop_reason=StopReason.MAX_TOKENS,
        )


def _fake_pipeline():
    runtime = FakeTorchRuntime()
    return SimpleNamespace(
        model=FakeDenseModel(),
        tokenizer=FakeTokenizer(),
        config=SimpleNamespace(hidden_size=4, num_hidden_layers=4),
        runtime=runtime,
    )


@pytest.mark.asyncio
async def test_virtual_expert_service_solve_uses_unified_pipeline_backend_threading() -> None:
    reset_default_registry()
    pipeline = _fake_pipeline()

    with (
        patch.object(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["2 + 2 = "], ["Tell me a joke"]),
        ),
        patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ) as mock_from_pretrained,
    ):
        result = await VirtualExpertService.solve(
            VirtualExpertConfig(
                model="fake/model",
                backend="torch",
                device="cuda:1",
                prompt="2 + 2 = ",
            )
        )

    call = mock_from_pretrained.call_args
    assert call.args == ("fake/model",)
    assert call.kwargs["pipeline_config"].backend_name == "torch"
    assert call.kwargs["pipeline_config"].device == "cuda:1"
    assert result.answer == "4"
    assert result.results[0]["used_virtual_expert"] is True


@pytest.mark.asyncio
async def test_virtual_expert_service_compare_uses_runtime_direct_generation() -> None:
    reset_default_registry()
    pipeline = _fake_pipeline()

    with (
        patch.object(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["2 + 2 = "], ["Tell me a joke"]),
        ),
        patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ),
    ):
        result = await VirtualExpertService.compare(
            VirtualExpertConfig(
                model="fake/model",
                backend="torch",
                device="cuda:0",
                prompt="2 + 2 = ",
                verbose=True,
            )
    )

    assert result.results[0]["with_expert"] == "4"
    assert result.results[0]["without_expert"] == "model-output"
    assert result.summary["with_expert"] == "4"
    assert result.summary["without_expert"] == "model-output"
    assert pipeline.runtime.generated_prompts == ["2 + 2 = "]


@pytest.mark.asyncio
async def test_virtual_expert_service_benchmark_runs_on_torch_runtime() -> None:
    reset_default_registry()
    pipeline = _fake_pipeline()

    with (
        patch.object(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["2 + 2 = "], ["Tell me a joke"]),
        ),
        patch(
            "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
            return_value=pipeline,
        ),
    ):
        result = await VirtualExpertService.benchmark(
            VirtualExpertConfig(
                model="fake/model",
                backend="torch",
                device="cuda:0",
                benchmark_problems=[{"prompt": "2 + 2 = ", "expected": "4"}],
            )
        )

    assert result.accuracy == 1.0
    assert result.results[0]["correct"] is True
    assert result.results[0]["plugin_used"] == "math"
