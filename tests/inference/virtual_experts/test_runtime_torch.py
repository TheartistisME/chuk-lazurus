"""Torch-runtime coverage for virtual expert wrappers without MLX."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends.types import LazarusBackend, ResidualState
from chuk_lazarus.inference.generation import GenerationResult, GenerationStats, StopReason
from chuk_lazarus.inference.virtual_experts.cot_rewriter import FewShotCoTRewriter
from chuk_lazarus.inference.virtual_experts.plugins.math import MathExpertPlugin
from chuk_lazarus.inference.virtual_experts.dense_wrapper import VirtualDenseWrapper
from chuk_lazarus.inference.virtual_experts.registry import reset_default_registry
from chuk_lazarus.inference.virtual_experts.wrapper import VirtualMoEWrapper


class FakeTokenizer:
    """Minimal tokenizer stub for runtime-backed wrapper tests."""

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


class FakeMoELayer:
    """MoE layer stub exposing the router shape expected by the wrapper."""

    def __init__(self):
        self.mlp = SimpleNamespace(
            router=SimpleNamespace(
                weight=torch.zeros((2, 4), dtype=torch.float32),
                bias=torch.zeros((2,), dtype=torch.float32),
                num_experts=2,
                num_experts_per_tok=1,
            )
        )


class FakeMoEModel:
    """MoE model stub with config only."""

    def __init__(self):
        self.config = SimpleNamespace(hidden_size=4, embedding_scale=None)


class FakeTorchRuntime:
    """Backend-aware runtime stub for exercising torch/CUDA wrapper paths."""

    backend = LazarusBackend.CUDA

    def __init__(self, layers):
        self._layers = layers
        self.generated_prompts: list[str] = []

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


def test_dense_wrapper_routes_with_torch_runtime() -> None:
    reset_default_registry()
    runtime = FakeTorchRuntime([SimpleNamespace() for _ in range(4)])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["2 + 2 = "], ["Tell me a joke"]),
        )
        wrapper = VirtualDenseWrapper(FakeDenseModel(), FakeTokenizer(), "dense", runtime=runtime)
        result = wrapper.solve("2 + 2 = ")

    assert result.used_virtual_expert is True
    assert result.plugin_name == "math"
    assert result.answer == "4"
    assert runtime.generated_prompts == []


def test_dense_wrapper_uses_runtime_generation_for_passthrough() -> None:
    reset_default_registry()
    runtime = FakeTorchRuntime([SimpleNamespace() for _ in range(4)])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["2 + 2 = "], ["Tell me a joke"]),
        )
        wrapper = VirtualDenseWrapper(FakeDenseModel(), FakeTokenizer(), "dense", runtime=runtime)
        result = wrapper.solve("Tell me a joke")

    assert result.used_virtual_expert is False
    assert result.answer == "model-output"
    assert runtime.generated_prompts == ["Tell me a joke"]


def test_few_shot_rewriter_uses_runtime_generate() -> None:
    runtime = FakeTorchRuntime([SimpleNamespace() for _ in range(2)])
    rewriter = FewShotCoTRewriter(
        FakeDenseModel(),
        FakeTokenizer(),
        runtime=runtime,
        max_examples_per_expert=1,
    )

    action = rewriter.rewrite("What is 2 + 2?", ["math"])

    assert action.expert == "math"
    assert action.operation == "evaluate"
    assert runtime.generated_prompts and "Action:" in runtime.generated_prompts[0]


def test_moe_wrapper_routes_with_torch_runtime_trace() -> None:
    reset_default_registry()
    runtime = FakeTorchRuntime([FakeMoELayer() for _ in range(3)])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            MathExpertPlugin,
            "get_calibration_prompts",
            lambda self: (["7 * 8 = "], ["Tell me a joke"]),
        )
        wrapper = VirtualMoEWrapper(FakeMoEModel(), FakeTokenizer(), "moe", runtime=runtime)
        result = wrapper.solve("7 * 8 = ", verbose=True)

    assert result.used_virtual_expert is True
    assert result.plugin_name == "math"
    assert result.answer == "56"
    assert result.routing_trace is not None
    assert result.routing_trace.decisions
