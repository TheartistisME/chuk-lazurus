"""Torch runtime coverage for circuit service entrypoints."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.introspection.circuit import (
    CircuitCaptureConfig,
    CircuitDecodeConfig,
    CircuitInvokeConfig,
    CircuitService,
    CircuitTestConfig,
)
from chuk_lazarus.introspection.models.circuit import CapturedCircuit, CircuitDirection, CircuitEntry


TORCH_DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


class TorchCircuitLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TorchCircuitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([TorchCircuitLayer()])
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.model = SimpleNamespace(
            layers=self.layers,
            embed_tokens=self.embed_tokens,
            norm=self.norm,
        )

        with torch.no_grad():
            self.embed_tokens.weight.zero_()
            self.embed_tokens.weight[1] = torch.tensor([2.0, 0.0, 0.0, 0.0])
            self.embed_tokens.weight[2] = torch.tensor([-2.0, 0.0, 0.0, 0.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[3] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            self.lm_head.weight[4] = torch.tensor([-1.0, 0.0, 0.0, 0.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


class TorchCircuitTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        token_id = self._token_id(prompt)
        input_ids = torch.tensor([[token_id]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def encode(self, prompt, return_tensors=None):
        token_id = self._token_id(prompt)
        if return_tensors == "pt":
            return torch.tensor([[token_id]], dtype=torch.long)
        return [token_id]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            return "".join(f"tok-{int(token_id)}" for token_id in ids)
        return f"tok-{int(ids)}"

    @staticmethod
    def _token_id(prompt: str) -> int:
        if prompt == "alpha":
            return 1
        if prompt == "beta":
            return 2
        return 0


def _make_pipeline(device: str):
    model = TorchCircuitModel().eval().to(device)
    tokenizer = TorchCircuitTokenizer()
    runtime = TorchInferenceRuntime(model, tokenizer, device=device)
    return SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        config=SimpleNamespace(hidden_size=4, num_hidden_layers=1),
        runtime=runtime,
    )


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_circuit_service_capture_uses_torch_runtime(monkeypatch, tmp_path, device):
    import chuk_lazarus.inference as inference_module

    pipeline = _make_pipeline(device)
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_id, pipeline_config=None, verbose=False):
        captured["model_id"] = model_id
        captured["pipeline_config"] = pipeline_config
        captured["verbose"] = verbose
        return pipeline

    monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", fake_from_pretrained)

    output_path = tmp_path / "capture.npz"
    result = asyncio.run(
        CircuitService.capture(
            CircuitCaptureConfig(
                model="fake/test-model",
                prompts=["alpha", "beta"],
                layer=0,
                output_path=str(output_path),
                backend="torch",
                device=device,
            )
        )
    )

    saved = CapturedCircuit.load(output_path)

    assert captured["model_id"] == "fake/test-model"
    assert captured["pipeline_config"].backend_name == "torch"
    assert captured["pipeline_config"].device == device
    assert captured["verbose"] is False
    assert result.activations_shape == [2, 4]
    assert saved.layer == 0
    assert saved.entries[0].prompt == "alpha"
    assert saved.activations is not None
    np.testing.assert_allclose(saved.activations[0], np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32))


@pytest.mark.parametrize("device", TORCH_DEVICES)
def test_circuit_service_invoke_decode_and_test_use_torch_runtime(monkeypatch, tmp_path, device):
    import chuk_lazarus.inference as inference_module

    pipeline = _make_pipeline(device)
    calls: list[object] = []

    def fake_from_pretrained(model_id, pipeline_config=None, verbose=False):
        calls.append((model_id, pipeline_config, verbose))
        return pipeline

    monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", fake_from_pretrained)

    circuit_path = tmp_path / "direction.npz"
    activations = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    CapturedCircuit(
        model_id="fake/test-model",
        layer=0,
        entries=[
            CircuitEntry(prompt="alpha", result=1, activation=activations[0]),
            CircuitEntry(prompt="beta", result=0, activation=activations[1]),
        ],
        direction=CircuitDirection(
            direction=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            norm=1.0,
        ),
        activations=activations,
    ).save(circuit_path)

    invoke_result = asyncio.run(
        CircuitService.invoke(
            CircuitInvokeConfig(
                model="fake/test-model",
                circuit_file=str(circuit_path),
                prompts=["alpha", "beta"],
                method="steer",
                layer=0,
                backend="torch",
                device=device,
            )
        )
    )
    decode_result = asyncio.run(
        CircuitService.decode(
            CircuitDecodeConfig(
                model="fake/test-model",
                circuit_file=str(circuit_path),
                top_k=2,
                backend="torch",
                device=device,
            )
        )
    )
    test_result = asyncio.run(
        CircuitService.test(
            CircuitTestConfig(
                model="fake/test-model",
                circuit_file=str(circuit_path),
                prompts=["alpha", "beta"],
                expected_results=[1, 0],
                threshold=0.0,
                backend="torch",
                device=device,
            )
        )
    )

    assert len(calls) == 3
    for model_id, pipeline_config, verbose in calls:
        assert model_id == "fake/test-model"
        assert pipeline_config.backend_name == "torch"
        assert pipeline_config.device == device
        assert verbose is False

    assert invoke_result.results[0]["prediction"] == "positive"
    assert invoke_result.results[0]["score"] > 0
    assert invoke_result.results[1]["prediction"] == "negative"
    assert invoke_result.results[1]["score"] < 0
    assert decode_result.top_tokens[0]["token"] == "tok-3"
    assert test_result.accuracy == 1.0
    assert test_result.correct == 2
