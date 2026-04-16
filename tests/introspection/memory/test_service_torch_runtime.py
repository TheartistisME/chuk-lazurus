"""Torch runtime coverage for the memory-analysis service."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chuk_lazarus.datasets import FactType
from chuk_lazarus.inference.backends import TorchInferenceRuntime
from chuk_lazarus.introspection.memory import MemoryAnalysisConfig, MemoryAnalysisService

torch = pytest.importorskip("torch")


class TorchIdentityLayer(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


class TorchMemoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.layers = torch.nn.ModuleList([TorchIdentityLayer()])
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(4, 16, bias=False)
        self.model = SimpleNamespace(
            layers=self.layers,
            embed_tokens=self.embed,
            norm=self.norm,
        )

        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[4] = torch.tensor([0.0, 0.0, 4.0, 0.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[7] = torch.tensor([0.0, 0.0, 1.0, 0.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


class TorchMemoryTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompt, return_tensors="pt"):
        token_id = 4 if prompt == "alpha" else 0
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def encode(self, prompt, return_tensors=None):
        token_id = 4 if prompt == "alpha" else 0
        tokens = [token_id]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, list):
            return f"tok-{ids[0]}"
        return f"tok-{ids}"


def test_memory_analysis_service_runs_real_torch_path(monkeypatch):
    import chuk_lazarus.inference as inference_module

    model = TorchMemoryModel().eval()
    tokenizer = TorchMemoryTokenizer()
    runtime = TorchInferenceRuntime(model, tokenizer, device="cpu")
    captured: dict[str, object] = {}

    fake_pipeline = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        config=SimpleNamespace(hidden_size=4, num_hidden_layers=1),
        runtime=runtime,
    )

    def fake_from_pretrained(model_id, pipeline_config=None, verbose=False):
        captured["model_id"] = model_id
        captured["pipeline_config"] = pipeline_config
        captured["verbose"] = verbose
        return fake_pipeline

    monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", fake_from_pretrained)

    result = asyncio.run(
        MemoryAnalysisService.analyze(
            MemoryAnalysisConfig(
                model="fake/test-model",
                facts=[{"query": "alpha", "answer": "tok-7", "category": "letters"}],
                fact_type=FactType.CUSTOM,
                layer=0,
                top_k=3,
                backend="torch",
                device="cpu",
            )
        )
    )

    assert captured["model_id"] == "fake/test-model"
    assert captured["pipeline_config"].backend_name == "torch"
    assert captured["pipeline_config"].device == "cpu"
    assert captured["verbose"] is False
    assert result.layer == 0
    assert result.top1_accuracy == 1
    assert result.top5_accuracy == 1
    assert result.not_found == 0
    assert result.results[0]["predictions"][0]["token"] == "tok-7"
