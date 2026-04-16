"""Torch runtime coverage for the analyzer bucket."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chuk_lazarus.introspection.analyzer.config import AnalysisConfig, LayerStrategy
from chuk_lazarus.introspection.analyzer.core import ModelAnalyzer
from chuk_lazarus.introspection.hooks import CaptureConfig, ModelHooks
from chuk_lazarus.introspection.logit_lens import LogitLens
from chuk_lazarus.models_v2.core.backend import get_backend, reset_backend

torch = pytest.importorskip("torch")


class TinyTorchBlock(torch.nn.Module):
    """Small residual block that can emit synthetic attention weights."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        output_attentions: bool = False,
        **_: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = x + self.activation(self.linear(self.norm(x)))
        if not output_attentions:
            return hidden

        batch, seq_len, _ = hidden.shape
        attn = torch.zeros(
            batch,
            1,
            seq_len,
            seq_len,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return hidden, attn


class TinyTorchBackbone(torch.nn.Module):
    """Backbone with the same structural access patterns as loaded models."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 12, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [TinyTorchBlock(hidden_size) for _ in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        output_attentions: bool = False,
        **_: object,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden = self.embed_tokens(input_ids)
        attentions: list[torch.Tensor] | None = [] if output_attentions else None

        for layer in self.layers:
            layer_out = layer(hidden, output_attentions=output_attentions)
            if output_attentions:
                hidden, attn = layer_out
                assert attentions is not None
                attentions.append(attn)
            else:
                hidden = layer_out

        return self.norm(hidden), attentions


class TinyTorchForCausalLM(torch.nn.Module):
    """Minimal causal LM wrapper for analyzer/hook tests."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 12, num_layers: int = 3):
        super().__init__()
        self.model = TinyTorchBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_word_embeddings = False

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        assert input_ids is not None
        hidden, attentions = self.model(
            input_ids=input_ids,
            output_attentions=output_attentions,
        )
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            attentions=attentions,
        )


class TinyTokenizer:
    """Simple tokenizer for deterministic tests."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        encoded = [ord(ch) % self.vocab_size for ch in text[:5]]
        return encoded or [0]

    def decode(self, ids: list[int]) -> str:
        return f"<{ids[0]}>"

    def get_vocab(self) -> dict[str, int]:
        return {f"<{idx}>": idx for idx in range(self.vocab_size)}


@pytest.fixture
def torch_backend():
    reset_backend()
    backend = get_backend("torch", device="cpu")
    yield backend
    reset_backend()


def test_model_hooks_and_logit_lens_run_on_torch_backend(torch_backend):
    model = TinyTorchForCausalLM()
    tokenizer = TinyTokenizer()

    hooks = ModelHooks(model)
    hooks.configure(
        CaptureConfig(
            layers=[0, 1, 2],
            capture_hidden_states=True,
            capture_attention_weights=True,
            positions="all",
        )
    )

    input_ids = torch_backend.from_numpy(np.array([[1, 2, 3]], dtype=np.int64))
    logits = hooks.forward(input_ids)

    assert isinstance(logits, torch.Tensor)
    assert hooks.state.captured_layers == [0, 1, 2]
    assert all(isinstance(hidden, torch.Tensor) for hidden in hooks.state.hidden_states.values())
    assert all(
        isinstance(attn, torch.Tensor) for attn in hooks.state.attention_weights.values()
    )

    lens = LogitLens(hooks, tokenizer)
    predictions = lens.get_layer_predictions(position=-1, top_k=3)
    evolution = lens.track_token(1, position=-1)

    assert len(predictions) == 3
    assert all(len(pred.top_ids) == 3 for pred in predictions)
    assert evolution.layers == [0, 1, 2]


def test_model_analyzer_runs_real_torch_path(torch_backend):
    model = TinyTorchForCausalLM()
    tokenizer = TinyTokenizer()
    config = SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=12,
        vocab_size=32,
        tie_word_embeddings=False,
    )

    analyzer = ModelAnalyzer(model, tokenizer, model_id="tiny-torch", config=config)
    result = analyzer._analyze_sync(
        "torch",
        AnalysisConfig(
            layer_strategy=LayerStrategy.ALL,
            top_k=3,
            compute_entropy=True,
            compute_transitions=True,
        ),
    )

    assert result.prompt == "torch"
    assert result.captured_layers == [0, 1, 2]
    assert len(result.layer_predictions) == 3
    assert len(result.final_prediction) == 3
    assert len(result.layer_transitions) == 2
    assert all(pred.probability >= 0.0 for pred in result.final_prediction)
