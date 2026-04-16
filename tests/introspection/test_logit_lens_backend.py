"""Smoke test: importing chuk_lazarus.introspection.logit_lens under CHUK_BACKEND=torch must not pull MLX."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - host-dependent optional dependency
    torch = None


def test_import_logit_lens_with_torch_backend_does_not_load_mlx():
    code = (
        "import sys\n"
        "import chuk_lazarus.introspection.logit_lens  # noqa: F401\n"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if m.startswith('mlx'))\n"
        "assert 'mlx_lm' not in sys.modules\n"
    )
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([os.path.abspath(repo_src), existing_pp]) if existing_pp else os.path.abspath(repo_src)
    env = {**os.environ, "CHUK_BACKEND": "torch", "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


class TinyTorchBlock(torch.nn.Module if torch is not None else object):
    """Minimal transformer block for standalone logit-lens torch coverage."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return x + self.activation(self.linear(self.norm(x)))


class TinyTorchBackbone(torch.nn.Module if torch is not None else object):
    """Backbone exposing the structural seams ModelHooks expects."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 12, num_layers: int = 3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [TinyTorchBlock(hidden_size) for _ in range(num_layers)]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyTorchForCausalLM(torch.nn.Module if torch is not None else object):
    """Tiny causal LM wrapper for the standalone service path."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 12, num_layers: int = 3):
        super().__init__()
        self.model = TinyTorchBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor | None = None, **_: object) -> SimpleNamespace:
        assert input_ids is not None
        hidden = self.model(input_ids)
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyTokenizer:
    """Tokenizer stub with deterministic single-token decoding."""

    vocab_size = 32

    def encode(self, text: str) -> list[int]:
        tokens = [ord(ch) % self.vocab_size for ch in text[:5]]
        return tokens or [0]

    def decode(self, ids: list[int]) -> str:
        return f"<{ids[0]}>"

    def get_vocab(self) -> dict[str, int]:
        return {f"<{idx}>": idx for idx in range(self.vocab_size)}


@pytest.mark.asyncio
@pytest.mark.skipif(torch is None, reason="torch not installed")
async def test_logit_lens_service_uses_unified_pipeline_torch_runtime_path() -> None:
    from chuk_lazarus.introspection.logit_lens import LogitLensConfig, LogitLensService

    fake_backend = SimpleNamespace(
        name="torch",
        device="cpu",
        stop_gradient=lambda tensor: tensor.detach(),
    )
    fake_pipeline = SimpleNamespace(
        model=TinyTorchForCausalLM(),
        tokenizer=TinyTokenizer(),
        config=SimpleNamespace(
            num_hidden_layers=3,
            hidden_size=12,
            vocab_size=32,
        ),
        runtime=SimpleNamespace(backend="cuda"),
    )

    with patch(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        return_value=fake_backend,
    ) as mock_get_backend, patch(
        "chuk_lazarus.inference.UnifiedPipeline.from_pretrained",
        return_value=fake_pipeline,
    ) as mock_from_pretrained:
        result = await LogitLensService.analyze(
            LogitLensConfig(
                model="fake/model",
                prompt="torch",
                layers=[0, 2],
                top_k=3,
                track_tokens=["<1>"],
                backend="torch",
                device="cuda:7",
            )
        )

    mock_get_backend.assert_called_once_with(name="torch", device="cuda:7")
    call = mock_from_pretrained.call_args
    assert call.args == ("fake/model",)
    assert call.kwargs["pipeline_config"].backend_name == "torch"
    assert call.kwargs["pipeline_config"].device == "cpu"
    assert result.layers == [0, 2]
    assert set(result.predictions) == {0, 2}
    assert result.final_prediction.startswith("<")
    assert result.tracked_tokens is not None
    assert "<1>" in result.tracked_tokens
