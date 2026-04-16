"""Torch-backend tests for ``chuk_lazarus.introspection.patcher``."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_import_patcher_with_torch_backend_does_not_load_mlx():
    code = (
        "import sys\n"
        "import chuk_lazarus.introspection.patcher  # noqa: F401\n"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if m.startswith('mlx'))\n"
        "assert 'mlx_lm' not in sys.modules\n"
    )
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = (
        os.pathsep.join([os.path.abspath(repo_src), existing_pp])
        if existing_pp
        else os.path.abspath(repo_src)
    )
    env = {**os.environ, "CHUK_BACKEND": "torch", "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def _make_runtime_bundle(device: str = "cpu"):
    torch = pytest.importorskip("torch")
    from chuk_lazarus.inference.backends import TorchInferenceRuntime
    from chuk_lazarus.introspection.patcher import ActivationPatcher, CommutativityAnalyzer

    class TinyTorchLayer(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, **_: object) -> torch.Tensor:
            return hidden_states

    class TinyTorchBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 3)
            self.layers = torch.nn.ModuleList([TinyTorchLayer(), TinyTorchLayer()])
            self.norm = torch.nn.Identity()

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            hidden = self.embed_tokens(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return self.norm(hidden)

    class TinyTorchForCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyTorchBackbone()
            self.lm_head = torch.nn.Linear(3, 8, bias=False)

            with torch.no_grad():
                self.model.embed_tokens.weight.zero_()
                self.model.embed_tokens.weight[1] = torch.tensor([2.0, 0.0, 0.0])
                self.model.embed_tokens.weight[2] = torch.tensor([0.0, 2.0, 0.0])
                self.model.embed_tokens.weight[3] = torch.tensor([0.0, 0.0, 2.0])
                self.model.embed_tokens.weight[4] = torch.tensor([0.0, 0.0, 2.0])

                self.lm_head.weight.zero_()
                self.lm_head.weight[5] = torch.tensor([1.0, 0.0, 0.0])
                self.lm_head.weight[6] = torch.tensor([0.0, 1.0, 0.0])
                self.lm_head.weight[7] = torch.tensor([0.0, 0.0, 1.0])

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            use_cache: bool = False,
        ) -> SimpleNamespace:
            del attention_mask, use_cache
            hidden = self.model(input_ids)
            return SimpleNamespace(logits=self.lm_head(hidden))

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def __init__(self) -> None:
            self._prompt_ids = {
                "source": 1,
                "target": 2,
                "2*3=": 3,
                "3*2=": 4,
            }
            self._token_text = {
                0: "<pad>",
                1: "source",
                2: "target",
                3: "2*3=",
                4: "3*2=",
                5: "source_answer",
                6: "target_answer",
                7: "comm_answer",
            }

        def __call__(self, prompt: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
            del return_tensors
            token_id = self._prompt_ids[prompt]
            input_ids = torch.tensor([[token_id]], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

        def encode(self, text: str, return_tensors: str | None = None):
            token_ids = [self._prompt_ids[text]]
            if return_tensors == "pt":
                return torch.tensor([token_ids], dtype=torch.long)
            return token_ids

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, list):
                return "".join(self._token_text[token_id] for token_id in ids)
            return self._token_text[ids]

    tokenizer = TinyTokenizer()
    model = TinyTorchForCausalLM().eval().to(device)
    runtime = TorchInferenceRuntime(model, tokenizer, device=device)
    config = SimpleNamespace(model_id=f"tiny-patcher:{device}")
    return torch, ActivationPatcher, CommutativityAnalyzer, model, tokenizer, config, runtime


def _available_torch_devices(torch_module) -> list[str]:
    devices = ["cpu"]
    if torch_module.cuda.is_available():
        devices.append("cuda:0")
    return devices


def test_activation_patcher_uses_torch_runtime_path():
    torch = pytest.importorskip("torch")

    for device in _available_torch_devices(torch):
        (
            _torch,
            ActivationPatcher,
            _CommutativityAnalyzer,
            model,
            tokenizer,
            config,
            runtime,
        ) = _make_runtime_bundle(device)
        patcher = ActivationPatcher(
            model=model,
            tokenizer=tokenizer,
            config=config,
            runtime=runtime,
        )

        activation = asyncio.run(patcher.capture_activation("source", layer=0))
        np.testing.assert_allclose(activation, np.array([2.0, 0.0, 0.0], dtype=np.float32))

        top_token, top_prob = asyncio.run(
            patcher.patch_and_predict(
                target_prompt="target",
                source_activation=activation,
                layer=0,
            )
        )
        assert top_token == "source_answer"
        assert 0.0 < top_prob <= 1.0

        result = asyncio.run(
            patcher.sweep_layers(
                target_prompt="target",
                source_prompt="source",
                layers=[0, 1],
                source_answer="source_answer",
                target_answer="target_answer",
            )
        )
        assert result.model_id == f"tiny-patcher:{device}"
        assert result.baseline_token == "target_answer"
        assert result.transferred_layers == [0, 1]


def test_commutativity_analyzer_uses_torch_runtime_path():
    torch = pytest.importorskip("torch")

    for device in _available_torch_devices(torch):
        (
            _torch,
            _ActivationPatcher,
            CommutativityAnalyzer,
            model,
            tokenizer,
            config,
            runtime,
        ) = _make_runtime_bundle(device)
        analyzer = CommutativityAnalyzer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            runtime=runtime,
        )

        result = asyncio.run(analyzer.analyze(layer=1, pairs=[("2*3=", "3*2=")]))

        assert result.model_id == f"tiny-patcher:{device}"
        assert result.num_pairs == 1
        assert result.layer == 1
        assert result.mean_similarity > 0.999
        assert result.pairs[0].similarity > 0.999
