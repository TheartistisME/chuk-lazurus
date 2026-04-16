"""Smoke test: importing chuk_lazarus.introspection.layer_analysis under CHUK_BACKEND=torch must not pull MLX."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_import_layer_analysis_with_torch_backend_does_not_load_mlx():
    code = (
        "import sys\n"
        "import chuk_lazarus.introspection.layer_analysis  # noqa: F401\n"
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


class TinyTokenizer:
    def encode(self, prompt: str):
        return [1] if prompt == "alpha" else [2]

    def decode(self, ids):
        if isinstance(ids, list):
            return "".join(f"tok-{int(token_id)}" for token_id in ids)
        return f"tok-{int(ids)}"


def test_layer_analyzer_load_threads_backend_and_device(monkeypatch):
    from chuk_lazarus.introspection.layer_analysis import LayerAnalyzer

    import chuk_lazarus.inference as inference_module

    fake_pipeline = SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=TinyTokenizer(),
        config=SimpleNamespace(num_hidden_layers=1),
        runtime=SimpleNamespace(backend="cuda", _device="cpu"),
    )
    captured: dict[str, object] = {}

    def fake_from_pretrained(model_id, pipeline_config=None, verbose=False):
        captured["model_id"] = model_id
        captured["pipeline_config"] = pipeline_config
        captured["verbose"] = verbose
        return fake_pipeline

    monkeypatch.setattr(inference_module.UnifiedPipeline, "from_pretrained", fake_from_pretrained)

    analyzer = LayerAnalyzer.load("fake/test-model", backend="torch", device="cpu", verbose=False)

    assert captured["model_id"] == "fake/test-model"
    assert captured["pipeline_config"].backend_name == "torch"
    assert captured["pipeline_config"].device == "cpu"
    assert captured["verbose"] is False
    assert analyzer._backend == "torch"
    assert analyzer._device == "cpu"


def test_layer_analyzer_analyze_representations_on_torch_backend(monkeypatch):
    from chuk_lazarus.introspection.layer_analysis import LayerAnalyzer
    import chuk_lazarus.introspection._backend_dispatch as backend_dispatch
    import chuk_lazarus.introspection.hooks as hooks_module
    import chuk_lazarus.introspection.layer_analysis as layer_analysis_module

    class FakeHooks:
        def __init__(self, model, model_config=None):
            self.state = SimpleNamespace(hidden_states={})

        def configure(self, config):
            return self

        def forward(self, input_ids):
            token_id = int(input_ids[0, 0])
            if token_id == 1:
                hidden = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
            else:
                hidden = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
            self.state.hidden_states = {0: hidden}

    monkeypatch.setattr(hooks_module, "ModelHooks", FakeHooks)
    monkeypatch.setattr(backend_dispatch, "to_backend_tensor", lambda tensor, backend=None: tensor)
    monkeypatch.setattr(layer_analysis_module, "_to_numpy", lambda tensor, **_: np.asarray(tensor))
    monkeypatch.setattr(
        layer_analysis_module,
        "_activate_backend",
        lambda backend, device=None: SimpleNamespace(name=backend, device=device),
    )

    analyzer = LayerAnalyzer(
        SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        TinyTokenizer(),
        model_id="fake/test-model",
        config=SimpleNamespace(num_hidden_layers=1),
        backend="torch",
        device="cpu",
    )

    result = analyzer.analyze_representations(
        prompts=["alpha", "beta"],
        layers=[0],
    )

    assert result.layers == [0]
    assert result.representations[0].similarity_matrix[0][0] == pytest.approx(1.0)
    assert result.representations[0].similarity_matrix[0][1] == pytest.approx(0.0)
    assert len(result.representations[0].similarity_matrix) == 2
