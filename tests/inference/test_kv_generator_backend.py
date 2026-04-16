"""Tests for kv_generator lazy-load behavior and backend import hygiene."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_kv_generator_does_not_load_mlx():
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        "importlib.import_module('chuk_lazarus.inference.context.kv_generator')\n"
        "bad = [k for k in sys.modules if k in {'mlx', 'mlx_lm'}]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_kv_generator_exports_make_factory():
    from chuk_lazarus.inference.context import kv_generator as m

    assert hasattr(m, "make_kv_generator")
    assert hasattr(m, "KVDirectGenerator")


def test_mask_dtype_helper_is_lazy(monkeypatch):
    from chuk_lazarus.inference.context import kv_generator as m

    calls: list[str] = []

    class _FakeMx:
        bfloat16 = "FAKE_BF16"

    def fake_mx():
        calls.append("mx")
        return _FakeMx()

    monkeypatch.setattr(m, "_mx", fake_mx)
    assert m._mask_dtype() == "FAKE_BF16"
    assert calls == ["mx"]


@pytest.mark.parametrize("backend_name", ["mlx", "torch"])
def test_resolve_pipeline_backend_threads_kv_config(monkeypatch, backend_name):
    """kv_generator shares the resolve_pipeline_backend dispatcher in loader."""
    from types import SimpleNamespace

    from chuk_lazarus.inference import loader as loader_mod

    calls: list[dict] = []

    def fake_get_backend(name=None, device=None, check_sm=True):
        calls.append({"name": name, "device": device, "check_sm": check_sm})
        return object()

    import chuk_lazarus.models_v2.core.backend as backend_pkg

    monkeypatch.setattr(backend_pkg, "get_backend", fake_get_backend)

    cfg = SimpleNamespace(backend_name=backend_name, device=None, cuda_check_sm=True)
    loader_mod.resolve_pipeline_backend(cfg)
    assert calls[-1]["name"] == backend_name
