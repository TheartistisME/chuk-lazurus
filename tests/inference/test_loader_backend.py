"""Tests for backend plumbing in inference.loader.resolve_pipeline_backend."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chuk_lazarus.inference import loader as loader_mod


class _Sentinel:
    pass


@pytest.mark.parametrize("name", ["mlx", "torch"])
def test_resolve_pipeline_backend_threads_config(monkeypatch, name):
    calls: list[dict] = []

    def fake_get_backend(name=None, device=None, check_sm=True):
        calls.append({"name": name, "device": device, "check_sm": check_sm})
        return _Sentinel()

    import chuk_lazarus.models_v2.core.backend as backend_pkg

    monkeypatch.setattr(backend_pkg, "get_backend", fake_get_backend)

    config = SimpleNamespace(
        backend_name=name,
        device="cuda:1" if name == "torch" else None,
        cuda_check_sm=False,
    )

    result = loader_mod.resolve_pipeline_backend(config)

    assert isinstance(result, _Sentinel)
    assert calls == [
        {
            "name": name,
            "device": "cuda:1" if name == "torch" else None,
            "check_sm": False,
        }
    ]


def test_resolve_pipeline_backend_defaults_none(monkeypatch):
    calls: list[dict] = []

    def fake_get_backend(name=None, device=None, check_sm=True):
        calls.append({"name": name, "device": device, "check_sm": check_sm})
        return _Sentinel()

    import chuk_lazarus.models_v2.core.backend as backend_pkg

    monkeypatch.setattr(backend_pkg, "get_backend", fake_get_backend)

    config = SimpleNamespace(backend_name=None, device=None, cuda_check_sm=True)
    loader_mod.resolve_pipeline_backend(config)

    assert calls == [{"name": None, "device": None, "check_sm": True}]
