"""Tests for UnifiedPipelineConfig backend fields."""

from __future__ import annotations

import pytest

from chuk_lazarus.inference.unified import UnifiedPipelineConfig


def test_config_has_backend_fields():
    cfg = UnifiedPipelineConfig()
    assert cfg.backend is None
    assert cfg.backend_name is None
    assert cfg.device is None
    assert cfg.cuda_check_sm is True


@pytest.mark.parametrize("name", ["mlx", "torch"])
def test_config_accepts_backend_name(name):
    cfg = UnifiedPipelineConfig(
        backend_name=name,
        device="cuda:0" if name == "torch" else None,
        cuda_check_sm=False,
    )
    assert cfg.backend_name == name
    assert cfg.device == ("cuda:0" if name == "torch" else None)
    assert cfg.cuda_check_sm is False


def test_backend_field_description_references_chuk_backend():
    schema = UnifiedPipelineConfig.model_fields
    assert "CHUK_BACKEND" in schema["backend"].description
    assert "LAZARUS_BACKEND" not in schema["backend"].description


def test_resolve_pipeline_backend_called_via_config(monkeypatch):
    from chuk_lazarus.inference import loader as loader_mod

    calls: list[dict] = []

    class _Sentinel:
        pass

    def fake_get_backend(name=None, device=None, check_sm=True):
        calls.append({"name": name, "device": device, "check_sm": check_sm})
        return _Sentinel()

    import chuk_lazarus.models_v2.core.backend as backend_pkg

    monkeypatch.setattr(backend_pkg, "get_backend", fake_get_backend)

    cfg = UnifiedPipelineConfig(backend_name="torch", device="cuda:2", cuda_check_sm=False)
    loader_mod.resolve_pipeline_backend(cfg)

    assert calls == [{"name": "torch", "device": "cuda:2", "check_sm": False}]
