"""Tests for inference backend resolution."""

from __future__ import annotations

import pytest

from chuk_lazarus.inference.backends.registry import (
    BACKEND_ENV_VAR,
    detect_default_backend,
    resolve_backend,
)
from chuk_lazarus.inference.backends.types import LazarusBackend


class TestResolveBackend:
    """Tests for runtime backend selection."""

    def test_explicit_backend_wins(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
        backend = resolve_backend("cuda")
        assert backend == LazarusBackend.CUDA

    def test_env_backend_is_respected(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "cuda")
        backend = resolve_backend()
        assert backend == LazarusBackend.CUDA

    def test_unavailable_backend_raises(self, monkeypatch):
        from chuk_lazarus.inference.backends import registry as registry_module

        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        monkeypatch.setattr(registry_module, "is_backend_available", lambda backend: False)

        with pytest.raises(RuntimeError, match="MLX backend requested"):
            resolve_backend("mlx")

    def test_detect_default_backend_prefers_cuda_when_available(self, monkeypatch):
        from chuk_lazarus.inference.backends import registry as registry_module

        def fake_is_available(backend):
            return backend == LazarusBackend.CUDA

        monkeypatch.setattr(registry_module, "is_backend_available", fake_is_available)
        assert detect_default_backend() == LazarusBackend.CUDA
