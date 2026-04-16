"""Tests for LAZARUS_BACKEND <-> CHUK_BACKEND env bridging in resolve_backend."""

from __future__ import annotations

import pytest

from chuk_lazarus.inference.backends import registry
from chuk_lazarus.inference.backends.registry import (
    BACKEND_ENV_VAR,
    CHUK_BACKEND_ENV_VAR,
    resolve_backend,
)
from chuk_lazarus.inference.backends.types import LazarusBackend


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
    monkeypatch.delenv(CHUK_BACKEND_ENV_VAR, raising=False)


@pytest.fixture
def _always_available(monkeypatch):
    monkeypatch.setattr(registry, "is_backend_available", lambda backend: True)


def test_lazarus_backend_only(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "cuda")
    assert resolve_backend() == LazarusBackend.CUDA


def test_lazarus_backend_only_mlx(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
    assert resolve_backend() == LazarusBackend.MLX


def test_chuk_backend_torch_only_maps_to_cuda(monkeypatch, _always_available):
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "torch")
    assert resolve_backend() == LazarusBackend.CUDA


def test_chuk_backend_mlx_only(monkeypatch, _always_available):
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "mlx")
    assert resolve_backend() == LazarusBackend.MLX


def test_both_agree(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "cuda")
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "torch")
    assert resolve_backend() == LazarusBackend.CUDA


def test_both_agree_mlx(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "mlx")
    assert resolve_backend() == LazarusBackend.MLX


def test_both_conflict_raises(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "torch")
    with pytest.raises(RuntimeError, match="Conflicting backend env vars"):
        resolve_backend()


def test_neither_set_uses_auto_detect(monkeypatch, _always_available):
    calls = {"n": 0}

    def fake_detect():
        calls["n"] += 1
        return LazarusBackend.CUDA

    monkeypatch.setattr(registry, "detect_default_backend", fake_detect)
    assert resolve_backend() == LazarusBackend.CUDA
    assert calls["n"] == 1


def test_explicit_arg_overrides_env(monkeypatch, _always_available):
    monkeypatch.setenv(BACKEND_ENV_VAR, "cuda")
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "torch")
    assert resolve_backend(LazarusBackend.MLX) == LazarusBackend.MLX


def test_chuk_backend_invalid_value_raises(monkeypatch, _always_available):
    monkeypatch.setenv(CHUK_BACKEND_ENV_VAR, "tpu")
    with pytest.raises(RuntimeError, match="Unsupported CHUK_BACKEND"):
        resolve_backend()
