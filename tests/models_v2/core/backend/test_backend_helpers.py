"""Round-trip tests for ``Backend.array``/``save``/``load`` (EWS-0)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _skip_if_missing(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception:
        pytest.skip(f"{name} not importable on this host")


SHAPES = [(), (4,), (2, 3)]


def _make_numpy(shape):
    if shape == ():
        return np.asarray(1.5, dtype=np.float32)
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(shape, dtype=np.float32)


# ---- Torch ------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_torch_array_roundtrip(shape, tmp_path):
    _skip_if_missing("torch")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("torch", device="cpu", check_sm=False)
    data = _make_numpy(shape)
    tensor = backend.array(data)
    assert tuple(tensor.shape) == shape
    np.testing.assert_allclose(backend.to_numpy(tensor), data)


@pytest.mark.parametrize("shape", SHAPES)
def test_torch_save_load_roundtrip(shape, tmp_path):
    _skip_if_missing("torch")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("torch", device="cpu", check_sm=False)
    data = _make_numpy(shape)
    tensor = backend.array(data)
    path = str(tmp_path / f"torch_{len(shape)}.pt")
    backend.save(tensor, path)
    loaded = backend.load(path)
    np.testing.assert_allclose(backend.to_numpy(loaded), data)


def test_torch_save_dict_roundtrip(tmp_path):
    _skip_if_missing("torch")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("torch", device="cpu", check_sm=False)
    data = {
        "a": backend.array(_make_numpy((4,))),
        "b": backend.array(_make_numpy((2, 3))),
    }
    path = str(tmp_path / "state.pt")
    backend.save(data, path)
    loaded = backend.load(path)
    assert set(loaded.keys()) == {"a", "b"}
    for k in data:
        np.testing.assert_allclose(backend.to_numpy(loaded[k]), backend.to_numpy(data[k]))


# ---- MLX --------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_mlx_array_roundtrip(shape):
    _skip_if_missing("mlx.core")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("mlx")
    data = _make_numpy(shape)
    tensor = backend.array(data)
    assert tuple(tensor.shape) == shape
    np.testing.assert_allclose(backend.to_numpy(tensor), data)


@pytest.mark.parametrize("shape", SHAPES)
def test_mlx_save_load_roundtrip(shape, tmp_path):
    _skip_if_missing("mlx.core")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("mlx")
    data = _make_numpy(shape)
    tensor = backend.array(data)
    path = str(tmp_path / f"mlx_{len(shape)}.npz")
    backend.save(tensor, path)
    loaded = backend.load(path)
    np.testing.assert_allclose(backend.to_numpy(loaded), data)


def test_mlx_save_dict_roundtrip(tmp_path):
    _skip_if_missing("mlx.core")
    from chuk_lazarus.models_v2.core.backend import get_backend

    backend = get_backend("mlx")
    data = {
        "a": backend.array(_make_numpy((4,))),
        "b": backend.array(_make_numpy((2, 3))),
    }
    path = str(tmp_path / "state.npz")
    backend.save(data, path)
    loaded = backend.load(path)
    assert set(loaded.keys()) == {"a", "b"}
    for k in data:
        np.testing.assert_allclose(backend.to_numpy(loaded[k]), backend.to_numpy(data[k]))
