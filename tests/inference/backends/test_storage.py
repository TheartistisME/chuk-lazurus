"""Tests for residual state serialization helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chuk_lazarus.inference.backends import (
    LazarusBackend,
    ResidualState,
    load_residual_state,
    prefetch_residual_state,
    save_residual_state,
)


def test_save_and_load_numpy_residual_state(tmp_path):
    state = ResidualState(
        backend=LazarusBackend.MLX,
        layer_index=3,
        tensor=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        sequence_length=10,
        hidden_size=3,
        dtype="float32",
        device="mps",
    )

    path = save_residual_state(tmp_path / "residual", state)
    loaded = load_residual_state(path)

    assert path.suffix == ".npz"
    assert loaded.backend == LazarusBackend.MLX
    assert loaded.layer_index == 3
    assert np.allclose(loaded.tensor, state.tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA storage test requires a CUDA GPU")
def test_save_and_load_cuda_residual_state(tmp_path):
    state = ResidualState(
        backend=LazarusBackend.CUDA,
        layer_index=1,
        tensor=torch.tensor([[1.0, 2.0, 3.0]], device="cuda"),
        sequence_length=12,
        hidden_size=3,
        dtype="float32",
        device="cuda",
    )

    path = save_residual_state(tmp_path / "residual", state)
    loaded = load_residual_state(path)

    assert path.suffix == ".safetensors"
    assert loaded.backend == LazarusBackend.CUDA
    assert tuple(loaded.tensor.shape) == (1, 3)


@pytest.mark.asyncio
async def test_prefetch_residual_state(tmp_path):
    state = ResidualState(
        backend=LazarusBackend.MLX,
        layer_index=0,
        tensor=np.array([[4.0, 5.0]], dtype=np.float32),
        sequence_length=4,
        hidden_size=2,
        dtype="float32",
        device="mps",
    )

    path = save_residual_state(tmp_path / "prefetch", state)
    loaded = await prefetch_residual_state(path)
    assert loaded.backend == LazarusBackend.MLX
    assert np.allclose(loaded.tensor, state.tensor)
