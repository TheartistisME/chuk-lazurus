"""Scaled-dot-product attention MLX-vs-torch parity test.

M-x1 extended-coverage deliverable (see ve-ins-0mo7014jk000083602f
axis-3 baseline: "No parity test covers ... SDPA/attention"). We compare
each backend arm against a numpy reference of SDPA (``softmax(Q Kᵀ /
sqrt(d)) V``) to pin the cross-backend tolerance.

Tolerance: atol=5e-5, rtol=1e-4. Slightly looser than the general
circuit-op band because the softmax exponentiation amplifies float32
error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _helpers.mlx_snapshots import mlx_snapshot_assert  # noqa: E402

pytestmark = pytest.mark.parity


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sdpa_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q, k, v: [B, H, T, D]
    d = q.shape[-1]
    scores = np.einsum("bhid,bhjd->bhij", q, k) / np.sqrt(d)
    attn = _softmax(scores, axis=-1)
    return np.einsum("bhij,bhjd->bhid", attn, v)


def _has(name: str) -> bool:
    try:
        if name == "mlx":
            import mlx.core  # noqa: F401
        elif name == "torch":
            import torch  # noqa: F401
        else:
            return False
        return True
    except Exception:
        return False


BACKENDS = [
    pytest.param(
        "torch",
        marks=pytest.mark.skipif(not _has("torch"), reason="torch unavailable"),
    ),
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(not _has("mlx"), reason="mlx unavailable"),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
def test_sdpa_parity(backend: str) -> None:
    rng = np.random.default_rng(20260420)
    B, H, T, D = 2, 4, 7, 8
    q = rng.standard_normal((B, H, T, D)).astype(np.float32)
    k = rng.standard_normal((B, H, T, D)).astype(np.float32)
    v = rng.standard_normal((B, H, T, D)).astype(np.float32)
    ref = _sdpa_reference(q, k, v)

    if backend == "torch":
        import torch

        qt = torch.as_tensor(q)
        kt = torch.as_tensor(k)
        vt = torch.as_tensor(v)
        scores = torch.matmul(qt, kt.transpose(-2, -1)) / (D**0.5)
        attn = torch.softmax(scores, dim=-1)
        got = torch.matmul(attn, vt).detach().cpu().numpy()
    else:
        import mlx.core as mx

        qt = mx.array(q)
        kt = mx.array(k)
        vt = mx.array(v)
        scores = mx.matmul(qt, mx.swapaxes(kt, -2, -1)) / (D**0.5)
        attn = mx.softmax(scores, axis=-1)
        got = np.array(mx.matmul(attn, vt))

    mlx_snapshot_assert(
        f"models_v2.components.attention.sdpa.{backend}",
        got,
        ref,
        atol=5e-5,
        rtol=1e-4,
        inputs=[q, k, v],
    )
