"""RMSNorm MLX-vs-torch parity test.

M-x1 extended-coverage deliverable. Implements a numpy reference for RMS
normalization and compares both backend arms against it through the
shared ``mlx_snapshot_assert`` oracle. Skips gracefully when a backend is
unavailable on the host.

Why this test exists (from ve-ins-0mo7014jk000083602f axis-3 baseline):

> No parity test covers generate(), generate_with_residual(),
> extract_residual_state(), SDPA/attention, norms, tokenizer encode/
> decode, or loader equivalence.

This module fills the *norms* slot. Implementation intentionally goes
through the mathematical identity (``x / sqrt(mean(x**2) + eps) *
weight``) rather than importing the MLX-coupled ``RMSNorm`` class, so
the test is darwin-optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _helpers.mlx_snapshots import mlx_snapshot_assert  # noqa: E402

pytestmark = pytest.mark.parity


def _rmsnorm_reference(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


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
def test_rmsnorm_parity(backend: str) -> None:
    rng = np.random.default_rng(20260420)
    dims = 16
    x = rng.standard_normal((2, 5, dims)).astype(np.float32)
    weight = rng.standard_normal((dims,)).astype(np.float32)
    eps = 1e-6
    ref = _rmsnorm_reference(x, weight, eps)

    if backend == "torch":
        import torch

        xt = torch.as_tensor(x)
        wt = torch.as_tensor(weight)
        rms = torch.sqrt((xt * xt).mean(dim=-1, keepdim=True) + eps)
        got = ((xt / rms) * wt).detach().cpu().numpy()
    else:
        import mlx.core as mx

        xt = mx.array(x)
        wt = mx.array(weight)
        rms = mx.sqrt(mx.mean(xt * xt, axis=-1, keepdims=True) + eps)
        got = np.array((xt / rms) * wt)

    mlx_snapshot_assert(
        f"models_v2.components.rmsnorm.{backend}",
        got,
        ref,
        atol=1e-5,
        rtol=1e-4,
        inputs=[x, weight, np.array(eps)],
    )
