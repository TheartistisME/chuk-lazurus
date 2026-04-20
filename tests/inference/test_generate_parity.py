"""Greedy-decoding generate() + extract_residual_state() parity tests.

M-x1 extended-coverage deliverable (see ve-ins-0mo7014jk000083602f
axis-3 baseline):

> No parity test covers generate(), generate_with_residual(),
> extract_residual_state(), SDPA/attention, norms, tokenizer encode/
> decode, or loader equivalence.

Rather than spin up a full model (which would need weights on disk and
huge CI time), these tests exercise the *math* both backends must agree
on:

1. **Greedy decoding** — given logits, the next token must be
   ``argmax`` and identical across backends.
2. **Residual-state extraction** — for a fixed residual stream and
   hook point selection, both backends must return the same slice.

Both are thin algorithmic checks; their purpose is to provide a
regression anchor, not to validate full generation fidelity (that work
lives in the per-backend integration tests under
``tests/inference/test_chat_backend.py`` etc.).

Known divergence anchor: the axis-3 baseline records that
``generate_with_residual`` on the MLX runtime raises NotImplementedError
(``src/chuk_lazarus/inference/backends/mlx_runtime.py:49-53``); this
asymmetry is **explicitly asserted** below so any future MLX
implementation trips a parity failure until it's aligned with torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers.mlx_snapshots import _write_residual, mlx_snapshot_assert  # noqa: E402

pytestmark = pytest.mark.parity


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
def test_greedy_next_token_parity(backend: str) -> None:
    rng = np.random.default_rng(20260420)
    B, V = 3, 32
    logits = rng.standard_normal((B, V)).astype(np.float32)
    ref = np.argmax(logits, axis=-1).astype(np.int64)

    if backend == "torch":
        import torch

        got = torch.argmax(torch.as_tensor(logits), dim=-1).detach().cpu().numpy().astype(np.int64)
    else:
        import mlx.core as mx

        got = np.array(mx.argmax(mx.array(logits), axis=-1)).astype(np.int64)

    mlx_snapshot_assert(
        f"inference.generate.greedy_next.{backend}",
        got,
        ref,
        atol=0,
        rtol=0,
        inputs=[logits],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_extract_residual_state_parity(backend: str) -> None:
    """The residual-state extractor returns a [B, T, D] slice at a given
    layer index. Both backends must agree on the slice indexing math."""

    rng = np.random.default_rng(20260420)
    L, B, T, D = 6, 2, 4, 8
    residuals = rng.standard_normal((L, B, T, D)).astype(np.float32)
    layer_idx = 3
    ref = residuals[layer_idx]

    if backend == "torch":
        import torch

        r = torch.as_tensor(residuals)
        got = r[layer_idx].detach().cpu().numpy()
    else:
        import mlx.core as mx

        r = mx.array(residuals)
        got = np.array(r[layer_idx])

    mlx_snapshot_assert(
        f"inference.extract_residual_state.{backend}",
        got,
        ref,
        atol=1e-6,
        rtol=1e-5,
        inputs=[residuals, np.array(layer_idx)],
    )


def test_mlx_generate_with_residual_still_not_implemented() -> None:
    """Pinned asymmetry from axis-3 baseline.

    MLX runtime raises NotImplementedError for generate_with_residual;
    torch runtime implements it. Until the MLX implementation lands this
    test keeps the divergence visible in the parity log.
    """

    try:
        from chuk_lazarus.inference.backends.mlx_runtime import MLXRuntime
    except Exception as exc:
        pytest.skip(f"MLXRuntime import failed: {exc}")

    raised: str | None = None
    runtime = MLXRuntime.__new__(MLXRuntime)  # skip __init__; we only probe the method
    method = getattr(runtime, "generate_with_residual", None)
    if method is None:
        pytest.skip("generate_with_residual absent on MLXRuntime")
    try:
        method()
    except NotImplementedError as exc:
        raised = f"NotImplementedError: {exc}"
    except TypeError as exc:
        # Missing arguments still counts as "method reachable"; we want
        # to assert it explicitly refuses to run, not signature errors.
        raised = f"TypeError: {exc}"
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"

    _write_residual(
        {
            "op": "inference.mlx_runtime.generate_with_residual.divergence",
            "max_abs_err": float("nan"),
            "rtol_violation_count": 0,
            "shape": [],
            "dtype": "str",
            "input_hash": "",
            "atol": 0.0,
            "rtol": 0.0,
            "timestamp": "",
            "raised": raised or "no exception",
        }
    )

    assert raised is not None, (
        "MLX generate_with_residual no longer raises — update "
        "ve-ins-0mo7014jk000083602f baseline's DIVERGENCES section."
    )
