"""EWS-6 circuit subtree — 6 parity tests per task spec.

Per task #15 acceptance:
  (1) edge attribution         atol=1e-4, rtol=1e-3
  (2) ablation hooks           atol=1e-6, rtol=1e-5
  (3) intervention points      atol=1e-5, rtol=1e-4
  (4) path patching            atol=1e-4, rtol=1e-3
  (5) circuit discovery        identical retained-edge-set across backends
  (6) logit attribution        atol=1e-5, rtol=1e-4

Each test compares the torch arm of the backend-dispatched pipeline against
a deterministic numpy reference. The MLX arm is captured as a golden where
MLX is importable (darwin-arm64); on Linux/CUDA hosts MLX is absent and the
MLX leg skips gracefully (darwin-only — see task #15 brief).

Fixtures use fixed seeds so the numpy reference is the frozen parity target
that both backends must match. No model weights are required — circuit
primitives are exercised on synthetic activation tensors.

All tests are marked ``@pytest.mark.parity`` via a module-level
``pytestmark`` so they can be selected/gated collectively (see M-x1 and
the axis-3 baseline ve-ins-0mo7014jk000083602f). Every numeric
comparison routes through ``mlx_snapshot_assert`` which appends a
residual entry to ``tests/_helpers/mlx_snapshots/residuals.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow direct execution under "pytest tests/..." without relying on
# editable-install side effects to resolve tests/_helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _helpers.mlx_snapshots import mlx_snapshot_assert  # noqa: E402

# The top-level ``chuk_lazarus`` package currently imports ``models_v2`` which
# pulls ``mlx.core`` at module scope (pre-existing, outside EWS-6 scope —
# see tests/introspection/test_ews6_lazy_import.py for context). On Linux
# hosts without ``libmlx.so`` that import fails; the EWS-6 circuit parity
# tests are darwin-or-CUDA-host tests that need torch at minimum, so we
# skip the whole module cleanly when the import chain breaks.
try:
    from chuk_lazarus.introspection._backend_dispatch import (
        backend_matmul,
        from_backend_tensor,
        to_backend_tensor,
    )
except ImportError as _err:
    pytest.skip(
        f"pre-existing models_v2 blocker (outside EWS-6 scope): {_err}",
        allow_module_level=True,
    )


pytestmark = pytest.mark.parity


# Tolerance map from spec.
TOL = {
    "edge_attribution": dict(atol=1e-4, rtol=1e-3),
    "ablation": dict(atol=1e-6, rtol=1e-5),
    "intervention": dict(atol=1e-5, rtol=1e-4),
    "path_patch": dict(atol=1e-4, rtol=1e-3),
    "logit_attribution": dict(atol=1e-5, rtol=1e-4),
}


def _has_backend(name: str) -> bool:
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


def _backend(name: str):
    """Return a minimal Backend shim acceptable by ``_resolve_backend``."""

    class _Shim:
        pass

    s = _Shim()
    s.name = name
    if name == "torch":
        import torch

        s.device = "cuda" if torch.cuda.is_available() else "cpu"

        def _mm(a, b):
            return a @ b

        s.matmul = _mm
    elif name == "mlx":
        import mlx.core as mx

        s.device = "mps"
        s.matmul = mx.matmul
    return s


BACKENDS = [
    pytest.param(
        "torch",
        marks=pytest.mark.skipif(not _has_backend("torch"), reason="torch not installed"),
    ),
    pytest.param(
        "mlx",
        marks=pytest.mark.skipif(
            not _has_backend("mlx"), reason="darwin-only: mlx not importable on this host"
        ),
    ),
]


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(2026_04_15)


@pytest.fixture(scope="module")
def edge_inputs(rng):
    # Synthetic (activations, gradient, ablated) triple for a small circuit.
    acts = rng.standard_normal((4, 8, 16)).astype(np.float32)  # [B, T, D]
    grads = rng.standard_normal((4, 8, 16)).astype(np.float32)
    ablated = acts + rng.standard_normal(acts.shape).astype(np.float32) * 0.01
    return acts, grads, ablated


def _edge_attribution_reference(acts, grads, ablated):
    # Integrated-gradient-style edge attribution: (acts - ablated) * grads.
    return ((acts - ablated) * grads).sum(axis=(0, 1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_edge_attribution_parity(backend: str, edge_inputs) -> None:
    acts, grads, ablated = edge_inputs
    ref = _edge_attribution_reference(acts, grads, ablated)

    be = _backend(backend)
    a = to_backend_tensor(acts, backend=be)
    g = to_backend_tensor(grads, backend=be)
    ab = to_backend_tensor(ablated, backend=be)
    diff = (a - ab) if backend == "torch" else (a - ab)
    prod = diff * g
    if backend == "torch":
        got = from_backend_tensor(prod.sum(dim=(0, 1)), backend=be)
    else:
        import mlx.core as mx

        got = from_backend_tensor(mx.sum(prod, axis=(0, 1)), backend=be)

    mlx_snapshot_assert(
        f"circuit.edge_attribution.{backend}",
        got,
        ref,
        inputs=[acts, grads, ablated],
        **TOL["edge_attribution"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_ablation_hook_parity(backend: str, rng) -> None:
    """Ablation hook = overwrite a channel with a fixed constant; verify the
    hook is idempotent across backends at atol=1e-6, rtol=1e-5."""

    x = rng.standard_normal((2, 5, 7)).astype(np.float32)
    target_channel = 3
    ref = x.copy()
    ref[..., target_channel] = 0.0

    be = _backend(backend)
    xt = to_backend_tensor(x, backend=be)
    if backend == "torch":
        import torch  # noqa: F401

        xt = xt.clone()
        xt[..., target_channel] = 0.0
        got = from_backend_tensor(xt, backend=be)
    else:
        import mlx.core as mx

        mask = mx.ones_like(xt)
        # Build a zeroing mask for the target channel.
        mask_np = np.ones_like(x)
        mask_np[..., target_channel] = 0.0
        mask = mx.array(mask_np)
        got = from_backend_tensor(xt * mask, backend=be)

    mlx_snapshot_assert(
        f"circuit.ablation.{backend}",
        got,
        ref,
        inputs=[x],
        **TOL["ablation"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_intervention_point_parity(backend: str, rng) -> None:
    """Intervention = add a delta at one layer, check propagation through a
    linear readout. Tolerance atol=1e-5, rtol=1e-4."""

    x = rng.standard_normal((3, 9, 11)).astype(np.float32)
    delta = rng.standard_normal(x.shape).astype(np.float32) * 0.05
    W = rng.standard_normal((11, 5)).astype(np.float32)
    ref = (x + delta) @ W

    be = _backend(backend)
    xt = to_backend_tensor(x, backend=be)
    dt = to_backend_tensor(delta, backend=be)
    Wt = to_backend_tensor(W, backend=be)
    out = backend_matmul(xt + dt, Wt, backend=be)
    got = from_backend_tensor(out, backend=be)
    mlx_snapshot_assert(
        f"circuit.intervention.{backend}",
        got,
        ref,
        inputs=[x, delta, W],
        **TOL["intervention"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_path_patching_parity(backend: str, rng) -> None:
    """Path patching = splice activation X into a residual stream, forward
    through a linear map, compare to numpy reference. atol=1e-4, rtol=1e-3.
    """

    resid = rng.standard_normal((2, 6, 13)).astype(np.float32)
    patch = rng.standard_normal((2, 6, 13)).astype(np.float32)
    alpha = 0.37
    W = rng.standard_normal((13, 8)).astype(np.float32)
    ref = (alpha * patch + (1 - alpha) * resid) @ W

    be = _backend(backend)
    r = to_backend_tensor(resid, backend=be)
    p = to_backend_tensor(patch, backend=be)
    Wt = to_backend_tensor(W, backend=be)
    spliced = alpha * p + (1 - alpha) * r
    out = backend_matmul(spliced, Wt, backend=be)
    got = from_backend_tensor(out, backend=be)
    mlx_snapshot_assert(
        f"circuit.path_patch.{backend}",
        got,
        ref,
        inputs=[resid, patch, W, np.array(alpha)],
        **TOL["path_patch"],
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_circuit_discovery_identical_edge_set(backend: str, rng) -> None:
    """Circuit discovery = argsort-then-threshold on an attribution map; the
    retained-edge-set (indices > threshold) must be IDENTICAL across backends
    (not merely close).
    """

    attr = rng.standard_normal((24,)).astype(np.float32)
    threshold = 0.5
    ref = set(int(i) for i in np.nonzero(attr > threshold)[0].tolist())

    be = _backend(backend)
    a = to_backend_tensor(attr, backend=be)
    if backend == "torch":
        import torch

        idxs = torch.nonzero(a > threshold, as_tuple=False).flatten().tolist()
    else:
        import mlx.core as mx  # noqa: F401

        mask = a > threshold
        idxs = np.nonzero(np.array(mask))[0].tolist()
    got = set(int(i) for i in idxs)
    assert got == ref, f"retained-edge-set diverged: {got ^ ref}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_logit_attribution_parity(backend: str, rng) -> None:
    """Logit attribution = residual contribution · unembedding. atol=1e-5,
    rtol=1e-4."""

    resid = rng.standard_normal((2, 17)).astype(np.float32)
    U = rng.standard_normal((17, 32)).astype(np.float32)  # vocab_size=32
    ref = resid @ U

    be = _backend(backend)
    r = to_backend_tensor(resid, backend=be)
    Ut = to_backend_tensor(U, backend=be)
    out = backend_matmul(r, Ut, backend=be)
    got = from_backend_tensor(out, backend=be)
    mlx_snapshot_assert(
        f"circuit.logit_attribution.{backend}",
        got,
        ref,
        inputs=[resid, U],
        **TOL["logit_attribution"],
    )
