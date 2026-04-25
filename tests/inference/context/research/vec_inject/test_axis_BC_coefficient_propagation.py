"""axis-BC PROP K.5 — coefficient propagation parametric tests (run-4).

Bug authority
-------------
ve-ins-0moe6w4su0000096c6a — the run-4 bug record establishing that
``vec_inject_to_kv_direct`` MUST scale the per-match raw (k_vec, v_vec)
returned by ``LocalVecInjectProvider.kv_for_match`` by ``float(m.coefficient)``
before stacking + broadcasting into the KVDirectGenerator page tensor.
The patch lives at ``kv_direct_adapter.py:225-236``.

Axis-1 end-state
----------------
ve-ins-0moebmc0b0000f914ac — acceptance criterion: parametric coverage of
the coefficient grid {0.0, 0.5, 1.0, 2.0} plus a sign-flip case (-2.0)
demonstrating that the adapter wires the upstream tier-aware ranking weight
all the way to the materialised K/V page; without scaling, the cold/warm/hot
ranking collapses to uniform weight at injection time.

Hardware
--------
CUDA-only per AMD 14 (RTX 5090 + Gemma-4-E2B-it bf16). The module-level
skipif below is the ONLY skip; per-test CPU fallbacks are forbidden.

Layer choice
------------
target_layer=29 — member of the axis-A global-attention fixture
``GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`` (AMD 11)
so the PROP K.0 sliding-window guard accepts every call.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chuk_lazarus.inference.context.research.vec_inject._types import (
    SourceType,
    VecInjectMatch,
)
from chuk_lazarus.inference.context.research.vec_inject.kv_direct_adapter import (
    vec_inject_to_kv_direct,
)
from chuk_lazarus.inference.context.research.vec_inject.providers._index_format import (
    VEC_INJECT_FILE,
    VecInjectMetaKey,
    VecInjectWindowKey,
)
from chuk_lazarus.inference.context.research.vec_inject.providers._local_file_torch import (
    LocalVecInjectProvider,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only: torch.cuda.is_available()==False",
)


HEAD_DIM = 256
N_FACTS = 4
WINDOW_ID = 0


def _write_synthetic_npz(path, head_dim: int = HEAD_DIM, n_facts: int = N_FACTS) -> None:
    """Build a single-window NPZ carrying both k_vecs and v_vecs.

    Seed is intentionally distinct from the sibling
    ``test_axis_BC_kv_direct_adapter.py`` (which uses 12345) so this suite
    exercises a different draw of K/V values — guarding against accidental
    fixture aliasing under the omitted-norm pipeline.
    """
    rng = np.random.default_rng(seed=54321)
    arrays: dict[str, np.ndarray] = {
        VecInjectMetaKey.LAYER: np.array(29, dtype=np.int32),
        VecInjectMetaKey.KV_HEAD: np.array(0, dtype=np.int32),
        VecInjectMetaKey.QUERY_HEAD: np.array(4, dtype=np.int32),
        VecInjectMetaKey.INJECT_LAYER: np.array(30, dtype=np.int32),
    }
    arrays[VecInjectWindowKey.k_vecs(WINDOW_ID)] = rng.standard_normal(
        (n_facts, head_dim)
    ).astype(np.float32)
    arrays[VecInjectWindowKey.v_vecs(WINDOW_ID)] = rng.standard_normal(
        (n_facts, head_dim)
    ).astype(np.float32)
    arrays[VecInjectWindowKey.token_ids(WINDOW_ID)] = np.arange(
        100, 100 + n_facts, dtype=np.int32
    )
    arrays[VecInjectWindowKey.coefs(WINDOW_ID)] = np.full(n_facts, 0.5, dtype=np.float32)
    arrays[VecInjectWindowKey.positions(WINDOW_ID)] = np.arange(n_facts, dtype=np.int32)
    arrays[VecInjectWindowKey.distinctive(WINDOW_ID)] = np.ones(n_facts, dtype=np.int32)
    np.savez(str(path), **arrays)


@pytest.fixture(scope="function")
def synthetic_provider(tmp_path):
    npz_path = tmp_path / VEC_INJECT_FILE
    _write_synthetic_npz(npz_path)
    return LocalVecInjectProvider._from_npz(
        npz_path,
        kv_gen=None,
        has_injection=True,
    )


def _make_match(
    coefficient: float,
    position: int = 0,
    window_id: int = WINDOW_ID,
) -> VecInjectMatch:
    """Single-match factory with a caller-supplied coefficient.

    The coefficient is the tested axis — defaults from the sibling test
    (0.5) are intentionally NOT honoured here, every test passes a value
    drawn from the axis-1 acceptance matrix {0.0, 0.5, 1.0, 2.0, -2.0}.
    """
    return VecInjectMatch(
        token_id=100 + position,
        coefficient=coefficient,
        score=0.9,
        window_id=window_id,
        position=position,
        distinctive=True,
        source_id=window_id,
        source_type=SourceType.DOCUMENT,
    )


def _expected_scaled(raw_vec: torch.Tensor, coef: float) -> torch.Tensor:
    """Replicate the adapter's _to_cuda_bf16 cast + scale step exactly.

    Provider.kv_for_match already returns cuda bf16 tensors, so
    _to_cuda_bf16 is a no-op; this helper is the per-fact ground truth for
    the (1, n_kv_heads, n_facts, head_dim) page slot after broadcast.
    """
    cast = raw_vec.to(device="cuda", dtype=torch.bfloat16)
    return cast * coef


# ── tests ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("position", [0, 1, 2])
def test_coefficient_zero_produces_all_zero_kv(synthetic_provider, position):
    """coefficient=0.0 (cold-state) MUST zero the materialised K/V page.

    Bug authority claim: scaling by float(m.coefficient) is applied
    per-match BEFORE the (1, n_kv_heads, n_facts, head_dim) broadcast. With
    coef=0 every element of the resulting page tensor is exactly zero —
    not "small", not "tolerance-bounded": strict zero, asserted with
    ``torch.equal``. This is the boundary case that proves the multiply
    is wired (a no-op identity adapter would leave the raw K/V intact).
    """
    match = _make_match(coefficient=0.0, position=position)

    pages, sizes, offset = vec_inject_to_kv_direct(
        [match],
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )

    k_page, v_page = pages[0][0]

    # Shape / dtype / device contract — same as sibling shape test.
    assert k_page.shape == (1, 1, 1, HEAD_DIM)
    assert v_page.shape == (1, 1, 1, HEAD_DIM)
    assert k_page.dtype == torch.bfloat16
    assert v_page.dtype == torch.bfloat16
    assert k_page.device.type == "cuda"
    assert v_page.device.type == "cuda"
    assert sizes == [1]
    assert offset == 0

    # The propagation claim — strict zero everywhere.
    assert torch.equal(k_page, torch.zeros_like(k_page)), (
        f"coefficient=0 at position={position} must produce all-zero K page; "
        f"observed nonzero count = {(k_page != 0).sum().item()}"
    )
    assert torch.equal(v_page, torch.zeros_like(v_page)), (
        f"coefficient=0 at position={position} must produce all-zero V page; "
        f"observed nonzero count = {(v_page != 0).sum().item()}"
    )


def test_coefficient_one_matches_unscaled_kv(synthetic_provider):
    """coefficient=1.0 MUST leave raw K/V intact (modulo bf16 cast).

    Bug authority claim: ``coef = float(m.coefficient); k * coef`` is the
    propagation operator. At coef=1.0 the scaling is the identity and the
    materialised page must equal the cast raw K/V broadcast across
    n_kv_heads. We assert ``torch.equal`` because no rounding is introduced
    once both sides are already bf16 — the multiply by exactly 1.0 in bf16
    is bit-exact.
    """
    match = _make_match(coefficient=1.0, position=0)
    raw_k, raw_v = synthetic_provider.kv_for_match(match)

    pages, sizes, _ = vec_inject_to_kv_direct(
        [match],
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )

    k_page, v_page = pages[0][0]
    n_kv_heads = k_page.shape[1]
    assert k_page.shape == (1, n_kv_heads, 1, HEAD_DIM)
    assert sizes == [1]

    expected_k = _expected_scaled(raw_k, 1.0)
    expected_v = _expected_scaled(raw_v, 1.0)

    # Single fact (n_facts=1) at the only kv-head slot in the synthetic provider.
    for h in range(n_kv_heads):
        assert torch.equal(k_page[0, h, 0, :], expected_k), (
            f"coefficient=1.0 must yield identity at kv-head={h}"
        )
        assert torch.equal(v_page[0, h, 0, :], expected_v), (
            f"coefficient=1.0 must yield identity at kv-head={h}"
        )


@pytest.mark.parametrize("coefficient_pair", [(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)])
def test_coefficient_two_matches_ratio_scaling(synthetic_provider, coefficient_pair):
    """Two matches at DIFFERENT positions must each carry their own coefficient.

    Bug authority claim: the per-match ``float(m.coefficient)`` is applied
    INSIDE the loop body, so a batch of matches with heterogeneous
    coefficients yields heterogeneous scalings — match i's slice uses c_i,
    not the mean / first / last coefficient of the batch.

    Different positions ensure different underlying raw K/V from the
    provider; this avoids a degenerate test where both rows happen to be
    identical and the ratio test reduces to checking c0/c1 against a
    uniform constant.
    """
    c0, c1 = coefficient_pair
    m0 = _make_match(coefficient=c0, position=0)
    m1 = _make_match(coefficient=c1, position=1)

    raw_k0, raw_v0 = synthetic_provider.kv_for_match(m0)
    raw_k1, raw_v1 = synthetic_provider.kv_for_match(m1)

    pages, sizes, _ = vec_inject_to_kv_direct(
        [m0, m1],
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )
    k_page, v_page = pages[0][0]
    n_kv_heads = k_page.shape[1]
    assert k_page.shape == (1, n_kv_heads, 2, HEAD_DIM)
    assert sizes == [2]

    expected_k0 = _expected_scaled(raw_k0, c0)
    expected_v0 = _expected_scaled(raw_v0, c0)
    expected_k1 = _expected_scaled(raw_k1, c1)
    expected_v1 = _expected_scaled(raw_v1, c1)

    # Per-fact assertions (broadcast across every kv-head slot).
    for h in range(n_kv_heads):
        actual_k0 = k_page[0, h, 0, :]
        actual_v0 = v_page[0, h, 0, :]
        actual_k1 = k_page[0, h, 1, :]
        actual_v1 = v_page[0, h, 1, :]

        assert torch.allclose(actual_k0, expected_k0, atol=1e-2, rtol=1e-2), (
            f"fact-0 K mismatch at kv-head={h} for c0={c0}"
        )
        assert torch.allclose(actual_v0, expected_v0, atol=1e-2, rtol=1e-2), (
            f"fact-0 V mismatch at kv-head={h} for c0={c0}"
        )
        assert torch.allclose(actual_k1, expected_k1, atol=1e-2, rtol=1e-2), (
            f"fact-1 K mismatch at kv-head={h} for c1={c1}"
        )
        assert torch.allclose(actual_v1, expected_v1, atol=1e-2, rtol=1e-2), (
            f"fact-1 V mismatch at kv-head={h} for c1={c1}"
        )

    # Per-head broadcast invariant — every kv-head slot must hold the same
    # (already-scaled) K/V vector. Replicates the /euclid PASS claim
    # documented in kv_direct_adapter.py.
    for h in range(1, n_kv_heads):
        assert torch.equal(k_page[0, 0, 0, :], k_page[0, h, 0, :])
        assert torch.equal(v_page[0, 0, 0, :], v_page[0, h, 0, :])
        assert torch.equal(k_page[0, 0, 1, :], k_page[0, h, 1, :])
        assert torch.equal(v_page[0, 0, 1, :], v_page[0, h, 1, :])


@pytest.mark.parametrize("coefficient", [0.0, 0.5, 1.0, 2.0])
def test_coefficient_full_matrix(synthetic_provider, coefficient):
    """axis-1 end-state acceptance grid: {0.0, 0.5, 1.0, 2.0}.

    For every coefficient on the run-4 acceptance matrix, independently
    fetch the raw K/V via ``provider.kv_for_match`` (the same call the
    adapter makes), cast to cuda bf16, multiply by the coefficient, and
    compare against the broadcast-contiguous adapter output. atol/rtol of
    1e-2 absorbs bf16 round-off in the multiply; the coefficient=0 case
    additionally enforces strict zero via ``torch.equal``.
    """
    match = _make_match(coefficient=coefficient, position=0)
    raw_k, raw_v = synthetic_provider.kv_for_match(match)

    pages, sizes, offset = vec_inject_to_kv_direct(
        [match],
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )

    k_page, v_page = pages[0][0]
    n_kv_heads = k_page.shape[1]
    assert k_page.shape == (1, n_kv_heads, 1, HEAD_DIM)
    assert v_page.shape == (1, n_kv_heads, 1, HEAD_DIM)
    assert sizes == [1]
    assert offset == 0

    expected_k = _expected_scaled(raw_k, coefficient)
    expected_v = _expected_scaled(raw_v, coefficient)

    for h in range(n_kv_heads):
        actual_k = k_page[0, h, 0, :]
        actual_v = v_page[0, h, 0, :]
        assert torch.allclose(actual_k, expected_k, atol=1e-2, rtol=1e-2), (
            f"K propagation failed at coefficient={coefficient}, kv-head={h}"
        )
        assert torch.allclose(actual_v, expected_v, atol=1e-2, rtol=1e-2), (
            f"V propagation failed at coefficient={coefficient}, kv-head={h}"
        )

    # Strict-zero boundary check at the cold-state coefficient.
    if coefficient == 0.0:
        assert torch.equal(k_page, torch.zeros_like(k_page)), (
            "coefficient=0.0 must yield strictly-zero K page (not just close-to-zero)"
        )
        assert torch.equal(v_page, torch.zeros_like(v_page)), (
            "coefficient=0.0 must yield strictly-zero V page (not just close-to-zero)"
        )


def test_coefficient_negative_two(synthetic_provider):
    """coefficient=-2.0 — sign flip + 2x magnitude.

    Run-4 acceptance criterion includes a negative-coefficient case to
    prove the multiply preserves sign. A na"ively-implemented adapter
    using ``abs(coef)`` or clamping to >=0 would silently lose the sign
    information that upstream tier ranking encodes (e.g. a "deprecated /
    contradicted" fact tagged with negative weight).

    Asserts:
      1. element-wise sign opposite to raw at every position where raw is
         non-zero (raw == 0 → 0 either way; ignored by the mask)
      2. magnitude == 2.0 * |raw| within bf16 tolerance
    """
    match = _make_match(coefficient=-2.0, position=0)
    raw_k, raw_v = synthetic_provider.kv_for_match(match)

    pages, _, _ = vec_inject_to_kv_direct(
        [match],
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )
    k_page, v_page = pages[0][0]
    n_kv_heads = k_page.shape[1]

    raw_k_bf16 = raw_k.to(device="cuda", dtype=torch.bfloat16)
    raw_v_bf16 = raw_v.to(device="cuda", dtype=torch.bfloat16)

    expected_k = raw_k_bf16 * -2.0
    expected_v = raw_v_bf16 * -2.0

    for h in range(n_kv_heads):
        actual_k = k_page[0, h, 0, :]
        actual_v = v_page[0, h, 0, :]

        # End-to-end value check.
        assert torch.allclose(actual_k, expected_k, atol=1e-2, rtol=1e-2)
        assert torch.allclose(actual_v, expected_v, atol=1e-2, rtol=1e-2)

        # Sign-flip check at every nonzero position.
        nonzero_k = raw_k_bf16 != 0
        nonzero_v = raw_v_bf16 != 0
        # For floats x != 0 with negative coefficient, sign(actual) == -sign(raw).
        assert torch.all(
            torch.sign(actual_k[nonzero_k]) == -torch.sign(raw_k_bf16[nonzero_k])
        ), f"K sign flip violated at kv-head={h}"
        assert torch.all(
            torch.sign(actual_v[nonzero_v]) == -torch.sign(raw_v_bf16[nonzero_v])
        ), f"V sign flip violated at kv-head={h}"

        # Magnitude check at every nonzero position.
        assert torch.allclose(
            actual_k[nonzero_k].abs(),
            2.0 * raw_k_bf16[nonzero_k].abs(),
            atol=1e-2,
            rtol=1e-2,
        ), f"K magnitude scaling violated at kv-head={h}"
        assert torch.allclose(
            actual_v[nonzero_v].abs(),
            2.0 * raw_v_bf16[nonzero_v].abs(),
            atol=1e-2,
            rtol=1e-2,
        ), f"V magnitude scaling violated at kv-head={h}"
