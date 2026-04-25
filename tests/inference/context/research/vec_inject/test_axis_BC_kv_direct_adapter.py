"""axis-BC PROP K.5 — vec_inject_to_kv_direct shape & contract tests.

CUDA-only per repo policy (Amendment 5). Builds a synthetic vec_inject.npz
inline so the test does not rely on a real model snapshot.
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
    SlidingWindowLayerRefusedError,
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
    """Build a single-window NPZ carrying both k_vecs and v_vecs."""
    rng = np.random.default_rng(seed=12345)
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


def _make_matches(positions: list[int], window_id: int = WINDOW_ID) -> list[VecInjectMatch]:
    return [
        VecInjectMatch(
            token_id=100 + p,
            coefficient=0.5,
            score=0.9,
            window_id=window_id,
            position=p,
            distinctive=True,
            source_id=window_id,
            source_type=SourceType.DOCUMENT,
        )
        for p in positions
    ]


# ── tests ────────────────────────────────────────────────────────────────────


def test_adapter_returns_correct_pages_shape_on_global_layer(synthetic_provider):
    matches = _make_matches([0, 1, 2])
    target_offsets = {WINDOW_ID: 100}

    pages, sizes, offset = vec_inject_to_kv_direct(
        matches,
        synthetic_provider,
        target_layer=29,
        target_offsets=target_offsets,
    )

    # n_kv_heads is resolved from the provider — synthetic provider has no raw
    # model so the resolver falls back to 1.
    assert isinstance(pages, list)
    assert len(pages) == 1, "single-window batch → one outer entry"
    assert isinstance(pages[0], list)
    assert len(pages[0]) == 1, "single-layer entry per recipe"
    assert isinstance(pages[0][0], tuple)
    assert len(pages[0][0]) == 2

    k_page, v_page = pages[0][0]
    n_facts = 3
    expected_n_kv_heads = 1  # synthetic provider has no raw model → fallback
    assert k_page.shape == (1, expected_n_kv_heads, n_facts, HEAD_DIM)
    assert v_page.shape == (1, expected_n_kv_heads, n_facts, HEAD_DIM)
    assert k_page.dtype == torch.bfloat16
    assert v_page.dtype == torch.bfloat16
    assert k_page.device.type == "cuda"
    assert v_page.device.type == "cuda"

    assert sizes == [n_facts]
    assert offset == 100


def test_adapter_broadcast_replicates_single_head_across_n_kv_heads(synthetic_provider):
    """Validate the n_kv_heads broadcast claim — with a stub raw_model providing
    num_key_value_heads=4 every kv-head slot must hold the same K/V vector."""

    class _StubCfg:
        num_key_value_heads = 4

    class _StubModel:
        config = _StubCfg()

    synthetic_provider._raw_model = _StubModel()
    matches = _make_matches([0, 1])

    pages, _, _ = vec_inject_to_kv_direct(
        matches,
        synthetic_provider,
        target_layer=29,
        target_offsets={WINDOW_ID: 0},
    )
    k_page, v_page = pages[0][0]
    assert k_page.shape == (1, 4, 2, HEAD_DIM)
    # head 0 == head 1 == head 2 == head 3
    for h in range(1, 4):
        assert torch.equal(k_page[0, 0], k_page[0, h])
        assert torch.equal(v_page[0, 0], v_page[0, h])


def test_adapter_raises_on_empty_matches(synthetic_provider):
    with pytest.raises(ValueError, match="empty"):
        vec_inject_to_kv_direct(
            [],
            synthetic_provider,
            target_layer=29,
            target_offsets={WINDOW_ID: 0},
        )


def test_adapter_raises_on_multi_window_matches(synthetic_provider):
    multi = [
        VecInjectMatch(
            token_id=1,
            coefficient=0.5,
            score=0.9,
            window_id=0,
            position=0,
            distinctive=True,
        ),
        VecInjectMatch(
            token_id=2,
            coefficient=0.5,
            score=0.9,
            window_id=1,
            position=0,
            distinctive=True,
        ),
    ]
    with pytest.raises(ValueError, match="single window_id"):
        vec_inject_to_kv_direct(
            multi,
            synthetic_provider,
            target_layer=29,
            target_offsets=None,
        )


def test_adapter_default_offset_zero_when_target_offsets_none(synthetic_provider, caplog):
    matches = _make_matches([0])
    with caplog.at_level("WARNING"):
        pages, sizes, offset = vec_inject_to_kv_direct(
            matches,
            synthetic_provider,
            target_layer=29,
            target_offsets=None,
        )
    assert offset == 0
    assert sizes == [1]
    assert pages[0][0][0].device.type == "cuda"
    # Warning should mention the defaulting behaviour
    assert any("offset" in rec.message.lower() for rec in caplog.records)


def test_adapter_raises_when_target_offsets_missing_window(synthetic_provider):
    matches = _make_matches([0])
    with pytest.raises(ValueError, match="missing"):
        vec_inject_to_kv_direct(
            matches,
            synthetic_provider,
            target_layer=29,
            target_offsets={999: 0},  # wrong window_id
        )


def test_kv_for_match_helper(synthetic_provider):
    match = _make_matches([0])[0]
    k_vec, v_vec = synthetic_provider.kv_for_match(match)
    assert isinstance(k_vec, torch.Tensor)
    assert isinstance(v_vec, torch.Tensor)
    assert k_vec.shape == (HEAD_DIM,)
    assert v_vec.shape == (HEAD_DIM,)
    assert k_vec.dtype == torch.bfloat16
    assert v_vec.dtype == torch.bfloat16
    assert k_vec.device.type == "cuda"
    assert v_vec.device.type == "cuda"


def test_kv_for_match_unknown_position_raises(synthetic_provider):
    bad = VecInjectMatch(
        token_id=999,
        coefficient=0.5,
        score=0.5,
        window_id=WINDOW_ID,
        position=42,  # not loaded
        distinctive=True,
    )
    with pytest.raises(KeyError):
        synthetic_provider.kv_for_match(bad)


def test_adapter_refuses_sliding_layer_via_full_call(synthetic_provider):
    """End-to-end: calling the adapter with a sliding layer triggers PROP K.0 guard."""
    matches = _make_matches([0])
    with pytest.raises(SlidingWindowLayerRefusedError) as exc_info:
        vec_inject_to_kv_direct(
            matches,
            synthetic_provider,
            target_layer=30,  # sliding per axis-A fixture
            target_offsets={WINDOW_ID: 0},
        )
    assert exc_info.value.target_layer == 30
    assert 30 in exc_info.value.sliding_set
    assert 30 not in exc_info.value.global_set
