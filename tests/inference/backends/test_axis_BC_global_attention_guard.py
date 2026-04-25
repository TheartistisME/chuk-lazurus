"""axis-BC PROP K.0 — sliding-window-hazard guard tests.

Validates that ``assert_global_attention_layer`` accepts every layer in
``GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS`` and refuses every sliding layer
with a structured ``SlidingWindowLayerRefusedError``.

CUDA-only per repo policy (Amendment 5). The guard itself is pure Python
but the import chain pulls torch — gating on CUDA keeps the test surface
homogeneous with the rest of the axis-BC suite.
"""

from __future__ import annotations

import pytest
import torch

from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
    GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
    GEMMA4_E2B_IT_LAYER_TYPES,
)
from chuk_lazarus.inference.context.research.vec_inject.kv_direct_adapter import (
    SlidingWindowLayerRefusedError,
    _gemma4_e2b_it_sliding_set,
    assert_global_attention_layer,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA-only: torch.cuda.is_available()==False",
)


# ── PASS branch ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "layer_idx", sorted(GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS)
)
def test_guard_passes_on_global_layers(layer_idx: int):
    """Global / full-attention layers MUST pass the guard silently."""
    assert assert_global_attention_layer(layer_idx) is None


# ── FAIL branch ──────────────────────────────────────────────────────────────


_SLIDING_SAMPLE = [0, 1, 13, 30, 33]


@pytest.mark.parametrize("layer_idx", _SLIDING_SAMPLE)
def test_guard_raises_on_sliding_layers(layer_idx: int):
    """Sliding-window layers MUST raise SlidingWindowLayerRefusedError."""
    # Sanity: confirm the sample really is sliding per the axis-A fixture.
    assert GEMMA4_E2B_IT_LAYER_TYPES[layer_idx] == "sliding_attention"

    with pytest.raises(SlidingWindowLayerRefusedError) as exc_info:
        assert_global_attention_layer(layer_idx)

    err = exc_info.value
    assert err.target_layer == layer_idx
    assert layer_idx in err.sliding_set
    assert layer_idx not in err.global_set
    assert err.global_set == GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS
    assert err.sliding_set == _gemma4_e2b_it_sliding_set()


def test_sliding_set_partition_is_complete():
    """global ∪ sliding == full layer set; intersection must be empty."""
    sliding = _gemma4_e2b_it_sliding_set()
    glob = GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS
    n = len(GEMMA4_E2B_IT_LAYER_TYPES)
    assert sliding.isdisjoint(glob)
    assert sliding | glob == frozenset(range(n))
