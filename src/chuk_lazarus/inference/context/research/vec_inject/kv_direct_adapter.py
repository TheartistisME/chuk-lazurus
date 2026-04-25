"""KV-direct adapter — vec_inject retrieval -> KVDirectGenerator.inject_pre_rope_kv args.

This module implements axis-BC PROP K.5 (KV-direct adapter) and PROP K.0
(sliding-window guard) per the kv-memory-implementation run-1 recipe.

PROP K.5 — adapter
==================
``vec_inject_to_kv_direct(matches, provider, target_layer, target_offsets)``
turns a sequence of ``VecInjectMatch`` records into the
``(windows_kv, window_sizes, offset)`` triple consumed by
``KVDirectGenerator.inject_pre_rope_kv`` (see
``src/chuk_lazarus/inference/context/kv_generator.py``:585).

Per-window K and V are loaded from the provider's NPZ payloads via
``LocalVecInjectProvider.kv_for_match(match)`` (added in this run). The
returned per-layer page has shape ``(1, n_kv_heads, n_facts, head_dim)``,
dtype ``torch.bfloat16``, device ``cuda``.

PROP K.0 — sliding-window guard
===============================
``assert_global_attention_layer(target_layer)`` REFUSES injection at any
layer whose Gemma-4-E2B-it ``layer_types[target_layer]`` is
``"sliding_attention"``. The full set of allowed (global / full-attention)
layers is the axis-A frozenset
``GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}``.

Refuse-on-sliding is the explicit policy here — automatic remap to the
nearest global layer is an OPTIONAL EXTENSION owned by axis-G (DEFERRED
per Q3 of the recipe).

/euclid CLAIM (n_kv_heads broadcast):
  The K extracted into ``vec_inject.npz`` is single-head (the meta's
  ``kv_head``). To populate the ``(1, n_kv_heads, n_facts, head_dim)``
  page slot the adapter REPLICATES the single-head K (and V) across all
  ``n_kv_heads`` slots. This matches the semantics expected by
  ``inject_pre_rope_kv`` because Gemma-4 uses GQA (grouped-query attn)
  where every KV head is attended-to by ``n_rep`` query heads — so a
  single-head K written to every kv-head slot acts as a constant K
  across the heads, which is the intended semantics of "inject this fact
  uniformly for routing".
/euclid PASS test:
  tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py
  ::test_adapter_returns_correct_pages_shape_on_global_layer asserts
  ``pages[0][0][0].shape == (1, n_kv_heads, n_facts, head_dim)`` and that
  every kv-head slot is identical to the single-head extraction.
/euclid FAIL test:
  ::test_adapter_refuses_sliding_layer_via_full_call (D6) — calling with
  target_layer=30 raises ``SlidingWindowLayerRefusedError``.
/euclid adaptation-status:
  PROP K.0 == REFUSE (no auto-remap). Auto-remap is deferred to axis-G.

/euclid CLAIM (default offset):
  When ``target_offsets is None`` the adapter falls back to ``offset=0``
  and emits a ``logging.warning`` so the caller is aware that no preamble
  offset was provided.
/euclid PASS test:
  ::test_adapter_default_offset_zero_when_target_offsets_none.

/euclid CLAIM (NPZ key schema):
  The provider sources raw K from NPZ key ``"w{wid}/k_vecs"`` and V from
  ``"w{wid}/v_vecs"`` per ``_index_format.VecInjectWindowKey``. Indices
  produced by axis-B's torch-side prefill pipeline contain v_vecs;
  legacy ``kv_route_index.npz`` does NOT and will raise from
  ``provider.kv_for_match`` with a clear RuntimeError.
/euclid PASS test:
  ::test_kv_for_match_helper validates round-trip on a synthetic NPZ
  carrying both keys.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import torch

from chuk_lazarus.inference.context.knowledge.gemma4_e2b_it_layers import (
    GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS,
    GEMMA4_E2B_IT_LAYER_TYPES,
)

from ._types import VecInjectMatch

if TYPE_CHECKING:
    from .providers._local_file_torch import LocalVecInjectProvider

_LOG = logging.getLogger(__name__)


__all__ = [
    "SlidingWindowLayerRefusedError",
    "assert_global_attention_layer",
    "inject_via_kv_direct",
    "vec_inject_to_kv_direct",
]


class SlidingWindowLayerRefusedError(RuntimeError):
    """PROP K.0 GUARD — refused KV-direct injection at a sliding-window layer.

    Carries:
      * target_layer  : the layer index that was refused
      * global_set    : frozenset of allowed (global / full-attention) layers
      * sliding_set   : frozenset of refused (sliding-window) layers
    """

    def __init__(
        self,
        target_layer: int,
        global_set: frozenset[int],
        sliding_set: frozenset[int],
    ) -> None:
        self.target_layer = target_layer
        self.global_set = global_set
        self.sliding_set = sliding_set
        super().__init__(
            f"PROP K.0 GUARD: target_layer={target_layer} is a sliding-window "
            f"layer; KV-direct injection is refused. "
            f"global_set={sorted(global_set)} sliding_set={sorted(sliding_set)}"
        )


def _gemma4_e2b_it_sliding_set() -> frozenset[int]:
    """Indices where ``GEMMA4_E2B_IT_LAYER_TYPES[i] == 'sliding_attention'``."""
    return frozenset(
        i for i, t in enumerate(GEMMA4_E2B_IT_LAYER_TYPES) if t == "sliding_attention"
    )


def assert_global_attention_layer(target_layer: int) -> None:
    """PROP K.0 GUARD — accept global-attn layers, refuse sliding-window layers.

    Refuse-on-sliding policy. Auto-remap to the nearest global layer is
    an OPTIONAL EXTENSION owned by axis-G (DEFERRED per recipe Q3).
    """
    global_set = GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS
    if target_layer in global_set:
        return
    sliding_set = _gemma4_e2b_it_sliding_set()
    raise SlidingWindowLayerRefusedError(target_layer, global_set, sliding_set)


def _to_cuda_bf16(t: torch.Tensor) -> torch.Tensor:
    """Normalise dtype/device to (cuda, bf16) without re-allocating when already correct."""
    if t.device.type != "cuda" or t.dtype != torch.bfloat16:
        return t.to(device="cuda", dtype=torch.bfloat16)
    return t


def vec_inject_to_kv_direct(
    matches: Iterable[VecInjectMatch],
    provider: LocalVecInjectProvider,
    target_layer: int,
    target_offsets: Mapping[int, int] | None = None,
) -> tuple[list[list[tuple[torch.Tensor, torch.Tensor]]], list[int], int]:
    """Convert vec_inject retrieval output into ``inject_pre_rope_kv`` arguments.

    Parameters
    ----------
    matches        : iterable of ``VecInjectMatch`` — must share a single
                     ``window_id`` (multi-window batches are deferred to axis-E).
    provider       : ``LocalVecInjectProvider`` exposing ``kv_for_match`` and
                     the ``kv_head`` count via its raw-model handle.
    target_layer   : index of the layer where the synthetic KV will land.
                     PROP K.0 guard runs first — sliding layers are refused.
    target_offsets : optional ``{window_id: rope_offset}`` mapping. When
                     None the adapter defaults ``offset = 0`` and logs a
                     warning.

    Returns
    -------
    pages         : ``list[list[tuple[K, V]]]`` — outer dim = N windows
                    (== 1 here), inner dim = 1 layer entry. K and V are
                    cuda bf16 of shape ``(1, n_kv_heads, n_facts, head_dim)``.
    window_sizes  : ``[n_facts]``
    offset        : starting RoPE position for the (single) window.

    Raises
    ------
    SlidingWindowLayerRefusedError : target_layer is a sliding-window layer.
    ValueError : empty ``matches`` or matches spanning multiple ``window_id``s.
    RuntimeError : provider lacks raw KV payloads (legacy index).
    """
    # PROP K.0 — refuse non-global target layers up-front.
    assert_global_attention_layer(target_layer)

    matches_list = list(matches)
    if not matches_list:
        raise ValueError(
            "vec_inject_to_kv_direct: matches is empty; KV-direct injection "
            "requires at least one fact to materialise a page."
        )

    window_ids = {int(m.window_id) for m in matches_list}
    if len(window_ids) != 1:
        raise ValueError(
            f"vec_inject_to_kv_direct: matches must share a single window_id; "
            f"got {sorted(window_ids)}. Multi-window batches are deferred to "
            f"axis-E E2E."
        )
    (only_window,) = window_ids

    # Resolve offset for the sole window.
    if target_offsets is None:
        _LOG.warning(
            "vec_inject_to_kv_direct: target_offsets=None — defaulting "
            "RoPE offset=0 for window_id=%d. Pass target_offsets to position "
            "synthetic facts after a real preamble.",
            only_window,
        )
        offset = 0
    else:
        if only_window not in target_offsets:
            raise ValueError(
                f"vec_inject_to_kv_direct: target_offsets is missing the "
                f"window_id={only_window} required by the matches."
            )
        offset = int(target_offsets[only_window])

    # Determine n_kv_heads from the provider's raw model when available;
    # otherwise fall back to the meta's kv_head as a single-head view.
    n_kv_heads = _resolve_n_kv_heads(provider)

    # Pull (k, v) for every match and stack into (n_facts, head_dim) pre-broadcast.
    k_list: list[torch.Tensor] = []
    v_list: list[torch.Tensor] = []
    for m in matches_list:
        k_vec, v_vec = provider.kv_for_match(m)
        k_list.append(_to_cuda_bf16(k_vec))
        v_list.append(_to_cuda_bf16(v_vec))

    k_stack = torch.stack(k_list, dim=0)  # (n_facts, head_dim)
    v_stack = torch.stack(v_list, dim=0)  # (n_facts, head_dim)
    n_facts, head_dim = k_stack.shape

    # Broadcast single-head K/V across n_kv_heads → (1, n_kv_heads, n_facts, head_dim).
    # Use expand+contiguous to avoid silent stride aliasing further down the stack.
    k_page = (
        k_stack.unsqueeze(0).unsqueeze(0).expand(1, n_kv_heads, n_facts, head_dim).contiguous()
    )
    v_page = (
        v_stack.unsqueeze(0).unsqueeze(0).expand(1, n_kv_heads, n_facts, head_dim).contiguous()
    )

    # The recipe's "pages = [[(K, V)]]" structure: outer = N windows, inner = 1 layer.
    # inject_pre_rope_kv iterates windows[wi][li]; the caller (axis-E) is responsible
    # for wiring the per-layer dimension when more than one layer is targeted.
    pages: list[list[tuple[torch.Tensor, torch.Tensor]]] = [[(k_page, v_page)]]
    window_sizes = [n_facts]
    return pages, window_sizes, offset


def _resolve_n_kv_heads(provider: LocalVecInjectProvider) -> int:
    """Best-effort lookup of the model's ``num_key_value_heads``.

    Order of resolution:
      1. ``provider._raw_model.config.num_key_value_heads`` (HF Gemma4 path)
      2. ``provider._raw_model.config.text_config.num_key_value_heads``
      3. fallback: 1 (treat single-head extraction as a single kv-head page —
         caller may broadcast downstream)
    """
    raw_model = getattr(provider, "_raw_model", None)
    if raw_model is not None:
        cfg = getattr(raw_model, "config", None)
        if cfg is not None:
            n = getattr(cfg, "num_key_value_heads", None)
            if isinstance(n, int) and n > 0:
                return n
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                n = getattr(text_cfg, "num_key_value_heads", None)
                if isinstance(n, int) and n > 0:
                    return n
    # Fallback — single-head broadcast view (n_kv_heads=1).
    return 1


def inject_via_kv_direct(
    matches: Iterable[VecInjectMatch],
    provider: LocalVecInjectProvider,
    target_layer: int,
    kv_direct_generator,  # KVDirectGenerator instance
    target_offsets: Mapping[int, int] | None = None,
):
    """Convenience wrapper — build adapter args + invoke ``inject_pre_rope_kv``.

    Returns whatever ``kv_direct_generator.inject_pre_rope_kv(pages, sizes, offset)``
    returns (a ``KVStore`` for the MLX backbone).
    """
    pages, sizes, offset = vec_inject_to_kv_direct(
        matches, provider, target_layer, target_offsets
    )
    return kv_direct_generator.inject_pre_rope_kv(pages, sizes, offset)
