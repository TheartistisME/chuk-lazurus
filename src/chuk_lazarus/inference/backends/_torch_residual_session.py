"""
Runtime-side glue for the session-keyed WARM residual cache.

Keeps the prefix-match, delta-extend, and cache-writeback logic out of the
main torch runtime module. Import-time torch-free: all torch work happens
inside the methods receiving a live runtime reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheHitResult:
    """Everything the runtime needs to continue into the decode loop.

    Returned by :func:`reuse_from_session_cache`. Mirrors the locals the
    non-streaming prefill path leaves in scope right before the first
    ``_sample_next_token`` call.
    """

    past_key_values: Any
    logits: Any
    boundary_residual: Any | None
    boundary_residual_position: int
    total_seen_tokens: int
    hot_window_start: int
    # The input_length is always the full tokenised prompt length so the
    # GenerationStats surface matches what the cold path would have produced.
    input_length: int
    delta_len: int


def _torch_cuda_free() -> None:
    """Best-effort CUDA cache release — only runs when torch/CUDA are live."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — never let cleanup sink generation
        pass


def make_cuda_eviction_hook():
    """Return an eviction callback that drops GPU storage explicitly.

    Installed by the runtime when it constructs the ``ResidualSessionCache``
    so evicted entries free the hot KV cache and the residual tensor before
    ``torch.cuda.empty_cache()`` runs.
    """

    def _hook(entry) -> None:
        pkv = getattr(entry, "past_key_values", None)
        residual = getattr(entry, "boundary_residual", None)
        entry.past_key_values = None
        entry.boundary_residual = None
        # Explicit del helps reference counting on the HF Cache object which
        # holds per-layer tensors via Python references.
        del pkv
        del residual
        _torch_cuda_free()

    return _hook


def reuse_from_session_cache(
    runtime,
    *,
    entry,
    input_ids,
    attention_mask,
    max_hot_tokens_pre: int,
    prefill_hot_limit: int,
    config,
) -> CacheHitResult | None:
    """Extend a cached hot KV with only the delta tokens.

    Returns ``None`` to signal "not reusable — fall through to the cold
    split-prefill path". The caller is expected to discard the stale cache
    entry in that case so the rewrite path repopulates it.

    Reuse is rejected when:
      * The cached ``cold_token_ids`` is not a proper prefix of the new
        ``input_ids`` (tokenisation drifted or the client sent an
        unrelated prompt).
      * There are no delta tokens (turn K's prompt == turn K-1's prompt
        verbatim — exceedingly rare but not our job to handle).
      * The cached KV is missing (a previous eviction hook already cleared
        it but the entry lingered — shouldn't happen, but defended).

    On success runs a single forward over the delta tokens with
    ``past_key_values=entry.past_key_values`` and ``use_cache=True``, then
    trims to ``prefill_hot_limit`` so the hot window matches the budget the
    cold path would land at.
    """
    import torch  # local: keeps the module torch-free at import time  # noqa: F401

    if entry.past_key_values is None:
        return None

    cached_cold = entry.cold_token_ids
    if not cached_cold:
        return None

    prompt_tokens = input_ids[0].tolist()
    cached_len = len(cached_cold)
    prompt_len = len(prompt_tokens)

    if prompt_len <= cached_len:
        # The new prompt doesn't extend the cached one — fall through.
        return None
    if prompt_tokens[:cached_len] != cached_cold:
        return None

    delta_tokens = prompt_tokens[cached_len:]
    delta_len = len(delta_tokens)
    if delta_len == 0:
        return None

    batch_size = int(input_ids.shape[0])
    device = input_ids.device

    # Install the boundary-residual capture hook so the forward over the
    # delta refreshes ``entry.boundary_residual`` to reflect the new last
    # token position. Avoids a stale residual if the next turn needs a
    # split-prefill rebuild.
    from ._torch_residual_bounded import install_boundary_residual_capture_hook

    capture_layer = runtime._resolve_residual_capture_layer()
    capture_hook, capture_handle = install_boundary_residual_capture_hook(capture_layer)

    try:
        delta_ids = runtime._torch.tensor(
            [delta_tokens], dtype=input_ids.dtype, device=device
        )
        hot_seq_len = runtime._cache_seq_length(entry.past_key_values)
        step_attention_mask = runtime._torch.ones(
            (batch_size, hot_seq_len + delta_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = runtime._build_text_only_position_ids(
            batch_size,
            entry.total_seen_tokens,
            delta_len,
            device,
        )
        outputs = runtime._model(
            input_ids=delta_ids,
            attention_mask=step_attention_mask,
            position_ids=position_ids,
            past_key_values=entry.past_key_values,
            use_cache=True,
            **runtime._kv_direct_forward_kwargs(),
        )
        logits, past_key_values = runtime._extract_logits_and_cache(outputs)
        if past_key_values is None:
            raise RuntimeError(
                "residual_bounded_kv_direct: delta forward returned no past_key_values."
            )

        boundary_residual = entry.boundary_residual
        boundary_residual_position = entry.boundary_residual_position
        if capture_hook.tensor is not None:
            boundary_residual = capture_hook.tensor
            boundary_residual_position = entry.total_seen_tokens + delta_len - 1
    finally:
        capture_handle.remove()

    new_total_seen = entry.total_seen_tokens + delta_len
    past_key_values, evicted = runtime._enforce_last_token_budget(
        past_key_values, prefill_hot_limit
    )
    new_hot_window_start = max(0, new_total_seen - runtime._cache_seq_length(past_key_values))

    return CacheHitResult(
        past_key_values=past_key_values,
        logits=logits,
        boundary_residual=boundary_residual,
        boundary_residual_position=boundary_residual_position,
        total_seen_tokens=new_total_seen,
        hot_window_start=new_hot_window_start,
        input_length=prompt_len,
        delta_len=delta_len,
    )


def write_entry_from_state(
    *,
    entry_cls,
    conversation_id: str,
    state,
    past_key_values,
    total_seen_tokens: int,
):
    """Build a ``ResidualSessionEntry`` from the end-of-generation state.

    Called after a miss path finishes — the runtime writes this entry back
    into the cache so the next turn can take the reuse path.
    """
    return entry_cls(
        cold_token_ids=list(state.cold_token_ids),
        past_key_values=past_key_values,
        boundary_residual=state.boundary_residual,
        boundary_residual_position=state.boundary_residual_position,
        total_seen_tokens=total_seen_tokens,
        hot_window_start=state.hot_window_start,
        conversation_id=conversation_id,
    )


__all__ = [
    "CacheHitResult",
    "make_cuda_eviction_hook",
    "reuse_from_session_cache",
    "write_entry_from_state",
]
