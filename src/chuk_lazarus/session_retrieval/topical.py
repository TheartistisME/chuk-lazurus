"""Topical (TF-IDF) routing across per-session checkpoints.

Given a natural-language ``query_text``, for every valid checkpoint we call
``TorchKnowledgeStore.route_top_k`` to get its best-matching window(s), then
use the store's own ``TFIDFRouter.score_window`` to compute comparable scores
across stores, and pick the global top-1.

Primitive delegated to
----------------------
``TorchKnowledgeStore.route_top_k`` and ``TFIDFRouter.score_window`` - TF-IDF
and tokenisation are owned by the primitive; this module only ranks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle, load_store


def _encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` to token ids without special tokens.

    Matches the encoding convention used by the primitive's TF-IDF router
    (see ``torch_query._encode_token_ids``).
    """
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(t) for t in token_ids]


def route_topical(
    handles: Sequence[CheckpointHandle],
    query_text: str,
    tokenizer: Any,
    *,
    top_k_per_checkpoint: int = 1,
) -> tuple[CheckpointHandle, int, float] | None:
    """Return the globally top-1 ``(handle, window_id, score)`` across stores.

    For each :class:`CheckpointHandle`:
    1. Load the store.
    2. ``store.route_top_k(query_text, tokenizer, k=top_k_per_checkpoint)`` -
       the primitive's own topical routing; may return an empty list.
    3. For each returned window id, score it via
       ``store._get_tfidf_router().score_window(query_ids, window_id)``.

    Globally rank all scored candidates by ``(-score, session_id, window_id)``
    for deterministic tie-breaking and return the top-1.

    Returns
    -------
    ``(handle, window_id, score)`` if any checkpoint produced a positive
    candidate, otherwise ``None``.
    """
    if top_k_per_checkpoint <= 0:
        return None

    scored: list[tuple[float, str, int, CheckpointHandle]] = []
    for handle in handles:
        store = load_store(handle)
        candidates = store.route_top_k(query_text, tokenizer, k=top_k_per_checkpoint)
        if not candidates:
            continue
        query_ids = _encode_token_ids(tokenizer, query_text)
        router = store._get_tfidf_router()
        for window_id in candidates:
            score = float(router.score_window(query_ids, int(window_id)))
            scored.append((score, handle.session_id, int(window_id), handle))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_score, _sid, top_window, top_handle = scored[0]
    return top_handle, top_window, top_score


__all__ = ["route_topical"]
