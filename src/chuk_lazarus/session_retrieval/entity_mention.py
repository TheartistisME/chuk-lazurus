"""Deterministic entity-mention routing across per-session checkpoints.

Extracts a query's "entity tokens" (capitalised, ALL-CAPS, quoted), builds
each candidate window's entity set from store keywords and per-turn
``topic_tags``, and picks the (handle, window_id) with the highest overlap
score.

Primitive delegated to
----------------------
- ``TorchKnowledgeStore.keywords`` - per-window keyword lists.
- ``TorchKnowledgeStore.window_metadata`` - ``window_id -> {clause_id, ...}``.
- ``chuk_lazarus.session_retrieval.enumeration.load_original_turn_records`` -
  axis-2 per-turn JSON records (for ``topic_tags``).

No re-implementation of clause-ID parsing or keyword routing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    load_original_turn_records,
    load_store,
)

# Stopwords removed from extracted entity tokens (shortlist per axis-4 spec).
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "to", "for", "of",
        "in", "on", "at", "that", "this", "these", "those", "who", "what",
        "when", "where", "why", "how", "and", "or", "but", "i", "you", "he",
        "she", "it", "we", "they", "with", "about",
    }
)

# Token regex: sequences of letters/digits/apostrophes/hyphens, length >= 2.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# Quoted strings: single- or double-quoted, non-greedy, content length >= 1.
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")


def extract_entity_tokens(query_text: str) -> list[str]:
    """Return the deterministic ordered, deduplicated list of entity tokens.

    Rules (spec §entity_mention.py):
    - Collect capitalised / Title-Case tokens of length >= 2.
    - Collect ALL-CAPS tokens of length >= 2.
    - Collect quoted-string contents (each token inside the quotes).
    - Drop a small stopword shortlist.
    - Normalise to lowercase for matching.
    - Preserve first-seen order; dedupe.
    """
    collected: list[str] = []
    seen: set[str] = set()

    def _push(raw: str) -> None:
        lowered = raw.lower()
        if len(lowered) < 2:
            return
        if lowered in _STOPWORDS:
            return
        if lowered in seen:
            return
        seen.add(lowered)
        collected.append(lowered)

    # Quoted contents first — then tokenise each quoted span.
    for match in _QUOTED_RE.finditer(query_text):
        quoted = match.group(1) if match.group(1) is not None else match.group(2)
        if not quoted:
            continue
        for sub in _TOKEN_RE.findall(quoted):
            _push(sub)

    # Capitalised / ALL-CAPS tokens.
    for sub in _TOKEN_RE.findall(query_text):
        if len(sub) < 2:
            continue
        if sub[0].isupper() and sub[1:].islower():
            _push(sub)
            continue
        if sub.isupper() and any(ch.isalpha() for ch in sub):
            _push(sub)
            continue

    return collected


def _window_entity_set(
    handle: CheckpointHandle,
    window_id: int,
    *,
    store_keywords: list[str],
    store_window_metadata: dict[int, dict[str, Any]],
    cached_records: list[dict[str, Any]] | None,
) -> set[str]:
    """Assemble the entity set attached to one window.

    Sources:
    1. ``store.keywords.get(window_id, [])`` lowercased.
    2. ``topic_tags`` from the per-turn JSON record whose ``clause_id``
       matches the window's metadata ``clause_id`` (axis-2 emission layout).
       Falls back to matching by the window's own ``clause_id`` field in
       ``store.window_metadata[window_id]``.
    """
    entity_set: set[str] = set()
    for keyword in store_keywords:
        lowered = str(keyword).strip().lower()
        if lowered:
            entity_set.add(lowered)

    window_meta = store_window_metadata.get(int(window_id)) or {}
    window_clause_id = str(window_meta.get("clause_id", "")).strip()

    if not window_clause_id or not cached_records:
        return entity_set

    for record in cached_records:
        if str(record.get("clause_id", "")).strip() != window_clause_id:
            continue
        for tag in record.get("topic_tags", []) or []:
            lowered = str(tag).strip().lower()
            if lowered:
                entity_set.add(lowered)
        break
    return entity_set


def score_window_entity_overlap(
    handle: CheckpointHandle,
    window_id: int,
    entity_tokens: set[str],
) -> float:
    """Return ``|entities ∩ window_set| / max(1, len(entity_tokens))``.

    This helper is used by :func:`route_entity_mention` but is exported for
    tests and for consumers that want to score against a fixed candidate.

    Note: loading the store on every call is not cheap, which is why the
    routing function below caches per-handle state. This top-level helper
    remains available for single-shot scoring use.
    """
    store = load_store(handle)
    store_keywords = list(store.keywords.get(int(window_id), []))
    cached_records = load_original_turn_records(handle)
    window_set = _window_entity_set(
        handle,
        int(window_id),
        store_keywords=store_keywords,
        store_window_metadata=store.window_metadata,
        cached_records=cached_records,
    )
    if not entity_tokens:
        return 0.0
    intersection = entity_tokens & window_set
    return float(len(intersection)) / float(max(1, len(entity_tokens)))


def route_entity_mention(
    handles: Sequence[CheckpointHandle],
    query_text: str,
) -> tuple[CheckpointHandle, int, float] | None:
    """Pick the globally best ``(handle, window_id, score)`` by entity overlap.

    Iterates all ``(handle, window_id)`` pairs across every valid checkpoint.
    Per-handle state (store keywords, window metadata, per-turn record list)
    is loaded exactly once. Returns the pair with the highest overlap score
    strictly greater than zero. Ties are broken lexicographically on
    ``(session_id, window_id)`` for determinism.
    """
    entity_tokens = extract_entity_tokens(query_text)
    if not entity_tokens:
        return None
    entity_set = set(entity_tokens)

    scored: list[tuple[float, str, int, CheckpointHandle]] = []
    for handle in handles:
        store = load_store(handle)
        cached_records = load_original_turn_records(handle)
        store_window_metadata = store.window_metadata
        for window_id in range(int(store.num_windows)):
            store_keywords = list(store.keywords.get(int(window_id), []))
            window_set = _window_entity_set(
                handle,
                int(window_id),
                store_keywords=store_keywords,
                store_window_metadata=store_window_metadata,
                cached_records=cached_records,
            )
            if not window_set:
                continue
            intersection = entity_set & window_set
            if not intersection:
                continue
            score = float(len(intersection)) / float(max(1, len(entity_tokens)))
            if score <= 0.0:
                continue
            scored.append((score, handle.session_id, int(window_id), handle))

    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    top_score, _sid, top_window, top_handle = scored[0]
    return top_handle, top_window, top_score


__all__ = [
    "extract_entity_tokens",
    "route_entity_mention",
    "score_window_entity_overlap",
]
