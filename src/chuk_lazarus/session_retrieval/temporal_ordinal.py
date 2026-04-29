"""Temporal ordinal selection for co-reference-heavy retrieval tasks."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_ORDINAL_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in _ORDINAL_WORDS) + r")\b",
    re.IGNORECASE,
)
_ORDINAL_DIGIT_RE = re.compile(r"\b([1-9][0-9]*)(?:st|nd|rd|th)?\b", re.IGNORECASE)


def parse_ordinal_requirement(text: str, *, default: int | None = None) -> int | None:
    """Extract the 1-indexed ordinal requested by ``text``.

    MRCR prompts usually phrase this as ``"the 2nd (1 indexed) ..."``. The
    digit form wins over word ordinals because it is the canonical benchmark
    form; word support keeps the helper useful for hand-authored prompts.
    """
    match = _ORDINAL_DIGIT_RE.search(str(text))
    if match is not None:
        return int(match.group(1))
    word_match = _ORDINAL_WORD_RE.search(str(text))
    if word_match is not None:
        return int(_ORDINAL_WORDS[word_match.group(1).lower()])
    return default


def candidate_session_id(candidate: Any) -> str:
    """Return the candidate's session id, if present."""
    if isinstance(candidate, dict):
        return str(candidate.get("session_id", ""))
    handle = getattr(candidate, "handle", None)
    if handle is not None:
        return str(getattr(handle, "session_id", ""))
    nested = getattr(candidate, "candidate", None)
    if nested is not None:
        return candidate_session_id(nested)
    return str(getattr(candidate, "session_id", ""))


def candidate_window_id(candidate: Any) -> int:
    """Return the candidate's window id, defaulting to ``0`` for tie-breaks."""
    if isinstance(candidate, dict):
        value = candidate.get("window_id", 0)
    else:
        value = getattr(candidate, "window_id", 0)
        if value == 0 and hasattr(candidate, "candidate"):
            value = getattr(candidate.candidate, "window_id", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def candidate_session_turn_index(candidate: Any) -> int:
    """Return the archival turn timestamp for temporal sorting.

    The ASI lifecycle now surfaces ``session_turn_index`` directly. This helper
    also accepts dict telemetry and tier assignments for runner code that deals
    with serialized candidates.
    """
    if isinstance(candidate, dict):
        for key in ("session_turn_index", "turn_index", "message_index"):
            value = candidate.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        return candidate_window_id(candidate)

    if hasattr(candidate, "session_turn_index"):
        try:
            return int(candidate.session_turn_index)
        except (TypeError, ValueError):
            pass

    nested = getattr(candidate, "candidate", None)
    if nested is not None:
        return candidate_session_turn_index(nested)

    return candidate_window_id(candidate)


def sort_temporally(candidates: Sequence[T]) -> list[T]:
    """Sort candidates by archival timestamp, then stable identity."""
    ordered = list(candidates)
    ordered.sort(
        key=lambda candidate: (
            candidate_session_id(candidate),
            candidate_session_turn_index(candidate),
            candidate_window_id(candidate),
        )
    )
    return ordered


def select_temporal_ordinal(
    candidates: Sequence[T],
    *,
    ordinal: int,
    top_k: int | None = None,
) -> T:
    """Select the ``ordinal``-th retrieved candidate after temporal sorting."""
    ordinal_int = int(ordinal)
    if ordinal_int <= 0:
        raise ValueError(f"ordinal must be >= 1, got {ordinal}")
    pool = list(candidates)
    if top_k is not None:
        pool = pool[: int(top_k)]
    ordered = sort_temporally(pool)
    if ordinal_int > len(ordered):
        raise IndexError(
            f"ordinal {ordinal_int} requested from {len(ordered)} retrieved candidates"
        )
    return ordered[ordinal_int - 1]


__all__ = [
    "candidate_session_id",
    "candidate_session_turn_index",
    "candidate_window_id",
    "parse_ordinal_requirement",
    "select_temporal_ordinal",
    "sort_temporally",
]
