"""Deterministic top-K frequency keyword extraction (design §6)."""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "to", "of", "in", "on", "at", "for", "with", "by", "it", "this",
        "that", "these", "those", "i", "you", "we", "they", "he", "she",
        "as", "be", "been",
    }
)

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def extract_topic_tags(text: str, *, top_k: int = 5) -> list[str]:
    """Return up to ``top_k`` keyword tags from ``text`` (design §6).

    Steps:
      1. Lowercase.
      2. Split on non-alphanumeric characters.
      3. Drop tokens shorter than 3 characters.
      4. Drop purely numeric tokens.
      5. Drop STOPWORDS.
      6. Count remaining tokens; keep first-occurrence index for tie-break.
      7. Sort by (count desc, first_occurrence asc); take top_k.
    """
    if not text:
        return []

    lowered = text.lower()
    tokens = _TOKEN_SPLIT.split(lowered)

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for token in tokens:
        if not token:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        if token in STOPWORDS:
            continue
        if token not in first_seen:
            first_seen[token] = position
            counts[token] = 1
        else:
            counts[token] += 1
        position += 1

    if not counts:
        return []

    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], first_seen[kv[0]]),
    )
    return [tok for tok, _ in ranked[:top_k]]
