"""Exact literal matching helpers for cross-session routing."""

from __future__ import annotations

import re
from typing import Any

_LITERAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{5,}")


def extract_high_entropy_literals(query_text: str) -> list[str]:
    """Return long query atoms that should route by exact token sequence.

    The memory harness plants random hex identifiers inside otherwise similar
    text. Token-set TF-IDF can collide on those fragments, so literals with
    digits get an exact subsequence check before ranking.
    """
    seen: set[str] = set()
    literals: list[str] = []
    for match in _LITERAL_RE.finditer(query_text):
        literal = match.group(0).strip("._:-")
        if len(literal) < 6 or not any(ch.isdigit() for ch in literal):
            continue
        key = literal.lower()
        if key in seen:
            continue
        seen.add(key)
        literals.append(literal)
    return literals


def encode_literal_sequences(tokenizer: Any, query_text: str) -> list[list[int]]:
    """Encode exact literals from ``query_text`` with the model tokenizer."""
    sequences: list[list[int]] = []
    for literal in extract_high_entropy_literals(query_text):
        try:
            token_ids = tokenizer.encode(literal, add_special_tokens=False)
        except TypeError:
            token_ids = tokenizer.encode(literal)
        sequence = [int(token_id) for token_id in token_ids]
        if sequence:
            sequences.append(sequence)
    return sequences


def contains_token_subsequence(haystack: list[int], needle: list[int]) -> bool:
    """Return True when ``needle`` occurs contiguously inside ``haystack``."""
    if not needle:
        return False
    needle_len = len(needle)
    if needle_len > len(haystack):
        return False
    first = needle[0]
    stop = len(haystack) - needle_len + 1
    for idx in range(stop):
        if haystack[idx] != first:
            continue
        if haystack[idx : idx + needle_len] == needle:
            return True
    return False


def literal_match_scores(
    store: Any,
    literal_token_sequences: list[list[int]],
) -> dict[int, float]:
    """Score windows containing any encoded high-entropy literal."""
    if not literal_token_sequences:
        return {}
    token_lists = getattr(store, "window_token_lists", None)
    if not isinstance(token_lists, dict) or not token_lists:
        return {}

    scores: dict[int, float] = {}
    for raw_window_id, raw_tokens in token_lists.items():
        try:
            window_id = int(raw_window_id)
            window_tokens = [int(token_id) for token_id in raw_tokens]
        except (TypeError, ValueError):
            continue
        score = 0.0
        for sequence in literal_token_sequences:
            if contains_token_subsequence(window_tokens, sequence):
                score += float(len(sequence))
        if score > 0.0:
            scores[window_id] = score
    return scores


__all__ = [
    "contains_token_subsequence",
    "encode_literal_sequences",
    "extract_high_entropy_literals",
    "literal_match_scores",
]
