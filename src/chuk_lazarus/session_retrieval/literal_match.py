"""Exact literal matching helpers for cross-session routing."""

from __future__ import annotations

import re
from typing import Any

_LITERAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{5,}")
_TOKEN_INDEX_CACHE: dict[int, tuple[int, dict[int, list[tuple[int, list[int]]]]]] = {}


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

    cache_key = id(token_lists)
    cached = _TOKEN_INDEX_CACHE.get(cache_key)
    if cached is None or cached[0] != len(token_lists):
        first_token_index: dict[int, list[tuple[int, list[int]]]] = {}
        for raw_window_id, raw_tokens in token_lists.items():
            try:
                window_id = int(raw_window_id)
                window_tokens = [int(token_id) for token_id in raw_tokens]
            except (TypeError, ValueError):
                continue
            for token_id in set(window_tokens):
                first_token_index.setdefault(int(token_id), []).append(
                    (window_id, window_tokens)
                )
        cached = (len(token_lists), first_token_index)
        _TOKEN_INDEX_CACHE[cache_key] = cached
    first_token_index = cached[1]

    scores: dict[int, float] = {}
    for sequence in literal_token_sequences:
        if not sequence:
            continue
        for window_id, window_tokens in first_token_index.get(int(sequence[0]), []):
            if contains_token_subsequence(window_tokens, sequence):
                scores[window_id] = scores.get(window_id, 0.0) + float(len(sequence))
    return scores


__all__ = [
    "contains_token_subsequence",
    "encode_literal_sequences",
    "extract_high_entropy_literals",
    "literal_match_scores",
]
