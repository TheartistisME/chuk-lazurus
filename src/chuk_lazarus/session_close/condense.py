"""Deterministic, no-LLM title condensation (design §5)."""

from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"\s+")

# Sentence terminators considered by §5 step 3.
_SINGLE_CHAR_TERMINATORS = (".", "!", "?")
_BLANK_LINE_TERMINATOR = "\n\n"


def condense_title(
    content: str,
    role: str,
    turn_index: int,
    *,
    max_chars: int = 160,
) -> str:
    """Produce a short, deterministic, non-empty title for a clause.

    Follows design §5:
      1. Strip leading/trailing whitespace.
      2. If empty, return ``"<role> turn <turn_index>"``.
      3. Take the substring up to (but not including) the first sentence
         terminator (``.``, ``!``, ``?``, or a blank line ``\\n\\n``).
      4. Collapse internal whitespace runs to single spaces.
      5. If longer than ``max_chars``, truncate to ``max_chars - 3`` and
         append ``"..."``.
    """
    stripped = content.strip()
    if not stripped:
        return f"{role} turn {turn_index}"

    terminator_positions: list[int] = []
    for terminator in _SINGLE_CHAR_TERMINATORS:
        idx = stripped.find(terminator)
        if idx != -1:
            terminator_positions.append(idx)
    blank_idx = stripped.find(_BLANK_LINE_TERMINATOR)
    if blank_idx != -1:
        terminator_positions.append(blank_idx)

    if terminator_positions:
        cut = min(terminator_positions)
        head = stripped[:cut]
    else:
        head = stripped

    # Collapse all internal whitespace (including newlines/tabs) to single spaces.
    collapsed = _WHITESPACE_RUN.sub(" ", head).strip()

    # The head may have been empty if the text begins with a terminator (e.g. ".Hello").
    # Fall back to the synthesized title in that case — clause_title MUST be non-empty.
    if not collapsed:
        return f"{role} turn {turn_index}"

    if len(collapsed) > max_chars:
        if max_chars <= 3:
            # Degenerate configuration; still return a non-empty string.
            return collapsed[:max_chars]
        return collapsed[: max_chars - 3] + "..."

    return collapsed
