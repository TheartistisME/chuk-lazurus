"""Compose/parse the dotted clause-id handle ``<session_id>.<turn_index>.<chunk_index>``."""

from __future__ import annotations


def compose_handle(session_id: str, turn_index: int, chunk_index: int) -> str:
    """Build the canonical dotted handle for a (turn, chunk) pair.

    The handle shape is ``"<session_id>.<turn_index>.<chunk_index>"`` where
    ``session_id`` is a 32-char lowercase hex string (axis-1 §6), and the
    two indices are non-negative decimal integers.
    """
    return f"{session_id}.{turn_index}.{chunk_index}"


def parse_handle(handle: str) -> tuple[str, int, int]:
    """Parse a dotted handle into ``(session_id, turn_index, chunk_index)``.

    Raises ``ValueError`` if the handle does not have exactly three parts
    or if the trailing two parts are not valid integers.
    """
    parts = handle.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 dot-separated parts in handle, got {len(parts)}: {handle!r}"
        )
    session_id, turn_str, chunk_str = parts
    if not session_id:
        raise ValueError(f"Empty session_id component in handle: {handle!r}")
    try:
        turn_index = int(turn_str)
        chunk_index = int(chunk_str)
    except ValueError as exc:
        raise ValueError(
            f"Non-integer turn_index or chunk_index in handle: {handle!r}"
        ) from exc
    return session_id, turn_index, chunk_index
