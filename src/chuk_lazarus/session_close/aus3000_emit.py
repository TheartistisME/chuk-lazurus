"""TurnRecord list -> per-chunk AUS3000-shaped dicts; JSON serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from chuk_lazarus.chat_loop.session import TurnRecord
from chuk_lazarus.session_close.condense import condense_title
from chuk_lazarus.session_close.dotted_handle import compose_handle, parse_handle
from chuk_lazarus.session_close.topic_tags import extract_topic_tags

TextSlicer = Callable[[str, int, int], str]


def session_title_from_turns(
    turns: list[TurnRecord],
    session_id: str,
    *,
    max_chars: int = 120,
) -> str:
    """First non-empty user-turn text trimmed to ``max_chars``.

    Falls back to ``"chat-session <session_id[:8]>"`` if no user turn has
    non-empty text. Guaranteed non-empty (axis-1 §6 guarantees a 32-char
    lowercase hex session_id).
    """
    for turn in turns:
        if turn.role.value != "user":
            continue
        candidate = turn.text.strip()
        if not candidate:
            continue
        if len(candidate) > max_chars:
            return candidate[:max_chars]
        return candidate
    return f"chat-session {session_id[:8]}"


def _resolve_chunk_text(
    turn: TurnRecord,
    chunk_start: int,
    chunk_end: int,
    text_slicer: TextSlicer | None,
) -> str:
    """Return the content string for a single chunk.

    Uses the caller-supplied ``text_slicer`` when given; otherwise falls
    back to the POC default (full-turn text per chunk) per design §4.
    """
    if text_slicer is not None:
        return text_slicer(turn.text, chunk_start, chunk_end)
    return turn.text


def _fallback_content(turn: TurnRecord) -> str:
    """Non-empty content fallback when turn.text is empty/whitespace.

    Loader tolerates empty clause_content (falls back to clause_title),
    but axis-2 emits non-empty clause_content as a matter of discipline
    (design §3).
    """
    stripped = turn.text.strip()
    if stripped:
        return turn.text
    return f"{turn.role.value} turn {turn.turn_index}"


def build_records(
    turns: list[TurnRecord],
    session_id: str,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[dict]:
    """Produce one AUS3000-shaped dict per (turn, chunk). See §3-§7.

    If a turn has empty ``chunk_boundaries``, exactly one record is
    emitted for that turn with ``chunk_index = 0`` (design §4 edge case).
    """
    title = session_title if session_title is not None else session_title_from_turns(
        turns, session_id
    )
    title = title.strip() or f"chat-session {session_id[:8]}"

    records: list[dict] = []
    for turn in turns:
        boundaries = turn.chunk_boundaries
        if not boundaries:
            chunk_text = _resolve_chunk_text(turn, 0, 0, text_slicer)
            if not chunk_text.strip():
                chunk_text = _fallback_content(turn)
            records.append(
                _build_single_record(
                    session_id=session_id,
                    session_title=title,
                    turn=turn,
                    chunk_index=0,
                    chunk_text=chunk_text,
                )
            )
            continue

        for boundary in boundaries:
            chunk_text = _resolve_chunk_text(
                turn,
                boundary.start_token_offset,
                boundary.end_token_offset,
                text_slicer,
            )
            if not chunk_text.strip():
                chunk_text = _fallback_content(turn)
            records.append(
                _build_single_record(
                    session_id=session_id,
                    session_title=title,
                    turn=turn,
                    chunk_index=boundary.chunk_index,
                    chunk_text=chunk_text,
                )
            )
    return records


def _build_single_record(
    *,
    session_id: str,
    session_title: str,
    turn: TurnRecord,
    chunk_index: int,
    chunk_text: str,
) -> dict:
    """Assemble one AUS3000-shaped dict for a single chunk."""
    clause_id = compose_handle(session_id, turn.turn_index, chunk_index)
    clause_title = condense_title(chunk_text, turn.role.value, turn.turn_index)

    iso_timestamp = turn.finished_at if turn.finished_at is not None else turn.started_at

    return {
        "standard_id": session_id,
        "standard_title": session_title,
        "clause_id": clause_id,
        "clause_title": clause_title,
        "clause_content": chunk_text,
        "iso_timestamp": iso_timestamp,
        "speaker_role": turn.role.value,
        "session_uuid": session_id,
        "topic_tags": extract_topic_tags(chunk_text),
    }


def write_records(
    out_dir: Path,
    session_id: str,
    records: list[dict],
) -> list[Path]:
    """Write records to ``<out_dir>/<session_id>/<NNN>_<ti>_<ci>.json``.

    Returns the written paths in emission order. Existing same-named
    files are overwritten; unrelated files in the target dir are
    preserved (POC behavior).
    """
    session_dir = Path(out_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    width = max(3, len(str(total - 1))) if total > 0 else 3

    written: list[Path] = []
    for emission_counter, record in enumerate(records):
        _, turn_index, chunk_index = parse_handle(record["clause_id"])
        filename = f"{emission_counter:0{width}d}_{turn_index}_{chunk_index}.json"
        path = session_dir / filename
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written.append(path)
    return written
