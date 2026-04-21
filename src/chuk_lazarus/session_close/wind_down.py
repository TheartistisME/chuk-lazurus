"""End-to-end session-close orchestration; public entry points."""

from __future__ import annotations

from pathlib import Path

from chuk_lazarus.chat_loop.handoff import TranscriptHandoff
from chuk_lazarus.chat_loop.session import TurnRecord
from chuk_lazarus.session_close.aus3000_emit import (
    TextSlicer,
    build_records,
    write_records,
)


class WindDownEmitter:
    """Configurable orchestrator for emitting a session's per-chunk records."""

    def __init__(
        self,
        *,
        session_title: str | None = None,
        text_slicer: TextSlicer | None = None,
    ) -> None:
        self._session_title = session_title
        self._text_slicer = text_slicer

    def emit(
        self,
        turns: list[TurnRecord],
        session_id: str,
        out_dir: Path,
    ) -> list[Path]:
        """Build records from ``turns`` and write them under ``out_dir/session_id/``."""
        records = build_records(
            turns,
            session_id,
            session_title=self._session_title,
            text_slicer=self._text_slicer,
        )
        return write_records(Path(out_dir), session_id, records)


def emit_session(
    turns: list[TurnRecord],
    session_id: str,
    out_dir: Path,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[Path]:
    """End-to-end emission: build records from turns and write them to disk."""
    emitter = WindDownEmitter(
        session_title=session_title,
        text_slicer=text_slicer,
    )
    return emitter.emit(turns, session_id, Path(out_dir))


def emit_from_handoff(
    handoff: TranscriptHandoff,
    out_dir: Path,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[Path]:
    """Convenience wrapper: pulls session_id from handoff.raw_session()
    and turns from handoff.snapshot()."""
    session = handoff.raw_session()
    turns = handoff.snapshot()
    return emit_session(
        turns,
        session.session_id,
        Path(out_dir),
        session_title=session_title,
        text_slicer=text_slicer,
    )
