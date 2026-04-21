"""End-to-end tests for session_close.wind_down."""

from __future__ import annotations

import json
import re
from pathlib import Path

from chuk_lazarus.chat_loop.handoff import TranscriptHandoff
from chuk_lazarus.chat_loop.session import ChatLoopSession, ChunkBoundary, TurnRecord
from chuk_lazarus.inference.chat import Role
from chuk_lazarus.session_close.dotted_handle import parse_handle
from chuk_lazarus.session_close.wind_down import (
    WindDownEmitter,
    emit_from_handoff,
    emit_session,
)


SID = "4f9b6d0e1a2b3c4d5e6f708192a3b4c5"
HANDLE_RE = re.compile(r"^[0-9a-f]{32}\.\d+\.\d+$")


def _synthetic_turns() -> list[TurnRecord]:
    return [
        TurnRecord(
            turn_index=0,
            role=Role.USER,
            text="What does ISO 9001 clause 4.2 require?",
            started_at="2026-04-21T10:00:00.000000Z",
            finished_at="2026-04-21T10:00:00.500000Z",
            chunk_boundaries=[],
        ),
        TurnRecord(
            turn_index=1,
            role=Role.ASSISTANT,
            text="Clause 4.2 requires documented information for the management system.",
            started_at="2026-04-21T10:00:01.000000Z",
            finished_at="2026-04-21T10:00:02.500000Z",
            chunk_boundaries=[
                ChunkBoundary(
                    chunk_index=0,
                    start_token_offset=0,
                    end_token_offset=512,
                    emitted_at="2026-04-21T10:00:02.000000Z",
                ),
                ChunkBoundary(
                    chunk_index=1,
                    start_token_offset=512,
                    end_token_offset=800,
                    emitted_at="2026-04-21T10:00:02.300000Z",
                ),
            ],
        ),
        TurnRecord(
            turn_index=2,
            role=Role.USER,
            text="Thank you.",
            started_at="2026-04-21T10:00:03.000000Z",
            finished_at="2026-04-21T10:00:03.100000Z",
            chunk_boundaries=[],
        ),
    ]


def test_emit_session_creates_directory_and_files(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    written = emit_session(turns, SID, tmp_path)
    session_dir = tmp_path / SID
    assert session_dir.is_dir()
    # 1 user + 2 assistant chunks + 1 user = 4
    assert len(written) == 4
    for p in written:
        assert p.parent == session_dir
        assert p.exists()
        assert p.suffix == ".json"


def test_emit_session_each_file_is_valid_json(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    written = emit_session(turns, SID, tmp_path)
    for p in written:
        obj = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(obj, dict)
        for field in (
            "standard_id",
            "standard_title",
            "clause_id",
            "clause_title",
            "clause_content",
        ):
            assert isinstance(obj[field], str)
            assert obj[field].strip()


def test_emit_session_clause_ids_are_three_part(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    written = emit_session(turns, SID, tmp_path)
    for p in written:
        obj = json.loads(p.read_text(encoding="utf-8"))
        handle = obj["clause_id"]
        assert HANDLE_RE.match(handle), handle
        session_id, turn_index, chunk_index = parse_handle(handle)
        assert session_id == SID
        assert isinstance(turn_index, int)
        assert isinstance(chunk_index, int)


def test_emit_session_emission_counter_monotonic(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    written = emit_session(turns, SID, tmp_path)
    # Filenames sorted by name should correspond to emission order (000, 001, 002, 003).
    names = sorted(p.name for p in written)
    prefixes = [int(n.split("_", 1)[0]) for n in names]
    assert prefixes == [0, 1, 2, 3]


def test_wind_down_emitter_class_matches_function(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    emit_session(turns, SID, out_a)
    WindDownEmitter().emit(turns, SID, out_b)

    a_files = sorted((out_a / SID).glob("*.json"))
    b_files = sorted((out_b / SID).glob("*.json"))
    assert [p.name for p in a_files] == [p.name for p in b_files]
    for a, b in zip(a_files, b_files):
        assert json.loads(a.read_text(encoding="utf-8")) == json.loads(
            b.read_text(encoding="utf-8")
        )


def test_wind_down_emitter_configured_session_title(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    emitter = WindDownEmitter(session_title="Hand-picked title")
    written = emitter.emit(turns, SID, tmp_path)
    for p in written:
        obj = json.loads(p.read_text(encoding="utf-8"))
        assert obj["standard_title"] == "Hand-picked title"


def test_emit_from_handoff(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    session = ChatLoopSession(
        session_id=SID,
        started_at="2026-04-21T10:00:00.000000Z",
        turns=turns,
    )
    handoff = TranscriptHandoff(session)
    written = emit_from_handoff(handoff, tmp_path)
    assert len(written) == 4
    for p in written:
        assert (tmp_path / SID / p.name).exists()


def test_emit_session_rerun_overwrites_same_files(tmp_path: Path) -> None:
    turns = _synthetic_turns()
    first = emit_session(turns, SID, tmp_path)
    # Second call with same inputs must produce same filenames + identical content.
    second = emit_session(turns, SID, tmp_path)
    assert [p.name for p in first] == [p.name for p in second]
    for p in second:
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["standard_id"] == SID
