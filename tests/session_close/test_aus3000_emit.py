"""Unit tests for session_close.aus3000_emit."""

from __future__ import annotations

import json
import re
from pathlib import Path

from chuk_lazarus.chat_loop.session import ChunkBoundary, TurnRecord
from chuk_lazarus.inference.chat import Role
from chuk_lazarus.session_close.aus3000_emit import (
    build_records,
    session_title_from_turns,
    write_records,
)


SID = "4f9b6d0e1a2b3c4d5e6f708192a3b4c5"  # 32-char lowercase hex
HANDLE_RE = re.compile(r"^[0-9a-f]{32}\.\d+\.\d+$")


def _make_turn(
    *,
    turn_index: int,
    role: Role,
    text: str,
    started_at: str = "2026-04-21T10:00:00.000000Z",
    finished_at: str | None = "2026-04-21T10:00:01.000000Z",
    boundaries: list[ChunkBoundary] | None = None,
) -> TurnRecord:
    return TurnRecord(
        turn_index=turn_index,
        role=role,
        text=text,
        started_at=started_at,
        finished_at=finished_at,
        chunk_boundaries=boundaries or [],
    )


def test_user_turn_emits_exactly_one_record_with_chunk_index_zero() -> None:
    turn = _make_turn(turn_index=0, role=Role.USER, text="Please summarize clause 4.2.")
    records = build_records([turn], SID)
    assert len(records) == 1
    r = records[0]
    assert r["clause_id"] == f"{SID}.0.0"
    assert r["speaker_role"] == "user"


def test_assistant_turn_with_two_chunks_emits_two_records() -> None:
    boundaries = [
        ChunkBoundary(
            chunk_index=0,
            start_token_offset=0,
            end_token_offset=512,
            emitted_at="2026-04-21T10:00:02.000000Z",
        ),
        ChunkBoundary(
            chunk_index=1,
            start_token_offset=512,
            end_token_offset=900,
            emitted_at="2026-04-21T10:00:03.000000Z",
        ),
    ]
    turn = _make_turn(
        turn_index=1,
        role=Role.ASSISTANT,
        text="Clause 4.2 requires documented procedures for clause review.",
        boundaries=boundaries,
    )
    records = build_records([turn], SID)
    assert len(records) == 2
    assert records[0]["clause_id"] == f"{SID}.1.0"
    assert records[1]["clause_id"] == f"{SID}.1.1"
    for r in records:
        assert r["speaker_role"] == "assistant"


def test_mapping_correctness_all_fields() -> None:
    turn = _make_turn(
        turn_index=0,
        role=Role.USER,
        text="What is ISO 9001?",
        started_at="2026-04-21T10:00:00.000000Z",
        finished_at="2026-04-21T10:00:01.000000Z",
    )
    records = build_records([turn], SID, session_title="ISO conversation")
    r = records[0]
    # Primary ClauseRecord fields
    assert r["standard_id"] == SID
    assert r["standard_title"] == "ISO conversation"
    assert r["clause_id"] == f"{SID}.0.0"
    assert r["clause_title"]  # non-empty
    assert r["clause_content"] == "What is ISO 9001?"
    # Secondary metadata
    assert r["iso_timestamp"] == "2026-04-21T10:00:01.000000Z"  # finished_at preferred
    assert r["speaker_role"] == "user"
    assert r["session_uuid"] == SID
    assert isinstance(r["topic_tags"], list)
    assert len(r["topic_tags"]) <= 5


def test_iso_timestamp_falls_back_to_started_at_when_finished_at_none() -> None:
    turn = _make_turn(
        turn_index=0,
        role=Role.USER,
        text="Hi there",
        started_at="2026-04-21T10:00:00.000000Z",
        finished_at=None,
    )
    records = build_records([turn], SID)
    assert records[0]["iso_timestamp"] == "2026-04-21T10:00:00.000000Z"


def test_session_title_fallback_used_when_no_user_turn() -> None:
    turn = _make_turn(
        turn_index=0,
        role=Role.ASSISTANT,
        text="Assistant-initiated greeting.",
    )
    records = build_records([turn], SID)
    expected_fallback = f"chat-session {SID[:8]}"
    assert records[0]["standard_title"] == expected_fallback


def test_session_title_from_turns_picks_first_nonempty_user_turn() -> None:
    turns = [
        _make_turn(turn_index=0, role=Role.ASSISTANT, text="Hello!"),
        _make_turn(turn_index=1, role=Role.USER, text="   "),  # empty-after-strip
        _make_turn(turn_index=2, role=Role.USER, text="Real question here."),
        _make_turn(turn_index=3, role=Role.USER, text="Ignored second user turn."),
    ]
    assert session_title_from_turns(turns, SID) == "Real question here."


def test_session_title_from_turns_truncates() -> None:
    long_text = "x" * 200
    turns = [_make_turn(turn_index=0, role=Role.USER, text=long_text)]
    title = session_title_from_turns(turns, SID, max_chars=120)
    assert len(title) == 120
    assert title == "x" * 120


def test_session_title_explicit_override_wins() -> None:
    turns = [_make_turn(turn_index=0, role=Role.USER, text="User asked something.")]
    records = build_records(turns, SID, session_title="Explicit title")
    assert records[0]["standard_title"] == "Explicit title"


def test_session_title_explicit_empty_falls_back() -> None:
    turns = [_make_turn(turn_index=0, role=Role.USER, text="Hello.")]
    # Explicit whitespace-only override should fall back to synthesized title.
    records = build_records(turns, SID, session_title="   ")
    assert records[0]["standard_title"] == f"chat-session {SID[:8]}"


def test_empty_turn_text_produces_nonempty_content_and_title() -> None:
    turn = _make_turn(turn_index=2, role=Role.ASSISTANT, text="")
    records = build_records([turn], SID)
    r = records[0]
    assert r["clause_title"].strip()
    assert r["clause_content"].strip()


def test_clause_id_matches_canonical_pattern() -> None:
    turns = [
        _make_turn(turn_index=0, role=Role.USER, text="Hello."),
        _make_turn(
            turn_index=1,
            role=Role.ASSISTANT,
            text="Reply body.",
            boundaries=[
                ChunkBoundary(
                    chunk_index=0,
                    start_token_offset=0,
                    end_token_offset=10,
                    emitted_at="2026-04-21T10:00:02.000000Z",
                )
            ],
        ),
    ]
    records = build_records(turns, SID)
    for r in records:
        assert HANDLE_RE.match(r["clause_id"]), r["clause_id"]


def test_text_slicer_invoked_when_supplied() -> None:
    seen: list[tuple[str, int, int]] = []

    def slicer(text: str, start: int, end: int) -> str:
        seen.append((text, start, end))
        return f"SLICE[{start}:{end}]{text}"

    boundaries = [
        ChunkBoundary(
            chunk_index=0,
            start_token_offset=3,
            end_token_offset=7,
            emitted_at="2026-04-21T10:00:02.000000Z",
        ),
    ]
    turn = _make_turn(
        turn_index=0,
        role=Role.ASSISTANT,
        text="Reply body with chunks.",
        boundaries=boundaries,
    )
    records = build_records([turn], SID, text_slicer=slicer)
    assert len(records) == 1
    assert seen == [("Reply body with chunks.", 3, 7)]
    assert records[0]["clause_content"] == "SLICE[3:7]Reply body with chunks."


def test_write_records_filename_layout(tmp_path: Path) -> None:
    turns = [
        _make_turn(turn_index=0, role=Role.USER, text="First question."),
        _make_turn(
            turn_index=1,
            role=Role.ASSISTANT,
            text="First answer.",
            boundaries=[
                ChunkBoundary(
                    chunk_index=0,
                    start_token_offset=0,
                    end_token_offset=10,
                    emitted_at="2026-04-21T10:00:02.000000Z",
                ),
                ChunkBoundary(
                    chunk_index=1,
                    start_token_offset=10,
                    end_token_offset=20,
                    emitted_at="2026-04-21T10:00:03.000000Z",
                ),
            ],
        ),
    ]
    records = build_records(turns, SID)
    written = write_records(tmp_path, SID, records)

    assert len(written) == 3
    session_dir = tmp_path / SID
    assert session_dir.is_dir()

    # Expected filenames: 000_0_0.json, 001_1_0.json, 002_1_1.json
    names = [p.name for p in written]
    assert names == ["000_0_0.json", "001_1_0.json", "002_1_1.json"]
    for p in written:
        # Each file is valid JSON with the required fields.
        obj = json.loads(p.read_text(encoding="utf-8"))
        for field in (
            "standard_id",
            "standard_title",
            "clause_id",
            "clause_title",
            "clause_content",
            "iso_timestamp",
            "speaker_role",
            "session_uuid",
            "topic_tags",
        ):
            assert field in obj, f"missing {field} in {p.name}"


def test_write_records_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "out"
    turn = _make_turn(turn_index=0, role=Role.USER, text="Hello.")
    records = build_records([turn], SID)
    written = write_records(nested, SID, records)
    assert len(written) == 1
    assert written[0].exists()


def test_write_records_utf8_no_bom(tmp_path: Path) -> None:
    turn = _make_turn(turn_index=0, role=Role.USER, text="Café naïve résumé.")
    records = build_records([turn], SID)
    written = write_records(tmp_path, SID, records)
    raw = written[0].read_bytes()
    # No UTF-8 BOM at the start
    assert not raw.startswith(b"\xef\xbb\xbf")
    # ensure_ascii=False preserves the unicode directly
    assert "Café" in raw.decode("utf-8")


def test_all_required_string_fields_nonempty_after_strip() -> None:
    # Pathological inputs that still must produce loader-safe records.
    turns = [
        _make_turn(turn_index=0, role=Role.USER, text=""),  # empty user turn
        _make_turn(turn_index=1, role=Role.ASSISTANT, text="   "),  # whitespace only
    ]
    records = build_records(turns, SID)
    for r in records:
        for field in ("standard_id", "standard_title", "clause_id", "clause_title", "clause_content"):
            assert r[field].strip(), f"{field} is empty/whitespace: {r[field]!r}"
