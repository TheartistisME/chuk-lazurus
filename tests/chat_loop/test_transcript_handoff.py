"""Unit tests for ChatLoopSession, TurnRecord, and TranscriptHandoff."""

from __future__ import annotations

import json
import re

from chuk_lazarus.chat_loop import (
    ChatLoopSession,
    ChunkBoundary,
    StreamingWindower,
    TranscriptHandoff,
    TurnRecord,
)
from chuk_lazarus.inference.chat import Role
from tests.chat_loop.conftest import make_text


ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
# session_id comes from uuid4().hex -> 32 lowercase hex chars, no dashes
UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _make_boundary(index: int, start: int, end: int) -> ChunkBoundary:
    return ChunkBoundary(
        chunk_index=index,
        start_token_offset=start,
        end_token_offset=end,
        emitted_at="2025-01-01T00:00:00.000000Z",
    )


def test_begin_turn_increments_index() -> None:
    session = ChatLoopSession()
    t0 = session.begin_turn(Role.USER, "hi")
    t1 = session.begin_turn(Role.ASSISTANT, "")
    assert t0.turn_index == 0
    assert t1.turn_index == 1
    assert session.turns == [t0, t1]


def test_finish_turn_sets_finished_at() -> None:
    session = ChatLoopSession()
    turn = session.begin_turn(Role.ASSISTANT, "hello")
    assert turn.finished_at is None
    assert ISO_Z_RE.match(turn.started_at) is not None

    session.finish_turn(turn)

    assert turn.finished_at is not None
    assert ISO_Z_RE.match(turn.finished_at) is not None
    # Same ISO format; lexical order matches temporal order
    assert turn.started_at <= turn.finished_at


def test_append_chunk_attaches_boundary() -> None:
    session = ChatLoopSession()
    turn = session.begin_turn(Role.ASSISTANT, "")
    b0 = _make_boundary(0, 0, 512)
    b1 = _make_boundary(1, 448, 960)

    session.append_chunk(turn, b0)
    session.append_chunk(turn, b1)

    assert turn.chunk_boundaries == [b0, b1]


def test_snapshot_is_deep_copy() -> None:
    session = ChatLoopSession()
    turn = session.begin_turn(Role.ASSISTANT, "x")
    session.append_chunk(turn, _make_boundary(0, 0, 512))

    handoff = TranscriptHandoff(session)
    snap = handoff.snapshot()

    assert len(snap) == 1
    assert len(snap[0].chunk_boundaries) == 1

    # Mutate the snapshot — live session must remain unchanged
    snap[0].chunk_boundaries.append(_make_boundary(1, 448, 960))
    assert len(snap[0].chunk_boundaries) == 2
    assert len(session.turns[0].chunk_boundaries) == 1


def test_raw_session_returns_live_object() -> None:
    session = ChatLoopSession()
    handoff = TranscriptHandoff(session)

    live = handoff.raw_session()
    assert live is session

    live.turns.append(
        TurnRecord(turn_index=0, role=Role.USER, text="live mutation")
    )
    assert len(session.turns) == 1
    assert session.turns[0].text == "live mutation"


def test_as_dict_list_round_trip() -> None:
    session = ChatLoopSession()
    user_turn = session.begin_turn(Role.USER, "hello")
    session.finish_turn(user_turn)
    asst = session.begin_turn(Role.ASSISTANT, "world")
    session.append_chunk(asst, _make_boundary(0, 0, 512))
    session.finish_turn(asst)

    handoff = TranscriptHandoff(session)
    dumped = handoff.as_dict_list()

    # JSON-serializable
    payload = json.dumps(dumped)
    assert isinstance(payload, str)

    assert len(dumped) == 2
    for item in dumped:
        assert "role" in item
        assert "text" in item
        assert "turn_index" in item
        assert "started_at" in item
        assert "chunk_boundaries" in item
        assert isinstance(item["chunk_boundaries"], list)

    # The assistant turn carries one chunk-boundary dict
    assert len(dumped[1]["chunk_boundaries"]) == 1
    assert dumped[1]["chunk_boundaries"][0]["start_token_offset"] == 0
    assert dumped[1]["chunk_boundaries"][0]["end_token_offset"] == 512


def test_session_ids_are_unique_uuid_hex() -> None:
    a = ChatLoopSession()
    b = ChatLoopSession()

    assert a.session_id != b.session_id
    # Implementation uses uuid4().hex (32 lowercase hex chars)
    assert UUID_HEX_RE.match(a.session_id) is not None
    assert UUID_HEX_RE.match(b.session_id) is not None


def test_end_to_end_windower_feeds_session(fake_tokenizer) -> None:
    session = ChatLoopSession()
    windower = StreamingWindower(fake_tokenizer)
    assistant_turn = session.begin_turn(Role.ASSISTANT, "")

    # 1024 tokens via two deltas: [0..512), [512..1024)
    for i, delta in enumerate([make_text(512, prefix="a"), make_text(512, prefix="b")]):
        for boundary in windower.feed_text(delta):
            session.append_chunk(assistant_turn, boundary)

    tail = windower.flush()
    if tail is not None:
        session.append_chunk(assistant_turn, tail)

    session.finish_turn(assistant_turn)

    # With window=512, overlap=64 and 1024 tokens:
    #   first emission  [0, 512)
    #   next_start = 448 -> 1024 - 448 = 576 >= 512 -> [448, 960)
    #   next_start = 896 -> 1024 - 896 = 128 < 512  -> stop
    #   flush tail -> [896, 1024)
    # Two full-window boundaries plus a tail.
    assert len(assistant_turn.chunk_boundaries) == 3

    offsets = [b.start_token_offset for b in assistant_turn.chunk_boundaries]
    assert offsets == sorted(offsets)
    assert offsets[0] < offsets[1] < offsets[2]

    handoff = TranscriptHandoff(session)
    snapped = handoff.snapshot()
    assert len(snapped) == 1
    assert len(snapped[0].chunk_boundaries) == len(assistant_turn.chunk_boundaries)
