"""Unit tests for StreamingWindower hot-path windowing logic."""

from __future__ import annotations

import re

import pytest

from chuk_lazarus.chat_loop import ChunkBoundary, StreamingWindower
from tests.chat_loop.conftest import make_text


ISO_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def test_no_window_below_threshold(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    # Feed 500 tokens across 5 deltas of 100 each
    emitted_total: list[ChunkBoundary] = []
    for i in range(5):
        emitted = w.feed_text(make_text(100, prefix=f"d{i}t"))
        assert emitted == []
        emitted_total.extend(emitted)

    assert emitted_total == []
    assert w.emitted_chunks == 0
    assert w.total_tokens == 500


def test_exact_boundary_emits_one_window(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    emitted = w.feed_text(make_text(512))

    assert len(emitted) == 1
    boundary = emitted[0]
    assert boundary.chunk_index == 0
    assert boundary.start_token_offset == 0
    assert boundary.end_token_offset == 512
    assert w.emitted_chunks == 1
    assert w.total_tokens == 512


def test_overlap_advances_by_448(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer, window_tokens=512, overlap_tokens=64)
    emitted = w.feed_text(make_text(960))

    assert len(emitted) == 2
    first, second = emitted
    assert (first.start_token_offset, first.end_token_offset) == (0, 512)
    assert first.chunk_index == 0
    assert (second.start_token_offset, second.end_token_offset) == (448, 960)
    assert second.chunk_index == 1
    assert w.total_tokens == 960
    assert w.emitted_chunks == 2


def test_streaming_multiple_small_deltas(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    last_emitted: list[ChunkBoundary] = []
    for i in range(16):
        last_emitted = w.feed_text(make_text(32, prefix=f"b{i}t"))

    # Final delta completes the 512-token window
    assert len(last_emitted) == 1
    assert last_emitted[0].chunk_index == 0
    assert last_emitted[0].start_token_offset == 0
    assert last_emitted[0].end_token_offset == 512
    assert w.emitted_chunks == 1
    assert w.total_tokens == 512


def test_flush_emits_tail(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    emitted = w.feed_text(make_text(700))
    assert len(emitted) == 1
    assert (emitted[0].start_token_offset, emitted[0].end_token_offset) == (0, 512)

    tail = w.flush()
    assert tail is not None
    assert tail.start_token_offset == 448
    assert tail.end_token_offset == 700
    assert tail.chunk_index == 1

    # Idempotent after tail emission
    second_flush = w.flush()
    assert second_flush is None


def test_flush_no_tail_when_aligned(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    emitted = w.feed_text(make_text(512))
    assert len(emitted) == 1
    assert w.flush() is None


def test_reset_clears_state(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    w.feed_text(make_text(700))
    w.flush()
    assert w.total_tokens == 700
    assert w.emitted_chunks == 2

    w.reset()
    assert w.total_tokens == 0
    assert w.emitted_chunks == 0
    # Private attribute check for the start-offset cursor
    assert w._next_window_start_offset == 0  # noqa: SLF001 - test intent


@pytest.mark.parametrize(
    "window_tokens,overlap_tokens",
    [
        (512, 512),  # equal -> invalid
        (512, 600),  # overlap > window -> invalid
        (-1, 0),  # negative window
        (0, 0),  # zero window
        (512, -1),  # negative overlap
    ],
)
def test_invalid_config_raises(fake_tokenizer, window_tokens, overlap_tokens) -> None:
    with pytest.raises(ValueError):
        StreamingWindower(
            fake_tokenizer,
            window_tokens=window_tokens,
            overlap_tokens=overlap_tokens,
        )


def test_callback_fires_on_emit(fake_tokenizer) -> None:
    received: list[tuple[ChunkBoundary, str]] = []

    def on_window(boundary: ChunkBoundary, decoded: str) -> None:
        received.append((boundary, decoded))

    w = StreamingWindower(fake_tokenizer, on_window=on_window)
    w.feed_text(make_text(512))

    assert len(received) == 1
    boundary, decoded_text = received[0]
    assert boundary.chunk_index == 0
    assert isinstance(decoded_text, str)
    assert decoded_text != ""
    # Flush tail with callback too
    tail_tokens = make_text(100, prefix="x")
    w.feed_text(tail_tokens)
    tail = w.flush()
    assert tail is not None
    assert len(received) == 2
    assert received[1][1] != ""


def test_timestamps_are_utc_iso_z(fake_tokenizer) -> None:
    w = StreamingWindower(fake_tokenizer)
    emitted = w.feed_text(make_text(512))
    assert len(emitted) == 1
    assert ISO_Z_RE.match(emitted[0].emitted_at) is not None

    # Tail timestamp too
    w.feed_text(make_text(100, prefix="y"))
    tail = w.flush()
    assert tail is not None
    assert ISO_Z_RE.match(tail.emitted_at) is not None
