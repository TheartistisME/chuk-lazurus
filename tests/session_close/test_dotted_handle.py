"""Unit tests for session_close.dotted_handle."""

from __future__ import annotations

import pytest

from chuk_lazarus.session_close.dotted_handle import compose_handle, parse_handle


SID = "0" * 32  # 32-char lowercase hex (axis-1 §6 invariant)
SID_REAL = "4f9b6d0e1a2b3c4d5e6f708192a3b4c5"


def test_compose_basic() -> None:
    assert compose_handle(SID, 0, 0) == f"{SID}.0.0"


def test_compose_nonzero_indices() -> None:
    assert compose_handle(SID_REAL, 7, 3) == f"{SID_REAL}.7.3"


def test_round_trip_zero() -> None:
    handle = compose_handle(SID, 0, 0)
    assert parse_handle(handle) == (SID, 0, 0)


def test_round_trip_nonzero() -> None:
    handle = compose_handle(SID_REAL, 12, 4)
    assert parse_handle(handle) == (SID_REAL, 12, 4)


def test_round_trip_large_indices() -> None:
    handle = compose_handle(SID, 999, 999)
    assert parse_handle(handle) == (SID, 999, 999)


def test_parse_raises_on_too_few_parts() -> None:
    with pytest.raises(ValueError):
        parse_handle("abc.0")


def test_parse_raises_on_too_many_parts() -> None:
    with pytest.raises(ValueError):
        parse_handle("abc.0.0.0")


def test_parse_raises_on_non_integer_turn() -> None:
    with pytest.raises(ValueError):
        parse_handle(f"{SID}.foo.0")


def test_parse_raises_on_non_integer_chunk() -> None:
    with pytest.raises(ValueError):
        parse_handle(f"{SID}.0.bar")


def test_parse_raises_on_empty_session_id() -> None:
    with pytest.raises(ValueError):
        parse_handle(".0.0")


def test_parse_raises_on_empty_string() -> None:
    with pytest.raises(ValueError):
        parse_handle("")
