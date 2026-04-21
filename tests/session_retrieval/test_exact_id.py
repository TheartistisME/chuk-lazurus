"""Tests for :mod:`chuk_lazarus.session_retrieval.exact_id` and handle helpers.

The new implementation bypasses the primitive's digit-only
``_CLAUSE_ID_RE`` by indexing into ``ClauseMetadataRouter.clause_id_to_primary_windows``
directly, which works for any verbatim clause_id string (including the
hex-prefixed shapes axis-2 actually emits). These tests therefore drive
``route_exact_id`` against REAL :class:`TorchKnowledgeStore` loads produced
by the ``two_sessions_checkpoint_root`` fixture — no ``load_store`` mocking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    iter_checkpoint_handles,
)
from chuk_lazarus.session_retrieval.exact_id import route_exact_id
from chuk_lazarus.session_retrieval.handle import (
    compose_handle,
    handle_belongs_to_session,
    parse_handle,
)


# ---------------------------------------------------------------------------
# parse_handle / compose_handle round-trip + predicate.
# ---------------------------------------------------------------------------


def test_compose_and_parse_roundtrip() -> None:
    """``parse_handle(compose_handle(...))`` returns the original tuple."""
    session_id = "a" * 32
    composed = compose_handle(session_id, 3, 5)
    parsed = parse_handle(composed)
    assert parsed == (session_id, 3, 5)


def test_parse_handle_malformed_raises() -> None:
    """Missing dotted parts raise ``ValueError``."""
    with pytest.raises(ValueError):
        parse_handle("only-one-piece")


def test_parse_handle_non_integer_raises() -> None:
    """Non-integer turn / chunk indices raise ``ValueError``."""
    with pytest.raises(ValueError):
        parse_handle(f"{'a' * 32}.x.0")


def test_handle_belongs_to_session_true_and_false() -> None:
    """Predicate reports True iff the session portion matches."""
    session_a = "a" * 32
    session_b = "b" * 32
    handle = compose_handle(session_a, 0, 0)
    assert handle_belongs_to_session(handle, session_a) is True
    assert handle_belongs_to_session(handle, session_b) is False


# ---------------------------------------------------------------------------
# route_exact_id: success / miss / malformed — real-store tests.
# ---------------------------------------------------------------------------


def test_route_exact_id_returns_handle_and_window_for_valid_dotted_handle(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """A valid ``<session_a>.0.0`` handle resolves to session_a + window 0."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    session_a = two_sessions_checkpoint_root["session_a"]
    handles = list(iter_checkpoint_handles(root))

    dotted = compose_handle(session_a, 0, 0)
    result = route_exact_id(handles, dotted)
    assert result is not None
    routed_handle, window_id = result
    assert routed_handle.session_id == session_a
    assert window_id == 0


def test_route_exact_id_returns_handle_and_window_for_session_a_window_1(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """``<session_a>.1.0`` resolves to session_a window 1.

    Asserts the lookup distinguishes between clauses within the same
    session and does not silently collapse to window 0.
    """
    root = two_sessions_checkpoint_root["checkpoint_root"]
    session_a = two_sessions_checkpoint_root["session_a"]
    handles = list(iter_checkpoint_handles(root))

    dotted = compose_handle(session_a, 1, 0)
    result = route_exact_id(handles, dotted)
    assert result is not None
    routed_handle, window_id = result
    assert routed_handle.session_id == session_a
    assert window_id == 1


def test_route_exact_id_returns_none_for_unknown_session(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Handle referring to a session not in ``handles`` returns ``None``."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    foreign_session = "z" * 32
    dotted = compose_handle(foreign_session, 0, 0)
    assert route_exact_id(handles, dotted) is None


def test_route_exact_id_returns_none_for_unknown_window_within_session(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Session exists but (turn, chunk) is not in window_metadata -> None."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    session_a = two_sessions_checkpoint_root["session_a"]
    handles = list(iter_checkpoint_handles(root))
    # session_a only has clauses "<sa>.0.0" and "<sa>.1.0"; .9.9 is absent.
    dotted = compose_handle(session_a, 9, 9)
    assert route_exact_id(handles, dotted) is None


def test_route_exact_id_raises_on_malformed_handle(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Malformed dotted handle propagates ``ValueError`` from ``parse_handle``."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    with pytest.raises(ValueError):
        route_exact_id(handles, "not-a-handle")


def test_route_exact_id_empty_handles_returns_none() -> None:
    """Empty handle list -> None without attempting to load any store."""
    session_a = "a" * 32
    dotted = compose_handle(session_a, 0, 0)
    assert route_exact_id([], dotted) is None


def test_route_exact_id_resolves_session_b(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Lookup into the second session resolves correctly (cross-session sanity)."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    session_b = two_sessions_checkpoint_root["session_b"]
    handles = list(iter_checkpoint_handles(root))

    dotted = compose_handle(session_b, 0, 0)
    result = route_exact_id(handles, dotted)
    assert result is not None
    routed_handle, window_id = result
    assert routed_handle.session_id == session_b
    assert window_id == 0


# ---------------------------------------------------------------------------
# CheckpointHandle type sanity — imported but not otherwise exercised here.
# ---------------------------------------------------------------------------


def test_checkpoint_handle_type_is_importable() -> None:
    """Minimal smoke-test that the dataclass is importable in this module."""
    assert CheckpointHandle.__name__ == "CheckpointHandle"
