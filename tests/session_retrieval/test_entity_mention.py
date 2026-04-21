"""Tests for :mod:`chuk_lazarus.session_retrieval.entity_mention`.

Covers:
- ``extract_entity_tokens`` regex + stopword filtering + quoted-string handling.
- ``score_window_entity_overlap`` against the real two-session fixture.
- ``route_entity_mention`` cross-session routing with deterministic tie-breaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chuk_lazarus.session_retrieval import entity_mention as entity_mention_module
from chuk_lazarus.session_retrieval.entity_mention import (
    extract_entity_tokens,
    route_entity_mention,
    score_window_entity_overlap,
)
from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    iter_checkpoint_handles,
)


# ---------------------------------------------------------------------------
# extract_entity_tokens
# ---------------------------------------------------------------------------


def test_extract_entity_tokens_capitalised() -> None:
    """Capitalised tokens are extracted, stopwords are dropped."""
    tokens = extract_entity_tokens(
        "Who was Alice talking about in that Dubai thread?"
    )
    assert "alice" in tokens
    assert "dubai" in tokens
    for stopword in ("who", "was", "that", "in", "about"):
        assert stopword not in tokens


def test_extract_entity_tokens_from_quoted_string() -> None:
    """Quoted-string contents contribute their inner tokens."""
    tokens = extract_entity_tokens('He said "Project Lazarus" matters.')
    assert "project" in tokens
    assert "lazarus" in tokens


def test_extract_entity_tokens_lowercase_only_returns_empty() -> None:
    """An all-lowercase sentence with no quotes yields no entity tokens."""
    tokens = extract_entity_tokens("who was he talking about in that thread?")
    assert tokens == []


def test_extract_entity_tokens_preserves_first_seen_order_and_dedupes() -> None:
    """Duplicates are dropped while preserving first-seen ordering."""
    tokens = extract_entity_tokens("Alice, Bob, then Alice again and Bob.")
    assert tokens == ["alice", "bob"]


def test_extract_entity_tokens_all_caps_accepted() -> None:
    """ALL-CAPS tokens of length >= 2 are kept (after lowercasing)."""
    tokens = extract_entity_tokens("Ask NASA about Voyager.")
    assert "nasa" in tokens
    assert "voyager" in tokens


# ---------------------------------------------------------------------------
# score_window_entity_overlap (real store load via fixture)
# ---------------------------------------------------------------------------


def test_score_window_entity_overlap_full_match(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """session_a w0 keywords {alice,dubai,meeting} vs {alice,dubai} -> 1.0."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    session_a_handle = handles[0]
    entity_tokens = {"alice", "dubai"}
    score = score_window_entity_overlap(session_a_handle, 0, entity_tokens)
    assert score == pytest.approx(1.0)


def test_score_window_entity_overlap_no_match(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Disjoint entity set vs window keywords -> 0.0."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    session_a_handle = handles[0]
    score = score_window_entity_overlap(session_a_handle, 0, {"zzz", "nope"})
    assert score == pytest.approx(0.0)


def test_score_window_entity_overlap_empty_entities(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Empty entity set -> 0.0 (helper is resilient to an empty query)."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    session_a_handle = handles[0]
    assert score_window_entity_overlap(session_a_handle, 0, set()) == 0.0


# ---------------------------------------------------------------------------
# route_entity_mention (real store load via fixture)
# ---------------------------------------------------------------------------


def test_route_entity_mention_alice_dubai_routes_to_session_a(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """'Alice Dubai' routes to session_a window 0 with score ~1.0."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    original_root = two_sessions_checkpoint_root["original_input_root"]
    handles = list(iter_checkpoint_handles(root, original_input_root=original_root))
    result = route_entity_mention(handles, "Alice Dubai")
    assert result is not None
    handle, window_id, score = result
    assert handle.session_id == two_sessions_checkpoint_root["session_a"]
    assert window_id == 0
    assert score == pytest.approx(1.0)


def test_route_entity_mention_charlie_lazarus_routes_to_session_b(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """'Charlie Lazarus' routes to session_b window 0 with score ~1.0."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    original_root = two_sessions_checkpoint_root["original_input_root"]
    handles = list(iter_checkpoint_handles(root, original_input_root=original_root))
    result = route_entity_mention(handles, "Charlie Lazarus")
    assert result is not None
    handle, window_id, score = result
    assert handle.session_id == two_sessions_checkpoint_root["session_b"]
    assert window_id == 0
    assert score == pytest.approx(1.0)


def test_route_entity_mention_no_overlap_returns_none(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Query with capitalised entities not in any store -> None."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    assert route_entity_mention(handles, "Xenon Krypton") is None


def test_route_entity_mention_empty_entity_tokens_returns_none(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """All-lowercase query -> no entity tokens -> short-circuit None."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    assert route_entity_mention(handles, "nothing here to find") is None


def test_route_entity_mention_tie_break_by_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Identical scores across sessions -> smaller session_id wins."""
    # Two handles, identical keyword content, identical window count. Both
    # windows match a single entity token; tie-break must pick session_a.
    session_a = "a" * 32
    session_b = "b" * 32

    def _make_handle(sid: str) -> CheckpointHandle:
        return CheckpointHandle(
            session_id=sid,
            checkpoint_dir=tmp_path / sid,
            torch_store_dir=tmp_path / sid / "torch_store",
            manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
            original_input_dir=None,
        )

    def _fake_store() -> Any:
        store = MagicMock()
        store.num_windows = 1
        store.keywords = {0: ["alice"]}
        store.window_metadata = {0: {"clause_id": "1.0.0"}}
        return store

    handle_a = _make_handle(session_a)
    handle_b = _make_handle(session_b)
    monkeypatch.setattr(entity_mention_module, "load_store", lambda _h: _fake_store())
    monkeypatch.setattr(
        entity_mention_module, "load_original_turn_records", lambda _h: []
    )

    result = route_entity_mention([handle_b, handle_a], "Alice")
    assert result is not None
    handle, window_id, score = result
    assert handle.session_id == session_a
    assert window_id == 0
    assert score == pytest.approx(1.0)
