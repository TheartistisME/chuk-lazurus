"""Tests for :mod:`chuk_lazarus.session_retrieval.topical`.

Uses monkeypatched ``load_store`` inside the ``topical`` module so we do not
depend on the primitive's TF-IDF routing for pure-unit coverage. The stub
store exposes ``route_top_k`` and ``_get_tfidf_router().score_window`` so the
topical router's global ranking + deterministic tie-break is exercised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chuk_lazarus.session_retrieval import topical as topical_module
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.topical import route_topical


def _make_handle(session_id: str, tmp_path: Path) -> CheckpointHandle:
    """Return a lightweight CheckpointHandle whose paths do not need to exist."""
    return CheckpointHandle(
        session_id=session_id,
        checkpoint_dir=tmp_path / session_id,
        torch_store_dir=tmp_path / session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=None,
    )


def _make_store(
    *, top_k_windows: list[int], scores_by_window: dict[int, float]
) -> Any:
    """Build a MagicMock store that returns the supplied routing behaviour."""
    store = MagicMock()
    store.route_top_k.return_value = list(top_k_windows)

    router = MagicMock()

    def _score_window(_query_ids: list[int], window_id: int) -> float:
        return float(scores_by_window.get(int(window_id), 0.0))

    router.score_window.side_effect = _score_window
    store._get_tfidf_router.return_value = router
    return store


def test_topical_returns_global_top_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_tokenizer: Any,
) -> None:
    """Two sessions: session_a scores 0.9, session_b scores 0.5 -> session_a wins."""
    session_a = "a" * 32
    session_b = "b" * 32
    handle_a = _make_handle(session_a, tmp_path)
    handle_b = _make_handle(session_b, tmp_path)

    store_a = _make_store(top_k_windows=[0], scores_by_window={0: 0.9})
    store_b = _make_store(top_k_windows=[0], scores_by_window={0: 0.5})

    mapping = {handle_a.session_id: store_a, handle_b.session_id: store_b}
    monkeypatch.setattr(
        topical_module, "load_store", lambda h: mapping[h.session_id]
    )

    result = route_topical(
        [handle_a, handle_b], "Alice Dubai", stub_tokenizer
    )
    assert result is not None
    handle, window_id, score = result
    assert handle is handle_a
    assert window_id == 0
    assert score == pytest.approx(0.9)


def test_topical_returns_none_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_tokenizer: Any,
) -> None:
    """No checkpoint returns any window -> None."""
    handle_a = _make_handle("a" * 32, tmp_path)
    handle_b = _make_handle("b" * 32, tmp_path)

    empty_store = _make_store(top_k_windows=[], scores_by_window={})
    monkeypatch.setattr(topical_module, "load_store", lambda _h: empty_store)

    assert (
        route_topical([handle_a, handle_b], "anything", stub_tokenizer) is None
    )


def test_topical_respects_nonpositive_k(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_tokenizer: Any,
) -> None:
    """``top_k_per_checkpoint <= 0`` short-circuits to None without loading."""
    handle_a = _make_handle("a" * 32, tmp_path)

    def _fail(_h: Any) -> Any:  # pragma: no cover - would signal a regression.
        raise AssertionError("load_store must not be invoked when k<=0")

    monkeypatch.setattr(topical_module, "load_store", _fail)
    assert route_topical(
        [handle_a], "alice", stub_tokenizer, top_k_per_checkpoint=0
    ) is None


def test_topical_ties_broken_deterministically_by_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_tokenizer: Any,
) -> None:
    """Equal scores -> lexicographically smallest session_id wins."""
    session_a = "a" * 32
    session_b = "b" * 32
    handle_a = _make_handle(session_a, tmp_path)
    handle_b = _make_handle(session_b, tmp_path)

    store_a = _make_store(top_k_windows=[3], scores_by_window={3: 0.7})
    store_b = _make_store(top_k_windows=[5], scores_by_window={5: 0.7})
    mapping = {handle_a.session_id: store_a, handle_b.session_id: store_b}
    monkeypatch.setattr(
        topical_module, "load_store", lambda h: mapping[h.session_id]
    )

    # Order input list reversed to prove tie-break does not rely on input order.
    result = route_topical([handle_b, handle_a], "tie", stub_tokenizer)
    assert result is not None
    handle, window_id, score = result
    assert handle is handle_a
    assert window_id == 3
    assert score == pytest.approx(0.7)


def test_topical_ties_broken_deterministically_by_window_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_tokenizer: Any,
) -> None:
    """Same score in same session -> smallest window_id wins."""
    session_a = "a" * 32
    handle_a = _make_handle(session_a, tmp_path)
    store = _make_store(
        top_k_windows=[9, 2], scores_by_window={9: 0.8, 2: 0.8}
    )
    monkeypatch.setattr(topical_module, "load_store", lambda _h: store)

    result = route_topical(
        [handle_a], "tie", stub_tokenizer, top_k_per_checkpoint=2
    )
    assert result is not None
    _handle, window_id, score = result
    assert window_id == 2
    assert score == pytest.approx(0.8)
