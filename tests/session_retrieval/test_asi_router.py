"""Tests for :mod:`chuk_lazarus.session_retrieval.asi_router`.

Covers:
- Numerical parity of :func:`compute_ucb1` (contract ve-ins-0mo9p7q1a000047eddd).
- Island rotation and migration bookkeeping in :func:`advance_island`.
- Deterministic feature-map island assignment via :func:`assign_island`.
- Round-trip persistence of :class:`AsiRouterState`.
- End-to-end behaviour of :func:`asi_route_candidates`.
- Adaptation-log parity between module docstring and docs/context/routing.
- Thin sibling :func:`route_candidates` in ``topical.py``.

All tests run on CPU; no network, no torch, no CUDA.
"""

from __future__ import annotations

import inspect
import json
import math
import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chuk_lazarus.session_retrieval import asi_router as asi_router_module
from chuk_lazarus.session_retrieval import topical as topical_module
from chuk_lazarus.session_retrieval.asi_router import (
    ASI_ROUTER_STATE_FILENAME,
    AsiRouterCandidate,
    AsiRouterState,
    advance_island,
    asi_route_candidates,
    assign_island,
    compute_ucb1,
    load_asi_router_state,
    save_asi_router_state,
)
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.topical import route_candidates, route_topical


# ---------------------------------------------------------------------------
# Local helpers / stubs (kept single-file per task brief).
# ---------------------------------------------------------------------------


def _make_handle(session_id: str, tmp_path: Path) -> CheckpointHandle:
    """Return a lightweight :class:`CheckpointHandle` — paths need not exist."""
    return CheckpointHandle(
        session_id=session_id,
        checkpoint_dir=tmp_path / session_id,
        torch_store_dir=tmp_path / session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=None,
    )


class _StubTokenizer:
    """Deterministic small-integer tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        # Stable token ids; not hashed so tests are reproducible.
        return [10, 11, 12]


def _make_stub_store(
    *,
    window_tokens: dict[int, set[int]],
    idf: dict[int, float],
    keywords: dict[int, list[str]] | None = None,
    num_windows: int | None = None,
) -> Any:
    store = MagicMock()
    store.window_tokens = window_tokens
    store.idf = idf
    store.keywords = keywords or {wid: [] for wid in window_tokens}
    store.num_windows = (
        num_windows if num_windows is not None else len(window_tokens)
    )
    return store


# ---------------------------------------------------------------------------
# 1. Numerical parity — compute_ucb1
# ---------------------------------------------------------------------------


class TestComputeUcb1:
    def test_ucb1_unvisited_returns_inf(self) -> None:
        assert compute_ucb1(q_w=0.5, n_w=0, total_visits=0, ucb1_c=1.414) == math.inf

    def test_ucb1_visited_parity_triple_a(self) -> None:
        # ln(1) = 0, so exploration term vanishes → score == q_w.
        score = compute_ucb1(q_w=0.5, n_w=1, total_visits=1, ucb1_c=1.414)
        assert score == pytest.approx(0.5, rel=1e-12)

    def test_ucb1_visited_parity_triple_b(self) -> None:
        expected = 0.7 + 1.414 * math.sqrt(math.log(10) / 2)
        score = compute_ucb1(q_w=0.7, n_w=2, total_visits=10, ucb1_c=1.414)
        assert score == pytest.approx(expected, rel=1e-12)

    def test_ucb1_visited_parity_triple_c(self) -> None:
        expected = 0.3 + 1.414 * math.sqrt(math.log(100) / 5)
        score = compute_ucb1(q_w=0.3, n_w=5, total_visits=100, ucb1_c=1.414)
        assert score == pytest.approx(expected, rel=1e-12)

    def test_ucb1_c_override(self) -> None:
        # Exploration term must scale linearly with ucb1_c.
        base = compute_ucb1(q_w=0.0, n_w=2, total_visits=10, ucb1_c=1.0)
        doubled = compute_ucb1(q_w=0.0, n_w=2, total_visits=10, ucb1_c=2.0)
        assert doubled == pytest.approx(2.0 * base, rel=1e-12)

    def test_ucb1_total_visits_clamped(self) -> None:
        # total_visits=0 must be clamped to max(1, 0)=1 → ln(1)=0 → score==q_w.
        score = compute_ucb1(q_w=0.42, n_w=3, total_visits=0, ucb1_c=1.414)
        assert score == pytest.approx(0.42, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Island rotation — advance_island
# ---------------------------------------------------------------------------


class TestAdvanceIsland:
    def test_advance_island_cycles_deterministically(self) -> None:
        state = AsiRouterState(
            current_island=0,
            total_selections=0,
            num_islands=5,
            migration_interval=10,
            migration_rate=0.1,
        )
        expected_cycle = [(i + 1) % 5 for i in range(25)]
        observed: list[int] = []
        starting_total = state.total_selections
        for call_idx in range(25):
            observed.append(advance_island(state))
            # Contract spec: total_selections must increment by 1 per call.
            assert state.total_selections == starting_total + (call_idx + 1), (
                f"total_selections did not increment on call {call_idx + 1}: "
                f"got {state.total_selections}, expected {starting_total + (call_idx + 1)}"
            )
        assert observed == expected_cycle

    def test_advance_island_migration_trigger(self) -> None:
        """With migration_interval=3 and migration_rate=0.5, calling
        advance_island 3 times should at least not raise and must leave
        ``feature_map`` a dict. Exact migration count is best-effort per the
        brief (document behavior)."""
        state = AsiRouterState(
            current_island=0,
            total_selections=0,
            num_islands=2,
            migration_interval=3,
            migration_rate=0.5,
            visit_counts={"s1:0": 1, "s1:1": 2, "s2:0": 3, "s2:1": 4},
            mean_rewards={"s1:0": 0.1, "s1:1": 0.2, "s2:0": 0.3, "s2:1": 0.4},
            feature_map={"s1:0": 0, "s1:1": 1, "s2:0": 0, "s2:1": 1},
        )
        for _ in range(3):
            advance_island(state)
        assert isinstance(state.feature_map, dict)

    def test_advance_island_zero_migration_rate(self) -> None:
        """migration_rate=0.0 must leave feature_map unchanged even when a
        migration trigger would otherwise fire."""
        original_fmap = {"s1:0": 0, "s1:1": 1, "s2:0": 0, "s2:1": 1}
        state = AsiRouterState(
            current_island=0,
            total_selections=3,  # divisible by migration_interval=3 immediately.
            num_islands=2,
            migration_interval=3,
            migration_rate=0.0,
            visit_counts={"s1:0": 1, "s1:1": 2, "s2:0": 3, "s2:1": 4},
            mean_rewards={"s1:0": 0.1, "s1:1": 0.2, "s2:0": 0.3, "s2:1": 0.4},
            feature_map=dict(original_fmap),
        )
        advance_island(state)
        assert state.feature_map == original_fmap


# ---------------------------------------------------------------------------
# 3. Feature-map / island assignment
# ---------------------------------------------------------------------------


class TestAssignIsland:
    def test_assign_island_deterministic(self) -> None:
        a = assign_island(
            "abc", 0, keyword_count=5, session_age_seconds=10.0, num_islands=5
        )
        b = assign_island(
            "abc", 0, keyword_count=5, session_age_seconds=10.0, num_islands=5
        )
        assert a == b

    def test_assign_island_range(self) -> None:
        rng = random.Random(1234)
        num_islands = 5
        for _ in range(100):
            session_id = f"sess_{rng.randint(0, 1_000_000):x}"
            window_id = rng.randint(0, 1_000)
            keyword_count = rng.randint(0, 50)
            session_age_seconds = rng.uniform(0.0, 1_000_000.0)
            island = assign_island(
                session_id,
                window_id,
                keyword_count=keyword_count,
                session_age_seconds=session_age_seconds,
                num_islands=num_islands,
            )
            assert 0 <= island < num_islands

    def test_assign_island_recency_bucket_monotonic(self) -> None:
        """For fixed keyword_count=0, island id should follow the documented
        formula ``floor(log2(1+age))`` clamped to [0, num_islands-1]."""
        num_islands = 5
        ages = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        for age in ages:
            expected_recency = max(
                0, min(int(math.floor(math.log2(1.0 + age))), num_islands - 1)
            )
            # keyword_count=0 → kw_bucket=0 → island = recency_bucket % num_islands
            expected_island = expected_recency % num_islands
            observed = assign_island(
                "session-x",
                0,
                keyword_count=0,
                session_age_seconds=age,
                num_islands=num_islands,
            )
            assert observed == expected_island, (
                f"age={age}: expected {expected_island}, got {observed}"
            )


# ---------------------------------------------------------------------------
# 4. State persistence round-trip
# ---------------------------------------------------------------------------


class TestStatePersistence:
    def test_state_filename_constant(self) -> None:
        assert ASI_ROUTER_STATE_FILENAME == "asi_router_state.json"

    def test_state_roundtrip(self, tmp_path: Path) -> None:
        state = AsiRouterState(
            current_island=3,
            total_selections=17,
            num_islands=5,
            migration_interval=10,
            migration_rate=0.1,
            visit_counts={"sess1:0": 2, "sess1:1": 5, "sess2:0": 1},
            mean_rewards={"sess1:0": 0.75, "sess1:1": 0.2, "sess2:0": 0.9},
            islands=[{"id": 0, "foo": "bar"}, {"id": 1}],
            feature_map={"sess1:0": 2, "sess1:1": 4, "sess2:0": 0},
        )
        written_path = save_asi_router_state(tmp_path, state)
        assert written_path == tmp_path / ASI_ROUTER_STATE_FILENAME
        reloaded = load_asi_router_state(tmp_path)
        assert reloaded.current_island == state.current_island
        assert reloaded.total_selections == state.total_selections
        assert reloaded.num_islands == state.num_islands
        assert reloaded.migration_interval == state.migration_interval
        assert reloaded.migration_rate == pytest.approx(state.migration_rate)
        assert reloaded.visit_counts == state.visit_counts
        assert reloaded.mean_rewards == state.mean_rewards
        assert reloaded.islands == state.islands
        assert reloaded.feature_map == state.feature_map
        assert reloaded.schema_version == state.schema_version

    def test_load_state_missing_file_returns_fresh(self, tmp_path: Path) -> None:
        # Fresh directory, no asi_router_state.json present.
        state = load_asi_router_state(tmp_path)
        assert state.current_island == 0
        assert state.total_selections == 0
        assert state.visit_counts == {}
        assert state.mean_rewards == {}
        assert state.islands == []
        assert state.feature_map == {}
        # Defaults from function signature.
        assert state.num_islands == 5
        assert state.migration_interval == 10
        assert state.migration_rate == pytest.approx(0.1)

    def test_load_state_bad_schema_raises(self, tmp_path: Path) -> None:
        bad_payload = {
            "schema_version": 999,
            "current_island": 0,
            "total_selections": 0,
            "num_islands": 5,
            "migration_interval": 10,
            "migration_rate": 0.1,
            "visit_counts": {},
            "mean_rewards": {},
            "islands": [],
            "feature_map": {},
        }
        (tmp_path / ASI_ROUTER_STATE_FILENAME).write_text(
            json.dumps(bad_payload), encoding="utf-8"
        )
        with pytest.raises(RuntimeError):
            load_asi_router_state(tmp_path)

    def test_save_state_atomic_replace(self, tmp_path: Path) -> None:
        state_a = AsiRouterState(
            current_island=1,
            total_selections=1,
            num_islands=5,
            migration_interval=10,
            migration_rate=0.1,
            visit_counts={"a:0": 1},
        )
        state_b = AsiRouterState(
            current_island=4,
            total_selections=9,
            num_islands=5,
            migration_interval=10,
            migration_rate=0.1,
            visit_counts={"b:0": 7},
            mean_rewards={"b:0": 0.5},
        )
        save_asi_router_state(tmp_path, state_a)
        save_asi_router_state(tmp_path, state_b)
        reloaded = load_asi_router_state(tmp_path)
        assert reloaded.current_island == state_b.current_island
        assert reloaded.total_selections == state_b.total_selections
        assert reloaded.visit_counts == state_b.visit_counts
        assert reloaded.mean_rewards == state_b.mean_rewards


# ---------------------------------------------------------------------------
# 5. asi_route_candidates end-to-end
# ---------------------------------------------------------------------------


class TestAsiRouteCandidates:
    def test_asi_route_candidates_returns_full_ranked_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle_a = _make_handle("a" * 32, tmp_path)
        handle_b = _make_handle("b" * 32, tmp_path)

        # Distinct token sets per window so scores differ.
        store_a = _make_stub_store(
            window_tokens={0: {10}, 1: {10, 11}, 2: {12}},
            idf={10: 1.0, 11: 1.0, 12: 1.0},
        )
        store_b = _make_stub_store(
            window_tokens={0: {10, 11, 12}, 1: {11}, 2: {13}},
            idf={10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0},
        )

        def _fake_load_store(handle: CheckpointHandle) -> Any:
            return store_a if handle.session_id == handle_a.session_id else store_b

        monkeypatch.setattr(asi_router_module, "load_store", _fake_load_store)

        candidates = asi_route_candidates(
            [handle_a, handle_b],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            candidate_pool=6,
            archive_root=None,
        )
        assert isinstance(candidates, list)
        assert len(candidates) <= 6
        for c in candidates:
            assert isinstance(c, AsiRouterCandidate)
        # Primary key: ucb1_score descending.
        scores = [c.ucb1_score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_asi_route_candidates_degenerate_reward_signal_uses_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When raw min == max, normalisation degenerates; all raw_router_score
        outputs must be 1.0 and deterministic, no NaN/inf leaking into q_w."""
        handle = _make_handle("a" * 32, tmp_path)
        # Identical token sets → identical raw scores → min == max.
        store = _make_stub_store(
            window_tokens={0: {10}, 1: {10}, 2: {10}},
            idf={10: 1.0},
        )
        monkeypatch.setattr(
            asi_router_module, "load_store", lambda _h: store
        )

        candidates = asi_route_candidates(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            candidate_pool=16,
            archive_root=None,
        )
        assert candidates, "expected candidates even when rewards are degenerate"
        for c in candidates:
            assert c.raw_router_score == pytest.approx(1.0)
            assert not math.isnan(c.raw_router_score)
            assert not math.isinf(c.raw_router_score)
        # Determinism: calling twice returns identical window_id ordering.
        second = asi_route_candidates(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            candidate_pool=16,
            archive_root=None,
        )
        assert [c.window_id for c in candidates] == [c.window_id for c in second]

    def test_asi_route_candidates_exploration_disabled_when_all_n_w_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle_a = _make_handle("a" * 32, tmp_path)
        handle_b = _make_handle("b" * 32, tmp_path)
        store_a = _make_stub_store(
            window_tokens={0: {10}, 1: {11}},
            idf={10: 1.0, 11: 1.0},
        )
        store_b = _make_stub_store(
            window_tokens={0: {10}, 1: {11}},
            idf={10: 1.0, 11: 1.0},
        )

        def _fake_load_store(handle: CheckpointHandle) -> Any:
            return store_a if handle.session_id == handle_a.session_id else store_b

        monkeypatch.setattr(asi_router_module, "load_store", _fake_load_store)

        candidates = asi_route_candidates(
            [handle_a, handle_b],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            archive_root=None,
        )
        assert candidates, "expected at least one candidate"
        # All n_w are zero → all ucb1 scores must be +inf.
        for c in candidates:
            assert c.ucb1_score == math.inf
            assert c.visit_count == 0
            assert c.mean_reward == 0.0
        # Deterministic tie-break by (session_id, window_id) ascending.
        key_seq = [(c.handle.session_id, c.window_id) for c in candidates]
        assert key_seq == sorted(key_seq)

    def test_asi_route_candidates_uses_prior_state(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        store = _make_stub_store(
            window_tokens={0: {10}, 1: {11}, 2: {12}},
            idf={10: 1.0, 11: 1.0, 12: 1.0},
        )
        monkeypatch.setattr(asi_router_module, "load_store", lambda _h: store)

        archive_root = tmp_path / "archive"
        archive_root.mkdir()

        pinned_key = f"{handle.session_id}:1"
        prior_state = AsiRouterState(
            current_island=0,
            total_selections=3,
            num_islands=5,
            migration_interval=10,
            migration_rate=0.1,
            visit_counts={pinned_key: 2},
            mean_rewards={pinned_key: 0.85},
        )
        save_asi_router_state(archive_root, prior_state)

        candidates = asi_route_candidates(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            archive_root=archive_root,
        )

        # Locate the pinned candidate.
        pinned = [c for c in candidates if c.window_id == 1]
        assert len(pinned) == 1, "expected exactly one candidate for window_id=1"
        pinned_c = pinned[0]
        assert math.isfinite(pinned_c.ucb1_score), (
            f"expected finite ucb1 for pinned candidate, got {pinned_c.ucb1_score}"
        )
        assert pinned_c.visit_count > 0
        assert pinned_c.mean_reward > 0.0

        # Contract says state is NOT saved back; caller owns persistence.
        # Verify the file contents on disk still match what we wrote.
        on_disk = json.loads(
            (archive_root / ASI_ROUTER_STATE_FILENAME).read_text(encoding="utf-8")
        )
        assert on_disk["visit_counts"] == {pinned_key: 2}
        assert on_disk["mean_rewards"] == {pinned_key: pytest.approx(0.85)}

    def test_asi_route_candidates_empty_handles(self) -> None:
        result = asi_route_candidates(
            [],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            archive_root=None,
        )
        assert result == []

    def test_asi_route_candidates_candidate_pool_truncates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        # 10 windows total.
        store = _make_stub_store(
            window_tokens={i: {10 + i} for i in range(10)},
            idf={10 + i: 1.0 for i in range(10)},
        )
        monkeypatch.setattr(asi_router_module, "load_store", lambda _h: store)
        candidates = asi_route_candidates(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            candidate_pool=3,
            archive_root=None,
        )
        assert len(candidates) <= 3

    def test_asi_route_candidates_no_silent_fallback_on_bad_store(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        # Object missing window_tokens/idf entirely.
        bad_store = object()
        monkeypatch.setattr(asi_router_module, "load_store", lambda _h: bad_store)
        with pytest.raises(RuntimeError):
            asi_route_candidates(
                [handle],
                query_text="hello",
                tokenizer=_StubTokenizer(),
                archive_root=None,
            )

    def test_asi_route_candidates_defaults_match_contract(self) -> None:
        sig = inspect.signature(asi_route_candidates)
        defaults = {
            name: p.default
            for name, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        assert defaults["ucb1_c"] == pytest.approx(1.414)
        assert defaults["num_islands"] == 5
        assert defaults["migration_interval"] == 10
        assert defaults["migration_rate"] == pytest.approx(0.1)
        assert defaults["exploration_ratio"] == pytest.approx(0.2)
        assert defaults["exploitation_ratio"] == pytest.approx(0.3)
        assert defaults["candidate_pool"] == 64


# ---------------------------------------------------------------------------
# 6. Adaptation-log parity
# ---------------------------------------------------------------------------


_ADAPTATION_SUBSTRINGS = [
    "(i) Reward signal adaptation",
    "(ii) Visit-count adaptation",
    "(iii) Island assignment adaptation",
    "(iv) Multi-candidate emission adaptation",
]


class TestAdaptationLogParity:
    def test_adaptation_log_present_in_module_docstring(self) -> None:
        asi_router_path = Path(asi_router_module.__file__)
        source = asi_router_path.read_text(encoding="utf-8")
        for needle in _ADAPTATION_SUBSTRINGS:
            assert needle in source, f"missing adaptation string in module: {needle!r}"

    def test_adaptation_log_parity_with_docs(self) -> None:
        repo_root = Path(asi_router_module.__file__).resolve().parents[3]
        docs_path = repo_root / "docs" / "context" / "routing" / "asi_router.md"
        assert docs_path.is_file(), f"docs file missing: {docs_path}"
        contents = docs_path.read_text(encoding="utf-8")
        for needle in _ADAPTATION_SUBSTRINGS:
            assert needle in contents, (
                f"missing adaptation string in docs: {needle!r}"
            )


# ---------------------------------------------------------------------------
# 7. route_candidates (topical.py sibling)
# ---------------------------------------------------------------------------


def _make_topical_store(
    *, window_tokens: dict[int, set[int]], idf: dict[int, float]
) -> Any:
    """Stub store exposing the surface topical.route_candidates uses."""
    store = MagicMock()
    store.window_tokens = window_tokens

    router = MagicMock()

    def _score_window(query_ids: list[int], window_id: int) -> float:
        tokens = window_tokens.get(int(window_id))
        if not tokens:
            return 0.0
        overlap = set(query_ids) & tokens
        return float(sum(idf.get(t, 0.0) for t in overlap))

    router.score_window.side_effect = _score_window
    store._get_tfidf_router.return_value = router

    # Provide a route_top_k that mirrors score-descending order for completeness.
    def _route_top_k(_query_text: str, _tokenizer: Any, k: int) -> list[int]:
        ordered = sorted(
            window_tokens.keys(),
            key=lambda wid: (-_score_window([10, 11, 12], wid), wid),
        )
        return ordered[: int(k)]

    store.route_top_k.side_effect = _route_top_k
    return store


class TestRouteCandidates:
    def test_route_candidates_returns_list_of_tuples(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        store = _make_topical_store(
            window_tokens={0: {10}, 1: {10, 11}, 2: {11, 12}},
            idf={10: 1.0, 11: 1.0, 12: 1.0},
        )
        monkeypatch.setattr(topical_module, "load_store", lambda _h: store)
        result = route_candidates(
            [handle], query_text="hello", tokenizer=_StubTokenizer()
        )
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)
        scores = [score for _h, _wid, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_route_candidates_respects_max_candidates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        store = _make_topical_store(
            window_tokens={i: {10 + i} for i in range(6)},
            idf={10 + i: 1.0 for i in range(6)},
        )
        monkeypatch.setattr(topical_module, "load_store", lambda _h: store)
        result = route_candidates(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            max_candidates=2,
        )
        assert len(result) <= 2

    def test_route_candidates_does_not_change_route_topical_behavior(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        handle = _make_handle("a" * 32, tmp_path)
        store = _make_topical_store(
            window_tokens={0: {10}, 1: {10, 11}, 2: {11}},
            idf={10: 1.0, 11: 1.0},
        )
        monkeypatch.setattr(topical_module, "load_store", lambda _h: store)
        topical_result = route_topical(
            [handle],
            query_text="hello",
            tokenizer=_StubTokenizer(),
            top_k_per_checkpoint=3,
        )
        candidates = route_candidates(
            [handle], query_text="hello", tokenizer=_StubTokenizer()
        )
        assert topical_result is not None
        top_handle, top_window, _top_score = topical_result
        assert candidates, "expected non-empty candidate list"
        first_handle, first_window, _first_score = candidates[0]
        assert first_handle.session_id == top_handle.session_id
        assert first_window == top_window
