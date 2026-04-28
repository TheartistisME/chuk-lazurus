from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chuk_lazarus.session_retrieval import asi_router as asi_router_module
from chuk_lazarus.session_retrieval.asi_router import (
    ASI_ROUTER_STATE_FILENAME,
    AsiRouterCandidate,
    asi_route_candidates,
    record_asi_feedback,
)
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.selector_eval import run_selector_eval
from chuk_lazarus.session_retrieval.tier_policy import (
    POLICY_VERSION_UTILITY_V2,
    TierLabel,
    assign_tiers,
)


def _handle(session_id: str, tmp_path: Path) -> CheckpointHandle:
    return CheckpointHandle(
        session_id=session_id,
        checkpoint_dir=tmp_path / session_id,
        torch_store_dir=tmp_path / session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=None,
    )


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        table = {
            "automobile": 10,
            "repair": 11,
            "bill": 12,
            "garden": 20,
        }
        return [table.get(part.lower(), 99) for part in text.split()]


class _Store:
    def __init__(self, texts: dict[int, str], tokens: dict[int, set[int]]) -> None:
        self._texts = texts
        self.window_tokens = tokens
        self.window_token_lists = {wid: list(token_set) for wid, token_set in tokens.items()}
        self.idf = {token: 1.0 for token_set in tokens.values() for token in token_set}
        self.keywords = {wid: [] for wid in tokens}
        self.num_windows = len(tokens)

    def get_window_text(self, window_id: int, _tokenizer: Any) -> str:
        return self._texts[int(window_id)]


def _candidate(
    name: str,
    *,
    utility: float,
    cost: float = 1.0,
    vector: tuple[float, ...] = (),
    tmp_path: Path,
) -> AsiRouterCandidate:
    return AsiRouterCandidate(
        handle=_handle(f"{name:0<32}"[:32], tmp_path),
        window_id=0,
        ucb1_score=float(utility),
        raw_router_score=float(utility),
        island_id=0,
        visit_count=0,
        mean_reward=0.0,
        rrf_score=0.05,
        relevance_score=float(utility),
        estimated_cost=float(cost),
        utility_score=float(utility),
        dense_vector=tuple(vector),
        content_fingerprint=name,
    )


def test_hybrid_lexical_dense_fusion_with_provided_vectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    handle = _handle("dense000000000000000000000000001", tmp_path)
    store = _Store(
        texts={
            0: "car mechanic invoice approved",
            1: "garden irrigation reminder",
        },
        tokens={0: {30}, 1: {20}},
    )
    monkeypatch.setattr(asi_router_module, "load_store", lambda _h: store)

    candidates = asi_route_candidates(
        [handle],
        "automobile repair bill",
        _Tokenizer(),
        candidate_pool=2,
        dense_scoring="provided",
        dense_query_vector=(1.0, 0.0),
        dense_window_vectors={
            (handle.session_id, 0): (1.0, 0.0),
            (handle.session_id, 1): (0.0, 1.0),
        },
        ranking_policy=POLICY_VERSION_UTILITY_V2,
    )

    assert [candidate.window_id for candidate in candidates] == [0, 1]
    assert candidates[0].dense_rank == 1
    assert candidates[0].rrf_score > 0.0
    assert candidates[0].selector_telemetry["dense_status"] == "provided"


def test_mmr_duplicate_suppression(tmp_path: Path) -> None:
    duplicate_a = _candidate("duplicate-a", utility=1.00, vector=(1.0, 0.0), tmp_path=tmp_path)
    duplicate_b = _candidate("duplicate-b", utility=0.98, vector=(1.0, 0.0), tmp_path=tmp_path)
    diverse = _candidate("diverse", utility=0.90, vector=(0.0, 1.0), tmp_path=tmp_path)

    assignments = assign_tiers(
        [duplicate_a, duplicate_b, diverse],
        K_HOT=2,
        K_WARM=0,
        candidate_pool=3,
        policy_version=POLICY_VERSION_UTILITY_V2,
        budget=2,
        mmr_lambda=0.75,
    )
    active = [
        assignment.candidate for assignment in assignments if assignment.tier != TierLabel.COLD
    ]

    assert duplicate_a in active
    assert diverse in active
    assert duplicate_b not in active


def test_budget_aware_utility_selection(tmp_path: Path) -> None:
    expensive = _candidate("expensive", utility=1.00, cost=3, tmp_path=tmp_path)
    efficient_a = _candidate("efficient-a", utility=0.80, cost=1, tmp_path=tmp_path)
    efficient_b = _candidate("efficient-b", utility=0.70, cost=1, tmp_path=tmp_path)

    assignments = assign_tiers(
        [expensive, efficient_a, efficient_b],
        K_HOT=2,
        K_WARM=0,
        candidate_pool=3,
        policy_version=POLICY_VERSION_UTILITY_V2,
        budget=2,
    )
    active = [
        assignment.candidate for assignment in assignments if assignment.tier != TierLabel.COLD
    ]

    assert active == [efficient_a, efficient_b]
    assert sum(candidate.estimated_cost for candidate in active) <= 2


def test_dynamic_tier_thresholds(tmp_path: Path) -> None:
    high = _candidate("high", utility=1.00, tmp_path=tmp_path)
    medium = _candidate("medium", utility=0.40, tmp_path=tmp_path)
    low = _candidate("low", utility=0.05, tmp_path=tmp_path)

    assignments = assign_tiers(
        [high, medium, low],
        K_HOT=1,
        K_WARM=1,
        candidate_pool=3,
        policy_version=POLICY_VERSION_UTILITY_V2,
        budget=3,
    )
    by_name = {
        assignment.candidate.content_fingerprint: assignment.tier for assignment in assignments
    }

    assert by_name["high"] == TierLabel.HOT
    assert by_name["medium"] == TierLabel.WARM
    assert by_name["low"] == TierLabel.COLD


def test_persisted_asi_feedback(tmp_path: Path) -> None:
    candidate = _candidate("feedback", utility=1.0, tmp_path=tmp_path)
    assignments = assign_tiers(
        [candidate],
        policy_version=POLICY_VERSION_UTILITY_V2,
        budget=1,
    )

    record_asi_feedback(tmp_path, assignments, reward=0.75, outcome="kv_direct")
    record_asi_feedback(tmp_path, assignments, reward=0.25, outcome="kv_direct")

    payload = json.loads((tmp_path / ASI_ROUTER_STATE_FILENAME).read_text(encoding="utf-8"))
    key = f"{candidate.handle.session_id}:0"
    assert payload["visit_counts"][key] == 2
    assert payload["mean_rewards"][key] == pytest.approx(0.50)
    assert payload["islands"][0]["outcome"] == "kv_direct"


def test_selector_fixture_eval_passes_hard_criteria() -> None:
    report = run_selector_eval()

    assert report["pass"] is True
    assert report["delta"] >= 0.15
    assert report["new_metrics"]["recall_at_hot"] >= report["baseline_metrics"]["recall_at_hot"]
    assert (
        report["new_metrics"]["irrelevant_hot_rate"]
        <= report["baseline_metrics"]["irrelevant_hot_rate"]
    )
    assert report["new_metrics"]["budget_overflow_rate"] == 0.0
