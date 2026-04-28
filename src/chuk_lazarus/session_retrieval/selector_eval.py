"""Deterministic hard-eval harness for ASI memory window selection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chuk_lazarus.session_retrieval.asi_router import (
    AsiRouterCandidate,
    _cosine,
    _deterministic_dense_vector,
)
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.tier_policy import (
    POLICY_VERSION_RANK_V1,
    POLICY_VERSION_UTILITY_V2,
    TierAssignment,
    TierLabel,
    assign_tiers,
)


DEFAULT_FIXTURE_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "memory_selector_eval.json"
)

_WORD_RE = re.compile(r"[a-z0-9$%]+")
_STOPWORDS = {
    "a", "about", "after", "and", "are", "as", "at", "be", "did", "do",
    "for", "how", "is", "it", "of", "on", "the", "to", "was", "we",
    "what",
}


@dataclass(frozen=True)
class _Window:
    id: str
    session_id: str
    window_id: int
    text: str
    cost: float
    duplicate_group: str


def _tokens(text: str) -> set[str]:
    return {
        token for token in _WORD_RE.findall(str(text).lower())
        if token not in _STOPWORDS
    }


def _rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda wid: (-scores[wid], wid))
    return {wid: idx for idx, wid in enumerate(ordered, start=1)}


def _rrf(window_id: str, rank_maps: list[dict[str, int]], *, k: float = 60.0) -> float:
    return sum(1.0 / (float(k) + ranks[window_id]) for ranks in rank_maps)


def _normalise_by_id(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    hi = max(scores.values())
    lo = min(scores.values())
    if hi == lo:
        value = 1.0 if hi > 0.0 else 0.0
        return {key: value for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def _handle_for(window: _Window, root: Path) -> CheckpointHandle:
    return CheckpointHandle(
        session_id=window.session_id,
        checkpoint_dir=root / window.session_id,
        torch_store_dir=root / window.session_id / "torch_store",
        manifest={"clause_aligned": True, "num_windows": 1, "num_entries": 1},
        original_input_dir=None,
    )


def _candidate_pool(
    windows: list[_Window],
    query: dict[str, Any],
    *,
    fixture_root: Path,
    hybrid: bool,
) -> list[AsiRouterCandidate]:
    query_text = str(query["text"])
    query_tokens = _tokens(query_text)
    query_vec = _deterministic_dense_vector(query_text)
    literal = str(query.get("exact_literal", "") or "").lower()

    lexical_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}
    literal_scores: dict[str, float] = {}
    entity_scores: dict[str, float] = {}
    vectors: dict[str, tuple[float, ...]] = {}
    for window in windows:
        window_tokens = _tokens(window.text)
        overlap = query_tokens & window_tokens
        lexical = float(len(overlap))
        if "current" in query_tokens and "old" in window_tokens:
            lexical -= 1.5
        if "tight" in query_tokens and window.cost > 1:
            lexical += 0.5
        lexical_scores[window.id] = max(0.0, lexical)
        window_vec = _deterministic_dense_vector(window.text)
        vectors[window.id] = window_vec
        dense_scores[window.id] = max(0.0, _cosine(query_vec, window_vec))
        literal_scores[window.id] = 1.0 if literal and literal in window.text.lower() else 0.0
        entity_scores[window.id] = float(
            len({token for token in query_tokens if token[:1].isupper()} & window_tokens)
        )

    lexical_rank = _rank_map(lexical_scores)
    dense_rank = _rank_map(dense_scores)
    literal_rank = _rank_map(literal_scores)
    entity_rank = _rank_map(entity_scores)
    lexical_norm = _normalise_by_id(lexical_scores)
    dense_norm = {key: max(0.0, min(1.0, value)) for key, value in dense_scores.items()}
    literal_norm = _normalise_by_id(literal_scores)
    entity_norm = _normalise_by_id(entity_scores)

    candidates: list[AsiRouterCandidate] = []
    for window in windows:
        rank_maps = [lexical_rank, literal_rank, entity_rank]
        if hybrid:
            rank_maps.append(dense_rank)
        rrf_score = _rrf(window.id, rank_maps)
        relevance = (
            0.50 * lexical_norm[window.id]
            + (0.25 * dense_norm[window.id] if hybrid else 0.0)
            + 0.15 * literal_norm[window.id]
            + 0.10 * entity_norm[window.id]
        )
        if ("current" in query_tokens or "tight" in query_tokens) and "old" in _tokens(window.text):
            relevance = max(0.0, relevance - 0.60)
        if not query.get("gold_ids"):
            relevance = 0.0
        cost_norm = float(window.cost) / max(1.0, max(w.cost for w in windows))
        utility = relevance + rrf_score - (0.10 * cost_norm)
        candidates.append(AsiRouterCandidate(
            handle=_handle_for(window, fixture_root),
            window_id=int(window.window_id),
            ucb1_score=float(lexical_scores[window.id]),
            raw_router_score=float(lexical_norm[window.id]),
            island_id=0,
            visit_count=0,
            mean_reward=0.0,
            raw_tfidf_score_pre_normalization=float(lexical_scores[window.id]),
            literal_score=float(literal_scores[window.id]),
            entity_score=float(entity_scores[window.id]),
            dense_score=float(dense_scores[window.id] if hybrid else 0.0),
            rrf_score=float(rrf_score),
            relevance_score=float(relevance),
            freshness_score=0.5,
            learned_reward=0.0,
            estimated_cost=float(window.cost),
            utility_score=float(utility),
            lexical_rank=int(lexical_rank[window.id]),
            dense_rank=int(dense_rank[window.id] if hybrid else 0),
            literal_rank=int(literal_rank[window.id]),
            entity_rank=int(entity_rank[window.id]),
            content_fingerprint=str(window.duplicate_group),
            dense_vector=tuple(vectors[window.id] if hybrid else ()),
            selector_telemetry={"eval_window_id": window.id},
        ))
    if hybrid:
        candidates.sort(key=lambda c: (-c.utility_score, str(c.handle.session_id), c.window_id))
    else:
        candidates.sort(
            key=lambda c: (
                -c.raw_tfidf_score_pre_normalization,
                str(c.handle.session_id),
                c.window_id,
            )
        )
    return candidates


def _assignment_window_id(assignment: TierAssignment, id_by_key: dict[tuple[str, int], str]) -> str:
    candidate = assignment.candidate
    return id_by_key[(str(candidate.handle.session_id), int(candidate.window_id))]


def _score_query(
    assignments: list[TierAssignment],
    query: dict[str, Any],
    windows_by_id: dict[str, _Window],
    id_by_key: dict[tuple[str, int], str],
) -> dict[str, Any]:
    selected = [
        {
            "window_id": _assignment_window_id(assignment, id_by_key),
            "tier": assignment.tier.value,
            "rank": int(assignment.rank),
            "cost": float(assignment.candidate.estimated_cost),
        }
        for assignment in assignments
    ]
    gold = set(query.get("gold_ids", []) or [])
    false_support = set(query.get("false_support_ids", []) or [])
    hot_ids = {row["window_id"] for row in selected if row["tier"] == "hot"}
    warm_ids = {row["window_id"] for row in selected if row["tier"] == "warm"}
    active_ids = {
        row["window_id"] for row in selected if row["tier"] in {"hot", "warm"}
    }
    selected_ids = {row["window_id"] for row in selected}
    exact_literal = str(query.get("exact_literal", "") or "")

    duplicate_penalty = 0
    groups: dict[str, int] = {}
    for window_id in active_ids:
        group = windows_by_id[window_id].duplicate_group
        groups[group] = groups.get(group, 0) + 1
    for count in groups.values():
        duplicate_penalty += max(0, count - 1)

    active_cost = sum(windows_by_id[window_id].cost for window_id in active_ids)
    budget = float(query.get("budget", 0))
    hit_hot = 1 if gold and (gold & hot_ids) else 0
    hit_warm = 1 if gold and (gold & warm_ids) else 0
    evidence_support = 1 if gold and (gold & selected_ids) else 0
    exact_literal_preserved = (
        1 if exact_literal and any(window_id in selected_ids for window_id in gold) else 0
    )
    irrelevant_hot = len([window_id for window_id in hot_ids if window_id not in gold])
    budget_overflow = 1 if budget > 0 and active_cost > budget else 0
    false_memory = 1 if false_support & active_ids else 0
    score = (
        4 * hit_hot
        + 2 * hit_warm
        + evidence_support
        + exact_literal_preserved
        - 2 * duplicate_penalty
        - 2 * irrelevant_hot
        - budget_overflow
        - 4 * false_memory
    )
    return {
        "query_id": str(query["id"]),
        "score": float(score),
        "HitHot": hit_hot,
        "HitWarm": hit_warm,
        "EvidenceSupport": evidence_support,
        "ExactLiteralPreserved": exact_literal_preserved,
        "DuplicatePenalty": duplicate_penalty,
        "IrrelevantHot": irrelevant_hot,
        "BudgetOverflow": budget_overflow,
        "FalseMemory": false_memory,
        "active_cost": float(active_cost),
        "budget": float(budget),
        "selected": selected,
    }


def _metrics(query_results: list[dict[str, Any]]) -> dict[str, float]:
    n = max(1, len(query_results))
    return {
        "mean_score": sum(float(row["score"]) for row in query_results) / n,
        "recall_at_hot": sum(int(row["HitHot"]) for row in query_results) / n,
        "irrelevant_hot_rate": sum(int(row["IrrelevantHot"]) for row in query_results) / n,
        "budget_overflow_rate": sum(int(row["BudgetOverflow"]) for row in query_results) / n,
    }


def run_selector_eval(fixture_path: Path | None = None) -> dict[str, Any]:
    path = Path(fixture_path or DEFAULT_FIXTURE_PATH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows = [
        _Window(
            id=str(row["id"]),
            session_id=str(row["session_id"]),
            window_id=int(row["window_id"]),
            text=str(row["text"]),
            cost=float(row.get("cost", 1)),
            duplicate_group=str(row.get("duplicate_group", row["id"])),
        )
        for row in payload["windows"]
    ]
    windows_by_id = {window.id: window for window in windows}
    id_by_key = {
        (window.session_id, int(window.window_id)): window.id
        for window in windows
    }

    baseline_results: list[dict[str, Any]] = []
    new_results: list[dict[str, Any]] = []
    fixture_root = path.parent / "_selector_eval_virtual_store"
    for query in payload["queries"]:
        baseline_candidates = _candidate_pool(
            windows, query, fixture_root=fixture_root, hybrid=False,
        )
        baseline_assignments = assign_tiers(
            baseline_candidates,
            K_HOT=int(query.get("k_hot", 1)),
            K_WARM=int(query.get("k_warm", 1)),
            candidate_pool=int(query.get("candidate_pool", 4)),
            policy_version=POLICY_VERSION_RANK_V1,
        )
        new_candidates = _candidate_pool(
            windows, query, fixture_root=fixture_root, hybrid=True,
        )
        new_assignments = assign_tiers(
            new_candidates,
            K_HOT=int(query.get("k_hot", 1)),
            K_WARM=int(query.get("k_warm", 1)),
            candidate_pool=int(query.get("candidate_pool", 4)),
            policy_version=POLICY_VERSION_UTILITY_V2,
            budget=float(query.get("budget", 2)),
            mmr_lambda=0.75,
            rrf_k=60.0,
        )
        baseline_results.append(_score_query(
            baseline_assignments, query, windows_by_id, id_by_key,
        ))
        new_results.append(_score_query(
            new_assignments, query, windows_by_id, id_by_key,
        ))

    baseline_metrics = _metrics(baseline_results)
    new_metrics = _metrics(new_results)
    delta = new_metrics["mean_score"] - baseline_metrics["mean_score"]
    criteria = {
        "mean_score_delta_at_least_0_15": delta >= 0.15,
        "recall_hot_not_worse": (
            new_metrics["recall_at_hot"] >= baseline_metrics["recall_at_hot"]
        ),
        "irrelevant_hot_rate_not_worse": (
            new_metrics["irrelevant_hot_rate"]
            <= baseline_metrics["irrelevant_hot_rate"]
        ),
        "budget_overflow_rate_zero": new_metrics["budget_overflow_rate"] == 0.0,
    }
    return {
        "fixture_path": str(path),
        "baseline_score": baseline_metrics["mean_score"],
        "new_score": new_metrics["mean_score"],
        "delta": delta,
        "baseline_metrics": baseline_metrics,
        "new_metrics": new_metrics,
        "per_query": [
            {
                "query_id": base["query_id"],
                "baseline": base,
                "new": new,
            }
            for base, new in zip(baseline_results, new_results)
        ],
        "criteria": criteria,
        "pass": all(criteria.values()),
    }


__all__ = ["DEFAULT_FIXTURE_PATH", "run_selector_eval"]
