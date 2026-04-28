"""Deterministic tier policy for ASI-router candidates.

Frozen contract: ``ve-ins-0mo9p8kou0000d20e0d`` (axis-3 of
``asi-kv-direct-chat`` run 1). Related upstream records: axis-2 closure
``ve-ins-0mo9vtofg00005ae032``; baseline-of-absence
``ve-ins-0mo9p63sh0000f78047``.

Policy name: ``rank-v1``. Given a candidate list already ranked descending
by ``ucb1_score`` (as produced by
:func:`chuk_lazarus.session_retrieval.asi_router.asi_route_candidates`), this
module assigns each candidate a :class:`TierLabel` (``HOT``/``WARM``/``COLD``)
purely as a function of its zero-based rank. Tier boundaries are controlled
by ``K_HOT`` and ``K_WARM`` with a hard cap of ``candidate_pool`` kept
candidates.

Upstream: ``asi_route_candidates`` — input is already ranked by
``ucb1_score`` descending; :func:`assign_tiers` does NOT re-sort.
Downstream consumers: axis-4 mute/compress/mask, axis-5 kv-direct-expansion.

JSON schema: version ``1`` (see :data:`TIER_POLICY_SCHEMA_VERSION`). All
serialized blobs are byte-deterministic; helpers raise ``ValueError`` on any
schema mismatch or mixed-policy envelope rather than falling back silently.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from chuk_lazarus.session_retrieval.asi_router import AsiRouterCandidate
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle

POLICY_VERSION_RANK_V1: str = "rank-v1"
POLICY_VERSION_UTILITY_V2: str = "utility-v2"
TIER_POLICY_SCHEMA_VERSION: int = 1
DEFAULT_MMR_LAMBDA: float = 0.75
DEFAULT_RRF_K: float = 60.0


class TierLabel(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass(frozen=True)
class TierAssignment:
    """One candidate's tier decision under a named policy.

    ``policy_params`` captures the exact hyperparameters (``K_HOT``,
    ``K_WARM``, ``candidate_pool``) the policy used, so downstream axes can
    reconstruct the decision without re-reading the router config.
    """

    candidate: AsiRouterCandidate
    tier: TierLabel
    rank: int
    policy_version: str
    policy_params: dict[str, int | float | str]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _candidate_cost(candidate: AsiRouterCandidate) -> float:
    return max(
        1.0,
        _as_float(
            getattr(candidate, "estimated_cost", getattr(candidate, "cost", 1.0)),
            1.0,
        ),
    )


def _candidate_similarity(a: AsiRouterCandidate, b: AsiRouterCandidate) -> float:
    vec_a = tuple(float(v) for v in (getattr(a, "dense_vector", ()) or ()))
    vec_b = tuple(float(v) for v in (getattr(b, "dense_vector", ()) or ()))
    if vec_a and vec_b:
        n = min(len(vec_a), len(vec_b))
        dot = sum(vec_a[i] * vec_b[i] for i in range(n))
        norm_a = math.sqrt(sum(v * v for v in vec_a[:n]))
        norm_b = math.sqrt(sum(v * v for v in vec_b[:n]))
        if norm_a > 0.0 and norm_b > 0.0:
            return max(0.0, min(1.0, dot / (norm_a * norm_b)))
    fp_a = str(getattr(a, "content_fingerprint", "") or "")
    fp_b = str(getattr(b, "content_fingerprint", "") or "")
    if fp_a and fp_a == fp_b:
        return 1.0
    if str(getattr(a.handle, "session_id", "")) == str(getattr(b.handle, "session_id", "")) and int(
        getattr(a, "window_id", -1)
    ) == int(getattr(b, "window_id", -2)):
        return 1.0
    return 0.0


def _utility_for_candidate(
    candidate: AsiRouterCandidate,
    *,
    max_cost: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    eta: float,
) -> float:
    rel = _as_float(
        getattr(candidate, "relevance_score", 0.0),
        _as_float(getattr(candidate, "raw_router_score", 0.0), 0.0),
    )
    if rel <= 0.0:
        rel = _as_float(getattr(candidate, "raw_router_score", 0.0), 0.0)
    rrf = _as_float(getattr(candidate, "rrf_score", 0.0), 0.0)
    fresh = _as_float(getattr(candidate, "freshness_score", 0.0), 0.0)
    reward = _as_float(
        getattr(candidate, "learned_reward", getattr(candidate, "mean_reward", 0.0)),
        0.0,
    )
    cost_norm = _candidate_cost(candidate) / max(1.0, float(max_cost))
    utility = (
        float(alpha) * rel
        + float(beta) * rrf
        + float(gamma) * fresh
        + float(delta) * reward
        - float(eta) * cost_norm
    )
    explicit = _as_float(getattr(candidate, "utility_score", 0.0), 0.0)
    if explicit > 0.0 and rel <= 0.0 and rrf <= 0.0:
        return explicit
    return utility


def _mmr_budget_order(
    candidates: Sequence[AsiRouterCandidate],
    *,
    utilities: dict[int, float],
    budget: float,
    mmr_lambda: float,
) -> list[AsiRouterCandidate]:
    selected: list[AsiRouterCandidate] = []
    remaining = list(candidates)
    spent = 0.0
    while remaining:
        feasible = [
            candidate
            for candidate in remaining
            if spent + _candidate_cost(candidate) <= float(budget)
        ]
        if not feasible:
            break
        best = max(
            feasible,
            key=lambda candidate: (
                float(mmr_lambda) * max(0.0, utilities[id(candidate)])
                - (1.0 - float(mmr_lambda))
                * max(
                    (_candidate_similarity(candidate, chosen) for chosen in selected),
                    default=0.0,
                ),
                utilities[id(candidate)] / _candidate_cost(candidate),
                utilities[id(candidate)],
                -_candidate_cost(candidate),
                str(candidate.handle.session_id),
                -int(candidate.window_id),
            ),
        )
        if utilities[id(best)] <= 0.0 and selected:
            break
        selected.append(best)
        spent += _candidate_cost(best)
        remaining.remove(best)
    return selected


def _assign_tiers_utility_v2(
    candidates: Sequence[AsiRouterCandidate],
    *,
    K_HOT: int,
    K_WARM: int,
    candidate_pool: int,
    policy_version: str,
    budget: int | float | None,
    mmr_lambda: float,
    rrf_k: float,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    eta: float,
    hot_utility_threshold: float,
    warm_utility_threshold: float,
) -> list[TierAssignment]:
    kept = list(candidates)[: int(candidate_pool)]
    if not kept:
        return []

    max_cost = max((_candidate_cost(candidate) for candidate in kept), default=1.0)
    utilities = {
        id(candidate): _utility_for_candidate(
            candidate,
            max_cost=max_cost,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
        )
        for candidate in kept
    }
    total_cost = sum(_candidate_cost(candidate) for candidate in kept)
    active_budget = float(total_cost if budget is None else max(0.0, float(budget)))
    active = _mmr_budget_order(
        kept,
        utilities=utilities,
        budget=active_budget,
        mmr_lambda=float(mmr_lambda),
    )
    active_ids = {id(candidate) for candidate in active}
    active_sorted = sorted(
        active,
        key=lambda candidate: (
            -utilities[id(candidate)],
            -_as_float(getattr(candidate, "rrf_score", 0.0), 0.0),
            str(candidate.handle.session_id),
            int(candidate.window_id),
        ),
    )
    inactive_sorted = sorted(
        [candidate for candidate in kept if id(candidate) not in active_ids],
        key=lambda candidate: (
            -utilities[id(candidate)],
            str(candidate.handle.session_id),
            int(candidate.window_id),
        ),
    )

    top_utility = max((utilities[id(candidate)] for candidate in active_sorted), default=0.0)
    hot_floor = max(float(hot_utility_threshold), top_utility * 0.60)
    warm_floor = max(float(warm_utility_threshold), top_utility * 0.20)
    tier_by_id: dict[int, TierLabel] = {}
    hot_count = 0
    warm_count = 0
    for candidate in active_sorted:
        utility = utilities[id(candidate)]
        if hot_count < int(K_HOT) and utility >= hot_floor and utility > 0.0:
            tier_by_id[id(candidate)] = TierLabel.HOT
            hot_count += 1
        elif warm_count < int(K_WARM) and utility >= warm_floor and utility > 0.0:
            tier_by_id[id(candidate)] = TierLabel.WARM
            warm_count += 1
        else:
            tier_by_id[id(candidate)] = TierLabel.COLD
    if (
        hot_count == 0
        and active_sorted
        and utilities[id(active_sorted[0])] >= float(hot_utility_threshold)
        and K_HOT > 0
    ):
        first = active_sorted[0]
        if tier_by_id.get(id(first)) == TierLabel.WARM:
            warm_count = max(0, warm_count - 1)
        tier_by_id[id(first)] = TierLabel.HOT
        hot_count = 1

    policy_params: dict[str, int | float | str] = {
        "K_HOT": int(K_HOT),
        "K_WARM": int(K_WARM),
        "candidate_pool": int(candidate_pool),
        "budget": float(active_budget),
        "mmr_lambda": float(mmr_lambda),
        "rrf_k": float(rrf_k),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "delta": float(delta),
        "eta": float(eta),
        "hot_utility_threshold": float(hot_utility_threshold),
        "warm_utility_threshold": float(warm_utility_threshold),
    }
    ordered = [
        *[candidate for candidate in active_sorted if tier_by_id[id(candidate)] == TierLabel.HOT],
        *[candidate for candidate in active_sorted if tier_by_id[id(candidate)] == TierLabel.WARM],
        *[candidate for candidate in active_sorted if tier_by_id[id(candidate)] == TierLabel.COLD],
        *inactive_sorted,
    ]
    assignments: list[TierAssignment] = []
    for rank, candidate in enumerate(ordered):
        tier = tier_by_id.get(id(candidate), TierLabel.COLD)
        assignments.append(
            TierAssignment(
                candidate=candidate,
                tier=tier,
                rank=int(rank),
                policy_version=str(policy_version),
                policy_params=dict(policy_params),
            )
        )
    return assignments


def assign_tiers(
    candidates: Sequence[AsiRouterCandidate],
    *,
    K_HOT: int = 4,
    K_WARM: int = 12,
    candidate_pool: int = 64,
    policy_version: str = POLICY_VERSION_RANK_V1,
    budget: int | float | None = None,
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    rrf_k: float = DEFAULT_RRF_K,
    utility_alpha: float = 1.0,
    utility_beta: float = 1.0,
    utility_gamma: float = 0.10,
    utility_delta: float = 0.50,
    utility_eta: float = 0.10,
    hot_utility_threshold: float = 0.35,
    warm_utility_threshold: float = 0.05,
) -> list[TierAssignment]:
    """Deterministic tier assignment from a ucb1-ranked candidate list.

    Invariants (verification obligations):

    - ``len(returned) == min(len(candidates), candidate_pool)``.
    - Tier counts are exactly ``(K_HOT, K_WARM, candidate_pool - K_HOT -
      K_WARM)`` when ``len(candidates) >= candidate_pool``; otherwise degrade
      gracefully (fewer COLD, then fewer WARM, then fewer HOT).
    - Assignment is stable: same input -> same output, byte-identical.
    """
    if K_HOT < 0:
        raise ValueError(f"K_HOT must be >= 0, got {K_HOT}")
    if K_WARM < 0:
        raise ValueError(f"K_WARM must be >= 0, got {K_WARM}")
    if candidate_pool <= 0:
        raise ValueError(f"candidate_pool must be >= 1, got {candidate_pool}")
    policy_name = str(policy_version).strip().lower().replace("_", "-")
    if policy_name in {POLICY_VERSION_UTILITY_V2, "hybrid-v2"}:
        return _assign_tiers_utility_v2(
            candidates,
            K_HOT=int(K_HOT),
            K_WARM=int(K_WARM),
            candidate_pool=int(candidate_pool),
            policy_version=POLICY_VERSION_UTILITY_V2,
            budget=budget,
            mmr_lambda=float(mmr_lambda),
            rrf_k=float(rrf_k),
            alpha=float(utility_alpha),
            beta=float(utility_beta),
            gamma=float(utility_gamma),
            delta=float(utility_delta),
            eta=float(utility_eta),
            hot_utility_threshold=float(hot_utility_threshold),
            warm_utility_threshold=float(warm_utility_threshold),
        )
    if policy_name != POLICY_VERSION_RANK_V1:
        raise ValueError(
            f"unknown tier policy_version={policy_version!r}; expected "
            f"{POLICY_VERSION_RANK_V1!r} or {POLICY_VERSION_UTILITY_V2!r}"
        )

    kept = list(candidates)[: int(candidate_pool)]
    policy_params: dict[str, int | float] = {
        "K_HOT": int(K_HOT),
        "K_WARM": int(K_WARM),
        "candidate_pool": int(candidate_pool),
    }
    warm_upper = int(K_HOT) + int(K_WARM)

    assignments: list[TierAssignment] = []
    for rank, candidate in enumerate(kept):
        if rank < int(K_HOT):
            tier = TierLabel.HOT
        elif rank < warm_upper:
            tier = TierLabel.WARM
        else:
            tier = TierLabel.COLD
        assignments.append(
            TierAssignment(
                candidate=candidate,
                tier=tier,
                rank=int(rank),
                policy_version=str(policy_version),
                policy_params=dict(policy_params),
            )
        )
    return assignments


def _handle_to_dict(handle: CheckpointHandle) -> dict[str, Any]:
    return {
        "session_id": str(handle.session_id),
        "checkpoint_dir": str(handle.checkpoint_dir),
        "torch_store_dir": str(handle.torch_store_dir),
        "manifest": handle.manifest,
        "original_input_dir": (
            None if handle.original_input_dir is None else str(handle.original_input_dir)
        ),
    }


def _handle_from_dict(data: dict[str, Any]) -> CheckpointHandle:
    for key in ("session_id", "checkpoint_dir", "torch_store_dir", "manifest"):
        if key not in data:
            raise ValueError(f"tier_policy: handle missing required key: {key!r}")
    original = data.get("original_input_dir")
    return CheckpointHandle(
        session_id=str(data["session_id"]),
        checkpoint_dir=Path(data["checkpoint_dir"]),
        torch_store_dir=Path(data["torch_store_dir"]),
        manifest=dict(data["manifest"]),
        original_input_dir=(None if original is None else Path(original)),
    )


def _candidate_to_dict(candidate: AsiRouterCandidate) -> dict[str, Any]:
    return {
        "window_id": int(candidate.window_id),
        "ucb1_score": float(candidate.ucb1_score),
        "raw_router_score": float(candidate.raw_router_score),
        "raw_tfidf_score_pre_normalization": float(candidate.raw_tfidf_score_pre_normalization),
        "island_id": int(candidate.island_id),
        "visit_count": int(candidate.visit_count),
        "mean_reward": float(candidate.mean_reward),
        "literal_score": float(getattr(candidate, "literal_score", 0.0)),
        "entity_score": float(getattr(candidate, "entity_score", 0.0)),
        "dense_score": float(getattr(candidate, "dense_score", 0.0)),
        "rrf_score": float(getattr(candidate, "rrf_score", 0.0)),
        "relevance_score": float(getattr(candidate, "relevance_score", 0.0)),
        "freshness_score": float(getattr(candidate, "freshness_score", 0.0)),
        "learned_reward": float(getattr(candidate, "learned_reward", 0.0)),
        "estimated_cost": float(getattr(candidate, "estimated_cost", 1.0)),
        "utility_score": float(getattr(candidate, "utility_score", 0.0)),
        "lexical_rank": int(getattr(candidate, "lexical_rank", 0)),
        "dense_rank": int(getattr(candidate, "dense_rank", 0)),
        "literal_rank": int(getattr(candidate, "literal_rank", 0)),
        "entity_rank": int(getattr(candidate, "entity_rank", 0)),
        "content_fingerprint": str(getattr(candidate, "content_fingerprint", "")),
        "selector_telemetry": dict(getattr(candidate, "selector_telemetry", {}) or {}),
        "handle": _handle_to_dict(candidate.handle),
    }


def _candidate_from_dict(data: dict[str, Any]) -> AsiRouterCandidate:
    for key in (
        "window_id",
        "ucb1_score",
        "raw_router_score",
        "island_id",
        "visit_count",
        "mean_reward",
        "handle",
    ):
        if key not in data:
            raise ValueError(f"tier_policy: candidate missing required key: {key!r}")
    return AsiRouterCandidate(
        handle=_handle_from_dict(data["handle"]),
        window_id=int(data["window_id"]),
        ucb1_score=float(data["ucb1_score"]),
        raw_router_score=float(data["raw_router_score"]),
        island_id=int(data["island_id"]),
        visit_count=int(data["visit_count"]),
        mean_reward=float(data["mean_reward"]),
        raw_tfidf_score_pre_normalization=float(data.get("raw_tfidf_score_pre_normalization", 0.0)),
        literal_score=float(data.get("literal_score", 0.0)),
        entity_score=float(data.get("entity_score", 0.0)),
        dense_score=float(data.get("dense_score", 0.0)),
        rrf_score=float(data.get("rrf_score", 0.0)),
        relevance_score=float(data.get("relevance_score", 0.0)),
        freshness_score=float(data.get("freshness_score", 0.0)),
        learned_reward=float(data.get("learned_reward", data.get("mean_reward", 0.0))),
        estimated_cost=float(data.get("estimated_cost", 1.0)),
        utility_score=float(data.get("utility_score", 0.0)),
        lexical_rank=int(data.get("lexical_rank", 0)),
        dense_rank=int(data.get("dense_rank", 0)),
        literal_rank=int(data.get("literal_rank", 0)),
        entity_rank=int(data.get("entity_rank", 0)),
        content_fingerprint=str(data.get("content_fingerprint", "")),
        selector_telemetry=dict(data.get("selector_telemetry", {}) or {}),
    )


def tier_assignment_to_dict(ta: TierAssignment) -> dict[str, Any]:
    """Serialize one :class:`TierAssignment` to a JSON-ready dict."""
    return {
        "tier": ta.tier.value,
        "rank": int(ta.rank),
        "policy_version": str(ta.policy_version),
        "policy_params": dict(ta.policy_params),
        "candidate": _candidate_to_dict(ta.candidate),
    }


def tier_assignment_from_dict(data: dict[str, Any]) -> TierAssignment:
    """Inverse of :func:`tier_assignment_to_dict`; raises ``ValueError`` on drift."""
    for key in ("tier", "rank", "policy_version", "policy_params", "candidate"):
        if key not in data:
            raise ValueError(f"tier_policy: assignment missing required key: {key!r}")
    tier_raw = data["tier"]
    try:
        tier = TierLabel(tier_raw)
    except ValueError as exc:
        raise ValueError(
            f"tier_policy: unknown tier label: {tier_raw!r}; "
            f"expected one of {[t.value for t in TierLabel]}"
        ) from exc
    params_raw = data["policy_params"]
    if not isinstance(params_raw, dict):
        raise ValueError(
            f"tier_policy: policy_params must be a dict, got {type(params_raw).__name__}"
        )
    return TierAssignment(
        candidate=_candidate_from_dict(data["candidate"]),
        tier=tier,
        rank=int(data["rank"]),
        policy_version=str(data["policy_version"]),
        policy_params=dict(params_raw),
    )


def tier_assignments_to_json(ts: Sequence[TierAssignment]) -> str:
    """Serialize a list of assignments to a byte-deterministic JSON string.

    Uses ``sort_keys=True`` and a compact separator so byte-identical inputs
    yield byte-identical outputs. The envelope's ``policy_version`` mirrors
    the first assignment's; an empty sequence uses :data:`POLICY_VERSION_RANK_V1`.
    """
    assignments = [tier_assignment_to_dict(ta) for ta in ts]
    envelope_policy = assignments[0]["policy_version"] if assignments else POLICY_VERSION_RANK_V1
    envelope: dict[str, Any] = {
        "schema_version": TIER_POLICY_SCHEMA_VERSION,
        "policy_version": envelope_policy,
        "assignments": assignments,
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


def tier_assignments_from_json(raw: str) -> list[TierAssignment]:
    """Inverse of :func:`tier_assignments_to_json`; no silent fallback."""
    envelope = json.loads(raw)
    if not isinstance(envelope, dict):
        raise ValueError(
            f"tier_policy: envelope must be a JSON object, got {type(envelope).__name__}"
        )
    for key in ("schema_version", "policy_version", "assignments"):
        if key not in envelope:
            raise ValueError(f"tier_policy: envelope missing required key: {key!r}")
    sv = envelope["schema_version"]
    if int(sv) != TIER_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"tier_policy: unsupported schema_version: {sv}; expected {TIER_POLICY_SCHEMA_VERSION}"
        )
    envelope_policy = str(envelope["policy_version"])
    raw_assignments = envelope["assignments"]
    if not isinstance(raw_assignments, list):
        raise ValueError(
            f"tier_policy: 'assignments' must be a list, got {type(raw_assignments).__name__}"
        )
    parsed = [tier_assignment_from_dict(entry) for entry in raw_assignments]
    if parsed and parsed[0].policy_version != envelope_policy:
        raise ValueError(
            f"tier_policy: envelope policy_version={envelope_policy!r} does not "
            f"match first assignment policy_version={parsed[0].policy_version!r}; "
            f"refusing to load mixed-policy blob"
        )
    return parsed


__all__ = [
    "POLICY_VERSION_UTILITY_V2",
    "POLICY_VERSION_RANK_V1",
    "TIER_POLICY_SCHEMA_VERSION",
    "TierAssignment",
    "TierLabel",
    "assign_tiers",
    "tier_assignment_from_dict",
    "tier_assignment_to_dict",
    "tier_assignments_from_json",
    "tier_assignments_to_json",
]
