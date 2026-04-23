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
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from chuk_lazarus.session_retrieval.asi_router import AsiRouterCandidate
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle

POLICY_VERSION_RANK_V1: str = "rank-v1"
TIER_POLICY_SCHEMA_VERSION: int = 1


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
    policy_params: dict[str, int | float]


def assign_tiers(
    candidates: Sequence[AsiRouterCandidate],
    *,
    K_HOT: int = 4,
    K_WARM: int = 12,
    candidate_pool: int = 64,
    policy_version: str = POLICY_VERSION_RANK_V1,
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
        assignments.append(TierAssignment(
            candidate=candidate,
            tier=tier,
            rank=int(rank),
            policy_version=str(policy_version),
            policy_params=dict(policy_params),
        ))
    return assignments


def _handle_to_dict(handle: CheckpointHandle) -> dict[str, Any]:
    return {
        "session_id": str(handle.session_id),
        "checkpoint_dir": str(handle.checkpoint_dir),
        "torch_store_dir": str(handle.torch_store_dir),
        "manifest": handle.manifest,
        "original_input_dir": (
            None if handle.original_input_dir is None
            else str(handle.original_input_dir)
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
        "island_id": int(candidate.island_id),
        "visit_count": int(candidate.visit_count),
        "mean_reward": float(candidate.mean_reward),
        "role": str(candidate.role),
        "turn_index": int(candidate.turn_index),
        "handle": _handle_to_dict(candidate.handle),
    }


def _candidate_from_dict(data: dict[str, Any]) -> AsiRouterCandidate:
    for key in (
        "window_id", "ucb1_score", "raw_router_score",
        "island_id", "visit_count", "mean_reward", "handle",
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
        role=str(data.get("role", "unknown")),
        turn_index=int(data.get("turn_index", -1)),
    )


def tier_assignment_to_dict(ta: TierAssignment) -> dict[str, Any]:
    """Serialize one :class:`TierAssignment` to a JSON-ready dict."""
    return {
        "tier": ta.tier.value,
        "rank": int(ta.rank),
        "policy_version": str(ta.policy_version),
        "policy_params": {str(k): int(v) for k, v in ta.policy_params.items()},
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
        policy_params={str(k): int(v) for k, v in params_raw.items()},
    )


def tier_assignments_to_json(ts: Sequence[TierAssignment]) -> str:
    """Serialize a list of assignments to a byte-deterministic JSON string.

    Uses ``sort_keys=True`` and a compact separator so byte-identical inputs
    yield byte-identical outputs. The envelope's ``policy_version`` mirrors
    the first assignment's; an empty sequence uses :data:`POLICY_VERSION_RANK_V1`.
    """
    assignments = [tier_assignment_to_dict(ta) for ta in ts]
    envelope_policy = (
        assignments[0]["policy_version"] if assignments else POLICY_VERSION_RANK_V1
    )
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
            f"tier_policy: unsupported schema_version: {sv}; "
            f"expected {TIER_POLICY_SCHEMA_VERSION}"
        )
    envelope_policy = str(envelope["policy_version"])
    raw_assignments = envelope["assignments"]
    if not isinstance(raw_assignments, list):
        raise ValueError(
            f"tier_policy: 'assignments' must be a list, got "
            f"{type(raw_assignments).__name__}"
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
