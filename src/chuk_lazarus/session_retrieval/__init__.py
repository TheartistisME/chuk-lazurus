"""Axis-4 session_retrieval module.

Consumes axis-3 per-session checkpoints and answers a question by routing
across all valid clause-aligned stores, then running the Apollo-11-pattern
residual-injection pipeline.

This module NEVER modifies the underlying primitive at
``src/chuk_lazarus/inference/context/knowledge/``; it imports the already-public
``TorchKnowledgeStore``, ``torch_query._residual_is_compatible``,
``TorchInferenceRuntime``, ``LazarusBackend``, ``ResidualState``, and
``GenerationConfig`` only.

Public API
----------
- :class:`SessionRetriever` - unified retrieval entry point.
- :class:`QueryResult` - structured result with 6 strict-mode assertions.
- :class:`CheckpointHandle` - per-session checkpoint descriptor.
- :class:`AsiRouterCandidate` / :class:`AsiRouterState` - UCB1-ranked
  window candidates adapted from ASI-Evolve (see :mod:`.asi_router`).
- :func:`route_candidates` - full scored list variant of :func:`route_topical`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chuk_lazarus.session_retrieval.asi_router import (
    ASI_ROUTER_STATE_FILENAME,
    AsiRouterCandidate,
    AsiRouterState,
    advance_island,
    asi_route_candidates,
    assign_island,
    compute_ucb1,
    load_asi_router_state,
    record_asi_feedback,
    save_asi_router_state,
)
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle
from chuk_lazarus.session_retrieval.tier_policy import (
    POLICY_VERSION_RANK_V1,
    POLICY_VERSION_UTILITY_V2,
    TIER_POLICY_SCHEMA_VERSION,
    TierAssignment,
    TierLabel,
    assign_tiers,
    tier_assignment_from_dict,
    tier_assignment_to_dict,
    tier_assignments_from_json,
    tier_assignments_to_json,
)
from chuk_lazarus.session_retrieval.temporal_ordinal import (
    parse_ordinal_requirement,
    select_temporal_ordinal,
    sort_temporally,
)
from chuk_lazarus.session_retrieval.topical import route_candidates

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chuk_lazarus.session_retrieval.retriever import (  # noqa: F401
        QueryResult,
        SessionRetriever,
    )


def __getattr__(name: str):
    """Lazy re-export for ``retriever`` to avoid a circular import.

    ``retriever`` imports ``TorchInferenceRuntime`` from
    ``chuk_lazarus.inference.backends``, which in turn pulls
    ``tier_policy.TierLabel`` through ``torch_runtime``. Importing
    ``retriever`` at package load time would cycle through
    ``inference.backends`` before it has finished initialising. Resolving
    ``QueryResult`` / ``SessionRetriever`` on demand breaks the cycle.
    """

    if name in {"QueryResult", "SessionRetriever"}:
        from chuk_lazarus.session_retrieval.retriever import QueryResult, SessionRetriever

        globals().update({"QueryResult": QueryResult, "SessionRetriever": SessionRetriever})
        return globals()[name]
    raise AttributeError(f"module 'chuk_lazarus.session_retrieval' has no attribute {name!r}")

__all__ = [
    "ASI_ROUTER_STATE_FILENAME",
    "AsiRouterCandidate",
    "AsiRouterState",
    "CheckpointHandle",
    "POLICY_VERSION_RANK_V1",
    "POLICY_VERSION_UTILITY_V2",
    "QueryResult",
    "SessionRetriever",
    "TIER_POLICY_SCHEMA_VERSION",
    "TierAssignment",
    "TierLabel",
    "advance_island",
    "asi_route_candidates",
    "assign_island",
    "assign_tiers",
    "compute_ucb1",
    "load_asi_router_state",
    "parse_ordinal_requirement",
    "record_asi_feedback",
    "route_candidates",
    "save_asi_router_state",
    "select_temporal_ordinal",
    "sort_temporally",
    "tier_assignment_from_dict",
    "tier_assignment_to_dict",
    "tier_assignments_from_json",
    "tier_assignments_to_json",
]
