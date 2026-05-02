"""Standalone centralized routing primitives.

This module is intentionally not wired into product code.  It provides a small,
stdlib-only router surface that other workers can import or copy into tests while
the production integration is still being designed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


TEMPORAL_ORDINAL = "temporal_ordinal"
SYMBOLIC_CHAIN = "symbolic_chain"
DEPENDENCY_SOURCE = "dependency_source"
PATCH_TARGET = "patch_target"
DURABLE_CHAT_MEMORY = "durable_chat_memory"
GENERAL_RECALL = "general_recall"

CAPABILITY_MODES = (
    TEMPORAL_ORDINAL,
    SYMBOLIC_CHAIN,
    DEPENDENCY_SOURCE,
    PATCH_TARGET,
    DURABLE_CHAT_MEMORY,
    GENERAL_RECALL,
)

HOT = "HOT"
WARM = "WARM"
COLD = "COLD"
TIER_ORDER = (HOT, WARM, COLD)


class RequestLike(Protocol):
    """Protocol-like request shape accepted by :class:`CentralRouter`."""

    query: str
    capability_mode: str


class WindowLike(Protocol):
    """Protocol-like window shape accepted by :class:`CentralRouter`."""

    window_id: str
    text: str


@dataclass
class RouteRequest:
    """Neutral request object for all router modes."""

    query: str
    capability_mode: str = GENERAL_RECALL
    ordinal: Optional[int] = None
    scope_key: Optional[str] = None
    canonical_request: Optional[str] = None
    path_hints: Tuple[str, ...] = ()
    identifiers: Tuple[str, ...] = ()
    entities: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteWindow:
    """Neutral routeable window object.

    Most fields are optional because callers may only have text and metadata.
    The router also accepts mappings or protocol-like objects and coerces them
    into this shape.
    """

    window_id: str
    text: str = ""
    source_path: Optional[str] = None
    session_turn_index: Optional[int] = None
    scope_key: Optional[str] = None
    canonical_request: Optional[str] = None
    memory_scope: Optional[str] = None
    source_authority: Optional[str] = None
    source_author: Optional[str] = None
    stale: bool = False
    superseded_by: Optional[str] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    tier: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteCandidate:
    """A scored window with mode-specific proof and trace data."""

    window: RouteWindow
    score: float
    mode: str
    reasons: Tuple[str, ...] = ()
    evidence: Tuple[str, ...] = ()
    route_trace: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0

    @property
    def window_id(self) -> str:
        return self.window.window_id


@dataclass
class TierAssignment:
    """Windows assigned to a materialization tier."""

    tier: str
    windows: Tuple[RouteWindow, ...]
    candidate_ids: Tuple[str, ...] = ()
    score_range: Tuple[float, float] = (0.0, 0.0)
    reason: str = ""


@dataclass
class MaterializationPlan:
    """Concrete plan for materializing routed windows."""

    windows: Tuple[RouteWindow, ...]
    tier_order: Tuple[str, ...] = TIER_ORDER
    tier_window_ids: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    per_tier_counts: Dict[str, int] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()

    def windows_for_tier(self, tier: str) -> Tuple[str, ...]:
        return self.tier_window_ids.get(tier, ())


@dataclass
class EvidenceSupport:
    """Proof-like support for why a window participates in a route."""

    window_id: str
    supports_claim: str
    confidence: float
    evidence: Tuple[str, ...] = ()
    mode: str = GENERAL_RECALL
    route_trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutePlan:
    """Complete router output consumed by downstream workers."""

    candidates: Tuple[RouteCandidate, ...]
    tier_assignments: Tuple[TierAssignment, ...]
    evidence_supports: Tuple[EvidenceSupport, ...]
    materialization_plan: MaterializationPlan
    router_metadata: Dict[str, Any] = field(default_factory=dict)
    selected_candidate: Optional[RouteCandidate] = None

    def tiers_present(self) -> Tuple[str, ...]:
        return tuple(
            assignment.tier
            for assignment in self.tier_assignments
            if assignment.windows
        )

    def assert_tier_coverage(self) -> "RoutePlan":
        """Assert the router's hard tier invariant and return ``self``.

        All routing modes must return HOT, WARM, and COLD windows whenever at
        least three eligible windows exist.
        """

        eligible_count = int(
            self.router_metadata.get("eligible_window_count", len(self.candidates))
        )
        if eligible_count >= 3:
            present = set(self.tiers_present())
            missing = [tier for tier in TIER_ORDER if tier not in present]
            if missing:
                raise AssertionError(
                    "missing required routing tiers for eligible windows: "
                    + ", ".join(missing)
                )

            materialized = self.materialization_plan.tier_window_ids
            missing_materialized = [
                tier for tier in TIER_ORDER if not materialized.get(tier)
            ]
            if missing_materialized:
                raise AssertionError(
                    "materialization plan omits tiers: "
                    + ", ".join(missing_materialized)
                )
        return self


@dataclass
class _Edge:
    left: str
    right: str
    window_id: str
    raw: str


class CentralRouter:
    """Deterministic standalone router for core capability modes."""

    def __init__(
        self,
        hot_count: int = 2,
        warm_count: int = 3,
        max_chain_depth: int = 8,
    ) -> None:
        self.hot_count = max(1, hot_count)
        self.warm_count = max(1, warm_count)
        self.max_chain_depth = max(1, max_chain_depth)

    def route(
        self,
        request: Any,
        windows: Sequence[Any],
    ) -> RoutePlan:
        route_request = self._coerce_request(request)
        route_windows = tuple(self._coerce_window(window, index) for index, window in enumerate(windows))
        mode = route_request.capability_mode or GENERAL_RECALL
        if mode not in CAPABILITY_MODES:
            mode = GENERAL_RECALL

        if mode == TEMPORAL_ORDINAL:
            candidates, supports, metadata = self._route_temporal_ordinal(
                route_request, route_windows
            )
        elif mode == SYMBOLIC_CHAIN:
            candidates, supports, metadata = self._route_symbolic_chain(
                route_request, route_windows
            )
        elif mode == DEPENDENCY_SOURCE:
            candidates, supports, metadata = self._route_dependency_source(
                route_request, route_windows
            )
        elif mode == PATCH_TARGET:
            candidates, supports, metadata = self._route_patch_target(
                route_request, route_windows
            )
        elif mode == DURABLE_CHAT_MEMORY:
            candidates, supports, metadata = self._route_durable_chat_memory(
                route_request, route_windows
            )
        else:
            candidates, supports, metadata = self._route_general_recall(
                route_request, route_windows
            )

        ranked = self._rank_candidates(candidates)
        assignments = self._assign_tiers(ranked)
        materialization_plan = self._build_materialization_plan(assignments)
        selected = ranked[0] if ranked else None

        metadata = dict(metadata)
        metadata.update(
            {
                "capability_mode": mode,
                "input_window_count": len(route_windows),
                "eligible_window_count": len(ranked),
                "selected_window_id": selected.window_id if selected else None,
                "tiers_present": tuple(
                    assignment.tier for assignment in assignments if assignment.windows
                ),
                "tier_invariant": (
                    "HOT_WARM_COLD_REQUIRED"
                    if len(ranked) >= 3
                    else "LESS_THAN_THREE_ELIGIBLE_WINDOWS"
                ),
            }
        )

        plan = RoutePlan(
            candidates=ranked,
            tier_assignments=assignments,
            evidence_supports=tuple(supports),
            materialization_plan=materialization_plan,
            router_metadata=metadata,
            selected_candidate=selected,
        )
        return plan.assert_tier_coverage()

    def _route_temporal_ordinal(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        scoped = [
            window
            for window in windows
            if self._matches_temporal_scope(request, window)
        ]
        pool = sorted(scoped, key=self._temporal_sort_key)
        ordinal = request.ordinal or _int_or_none(request.metadata.get("ordinal"))
        parsed_ordinal = self._parse_ordinal(request.query)
        if ordinal is None:
            ordinal = parsed_ordinal
        if ordinal is None:
            ordinal = 1

        if ordinal < 0:
            selected_index = len(pool) + ordinal
        else:
            selected_index = ordinal - 1

        selected_id = None
        if 0 <= selected_index < len(pool):
            selected_id = pool[selected_index].window_id

        candidates: List[RouteCandidate] = []
        supports: List[EvidenceSupport] = []
        for index, window in enumerate(pool):
            distance = abs(index - selected_index)
            selected = window.window_id == selected_id
            score = 1.0 if selected else max(0.05, 0.55 / (distance + 1))
            position = index + 1
            reason = (
                f"selected 1-indexed ordinal {ordinal}"
                if selected
                else f"temporal neighbor at position {position}"
            )
            evidence = (
                f"temporal position {position} of {len(pool)}",
                f"window_id={window.window_id}",
            )
            trace = {
                "ordinal": ordinal,
                "parsed_ordinal": parsed_ordinal,
                "position": position,
                "selected_position": selected_index + 1
                if 0 <= selected_index < len(pool)
                else None,
                "scope_key": request.scope_key,
                "canonical_request": request.canonical_request,
            }
            candidates.append(
                RouteCandidate(
                    window=window,
                    score=score,
                    mode=TEMPORAL_ORDINAL,
                    reasons=(reason,),
                    evidence=evidence,
                    route_trace=trace,
                )
            )
            supports.append(
                EvidenceSupport(
                    window_id=window.window_id,
                    supports_claim=reason,
                    confidence=score,
                    evidence=evidence,
                    mode=TEMPORAL_ORDINAL,
                    route_trace=trace,
                )
            )

        metadata = {
            "temporal_pool": tuple(window.window_id for window in pool),
            "temporal_pool_size": len(pool),
            "ordinal": ordinal,
            "selected_temporal_window_id": selected_id,
            "temporal_scope": {
                "scope_key": request.scope_key,
                "canonical_request": request.canonical_request,
            },
        }
        return candidates, supports, metadata

    def _route_symbolic_chain(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        edges = self._extract_edges(windows)
        graph: Dict[str, List[_Edge]] = {}
        for edge in edges:
            graph.setdefault(_symbol_key(edge.left), []).append(edge)

        starts = self._symbolic_starts(request, edges)
        chains: List[Dict[str, Any]] = []
        proved_window_ids = set()
        chain_edge_text: Dict[str, List[str]] = {}
        cycle_detected = False

        for start in starts:
            resolved = self._resolve_chain(start, graph)
            chains.append(resolved)
            cycle_detected = cycle_detected or bool(resolved["cycle_detected"])
            for edge in resolved["edges"]:
                proved_window_ids.add(edge.window_id)
                chain_edge_text.setdefault(edge.window_id, []).append(edge.raw)

        query_tokens = _tokens(request.query)
        candidates: List[RouteCandidate] = []
        supports: List[EvidenceSupport] = []
        for window in windows:
            window_edges = [edge for edge in edges if edge.window_id == window.window_id]
            edge_text = [edge.raw for edge in window_edges]
            chain_hits = chain_edge_text.get(window.window_id, [])
            lexical = len(query_tokens & _tokens(window.text))
            score = 0.05 + (0.12 * lexical)
            reasons: List[str] = []
            if window_edges:
                score += min(0.25, 0.08 * len(window_edges))
                reasons.append("contains symbolic assignment edges")
            if chain_hits:
                score += 0.65
                reasons.append("proves resolved symbolic chain edge")
            if any(_symbol_key(start) in {_symbol_key(edge.left), _symbol_key(edge.right)} for start in starts for edge in window_edges):
                score += 0.15
                reasons.append("mentions requested chain symbol")
            if not reasons:
                reasons.append("symbolic fallback lexical context")

            trace = {
                "starts": tuple(starts),
                "edges_in_window": tuple(edge_text),
                "chain_edges_in_window": tuple(chain_hits),
                "max_chain_depth": self.max_chain_depth,
                "cycle_detected": cycle_detected,
            }
            candidate = RouteCandidate(
                window=window,
                score=min(score, 1.0),
                mode=SYMBOLIC_CHAIN,
                reasons=tuple(reasons),
                evidence=tuple(chain_hits or edge_text[:3] or (window.window_id,)),
                route_trace=trace,
            )
            candidates.append(candidate)
            supports.append(
                EvidenceSupport(
                    window_id=window.window_id,
                    supports_claim="; ".join(reasons),
                    confidence=candidate.score,
                    evidence=candidate.evidence,
                    mode=SYMBOLIC_CHAIN,
                    route_trace=trace,
                )
            )

        metadata = {
            "symbolic_edge_count": len(edges),
            "symbolic_starts": tuple(starts),
            "symbolic_chains": tuple(
                {
                    "start": chain["start"],
                    "resolved_to": chain["resolved_to"],
                    "edge_count": len(chain["edges"]),
                    "cycle_detected": chain["cycle_detected"],
                    "stopped_by_depth": chain["stopped_by_depth"],
                }
                for chain in chains
            ),
            "cycle_detected": cycle_detected,
        }
        return candidates, supports, metadata

    def _route_dependency_source(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        candidates = []
        supports = []
        query_tokens = _tokens(request.query)
        path_hints = self._path_hints(request)
        identifiers = self._identifiers(request)

        for window in windows:
            score, reasons, evidence, trace = self._dependency_score_parts(
                request=request,
                window=window,
                query_tokens=query_tokens,
                path_hints=path_hints,
                identifiers=identifiers,
            )
            candidate = RouteCandidate(
                window=window,
                score=score,
                mode=DEPENDENCY_SOURCE,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                route_trace=trace,
            )
            candidates.append(candidate)
            supports.append(
                EvidenceSupport(
                    window_id=window.window_id,
                    supports_claim="; ".join(reasons) if reasons else "dependency route context",
                    confidence=score,
                    evidence=candidate.evidence,
                    mode=DEPENDENCY_SOURCE,
                    route_trace=trace,
                )
            )

        metadata = {
            "path_hints": tuple(path_hints),
            "identifiers": tuple(sorted(identifiers)),
            "route_trace_note": (
                "recursive_map and hop metadata are preserved as route traces, "
                "not asserted as static dependency truth"
            ),
        }
        return candidates, supports, metadata

    def _route_patch_target(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        dep_candidates, _supports, dep_metadata = self._route_dependency_source(request, windows)
        candidates: List[RouteCandidate] = []
        supports: List[EvidenceSupport] = []
        triad_roles: Dict[str, Tuple[str, ...]] = {
            "implementation_source": (),
            "tests": (),
            "docs": (),
            "assets": (),
            "padding": (),
        }
        role_buckets: Dict[str, List[str]] = {key: [] for key in triad_roles}

        for candidate in dep_candidates:
            path = self._window_path(candidate.window)
            role = self._patch_role(path, candidate.window)
            role_buckets.setdefault(role, []).append(candidate.window_id)
            score = candidate.score
            reasons = list(candidate.reasons)
            evidence = list(candidate.evidence)

            if role == "implementation_source":
                score += 0.35
                reasons.append("implementation source patch target")
            elif role == "tests":
                score -= 0.08
                reasons.append("test window kept as verification context")
            elif role == "docs":
                score -= 0.18
                reasons.append("documentation is lower priority than source")
            elif role == "assets":
                score -= 0.22
                reasons.append("asset is lower priority than source")
            elif role == "padding":
                score -= 0.25
                reasons.append("padding or generated context is lowest priority")
            else:
                reasons.append(f"{role} patch context")

            trace = dict(candidate.route_trace)
            trace.update(
                {
                    "patch_role": role,
                    "source_hints": dep_metadata["path_hints"],
                    "triad_role": role,
                }
            )
            patched = RouteCandidate(
                window=candidate.window,
                score=max(0.0, min(score, 1.2)),
                mode=PATCH_TARGET,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                route_trace=trace,
            )
            candidates.append(patched)

        source_scores = [
            candidate.score
            for candidate in candidates
            if candidate.route_trace.get("patch_role") == "implementation_source"
        ]
        if source_scores:
            source_floor = min(source_scores)
            for candidate in candidates:
                role = candidate.route_trace.get("patch_role")
                if role in {"tests", "docs", "assets", "padding"} and candidate.score >= source_floor:
                    candidate.score = max(0.0, source_floor - 0.001)
                    candidate.reasons = candidate.reasons + (
                        "capped below implementation source for patch target selection",
                    )

        for candidate in candidates:
            supports.append(
                EvidenceSupport(
                    window_id=candidate.window_id,
                    supports_claim="; ".join(candidate.reasons),
                    confidence=min(candidate.score, 1.0),
                    evidence=candidate.evidence,
                    mode=PATCH_TARGET,
                    route_trace=candidate.route_trace,
                )
            )

        ranked_tests = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.route_trace.get("patch_role") == "tests"
            ),
            key=lambda item: (-item.score, item.window_id),
        )
        selected_tests = tuple(candidate.window_id for candidate in ranked_tests[:5])

        metadata = dict(dep_metadata)
        metadata.update(
            {
                "source_hints": dep_metadata["path_hints"],
                "triad_roles": {
                    role: tuple(ids) for role, ids in role_buckets.items()
                },
                "selected_tests": selected_tests,
                "patch_target_policy": (
                    "implementation source receives priority over tests, docs, "
                    "assets, and padding"
                ),
            }
        )
        return candidates, supports, metadata

    def _route_durable_chat_memory(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        active: List[RouteWindow] = []
        stale_ids: List[str] = []
        superseded_ids: List[str] = []
        for window in windows:
            if self._window_bool(window, "stale") or window.stale:
                stale_ids.append(window.window_id)
                continue
            if self._window_value(window, "superseded_by", window.superseded_by):
                superseded_ids.append(window.window_id)
                continue
            active.append(window)

        query_tokens = _tokens(request.query)
        scope = _norm(request.scope_key or request.metadata.get("memory_scope") or "")
        user_memory: List[str] = []
        task_tool_memory: List[str] = []
        candidates: List[RouteCandidate] = []
        supports: List[EvidenceSupport] = []

        recency_values = [self._recency_value(window) for window in active]
        min_recency = min(recency_values) if recency_values else 0.0
        max_recency = max(recency_values) if recency_values else 0.0

        for window in active:
            memory_scope = _norm(
                window.memory_scope
                or self._window_value(window, "memory_scope", "")
                or self._window_value(window, "scope_key", "")
            )
            authority = _norm(
                window.source_authority
                or self._window_value(window, "source_authority", "")
                or window.source_author
                or self._window_value(window, "source_author", "")
            )
            memory_group = "user_memory" if authority in {"user", "human", "owner"} else "task_tool_memory"
            if memory_group == "user_memory":
                user_memory.append(window.window_id)
            else:
                task_tool_memory.append(window.window_id)

            lexical = len(query_tokens & _tokens(window.text))
            recency = self._recency_value(window)
            recency_score = _normalize(recency, min_recency, max_recency)
            score = 0.10 + min(0.25, 0.05 * lexical)
            reasons = []
            if scope and memory_scope == scope:
                score += 0.30
                reasons.append("memory scope match")
            elif scope and memory_scope:
                score += 0.05
                reasons.append("different memory scope retained as cold context")

            if memory_group == "user_memory":
                score += 0.25
                reasons.append("user memory source")
            elif authority in {"tool", "task", "assistant", "system"}:
                score += 0.10
                reasons.append("task or tool memory source")
            else:
                reasons.append("unspecified memory authority")

            if recency_score > 0.0:
                score += 0.20 * recency_score
                reasons.append("recent memory")
            if lexical:
                reasons.append("lexical overlap")

            trace = {
                "memory_scope": memory_scope,
                "source_authority": authority,
                "memory_group": memory_group,
                "recency_value": recency,
                "recency_score": recency_score,
                "stale": False,
                "superseded_by": None,
            }
            candidate = RouteCandidate(
                window=window,
                score=min(score, 1.0),
                mode=DURABLE_CHAT_MEMORY,
                reasons=tuple(reasons),
                evidence=(f"memory_group={memory_group}", f"memory_scope={memory_scope}"),
                route_trace=trace,
            )
            candidates.append(candidate)
            supports.append(
                EvidenceSupport(
                    window_id=window.window_id,
                    supports_claim="; ".join(reasons),
                    confidence=candidate.score,
                    evidence=candidate.evidence,
                    mode=DURABLE_CHAT_MEMORY,
                    route_trace=trace,
                )
            )

        metadata = {
            "filtered_stale_window_ids": tuple(stale_ids),
            "filtered_superseded_window_ids": tuple(superseded_ids),
            "user_memory_window_ids": tuple(user_memory),
            "task_tool_memory_window_ids": tuple(task_tool_memory),
            "memory_groups": {
                "user_memory": tuple(user_memory),
                "task_tool_memory": tuple(task_tool_memory),
            },
        }
        return candidates, supports, metadata

    def _route_general_recall(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> Tuple[List[RouteCandidate], List[EvidenceSupport], Dict[str, Any]]:
        query_tokens = _tokens(request.query)
        query_literal = _norm(request.query)
        entities = set(_norm(entity) for entity in request.entities if entity)
        entities.update(_extract_entities(request.query))
        candidates: List[RouteCandidate] = []
        supports: List[EvidenceSupport] = []

        for window in windows:
            text_norm = _norm(window.text)
            window_tokens = _tokens(window.text)
            window_entities = _extract_entities(window.text)
            lexical_hits = query_tokens & window_tokens
            entity_hits = entities & window_entities
            literal_hit = bool(query_literal and query_literal in text_norm)
            score = 0.05
            reasons: List[str] = []
            evidence: List[str] = []
            if literal_hit:
                score += 0.45
                reasons.append("literal query match")
                evidence.append("literal")
            if lexical_hits:
                score += min(0.35, 0.06 * len(lexical_hits))
                reasons.append("lexical overlap")
                evidence.extend(sorted(lexical_hits)[:5])
            if entity_hits:
                score += min(0.25, 0.08 * len(entity_hits))
                reasons.append("entity-style overlap")
                evidence.extend(sorted(entity_hits)[:5])
            if not reasons:
                reasons.append("general recall fallback context")
                evidence.append(window.window_id)

            trace = {
                "literal_hit": literal_hit,
                "lexical_hits": tuple(sorted(lexical_hits)),
                "entity_hits": tuple(sorted(entity_hits)),
            }
            candidate = RouteCandidate(
                window=window,
                score=min(score, 1.0),
                mode=GENERAL_RECALL,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                route_trace=trace,
            )
            candidates.append(candidate)
            supports.append(
                EvidenceSupport(
                    window_id=window.window_id,
                    supports_claim="; ".join(reasons),
                    confidence=candidate.score,
                    evidence=candidate.evidence,
                    mode=GENERAL_RECALL,
                    route_trace=trace,
                )
            )

        metadata = {
            "query_entities": tuple(sorted(entities)),
            "query_tokens": tuple(sorted(query_tokens)),
            "fallback": "hybrid lexical/literal/entity-style recall",
        }
        return candidates, supports, metadata

    def _dependency_score_parts(
        self,
        request: RouteRequest,
        window: RouteWindow,
        query_tokens: Iterable[str],
        path_hints: Sequence[str],
        identifiers: Iterable[str],
    ) -> Tuple[float, List[str], List[str], Dict[str, Any]]:
        path = self._window_path(window)
        path_norm = _norm_path(path)
        text_tokens = _tokens(window.text)
        path_tokens = _tokens(path_norm.replace("/", " "))
        query_token_set = set(query_tokens)
        identifier_set = set(_norm(identifier) for identifier in identifiers if identifier)
        path_matches = [
            hint
            for hint in path_hints
            if hint and (path_norm == _norm_path(hint) or path_norm.endswith(_norm_path(hint)))
        ]
        identifier_hits = identifier_set & (text_tokens | path_tokens)
        lexical_hits = query_token_set & (text_tokens | path_tokens)
        activation = self._activation_score(window)
        recursive_map = self._window_value(window, "recursive_map", None)
        hop = self._window_value(window, "hop", self._window_value(window, "dependency_hop", None))

        score = 0.05
        reasons: List[str] = []
        evidence: List[str] = []
        if path_matches:
            score += 0.45
            reasons.append("exact or suffix path hint match")
            evidence.extend(path_matches)
        if identifier_hits:
            score += min(0.30, 0.06 * len(identifier_hits))
            reasons.append("identifier overlap")
            evidence.extend(sorted(identifier_hits)[:5])
        if lexical_hits:
            score += min(0.20, 0.03 * len(lexical_hits))
            reasons.append("query/path/text overlap")
            evidence.extend(sorted(lexical_hits)[:5])
        if activation is not None:
            score += min(0.20, max(0.0, activation) * 0.20)
            reasons.append("activation metadata present")
            evidence.append(f"activation={activation:.3f}")
        if recursive_map is not None or hop is not None:
            reasons.append("recursive route trace metadata preserved")
        if not reasons:
            reasons.append("low-confidence dependency context")
            evidence.append(path or window.window_id)

        trace = {
            "path": path,
            "matched_path_hints": tuple(path_matches),
            "identifier_overlap": tuple(sorted(identifier_hits)),
            "lexical_overlap": tuple(sorted(lexical_hits)),
            "activation": activation,
            "recursive_map": recursive_map,
            "hop": hop,
            "recursive_map_is_route_trace_only": True,
        }
        return min(score, 1.0), reasons, evidence, trace

    def _assign_tiers(
        self,
        candidates: Tuple[RouteCandidate, ...],
    ) -> Tuple[TierAssignment, ...]:
        count = len(candidates)
        buckets: Dict[str, Tuple[RouteCandidate, ...]]
        if count == 0:
            buckets = {HOT: (), WARM: (), COLD: ()}
        elif count == 1:
            buckets = {HOT: candidates, WARM: (), COLD: ()}
        elif count == 2:
            buckets = {HOT: candidates[:1], WARM: candidates[1:], COLD: ()}
        else:
            hot_count = min(self.hot_count, count - 2)
            warm_count = min(self.warm_count, count - hot_count - 1)
            if warm_count < 1:
                warm_count = 1
            cold_start = hot_count + warm_count
            if cold_start >= count:
                warm_count = max(1, count - hot_count - 1)
                cold_start = hot_count + warm_count
            buckets = {
                HOT: candidates[:hot_count],
                WARM: candidates[hot_count:cold_start],
                COLD: candidates[cold_start:],
            }

        assignments: List[TierAssignment] = []
        for tier in TIER_ORDER:
            tier_candidates = buckets[tier]
            scores = [candidate.score for candidate in tier_candidates]
            assignments.append(
                TierAssignment(
                    tier=tier,
                    windows=tuple(candidate.window for candidate in tier_candidates),
                    candidate_ids=tuple(candidate.window_id for candidate in tier_candidates),
                    score_range=(
                        min(scores) if scores else 0.0,
                        max(scores) if scores else 0.0,
                    ),
                    reason=self._tier_reason(tier, len(tier_candidates), count),
                )
            )
        return tuple(assignments)

    def _build_materialization_plan(
        self,
        assignments: Tuple[TierAssignment, ...],
    ) -> MaterializationPlan:
        windows: List[RouteWindow] = []
        tier_window_ids: Dict[str, Tuple[str, ...]] = {}
        per_tier_counts: Dict[str, int] = {}
        for assignment in assignments:
            windows.extend(assignment.windows)
            tier_window_ids[assignment.tier] = assignment.candidate_ids
            per_tier_counts[assignment.tier] = len(assignment.windows)

        notes = (
            "materialization preserves HOT, WARM, and COLD tiers when at least three eligible windows exist",
        )
        return MaterializationPlan(
            windows=tuple(windows),
            tier_order=TIER_ORDER,
            tier_window_ids=tier_window_ids,
            per_tier_counts=per_tier_counts,
            notes=notes,
        )

    def _rank_candidates(
        self,
        candidates: Sequence[RouteCandidate],
    ) -> Tuple[RouteCandidate, ...]:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                self._temporal_sort_key(candidate.window),
                candidate.window_id,
            ),
        )
        output = []
        for index, candidate in enumerate(ranked, start=1):
            candidate.rank = index
            output.append(candidate)
        return tuple(output)

    def _coerce_request(self, request: Any) -> RouteRequest:
        if isinstance(request, RouteRequest):
            return request
        if isinstance(request, Mapping):
            data = dict(request)
            metadata = dict(data.get("metadata") or {})
            return RouteRequest(
                query=str(data.get("query", "")),
                capability_mode=str(data.get("capability_mode", GENERAL_RECALL)),
                ordinal=_int_or_none(data.get("ordinal")),
                scope_key=_optional_str(data.get("scope_key")),
                canonical_request=_optional_str(data.get("canonical_request")),
                path_hints=tuple(data.get("path_hints") or ()),
                identifiers=tuple(data.get("identifiers") or ()),
                entities=tuple(data.get("entities") or ()),
                metadata=metadata,
            )

        metadata = dict(getattr(request, "metadata", {}) or {})
        return RouteRequest(
            query=str(getattr(request, "query", "")),
            capability_mode=str(getattr(request, "capability_mode", GENERAL_RECALL)),
            ordinal=_int_or_none(getattr(request, "ordinal", None)),
            scope_key=_optional_str(getattr(request, "scope_key", None)),
            canonical_request=_optional_str(getattr(request, "canonical_request", None)),
            path_hints=tuple(getattr(request, "path_hints", ()) or ()),
            identifiers=tuple(getattr(request, "identifiers", ()) or ()),
            entities=tuple(getattr(request, "entities", ()) or ()),
            metadata=metadata,
        )

    def _coerce_window(self, window: Any, index: int) -> RouteWindow:
        if isinstance(window, RouteWindow):
            if not window.window_id:
                window.window_id = f"window-{index}"
            return window
        if isinstance(window, Mapping):
            data = dict(window)
            metadata = dict(data.get("metadata") or {})
            return RouteWindow(
                window_id=str(data.get("window_id") or data.get("id") or f"window-{index}"),
                text=str(data.get("text") or data.get("content") or ""),
                source_path=_optional_str(data.get("source_path") or data.get("path")),
                session_turn_index=_int_or_none(data.get("session_turn_index")),
                scope_key=_optional_str(data.get("scope_key")),
                canonical_request=_optional_str(data.get("canonical_request")),
                memory_scope=_optional_str(data.get("memory_scope")),
                source_authority=_optional_str(data.get("source_authority")),
                source_author=_optional_str(data.get("source_author")),
                stale=bool(data.get("stale", False)),
                superseded_by=_optional_str(data.get("superseded_by")),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                tier=_optional_str(data.get("tier")),
                metadata=metadata,
            )

        metadata = dict(getattr(window, "metadata", {}) or {})
        return RouteWindow(
            window_id=str(getattr(window, "window_id", getattr(window, "id", f"window-{index}"))),
            text=str(getattr(window, "text", getattr(window, "content", ""))),
            source_path=_optional_str(getattr(window, "source_path", getattr(window, "path", None))),
            session_turn_index=_int_or_none(getattr(window, "session_turn_index", None)),
            scope_key=_optional_str(getattr(window, "scope_key", None)),
            canonical_request=_optional_str(getattr(window, "canonical_request", None)),
            memory_scope=_optional_str(getattr(window, "memory_scope", None)),
            source_authority=_optional_str(getattr(window, "source_authority", None)),
            source_author=_optional_str(getattr(window, "source_author", None)),
            stale=bool(getattr(window, "stale", False)),
            superseded_by=_optional_str(getattr(window, "superseded_by", None)),
            created_at=getattr(window, "created_at", None),
            updated_at=getattr(window, "updated_at", None),
            tier=_optional_str(getattr(window, "tier", None)),
            metadata=metadata,
        )

    def _matches_temporal_scope(self, request: RouteRequest, window: RouteWindow) -> bool:
        request_scope = _norm(request.scope_key or "")
        request_canonical = _norm(request.canonical_request or "")
        window_scope = _norm(
            window.scope_key or self._window_value(window, "scope_key", "") or ""
        )
        window_canonical = _norm(
            window.canonical_request
            or self._window_value(window, "canonical_request", "")
            or ""
        )
        if request_scope and window_scope != request_scope:
            return False
        if request_canonical and window_canonical != request_canonical:
            return False
        return True

    def _temporal_sort_key(self, window: RouteWindow) -> Tuple[int, str]:
        turn = window.session_turn_index
        if turn is None:
            turn = _int_or_none(self._window_value(window, "session_turn_index", None))
        if turn is None:
            turn = 10**9
        return int(turn), window.window_id

    def _parse_ordinal(self, query: str) -> Optional[int]:
        text = _norm(query)
        for word, value in _ORDINAL_WORDS.items():
            if re.search(r"\b" + re.escape(word) + r"\b", text):
                return value
        match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\b", text)
        if match:
            return int(match.group(1))
        if re.search(r"\blast\b", text):
            return -1
        return None

    def _extract_edges(self, windows: Sequence[RouteWindow]) -> List[_Edge]:
        edges: List[_Edge] = []
        for window in windows:
            for match in _EDGE_RE.finditer(window.text):
                left = match.group("left")
                right = match.group("right")
                raw = match.group(0).strip()
                if left and right and _symbol_key(left) != _symbol_key(right):
                    edges.append(_Edge(left=left, right=right, window_id=window.window_id, raw=raw))
        return edges

    def _symbolic_starts(self, request: RouteRequest, edges: Sequence[_Edge]) -> Tuple[str, ...]:
        starts = [_clean_symbol(identifier) for identifier in request.identifiers if identifier]
        if not starts:
            query_symbols = [
                _clean_symbol(match.group(0))
                for match in _SYMBOL_RE.finditer(request.query)
                if match.group(0).lower() not in _STOPWORDS
            ]
            edge_lefts = {_symbol_key(edge.left): edge.left for edge in edges}
            starts = [edge_lefts[_symbol_key(symbol)] for symbol in query_symbols if _symbol_key(symbol) in edge_lefts]
        if not starts and edges:
            starts = [edges[0].left]

        seen = set()
        output = []
        for start in starts:
            key = _symbol_key(start)
            if key and key not in seen:
                output.append(start)
                seen.add(key)
        return tuple(output)

    def _resolve_chain(self, start: str, graph: Mapping[str, Sequence[_Edge]]) -> Dict[str, Any]:
        current = start
        visited = set()
        selected_edges: List[_Edge] = []
        cycle_detected = False
        stopped_by_depth = False

        for _depth in range(self.max_chain_depth):
            key = _symbol_key(current)
            if key in visited:
                cycle_detected = True
                break
            visited.add(key)
            outgoing = graph.get(key, ())
            if not outgoing:
                break
            edge = sorted(outgoing, key=lambda item: (item.window_id, item.raw))[0]
            selected_edges.append(edge)
            current = edge.right
        else:
            stopped_by_depth = True

        return {
            "start": start,
            "resolved_to": current,
            "edges": tuple(selected_edges),
            "cycle_detected": cycle_detected,
            "stopped_by_depth": stopped_by_depth,
        }

    def _path_hints(self, request: RouteRequest) -> Tuple[str, ...]:
        hints = [_norm_path(path) for path in request.path_hints if path]
        hints.extend(_norm_path(match.group(0)) for match in _PATH_HINT_RE.finditer(request.query))
        seen = set()
        output = []
        for hint in hints:
            if hint and hint not in seen:
                output.append(hint)
                seen.add(hint)
        return tuple(output)

    def _identifiers(self, request: RouteRequest) -> Tuple[str, ...]:
        identifiers = [_norm(identifier) for identifier in request.identifiers if identifier]
        identifiers.extend(
            token
            for token in _tokens(request.query)
            if len(token) > 2 and token not in _STOPWORDS
        )
        seen = set()
        output = []
        for identifier in identifiers:
            if identifier and identifier not in seen:
                output.append(identifier)
                seen.add(identifier)
        return tuple(output)

    def _patch_role(self, path: str, window: RouteWindow) -> str:
        explicit_role = _norm(self._window_value(window, "triad_role", self._window_value(window, "role", "")))
        if explicit_role:
            if "test" in explicit_role:
                return "tests"
            if "doc" in explicit_role:
                return "docs"
            if "asset" in explicit_role:
                return "assets"
            if "pad" in explicit_role or "generated" in explicit_role:
                return "padding"
            if "source" in explicit_role or "implementation" in explicit_role:
                return "implementation_source"

        path_norm = _norm_path(path)
        parts = set(path_norm.split("/"))
        file_name = path_norm.rsplit("/", 1)[-1]
        if not path_norm:
            return "implementation_source"
        if "test" in file_name or "tests" in parts or "__tests__" in parts:
            return "tests"
        if "docs" in parts or path_norm.endswith((".md", ".rst", ".txt")):
            return "docs"
        if path_norm.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".mp4", ".mp3", ".woff", ".woff2")):
            return "assets"
        if "padding" in parts or "generated" in parts or file_name.endswith((".map", ".lock")):
            return "padding"
        if path_norm.endswith((".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php")):
            return "implementation_source"
        if path_norm.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg")):
            return "config"
        return "implementation_source"

    def _window_path(self, window: RouteWindow) -> str:
        return str(
            window.source_path
            or self._window_value(window, "source_path", "")
            or self._window_value(window, "path", "")
            or ""
        )

    def _activation_score(self, window: RouteWindow) -> Optional[float]:
        for key in ("activation_score", "activation", "activation_strength"):
            value = self._window_value(window, key, None)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 1.0
        if self._window_value(window, "activated_by", None) is not None:
            return 1.0
        return None

    def _recency_value(self, window: RouteWindow) -> float:
        value = (
            window.updated_at
            or window.created_at
            or self._window_value(window, "updated_at", None)
            or self._window_value(window, "created_at", None)
            or self._window_value(window, "timestamp", None)
        )
        if value is None:
            turn = window.session_turn_index
            if turn is not None:
                return float(turn)
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            dt_value = value
            if dt_value.tzinfo is None:
                dt_value = dt_value.replace(tzinfo=timezone.utc)
            return dt_value.timestamp()
        text = str(value).strip()
        if not text:
            return 0.0
        try:
            return float(text)
        except ValueError:
            pass
        try:
            normalized = text.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized).timestamp()
        except ValueError:
            return 0.0

    def _window_value(self, window: RouteWindow, key: str, default: Any = None) -> Any:
        if key in window.metadata:
            return window.metadata[key]
        return default

    def _window_bool(self, window: RouteWindow, key: str) -> bool:
        value = self._window_value(window, key, False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _tier_reason(self, tier: str, tier_count: int, total_count: int) -> str:
        if total_count >= 3:
            return f"{tier} receives {tier_count} window(s) under full tier coverage"
        return f"{tier} receives {tier_count} window(s); fewer than three eligible windows"


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return str(value).strip().lower()


def _norm_path(value: Any) -> str:
    return re.sub(r"/+", "/", str(value).strip().replace("\\", "/").lower())


def _tokens(text: Any) -> set:
    return {
        token
        for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", str(text).lower())
        if token not in _STOPWORDS
    }


def _extract_entities(text: Any) -> set:
    entities = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_]*(?:\s+[A-Z][A-Za-z0-9_]*)*\b", str(text)):
        entity = _norm(match.group(0))
        if entity and entity not in _STOPWORDS:
            entities.add(entity)
    return entities


def _clean_symbol(value: str) -> str:
    return re.sub(r"^[^\w.]+|[^\w.]+$", "", value.strip())


def _symbol_key(value: str) -> str:
    return _clean_symbol(value).lower()


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*\b")
_EDGE_RE = re.compile(
    r"(?:\bVAR\s+)?(?P<left>[A-Za-z_][A-Za-z0-9_.]*)\s*(?:=|->)\s*(?P<right>[A-Za-z_][A-Za-z0-9_.]*)"
)
_PATH_HINT_RE = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+[/\\])+[A-Za-z0-9_.-]+\b|\b[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|hpp|md|rst|txt|json|yaml|yml|toml)\b"
)


__all__ = [
    "CAPABILITY_MODES",
    "TEMPORAL_ORDINAL",
    "SYMBOLIC_CHAIN",
    "DEPENDENCY_SOURCE",
    "PATCH_TARGET",
    "DURABLE_CHAT_MEMORY",
    "GENERAL_RECALL",
    "HOT",
    "WARM",
    "COLD",
    "RouteRequest",
    "RouteWindow",
    "RouteCandidate",
    "TierAssignment",
    "MaterializationPlan",
    "EvidenceSupport",
    "RoutePlan",
    "CentralRouter",
]
