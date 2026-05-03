"""Standalone centralized routing primitives.

This module is intentionally not wired into product code.  It provides a small,
stdlib-only router surface that other workers can import or copy into tests while
the production integration is still being designed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
class ModelAdapterMetadata:
    """Adapter identity that must match across routed request/window indexes."""

    model_family: Optional[str] = None
    model_revision: Optional[str] = None
    model_hash: Optional[str] = None
    tokenizer_id: Optional[str] = None
    tokenizer_hash: Optional[str] = None
    adapter_version: Optional[str] = None
    route_layer: Optional[str] = None
    boundary_layer: Optional[str] = None
    injection_layer: Optional[str] = None
    route_dimension: Optional[int] = None
    kv_layout: Optional[str] = None

    def identity(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "model_family": self.model_family,
                "model_revision": self.model_revision,
                "model_hash": self.model_hash,
                "tokenizer_id": self.tokenizer_id,
                "tokenizer_hash": self.tokenizer_hash,
                "adapter_version": self.adapter_version,
                "route_layer": self.route_layer,
                "boundary_layer": self.boundary_layer,
                "injection_layer": self.injection_layer,
                "route_dimension": self.route_dimension,
                "kv_layout": self.kv_layout,
            }.items()
            if value not in (None, "")
        }

    def is_empty(self) -> bool:
        return not self.identity()


@dataclass
class IndexReadinessMetadata:
    """Index state for a given adapter identity."""

    index_id: Optional[str] = None
    adapter_key: Optional[str] = None
    status: Optional[str] = None
    built_at: Optional[Any] = None
    dimension: Optional[int] = None
    model_family: Optional[str] = None
    model_revision: Optional[str] = None
    tokenizer_hash: Optional[str] = None
    tokenizer_id: Optional[str] = None
    model_hash: Optional[str] = None
    adapter_version: Optional[str] = None
    route_layer: Optional[str] = None
    boundary_layer: Optional[str] = None
    injection_layer: Optional[str] = None
    route_dimension: Optional[int] = None
    kv_layout: Optional[str] = None


@dataclass
class ApolloResidualReadinessMetadata:
    """Residual/KV materialization state independent from fast route indexes."""

    status: Optional[str] = None
    readiness_source: Optional[str] = None
    manifest_path: Optional[str] = None
    manifest_id: Optional[str] = None
    torch_store_path: Optional[str] = None
    window_count: Optional[int] = None
    boundary_residual_count: Optional[int] = None
    residual_stream_count: Optional[int] = None
    boundary_residual_dir: Optional[str] = None
    residual_stream_dir: Optional[str] = None
    has_residual_streams: bool = False
    kv_direct_ready: bool = False
    pending_reason: Optional[str] = None
    source_layer: Optional[str] = None
    target_layer: Optional[str] = None
    chain_id: Optional[str] = None
    parent_manifest_ids: Tuple[str, ...] = ()
    source_window_refs: Tuple[str, ...] = ()

    def identity(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in {
                "status": self.status,
                "readiness_source": self.readiness_source,
                "manifest_path": self.manifest_path,
                "manifest_id": self.manifest_id,
                "torch_store_path": self.torch_store_path,
                "window_count": self.window_count,
                "boundary_residual_count": self.boundary_residual_count,
                "residual_stream_count": self.residual_stream_count,
                "boundary_residual_dir": self.boundary_residual_dir,
                "residual_stream_dir": self.residual_stream_dir,
                "has_residual_streams": self.has_residual_streams,
                "kv_direct_ready": self.kv_direct_ready,
                "source_layer": self.source_layer,
                "target_layer": self.target_layer,
                "chain_id": self.chain_id,
                "parent_manifest_ids": self.parent_manifest_ids,
                "source_window_refs": self.source_window_refs,
            }.items()
            if value not in (None, "", (), ())
        }

    def is_empty(self) -> bool:
        return not self.identity() and not self.pending_reason


@dataclass
class ChatUserMemoryArtifact:
    """First-class user memory artifact carried by request/window metadata."""

    artifact_id: Optional[str] = None
    user_id: Optional[str] = None
    memory_scope: Optional[str] = None
    text: Optional[str] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    stale: bool = False
    superseded_by: Optional[str] = None
    source: Optional[str] = None
    artifact_path: Optional[str] = None
    local_timezone: Optional[str] = None
    local_datetime: Optional[Any] = None
    observed_at: Optional[Any] = None
    asserted_at: Optional[Any] = None
    effective_at: Optional[Any] = None
    deadline_at: Optional[Any] = None
    stale_after: Optional[Any] = None
    last_confirmed_at: Optional[Any] = None
    conflict_ids: Tuple[str, ...] = ()
    confidence: Optional[float] = None
    sensitivity: Optional[str] = None


@dataclass
class CodeTaskMemoryArtifact:
    """First-class code/task memory artifact carried by request/window metadata."""

    artifact_id: Optional[str] = None
    workspace: Optional[str] = None
    task_id: Optional[str] = None
    source_path: Optional[str] = None
    symbol: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    stale: bool = False
    superseded_by: Optional[str] = None
    artifact_path: Optional[str] = None
    workspace_id: Optional[str] = None
    repo_path: Optional[str] = None
    branch: Optional[str] = None
    base_commit: Optional[str] = None
    current_commit: Optional[str] = None
    dirty_state_hash: Optional[str] = None
    symbols: Tuple[str, ...] = ()
    imports: Tuple[str, ...] = ()
    exports: Tuple[str, ...] = ()
    dependency_edges: Tuple[str, ...] = ()
    test_links: Tuple[str, ...] = ()
    failure_links: Tuple[str, ...] = ()
    patch_target_role: Optional[str] = None


@dataclass
class UserTemporalMetadata:
    """User-memory temporal fields without imposing a product ontology."""

    user_id: Optional[str] = None
    memory_scope: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    created_at: Optional[Any] = None
    updated_at: Optional[Any] = None
    expires_at: Optional[Any] = None
    stale_after: Optional[Any] = None
    allow_stale: bool = False
    local_timezone: Optional[str] = None
    local_datetime: Optional[Any] = None
    observed_at: Optional[Any] = None
    asserted_at: Optional[Any] = None
    effective_at: Optional[Any] = None
    deadline_at: Optional[Any] = None
    last_confirmed_at: Optional[Any] = None
    superseded_by: Optional[str] = None
    conflict_ids: Tuple[str, ...] = ()
    confidence: Optional[float] = None
    sensitivity: Optional[str] = None


@dataclass
class CodeTaskMetadata:
    """Code/task fields used for workspace and artifact compatibility checks."""

    workspace: Optional[str] = None
    workspace_id: Optional[str] = None
    repo_path: Optional[str] = None
    task_id: Optional[str] = None
    source_path: Optional[str] = None
    symbol: Optional[str] = None
    role: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None
    base_commit: Optional[str] = None
    current_commit: Optional[str] = None
    dirty_state_hash: Optional[str] = None
    symbols: Tuple[str, ...] = ()
    imports: Tuple[str, ...] = ()
    exports: Tuple[str, ...] = ()
    dependency_edges: Tuple[str, ...] = ()
    test_links: Tuple[str, ...] = ()
    failure_links: Tuple[str, ...] = ()
    patch_target_role: Optional[str] = None


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
    ranking_policy: Optional[str] = None
    activation_query_vector: Optional[Tuple[float, ...]] = None
    dense_query_vector: Optional[Tuple[float, ...]] = None
    activation_gate_threshold: Optional[float] = None
    rrf_k: Optional[float] = None
    candidate_pool: Optional[int] = None
    adapter: Optional[ModelAdapterMetadata] = None
    index: Optional[IndexReadinessMetadata] = None
    apollo_residual: Optional[ApolloResidualReadinessMetadata] = None
    user_temporal: Optional[UserTemporalMetadata] = None
    code_task: Optional[CodeTaskMetadata] = None
    chat_memory: Optional[ChatUserMemoryArtifact] = None
    code_memory: Optional[CodeTaskMemoryArtifact] = None


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
    raw_router_score: Optional[float] = None
    relevance_score: Optional[float] = None
    rrf_score: Optional[float] = None
    utility_score: Optional[float] = None
    activation_score: Optional[float] = None
    activation_rank: Optional[int] = None
    activation_source: Optional[str] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    literal_score: Optional[float] = None
    literal_rank: Optional[int] = None
    entity_score: Optional[float] = None
    entity_rank: Optional[int] = None
    lexical_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    freshness_score: Optional[float] = None
    learned_reward: Optional[float] = None
    estimated_cost: Optional[float] = None
    dense_vector: Optional[Tuple[float, ...]] = None
    activation_route_vector: Optional[Tuple[float, ...]] = None
    content_fingerprint: Optional[str] = None
    selector_telemetry: Dict[str, Any] = field(default_factory=dict)
    adapter: Optional[ModelAdapterMetadata] = None
    index: Optional[IndexReadinessMetadata] = None
    apollo_residual: Optional[ApolloResidualReadinessMetadata] = None
    user_temporal: Optional[UserTemporalMetadata] = None
    code_task: Optional[CodeTaskMetadata] = None
    chat_memory: Optional[ChatUserMemoryArtifact] = None
    code_memory: Optional[CodeTaskMemoryArtifact] = None


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
    raw_router_score: Optional[float] = None
    relevance_score: Optional[float] = None
    rrf_score: Optional[float] = None
    utility_score: Optional[float] = None
    activation_score: Optional[float] = None
    activation_rank: Optional[int] = None
    activation_source: Optional[str] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    literal_score: Optional[float] = None
    literal_rank: Optional[int] = None
    entity_score: Optional[float] = None
    entity_rank: Optional[int] = None
    lexical_score: Optional[float] = None
    lexical_rank: Optional[int] = None
    freshness_score: Optional[float] = None
    learned_reward: Optional[float] = None
    estimated_cost: Optional[float] = None
    content_fingerprint: Optional[str] = None
    selector_telemetry: Dict[str, Any] = field(default_factory=dict)

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
    token_budget: Optional[int] = None
    payload_plan: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    boundary_residual_paths: Tuple[str, ...] = ()
    residual_stream_paths: Tuple[str, ...] = ()
    kv_direct_ready: bool = False
    recapture_requirements: Tuple[str, ...] = ()
    insertion_layer: Optional[str] = None
    decode_constraints: Dict[str, Any] = field(default_factory=dict)
    verification_expectations: Tuple[str, ...] = ()
    write_back_targets: Tuple[str, ...] = ()
    fast_route_ready: bool = True
    apollo_residual_ready: bool = False
    apollo_residual_identity: Dict[str, Any] = field(default_factory=dict)
    apollo_manifest_path: Optional[str] = None
    apollo_torch_store_path: Optional[str] = None

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
class CompatibilityProof:
    """Compatibility evidence for the selected route."""

    adapter_compatible: bool
    index_ready: bool
    fast_route_ready: Optional[bool] = None
    apollo_residual_ready: bool = False
    checked_window_ids: Tuple[str, ...] = ()
    adapter_identity: Dict[str, Any] = field(default_factory=dict)
    index_identity: Dict[str, Any] = field(default_factory=dict)
    apollo_residual_identity: Dict[str, Any] = field(default_factory=dict)
    jit_indexing_actions: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.fast_route_ready is None:
            self.fast_route_ready = self.index_ready


class JITIndexingRequired(RuntimeError):
    """Stop signal emitted before routing when index readiness requires JIT."""

    def __init__(
        self,
        compatibility_proof: CompatibilityProof,
        jit_indexing_actions: Sequence[str],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.compatibility_proof = compatibility_proof
        self.jit_indexing_actions = tuple(jit_indexing_actions)
        self.metadata = dict(metadata or {})
        actions = ", ".join(self.jit_indexing_actions) or "none"
        super().__init__(f"JIT indexing required before routing: {actions}")


@dataclass
class DecodePolicy:
    """Decode-time policy emitted by the router contract."""

    constraints: Dict[str, Any] = field(default_factory=dict)
    insertion_layer: Optional[str] = None
    boundary_layer: Optional[str] = None
    injection_layer: Optional[str] = None
    kv_layout: Optional[str] = None


@dataclass
class VerificationPlan:
    """Assertions downstream materializers should enforce."""

    assertions: Tuple[str, ...] = ()
    expectations: Tuple[str, ...] = ()
    required_artifacts: Tuple[str, ...] = ()


@dataclass
class WriteBackPolicy:
    """Write-back targets and restrictions for memory artifacts."""

    targets: Tuple[str, ...] = ()
    allow_user_memory_writeback: bool = False
    allow_code_task_writeback: bool = False
    notes: Tuple[str, ...] = ()


@dataclass
class RoutePacket:
    """Full harness-router packet alongside the legacy RoutePlan fields."""

    selected_memory: Tuple[RouteWindow, ...]
    evidence: Tuple[EvidenceSupport, ...]
    tiering: Tuple[TierAssignment, ...]
    compatibility_proof: CompatibilityProof
    materialization_plan: MaterializationPlan
    decode_policy: DecodePolicy
    verification_plan: VerificationPlan
    write_back_policy: WriteBackPolicy


@dataclass
class RoutePlan:
    """Complete router output consumed by downstream workers."""

    candidates: Tuple[RouteCandidate, ...]
    tier_assignments: Tuple[TierAssignment, ...]
    evidence_supports: Tuple[EvidenceSupport, ...]
    materialization_plan: MaterializationPlan
    router_metadata: Dict[str, Any] = field(default_factory=dict)
    selected_candidate: Optional[RouteCandidate] = None
    route_packet: Optional[RoutePacket] = None

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
        route_request.activation_query_vector = _coerce_vector(route_request.activation_query_vector)
        route_request.dense_query_vector = _coerce_vector(route_request.dense_query_vector)
        route_windows = tuple(self._coerce_window(window, index) for index, window in enumerate(windows))
        mode = route_request.capability_mode or GENERAL_RECALL
        if mode not in CAPABILITY_MODES:
            if not _metadata_bool(route_request.metadata, "allow_mode_fallback"):
                raise ValueError(f"unknown capability mode: {mode}")
            route_request.metadata["mode_fallback"] = {
                "requested_mode": mode,
                "fallback_mode": GENERAL_RECALL,
            }
            mode = GENERAL_RECALL

        compatibility = self._validate_adapter_and_index(route_request, route_windows)

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

        ranked = self._rank_candidates(candidates, route_request)
        assignments = self._assign_tiers(ranked)
        materialization_plan = self._build_materialization_plan(
            assignments, route_request, compatibility
        )
        selected = ranked[0] if ranked else None
        selected_windows = tuple(candidate.window for candidate in ranked)
        self._assert_route_contract(route_request, selected_windows, materialization_plan, compatibility)

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
                "compatibility_proof": compatibility.adapter_identity,
                "index_identity": compatibility.index_identity,
                "jit_indexing_actions": compatibility.jit_indexing_actions,
            }
        )
        if "mode_fallback" in route_request.metadata:
            metadata["mode_fallback"] = route_request.metadata["mode_fallback"]

        route_packet = self._build_route_packet(
            selected_windows=selected_windows,
            supports=tuple(supports),
            assignments=assignments,
            compatibility=compatibility,
            materialization_plan=materialization_plan,
            request=route_request,
        )

        plan = RoutePlan(
            candidates=ranked,
            tier_assignments=assignments,
            evidence_supports=tuple(supports),
            materialization_plan=materialization_plan,
            router_metadata=metadata,
            selected_candidate=selected,
            route_packet=route_packet,
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
            if self._is_stale_user_memory(window, request) and not _metadata_bool(request.metadata, "allow_stale_user_memory"):
                stale_ids.append(window.window_id)
                continue
            if self._superseded_by(window) and not _metadata_bool(request.metadata, "allow_stale_user_memory"):
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

            raw_router_score = min(score, 1.0)
            asi = self._asi_components(
                request,
                window,
                raw_router_score=raw_router_score,
            )
            asi_bonus = self._asi_score_bonus(asi) if _metadata_bool(request.metadata, "merge_asi_scores") else 0.0
            if asi_bonus:
                score += asi_bonus
                reasons.append("optional ASI scoring signals")

            trace = {
                "memory_scope": memory_scope,
                "source_authority": authority,
                "memory_group": memory_group,
                "recency_value": recency,
                "recency_score": recency_score,
                "stale": False,
                "superseded_by": None,
            }
            trace.update(_asi_trace(asi))
            candidate = RouteCandidate(
                window=window,
                score=min(score, 1.0),
                mode=DURABLE_CHAT_MEMORY,
                reasons=tuple(reasons),
                evidence=(f"memory_group={memory_group}", f"memory_scope={memory_scope}"),
                route_trace=trace,
                **_candidate_asi_kwargs(asi),
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

            raw_router_score = min(score, 1.0)
            asi = self._asi_components(
                request,
                window,
                raw_router_score=raw_router_score,
            )
            asi_bonus = self._asi_score_bonus(asi) if _metadata_bool(request.metadata, "merge_asi_scores") else 0.0
            if asi_bonus:
                score += asi_bonus
                reasons.append("optional ASI scoring signals")

            trace = {
                "literal_hit": literal_hit,
                "lexical_hits": tuple(sorted(lexical_hits)),
                "entity_hits": tuple(sorted(entity_hits)),
            }
            trace.update(_asi_trace(asi))
            candidate = RouteCandidate(
                window=window,
                score=min(score, 1.0),
                mode=GENERAL_RECALL,
                reasons=tuple(reasons),
                evidence=tuple(evidence),
                route_trace=trace,
                **_candidate_asi_kwargs(asi),
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
        request: Optional[RouteRequest] = None,
        compatibility: Optional[CompatibilityProof] = None,
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
        metadata = request.metadata if request is not None else {}
        notes = notes + (
            (
                "activation route metadata is preserved for selected windows"
                if _metadata_bool(metadata, "require_activation_routes")
                or any(self._window_vector(window, "activation_route_vector") for window in windows)
                else ""
            ),
            (
                "dense vector metadata is preserved for selected windows"
                if _metadata_bool(metadata, "require_dense_vectors")
                or any(self._window_vector(window, "dense_vector") for window in windows)
                else ""
            ),
        )
        notes = tuple(note for note in notes if note)
        adapter = request.adapter if request is not None else None
        apollo = request.apollo_residual if request is not None else None
        apollo_identity = compatibility.apollo_residual_identity if compatibility else _apollo_identity(apollo)
        payload_plan = {
            tier: tuple(ids) for tier, ids in tier_window_ids.items()
        }
        boundary_residual_paths = tuple(
            _collect_window_metadata_values(windows, "boundary_residual_path")
        )
        residual_stream_paths = tuple(
            _collect_window_metadata_values(windows, "residual_stream_path")
        )
        kv_paths = tuple(_collect_window_metadata_values(windows, "kv_artifact_path"))
        decode_constraints = _dict_from_metadata(metadata, "decode_constraints")
        verification_expectations = _tuple_from_metadata(metadata, "verification_expectations")
        if _metadata_bool(metadata, "require_activation_routes"):
            verification_expectations = verification_expectations + ("activation route vectors present",)
        if _metadata_bool(metadata, "require_dense_vectors"):
            verification_expectations = verification_expectations + ("dense vectors present",)
        write_back_targets = _tuple_from_metadata(metadata, "write_back_targets")
        recapture_requirements = _tuple_from_metadata(metadata, "recapture_requirements")
        if compatibility and compatibility.jit_indexing_actions:
            recapture_requirements = recapture_requirements + compatibility.jit_indexing_actions
        apollo_ready = bool(compatibility.apollo_residual_ready) if compatibility else False
        return MaterializationPlan(
            windows=tuple(windows),
            tier_order=TIER_ORDER,
            tier_window_ids=tier_window_ids,
            per_tier_counts=per_tier_counts,
            notes=notes,
            token_budget=_int_or_none(metadata.get("token_budget")),
            payload_plan=payload_plan,
            boundary_residual_paths=boundary_residual_paths,
            residual_stream_paths=residual_stream_paths,
            kv_direct_ready=bool(kv_paths) or (apollo_ready and bool(apollo and apollo.kv_direct_ready)),
            recapture_requirements=recapture_requirements,
            insertion_layer=_optional_str(
                metadata.get("insertion_layer")
                or (adapter.injection_layer if adapter else None)
            ),
            decode_constraints=decode_constraints,
            verification_expectations=verification_expectations,
            write_back_targets=write_back_targets,
            fast_route_ready=bool(compatibility.fast_route_ready) if compatibility else True,
            apollo_residual_ready=apollo_ready,
            apollo_residual_identity=apollo_identity,
            apollo_manifest_path=apollo.manifest_path if apollo else None,
            apollo_torch_store_path=apollo.torch_store_path if apollo else None,
        )

    def _validate_adapter_and_index(
        self,
        request: RouteRequest,
        windows: Tuple[RouteWindow, ...],
    ) -> CompatibilityProof:
        adapter = request.adapter or _coerce_adapter_metadata(request.metadata)
        request.adapter = adapter
        active_adapter = adapter if adapter and not adapter.is_empty() else None
        allow_jit = _metadata_bool(request.metadata, "allow_jit_indexing")
        require_ready = active_adapter is not None or _metadata_bool(request.metadata, "require_index_ready")
        require_apollo = _requires_apollo_residual_readiness(request.metadata)
        checked_ids: List[str] = []
        notes: List[str] = []
        jit_actions: List[str] = []
        require_activation_routes = _metadata_bool(request.metadata, "require_activation_routes")
        require_dense_vectors = _metadata_bool(request.metadata, "require_dense_vectors")
        if require_activation_routes and active_adapter and active_adapter.route_dimension is not None:
            self._assert_vector_dimension(
                request.activation_query_vector,
                int(active_adapter.route_dimension),
                "activation query vector",
            )

        if (
            not active_adapter
            and not require_ready
            and not require_apollo
            and not require_activation_routes
            and not require_dense_vectors
            and request.activation_query_vector is None
            and request.dense_query_vector is None
        ):
            return CompatibilityProof(
                adapter_compatible=True,
                index_ready=True,
                fast_route_ready=True,
                notes=("no active adapter metadata supplied",),
            )

        adapter_identity = active_adapter.identity() if active_adapter else {}
        request_index = request.index or _coerce_index_metadata(request.metadata)
        request.index = request_index
        request_index_identity = _index_identity(request_index)
        if require_ready and not self._index_ready(request_index, active_adapter):
            action = "jit_index_request"
            if not allow_jit:
                raise RuntimeError("request index is not ready for active adapter")
            jit_actions.append(action)
            notes.append("request index scheduled for JIT indexing")

        for window in windows:
            checked_ids.append(window.window_id)
            window_adapter = window.adapter or _coerce_adapter_metadata(window.metadata)
            window.adapter = window_adapter
            if active_adapter and window_adapter and not window_adapter.is_empty():
                mismatch = _adapter_mismatch(active_adapter, window_adapter)
                if mismatch:
                    raise RuntimeError(
                        "window adapter metadata incompatible for "
                        f"{window.window_id}: {', '.join(mismatch)}"
                    )

            window_index = window.index or _coerce_index_metadata(window.metadata)
            window.index = window_index
            if require_ready and not self._index_ready(window_index, active_adapter):
                if not allow_jit:
                    raise RuntimeError(
                        f"window index is not ready for active adapter: {window.window_id}"
                    )
                jit_actions.append(f"jit_index_window:{window.window_id}")
            elif window_index and _norm(window_index.status or "") == "in_flight":
                raise RuntimeError(f"window index is in_flight: {window.window_id}")

            activation_vector = self._window_vector(window, "activation_route_vector")
            dense_vector = self._window_vector(window, "dense_vector")
            if request.activation_query_vector is not None and activation_vector is not None:
                self._assert_matching_vector_dimensions(
                    request.activation_query_vector,
                    activation_vector,
                    f"activation query/window vector for {window.window_id}",
                )
            if request.dense_query_vector is not None and dense_vector is not None:
                self._assert_matching_vector_dimensions(
                    request.dense_query_vector,
                    dense_vector,
                    f"dense query/window vector for {window.window_id}",
                )
            if require_activation_routes and active_adapter and active_adapter.route_dimension is not None:
                self._assert_vector_dimension(
                    activation_vector,
                    int(active_adapter.route_dimension),
                    f"activation route vector for {window.window_id}",
                )
            if require_dense_vectors and request.dense_query_vector is not None and dense_vector is not None:
                self._assert_matching_vector_dimensions(
                    request.dense_query_vector,
                    dense_vector,
                    f"required dense vector for {window.window_id}",
                )

        apollo_residual = request.apollo_residual or _coerce_apollo_residual_metadata(request.metadata)
        request.apollo_residual = apollo_residual
        apollo_identity = _apollo_identity(apollo_residual)
        apollo_ready = _apollo_residual_ready(
            apollo_residual,
            require_kv_direct=_metadata_bool(request.metadata, "require_kv_direct"),
        )
        if require_apollo and not apollo_ready:
            reason = _apollo_unready_reason(
                apollo_residual,
                require_kv_direct=_metadata_bool(request.metadata, "require_kv_direct"),
            )
            action = "jit_apollo_residual_materialization"
            if not allow_jit:
                raise RuntimeError(f"apollo residual readiness is not ready: {reason}")
            jit_actions.append(action)
            notes.append(f"apollo residual materialization scheduled: {reason}")

        if jit_actions:
            fast_route_ready = not any(action.startswith("jit_index") for action in jit_actions)
            request.metadata["jit_indexing_actions"] = tuple(jit_actions)
            request.metadata["jit_indexing_required"] = True
            request.metadata["jit_indexing_stop"] = "pre_route"
            proof = CompatibilityProof(
                adapter_compatible=True,
                index_ready=fast_route_ready,
                fast_route_ready=fast_route_ready,
                apollo_residual_ready=apollo_ready,
                checked_window_ids=tuple(checked_ids),
                adapter_identity=adapter_identity,
                index_identity=request_index_identity,
                apollo_residual_identity=apollo_identity,
                jit_indexing_actions=tuple(jit_actions),
                notes=tuple(notes),
            )
            raise JITIndexingRequired(
                compatibility_proof=proof,
                jit_indexing_actions=tuple(jit_actions),
                metadata={
                    "jit_indexing_actions": tuple(jit_actions),
                    "jit_indexing_required": True,
                    "jit_indexing_stop": "pre_route",
                    "checked_window_ids": tuple(checked_ids),
                    "adapter_identity": adapter_identity,
                    "index_identity": request_index_identity,
                    "apollo_residual_identity": apollo_identity,
                },
            )

        return CompatibilityProof(
            adapter_compatible=True,
            index_ready=not jit_actions,
            fast_route_ready=True,
            apollo_residual_ready=apollo_ready,
            checked_window_ids=tuple(checked_ids),
            adapter_identity=adapter_identity,
            index_identity=request_index_identity,
            apollo_residual_identity=apollo_identity,
            jit_indexing_actions=tuple(jit_actions),
            notes=tuple(notes),
        )

    def _index_ready(
        self,
        index: Optional[IndexReadinessMetadata],
        adapter: Optional[ModelAdapterMetadata],
    ) -> bool:
        if index is None:
            return False
        status = _norm(index.status or "")
        if status in {"in_flight", "building", "pending", "missing", "failed"}:
            return False
        if status and status not in {"ready", "complete", "built", "available"}:
            return False
        if adapter:
            for key, expected in adapter.identity().items():
                actual = _index_adapter_value(index, key)
                if actual in (None, ""):
                    return False
                if key == "route_dimension":
                    try:
                        if int(actual) != int(expected):
                            return False
                    except (TypeError, ValueError):
                        return False
                elif actual != expected:
                    return False
        return bool(status)

    def _build_route_packet(
        self,
        selected_windows: Tuple[RouteWindow, ...],
        supports: Tuple[EvidenceSupport, ...],
        assignments: Tuple[TierAssignment, ...],
        compatibility: CompatibilityProof,
        materialization_plan: MaterializationPlan,
        request: RouteRequest,
    ) -> RoutePacket:
        adapter = request.adapter
        decode_policy = DecodePolicy(
            constraints=materialization_plan.decode_constraints,
            insertion_layer=materialization_plan.insertion_layer,
            boundary_layer=adapter.boundary_layer if adapter else None,
            injection_layer=adapter.injection_layer if adapter else None,
            kv_layout=adapter.kv_layout if adapter else None,
        )
        verification_plan = VerificationPlan(
            assertions=(
                "adapter compatibility",
                "selected windows exist",
                "selected indexes not in_flight",
                "workspace compatibility when declared",
                "active user memory is not stale or superseded unless allowed",
            )
            + (
                ("apollo residual readiness",)
                if _requires_apollo_residual_readiness(request.metadata)
                else ()
            ),
            expectations=materialization_plan.verification_expectations,
            required_artifacts=tuple(
                value
                for value in (
                    "kv_artifact_path" if _metadata_bool(request.metadata, "require_kv_direct") else "",
                    "residual_stream_path" if _metadata_bool(request.metadata, "require_residual_stream") else "",
                    "boundary_residual_path" if _metadata_bool(request.metadata, "require_boundary_residual") else "",
                    "activation_route_vector" if _metadata_bool(request.metadata, "require_activation_routes") else "",
                    "dense_vector" if _metadata_bool(request.metadata, "require_dense_vectors") else "",
                    "apollo_residual_manifest" if _requires_apollo_residual_readiness(request.metadata) else "",
                )
                if value
            ),
        )
        write_back_policy = WriteBackPolicy(
            targets=materialization_plan.write_back_targets,
            allow_user_memory_writeback=_metadata_bool(request.metadata, "allow_user_memory_writeback"),
            allow_code_task_writeback=_metadata_bool(request.metadata, "allow_code_task_writeback"),
            notes=_tuple_from_metadata(request.metadata, "write_back_notes"),
        )
        return RoutePacket(
            selected_memory=selected_windows,
            evidence=supports,
            tiering=assignments,
            compatibility_proof=compatibility,
            materialization_plan=materialization_plan,
            decode_policy=decode_policy,
            verification_plan=verification_plan,
            write_back_policy=write_back_policy,
        )

    def _assert_route_contract(
        self,
        request: RouteRequest,
        selected_windows: Tuple[RouteWindow, ...],
        materialization_plan: MaterializationPlan,
        compatibility: CompatibilityProof,
    ) -> None:
        if not compatibility.adapter_compatible:
            raise AssertionError("adapter compatibility proof failed")
        apollo_satisfies_residual = compatibility.apollo_residual_ready
        if _requires_apollo_residual_readiness(request.metadata) and not apollo_satisfies_residual:
            raise AssertionError("apollo residual readiness proof failed")
        selected_ids = {window.window_id for window in selected_windows}
        materialized_ids = {window.window_id for window in materialization_plan.windows}
        missing = selected_ids - materialized_ids
        if missing:
            raise AssertionError("selected windows are not materialized: " + ", ".join(sorted(missing)))
        source_refs = set(_apollo_source_window_refs(request.apollo_residual))
        if source_refs:
            missing_refs = selected_ids - source_refs
            if missing_refs:
                raise AssertionError(
                    "selected windows are missing from Apollo residual source refs: "
                    + ", ".join(sorted(missing_refs))
                )
        for window in selected_windows:
            index = window.index or _coerce_index_metadata(window.metadata)
            if index and _norm(index.status or "") == "in_flight":
                raise AssertionError(f"selected window index is in_flight: {window.window_id}")
            if _metadata_bool(request.metadata, "require_kv_direct") and not _window_has_any(
                window, ("kv_artifact_path", "kv_cache_path", "kv_path")
            ) and not (request.apollo_residual and request.apollo_residual.kv_direct_ready):
                raise AssertionError(f"selected window lacks KV artifact: {window.window_id}")
            if _metadata_bool(request.metadata, "require_residual_stream") and not _window_has_any(
                window, ("residual_stream_path", "residual_path")
            ) and not apollo_satisfies_residual:
                raise AssertionError(f"selected window lacks residual stream: {window.window_id}")
            if _metadata_bool(request.metadata, "require_boundary_residual") and not _window_has_any(
                window, ("boundary_residual_path",)
            ) and not apollo_satisfies_residual:
                raise AssertionError(f"selected window lacks boundary residual: {window.window_id}")
            if _metadata_bool(request.metadata, "require_window_files"):
                path = self._window_path(window)
                if not path or not _path_exists(path):
                    raise AssertionError(f"selected window file does not exist: {window.window_id}")
            if _metadata_bool(request.metadata, "require_artifact_files"):
                self._assert_required_artifact_files(window, request)

        for window in selected_windows:
            mismatch = self._workspace_state_mismatch(request, window)
            if mismatch:
                raise AssertionError(
                    f"selected code/task workspace mismatch for {window.window_id}: {mismatch}"
                )

        allow_stale = _metadata_bool(request.metadata, "allow_stale_user_memory")
        for window in selected_windows:
            artifact = window.chat_memory or _coerce_chat_memory_artifact(window.metadata)
            if artifact or window.user_temporal or self._window_value(window, "user_id", None):
                if self._is_stale_user_memory(window, request) and not allow_stale:
                    raise AssertionError(f"selected user memory is stale: {window.window_id}")
                if self._superseded_by(window) and not allow_stale:
                    raise AssertionError(f"selected user memory is superseded: {window.window_id}")
            if _metadata_bool(request.metadata, "require_activation_routes"):
                if not self._window_vector(window, "activation_route_vector"):
                    raise AssertionError(f"selected window lacks activation route vector: {window.window_id}")
            if _metadata_bool(request.metadata, "require_dense_vectors"):
                if not self._window_vector(window, "dense_vector"):
                    raise AssertionError(f"selected window lacks dense vector: {window.window_id}")

    def _asi_components(
        self,
        request: RouteRequest,
        window: RouteWindow,
        *,
        raw_router_score: float,
    ) -> Dict[str, Any]:
        activation_vector = self._window_vector(window, "activation_route_vector")
        dense_vector = self._window_vector(window, "dense_vector")
        activation_score = _field_float(window, "activation_score")
        dense_score = _field_float(window, "dense_score")
        activation_source = _field_str(window, "activation_source")

        if request.activation_query_vector is not None and activation_vector is not None:
            activation_score = _cosine_similarity(request.activation_query_vector, activation_vector)
            activation_source = activation_source or "activation_route_vector"
        if request.dense_query_vector is not None and dense_vector is not None:
            dense_score = _cosine_similarity(request.dense_query_vector, dense_vector)

        present = set()
        for key, value in (
            ("relevance_score", _field_float(window, "relevance_score")),
            ("rrf_score", _field_float(window, "rrf_score")),
            ("utility_score", _field_float(window, "utility_score")),
            ("activation_score", activation_score),
            ("dense_score", dense_score),
            ("literal_score", _field_float(window, "literal_score")),
            ("entity_score", _field_float(window, "entity_score")),
            ("lexical_score", _field_float(window, "lexical_score")),
            ("freshness_score", _field_float(window, "freshness_score")),
            ("learned_reward", _field_float(window, "learned_reward")),
            ("estimated_cost", _field_float(window, "estimated_cost")),
        ):
            if value is not None:
                present.add(key)

        gate_threshold = request.activation_gate_threshold
        gate_passed = None
        if gate_threshold is not None and activation_score is not None:
            gate_passed = activation_score >= gate_threshold

        relevance_score = _field_float(window, "relevance_score")
        if relevance_score is None:
            relevance_score = raw_router_score

        explicit_raw_score = _field_float(window, "raw_router_score")
        return {
            "raw_router_score": explicit_raw_score if explicit_raw_score is not None else raw_router_score,
            "relevance_score": relevance_score,
            "rrf_score": _field_float(window, "rrf_score"),
            "utility_score": _field_float(window, "utility_score"),
            "activation_score": activation_score,
            "activation_rank": _field_int(window, "activation_rank"),
            "activation_source": activation_source,
            "dense_score": dense_score,
            "dense_rank": _field_int(window, "dense_rank"),
            "literal_score": _field_float(window, "literal_score"),
            "literal_rank": _field_int(window, "literal_rank"),
            "entity_score": _field_float(window, "entity_score"),
            "entity_rank": _field_int(window, "entity_rank"),
            "lexical_score": _field_float(window, "lexical_score"),
            "lexical_rank": _field_int(window, "lexical_rank"),
            "freshness_score": _field_float(window, "freshness_score"),
            "learned_reward": _field_float(window, "learned_reward"),
            "estimated_cost": _field_float(window, "estimated_cost"),
            "content_fingerprint": _field_str(window, "content_fingerprint"),
            "selector_telemetry": _field_dict(window, "selector_telemetry"),
            "activation_gate_threshold": gate_threshold,
            "activation_gate_passed": gate_passed,
            "present_components": tuple(sorted(present)),
        }

    def _asi_score_bonus(self, asi: Mapping[str, Any]) -> float:
        present = set(asi.get("present_components", ()))
        weights = (
            ("activation_score", 0.15),
            ("dense_score", 0.15),
            ("literal_score", 0.10),
            ("entity_score", 0.10),
            ("lexical_score", 0.10),
            ("freshness_score", 0.10),
        )
        bonus = 0.0
        for key, weight in weights:
            if key in present:
                bonus += max(0.0, min(1.0, float(asi.get(key) or 0.0))) * weight
        if "estimated_cost" in present:
            bonus -= max(0.0, float(asi.get("estimated_cost") or 0.0)) * 0.05
        return max(-0.25, min(0.50, bonus))

    def _utility_v2_score(
        self,
        candidate: RouteCandidate,
        request: RouteRequest,
        *,
        max_cost: float = 1.0,
    ) -> float:
        if candidate.utility_score is not None:
            return float(candidate.utility_score)
        relevance = candidate.relevance_score
        if relevance is None:
            relevance = candidate.raw_router_score if candidate.raw_router_score is not None else candidate.score
        rrf = candidate.rrf_score
        if rrf is None:
            ranks = tuple(
                rank
                for rank in (
                    candidate.activation_rank,
                    candidate.dense_rank,
                    candidate.literal_rank,
                    candidate.entity_rank,
                    candidate.lexical_rank,
                )
                if rank is not None and rank > 0
            )
            k = request.rrf_k if request.rrf_k is not None else 60
            rrf = sum(1.0 / (float(k) + float(rank)) for rank in ranks)
        freshness = candidate.freshness_score or 0.0
        learned_reward = candidate.learned_reward or 0.0
        cost = candidate.estimated_cost or 0.0
        cost_norm = max(0.0, float(cost)) / max(1.0, float(max_cost))
        return (
            1.0 * float(relevance)
            + 1.0 * float(rrf)
            + 0.10 * float(freshness)
            + 0.50 * float(learned_reward)
            - 0.10 * cost_norm
        )

    def _candidate_session_id(self, candidate: RouteCandidate) -> str:
        return str(
            self._window_value(candidate.window, "session_id", "")
            or self._window_value(candidate.window, "source_session", "")
            or ""
        )

    def _rank_candidates(
        self,
        candidates: Sequence[RouteCandidate],
        request: Optional[RouteRequest] = None,
    ) -> Tuple[RouteCandidate, ...]:
        policy = _ranking_policy_value(request)
        if not policy:
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    self._temporal_sort_key(candidate.window),
                    candidate.window_id,
                ),
            )
        elif policy == "utility-v2":
            max_cost = max(
                (
                    float(candidate.estimated_cost)
                    for candidate in candidates
                    if candidate.estimated_cost is not None
                ),
                default=1.0,
            )

            def utility(candidate: RouteCandidate) -> float:
                return self._utility_v2_score(candidate, request, max_cost=max_cost)

            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    -utility(candidate),
                    -float(candidate.rrf_score or 0.0),
                    -float(
                        candidate.relevance_score
                        if candidate.relevance_score is not None
                        else candidate.raw_router_score
                        if candidate.raw_router_score is not None
                        else candidate.score
                    ),
                    self._candidate_session_id(candidate),
                    candidate.window_id,
                ),
            )
            for candidate in ranked:
                utility_score = utility(candidate)
                candidate.utility_score = utility_score
                candidate.route_trace["utility_v2_score"] = utility_score
                candidate.route_trace["ranking_policy"] = "utility-v2"
        else:
            raise ValueError(f"unknown ranking policy: {policy}")
        if request and request.candidate_pool is not None and request.candidate_pool > 0:
            ranked = ranked[: request.candidate_pool]
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
            ranking_policy=_optional_str(
                data.get("ranking_policy")
                or data.get("selector_policy")
                or metadata.get("ranking_policy")
                or metadata.get("selector_policy")
            ),
                activation_query_vector=_coerce_vector(
                    data.get("activation_query_vector", metadata.get("activation_query_vector"))
                ),
                dense_query_vector=_coerce_vector(
                    data.get("dense_query_vector", metadata.get("dense_query_vector"))
                ),
                activation_gate_threshold=_float_or_none(
                    data.get("activation_gate_threshold", metadata.get("activation_gate_threshold"))
                ),
                rrf_k=_float_or_none(data.get("rrf_k", metadata.get("rrf_k"))),
                candidate_pool=_int_or_none(data.get("candidate_pool", metadata.get("candidate_pool"))),
                adapter=_coerce_adapter_metadata(data.get("adapter") or metadata),
                index=_coerce_index_metadata(data.get("index") or metadata),
                apollo_residual=_coerce_apollo_residual_metadata(data.get("apollo_residual") or metadata),
                user_temporal=_coerce_user_temporal_metadata(data.get("user_temporal") or metadata),
                code_task=_coerce_code_task_metadata(data.get("code_task") or metadata),
                chat_memory=_coerce_chat_memory_artifact(data.get("chat_memory") or metadata),
                code_memory=_coerce_code_memory_artifact(data.get("code_memory") or metadata),
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
            ranking_policy=_optional_str(
                getattr(request, "ranking_policy", None)
                or metadata.get("ranking_policy")
                or metadata.get("selector_policy")
            ),
            activation_query_vector=_coerce_vector(
                getattr(request, "activation_query_vector", None)
                if getattr(request, "activation_query_vector", None) is not None
                else metadata.get("activation_query_vector")
            ),
            dense_query_vector=_coerce_vector(
                getattr(request, "dense_query_vector", None)
                if getattr(request, "dense_query_vector", None) is not None
                else metadata.get("dense_query_vector")
            ),
            activation_gate_threshold=_float_or_none(
                getattr(request, "activation_gate_threshold", None)
                if getattr(request, "activation_gate_threshold", None) is not None
                else metadata.get("activation_gate_threshold")
            ),
            rrf_k=_float_or_none(
                getattr(request, "rrf_k", None)
                if getattr(request, "rrf_k", None) is not None
                else metadata.get("rrf_k")
            ),
            candidate_pool=_int_or_none(
                getattr(request, "candidate_pool", None)
                if getattr(request, "candidate_pool", None) is not None
                else metadata.get("candidate_pool")
            ),
            adapter=_coerce_adapter_metadata(getattr(request, "adapter", None) or metadata),
            index=_coerce_index_metadata(getattr(request, "index", None) or metadata),
            apollo_residual=_coerce_apollo_residual_metadata(getattr(request, "apollo_residual", None) or metadata),
            user_temporal=_coerce_user_temporal_metadata(getattr(request, "user_temporal", None) or metadata),
            code_task=_coerce_code_task_metadata(getattr(request, "code_task", None) or metadata),
            chat_memory=_coerce_chat_memory_artifact(getattr(request, "chat_memory", None) or metadata),
            code_memory=_coerce_code_memory_artifact(getattr(request, "code_memory", None) or metadata),
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
                raw_router_score=_float_or_none(data.get("raw_router_score", metadata.get("raw_router_score"))),
                relevance_score=_float_or_none(data.get("relevance_score", metadata.get("relevance_score"))),
                rrf_score=_float_or_none(data.get("rrf_score", metadata.get("rrf_score"))),
                utility_score=_float_or_none(data.get("utility_score", metadata.get("utility_score"))),
                activation_score=_float_or_none(data.get("activation_score", metadata.get("activation_score"))),
                activation_rank=_int_or_none(data.get("activation_rank", metadata.get("activation_rank"))),
                activation_source=_optional_str(data.get("activation_source", metadata.get("activation_source"))),
                dense_score=_float_or_none(data.get("dense_score", metadata.get("dense_score"))),
                dense_rank=_int_or_none(data.get("dense_rank", metadata.get("dense_rank"))),
                literal_score=_float_or_none(data.get("literal_score", metadata.get("literal_score"))),
                literal_rank=_int_or_none(data.get("literal_rank", metadata.get("literal_rank"))),
                entity_score=_float_or_none(data.get("entity_score", metadata.get("entity_score"))),
                entity_rank=_int_or_none(data.get("entity_rank", metadata.get("entity_rank"))),
                lexical_score=_float_or_none(data.get("lexical_score", metadata.get("lexical_score"))),
                lexical_rank=_int_or_none(data.get("lexical_rank", metadata.get("lexical_rank"))),
                freshness_score=_float_or_none(data.get("freshness_score", metadata.get("freshness_score"))),
                learned_reward=_float_or_none(data.get("learned_reward", metadata.get("learned_reward"))),
                estimated_cost=_float_or_none(data.get("estimated_cost", metadata.get("estimated_cost"))),
                dense_vector=_coerce_vector(data.get("dense_vector", metadata.get("dense_vector"))),
                activation_route_vector=_coerce_vector(
                    data.get("activation_route_vector", metadata.get("activation_route_vector"))
                ),
                content_fingerprint=_optional_str(
                    data.get("content_fingerprint", metadata.get("content_fingerprint"))
                ),
                selector_telemetry=_dict_from_metadata(
                    {"selector_telemetry": data.get("selector_telemetry", metadata.get("selector_telemetry"))},
                    "selector_telemetry",
                ),
                adapter=_coerce_adapter_metadata(data.get("adapter") or metadata),
                index=_coerce_index_metadata(data.get("index") or metadata),
                apollo_residual=_coerce_apollo_residual_metadata(data.get("apollo_residual") or metadata),
                user_temporal=_coerce_user_temporal_metadata(data.get("user_temporal") or metadata),
                code_task=_coerce_code_task_metadata(data.get("code_task") or metadata),
                chat_memory=_coerce_chat_memory_artifact(data.get("chat_memory") or metadata),
                code_memory=_coerce_code_memory_artifact(data.get("code_memory") or metadata),
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
            raw_router_score=_float_or_none(
                getattr(window, "raw_router_score", None)
                if getattr(window, "raw_router_score", None) is not None
                else metadata.get("raw_router_score")
            ),
            relevance_score=_float_or_none(
                getattr(window, "relevance_score", None)
                if getattr(window, "relevance_score", None) is not None
                else metadata.get("relevance_score")
            ),
            rrf_score=_float_or_none(
                getattr(window, "rrf_score", None)
                if getattr(window, "rrf_score", None) is not None
                else metadata.get("rrf_score")
            ),
            utility_score=_float_or_none(
                getattr(window, "utility_score", None)
                if getattr(window, "utility_score", None) is not None
                else metadata.get("utility_score")
            ),
            activation_score=_float_or_none(
                getattr(window, "activation_score", None)
                if getattr(window, "activation_score", None) is not None
                else metadata.get("activation_score")
            ),
            activation_rank=_int_or_none(
                getattr(window, "activation_rank", None)
                if getattr(window, "activation_rank", None) is not None
                else metadata.get("activation_rank")
            ),
            activation_source=_optional_str(
                getattr(window, "activation_source", None)
                or metadata.get("activation_source")
            ),
            dense_score=_float_or_none(
                getattr(window, "dense_score", None)
                if getattr(window, "dense_score", None) is not None
                else metadata.get("dense_score")
            ),
            dense_rank=_int_or_none(
                getattr(window, "dense_rank", None)
                if getattr(window, "dense_rank", None) is not None
                else metadata.get("dense_rank")
            ),
            literal_score=_float_or_none(
                getattr(window, "literal_score", None)
                if getattr(window, "literal_score", None) is not None
                else metadata.get("literal_score")
            ),
            literal_rank=_int_or_none(
                getattr(window, "literal_rank", None)
                if getattr(window, "literal_rank", None) is not None
                else metadata.get("literal_rank")
            ),
            entity_score=_float_or_none(
                getattr(window, "entity_score", None)
                if getattr(window, "entity_score", None) is not None
                else metadata.get("entity_score")
            ),
            entity_rank=_int_or_none(
                getattr(window, "entity_rank", None)
                if getattr(window, "entity_rank", None) is not None
                else metadata.get("entity_rank")
            ),
            lexical_score=_float_or_none(
                getattr(window, "lexical_score", None)
                if getattr(window, "lexical_score", None) is not None
                else metadata.get("lexical_score")
            ),
            lexical_rank=_int_or_none(
                getattr(window, "lexical_rank", None)
                if getattr(window, "lexical_rank", None) is not None
                else metadata.get("lexical_rank")
            ),
            freshness_score=_float_or_none(
                getattr(window, "freshness_score", None)
                if getattr(window, "freshness_score", None) is not None
                else metadata.get("freshness_score")
            ),
            learned_reward=_float_or_none(
                getattr(window, "learned_reward", None)
                if getattr(window, "learned_reward", None) is not None
                else metadata.get("learned_reward")
            ),
            estimated_cost=_float_or_none(
                getattr(window, "estimated_cost", None)
                if getattr(window, "estimated_cost", None) is not None
                else metadata.get("estimated_cost")
            ),
            dense_vector=_coerce_vector(
                getattr(window, "dense_vector", None)
                if getattr(window, "dense_vector", None) is not None
                else metadata.get("dense_vector")
            ),
            activation_route_vector=_coerce_vector(
                getattr(window, "activation_route_vector", None)
                if getattr(window, "activation_route_vector", None) is not None
                else metadata.get("activation_route_vector")
            ),
            content_fingerprint=_optional_str(
                getattr(window, "content_fingerprint", None)
                or metadata.get("content_fingerprint")
            ),
            selector_telemetry=(
                dict(getattr(window, "selector_telemetry", {}) or {})
                if isinstance(getattr(window, "selector_telemetry", {}) or {}, Mapping)
                else _dict_from_metadata(metadata, "selector_telemetry")
            ),
            adapter=_coerce_adapter_metadata(getattr(window, "adapter", None) or metadata),
            index=_coerce_index_metadata(getattr(window, "index", None) or metadata),
            apollo_residual=_coerce_apollo_residual_metadata(getattr(window, "apollo_residual", None) or metadata),
            user_temporal=_coerce_user_temporal_metadata(getattr(window, "user_temporal", None) or metadata),
            code_task=_coerce_code_task_metadata(getattr(window, "code_task", None) or metadata),
            chat_memory=_coerce_chat_memory_artifact(getattr(window, "chat_memory", None) or metadata),
            code_memory=_coerce_code_memory_artifact(getattr(window, "code_memory", None) or metadata),
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
        if window.activation_score is not None:
            return window.activation_score
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
        value = getattr(window, key, None)
        if value is not None and key not in {"metadata", "selector_telemetry"}:
            return value
        if key in window.metadata:
            return window.metadata[key]
        return default

    def _window_vector(self, window: RouteWindow, key: str) -> Optional[Tuple[float, ...]]:
        value = getattr(window, key, None)
        if value is None:
            value = window.metadata.get(key)
        return _coerce_vector(value)

    def _assert_vector_dimension(
        self,
        vector: Optional[Tuple[float, ...]],
        dimension: int,
        label: str,
    ) -> None:
        if vector is None:
            return
        if len(vector) != dimension:
            raise ValueError(
                f"{label} dimension mismatch: expected {dimension}, got {len(vector)}"
            )

    def _assert_matching_vector_dimensions(
        self,
        left: Tuple[float, ...],
        right: Tuple[float, ...],
        label: str,
    ) -> None:
        if len(left) != len(right):
            raise ValueError(
                f"{label} dimension mismatch: query has {len(left)}, window has {len(right)}"
            )

    def _window_bool(self, window: RouteWindow, key: str) -> bool:
        value = self._window_value(window, key, False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _assert_required_artifact_files(self, window: RouteWindow, request: RouteRequest) -> None:
        required_keys: List[Tuple[str, Tuple[str, ...]]] = []
        if _metadata_bool(request.metadata, "require_kv_direct"):
            required_keys.append(("KV artifact", ("kv_artifact_path", "kv_cache_path", "kv_path")))
        if _metadata_bool(request.metadata, "require_residual_stream"):
            required_keys.append(("residual stream", ("residual_stream_path", "residual_path")))
        if _metadata_bool(request.metadata, "require_boundary_residual"):
            required_keys.append(("boundary residual", ("boundary_residual_path",)))
        required_keys.append(("memory artifact", ("artifact_path", "artifact_file", "file_path")))

        for label, keys in required_keys:
            paths = [str(window.metadata[key]) for key in keys if window.metadata.get(key)]
            if label == "memory artifact":
                for artifact in (window.chat_memory, window.code_memory):
                    if artifact and artifact.artifact_path:
                        paths.append(artifact.artifact_path)
                if not paths:
                    continue
            if not paths:
                raise AssertionError(f"selected window lacks {label}: {window.window_id}")
            missing = [path for path in paths if not _path_exists(path)]
            if missing:
                raise AssertionError(
                    f"selected window {label} file does not exist for {window.window_id}: "
                    + ", ".join(missing)
                )

    def _workspace_state_mismatch(self, request: RouteRequest, window: RouteWindow) -> Optional[str]:
        for key in (
            "workspace_id",
            "repo_path",
            "branch",
            "base_commit",
            "current_commit",
            "dirty_state_hash",
        ):
            request_value = _workspace_state_value(request.code_task, request.metadata, key)
            code_task = window.code_task or _coerce_code_task_metadata(window.metadata)
            window_value = _workspace_state_value(code_task, window.metadata, key)
            if request_value and window_value:
                if key == "repo_path":
                    if _norm_path(request_value) != _norm_path(window_value):
                        return key
                elif str(request_value) != str(window_value):
                    return key
        return None

    def _superseded_by(self, window: RouteWindow) -> Optional[str]:
        artifact = window.chat_memory or _coerce_chat_memory_artifact(window.metadata)
        temporal = window.user_temporal or _coerce_user_temporal_metadata(window.metadata)
        return _optional_str(
            window.superseded_by
            or self._window_value(window, "superseded_by", None)
            or (artifact.superseded_by if artifact else None)
            or (temporal.superseded_by if temporal else None)
        )

    def _is_stale_user_memory(self, window: RouteWindow, request: RouteRequest) -> bool:
        artifact = window.chat_memory or _coerce_chat_memory_artifact(window.metadata)
        temporal = window.user_temporal or _coerce_user_temporal_metadata(window.metadata)
        if window.stale or self._window_bool(window, "stale"):
            return True
        if artifact and artifact.stale:
            return True
        if temporal and temporal.allow_stale:
            return False
        stale_after = (
            self._window_value(window, "stale_after", None)
            or (artifact.stale_after if artifact else None)
            or (temporal.stale_after if temporal else None)
        )
        if stale_after is None:
            return False
        reference = request.metadata.get("current_time") or request.metadata.get("now")
        reference_dt = _parse_datetime(reference) or datetime.now(timezone.utc)
        stale_dt = _parse_datetime(stale_after)
        if stale_dt is None:
            return False
        return stale_dt <= reference_dt

    def _tier_reason(self, tier: str, tier_count: int, total_count: int) -> str:
        if total_count >= 3:
            return f"{tier} receives {tier_count} window(s) under full tier coverage"
        return f"{tier} receives {tier_count} window(s); fewer than three eligible windows"


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _metadata_bool(metadata: Mapping[str, Any], key: str) -> bool:
    aliases = {
        "allow_mode_fallback": ("allow_mode_fallback", "allow_fallback", "mode_fallback_allowed"),
        "allow_jit_indexing": ("allow_jit_indexing", "jit_indexing_allowed"),
        "require_activation_routes": (
            "require_activation_routes",
            "require_activation_route_vector",
            "require_activation_route_vectors",
        ),
        "require_dense_vectors": (
            "require_dense_vectors",
            "require_dense_vector",
        ),
        "merge_asi_scores": (
            "merge_asi_scores",
            "enable_asi_score_merge",
            "enable_asi_scoring",
        ),
    }.get(key, (key,))
    for alias in aliases:
        if alias in metadata:
            value = metadata[alias]
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)
    return False


def _ranking_policy_value(request: Optional[RouteRequest]) -> str:
    if request is None:
        return ""
    raw = (
        request.ranking_policy
        or request.metadata.get("ranking_policy")
        or request.metadata.get("selector_policy")
        or ""
    )
    return _norm(raw).replace("_", "-")


def _coerce_vector(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None

    def flatten(item: Any) -> List[Any]:
        if hasattr(item, "detach"):
            try:
                item = item.detach()
            except Exception:  # noqa: BLE001 - vector-like adapters are best-effort.
                pass
        if hasattr(item, "cpu"):
            try:
                item = item.cpu()
            except Exception:  # noqa: BLE001 - keep non-tensor vector-like objects usable.
                pass
        if hasattr(item, "tolist"):
            try:
                item = item.tolist()
            except Exception:  # noqa: BLE001 - fall through to normal type validation.
                pass
        if isinstance(item, (tuple, list)):
            flattened_items: List[Any] = []
            for child in item:
                flattened_items.extend(flatten(child))
            return flattened_items
        return [item]

    flattened = flatten(value)
    if not flattened:
        return ()
    return tuple(float(item) for item in flattened)


def _dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"vector dimension mismatch: {len(left)} != {len(right)}")
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"vector dimension mismatch: {len(left)} != {len(right)}")
    left_norm = _dot_product(left, left) ** 0.5
    right_norm = _dot_product(right, right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return _dot_product(left, right) / (left_norm * right_norm)


def _field_float(window: RouteWindow, key: str) -> Optional[float]:
    value = getattr(window, key, None)
    if value is None:
        value = window.metadata.get(key)
    return _float_or_none(value)


def _field_int(window: RouteWindow, key: str) -> Optional[int]:
    value = getattr(window, key, None)
    if value is None:
        value = window.metadata.get(key)
    return _int_or_none(value)


def _field_str(window: RouteWindow, key: str) -> Optional[str]:
    value = getattr(window, key, None)
    if value is None:
        value = window.metadata.get(key)
    return _optional_str(value)


def _field_dict(window: RouteWindow, key: str) -> Dict[str, Any]:
    value = getattr(window, key, None)
    if isinstance(value, Mapping):
        return dict(value)
    value = window.metadata.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _candidate_asi_kwargs(asi: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "raw_router_score",
        "relevance_score",
        "rrf_score",
        "utility_score",
        "activation_score",
        "activation_rank",
        "activation_source",
        "dense_score",
        "dense_rank",
        "literal_score",
        "literal_rank",
        "entity_score",
        "entity_rank",
        "lexical_score",
        "lexical_rank",
        "freshness_score",
        "learned_reward",
        "estimated_cost",
        "content_fingerprint",
        "selector_telemetry",
    )
    return {key: asi.get(key) for key in keys}


def _asi_trace(asi: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "asi_components": _candidate_asi_kwargs(asi),
        "activation_gate_threshold": asi.get("activation_gate_threshold"),
        "activation_gate_passed": asi.get("activation_gate_passed"),
        "present_asi_components": tuple(asi.get("present_components", ())),
    }


def _coerce_adapter_metadata(value: Any) -> Optional[ModelAdapterMetadata]:
    if isinstance(value, ModelAdapterMetadata):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    adapter_data = _mapping_from(data.get("adapter_metadata") or data.get("model_adapter") or data.get("adapter"))
    merged = dict(data)
    merged.update(adapter_data)
    adapter = ModelAdapterMetadata(
        model_family=_optional_str(merged.get("model_family")),
        model_revision=_optional_str(merged.get("model_revision") or merged.get("model_rev")),
        model_hash=_optional_str(merged.get("model_hash")),
        tokenizer_id=_optional_str(merged.get("tokenizer_id")),
        tokenizer_hash=_optional_str(merged.get("tokenizer_hash")),
        adapter_version=_optional_str(merged.get("adapter_version")),
        route_layer=_optional_str(merged.get("route_layer")),
        boundary_layer=_optional_str(merged.get("boundary_layer")),
        injection_layer=_optional_str(merged.get("injection_layer")),
        route_dimension=_int_or_none(merged.get("route_dimension") or merged.get("dimension")),
        kv_layout=_optional_str(merged.get("kv_layout")),
    )
    return None if adapter.is_empty() else adapter


def _coerce_index_metadata(value: Any) -> Optional[IndexReadinessMetadata]:
    if isinstance(value, IndexReadinessMetadata):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    index_data = _mapping_from(data.get("index_metadata") or data.get("route_index") or data.get("index"))
    merged = dict(data)
    merged.update(index_data)
    index = IndexReadinessMetadata(
        index_id=_optional_str(merged.get("index_id")),
        adapter_key=_optional_str(merged.get("adapter_key") or merged.get("adapter_id")),
        status=_optional_str(merged.get("index_status") or merged.get("status")),
        built_at=merged.get("built_at") or merged.get("index_built_at"),
        dimension=_int_or_none(merged.get("index_dimension") or merged.get("dimension")),
        model_family=_optional_str(merged.get("index_model_family") or merged.get("model_family")),
        model_revision=_optional_str(
            merged.get("index_model_revision")
            or merged.get("model_revision")
            or merged.get("model_revision_or_hash")
        ),
        tokenizer_hash=_optional_str(merged.get("index_tokenizer_hash") or merged.get("tokenizer_hash")),
        tokenizer_id=_optional_str(merged.get("index_tokenizer_id") or merged.get("tokenizer_id")),
        model_hash=_optional_str(merged.get("index_model_hash") or merged.get("model_hash")),
        adapter_version=_optional_str(
            merged.get("index_adapter_version")
            or merged.get("adapter_version")
            or merged.get("adapter_key")
            or merged.get("adapter_id")
        ),
        route_layer=_optional_str(merged.get("index_route_layer") or merged.get("route_layer")),
        boundary_layer=_optional_str(merged.get("index_boundary_layer") or merged.get("boundary_layer")),
        injection_layer=_optional_str(merged.get("index_injection_layer") or merged.get("injection_layer")),
        route_dimension=_int_or_none(
            merged.get("index_route_dimension")
            or merged.get("route_dimension")
            or merged.get("index_dimension")
            or merged.get("dimension")
        ),
        kv_layout=_optional_str(merged.get("index_kv_layout") or merged.get("kv_layout")),
    )
    if not any(_index_identity(index).values()) and not index.status:
        return None
    return index


def _coerce_apollo_residual_metadata(value: Any) -> Optional[ApolloResidualReadinessMetadata]:
    if isinstance(value, ApolloResidualReadinessMetadata):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    apollo_data = _mapping_from(
        data.get("apollo_residual")
        or data.get("apollo_residual_metadata")
        or data.get("apollo_readiness")
    )
    merged = dict(data)
    merged.update(apollo_data)
    has_signal = any(
        key in merged
        for key in (
            "apollo_residual",
            "apollo_residual_metadata",
            "apollo_readiness",
            "apollo_ready",
            "readiness_source",
            "manifest_path",
            "apollo_manifest_path",
            "manifest_id",
            "torch_store_path",
            "window_count",
            "num_windows",
            "boundary_residual_count",
            "residual_stream_count",
            "boundary_residual_dir",
            "boundaries_dir",
            "residual_stream_dir",
            "residual_streams_dir",
            "source_layer",
            "target_layer",
        )
    )
    if not has_signal:
        return None
    status = merged.get("apollo_status") or merged.get("status")
    if status is None and merged.get("apollo_ready") is True:
        status = "ready"
    metadata = ApolloResidualReadinessMetadata(
        status=_optional_str(status),
        readiness_source=_optional_str(merged.get("readiness_source") or merged.get("apollo_readiness_source")),
        manifest_path=_optional_str(merged.get("manifest_path") or merged.get("apollo_manifest_path")),
        manifest_id=_optional_str(merged.get("manifest_id")),
        torch_store_path=_optional_str(merged.get("torch_store_path") or merged.get("store_path")),
        window_count=_int_or_none(merged.get("window_count") or merged.get("num_windows")),
        boundary_residual_count=_int_or_none(merged.get("boundary_residual_count")),
        residual_stream_count=_int_or_none(merged.get("residual_stream_count")),
        boundary_residual_dir=_optional_str(
            merged.get("boundary_residual_dir") or merged.get("boundaries_dir")
        ),
        residual_stream_dir=_optional_str(
            merged.get("residual_stream_dir") or merged.get("residual_streams_dir")
        ),
        has_residual_streams=_metadata_bool(merged, "has_residual_streams"),
        kv_direct_ready=_metadata_bool(merged, "kv_direct_ready"),
        pending_reason=_optional_str(merged.get("pending_reason")),
        source_layer=_optional_str(merged.get("source_layer") or merged.get("layer") or merged.get("crystal_layer")),
        target_layer=_optional_str(merged.get("target_layer") or merged.get("injection_layer")),
        chain_id=_optional_str(merged.get("chain_id")),
        parent_manifest_ids=_tuple_from_metadata(merged, "parent_manifest_ids"),
        source_window_refs=_tuple_from_metadata(merged, "source_window_refs")
        or _tuple_from_metadata(merged, "document_order"),
    )
    return None if metadata.is_empty() else metadata


def _coerce_chat_memory_artifact(value: Any) -> Optional[ChatUserMemoryArtifact]:
    if isinstance(value, ChatUserMemoryArtifact):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    artifact_data = _mapping_from(data.get("chat_memory") or data.get("user_memory_artifact"))
    merged = dict(data)
    merged.update(artifact_data)
    has_signal = any(
        key in merged
        for key in ("user_id", "memory_scope", "chat_memory_id", "artifact_id", "source_authority")
    )
    if not has_signal:
        return None
    return ChatUserMemoryArtifact(
        artifact_id=_optional_str(merged.get("artifact_id") or merged.get("chat_memory_id")),
        user_id=_optional_str(merged.get("user_id")),
        memory_scope=_optional_str(merged.get("memory_scope")),
        text=_optional_str(merged.get("memory_text")),
        created_at=merged.get("created_at"),
        updated_at=merged.get("updated_at"),
        stale=_metadata_bool(merged, "stale"),
        superseded_by=_optional_str(merged.get("superseded_by")),
        source=_optional_str(merged.get("source_authority") or merged.get("source")),
        artifact_path=_optional_str(merged.get("artifact_path") or merged.get("artifact_file") or merged.get("file_path")),
        local_timezone=_optional_str(merged.get("local_timezone")),
        local_datetime=merged.get("local_datetime"),
        observed_at=merged.get("observed_at"),
        asserted_at=merged.get("asserted_at"),
        effective_at=merged.get("effective_at"),
        deadline_at=merged.get("deadline_at"),
        stale_after=merged.get("stale_after"),
        last_confirmed_at=merged.get("last_confirmed_at"),
        conflict_ids=_tuple_from_metadata(merged, "conflict_ids"),
        confidence=_float_or_none(merged.get("confidence")),
        sensitivity=_optional_str(merged.get("sensitivity")),
    )


def _coerce_code_memory_artifact(value: Any) -> Optional[CodeTaskMemoryArtifact]:
    if isinstance(value, CodeTaskMemoryArtifact):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    artifact_data = _mapping_from(data.get("code_memory") or data.get("code_task_memory"))
    merged = dict(data)
    merged.update(artifact_data)
    has_signal = any(
        key in merged
        for key in (
            "workspace",
            "workspace_id",
            "repo_path",
            "task_id",
            "source_path",
            "path",
            "symbol",
            "symbols",
            "code_memory_id",
        )
    )
    if not has_signal:
        return None
    return CodeTaskMemoryArtifact(
        artifact_id=_optional_str(merged.get("artifact_id") or merged.get("code_memory_id")),
        workspace=_optional_str(merged.get("workspace")),
        task_id=_optional_str(merged.get("task_id")),
        source_path=_optional_str(merged.get("source_path") or merged.get("path")),
        symbol=_optional_str(merged.get("symbol")),
        role=_optional_str(merged.get("role") or merged.get("triad_role")),
        created_at=merged.get("created_at"),
        updated_at=merged.get("updated_at"),
        stale=_metadata_bool(merged, "stale"),
        superseded_by=_optional_str(merged.get("superseded_by")),
        artifact_path=_optional_str(merged.get("artifact_path") or merged.get("artifact_file") or merged.get("file_path")),
        workspace_id=_optional_str(merged.get("workspace_id")),
        repo_path=_optional_str(merged.get("repo_path")),
        branch=_optional_str(merged.get("branch")),
        base_commit=_optional_str(merged.get("base_commit")),
        current_commit=_optional_str(merged.get("current_commit") or merged.get("commit")),
        dirty_state_hash=_optional_str(merged.get("dirty_state_hash")),
        symbols=_tuple_or_single(merged, "symbols", "symbol"),
        imports=_tuple_from_metadata(merged, "imports"),
        exports=_tuple_from_metadata(merged, "exports"),
        dependency_edges=_tuple_from_metadata(merged, "dependency_edges"),
        test_links=_tuple_from_metadata(merged, "test_links"),
        failure_links=_tuple_from_metadata(merged, "failure_links"),
        patch_target_role=_optional_str(merged.get("patch_target_role") or merged.get("role") or merged.get("triad_role")),
    )


def _coerce_user_temporal_metadata(value: Any) -> Optional[UserTemporalMetadata]:
    if isinstance(value, UserTemporalMetadata):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    temporal_data = _mapping_from(data.get("user_temporal") or data.get("temporal_metadata"))
    merged = dict(data)
    merged.update(temporal_data)
    if not any(
        key in merged
        for key in (
            "user_id",
            "session_id",
            "turn_index",
            "memory_scope",
            "local_timezone",
            "observed_at",
            "asserted_at",
            "effective_at",
            "stale_after",
            "superseded_by",
        )
    ):
        return None
    return UserTemporalMetadata(
        user_id=_optional_str(merged.get("user_id")),
        memory_scope=_optional_str(merged.get("memory_scope")),
        session_id=_optional_str(merged.get("session_id")),
        turn_index=_int_or_none(merged.get("turn_index") or merged.get("session_turn_index")),
        created_at=merged.get("created_at"),
        updated_at=merged.get("updated_at"),
        expires_at=merged.get("expires_at"),
        stale_after=merged.get("stale_after"),
        allow_stale=_metadata_bool(merged, "allow_stale"),
        local_timezone=_optional_str(merged.get("local_timezone")),
        local_datetime=merged.get("local_datetime"),
        observed_at=merged.get("observed_at"),
        asserted_at=merged.get("asserted_at"),
        effective_at=merged.get("effective_at"),
        deadline_at=merged.get("deadline_at"),
        last_confirmed_at=merged.get("last_confirmed_at"),
        superseded_by=_optional_str(merged.get("superseded_by")),
        conflict_ids=_tuple_from_metadata(merged, "conflict_ids"),
        confidence=_float_or_none(merged.get("confidence")),
        sensitivity=_optional_str(merged.get("sensitivity")),
    )


def _coerce_code_task_metadata(value: Any) -> Optional[CodeTaskMetadata]:
    if isinstance(value, CodeTaskMetadata):
        return value
    data = _mapping_from(value)
    if not data:
        return None
    code_data = _mapping_from(data.get("code_task") or data.get("task_metadata"))
    merged = dict(data)
    merged.update(code_data)
    if not any(
        key in merged
        for key in (
            "workspace",
            "workspace_id",
            "repo_path",
            "task_id",
            "source_path",
            "path",
            "symbol",
            "symbols",
            "branch",
            "base_commit",
            "current_commit",
            "dirty_state_hash",
        )
    ):
        return None
    return CodeTaskMetadata(
        workspace=_optional_str(merged.get("workspace")),
        workspace_id=_optional_str(merged.get("workspace_id")),
        repo_path=_optional_str(merged.get("repo_path")),
        task_id=_optional_str(merged.get("task_id")),
        source_path=_optional_str(merged.get("source_path") or merged.get("path")),
        symbol=_optional_str(merged.get("symbol")),
        role=_optional_str(merged.get("role") or merged.get("triad_role")),
        branch=_optional_str(merged.get("branch")),
        commit=_optional_str(merged.get("commit")),
        base_commit=_optional_str(merged.get("base_commit")),
        current_commit=_optional_str(merged.get("current_commit") or merged.get("commit")),
        dirty_state_hash=_optional_str(merged.get("dirty_state_hash")),
        symbols=_tuple_or_single(merged, "symbols", "symbol"),
        imports=_tuple_from_metadata(merged, "imports"),
        exports=_tuple_from_metadata(merged, "exports"),
        dependency_edges=_tuple_from_metadata(merged, "dependency_edges"),
        test_links=_tuple_from_metadata(merged, "test_links"),
        failure_links=_tuple_from_metadata(merged, "failure_links"),
        patch_target_role=_optional_str(merged.get("patch_target_role") or merged.get("role") or merged.get("triad_role")),
    )


def _mapping_from(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(getattr(value, "__dict__") or {})
    return {}


def _adapter_mismatch(
    expected: ModelAdapterMetadata,
    actual: ModelAdapterMetadata,
) -> Tuple[str, ...]:
    mismatches: List[str] = []
    for key, expected_value in expected.identity().items():
        actual_value = getattr(actual, key)
        if actual_value not in (None, "") and actual_value != expected_value:
            mismatches.append(key)
    return tuple(mismatches)


def _index_identity(index: Optional[IndexReadinessMetadata]) -> Dict[str, Any]:
    if index is None:
        return {}
    return {
        key: value
        for key, value in {
            "index_id": index.index_id,
            "adapter_key": index.adapter_key,
            "status": index.status,
            "dimension": index.dimension,
            "model_family": index.model_family,
            "model_revision": index.model_revision,
            "tokenizer_id": index.tokenizer_id,
            "tokenizer_hash": index.tokenizer_hash,
            "model_hash": index.model_hash,
            "adapter_version": index.adapter_version,
            "route_layer": index.route_layer,
            "boundary_layer": index.boundary_layer,
            "injection_layer": index.injection_layer,
            "route_dimension": index.route_dimension,
            "kv_layout": index.kv_layout,
        }.items()
        if value not in (None, "")
    }


def _index_adapter_value(index: IndexReadinessMetadata, key: str) -> Any:
    if key == "adapter_version":
        return index.adapter_version or index.adapter_key
    if key == "route_dimension":
        return index.route_dimension if index.route_dimension is not None else index.dimension
    return getattr(index, key, None)


def _requires_apollo_residual_readiness(metadata: Mapping[str, Any]) -> bool:
    return any(
        _metadata_bool(metadata, key)
        for key in (
            "require_apollo_residual_ready",
            "require_kv_direct",
            "require_residual_stream",
            "require_boundary_residual",
        )
    )


def _apollo_identity(apollo: Optional[ApolloResidualReadinessMetadata]) -> Dict[str, Any]:
    return apollo.identity() if apollo else {}


def _apollo_source_window_refs(
    apollo: Optional[ApolloResidualReadinessMetadata],
) -> Tuple[str, ...]:
    if not apollo:
        return ()
    return tuple(str(ref) for ref in apollo.source_window_refs if ref not in (None, ""))


def _apollo_residual_ready(
    apollo: Optional[ApolloResidualReadinessMetadata],
    *,
    require_kv_direct: bool = False,
) -> bool:
    return _apollo_unready_reason(apollo, require_kv_direct=require_kv_direct) is None


def _apollo_unready_reason(
    apollo: Optional[ApolloResidualReadinessMetadata],
    *,
    require_kv_direct: bool = False,
) -> Optional[str]:
    if apollo is None:
        return "missing Apollo residual metadata"
    status = _norm(apollo.status or "")
    if status in {"", "missing", "pending", "building", "in_flight", "failed", "incomplete"}:
        return apollo.pending_reason or f"status is {status or 'missing'}"
    if status not in {"ready", "complete", "available", "built"}:
        return apollo.pending_reason or f"status is {status}"
    if require_kv_direct and not apollo.kv_direct_ready:
        return "kv_direct_ready is false"

    has_path_evidence = any(
        (
            apollo.manifest_path,
            apollo.torch_store_path,
            apollo.boundary_residual_dir,
            apollo.residual_stream_dir,
        )
    )
    if not has_path_evidence:
        return "ready status is sparse without manifest or artifact path evidence"
    if not apollo.window_count or apollo.window_count <= 0:
        return "ready status is sparse without positive window_count"
    if not apollo.source_layer:
        return "ready status is sparse without source_layer"
    residual_count = apollo.residual_stream_count or 0
    boundary_count = apollo.boundary_residual_count or 0
    if residual_count <= 0 and boundary_count <= 0:
        return "ready status is sparse without residual stream or boundary residual evidence"
    if apollo.has_residual_streams and residual_count <= 0:
        return "has_residual_streams is true without residual_stream_count"
    return None


def _tuple_from_metadata(metadata: Mapping[str, Any], key: str) -> Tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    try:
        return tuple(str(item) for item in value if item)
    except TypeError:
        return (str(value),)


def _tuple_or_single(metadata: Mapping[str, Any], tuple_key: str, single_key: str) -> Tuple[str, ...]:
    values = _tuple_from_metadata(metadata, tuple_key)
    if values:
        return values
    value = metadata.get(single_key)
    return (str(value),) if value else ()


def _dict_from_metadata(metadata: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = metadata.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _collect_window_metadata_values(windows: Sequence[RouteWindow], key: str) -> List[str]:
    values: List[str] = []
    for window in windows:
        value = window.metadata.get(key)
        if value:
            values.append(str(value))
    return values


def _window_has_any(window: RouteWindow, keys: Sequence[str]) -> bool:
    return any(bool(window.metadata.get(key)) for key in keys)


def _workspace_state_value(
    code_task: Optional[CodeTaskMetadata],
    metadata: Mapping[str, Any],
    key: str,
) -> Optional[str]:
    if code_task:
        value = getattr(code_task, key, None)
        if value not in (None, ""):
            return str(value)
        if key == "current_commit" and code_task.commit:
            return str(code_task.commit)
        if key == "workspace_id" and code_task.workspace:
            return str(code_task.workspace)
    aliases = {
        "workspace_id": ("workspace_id", "workspace"),
        "repo_path": ("repo_path", "repository_path"),
        "current_commit": ("current_commit", "commit"),
    }.get(key, (key,))
    for alias in aliases:
        value = metadata.get(alias)
        if value not in (None, ""):
            return str(value)
    return None


def _path_exists(value: Any) -> bool:
    if not value:
        return False
    try:
        return Path(str(value)).expanduser().exists()
    except (OSError, ValueError):
        return False


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    "ModelAdapterMetadata",
    "IndexReadinessMetadata",
    "ApolloResidualReadinessMetadata",
    "ChatUserMemoryArtifact",
    "CodeTaskMemoryArtifact",
    "UserTemporalMetadata",
    "CodeTaskMetadata",
    "RouteRequest",
    "RouteWindow",
    "RouteCandidate",
    "TierAssignment",
    "MaterializationPlan",
    "EvidenceSupport",
    "CompatibilityProof",
    "JITIndexingRequired",
    "DecodePolicy",
    "VerificationPlan",
    "WriteBackPolicy",
    "RoutePacket",
    "RoutePlan",
    "CentralRouter",
]
