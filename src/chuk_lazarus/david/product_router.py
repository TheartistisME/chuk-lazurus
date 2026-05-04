"""Product-callable David router wrappers.

The product router translates benchmark-proven capabilities into stable
methodology metadata without importing protected proof-rig scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from .patch_routing import PatchRoutePlan, route_patch_targets
from .routing import CentralRouter, MethodDetector, RoutePacket


METHOD_TO_METHODOLOGY = {
    "temporal_recall": "temporal_ordinal",
    "symbolic_multi_hop": "symbolic_chain",
    "source_dependency": "dependency_source",
    "repo_patch": "patch_target",
    "user_continuity": "durable_chat_memory",
    "verify": "patch_target",
}
METHODOLOGY_TO_METHOD = {
    "temporal_ordinal": "temporal_recall",
    "symbolic_chain": "symbolic_multi_hop",
    "dependency_source": "source_dependency",
    "patch_target": "repo_patch",
    "durable_chat_memory": "user_continuity",
}
METHODOLOGY_TO_PROOF_RIG = {
    "temporal_ordinal": "MRCR/chat temporal recall",
    "symbolic_chain": "RULER symbolic multi-hop",
    "dependency_source": "LoCoBench source/dependency routing",
    "patch_target": "SWE-bench patch-target routing",
    "durable_chat_memory": "chat durable user/task memory",
}
METHODOLOGY_TO_CAPABILITY = {
    "temporal_ordinal": "temporal ordinal recall",
    "symbolic_chain": "symbolic chain routing",
    "dependency_source": "source/dependency routing",
    "patch_target": "repo patch-target routing",
    "durable_chat_memory": "durable chat/user memory",
}


class MiniRouter(Protocol):
    def route(
        self,
        *,
        method: str,
        prompt: str,
        session_id: str,
        evidence: list[dict[str, Any]],
        max_tokens: int,
    ) -> RoutePacket:
        ...


@dataclass(frozen=True)
class ProductRoutePacket:
    """Route packet enriched with product methodology metadata."""

    method: str
    methodology: str
    capability: str
    proof_rig: str
    selected_windows: list[str]
    memory_family: str
    session_id: str
    tier: str
    route_reason: str
    route_reasons: list[str]
    evidence: list[dict[str, Any]]
    token_cost: int
    activation_score: float = 0.0
    lexical_score: float = 0.0
    ordinal_score: float = 0.0
    recency_score: float = 0.0
    residual_available: bool = False
    kv_ready: bool = False
    selected_paths: list[str] = field(default_factory=list)
    selected_tests: list[str] = field(default_factory=list)
    source_hint_paths: list[str] = field(default_factory=list)
    triad_augmented_paths: list[str] = field(default_factory=list)
    patch_rejected_paths: dict[str, list[str]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ProductRouter:
    """Stable product wrapper around David routing capabilities."""

    def __init__(
        self,
        *,
        router: MiniRouter | None = None,
        detector: MethodDetector | None = None,
        proof_router_available: bool = False,
    ) -> None:
        self._router = router or CentralRouter()
        self._detector = detector or MethodDetector()
        self._proof_router_available = proof_router_available

    def route(
        self,
        prompt: str,
        *,
        session_id: str = "default",
        evidence: Sequence[Mapping[str, Any]] = (),
        files: Mapping[str, str] | None = None,
        windows: Sequence[Mapping[str, Any]] = (),
        path_hints: Sequence[str] = (),
        method: str | None = None,
        methodology: str | None = None,
        max_tokens: int = 4096,
    ) -> ProductRoutePacket:
        selected_method = _resolve_method(method, methodology, self._detector.detect(prompt))
        selected_methodology = METHOD_TO_METHODOLOGY.get(selected_method, "dependency_source")
        patch_plan: PatchRoutePlan | None = None
        evidence_items = _coerce_evidence(evidence)

        if selected_methodology == "patch_target":
            patch_plan = route_patch_targets(
                prompt,
                files=files or {},
                windows=windows,
                path_hints=path_hints,
            )
            evidence_items = [
                *evidence_items,
                *(_coerce_evidence(item.to_dict() for item in patch_plan.evidence)),
            ]

        base = self._route_fail_safe(
            method=selected_method,
            prompt=prompt,
            session_id=session_id,
            evidence=evidence_items,
            max_tokens=max_tokens,
        )
        return _enrich_route_packet(
            base,
            methodology=selected_methodology,
            proof_router_available=self._proof_router_available,
            patch_plan=patch_plan,
        )

    def _route_fail_safe(
        self,
        *,
        method: str,
        prompt: str,
        session_id: str,
        evidence: list[dict[str, Any]],
        max_tokens: int,
    ) -> RoutePacket:
        try:
            return self._router.route(
                method=method,
                prompt=prompt,
                session_id=session_id,
                evidence=evidence,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            fallback = CentralRouter().route(
                method=method,
                prompt=prompt,
                session_id=session_id,
                evidence=evidence,
                max_tokens=max_tokens,
            )
            return RoutePacket(
                **{
                    **fallback.to_json(),
                    "tier": "cold" if not evidence else "warm",
                    "route_reason": "fail-safe product routing used after router error",
                    "provenance": {
                        "router": "david.product_router.fail_safe",
                        "error_type": type(exc).__name__,
                        "proof_router_available": False,
                    },
                }
            )


def route_product(
    prompt: str,
    *,
    session_id: str = "default",
    evidence: Sequence[Mapping[str, Any]] = (),
    files: Mapping[str, str] | None = None,
    windows: Sequence[Mapping[str, Any]] = (),
    path_hints: Sequence[str] = (),
    method: str | None = None,
    methodology: str | None = None,
    max_tokens: int = 4096,
) -> ProductRoutePacket:
    return ProductRouter().route(
        prompt,
        session_id=session_id,
        evidence=evidence,
        files=files,
        windows=windows,
        path_hints=path_hints,
        method=method,
        methodology=methodology,
        max_tokens=max_tokens,
    )


def _resolve_method(method: str | None, methodology: str | None, detected: str) -> str:
    if method:
        return method
    if methodology:
        return METHODOLOGY_TO_METHOD.get(methodology, detected)
    return detected


def _coerce_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    coerced: list[dict[str, Any]] = []
    for item in evidence:
        data = dict(item)
        if "text" not in data:
            data["text"] = str(data.get("reason") or data.get("path") or data)
        coerced.append(data)
    return coerced


def _enrich_route_packet(
    base: RoutePacket,
    *,
    methodology: str,
    proof_router_available: bool,
    patch_plan: PatchRoutePlan | None,
) -> ProductRoutePacket:
    route_reasons = [base.route_reason]
    provenance = {
        **base.provenance,
        "product_router": "chuk_lazarus.david.product_router",
        "methodology": methodology,
        "proof_rig": METHODOLOGY_TO_PROOF_RIG[methodology],
        "proof_router_available": proof_router_available,
        "protected_imports": "not_imported",
    }
    selected_paths: list[str] = []
    selected_tests: list[str] = []
    source_hint_paths: list[str] = []
    triad_augmented_paths: list[str] = []
    rejected_paths: dict[str, list[str]] = {}
    if patch_plan is not None:
        selected_paths = list(patch_plan.selected_paths)
        selected_tests = list(patch_plan.selected_tests)
        source_hint_paths = list(patch_plan.source_hint_paths)
        triad_augmented_paths = list(patch_plan.triad_augmented_paths)
        rejected_paths = {path: list(reasons) for path, reasons in patch_plan.rejected_paths.items()}
        route_reasons.extend(
            reason
            for path in [*selected_paths, *source_hint_paths, *triad_augmented_paths]
            for reason in patch_plan.reasons.get(path, ())
        )
        provenance["patch_plan"] = {
            "selected_path_count": len(selected_paths),
            "selected_test_count": len(selected_tests),
            "rejected_path_count": len(rejected_paths),
        }
    return ProductRoutePacket(
        method=base.method,
        methodology=methodology,
        capability=METHODOLOGY_TO_CAPABILITY[methodology],
        proof_rig=METHODOLOGY_TO_PROOF_RIG[methodology],
        selected_windows=base.selected_windows,
        memory_family=base.memory_family,
        session_id=base.session_id,
        tier=base.tier,
        route_reason=base.route_reason,
        route_reasons=list(dict.fromkeys(route_reasons)),
        evidence=base.evidence,
        token_cost=base.token_cost,
        activation_score=base.activation_score,
        lexical_score=base.lexical_score,
        ordinal_score=base.ordinal_score,
        recency_score=base.recency_score,
        residual_available=base.residual_available,
        kv_ready=base.kv_ready,
        selected_paths=selected_paths,
        selected_tests=selected_tests,
        source_hint_paths=source_hint_paths,
        triad_augmented_paths=triad_augmented_paths,
        patch_rejected_paths=rejected_paths,
        provenance=provenance,
    )
