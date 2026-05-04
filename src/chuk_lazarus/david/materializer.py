"""Safe residual/KV materialization metadata for offline David."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import AdapterSessionMetadata
from .routing import RoutePacket


@dataclass(frozen=True)
class MaterializedContext:
    strategy: str
    text_context: str
    compatibility: dict[str, Any]
    refused: bool = False
    reason: str = "ok"


class Materializer:
    def materialize(self, route: RoutePacket, adapter: AdapterSessionMetadata) -> MaterializedContext:
        route_scope = _route_materialization_scope(route)
        refusal_reasons = _compatibility_refusals(route, adapter, route_scope)
        strategy = _select_strategy(route, adapter, refusal_reasons)
        compatibility = adapter.scope() | {
            "route_memory_family": route.memory_family,
            "route_tier": route.tier,
            "residual_available": route.residual_available,
            "kv_ready": route.kv_ready,
            "route_scope": route_scope,
            "refusal_reasons": refusal_reasons,
            "materialization_safe": not refusal_reasons,
        }
        if refusal_reasons:
            return MaterializedContext(
                strategy="refuse",
                text_context="",
                compatibility=compatibility,
                refused=True,
                reason="; ".join(refusal_reasons),
            )
        return MaterializedContext(
            strategy=strategy,
            text_context="\n".join(route.selected_windows),
            compatibility=compatibility,
        )


def _route_materialization_scope(route: RoutePacket) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    provenance_scope = route.provenance.get("materialization_scope")
    if isinstance(provenance_scope, Mapping):
        scope.update({str(key): value for key, value in provenance_scope.items()})

    for item in route.evidence:
        item_scope = item.get("materialization_scope") or item.get("adapter_scope")
        if isinstance(item_scope, Mapping):
            for key, value in item_scope.items():
                scope.setdefault(str(key), value)

    for key in (
        "model_id",
        "tokenizer_id",
        "adapter_family",
        "model_revision",
        "kv_source_layer",
        "kv_target_layer",
        "insertion_family",
        "memory_family",
    ):
        if key in route.provenance:
            scope.setdefault(key, route.provenance[key])
    return scope


def _compatibility_refusals(
    route: RoutePacket,
    adapter: AdapterSessionMetadata,
    route_scope: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if route.kv_ready:
        if adapter.kv_source_layer is None:
            reasons.append("kv route requested without adapter kv_source_layer")
        if adapter.kv_target_layer is None:
            reasons.append("kv route requested without adapter kv_target_layer")
        if "kv" not in adapter.insertion_family.lower() and "attention" not in adapter.insertion_family.lower():
            reasons.append("kv route requested with non-kv insertion family")
    if route.residual_available and adapter.boundary_layer is None and adapter.kv_source_layer is None:
        reasons.append("residual route requested without adapter residual layer")

    comparisons = {
        "model_id": adapter.model_id,
        "tokenizer_id": adapter.tokenizer_id,
        "adapter_family": adapter.adapter_family,
        "model_revision": adapter.model_revision,
        "kv_source_layer": adapter.kv_source_layer,
        "kv_target_layer": adapter.kv_target_layer,
        "insertion_family": adapter.insertion_family,
    }
    for key, expected in comparisons.items():
        actual = route_scope.get(key)
        if actual is None or expected is None:
            continue
        if str(actual) != str(expected):
            reasons.append(f"{key} mismatch: route={actual} adapter={expected}")

    scoped_memory = route_scope.get("memory_family")
    if scoped_memory is not None and str(scoped_memory) != route.memory_family:
        reasons.append(f"memory_family mismatch: route={scoped_memory} packet={route.memory_family}")
    if route.memory_family not in {"user", "task", "code", "decoder_prior"}:
        reasons.append(f"unsupported memory_family: {route.memory_family}")
    return reasons


def _select_strategy(
    route: RoutePacket,
    adapter: AdapterSessionMetadata,
    refusal_reasons: list[str],
) -> str:
    if refusal_reasons:
        return "refuse"
    if route.kv_ready:
        return "kv_direct"
    if route.residual_available and adapter.boundary_layer is not None:
        return "boundary_residual"
    if route.selected_windows:
        return "boundary_text"
    return "none"
