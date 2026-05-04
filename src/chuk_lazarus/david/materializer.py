"""Safe residual/KV materialization planning for offline David."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import AdapterSessionMetadata
from .materialization_replay import (
    ReplayConsumerInput,
    normalize_replay_consumer,
    replay_consumer_refusals,
    replay_contract_for_strategy,
)
from .residual_store import ResidualSidecarManifest, ResidualStore, ResidualStoreError
from .routing import RoutePacket


@dataclass(frozen=True)
class MaterializedContext:
    strategy: str
    text_context: str
    compatibility: dict[str, Any]
    refused: bool = False
    reason: str = "ok"
    materialization_plan: dict[str, Any] = field(default_factory=dict)


class Materializer:
    def __init__(self, residual_store: ResidualStore | None = None) -> None:
        self.residual_store = residual_store or ResidualStore()

    def materialize(
        self,
        route: RoutePacket,
        adapter: AdapterSessionMetadata,
        replay_consumer: ReplayConsumerInput = None,
    ) -> MaterializedContext:
        consumer = normalize_replay_consumer(replay_consumer)
        route_scope = _route_materialization_scope(route)
        sidecars, sidecar_refusals = _load_sidecars(route, self.residual_store)
        readiness_failures = _artifact_readiness_failures(route, sidecars)
        refusal_reasons = _compatibility_refusals(route, adapter, route_scope)
        refusal_reasons.extend(_sidecar_refusals(route, adapter, sidecars))
        refusal_reasons.extend(sidecar_refusals)
        refusal_reasons.extend(failure["reason"] for failure in readiness_failures)
        requested_strategy = _select_strategy(route, adapter, sidecars, refusal_reasons)
        refusal_reasons.extend(
            replay_consumer_refusals(
                requested_strategy,
                adapter=adapter,
                memory_family=route.memory_family,
                consumer=consumer,
            )
        )
        strategy = "refuse" if refusal_reasons else requested_strategy
        plan = _materialization_plan(
            route,
            adapter,
            strategy,
            sidecars,
            refusal_reasons,
            requested_strategy=requested_strategy,
            replay_consumer=consumer,
            readiness_failures=readiness_failures,
        )
        compatibility = adapter.scope() | {
            "route_memory_family": route.memory_family,
            "route_tier": route.tier,
            "residual_available": route.residual_available,
            "kv_ready": route.kv_ready,
            "route_scope": route_scope,
            "sidecar_count": len(sidecars),
            "sidecar_artifact_ids": [sidecar.artifact_id for sidecar in sidecars],
            "refusal_reasons": refusal_reasons,
            "artifact_readiness_failures": readiness_failures,
            "recapture_plans": [failure["plan"] for failure in readiness_failures],
            "materialization_safe": not refusal_reasons,
            "materialization_plan": plan,
        }
        if refusal_reasons:
            return MaterializedContext(
                strategy="refuse",
                text_context="",
                compatibility=compatibility,
                refused=True,
                reason="; ".join(refusal_reasons),
                materialization_plan=plan,
            )
        return MaterializedContext(
            strategy=strategy,
            text_context="\n".join(route.selected_windows),
            compatibility=compatibility,
            materialization_plan=plan,
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


def _load_sidecars(route: RoutePacket, residual_store: ResidualStore) -> tuple[list[ResidualSidecarManifest], list[str]]:
    sidecars: list[ResidualSidecarManifest] = []
    refusal_reasons: list[str] = []
    for ref in _sidecar_manifest_refs(route):
        try:
            sidecars.append(residual_store.load_manifest(ref))
        except ResidualStoreError as exc:
            refusal_reasons.append(f"sidecar load failed: {exc}")
    return sidecars, refusal_reasons


def _sidecar_manifest_refs(route: RoutePacket) -> list[Any]:
    refs: list[Any] = []
    seen: set[str] = set()
    for ref in _iter_sidecar_refs(route.provenance):
        _append_sidecar_ref(refs, seen, ref)
    for item in route.evidence:
        for ref in _iter_sidecar_refs(item):
            _append_sidecar_ref(refs, seen, ref)
    return refs


_SIDECAR_REF_KEYS = {
    "sidecar_manifest",
    "sidecar_manifests",
    "sidecar_ref",
    "sidecar_refs",
    "residual_sidecar_manifest",
    "residual_sidecar_manifests",
    "residual_sidecar_ref",
    "residual_sidecar_refs",
    "kv_sidecar_manifest",
    "kv_sidecar_manifests",
    "kv_sidecar_ref",
    "kv_sidecar_refs",
}


def _iter_sidecar_refs(value: Any) -> list[Any]:
    refs: list[Any] = []
    if isinstance(value, Mapping):
        if value.get("schema_version") is not None and value.get("refs") is not None:
            refs.append(value)
            return refs
        for key, item in value.items():
            key_text = str(key)
            if key_text in _SIDECAR_REF_KEYS:
                refs.extend(_coerce_sidecar_refs(item))
            elif key_text in {
                "product_route",
                "provenance",
                "route_metadata",
                "index_metadata",
                "index_readiness",
                "source_index",
                "window_metadata",
                "window",
                "windows",
                "selected_windows",
                "materialization",
                "materialization_metadata",
                "artifact_metadata",
            }:
                refs.extend(_iter_sidecar_refs(item))
    elif isinstance(value, list | tuple):
        for item in value:
            refs.extend(_iter_sidecar_refs(item))
    return refs


def _coerce_sidecar_refs(value: Any) -> list[Any]:
    if isinstance(value, list | tuple):
        refs: list[Any] = []
        for item in value:
            refs.extend(_coerce_sidecar_refs(item))
        return refs
    if isinstance(value, Mapping):
        if value.get("schema_version") is not None and value.get("refs") is not None:
            return [value]
        manifest = value.get("manifest")
        if isinstance(manifest, Mapping):
            return [manifest]
        for key in ("manifest_path", "path", "uri"):
            ref = value.get(key)
            if isinstance(ref, str) and ref:
                return [ref]
    return [value]


def _append_sidecar_ref(refs: list[Any], seen: set[str], ref: Any) -> None:
    key = str(ref)
    if key in seen:
        return
    refs.append(ref)
    seen.add(key)


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


def _sidecar_refusals(
    route: RoutePacket,
    adapter: AdapterSessionMetadata,
    sidecars: list[ResidualSidecarManifest],
) -> list[str]:
    reasons: list[str] = []
    for sidecar in sidecars:
        scope = sidecar.scope()
        comparisons = {
            "model_id": adapter.model_id,
            "tokenizer_id": adapter.tokenizer_id,
            "model_revision": adapter.model_revision,
            "adapter_family": adapter.adapter_family,
            "insertion_family": adapter.insertion_family,
            "boundary_layer": adapter.boundary_layer,
            "kv_source_layer": adapter.kv_source_layer,
            "kv_target_layer": adapter.kv_target_layer,
            "hidden_size": adapter.hidden_size,
        }
        for key, expected in comparisons.items():
            actual = scope.get(key)
            if actual is None or expected is None:
                continue
            if str(actual) != str(expected):
                reasons.append(f"sidecar {sidecar.artifact_id} {key} mismatch: sidecar={actual} adapter={expected}")
        if sidecar.memory_family != route.memory_family:
            reasons.append(
                f"sidecar {sidecar.artifact_id} memory_family mismatch: "
                f"sidecar={sidecar.memory_family} route={route.memory_family}"
            )
        if route.residual_available and not any(ref.kind in {"boundary_residual", "residual_stream"} for ref in sidecar.refs):
            reasons.append(f"sidecar {sidecar.artifact_id} has no residual refs")
        if route.kv_ready and not any(ref.kind == "kv_cache" for ref in sidecar.refs):
            reasons.append(f"sidecar {sidecar.artifact_id} has no kv_cache refs")
        for ref in sidecar.refs:
            if ref.kind == "boundary_residual" and adapter.boundary_layer is not None and ref.layer != adapter.boundary_layer:
                reasons.append(
                    f"sidecar {sidecar.artifact_id} boundary ref layer mismatch: "
                    f"ref={ref.layer} adapter={adapter.boundary_layer}"
                )
            if ref.kind == "residual_stream" and sidecar.residual_layer is not None and ref.layer != sidecar.residual_layer:
                reasons.append(
                    f"sidecar {sidecar.artifact_id} residual ref layer mismatch: "
                    f"ref={ref.layer} manifest={sidecar.residual_layer}"
                )
            if ref.kind == "kv_cache" and adapter.kv_target_layer is not None and ref.layer != adapter.kv_target_layer:
                reasons.append(
                    f"sidecar {sidecar.artifact_id} kv ref layer mismatch: "
                    f"ref={ref.layer} adapter={adapter.kv_target_layer}"
                )
    return reasons


def _artifact_readiness_failures(
    route: RoutePacket,
    sidecars: list[ResidualSidecarManifest],
) -> list[dict[str, Any]]:
    requirements = _artifact_requirements(route)
    failures: list[dict[str, Any]] = []
    for artifact in ("activation_routes", "boundary_residuals", "kv_cache"):
        if artifact not in requirements:
            continue
        if _artifact_ready(route, sidecars, artifact):
            continue
        failures.append(
            {
                "artifact": artifact,
                "reason": _artifact_missing_reason(artifact),
                "plan": _artifact_recapture_plan(route, artifact),
            }
        )
    return failures


def _artifact_requirements(route: RoutePacket) -> set[str]:
    requirements: set[str] = set()
    for document in _route_metadata_documents(route):
        for key in ("materialization_requirements", "artifact_requirements"):
            raw = document.get(key)
            if isinstance(raw, Mapping):
                for raw_key, value in raw.items():
                    if value:
                        mapped = _artifact_alias(raw_key)
                        if mapped is not None:
                            requirements.add(mapped)
            elif isinstance(raw, list | tuple | set):
                for item in raw:
                    mapped = _artifact_alias(item)
                    if mapped is not None:
                        requirements.add(mapped)
        raw_required = document.get("required_artifacts")
        if isinstance(raw_required, list | tuple | set):
            for item in raw_required:
                mapped = _artifact_alias(item)
                if mapped is not None:
                    requirements.add(mapped)
        for key, artifact in (
            ("requires_activation_artifact", "activation_routes"),
            ("activation_artifact_required", "activation_routes"),
            ("requires_residual_artifact", "boundary_residuals"),
            ("residual_artifact_required", "boundary_residuals"),
            ("requires_kv_artifact", "kv_cache"),
            ("kv_artifact_required", "kv_cache"),
        ):
            if document.get(key):
                requirements.add(artifact)
        readiness = document.get("artifact_readiness")
        if isinstance(readiness, Mapping):
            for key, value in readiness.items():
                mapped = _artifact_alias(key)
                if mapped is None:
                    continue
                if value is False:
                    requirements.add(mapped)
                elif isinstance(value, Mapping) and value.get("required"):
                    requirements.add(mapped)
    return requirements


def _artifact_ready(
    route: RoutePacket,
    sidecars: list[ResidualSidecarManifest],
    artifact: str,
) -> bool:
    for document in _route_metadata_documents(route):
        readiness = document.get("artifact_readiness")
        if not isinstance(readiness, Mapping):
            continue
        for key, value in readiness.items():
            if _artifact_alias(key) != artifact:
                continue
            if value is True:
                return True
            if isinstance(value, Mapping) and value.get("ready") is True:
                return True
    if artifact == "boundary_residuals":
        return any(any(ref.kind in {"boundary_residual", "residual_stream"} for ref in sidecar.refs) for sidecar in sidecars)
    if artifact == "kv_cache":
        return any(any(ref.kind == "kv_cache" for ref in sidecar.refs) for sidecar in sidecars)
    return _metadata_has_artifact_ref(route, artifact)


def _metadata_has_artifact_ref(route: RoutePacket, artifact: str) -> bool:
    ref_keys = {
        "activation_routes": {"activation_artifact", "activation_ref", "activation_refs", "activation_route_ref"},
        "boundary_residuals": {"residual_artifact", "residual_ref", "residual_refs", "boundary_residual_ref"},
        "kv_cache": {"kv_artifact", "kv_ref", "kv_refs", "kv_cache_ref"},
    }[artifact]
    for document in _route_metadata_documents(route):
        if any(key in document for key in ref_keys):
            return True
    return False


def _route_metadata_documents(route: RoutePacket) -> list[Mapping[str, Any]]:
    documents: list[Mapping[str, Any]] = []
    for value in [route.provenance, *route.evidence]:
        documents.extend(_iter_metadata_documents(value))
    return documents


def _iter_metadata_documents(value: Any) -> list[Mapping[str, Any]]:
    documents: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        documents.append(value)
        for key in (
            "product_route",
            "provenance",
            "route_metadata",
            "index_metadata",
            "index_readiness",
            "source_index",
            "window_metadata",
            "window",
            "windows",
            "selected_windows",
            "materialization",
            "materialization_metadata",
            "artifact_metadata",
        ):
            nested = value.get(key)
            if isinstance(nested, Mapping) or isinstance(nested, list | tuple):
                documents.extend(_iter_metadata_documents(nested))
    elif isinstance(value, list | tuple):
        for item in value:
            documents.extend(_iter_metadata_documents(item))
    return documents


def _artifact_alias(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"activation", "activation_route", "activation_routes", "activation_artifact"}:
        return "activation_routes"
    if text in {
        "residual",
        "residual_stream",
        "boundary_residual",
        "boundary_residuals",
        "residual_artifact",
    }:
        return "boundary_residuals"
    if text in {"kv", "kv_cache", "kv_caches", "kv_direct", "kv_artifact"}:
        return "kv_cache"
    return None


def _artifact_missing_reason(artifact: str) -> str:
    if artifact == "activation_routes":
        return "activation artifact missing: run JIT indexing with activation_routes capture before routing"
    if artifact == "boundary_residuals":
        return "residual artifact missing: recapture HOT windows with boundary_residuals before materialization"
    return "kv artifact missing: recapture HOT windows with kv_cache before KV materialization"


def _artifact_recapture_plan(route: RoutePacket, artifact: str) -> dict[str, Any]:
    if artifact == "activation_routes":
        action = "jit_index_workspace"
        capture = ["activation_routes", "metadata", "provenance"]
    elif artifact == "boundary_residuals":
        action = "recapture_hot_windows"
        capture = ["boundary_residuals", "residual_stream", "metadata", "provenance"]
    else:
        action = "recapture_hot_windows"
        capture = ["kv_cache", "metadata", "provenance"]
    return {
        "action": action,
        "capture": capture,
        "session_id": route.session_id,
        "memory_family": route.memory_family,
        "tier": route.tier,
        "route_reason": route.route_reason,
    }


def _select_strategy(
    route: RoutePacket,
    adapter: AdapterSessionMetadata,
    sidecars: list[ResidualSidecarManifest],
    refusal_reasons: list[str],
) -> str:
    if refusal_reasons:
        return "refuse"
    if route.kv_ready and sidecars:
        return "kv_sidecar"
    if route.kv_ready:
        return "kv_direct"
    if route.residual_available and sidecars:
        return "residual_sidecar"
    if route.residual_available and adapter.boundary_layer is not None:
        return "boundary_residual"
    if route.selected_windows:
        return "boundary_text"
    return "none"


def _materialization_plan(
    route: RoutePacket,
    adapter: AdapterSessionMetadata,
    strategy: str,
    sidecars: list[ResidualSidecarManifest],
    refusal_reasons: list[str],
    *,
    requested_strategy: str,
    replay_consumer: Any,
    readiness_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    replay_contract = replay_contract_for_strategy(
        requested_strategy,
        adapter=adapter,
        memory_family=route.memory_family,
        consumer=replay_consumer,
    )
    replay_state = _plan_replay_state(
        requires_runtime_replay=replay_contract["requires_runtime_replay"],
        refused=bool(refusal_reasons),
    )
    return {
        "version": 1,
        "strategy": strategy,
        "requested_strategy": requested_strategy,
        "replay_family": replay_contract["replay_family"],
        "replay_status": replay_state,
        "guarded": replay_state == "guarded",
        "refused": bool(refusal_reasons),
        "reason": "; ".join(refusal_reasons) if refusal_reasons else "ok",
        "session_id": route.session_id,
        "memory_family": route.memory_family,
        "tier": route.tier,
        "adapter_scope": adapter.scope(),
        "requires_runtime_replay": replay_contract["requires_runtime_replay"],
        "tensor_replay_applied": False,
        "runtime_replay": replay_contract,
        "text_window_count": len(route.selected_windows),
        "sidecars": [sidecar.to_plan_ref() for sidecar in sidecars],
        "replay_refs": _replay_refs(sidecars),
        "readiness_failures": readiness_failures,
        "recapture_plans": [failure["plan"] for failure in readiness_failures],
    }


def _replay_refs(sidecars: list[ResidualSidecarManifest]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for sidecar in sidecars:
        for ordinal, ref in enumerate(sidecar.refs):
            refs.append(
                {
                    "artifact_id": sidecar.artifact_id,
                    "manifest_path": sidecar.manifest_path,
                    "ordinal": ordinal,
                    "kind": ref.kind,
                    "layer": ref.layer,
                    "dtype": ref.dtype,
                    "shape": list(ref.shape),
                    "uri": ref.uri,
                    "inline_value_count": _nested_inline_count(ref.inline_values),
                    "metadata": dict(ref.metadata),
                    "scope": sidecar.scope(),
                }
            )
    return refs


def _nested_inline_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_nested_inline_count(item) for item in value)
    return 1 if value is not None else 0


def _plan_replay_state(*, requires_runtime_replay: bool, refused: bool) -> str:
    if refused:
        return "refused"
    if requires_runtime_replay:
        return "guarded"
    return "not_required"
