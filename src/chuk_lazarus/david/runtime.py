"""Offline David runtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

from .agent_loop import (
    ActionPayload,
    AgentLoopResult,
    AgentLoopState,
    parse_agent_action,
    render_model_action_prompt,
    run_agent_loop as run_agent_loop_core,
)
from .config import AdapterSessionMetadata, DavidConfig
from .decoder import DecoderController, DecoderPlan
from .decoder_prior_store import DecoderPriorProductStore, DecoderPriorScope
from .indexing import IndexReadiness, WorkspaceIndex
from .live_indexer import LiveIndexer, LiveIndexRefresh, load_live_index_state
from .materializer import MaterializedContext, Materializer
from .materialization_replay import ReplayConsumerInput
from .memory import JsonlMemoryStore, MemoryBank
from .model_attestation import ManualModelAttestationResult, verify_model_attestation
from .model_artifacts import is_vindex_artifact_path
from .model_backend import (
    ModelBackend,
    ModelBackendResult,
    OfflineModelBackend,
    TransformersCausalLMBackend,
    VindexArtifactBackend,
)
from .patch_routing import DOC_SUFFIXES, SOURCE_SUFFIXES, is_protected_path, route_patch_targets
from .patching import PatchApplyDiagnostic, apply_patch_candidate, validate_patch_candidate
from .central_router_adapter import CentralRouterAdapter
from .product_router import ProductRoutePacket, ProductRouter
from .resume import SessionSnapshot, load_session_snapshot, save_session_snapshot, summarize_result
from .routing import CentralRouter, MethodDetector, RoutePacket
from .source_index import SourceIndexManifest, load_source_index
from .steering import DecoderSteeringPolicy, TOKEN_FAMILY_MARKERS, build_decoder_logits_processor
from .tools import LocalTools
from .torch_backend import TorchRuntimeModelBackend
from .verifier import VerificationResult, Verifier

try:
    from chuk_lazarus.harness.boot import boot_harness
except ImportError:  # pragma: no cover - only used if harness package is unavailable.
    boot_harness = None


DAVID_RESPONSE_SENTINEL = "David response:"
DEFAULT_GENERATION_CONTEXT_CHARS = 8_192
_LEGACY_GENERATION_INSTRUCTIONS = (
    "Respond with the next concise coding-agent action.",
    "Respond with the next concise coding-agent action",
    "Respond with the next concise coding-",
)


def _materialization_replay_metadata(model_result: ModelBackendResult | None) -> dict[str, Any] | None:
    if model_result is None:
        return None
    replay = model_result.metadata.get("materialization_replay")
    return replay if isinstance(replay, dict) else None


def _clean_runtime_model_text(text: str) -> tuple[str, dict[str, Any]]:
    raw_text = text
    cleaned = text.strip()
    sentinel_index = cleaned.rfind(DAVID_RESPONSE_SENTINEL)
    if sentinel_index >= 0:
        cleaned = cleaned[sentinel_index + len(DAVID_RESPONSE_SENTINEL) :].strip()
    lines = [
        line
        for line in cleaned.splitlines()
        if line.strip() and line.strip() not in _LEGACY_GENERATION_INSTRUCTIONS
    ]
    cleaned = "\n".join(lines).strip()
    metadata = {
        "cleaned": cleaned != raw_text,
        "sentinel": DAVID_RESPONSE_SENTINEL,
    }
    if metadata["cleaned"]:
        metadata["raw_model_text"] = raw_text
    return cleaned, metadata


def _truncate_generation_context(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    marker = "\n[... routed context truncated ...]"
    keep = max(0, max_chars - len(marker))
    return text[:keep].rstrip() + marker, True


def _generation_context_char_budget(max_route_tokens: int) -> int:
    token_budget_chars = max(0, int(max_route_tokens)) * 4
    if token_budget_chars <= 0:
        return DEFAULT_GENERATION_CONTEXT_CHARS
    return min(DEFAULT_GENERATION_CONTEXT_CHARS, token_budget_chars)


def _render_evidence_context(evidence: Sequence[dict[str, Any]]) -> tuple[str, int]:
    blocks: list[str] = []
    included = 0
    for ordinal, item in enumerate(evidence, start=1):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        path = str(item.get("path") or item.get("artifact_id") or f"evidence-{ordinal}")
        kind = str(item.get("kind") or "evidence")
        blocks.append(f"[{ordinal}] {kind}: {path}\n{text}")
        included += 1
    return "\n\n".join(blocks), included


def _unprotected_route_evidence(evidence: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in evidence:
        path = item.get("path") or item.get("artifact_id") or ""
        if path and is_protected_path(str(path).removeprefix("workspace:")):
            continue
        filtered.append(dict(item))
    return filtered


def _bounded_selected_tests(*sources: Sequence[Any], limit: int = 4) -> list[str]:
    tests: list[str] = []
    seen: set[str] = set()
    for source in sources:
        for item in source:
            if isinstance(item, dict):
                raw_values = item.get("selected_tests") or ()
                if isinstance(raw_values, str):
                    raw_values = (raw_values,)
            else:
                raw_values = (item,)
            for raw in raw_values:
                path = str(raw or "").replace("\\", "/").strip()
                while path.startswith("./"):
                    path = path[2:]
                if not path or path in seen or is_protected_path(path):
                    continue
                tests.append(path)
                seen.add(path)
                if len(tests) >= limit:
                    return tests
    return tests


def _bounded_selected_paths(evidence: Sequence[dict[str, Any]], limit: int = 8) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        path = str(item.get("path") or "").strip()
        if not path or path in seen or is_protected_path(path):
            continue
        paths.append(path)
        seen.add(path)
        if len(paths) >= limit:
            return paths
    return paths


def _render_patch_action_guidance(method: str, selected_tests: Sequence[str], patch_context_count: int) -> str:
    if method not in {"repo_patch", "source_dependency"}:
        return ""
    lines = [
        "Patch action context:",
        "- For repo/source tasks, use the patch action when available; otherwise use read/write actions with minimal patch-compatible edits.",
        "- Read target files before editing, keep edits scoped to routed workspace paths, and never touch protected proof-rig benchmark paths.",
    ]
    if selected_tests:
        lines.append(f"- Selected test hints: {', '.join(selected_tests[:4])}.")
    else:
        lines.append("- Selected test hints: none routed; choose the narrowest relevant verification command.")
    lines.append(f"- Patch context entries available: {patch_context_count}.")
    return "\n".join(lines)


def _render_generation_route_guidance(product_route: ProductRoutePacket) -> str:
    route_metadata = _product_route_runtime_metadata(product_route)
    lines = [
        "Product route context:",
        f"- Method: {product_route.method}",
        f"- Methodology: {product_route.methodology}",
        f"- Capability: {product_route.capability}",
        f"- Provenance proof rig: {product_route.proof_rig}",
    ]
    selected_paths = [
        path
        for path in route_metadata.get("selected_paths") or ()
        if path and not is_protected_path(str(path))
    ]
    selected_tests = [
        path
        for path in route_metadata.get("selected_tests") or ()
        if path and not is_protected_path(str(path))
    ]
    route_reasons = [str(reason) for reason in route_metadata.get("route_reasons") or () if reason]
    protected_exclusions = route_metadata.get("protected_path_exclusions") or ()
    provenance = route_metadata.get("provenance") or {}
    lines.append(
        "- Selected paths: "
        + (", ".join(str(path) for path in selected_paths[:8]) if selected_paths else "none routed")
    )
    lines.append(
        "- Selected tests: "
        + (", ".join(str(path) for path in selected_tests[:4]) if selected_tests else "none routed")
    )
    if route_reasons:
        lines.append("- Route reasons: " + " | ".join(route_reasons[:4]))
    if protected_exclusions:
        excluded_paths = [
            str(item.get("path"))
            for item in protected_exclusions
            if isinstance(item, dict) and item.get("path")
        ]
        if excluded_paths:
            lines.append("- Protected proof-rig exclusions: " + ", ".join(excluded_paths[:4]))
    if isinstance(provenance, dict):
        provenance_bits = [
            f"{key}={provenance[key]}"
            for key in ("router", "product_router", "methodology", "proof_rig")
            if key in provenance
        ]
        if provenance_bits:
            lines.append("- Route provenance: " + "; ".join(provenance_bits))
    if product_route.method in {"repo_patch", "source_dependency"}:
        lines.append(
            "- Use selected paths and tests as first-class grounding for the answer; "
            "keep edits and verification scoped to those hints unless evidence says otherwise."
        )
    return "\n".join(lines)


def _protected_path_exclusions(product_route: ProductRoutePacket) -> list[dict[str, Any]]:
    exclusions: dict[str, dict[str, Any]] = {}
    for path, reasons in product_route.patch_rejected_paths.items():
        normalized = str(path or "").replace("\\", "/").strip()
        if not normalized:
            continue
        reason_items = [str(reason) for reason in reasons]
        if is_protected_path(normalized) or any("protected" in reason.lower() for reason in reason_items):
            exclusions[normalized] = {"path": normalized, "reasons": reason_items or ["protected path excluded"]}
    for item in product_route.evidence:
        path = str(item.get("path") or item.get("artifact_id") or "").removeprefix("workspace:")
        normalized = path.replace("\\", "/").strip()
        if normalized and is_protected_path(normalized):
            exclusions.setdefault(
                normalized,
                {"path": normalized, "reasons": ["protected route evidence excluded"]},
            )
    return list(exclusions.values())


def _product_route_runtime_metadata(product_route: ProductRoutePacket) -> dict[str, Any]:
    index_readiness = product_route.provenance.get("index_readiness")
    source_index = product_route.provenance.get("source_index")
    live_index_refresh = product_route.provenance.get("live_index_refresh")
    capture_metadata = product_route.provenance.get("capture_metadata")
    return {
        "method": product_route.method,
        "methodology": product_route.methodology,
        "capability": product_route.capability,
        "proof_rig": product_route.proof_rig,
        "selected_paths": list(product_route.selected_paths),
        "selected_tests": list(product_route.selected_tests),
        "selected_symbols": list(product_route.selected_symbols),
        "import_tokens": list(product_route.import_tokens),
        "dependency_source_hints": list(product_route.dependency_source_hints),
        "source_index_paths": list(product_route.source_index_paths),
        "source_hint_paths": list(product_route.source_hint_paths),
        "triad_augmented_paths": list(product_route.triad_augmented_paths),
        "route_reasons": list(product_route.route_reasons),
        "evidence_count": len(product_route.evidence),
        "protected_path_exclusions": _protected_path_exclusions(product_route),
        "index_readiness": dict(index_readiness) if isinstance(index_readiness, dict) else None,
        "source_index": dict(source_index) if isinstance(source_index, dict) else None,
        "live_index_refresh": dict(live_index_refresh) if isinstance(live_index_refresh, dict) else None,
        "capture_metadata": dict(capture_metadata) if isinstance(capture_metadata, dict) else None,
        "provenance": dict(product_route.provenance),
    }


def _text_windows_from_evidence(evidence: Sequence[dict[str, Any]], *, limit: int = 3) -> list[str]:
    windows: list[str] = []
    for item in evidence:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        windows.append(text)
        if len(windows) >= limit:
            break
    return windows


def _route_packet_from_product_route(
    product_route: ProductRoutePacket,
    *,
    fallback: RoutePacket | None = None,
) -> RoutePacket:
    """Keep ProductRoutePacket authoritative while preserving RoutePacket consumers."""

    safe_evidence = _unprotected_route_evidence(product_route.evidence)
    if not safe_evidence and fallback is not None:
        safe_evidence = _unprotected_route_evidence(fallback.evidence)
    selected_windows = _text_windows_from_evidence(safe_evidence) or list(product_route.selected_windows)
    if not selected_windows and fallback is not None:
        selected_windows = list(fallback.selected_windows)
    product_metadata = _product_route_runtime_metadata(product_route)
    fallback_provenance = fallback.provenance if fallback is not None else {}
    provenance = {
        **fallback_provenance,
        **product_route.provenance,
        "product_methodology": product_route.methodology,
        "product_capability": product_route.capability,
        "product_proof_rig": product_route.proof_rig,
        "selected_paths": list(product_route.selected_paths),
        "selected_tests": list(product_route.selected_tests),
        "route_reasons": list(product_route.route_reasons),
        "protected_path_exclusions": product_metadata["protected_path_exclusions"],
        "product_route": product_metadata,
    }
    return RoutePacket(
        method=product_route.method or (fallback.method if fallback is not None else ""),
        selected_windows=selected_windows,
        memory_family=product_route.memory_family or (fallback.memory_family if fallback is not None else "task"),
        session_id=product_route.session_id or (fallback.session_id if fallback is not None else "default"),
        tier=product_route.tier or (fallback.tier if fallback is not None else "cold"),
        route_reason=product_route.route_reason or (fallback.route_reason if fallback is not None else "product route"),
        evidence=safe_evidence,
        token_cost=product_route.token_cost,
        activation_score=max(fallback.activation_score if fallback is not None else 0.0, product_route.activation_score),
        lexical_score=max(fallback.lexical_score if fallback is not None else 0.0, product_route.lexical_score),
        ordinal_score=max(fallback.ordinal_score if fallback is not None else 0.0, product_route.ordinal_score),
        recency_score=max(fallback.recency_score if fallback is not None else 0.0, product_route.recency_score),
        residual_available=product_route.residual_available or (fallback.residual_available if fallback is not None else False),
        kv_ready=product_route.kv_ready or (fallback.kv_ready if fallback is not None else False),
        provenance=provenance,
    )


def _bridge_product_route_packet(route: RoutePacket, product_route: ProductRoutePacket) -> RoutePacket:
    """Compatibility wrapper for older tests/callers."""

    return _route_packet_from_product_route(product_route, fallback=route)


def _route_runtime_metadata(route: RoutePacket) -> dict[str, Any]:
    product = route.provenance.get("product_route")
    if isinstance(product, dict):
        return dict(product)
    return {
        "method": route.method,
        "methodology": route.provenance.get("product_methodology"),
        "capability": route.provenance.get("product_capability"),
        "proof_rig": route.provenance.get("product_proof_rig"),
        "selected_paths": list(route.provenance.get("selected_paths") or ()),
        "selected_tests": list(route.provenance.get("selected_tests") or ()),
        "route_reasons": list(route.provenance.get("route_reasons") or ()),
        "evidence_count": len(route.evidence),
        "protected_path_exclusions": list(route.provenance.get("protected_path_exclusions") or ()),
        "index_readiness": route.provenance.get("index_readiness"),
        "source_index": route.provenance.get("source_index"),
        "live_index_refresh": route.provenance.get("live_index_refresh"),
        "capture_metadata": route.provenance.get("capture_metadata"),
        "provenance": dict(route.provenance),
    }


def _materialized_with_route_metadata(
    materialized: MaterializedContext,
    route: RoutePacket,
) -> MaterializedContext:
    route_metadata = _route_runtime_metadata(route)
    compatibility = {
        **materialized.compatibility,
        "route_metadata": route_metadata,
        "route_provenance": dict(route.provenance),
    }
    plan = {
        **materialized.materialization_plan,
        "route_metadata": route_metadata,
        "route_provenance": dict(route.provenance),
    }
    return MaterializedContext(
        strategy=materialized.strategy,
        text_context=materialized.text_context,
        compatibility=compatibility,
        refused=materialized.refused,
        reason=materialized.reason,
        materialization_plan=plan,
    )


def _decoder_with_route_metadata(decoder: DecoderPlan, route: RoutePacket) -> DecoderPlan:
    route_metadata = _route_runtime_metadata(route)
    constraints = {
        **decoder.constraints,
        "route_metadata": route_metadata,
    }
    prior_scope = dict(decoder.prior_scope)
    methodology = route_metadata.get("methodology")
    if methodology:
        prior_scope.setdefault("methodology", methodology)
    return DecoderPlan(constraints=constraints, prior_scope=prior_scope)


def _scope_value(scope: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        value = scope.get(name)
        if value is not None and value != "":
            return value
    return None


def _adapter_decoder_prior_layer(adapter: AdapterSessionMetadata) -> int:
    for value in (adapter.kv_target_layer, adapter.boundary_layer, adapter.route_layer):
        if value is not None:
            return int(value)
    return 0


def _decoder_prior_scope_from_plan(
    decoder: DecoderPlan,
    adapter: AdapterSessionMetadata,
) -> DecoderPriorScope:
    prior_scope = dict(decoder.prior_scope)
    steering = decoder.constraints.get("steering")
    steering_scope = steering if isinstance(steering, dict) else {}
    layer = _scope_value(prior_scope, "layer", "kv_target_layer", "injection_layer", "boundary_layer", "route_layer")
    task_type = (
        _scope_value(prior_scope, "task_type")
        or steering_scope.get("task_type")
        or prior_scope.get("method")
        or "unknown"
    )
    steering_version = _scope_value(prior_scope, "steering_version") or steering_scope.get("policy") or "unknown"
    return DecoderPriorScope(
        model_id=str(_scope_value(prior_scope, "model_id") or adapter.model_id),
        tokenizer_id=str(_scope_value(prior_scope, "tokenizer_id") or adapter.tokenizer_id),
        adapter_family=str(_scope_value(prior_scope, "adapter_family") or adapter.adapter_family),
        layer=int(layer) if layer is not None else _adapter_decoder_prior_layer(adapter),
        task_type=str(task_type),
        steering_version=str(steering_version),
        model_revision=str(_scope_value(prior_scope, "model_revision") or adapter.model_revision),
        adapter_config_id=str(_scope_value(prior_scope, "adapter_config_id") or ""),
        insertion_family=str(_scope_value(prior_scope, "insertion_family") or adapter.insertion_family),
    )


def _empty_prior_lookup_metadata(*, refused_reason: str | None = None) -> dict[str, Any]:
    return {
        "attempted": False,
        "hit": False,
        "miss_reason": None,
        "refused_reason": refused_reason,
        "scope": None,
        "seed_alpha": None,
        "applied_seed_alpha": None,
    }


def _decoder_temptation_event(decoder: DecoderPlan, generated_text: str) -> bool:
    if not generated_text:
        return False
    steering = decoder.constraints.get("steering")
    if not isinstance(steering, dict):
        return False
    lowered = generated_text.lower()
    for family in steering.get("forbidden_token_families") or ():
        for marker in TOKEN_FAMILY_MARKERS.get(str(family), ()):
            marker_text = str(marker).lower()
            if marker_text and marker_text in lowered:
                return True
    return False


def _decoder_prior_for_capability_verifier(decoder_prior: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(decoder_prior, dict):
        return decoder_prior
    verifier_prior = dict(decoder_prior)
    scope = verifier_prior.pop("scope", None)
    if isinstance(scope, dict):
        verifier_prior["product_scope"] = scope
    return verifier_prior


def _index_runtime_metadata(
    readiness: IndexReadiness,
    *,
    source_index: SourceIndexManifest | None,
    live_index_refresh: dict[str, Any] | None,
) -> dict[str, Any]:
    capture = []
    if readiness.jit_plan is not None:
        capture = list(readiness.jit_plan.get("capture") or ())
    return {
        "index_readiness": {
            "ready": readiness.ready,
            "required": readiness.required,
            "manifest_path": str(readiness.manifest_path),
            "reason": readiness.reason,
            "jit_plan": readiness.jit_plan,
            "capture": capture,
        },
        "source_index": None
        if source_index is None
        else {
            "schema": source_index.schema,
            "indexed_at": source_index.indexed_at,
            "source_index_path": None,
            "workspace_root": source_index.workspace_root,
            "file_count": len(source_index.files),
            "truncated": source_index.truncated,
            "adapter_scope": dict(source_index.adapter_scope),
            "paths": [record.path for record in source_index.files],
        },
        "live_index_refresh": live_index_refresh,
        "capture_metadata": {
            "capture": capture,
            "source_index_ready": source_index is not None,
            "live_refresh_ready": live_index_refresh is not None,
            "activation_routes_expected": "activation_routes" in capture,
            "boundary_residuals_expected": "boundary_residuals" in capture,
            "kv_cache_expected": "kv_cache" in capture or "kv_caches" in capture,
        },
    }


def _product_route_with_index_metadata(
    product_route: ProductRoutePacket,
    index_metadata: dict[str, Any],
) -> ProductRoutePacket:
    provenance = {
        **product_route.provenance,
        "index_readiness": index_metadata["index_readiness"],
        "source_index": index_metadata["source_index"],
        "live_index_refresh": index_metadata["live_index_refresh"],
        "capture_metadata": index_metadata["capture_metadata"],
    }
    return replace(product_route, provenance=provenance)


def _route_has_materialization_metadata(route: RoutePacket) -> bool:
    return bool(route.residual_available or route.kv_ready or _metadata_has_materialization_handles(route.provenance))


def _product_route_has_materialization_metadata(product_route: ProductRoutePacket) -> bool:
    return bool(
        product_route.residual_available
        or product_route.kv_ready
        or _metadata_has_materialization_handles(product_route.provenance)
    )


def _metadata_has_materialization_handles(metadata: dict[str, Any]) -> bool:
    if any(
        key in metadata
        for key in (
            "materialization_scope",
            "materialization_requirements",
            "artifact_requirements",
            "required_artifacts",
            "sidecar_manifest",
            "sidecar_manifests",
            "sidecar_ref",
            "sidecar_refs",
            "residual_sidecar_manifest",
            "residual_sidecar_manifests",
            "kv_sidecar_manifest",
            "kv_sidecar_manifests",
        )
    ):
        return True
    product = metadata.get("product_route")
    return isinstance(product, dict) and _metadata_has_materialization_handles(product)


@dataclass(frozen=True)
class RuntimeResult:
    prompt: str
    method: str
    answer: str
    index: IndexReadiness
    route: RoutePacket
    materialized: MaterializedContext
    decoder: DecoderPlan
    verification: VerificationResult
    writeback: dict[str, Any]
    product_route: ProductRoutePacket | None = None
    model_result: ModelBackendResult | None = None
    source_index: dict[str, Any] | None = None
    live_index_refresh: dict[str, Any] | None = None
    decoder_prior: dict[str, Any] | None = None
    resume_snapshot: dict[str, Any] | None = None
    harness_session: dict[str, Any] | None = None
    writeback_verification: VerificationResult | None = None

    def to_json(self) -> dict[str, Any]:
        data = {
            "prompt": self.prompt,
            "method": self.method,
            "answer": self.answer,
            "index": {
                "ready": self.index.ready,
                "required": self.index.required,
                "manifest_path": str(self.index.manifest_path),
                "reason": self.index.reason,
                "jit_plan": self.index.jit_plan,
            },
            "route": self.route.to_json(),
            "materialized": {
                "strategy": self.materialized.strategy,
                "text_context": self.materialized.text_context,
                "compatibility": self.materialized.compatibility,
                "refused": self.materialized.refused,
                "reason": self.materialized.reason,
                "materialization_plan": self.materialized.materialization_plan,
                "materialization_replay": _materialization_replay_metadata(self.model_result),
            },
            "decoder": {
                "constraints": self.decoder.constraints,
                "prior_scope": self.decoder.prior_scope,
            },
            "verification": self.verification.to_json(),
            "writeback": self.writeback,
        }
        if self.writeback_verification is not None:
            data["writeback_verification"] = self.writeback_verification.to_json()
        if self.product_route is not None:
            data["product_route"] = self.product_route.to_json()
        if self.model_result is not None:
            data["model_result"] = {
                "text": self.model_result.text,
                "backend": self.model_result.backend,
                "ok": self.model_result.ok,
                "error": self.model_result.error,
                "metadata": self.model_result.metadata,
            }
        if self.source_index is not None:
            data["source_index"] = self.source_index
        if self.live_index_refresh is not None:
            data["live_index_refresh"] = self.live_index_refresh
        if self.decoder_prior is not None:
            data["decoder_prior"] = self.decoder_prior
        if self.resume_snapshot is not None:
            data["resume_snapshot"] = self.resume_snapshot
        if self.harness_session is not None:
            data["harness_session"] = self.harness_session
        return data


@dataclass(frozen=True)
class RuntimeAgentLoopResult:
    prompt: str
    loop: AgentLoopResult
    writeback: dict[str, Any] | None = None
    resume_snapshot: dict[str, Any] | None = None
    repair_summary: dict[str, Any] | None = None

    @property
    def answer(self) -> str:
        summary = self._repair_summary()
        verification = summary["verification"]["outcome"]
        return (
            f"agent_loop status={self.loop.status}; steps={self.loop.steps}; "
            f"verified={self.loop.verified}; failures={summary['failure_count']}; "
            f"repairs={summary['repair_attempt_count']}; retries={summary['retry_count']}; "
            f"verification={verification}; reason={self.loop.reason}"
        )

    @property
    def ok(self) -> bool:
        return self.loop.ok

    def _repair_summary(self) -> dict[str, Any]:
        return self.repair_summary or _agent_loop_repair_summary(self.loop)

    def to_json(self) -> dict[str, Any]:
        data = {
            "prompt": self.prompt,
            "answer": self.answer,
            "loop": self.loop.to_dict(),
            "repair_summary": self._repair_summary(),
            "writeback": self.writeback,
            "resume_snapshot": self.resume_snapshot,
        }
        return data


@dataclass(frozen=True)
class RuntimeQualityGateCandidate:
    name: str
    command: tuple[str, ...]
    reason: str
    source: str = "runtime.discovery"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "display_command": _display_command(self.command),
            "reason": self.reason,
            "source": self.source,
        }


def _display_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def _candidate_gate(
    name: str,
    command: Sequence[str],
    reason: str,
    *,
    source: str = "runtime.discovery",
) -> RuntimeQualityGateCandidate:
    return RuntimeQualityGateCandidate(
        name=name,
        command=tuple(str(part) for part in command),
        reason=reason,
        source=source,
    )


def _coerce_quality_gate_candidates(value: Any, *, source: str) -> list[RuntimeQualityGateCandidate]:
    if value is None:
        return []
    raw_items = value
    if isinstance(value, dict):
        raw_items = value.get("candidates") or value.get("gates") or value.get("commands") or []
    if isinstance(raw_items, (str, bytes)) or not isinstance(raw_items, Sequence):
        return []

    candidates: list[RuntimeQualityGateCandidate] = []
    for index, item in enumerate(raw_items, start=1):
        if isinstance(item, RuntimeQualityGateCandidate):
            candidates.append(item)
            continue
        to_json = getattr(item, "to_json", None)
        if callable(to_json):
            item = to_json()
        if isinstance(item, dict):
            raw_command = item.get("command") or item.get("argv") or item.get("args")
            if isinstance(raw_command, str):
                command: tuple[str, ...] = (raw_command,)
            elif isinstance(raw_command, Sequence) and not isinstance(raw_command, (bytes, bytearray)):
                command = tuple(str(part) for part in raw_command if str(part))
            else:
                command = ()
            if not command:
                continue
            candidates.append(
                RuntimeQualityGateCandidate(
                    name=str(item.get("name") or item.get("id") or f"candidate-{index}"),
                    command=command,
                    reason=str(item.get("reason") or item.get("description") or "quality gate candidate"),
                    source=str(item.get("source") or source),
                )
            )
    return candidates


def _quality_gates_module_candidates(workspace_root: Path) -> list[RuntimeQualityGateCandidate]:
    try:
        from . import quality_gates  # type: ignore[attr-defined]
    except ImportError:
        return []

    for name in (
        "discover_quality_gates",
        "discover_candidate_gates",
        "discover_gates",
        "candidate_gates",
    ):
        discover = getattr(quality_gates, name, None)
        if not callable(discover):
            continue
        try:
            raw = discover(workspace_root=workspace_root)
        except TypeError:
            raw = discover(workspace_root)
        return _coerce_quality_gate_candidates(raw, source=f"quality_gates.{name}")
    return []


def _safe_read_text(path: Path, *, max_bytes: int = 256_000) -> str:
    try:
        if path.stat().st_size > max_bytes:
            return ""
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _python_quality_gate_candidates(workspace_root: Path) -> list[RuntimeQualityGateCandidate]:
    markers = (
        "pyproject.toml",
        "pytest.ini",
        "tox.ini",
        "setup.cfg",
        "setup.py",
    )
    has_python_marker = any((workspace_root / marker).exists() for marker in markers)
    has_tests_dir = (workspace_root / "tests").is_dir()
    if not (has_python_marker or has_tests_dir):
        return []
    reason_parts = []
    if has_python_marker:
        reason_parts.append("Python project metadata found")
    if has_tests_dir:
        reason_parts.append("tests/ directory found")
    return [
        _candidate_gate(
            "pytest",
            ("python", "-m", "pytest", "-q"),
            "; ".join(reason_parts),
        )
    ]


def _node_quality_gate_candidates(workspace_root: Path) -> list[RuntimeQualityGateCandidate]:
    package_json = workspace_root / "package.json"
    if not package_json.exists():
        return []
    try:
        package = json.loads(_safe_read_text(package_json) or "{}")
    except json.JSONDecodeError:
        return [
            _candidate_gate(
                "npm-test",
                ("npm", "test"),
                "package.json found, but scripts could not be parsed",
            )
        ]
    scripts = package.get("scripts") if isinstance(package, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}
    candidates: list[RuntimeQualityGateCandidate] = []
    test_script = str(scripts.get("test") or "").strip()
    if test_script and "no test specified" not in test_script.lower():
        candidates.append(
            _candidate_gate(
                "npm-test",
                ("npm", "test"),
                "package.json defines a test script",
            )
        )
    for script_name, command in (
        ("lint", ("npm", "run", "lint")),
        ("typecheck", ("npm", "run", "typecheck")),
        ("check", ("npm", "run", "check")),
    ):
        if scripts.get(script_name):
            candidates.append(
                _candidate_gate(
                    f"npm-{script_name}",
                    command,
                    f"package.json defines a {script_name} script",
                )
            )
    if not candidates:
        candidates.append(
            _candidate_gate(
                "npm-test",
                ("npm", "test"),
                "package.json found; no explicit test-like scripts discovered",
            )
        )
    return candidates


def _makefile_quality_gate_candidates(workspace_root: Path) -> list[RuntimeQualityGateCandidate]:
    for filename in ("Makefile", "makefile"):
        makefile = workspace_root / filename
        if not makefile.exists():
            continue
        text = _safe_read_text(makefile)
        if "\ntest:" in f"\n{text}" or "\ncheck:" in f"\n{text}":
            target = "test" if "\ntest:" in f"\n{text}" else "check"
            return [
                _candidate_gate(
                    f"make-{target}",
                    ("make", target),
                    f"{filename} defines a {target} target",
                )
            ]
    return []


def _bounded_builtin_quality_gate_candidates(workspace_root: Path) -> list[RuntimeQualityGateCandidate]:
    candidates: list[RuntimeQualityGateCandidate] = []
    candidates.extend(_python_quality_gate_candidates(workspace_root))
    candidates.extend(_node_quality_gate_candidates(workspace_root))
    if (workspace_root / "Cargo.toml").exists():
        candidates.append(_candidate_gate("cargo-test", ("cargo", "test"), "Cargo.toml found"))
    if (workspace_root / "go.mod").exists():
        candidates.append(_candidate_gate("go-test", ("go", "test", "./..."), "go.mod found"))
    candidates.extend(_makefile_quality_gate_candidates(workspace_root))

    unique: dict[tuple[str, ...], RuntimeQualityGateCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.command, candidate)
    return list(unique.values())


def _quality_gate_evidence(candidates: Sequence[RuntimeQualityGateCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "quality_gate_candidate",
            "name": candidate.name,
            "command": list(candidate.command),
            "display_command": _display_command(candidate.command),
            "reason": candidate.reason,
            "source": candidate.source,
        }
        for candidate in candidates
    ]


def _builtin_verification_result(
    *,
    capability: str,
    candidates: Sequence[RuntimeQualityGateCandidate],
    selected: Sequence[RuntimeQualityGateCandidate],
    run_reason: str,
) -> VerificationResult:
    checks = {
        "candidate_quality_gates": {
            "ok": True,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "commands_run": 0,
            "run_reason": run_reason,
            "discovered_commands": [_display_command(candidate.command) for candidate in candidates],
            "selected_commands": [_display_command(candidate.command) for candidate in selected],
        }
    }
    reason = "candidate quality gates discovered" if candidates else "no candidate quality gates discovered"
    return VerificationResult(
        ok=True,
        capability=capability,
        evidence=_quality_gate_evidence(candidates),
        reason=reason,
        checks=checks,
    )


def _format_builtin_verify_report(
    *,
    workspace_root: Path,
    mode: str,
    verification: VerificationResult,
    candidates: Sequence[RuntimeQualityGateCandidate],
    selected: Sequence[RuntimeQualityGateCandidate],
    run_reason: str,
) -> str:
    lines = [
        "David verification",
        f"mode: {mode}",
        f"workspace: {workspace_root}",
        "selected commands:",
    ]
    if selected:
        lines.extend(f"- {_display_command(candidate.command)} ({candidate.reason})" for candidate in selected)
    else:
        lines.append("- none")

    lines.append("discovered candidate gates:")
    if candidates:
        lines.extend(
            f"- {candidate.name}: {_display_command(candidate.command)} ({candidate.reason})"
            for candidate in candidates
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "commands run:",
            "- none",
            f"reason: {run_reason}",
            "verification metadata:",
            f"- capability: {verification.capability}",
            f"- ok: {str(verification.ok).lower()}",
            f"- reason: {verification.reason}",
            f"- candidate_count: {len(candidates)}",
            f"- selected_count: {len(selected)}",
            "- command_count: 0",
        ]
    )
    return "\n".join(lines)


def _agent_loop_repair_summary(loop: AgentLoopResult) -> dict[str, Any]:
    """Summarize repair/retry signals from current and future loop trace metadata."""

    loop_json = loop.to_dict()
    trace = [step.to_dict() for step in loop.trace]
    failed_steps = [_agent_loop_step_failure(step) for step in trace if not bool(step.get("ok"))]
    first_failed_step = failed_steps[0]["step"] if failed_steps else None
    repair_steps = [
        _agent_loop_step_reference(step)
        for step in trace
        if (
            first_failed_step is not None
            and int(step.get("step") or 0) > first_failed_step
            and bool(step.get("ok"))
            and step.get("action") not in {"verify", "done", "no_action"}
        )
    ]
    verification_step = next((step for step in reversed(trace) if step.get("action") == "verify"), None)
    verification_passed = bool(
        verification_step.get("observation", {}).get("passed")
        if isinstance(verification_step, dict)
        else loop.verified
    )
    if verification_step is None:
        verification_outcome = "not_run"
    elif verification_passed:
        verification_outcome = "passed"
    else:
        verification_outcome = "failed"

    explicit_repair_count = _first_nested_count(
        {key: value for key, value in loop_json.items() if key != "trace"},
        {
            "repair_attempt_count",
            "repair_attempts",
            "repair_count",
            "repairs",
            "recovery_attempt_count",
            "recovery_attempts",
        },
    )
    explicit_retry_count = _first_nested_count(
        {key: value for key, value in loop_json.items() if key != "trace"},
        {"retry_count", "retries", "retry_attempt_count", "retry_attempts"},
    )
    recovered = bool(failed_steps and loop.ok)
    if recovered:
        recovery_outcome = "verified_after_failure"
    elif failed_steps:
        recovery_outcome = "unrecovered_failure"
    else:
        recovery_outcome = "no_failure"
    return {
        "schema": "david.agent_loop.repair_summary.v1",
        "source": "agent_loop.trace",
        "status": loop.status,
        "reason": loop.reason,
        "trace_step_count": len(trace),
        "failure_count": len(failed_steps),
        "failed_step_count": len(failed_steps),
        "failed_steps": failed_steps,
        "repair_attempt_count": explicit_repair_count if explicit_repair_count is not None else len(repair_steps),
        "repair_steps": repair_steps,
        "retry_count": explicit_retry_count if explicit_retry_count is not None else 0,
        "recovered": recovered,
        "recovery_outcome": recovery_outcome,
        "verification": {
            "status": loop.status,
            "verified": loop.verified,
            "outcome": verification_outcome,
            "passed": verification_passed,
            "step": verification_step.get("step") if isinstance(verification_step, dict) else None,
            "reason": loop.reason,
        },
    }


def _agent_loop_step_reference(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step.get("step"),
        "action": step.get("action"),
        "ok": bool(step.get("ok")),
    }


def _agent_loop_step_failure(step: dict[str, Any]) -> dict[str, Any]:
    observation = step.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    failure = _agent_loop_step_reference(step)
    error = observation.get("error") or observation.get("stderr") or observation.get("reason")
    if error:
        failure["error"] = str(error)
    if "returncode" in observation:
        failure["returncode"] = observation["returncode"]
    requested_action = observation.get("requested_action")
    if requested_action:
        failure["requested_action"] = requested_action
    return failure


def _first_nested_count(value: Any, names: set[str]) -> int | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in names:
                count = _nonnegative_int(item)
                if count is not None:
                    return count
        for item in value.values():
            count = _first_nested_count(item, names)
            if count is not None:
                return count
    return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


class DavidRuntime:
    def __init__(self, config: DavidConfig) -> None:
        self.config = config
        self.boot_errors: list[str] = []
        self.product_router_errors: list[str] = []
        self.harness_session = self._boot_session()
        self.adapter = self._adapter_from_harness() or config.adapter
        self.model_attestation = self._verify_model_attestation()
        self.index = WorkspaceIndex(config.workspace_root, self._index_manifest_path(), self.adapter)
        self.memory = MemoryBank(
            JsonlMemoryStore(config.user_memory_path, "user"),
            JsonlMemoryStore(config.task_memory_path, "task"),
        )
        self.detector = MethodDetector()
        self.router = CentralRouter()
        product_router = self._product_central_router()
        self.product_router = ProductRouter(
            router=product_router or self.router,
            detector=self.detector,
            proof_router_available=product_router is not None,
        )
        self.materializer = Materializer()
        self.decoder = DecoderController()
        self.tools = LocalTools(config.workspace_root)
        self.verifier = Verifier(self.tools)
        self.backend = self._create_backend()
        self.source_index_path = config.state_dir / "indexes" / f"{self._adapter_file_stem()}-source.json"
        self.live_index_state_path = config.state_dir / "indexes" / f"{self._adapter_file_stem()}-live-source-state.json"
        self.latest_live_index_refresh: LiveIndexRefresh | None = None
        self.decoder_prior_path = config.state_dir / "decoder_priors.json"
        self.decoder_prior_store = DecoderPriorProductStore.load(self.decoder_prior_path)
        self.resume_path = config.state_dir / "resume.json"
        self.resume_snapshot = self._load_resume_snapshot()
        self.auto_jit_summary = self.jit_index() if self.config.auto_jit_index else None

    @classmethod
    def create(cls, config: DavidConfig) -> "DavidRuntime":
        return cls(config)

    def run_once(self, prompt: str, *, verify_command: Sequence[str] | None = None) -> RuntimeResult:
        readiness = self.index.check()
        if readiness.required and self.config.auto_jit_index:
            self.jit_index()
            readiness = self.index.check()
        source_index = self._ensure_source_index() if self.config.auto_jit_index else self._loaded_source_index()
        live_index_refresh = self._latest_live_index_refresh_json()
        index_metadata = _index_runtime_metadata(
            readiness,
            source_index=source_index,
            live_index_refresh=live_index_refresh,
        )

        method = self.detector.detect(prompt)
        workspace_files = self._workspace_text_files() if method in {"repo_patch", "source_dependency"} else {}
        evidence = self._recall(method, prompt, workspace_files=workspace_files)
        fallback_route: RoutePacket | None = None
        try:
            product_route = self.product_router.route(
                prompt,
                session_id=self.config.session_id,
                evidence=evidence,
                files=workspace_files or None,
                source_index=source_index,
                method=method,
                max_tokens=self.config.max_route_tokens,
            )
        except Exception as exc:
            self.product_router_errors.append(f"product route failed: {type(exc).__name__}: {exc}")
            fallback_route = self.router.route(
                method=method,
                prompt=prompt,
                session_id=self.config.session_id,
                evidence=evidence,
                max_tokens=self.config.max_route_tokens,
            )
            product_route = ProductRouter(router=CentralRouter(), detector=self.detector).route(
                prompt,
                session_id=self.config.session_id,
                evidence=fallback_route.evidence,
                files=workspace_files or None,
                source_index=source_index,
                method=fallback_route.method,
                max_tokens=self.config.max_route_tokens,
            )
        if fallback_route is None and not _product_route_has_materialization_metadata(product_route):
            try:
                fallback_candidate = self.router.route(
                    method=method,
                    prompt=prompt,
                    session_id=self.config.session_id,
                    evidence=evidence,
                    max_tokens=self.config.max_route_tokens,
                )
            except Exception as exc:
                self.product_router_errors.append(f"offline materialization fallback failed: {type(exc).__name__}: {exc}")
            else:
                if _route_has_materialization_metadata(fallback_candidate):
                    fallback_route = fallback_candidate
        product_route = _product_route_with_index_metadata(product_route, index_metadata)
        route = _route_packet_from_product_route(product_route, fallback=fallback_route)
        replay_consumer = self._backend_replay_consumer()
        materialized = self.materializer.materialize(route, self.adapter, replay_consumer=replay_consumer)
        materialized = _materialized_with_route_metadata(materialized, route)
        decoder = self.decoder.plan(route=route, adapter=self.adapter, session_id=self.config.session_id)
        decoder = _decoder_with_route_metadata(decoder, route)
        model_result = self._generate(prompt, method, product_route, materialized, decoder, replay_consumer)
        verification = self.verifier.verify(
            capability=method,
            evidence=evidence,
            command=verify_command if method == "verify" else None,
            metadata=self._verification_metadata(
                route=route,
                materialized=materialized,
                decoder=decoder,
                product_route=product_route,
                model_result=model_result,
            ),
        )
        answer = model_result.text if self.config.model_path and model_result.ok and model_result.text else self._answer(
            prompt, method, readiness, route, materialized, verification, model_result
        )
        decoder_prior = self._update_decoder_prior(method, decoder, verification, model_result)
        writeback_metadata = {
            "provenance": "david.runtime.run_once",
            **self._verification_metadata(
                route=route,
                materialized=materialized,
                decoder=decoder,
                product_route=product_route,
                model_result=model_result,
                decoder_prior=decoder_prior,
            ),
            "verification": verification.to_json(),
            "harness_session_id": getattr(self.harness_session, "session_id", None),
            "live_index_refresh": live_index_refresh,
            "index_metadata": index_metadata,
        }
        artifact = self.memory.writeback(
            method=method,
            user_id=self.config.user_id,
            session_id=self.config.session_id,
            text=f"Prompt: {prompt}\nAnswer: {answer}",
            metadata=writeback_metadata,
        )
        writeback_verification = self.verifier.verify(
            capability=method,
            evidence=evidence,
            metadata={
                **writeback_metadata,
                "decoder_prior": _decoder_prior_for_capability_verifier(decoder_prior),
                "writeback": artifact.to_json(),
            },
        )
        result = RuntimeResult(
            prompt=prompt,
            method=method,
            answer=answer,
            index=readiness,
            route=route,
            materialized=materialized,
            decoder=decoder,
            verification=verification,
            writeback=artifact.to_json(),
            product_route=product_route,
            model_result=model_result,
            source_index=source_index.to_json() if source_index is not None else None,
            live_index_refresh=live_index_refresh,
            decoder_prior=decoder_prior,
            harness_session=self._harness_session_json(),
            writeback_verification=writeback_verification,
        )
        snapshot = self._save_resume_snapshot(result)
        return RuntimeResult(
            **{
                **result.__dict__,
                "resume_snapshot": snapshot.to_json(),
            }
        )

    def run_agent_loop(
        self,
        prompt: str | Sequence[ActionPayload],
        *,
        max_steps: int = 8,
        persist: bool = True,
    ) -> RuntimeAgentLoopResult:
        requests = self._agent_loop_requests(prompt)
        original_prompt = prompt if isinstance(prompt, str) else repr(list(prompt))
        if requests is None:
            loop = run_agent_loop_core(
                self._model_driven_agent_step(original_prompt),
                self.tools,
                max_steps=max_steps,
                objective=original_prompt,
                mode="model_driven",
            )
        else:
            loop = run_agent_loop_core(
                requests,
                self.tools,
                max_steps=max_steps,
                objective=original_prompt,
                mode="explicit",
            )
        repair_summary = _agent_loop_repair_summary(loop)
        writeback: dict[str, Any] | None = None
        if persist:
            artifact = self.memory.writeback(
                method="agent_loop",
                user_id=self.config.user_id,
                session_id=self.config.session_id,
                text=(
                    f"Agent loop: {original_prompt}\n"
                    f"Status: {loop.status}\n"
                    f"Reason: {loop.reason}\n"
                    "Repair summary: "
                    f"failures={repair_summary['failure_count']} "
                    f"repairs={repair_summary['repair_attempt_count']} "
                    f"retries={repair_summary['retry_count']} "
                    f"verification={repair_summary['verification']['outcome']}"
                ),
                metadata={
                    "provenance": "david.runtime.run_agent_loop",
                    "adapter": self.adapter.scope(),
                    "adapter_scope": self.adapter.scope(),
                    "loop": loop.to_dict(),
                    "repair_summary": repair_summary,
                    "failure_recovery": repair_summary,
                    "verification_outcome": repair_summary["verification"],
                    "harness_session_id": getattr(self.harness_session, "session_id", None),
                    "live_index_refresh": self._latest_live_index_refresh_json(),
                },
            )
            writeback = artifact.to_json()
        result = RuntimeAgentLoopResult(
            prompt=original_prompt,
            loop=loop,
            writeback=writeback,
            repair_summary=repair_summary,
        )
        snapshot = self._save_resume_snapshot(result)
        return RuntimeAgentLoopResult(
            prompt=original_prompt,
            loop=loop,
            writeback=writeback,
            resume_snapshot=snapshot.to_json(),
            repair_summary=repair_summary,
        )

    def agent_loop(
        self,
        prompt: str | Sequence[ActionPayload],
        *,
        max_steps: int = 8,
        persist: bool = True,
    ) -> RuntimeAgentLoopResult:
        return self.run_agent_loop(prompt, max_steps=max_steps, persist=persist)

    def readiness(self) -> dict[str, str]:
        index = self.index.check()
        backend = self.backend.status()
        source = self._loaded_source_index()
        return {
            "model validation": self._model_validation_status(),
            "backend": f"{backend.name}: {'loaded' if backend.loaded else backend.reason}",
            "index": "ready" if index.ready else f"missing: {index.reason}",
            "source index": self._source_index_status(source),
            "memory": self._memory_readiness(),
            "resume": "ready" if self.resume_snapshot is not None else "missing",
            "workspace": str(self.config.workspace_root),
        }

    def memory_status(self) -> str:
        user_count = len(self.memory.user.all())
        task_count = len(self.memory.task.all())
        return (
            "memory:\n"
            f"- user artifact: {self.config.user_memory_path} ({user_count} records)\n"
            f"- task artifact: {self.config.task_memory_path} ({task_count} records)"
        )

    def index_status(self) -> str:
        readiness = self.index.check()
        source = self._loaded_source_index()
        source_line = (
            f"source index: {self._source_index_status(source)} at {self.source_index_path}"
            if source is not None
            else f"source index: missing at {self.source_index_path}"
        )
        if readiness.ready:
            return f"index: ready at {readiness.manifest_path}\n{source_line}"
        return f"index: {readiness.reason}\nplan: {readiness.jit_plan}\n{source_line}"

    def discover_verify_gates(self) -> list[RuntimeQualityGateCandidate]:
        candidates = _quality_gates_module_candidates(self.config.workspace_root)
        if candidates:
            return candidates
        return _bounded_builtin_quality_gate_candidates(self.config.workspace_root)

    def verify_patch(self) -> str:
        return self._builtin_verify_report(
            mode="built-in patch verification",
            capability="repo_patch",
            select_candidates=True,
        )

    def verify(self, command: str | None = None) -> str:
        command = command or getattr(self.config, "verify_command", None)
        if not command:
            return self._builtin_verify_report(
                mode="candidate discovery",
                capability="verify",
                select_candidates=False,
            )
        shell = self.run_shell(command)
        return shell

    def _builtin_verify_report(
        self,
        *,
        mode: str,
        capability: str,
        select_candidates: bool,
    ) -> str:
        candidates = self.discover_verify_gates()
        selected = list(candidates) if select_candidates else []
        if selected:
            run_reason = "built-in verification selected candidate gates but did not execute them without --cmd"
        elif candidates:
            run_reason = "candidate gates were discovered but not executed without --cmd"
        else:
            run_reason = "no candidate quality gates were discovered in the workspace root"
        verification = _builtin_verification_result(
            capability=capability,
            candidates=candidates,
            selected=selected,
            run_reason=run_reason,
        )
        return _format_builtin_verify_report(
            workspace_root=self.config.workspace_root,
            mode=mode,
            verification=verification,
            candidates=candidates,
            selected=selected,
            run_reason=run_reason,
        )

    def jit_index(self) -> str:
        self.index.jit()
        readiness = self.index.check()
        refresh = self._refresh_source_index()
        return (
            f"index: {'ready' if readiness.ready else readiness.reason} at {readiness.manifest_path}\n"
            f"source index: {refresh.file_count} files at {self.source_index_path}\n"
            "source refresh: "
            f"changed={len(refresh.changed_paths)} indexed={len(refresh.indexed_paths)} "
            f"deleted={len(refresh.deleted_paths)} manifest_sha256={refresh.manifest_sha256}"
        )

    def build_index(self) -> str:
        return self.jit_index()

    def refresh_index(self) -> str:
        return self.jit_index()

    def run_shell(self, command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=self.config.workspace_root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=getattr(self.config, "command_timeout_seconds", None) or 30,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip() or "(no output)"
        return f"$ {command}\nrc={completed.returncode}\n{output}"

    def apply_patch(self, patch_text: str) -> PatchApplyDiagnostic:
        diagnostic = validate_patch_candidate(patch_text)
        if diagnostic.failures:
            return diagnostic
        return apply_patch_candidate(self.config.workspace_root, patch_text)

    def read_file(self, path: str) -> str:
        return self.tools.read(path)

    def write_file(self, spec: str) -> str:
        path, separator, content = spec.partition(" ")
        if not separator or not path:
            return "write: expected /write <path> <content>"
        written = self.tools.write(path, content)
        return f"wrote {written.relative_to(self.config.workspace_root)}"

    def _agent_loop_requests(self, prompt: str | Sequence[ActionPayload]) -> Sequence[ActionPayload] | None:
        if not isinstance(prompt, str):
            return prompt

        stripped = prompt.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return [prompt]

        try:
            if parse_agent_action(stripped) is not None:
                return [prompt]
        except Exception:
            return [prompt]
        return None

    def _model_driven_agent_step(self, objective: str):
        action_context = self._model_action_workspace_context(objective)

        def next_action(state: AgentLoopState) -> ActionPayload:
            model_prompt = render_model_action_prompt(state)
            patch_guidance = action_context.get("patch_guidance") or ""
            if patch_guidance:
                model_prompt = f"{model_prompt}\n{patch_guidance}"
            if action_context["included"]:
                model_prompt = (
                    f"{model_prompt}\n"
                    "Routed workspace context:\n"
                    f"{action_context['text']}\n"
                    "Use this bounded workspace context as file/path evidence when choosing read, write, or verify actions."
                )
            model_result = self.backend.generate(model_prompt, max_new_tokens=320)
            if not model_result.ok:
                raise RuntimeError(f"backend refused action generation: {model_result.error or 'unknown error'}")

            raw_text = model_result.text.strip()
            try:
                action = parse_agent_action(raw_text)
            except Exception:
                return raw_text
            if action is None:
                return raw_text
            payload = action.to_dict()
            payload["_model_provenance"] = {
                "backend": model_result.backend,
                "ok": model_result.ok,
                "error": model_result.error,
                "metadata": model_result.metadata,
                "prompt": model_prompt,
                "objective": objective,
                "workspace_context": action_context["metadata"],
            }
            payload["_model_text"] = raw_text
            return payload

        return next_action

    def _model_action_workspace_context(self, objective: str) -> dict[str, Any]:
        method = self.detector.detect(objective)
        budget = _generation_context_char_budget(self.config.max_route_tokens)
        metadata: dict[str, Any] = {
            "source": "none",
            "method": method,
            "context_char_count": 0,
            "context_char_budget": budget,
            "workspace_file_count": 0,
            "evidence_count_available": 0,
            "evidence_count_included": 0,
            "selected_paths": [],
            "selected_tests": [],
            "protected_path_exclusions": [],
            "methodology": None,
            "capability": None,
            "proof_rig": None,
            "route_reasons": [],
            "route_provenance": {},
            "patch_context_count": 0,
            "product_route_available": False,
            "omitted": False,
            "omitted_reason": None,
            "truncated": False,
        }
        if method not in {"repo_patch", "source_dependency"}:
            metadata.update({"omitted": True, "omitted_reason": "non_workspace_method"})
            return {"included": False, "text": "", "metadata": metadata}

        workspace_files = self._workspace_text_files(max_files=40, max_bytes=6_000)
        metadata["workspace_file_count"] = len(workspace_files)
        evidence = self._recall(method, objective, workspace_files=workspace_files)
        metadata["evidence_count_available"] = len(evidence)
        product_route: ProductRoutePacket | None = None
        try:
            product_route = self.product_router.route(
                objective,
                session_id=self.config.session_id,
                evidence=evidence,
                files=workspace_files or None,
                method=method,
                max_tokens=self.config.max_route_tokens,
            )
        except Exception:
            product_route = None
        route_evidence = product_route.evidence if product_route is not None else evidence
        safe_evidence = _unprotected_route_evidence(route_evidence)
        route_metadata = _product_route_runtime_metadata(product_route) if product_route is not None else {}
        selected_tests = _bounded_selected_tests(
            product_route.selected_tests if product_route is not None else (),
            safe_evidence,
        )
        patch_context_count = len(safe_evidence[:8])
        metadata["evidence_count_available"] = len(route_evidence)
        metadata["selected_paths"] = list(route_metadata.get("selected_paths") or _bounded_selected_paths(safe_evidence))
        metadata["selected_tests"] = selected_tests
        metadata["protected_path_exclusions"] = list(route_metadata.get("protected_path_exclusions") or ())
        metadata["methodology"] = route_metadata.get("methodology")
        metadata["capability"] = route_metadata.get("capability")
        metadata["proof_rig"] = route_metadata.get("proof_rig")
        metadata["route_reasons"] = list(route_metadata.get("route_reasons") or ())
        metadata["route_provenance"] = dict(route_metadata.get("provenance") or {})
        metadata["patch_context_count"] = patch_context_count
        metadata["product_route_available"] = product_route is not None
        patch_guidance = _render_patch_action_guidance(method, selected_tests, patch_context_count)
        raw_context, evidence_included = _render_evidence_context(safe_evidence[:8])
        metadata["evidence_count_included"] = evidence_included
        if not raw_context:
            metadata.update({"omitted": True, "omitted_reason": "no_context_available"})
            return {"included": False, "text": "", "metadata": metadata, "patch_guidance": patch_guidance}

        text, truncated = _truncate_generation_context(raw_context, budget)
        metadata.update(
            {
                "source": "workspace_route_evidence",
                "context_char_count": len(text),
                "truncated": truncated,
            }
        )
        return {"included": True, "text": text, "metadata": metadata, "patch_guidance": patch_guidance}

    def _recall(
        self,
        method: str,
        prompt: str,
        *,
        workspace_files: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        if method == "symbolic_multi_hop":
            return self.memory.symbolic_chain(prompt)
        if method == "temporal_recall":
            latest = self.memory.user.temporal(prompt, ordinal="latest")
            return [latest] if latest else []
        evidence = self.memory.recall_for_method(method, prompt)
        if method in {"repo_patch", "source_dependency"}:
            evidence = [*evidence, *self._workspace_route_evidence(prompt, workspace_files=workspace_files)]
        return evidence

    def _workspace_route_evidence(
        self,
        prompt: str,
        *,
        workspace_files: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        files = workspace_files if workspace_files is not None else self._workspace_text_files()
        if not files:
            return []
        plan = route_patch_targets(prompt, files=files, limit=8)
        evidence: list[dict[str, Any]] = []
        for ordinal, item in enumerate(plan.evidence):
            windows = plan.windows_by_path.get(item.path, ())
            text = "\n".join(str(window.get("text") or "")[:2_000] for window in windows)
            evidence.append(
                {
                    "artifact_id": f"workspace:{item.path}",
                    "family": "task",
                    "kind": "patch_target",
                    "path": item.path,
                    "text": f"{item.path}\n{text}".rstrip(),
                    "timestamp": "workspace-scan",
                    "score": item.score,
                    "ordinal": ordinal,
                    "provenance": "david.patch_routing",
                    "route_reason": item.reason,
                    "classification": item.kind,
                    "selected_tests": list(plan.selected_tests),
                }
            )
        return evidence

    def _workspace_text_files(self, *, max_files: int = 80, max_bytes: int = 12_000) -> dict[str, str]:
        root = self.config.workspace_root
        skipped_dirs = {
            ".git",
            ".david",
            ".chuk_lazarus",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            ".venv",
            "venv",
            "dist",
            "build",
            "target",
        }
        allowed_suffixes = SOURCE_SUFFIXES | DOC_SUFFIXES
        files: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in skipped_dirs)
            for filename in sorted(filenames):
                if len(files) >= max_files:
                    return files
                path = Path(dirpath) / filename
                try:
                    rel = path.relative_to(root)
                except ValueError:
                    continue
                rel_parts = set(rel.parts)
                rel_text = rel.as_posix()
                if rel_parts & skipped_dirs or is_protected_path(rel_text):
                    continue
                if path.suffix.lower() not in allowed_suffixes:
                    continue
                try:
                    if path.stat().st_size > max_bytes:
                        text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]
                    else:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                files[rel_text] = text
        return files

    def _memory_readiness(self) -> str:
        user_exists = self.config.user_memory_path.exists()
        task_exists = self.config.task_memory_path.exists()
        if user_exists and task_exists:
            return "ready"
        missing = []
        if not user_exists:
            missing.append("user")
        if not task_exists:
            missing.append("task")
        return "missing: " + ", ".join(missing)

    def _boot_session(self) -> Any | None:
        if boot_harness is None:
            self.boot_errors.append("harness boot module unavailable")
            return None
        if not (self.config.model_path or self.config.validation_report_path):
            return None
        require_validated_model = self.config.require_validated_model
        if self.config.model_attestation_path and self.config.require_validated_model:
            require_validated_model = False
        try:
            return boot_harness(
                model_path=self.config.model_path or "offline-deterministic",
                workspace_path=str(self.config.workspace_root),
                validation_report_path=self.config.validation_report_path,
                require_validated_model=require_validated_model,
            )
        except Exception as exc:
            self.boot_errors.append(f"{type(exc).__name__}: {exc}")
            return None

    def _adapter_from_harness(self) -> AdapterSessionMetadata | None:
        session = self.harness_session
        model_adapter = getattr(session, "model_adapter", None)
        if model_adapter is None:
            return None
        return AdapterSessionMetadata(
            model_id=str(model_adapter.model_identity),
            tokenizer_id=str(model_adapter.tokenizer_identity),
            model_revision=str(model_adapter.model_revision or "validated"),
            adapter_family=str(model_adapter.adapter_family),
            hidden_size=model_adapter.hidden_size,
            route_layer=model_adapter.route_layer_candidate,
            route_query_head=model_adapter.route_query_head_candidate,
            boundary_layer=model_adapter.boundary_layer_candidate,
            kv_source_layer=model_adapter.kv_source_layer_candidate,
            kv_target_layer=model_adapter.kv_target_layer_candidate,
            insertion_family=str(model_adapter.insertion_family or "text_only"),
            memory_family="david-runtime",
        )

    def _create_backend(self) -> ModelBackend:
        if self.config.model_path and is_vindex_artifact_path(self.config.model_path):
            return VindexArtifactBackend(self.config.model_path)
        can_auto_load = bool(getattr(self.harness_session, "can_auto_load", False))
        can_standard_decode = bool(
            self.config.require_validated_model
            and (can_auto_load or self._manual_review_standard_decode_allowed())
        )
        if self.config.model_path and can_standard_decode and not self.boot_errors:
            backend_selector = str(self.config.model_backend or "transformers").strip().lower()
            if backend_selector in {"torch-runtime", "torch_runtime", "torch"}:
                return TorchRuntimeModelBackend(
                    self.config.model_path,
                    local_files_only=True,
                    device=self.config.model_device,
                    torch_dtype=self.config.model_dtype,
                )
            if backend_selector not in {"transformers", "transformers-causal-lm", "hf"}:
                self.boot_errors.append(f"unsupported model backend selector: {self.config.model_backend}")
                return OfflineModelBackend(prefix="david")
            return TransformersCausalLMBackend(
                self.config.model_path,
                local_files_only=True,
                device=self.config.model_device,
                torch_dtype=self.config.model_dtype,
            )
        return OfflineModelBackend(prefix="david")

    def _product_central_router(self) -> CentralRouterAdapter | None:
        try:
            return CentralRouterAdapter.from_stable_wrapper_if_available()
        except Exception as exc:
            self.product_router_errors.append(f"{type(exc).__name__}: {exc}")
            return None

    def _model_validation_status(self) -> str:
        if self.boot_errors:
            return "blocked: " + "; ".join(self.boot_errors)
        if self._manual_review_standard_decode_allowed():
            return f"manual_reviewed ({self.adapter.adapter_family}:{self.adapter.model_id}; standard_decode_only)"
        if self.harness_session is not None:
            status = getattr(self.harness_session, "validation_status", None) or "unknown"
            return f"{status} ({self.adapter.adapter_family}:{self.adapter.model_id})"
        return f"offline shell mode ({self.adapter.adapter_family}:{self.adapter.model_id})"

    def _verify_model_attestation(self) -> ManualModelAttestationResult | None:
        if not self.config.model_attestation_path:
            return None
        result = verify_model_attestation(
            self.config.model_attestation_path,
            validation_report_path=self.config.validation_report_path,
            model_path=self.config.model_path,
        )
        can_auto_load = bool(getattr(self.harness_session, "can_auto_load", False))
        if self.config.require_validated_model and not can_auto_load and not result.standard_decode_allowed:
            self.boot_errors.append(f"model attestation invalid: {result.reason}")
        return result

    def _manual_review_standard_decode_allowed(self) -> bool:
        return bool(
            self.config.require_validated_model
            and self.model_attestation is not None
            and self.model_attestation.standard_decode_allowed
        )

    def _index_manifest_path(self) -> Path:
        return self.config.state_dir / "indexes" / f"{self._adapter_file_stem()}.json"

    def _adapter_file_stem(self) -> str:
        safe_model = self.adapter.model_id.replace("/", "_").replace("\\", "_")
        safe_revision = self.adapter.model_revision.replace("/", "_").replace("\\", "_")
        return f"{safe_model}-{safe_revision}"

    def _loaded_source_index(self) -> SourceIndexManifest | None:
        try:
            return load_source_index(self.source_index_path)
        except (OSError, ValueError):
            return None

    def _ensure_source_index(self) -> SourceIndexManifest:
        existing = self._loaded_source_index()
        if existing is not None and existing.adapter_scope == self.adapter.scope():
            return existing
        refresh = self._refresh_source_index()
        manifest = self._loaded_source_index()
        if manifest is None:
            raise RuntimeError(f"live source index refresh did not write {refresh.source_index_path}")
        return manifest

    def _refresh_source_index(self) -> LiveIndexRefresh:
        refresh = LiveIndexer(
            self.config.workspace_root,
            self.adapter.scope(),
            state_path=self.live_index_state_path,
            source_index_path=self.source_index_path,
            max_files=120,
            max_file_bytes=128_000,
        ).refresh()
        self.latest_live_index_refresh = refresh
        return refresh

    def _latest_live_index_refresh_json(self) -> dict[str, Any] | None:
        if self.latest_live_index_refresh is None:
            return None
        return self.latest_live_index_refresh.to_session_refresh_handle()

    def _source_index_status(self, source: SourceIndexManifest | None = None) -> str:
        source = self._loaded_source_index() if source is None else source
        if source is None:
            return "missing"
        parts = [f"ready ({len(source.files)} files)"]
        if source.adapter_scope != self.adapter.scope():
            parts.append("stale: adapter scope mismatch")
        elif self.live_index_state_path.exists():
            try:
                state = load_live_index_state(self.live_index_state_path)
                parts.append(f"live indexed_at={state.indexed_at or 'unknown'}")
            except (OSError, ValueError):
                parts.append("live state unreadable")
        else:
            parts.append("live state missing")
        if self.latest_live_index_refresh is not None:
            parts.append(f"manifest_sha256={self.latest_live_index_refresh.manifest_sha256}")
        elif source.indexed_at:
            parts.append(f"source indexed_at={source.indexed_at}")
        if source.truncated:
            parts.append("truncated")
        return "; ".join(parts)

    def _load_resume_snapshot(self) -> SessionSnapshot | None:
        try:
            return load_session_snapshot(self.resume_path)
        except (OSError, ValueError):
            return None

    def _save_resume_snapshot(self, result: Any) -> SessionSnapshot:
        live_index_refresh = getattr(result, "live_index_refresh", None)
        summary_input: Any = (
            result.answer
            if live_index_refresh is None
            else {"answer": result.answer, "live_index_refresh": live_index_refresh}
        )
        snapshot = SessionSnapshot(
            session_id=self.config.session_id,
            workspace=str(self.config.workspace_root),
            adapter_scope=self.adapter.scope(),
            memory_paths={
                "user": str(self.config.user_memory_path),
                "task": str(self.config.task_memory_path),
                "decoder_prior": str(self.decoder_prior_path),
                "source_index": str(self.source_index_path),
                "live_index_state": str(self.live_index_state_path),
            },
            last_result_summary=summarize_result(summary_input),
        )
        self.resume_snapshot = save_session_snapshot(snapshot, self.resume_path)
        return self.resume_snapshot

    def _generate(
        self,
        prompt: str,
        method: str,
        product_route: ProductRoutePacket,
        materialized: MaterializedContext,
        decoder: DecoderPlan,
        replay_consumer: ReplayConsumerInput = None,
    ) -> ModelBackendResult:
        context = self._generation_context(product_route, materialized)
        context_block = ""
        if context["included"]:
            context_block = (
                "Routed context:\n"
                f"{context['text']}\n"
                "Use the routed context above as grounding evidence. If it is insufficient, say what is missing.\n"
            )
        generation_prompt = (
            "You are David, a terminal coding agent operating inside the user's workspace.\n"
            "Use the routed methodology and evidence to produce the smallest useful next answer.\n"
            f"Task: {prompt}\n"
            f"Evidence count: {len(product_route.evidence)}\n"
            f"{_render_generation_route_guidance(product_route)}\n"
            f"{context_block}"
            "Answer only in the response slot below. Do not repeat this prompt, labels, or instructions.\n"
            f"{DAVID_RESPONSE_SENTINEL}"
        )
        steering = self._decoder_steering_processor(decoder)
        model_result = self.backend.generate(
            generation_prompt,
            max_new_tokens=self.config.model_max_new_tokens,
            logits_processor=steering["processors"],
            materialization_plan=materialized.materialization_plan,
            replay_consumer=replay_consumer,
        )
        cleaned_text, postprocess_metadata = _clean_runtime_model_text(model_result.text)
        return ModelBackendResult(
            text=cleaned_text,
            backend=model_result.backend,
            ok=model_result.ok,
            error=model_result.error,
            metadata={
                **model_result.metadata,
                "decoder_steering": steering["metadata"],
                "generation_context": context["metadata"],
                "answer_postprocess": postprocess_metadata,
            },
        )

    def _generation_context(
        self,
        product_route: ProductRoutePacket,
        materialized: MaterializedContext,
    ) -> dict[str, Any]:
        route_metadata = _product_route_runtime_metadata(product_route)
        metadata: dict[str, Any] = {
            "context_char_count": 0,
            "context_char_budget": _generation_context_char_budget(self.config.max_route_tokens),
            "evidence_count_available": len(product_route.evidence),
            "evidence_count_included": 0,
            "omitted": False,
            "omitted_reason": None,
            "refused": materialized.refused,
            "truncated": False,
            "source": "none",
            "methodology": product_route.methodology,
            "capability": product_route.capability,
            "selected_paths": route_metadata["selected_paths"],
            "selected_tests": route_metadata["selected_tests"],
            "protected_path_exclusions": route_metadata["protected_path_exclusions"],
            "route_reasons": route_metadata["route_reasons"],
            "route_provenance": route_metadata["provenance"],
            "index_readiness": route_metadata.get("index_readiness"),
            "source_index": route_metadata.get("source_index"),
            "live_index_refresh": route_metadata.get("live_index_refresh"),
            "capture_metadata": route_metadata.get("capture_metadata"),
        }
        if materialized.refused:
            metadata.update({"omitted": True, "omitted_reason": "materializer_refused"})
            return {"included": False, "text": "", "metadata": metadata}

        raw_context = materialized.text_context.strip()
        if raw_context:
            metadata["source"] = "materialized.text_context"
            metadata["evidence_count_included"] = len(product_route.evidence)
        else:
            raw_context, evidence_included = _render_evidence_context(product_route.evidence)
            metadata["source"] = "product_route.evidence" if raw_context else "none"
            metadata["evidence_count_included"] = evidence_included

        if not raw_context:
            metadata.update({"omitted": True, "omitted_reason": "no_context_available"})
            return {"included": False, "text": "", "metadata": metadata}

        text, truncated = _truncate_generation_context(raw_context, int(metadata["context_char_budget"]))
        metadata["context_char_count"] = len(text)
        metadata["truncated"] = truncated
        return {"included": True, "text": text, "metadata": metadata}

    def _backend_replay_consumer(self) -> ReplayConsumerInput:
        reporter = getattr(self.backend, "replay_consumer_capabilities", None)
        if not callable(reporter):
            return None
        try:
            return reporter(self.adapter)
        except Exception:
            return None

    def _decoder_prior_lookup(
        self,
        decoder: DecoderPlan,
    ) -> tuple[DecoderPriorScope | None, Any | None, dict[str, Any]]:
        metadata = _empty_prior_lookup_metadata()
        metadata["attempted"] = True
        try:
            scope = _decoder_prior_scope_from_plan(decoder, self.adapter)
        except Exception as exc:
            metadata["refused_reason"] = f"{type(exc).__name__}: {exc}"
            return None, None, metadata
        metadata["scope"] = scope.to_json()
        try:
            record = self.decoder_prior_store.get(scope)
        except Exception as exc:
            metadata["refused_reason"] = f"{type(exc).__name__}: {exc}"
            return scope, None, metadata
        if record is None:
            metadata["miss_reason"] = "no compatible prior record"
            return scope, None, metadata
        metadata.update(
            {
                "hit": True,
                "miss_reason": None,
                "seed_alpha": record.seed_alpha,
            }
        )
        return scope, record, metadata

    def _decoder_steering_processor(self, decoder: DecoderPlan) -> dict[str, Any]:
        steering = decoder.constraints.get("steering")
        metadata: dict[str, Any] = {
            "attempted": bool(steering),
            "applied": False,
            "processor_count": 0,
            "refused_reason": None,
            "prior_lookup": _empty_prior_lookup_metadata(),
        }
        if isinstance(steering, dict):
            metadata.update(
                {
                    "target_language": steering.get("target_language"),
                    "task_type": steering.get("task_type"),
                    "logit_lock": bool(steering.get("logit_lock")),
                    "forbidden_token_families": list(steering.get("forbidden_token_families") or ()),
                }
            )
        else:
            metadata["refused_reason"] = "decoder plan has no steering metadata"
            metadata["prior_lookup"]["refused_reason"] = "decoder plan has no steering metadata"
            return {"processors": None, "metadata": metadata}

        _prior_scope, prior_record, prior_lookup = self._decoder_prior_lookup(decoder)
        metadata["prior_lookup"] = prior_lookup
        if prior_lookup.get("refused_reason"):
            metadata["refused_reason"] = f"decoder prior lookup refused: {prior_lookup['refused_reason']}"
            return {"processors": None, "metadata": metadata}

        if not isinstance(self.backend, TransformersCausalLMBackend):
            metadata["refused_reason"] = "backend does not expose tokenizer for live steering"
            return {"processors": None, "metadata": metadata}

        status = self.backend.load()
        if not status.loaded:
            metadata["refused_reason"] = f"backend not loaded: {status.reason}"
            return {"processors": None, "metadata": metadata}
        tokenizer = self.backend.tokenizer
        if tokenizer is None:
            metadata["refused_reason"] = "backend tokenizer unavailable"
            return {"processors": None, "metadata": metadata}

        alpha_bounds = steering.get("alpha_bounds") if isinstance(steering.get("alpha_bounds"), dict) else {}
        try:
            policy = DecoderSteeringPolicy(
                task_type=str(
                    steering.get("task_type")
                    or decoder.prior_scope.get("task_type")
                    or decoder.prior_scope.get("method")
                    or "unknown"
                ),
                target_language=str(steering.get("target_language") or "unknown"),
                forbidden_token_families=tuple(str(item) for item in steering.get("forbidden_token_families") or ()),
                alpha_min=float(alpha_bounds.get("min", 0.0)),
                alpha_max=float(alpha_bounds.get("max", 0.0)),
                logit_lock=bool(steering.get("logit_lock")),
                steering_version=str(
                    steering.get("policy")
                    or decoder.prior_scope.get("steering_version")
                    or "unknown"
                ),
            )
            prior_alpha = prior_record.seed_alpha if prior_record is not None else None
            processor = build_decoder_logits_processor(
                policy=policy,
                adapter=self.adapter,
                tokenizer=tokenizer,
                alpha=prior_alpha,
                scope=decoder.prior_scope,
            )
        except Exception as exc:
            metadata["refused_reason"] = f"{type(exc).__name__}: {exc}"
            return {"processors": None, "metadata": metadata}

        applied_alpha = processor.spec.alpha
        if prior_record is not None:
            metadata["prior_lookup"]["applied_seed_alpha"] = applied_alpha
        metadata.update(
            {
                "applied": True,
                "processor_count": 1,
                "refused_reason": None,
                "forbidden_token_count": len(processor.forbidden_token_ids),
                "alpha": applied_alpha,
                "prior_seed_alpha_applied": prior_record is not None,
            }
        )
        return {"processors": [processor], "metadata": metadata}

    def _verification_metadata(
        self,
        *,
        route: RoutePacket,
        materialized: MaterializedContext,
        decoder: DecoderPlan,
        product_route: ProductRoutePacket,
        model_result: ModelBackendResult,
        decoder_prior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        materialized_json = {
            "strategy": materialized.strategy,
            "text_context": materialized.text_context,
            "compatibility": (
                materialized.compatibility if route.evidence or materialized.strategy != "none" else None
            ),
            "refused": materialized.refused,
            "reason": materialized.reason,
            "materialization_plan": materialized.materialization_plan,
        }
        materialization_replay = _materialization_replay_metadata(model_result)
        if materialization_replay is not None:
            materialized_json["materialization_replay"] = materialization_replay
        route_metadata = _route_runtime_metadata(route)
        return {
            "adapter": self.adapter.scope(),
            "adapter_scope": self.adapter.scope(),
            "route": route.to_json(),
            "route_metadata": route_metadata,
            "route_provenance": dict(route.provenance),
            "methodology": route_metadata.get("methodology"),
            "selected_paths": list(route_metadata.get("selected_paths") or ()),
            "selected_tests": list(route_metadata.get("selected_tests") or ()),
            "protected_path_exclusions": list(route_metadata.get("protected_path_exclusions") or ()),
            "route_evidence_chain": [
                {
                    "artifact_id": item.get("artifact_id"),
                    "path": item.get("path"),
                    "kind": item.get("kind"),
                    "ordinal": item.get("ordinal"),
                    "provenance": item.get("provenance"),
                    "route_reason": item.get("route_reason"),
                }
                for item in route.evidence
            ],
            "product_route": product_route.to_json(),
            "materialized": materialized_json,
            "materialization": materialized_json,
            "compatibility": (
                materialized.compatibility if route.evidence or materialized.strategy != "none" else None
            ),
            "decoder": {
                "constraints": decoder.constraints,
                "prior_scope": decoder.prior_scope,
            },
            "decoder_prior_scope": decoder.prior_scope,
            "decoder_prior": decoder_prior,
            "model_attestation": self._model_attestation_json(),
            "backend": {
                "name": model_result.backend,
                "ok": model_result.ok,
                "error": model_result.error,
                "metadata": model_result.metadata,
            },
        }

    def _update_decoder_prior(
        self,
        method: str,
        decoder: DecoderPlan,
        verification: VerificationResult,
        model_result: ModelBackendResult,
    ) -> dict[str, Any]:
        del method
        scope = _decoder_prior_scope_from_plan(decoder, self.adapter)
        steering_applied = self._live_steering_applied(model_result)
        temptation_event = _decoder_temptation_event(decoder, model_result.text)
        steering_metadata = model_result.metadata.get("decoder_steering")
        steering_refused_reason = (
            steering_metadata.get("refused_reason") if isinstance(steering_metadata, dict) else None
        )
        record = self.decoder_prior_store.update(
            scope,
            accepted=verification.ok,
            temptation_event=temptation_event,
            steering_applied=steering_applied,
        )
        record.metadata.update(
            {
                "last_scope_source": "decoder.prior_scope",
                "last_steering_applied": steering_applied,
                "last_steering_refused_reason": steering_refused_reason,
                "last_temptation_event": temptation_event,
            }
        )
        self.decoder_prior_store.save()
        return record.to_json()

    @staticmethod
    def _live_steering_applied(model_result: ModelBackendResult) -> bool:
        steering = model_result.metadata.get("decoder_steering")
        if not isinstance(steering, dict):
            return False
        return bool(steering.get("applied"))

    def _harness_session_json(self) -> dict[str, Any] | None:
        if self.harness_session is None:
            return None
        if hasattr(self.harness_session, "to_dict"):
            data = self.harness_session.to_dict()
            data["model_attestation"] = self._model_attestation_json()
            return data
        if hasattr(self.harness_session, "__dict__"):
            data = dict(self.harness_session.__dict__)
            data["model_attestation"] = self._model_attestation_json()
            return data
        return {"repr": repr(self.harness_session), "model_attestation": self._model_attestation_json()}

    def _model_attestation_json(self) -> dict[str, Any] | None:
        if self.model_attestation is None:
            return None
        return self.model_attestation.to_dict()

    def _answer(
        self,
        prompt: str,
        method: str,
        readiness: IndexReadiness,
        route: RoutePacket,
        materialized: MaterializedContext,
        verification: VerificationResult,
        model_result: ModelBackendResult | None = None,
    ) -> str:
        parts = [f"method={method}", f"tier={route.tier}", f"evidence={len(route.evidence)}"]
        if readiness.required:
            parts.append(f"jit_required={readiness.reason}")
        if materialized.refused:
            parts.append(f"materializer_refused={materialized.reason}")
        if method == "verify":
            parts.append(f"verification={'passed' if verification.ok else 'failed'}")
        if model_result is not None and not model_result.ok:
            parts.append(f"backend_blocked={model_result.error}")
        if self.config.model_tool_protocol:
            parts.append("tool_protocol=available")
        parts.append(f"summary={prompt[:120]}")
        return "; ".join(parts)
