"""Verification layer for David runtime capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Sequence

from .patch_routing import classify_path
from .tools import LocalTools


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    capability: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    command_result: dict[str, Any] | None = None
    reason: str = "ok"
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "capability": self.capability,
            "evidence": self.evidence,
            "command_result": self.command_result,
            "reason": self.reason,
            "checks": self.checks,
        }


class Verifier:
    def __init__(self, tools: LocalTools) -> None:
        self.tools = tools

    def verify(
        self,
        capability: str,
        evidence: list[dict[str, Any]],
        command: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VerificationResult:
        checks = self._structured_checks(capability, evidence, metadata or {})
        if command is not None:
            result = self.tools.run(command)
            checks["command"] = {
                "ok": result["returncode"] == 0,
                "returncode": result["returncode"],
            }
            ok = self._checks_ok(checks)
            return VerificationResult(
                ok=ok,
                capability=capability,
                evidence=evidence,
                command_result=result,
                reason=self._reason(checks, "command passed" if result["returncode"] == 0 else "command failed"),
                checks=checks,
            )
        if checks:
            return VerificationResult(
                self._checks_ok(checks),
                capability,
                evidence,
                reason=self._reason(checks, "structured evidence passed"),
                checks=checks,
            )
        return VerificationResult(True, capability, evidence)

    def _structured_checks(
        self,
        capability: str,
        evidence: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        checks: dict[str, dict[str, Any]] = {}
        if capability == "temporal_recall":
            checks["temporal_ordinal"] = self._check_temporal_ordinal(evidence, metadata)
        elif capability == "symbolic_multi_hop":
            checks["symbolic_chain"] = self._check_symbolic_chain(evidence, metadata)
        elif capability == "repo_patch":
            checks["patch_targets"] = self._check_patch_targets(evidence, metadata)
        elif capability == "source_dependency":
            checks["source_evidence"] = self._check_source_evidence(evidence)

        compatibility = self._check_compatibility_metadata(metadata)
        if compatibility is not None:
            checks["adapter_materialization_compatibility"] = compatibility

        writeback = self._check_writeback_metadata(metadata, capability)
        if writeback is not None:
            checks["memory_writeback"] = writeback

        decoder_prior = self._check_decoder_prior_metadata(metadata, capability)
        if decoder_prior is not None:
            checks["decoder_prior_scope"] = decoder_prior

        product_route = self._check_product_route_metadata(metadata, capability, evidence)
        if product_route is not None:
            checks["product_route"] = product_route

        backend = self._check_backend_metadata(metadata)
        if backend is not None:
            checks["backend"] = backend
        return checks

    @staticmethod
    def _check_temporal_ordinal(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        requested = metadata.get("requested_ordinal") or metadata.get("ordinal")
        ordinals = [item.get("ordinal") for item in evidence if item.get("ordinal") is not None]
        occurrences = [
            item.get(key)
            for item in evidence
            for key in ("occurrence", "occurrence_index", "occurrence_id")
            if item.get(key) is not None
        ]
        timestamps = [item.get("timestamp") for item in evidence if item.get("timestamp")]
        ids = [item.get("artifact_id") for item in evidence if item.get("artifact_id")]
        occurrence_required = bool(
            requested
            or metadata.get("requires_occurrence_metadata")
            or metadata.get("requested_occurrence")
        )
        has_occurrence = bool(ordinals or occurrences)
        ok = bool(evidence and timestamps and ids and (has_occurrence or not occurrence_required))
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "requested_ordinal": requested,
            "has_ordinal": bool(ordinals),
            "has_occurrence": has_occurrence,
            "has_timestamp": bool(timestamps),
            "has_artifact_id": bool(ids),
            "ordinals": ordinals,
            "occurrences": occurrences,
            "occurrence_required": occurrence_required,
        }

    @staticmethod
    def _check_symbolic_chain(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        expected = [str(item) for item in metadata.get("expected_hops", [])]
        hop_values = {
            str(value)
            for item in evidence
            for key in ("hop", "chain_hop", "symbol", "artifact_id")
            if (value := item.get(key)) is not None
        }
        ordered = [item for item in evidence if item.get("ordinal") is not None or item.get("hop") is not None]
        linked = [
            item
            for item in evidence
            if item.get("depends_on") is not None or item.get("links_to") is not None or item.get("chain_id") is not None
        ]
        provenance = [item for item in evidence if item.get("provenance") or item.get("source") or item.get("metadata")]
        missing = [hop for hop in expected if hop not in hop_values]
        complete = not missing if expected else len(evidence) >= 2 and bool(ordered)
        multi_hop = len(evidence) >= 2 and len(ordered) >= 2
        ok = bool(evidence) and complete and multi_hop and len(provenance) == len(evidence)
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "expected_hops": expected,
            "missing_hops": missing,
            "ordered_hops": len(ordered),
            "linked_hops": len(linked),
            "provenance_count": len(provenance),
            "requires_multi_hop": True,
        }

    def _check_patch_targets(self, evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        target_paths = self._patch_target_paths(evidence, metadata)
        paths = self._candidate_paths(evidence, metadata)
        unsafe = [path for path in paths if not self._is_safe_patch_path(path)]
        protected = [path for path in paths if classify_path(path).is_protected]
        ok = bool(evidence and target_paths) and not unsafe and not protected
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "paths": paths,
            "target_paths": target_paths,
            "requires_selected_or_patch_target_evidence": True,
            "unsafe_paths": unsafe,
            "protected_paths": protected,
        }

    @staticmethod
    def _check_source_evidence(evidence: list[dict[str, Any]]) -> dict[str, Any]:
        paths = [str(item.get("path")) for item in evidence if item.get("path")]
        symbols = [str(item.get("symbol")) for item in evidence if item.get("symbol")]
        ok = bool(evidence and (paths or symbols))
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "paths": paths,
            "symbols": symbols,
        }

    @staticmethod
    def _candidate_paths(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for item in evidence:
            for key in ("path", "file", "target_path"):
                value = item.get(key)
                if value:
                    paths.append(str(value))
            for value in item.get("selected_tests", []) or []:
                paths.append(str(value))
        product_route = metadata.get("product_route") or {}
        for value in product_route.get("selected_paths", []) or []:
            paths.append(str(value))
        return list(dict.fromkeys(paths))

    @staticmethod
    def _patch_target_paths(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        patch_kinds = {"patch_target", "edit_target", "selected_path", "repo_patch"}
        for item in evidence:
            kind = str(item.get("kind") or "")
            is_patch_target = kind in patch_kinds or bool(item.get("patch_target"))
            if not is_patch_target:
                continue
            for key in ("path", "file", "target_path"):
                value = item.get(key)
                if value:
                    paths.append(str(value))
        product_route = metadata.get("product_route") or {}
        for value in product_route.get("selected_paths", []) or []:
            paths.append(str(value))
        for value in metadata.get("selected_paths", []) or []:
            paths.append(str(value))
        return list(dict.fromkeys(paths))

    @staticmethod
    def _is_safe_patch_path(path: str) -> bool:
        normalized = path.replace("\\", "/")
        pure = PurePosixPath(normalized)
        return bool(
            normalized
            and not pure.is_absolute()
            and ".." not in pure.parts
            and "\x00" not in normalized
            and not (pure.parts and ":" in pure.parts[0])
        )

    @staticmethod
    def _check_compatibility_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        adapter = metadata.get("adapter") or metadata.get("adapter_scope")
        materialized = metadata.get("materialized") or metadata.get("materialization")
        compatibility = metadata.get("compatibility")
        if isinstance(materialized, dict):
            compatibility = materialized.get("compatibility") or compatibility
        if not isinstance(adapter, dict) or not isinstance(compatibility, dict):
            return None

        scope_keys = {
            "model_id",
            "tokenizer_id",
            "model_revision",
            "adapter_family",
            "route_layer",
            "boundary_layer",
            "kv_source_layer",
            "kv_target_layer",
            "insertion_family",
            "memory_family",
        }
        mismatches = {
            key: {"adapter": adapter.get(key), "materialized": compatibility.get(key)}
            for key in sorted(scope_keys)
            if adapter.get(key) is not None
            and compatibility.get(key) is not None
            and adapter.get(key) != compatibility.get(key)
        }
        refused = bool(materialized.get("refused")) if isinstance(materialized, dict) else False
        evidence_chain = []
        if isinstance(materialized, dict):
            evidence_chain = materialized.get("evidence_chain") or materialized.get("route_evidence_chain") or []
        if not evidence_chain:
            evidence_chain = metadata.get("route_evidence_chain") or []
        chain_required = bool(isinstance(materialized, dict) and not refused)
        chain_ok = isinstance(evidence_chain, list) and (bool(evidence_chain) or not chain_required)
        ok = not mismatches and not refused and chain_ok
        return {
            "ok": ok,
            "mismatches": mismatches,
            "materializer_refused": refused,
            "strategy": materialized.get("strategy") if isinstance(materialized, dict) else None,
            "evidence_chain_count": len(evidence_chain) if isinstance(evidence_chain, list) else 0,
            "evidence_chain_required": chain_required,
        }

    @staticmethod
    def _check_writeback_metadata(metadata: dict[str, Any], capability: str) -> dict[str, Any] | None:
        writeback = metadata.get("writeback") or metadata.get("memory_writeback")
        if not isinstance(writeback, dict):
            return None
        expected_family = "user" if capability in {"user_continuity", "temporal_recall"} else "task"
        required = ("artifact_id", "family", "kind", "text", "timestamp")
        missing = [key for key in required if not writeback.get(key)]
        family_ok = writeback.get("family") == expected_family
        evidence_chain = metadata.get("route_evidence_chain") or writeback.get("route_evidence_chain") or []
        if not evidence_chain and isinstance(writeback.get("metadata"), dict):
            evidence_chain = writeback["metadata"].get("route_evidence_chain") or []
        chain_required = capability in {"repo_patch", "source_dependency", "symbolic_multi_hop"}
        chain_ok = isinstance(evidence_chain, list) and (bool(evidence_chain) or not chain_required)
        return {
            "ok": not missing and family_ok and chain_ok,
            "missing": missing,
            "expected_family": expected_family,
            "actual_family": writeback.get("family"),
            "kind": writeback.get("kind"),
            "evidence_chain_count": len(evidence_chain) if isinstance(evidence_chain, list) else 0,
            "evidence_chain_required": chain_required,
        }

    @staticmethod
    def _check_decoder_prior_metadata(metadata: dict[str, Any], capability: str) -> dict[str, Any] | None:
        decoder = metadata.get("decoder") if isinstance(metadata.get("decoder"), dict) else {}
        prior_scope = metadata.get("decoder_prior_scope") or decoder.get("prior_scope")
        prior = metadata.get("decoder_prior")
        adapter = metadata.get("adapter") or metadata.get("adapter_scope") or {}
        if not isinstance(prior_scope, dict):
            return None

        required = ("model_id", "tokenizer_id", "adapter_family")
        missing = [key for key in required if not prior_scope.get(key)]
        expected_method = prior_scope.get("method") or prior_scope.get("task_type")
        method_ok = expected_method in {None, "", capability}
        scope = prior.get("scope") if isinstance(prior, dict) and isinstance(prior.get("scope"), dict) else {}
        comparable_scope_keys = {
            "model_id",
            "tokenizer_id",
            "adapter_family",
            "model_revision",
            "insertion_family",
            "layer",
            "kv_target_layer",
        }
        mismatches = {
            key: {"decoder": prior_scope.get(key), "prior": scope.get(key)}
            for key in sorted((set(prior_scope) & set(scope)) & comparable_scope_keys)
            if prior_scope.get(key) is not None
            and scope.get(key) is not None
            and str(prior_scope.get(key)) != str(scope.get(key))
        }
        prior_method = scope.get("task_type") or scope.get("method") if scope else None
        prior_method_ok = prior_method in {None, "", capability}
        adapter_mismatches = {
            key: {"adapter": adapter.get(key), "decoder": prior_scope.get(key)}
            for key in ("model_id", "tokenizer_id", "adapter_family", "model_revision", "insertion_family")
            if isinstance(adapter, dict)
            and adapter.get(key) is not None
            and prior_scope.get(key) is not None
            and str(adapter.get(key)) != str(prior_scope.get(key))
        }
        return {
            "ok": not missing and method_ok and prior_method_ok and not mismatches and not adapter_mismatches,
            "missing": missing,
            "method": expected_method,
            "prior_method": prior_method,
            "expected_method": capability,
            "prior_scope_mismatches": mismatches,
            "adapter_scope_mismatches": adapter_mismatches,
            "has_persisted_prior": isinstance(prior, dict),
        }

    @staticmethod
    def _check_product_route_metadata(
        metadata: dict[str, Any],
        capability: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        product_route = metadata.get("product_route")
        if not isinstance(product_route, dict):
            return None
        evidence_chain = metadata.get("route_evidence_chain")
        route_reasons = product_route.get("route_reasons") or []
        method = product_route.get("method")
        product_evidence = product_route.get("evidence") or []
        selected_paths = product_route.get("selected_paths") or []
        ok = (
            method == capability
            and bool(product_route.get("methodology"))
            and bool(product_route.get("capability"))
            and bool(product_route.get("proof_rig"))
            and len(product_evidence) >= len(evidence)
            and (not evidence or isinstance(evidence_chain, list))
        )
        return {
            "ok": ok,
            "method": method,
            "expected_method": capability,
            "methodology": product_route.get("methodology"),
            "proof_rig": product_route.get("proof_rig"),
            "route_reason_count": len(route_reasons),
            "evidence_count": len(product_evidence),
            "runtime_evidence_count": len(evidence),
            "route_evidence_chain_count": len(evidence_chain) if isinstance(evidence_chain, list) else 0,
            "selected_paths": selected_paths,
        }

    @staticmethod
    def _check_backend_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        backend = metadata.get("backend")
        if not isinstance(backend, dict):
            return None
        ok_value = backend.get("ok")
        name = backend.get("name") or backend.get("backend")
        return {
            "ok": bool(name) and isinstance(ok_value, bool) and (ok_value or bool(backend.get("error"))),
            "name": name,
            "backend_ok": ok_value,
            "error": backend.get("error"),
            "metadata_keys": sorted((backend.get("metadata") or {}).keys())
            if isinstance(backend.get("metadata"), dict)
            else [],
        }

    @staticmethod
    def _checks_ok(checks: dict[str, dict[str, Any]]) -> bool:
        return all(bool(check.get("ok")) for check in checks.values())

    @staticmethod
    def _reason(checks: dict[str, dict[str, Any]], success: str) -> str:
        failed = [name for name, check in checks.items() if not check.get("ok")]
        if failed:
            return "failed checks: " + ", ".join(failed)
        return success
