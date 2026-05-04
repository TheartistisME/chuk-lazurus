"""Verification layer for David runtime capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

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

        sidecar_catalog = self._check_sidecar_catalog_metadata(metadata)
        if sidecar_catalog is not None:
            checks["sidecar_catalog_compatibility"] = sidecar_catalog

        sidecar_replay = self._check_sidecar_replay_metadata(metadata)
        if sidecar_replay is not None:
            checks["sidecar_replay_evidence"] = sidecar_replay

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
        plan_evidence_count = 0
        if isinstance(materialized, dict) and isinstance(materialized.get("materialization_plan"), dict):
            plan = materialized["materialization_plan"]
            plan_evidence_count += len(plan.get("sidecars") or []) if isinstance(plan.get("sidecars"), list) else 0
            plan_evidence_count += len(plan.get("replay_refs") or []) if isinstance(plan.get("replay_refs"), list) else 0
        chain_required = bool(isinstance(materialized, dict) and not refused)
        chain_ok = isinstance(evidence_chain, list) and (bool(evidence_chain) or bool(plan_evidence_count) or not chain_required)
        ok = not mismatches and not refused and chain_ok
        return {
            "ok": ok,
            "mismatches": mismatches,
            "materializer_refused": refused,
            "strategy": materialized.get("strategy") if isinstance(materialized, dict) else None,
            "evidence_chain_count": (len(evidence_chain) if isinstance(evidence_chain, list) else 0) + plan_evidence_count,
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
        policy = _first_mapping(
            metadata.get("write_back_policy"),
            metadata.get("writeback_policy"),
            _nested_mapping(writeback, "metadata", "write_back_policy"),
            _nested_mapping(writeback, "metadata", "writeback_policy"),
        )
        policy_check = _check_writeback_policy(policy, expected_family)
        lifecycle_check = _check_user_lifecycle(writeback, expected_family)
        evidence_chain = metadata.get("route_evidence_chain") or writeback.get("route_evidence_chain") or []
        if not evidence_chain and isinstance(writeback.get("metadata"), dict):
            evidence_chain = writeback["metadata"].get("route_evidence_chain") or []
        chain_required = capability in {"repo_patch", "source_dependency", "symbolic_multi_hop"}
        chain_ok = isinstance(evidence_chain, list) and (bool(evidence_chain) or not chain_required)
        return {
            "ok": (
                not missing
                and family_ok
                and chain_ok
                and policy_check["ok"]
                and lifecycle_check["ok"]
            ),
            "missing": missing,
            "expected_family": expected_family,
            "actual_family": writeback.get("family"),
            "kind": writeback.get("kind"),
            "policy": policy_check,
            "user_lifecycle": lifecycle_check,
            "evidence_chain_count": len(evidence_chain) if isinstance(evidence_chain, list) else 0,
            "evidence_chain_required": chain_required,
        }

    @staticmethod
    def _check_sidecar_catalog_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        materialized = metadata.get("materialized") or metadata.get("materialization")
        plan = materialized.get("materialization_plan") if isinstance(materialized, dict) else None
        if not isinstance(plan, dict):
            return None
        sidecars = plan.get("sidecars") or []
        catalog = _first_mapping(
            metadata.get("sidecar_catalog"),
            metadata.get("residual_sidecar_catalog"),
            metadata.get("kv_sidecar_catalog"),
            plan.get("sidecar_catalog"),
            _nested_mapping(materialized, "compatibility", "sidecar_catalog") if isinstance(materialized, dict) else None,
        )
        if not sidecars and catalog is None:
            return None
        adapter = metadata.get("adapter") or metadata.get("adapter_scope") or plan.get("adapter_scope") or {}
        route_memory_family = None
        if isinstance(materialized, dict):
            route_memory_family = plan.get("memory_family") or _nested_value(
                materialized,
                "compatibility",
                "route_memory_family",
            )
        scope_keys = {
            "model_id",
            "tokenizer_id",
            "model_revision",
            "adapter_family",
            "insertion_family",
            "boundary_layer",
            "kv_source_layer",
            "kv_target_layer",
            "hidden_size",
        }
        mismatches: list[dict[str, Any]] = []
        ref_count = 0
        artifact_ids: list[str] = []
        for sidecar in sidecars if isinstance(sidecars, list) else []:
            if not isinstance(sidecar, dict):
                mismatches.append({"sidecar": sidecar, "reason": "sidecar entry is not an object"})
                continue
            artifact_id = str(sidecar.get("artifact_id") or "")
            if artifact_id:
                artifact_ids.append(artifact_id)
            scope = sidecar.get("scope") if isinstance(sidecar.get("scope"), dict) else {}
            refs = sidecar.get("refs") or []
            if isinstance(refs, list):
                ref_count += len(refs)
            else:
                mismatches.append({"artifact_id": artifact_id, "reason": "sidecar refs are not a list"})
            for key in sorted(scope_keys):
                adapter_value = adapter.get(key) if isinstance(adapter, dict) else None
                sidecar_value = scope.get(key)
                if adapter_value is not None and sidecar_value is not None and str(adapter_value) != str(sidecar_value):
                    mismatches.append(
                        {
                            "artifact_id": artifact_id,
                            "key": key,
                            "adapter": adapter_value,
                            "sidecar": sidecar_value,
                        }
                    )
            sidecar_family = sidecar.get("memory_family") or scope.get("memory_family")
            if (
                route_memory_family is not None
                and sidecar_family is not None
                and str(sidecar_family) != str(route_memory_family)
            ):
                mismatches.append(
                    {
                        "artifact_id": artifact_id,
                        "key": "memory_family",
                        "route": route_memory_family,
                        "sidecar": sidecar_family,
                    }
                )
        catalog_count = None if catalog is None else catalog.get("sidecar_count") or catalog.get("count")
        catalog_count_ok = True
        if catalog_count is not None:
            catalog_count_ok = int(catalog_count) == len(sidecars) if isinstance(catalog_count, int) else False
        return {
            "ok": not mismatches and catalog_count_ok and (bool(sidecars) or catalog is not None),
            "sidecar_count": len(sidecars) if isinstance(sidecars, list) else 0,
            "catalog_count": catalog_count,
            "catalog_count_ok": catalog_count_ok,
            "sidecar_artifact_ids": artifact_ids,
            "replay_ref_count": ref_count,
            "mismatches": mismatches,
            "has_catalog": catalog is not None,
        }

    @staticmethod
    def _check_sidecar_replay_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        materialized = metadata.get("materialized") or metadata.get("materialization")
        if not isinstance(materialized, dict):
            return None
        plan = materialized.get("materialization_plan")
        if not isinstance(plan, dict):
            return None
        replay = materialized.get("materialization_replay")
        if not isinstance(replay, dict):
            backend = metadata.get("backend")
            backend_metadata = backend.get("metadata") if isinstance(backend, dict) else None
            replay = backend_metadata.get("materialization_replay") if isinstance(backend_metadata, dict) else None
        requested_strategy = str(plan.get("requested_strategy") or plan.get("strategy") or "")
        replay_family = None
        if isinstance(replay, dict):
            replay_family = replay.get("replay_family") or _nested_value(replay, "state", "family")
        sidecar_strategy = requested_strategy in {"kv_sidecar", "residual_sidecar"} or replay_family in {
            "kv_sidecar",
            "residual_sidecar",
        }
        if not sidecar_strategy:
            return None

        route_chain = metadata.get("route_evidence_chain") or materialized.get("route_evidence_chain") or []
        replay_refs = plan.get("replay_refs") if isinstance(plan.get("replay_refs"), list) else []
        sidecars = plan.get("sidecars") if isinstance(plan.get("sidecars"), list) else []
        evidence_chain_count = (
            (len(route_chain) if isinstance(route_chain, list) else 0)
            + len(replay_refs)
            + len(sidecars)
        )
        refused = bool(materialized.get("refused") or plan.get("refused") or (replay or {}).get("refused"))
        applied = bool((replay or {}).get("applied") or (replay or {}).get("tensor_replay_applied"))
        refusal_reasons = []
        if isinstance(replay, dict):
            refusal_reasons.extend(str(item) for item in replay.get("refusal_reasons") or [])
            refusal_reasons.extend(str(item) for item in replay.get("guard_reasons") or [])
        if plan.get("reason") and plan.get("reason") != "ok":
            refusal_reasons.append(str(plan.get("reason")))
        if materialized.get("reason") and materialized.get("reason") != "ok":
            refusal_reasons.append(str(materialized.get("reason")))
        reason_ok = applied or not refused or bool(refusal_reasons)
        ok = evidence_chain_count > 0 and reason_ok and bool(replay_refs or sidecars)
        return {
            "ok": ok,
            "strategy": plan.get("strategy"),
            "requested_strategy": requested_strategy,
            "replay_family": replay_family,
            "applied": applied,
            "refused": refused,
            "evidence_chain_count": evidence_chain_count,
            "route_evidence_chain_count": len(route_chain) if isinstance(route_chain, list) else 0,
            "sidecar_count": len(sidecars),
            "replay_ref_count": len(replay_refs),
            "refusal_reason_count": len(refusal_reasons),
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


def _first_mapping(*values: Any) -> dict[str, Any] | None:
    for value in values:
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _nested_value(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _nested_mapping(value: Any, *keys: str) -> dict[str, Any] | None:
    nested = _nested_value(value, *keys)
    return dict(nested) if isinstance(nested, Mapping) else None


def _check_writeback_policy(policy: dict[str, Any] | None, expected_family: str) -> dict[str, Any]:
    if policy is None:
        return {"ok": True, "present": False, "targets": [], "family": None}
    targets = _string_list(policy.get("targets") or policy.get("target") or policy.get("stores"))
    family = policy.get("family") or policy.get("memory_family")
    expected_targets = {
        expected_family,
        f"{expected_family}_memory",
        f"{expected_family}_memory_artifact",
    }
    if expected_family == "user":
        expected_targets.update({"chat_user_memory", "user_continuity_memory", "person_in_time_memory"})
    else:
        expected_targets.update({"task_memory", "code_task_memory", "workspace_codebase_memory"})
    target_ok = not targets or any(target in expected_targets for target in targets)
    family_ok = family in {None, "", expected_family}
    return {
        "ok": target_ok and family_ok,
        "present": True,
        "targets": targets,
        "expected_targets": sorted(expected_targets),
        "target_ok": target_ok,
        "family": family,
        "family_ok": family_ok,
    }


def _check_user_lifecycle(writeback: dict[str, Any], expected_family: str) -> dict[str, Any]:
    if expected_family != "user":
        return {"ok": True, "applicable": False}
    artifact_id = str(writeback.get("artifact_id") or "")
    metadata = writeback.get("metadata") if isinstance(writeback.get("metadata"), dict) else {}
    expires_at = writeback.get("expires_at") or metadata.get("expires_at") or metadata.get("expiry")
    supersedes = _string_list(writeback.get("supersedes") or metadata.get("supersedes"))
    superseded_by = _string_list(writeback.get("superseded_by") or metadata.get("superseded_by"))
    supersession_status = writeback.get("supersession_status") or metadata.get("supersession_status")
    timestamp = writeback.get("timestamp")

    expiry_ok = True
    expiry_reason = None
    if expires_at:
        expiry_time = _parse_iso_datetime(str(expires_at))
        timestamp_time = _parse_iso_datetime(str(timestamp)) if timestamp else None
        if expiry_time is None:
            expiry_ok = False
            expiry_reason = "expires_at is not a valid ISO timestamp"
        elif timestamp_time is not None and expiry_time <= timestamp_time:
            expiry_ok = False
            expiry_reason = "expires_at is not after writeback timestamp"

    supersedes_ok = artifact_id not in supersedes
    superseded_by_ok = artifact_id not in superseded_by
    status_ok = supersession_status in {
        None,
        "",
        "active",
        "current",
        "stale",
        "superseded",
        "superseding",
    }
    return {
        "ok": expiry_ok and supersedes_ok and superseded_by_ok and status_ok,
        "applicable": True,
        "expires_at": expires_at,
        "expiry_ok": expiry_ok,
        "expiry_reason": expiry_reason,
        "supersedes": supersedes,
        "superseded_by": superseded_by,
        "supersedes_ok": supersedes_ok,
        "superseded_by_ok": superseded_by_ok,
        "supersession_status": supersession_status,
        "supersession_status_ok": status_ok,
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
