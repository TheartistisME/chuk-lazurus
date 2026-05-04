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

        compatibility = self._check_compatibility_metadata(metadata, evidence)
        if compatibility is not None:
            checks["adapter_materialization_compatibility"] = compatibility

        writeback = self._check_writeback_metadata(metadata, capability)
        if writeback is not None:
            checks["memory_writeback"] = writeback

        agent_loop = self._check_agent_loop_repair_metadata(metadata)
        if agent_loop is not None:
            checks["agent_loop_repair"] = agent_loop

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
        requested = _first_present_value(metadata, "requested_ordinal", "ordinal")
        requested_occurrence = _first_present_value(
            metadata,
            "requested_occurrence",
            "occurrence",
            "occurrence_index",
            "occurrence_id",
        )
        requested_values = [value for value in (requested, requested_occurrence) if _is_present_value(value)]
        ordinals = [item.get("ordinal") for item in evidence if item.get("ordinal") is not None]
        occurrences = [
            item.get(key)
            for item in evidence
            for key in ("occurrence", "occurrence_index", "occurrence_id")
            if item.get(key) is not None
        ]
        occurrence_entries = _temporal_occurrence_entries(evidence)
        timestamps = [item.get("timestamp") for item in evidence if item.get("timestamp")]
        ids = [item.get("artifact_id") for item in evidence if item.get("artifact_id")]
        occurrence_required = bool(
            requested_values
            or metadata.get("requires_occurrence_metadata")
        )
        has_occurrence = bool(occurrence_entries)
        matching_occurrences = [
            entry
            for entry in occurrence_entries
            if any(_ordinal_value_matches(requested_value, entry["value"], len(evidence)) for requested_value in requested_values)
        ]
        ordinal_match_required = bool(requested_values)
        ordinal_match_ok = not ordinal_match_required or not has_occurrence or bool(matching_occurrences)
        ordinal_mismatch = ordinal_match_required and has_occurrence and not matching_occurrences
        ok = bool(evidence and timestamps and ids and (has_occurrence or not occurrence_required) and ordinal_match_ok)
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "requested_ordinal": requested,
            "requested_occurrence": requested_occurrence,
            "has_ordinal": bool(ordinals),
            "has_occurrence": has_occurrence,
            "has_timestamp": bool(timestamps),
            "has_artifact_id": bool(ids),
            "ordinals": ordinals,
            "occurrences": occurrences,
            "occurrence_required": occurrence_required,
            "ordinal_match_required": ordinal_match_required,
            "ordinal_match_ok": ordinal_match_ok,
            "ordinal_mismatch": ordinal_mismatch,
            "matching_occurrences": matching_occurrences,
            "occurrence_metadata": occurrence_entries,
        }

    @staticmethod
    def _check_symbolic_chain(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        expected = [_expected_hop_value(item) for item in metadata.get("expected_hops", [])]
        records = [_chain_hop_record(index, item) for index, item in enumerate(evidence)]
        hop_ids = [record["hop"] for record in records if record["hop"] is not None]
        hop_values = set(hop_ids)
        ordered = [record for record in records if record["has_order"]]
        linked = [record for record in records if record["link_values"] or record["chain_id"] is not None]
        provenance = [record for record in records if record["has_provenance"]]
        missing = [hop for hop in expected if hop not in hop_values]
        duplicate_hops = _duplicate_values(hop_ids)
        duplicate_order_values = _duplicate_values(
            [str(record["order"]) for record in records if record["order"] is not None]
        )
        out_of_order_hops = _chain_order_errors(records, expected)
        missing_hop_metadata = _missing_chain_metadata(records)
        link_gaps = _chain_link_gaps(records, expected)
        complete = not missing if expected else len(evidence) >= 2 and bool(ordered)
        multi_hop = len(evidence) >= 2 and len(ordered) >= 2
        provenance_ok = len(provenance) == len(evidence)
        order_ok = not out_of_order_hops and not duplicate_order_values
        hop_metadata_ok = not missing_hop_metadata
        duplicate_ok = not duplicate_hops
        link_ok = not link_gaps
        ok = (
            bool(evidence)
            and complete
            and multi_hop
            and provenance_ok
            and order_ok
            and hop_metadata_ok
            and duplicate_ok
            and link_ok
        )
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "expected_hops": expected,
            "missing_hops": missing,
            "ordered_hops": len(ordered),
            "linked_hops": len(linked),
            "provenance_count": len(provenance),
            "requires_multi_hop": True,
            "duplicate_hops": duplicate_hops,
            "duplicate_order_values": duplicate_order_values,
            "out_of_order_hops": out_of_order_hops,
            "missing_hop_metadata": missing_hop_metadata,
            "link_gaps": link_gaps,
            "provenance_missing": [
                {"index": record["index"], "hop": record["hop"]}
                for record in records
                if not record["has_provenance"]
            ],
            "order_ok": order_ok,
            "provenance_ok": provenance_ok,
            "link_ok": link_ok,
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
    def _check_compatibility_metadata(
        metadata: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
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
        route_chain = evidence_chain if isinstance(evidence_chain, list) else []
        route_chain_check = _route_evidence_chain_check(evidence, route_chain)
        plan_evidence_count = 0
        if isinstance(materialized, dict) and isinstance(materialized.get("materialization_plan"), dict):
            plan = materialized["materialization_plan"]
            plan_evidence_count += len(plan.get("sidecars") or []) if isinstance(plan.get("sidecars"), list) else 0
            plan_evidence_count += len(plan.get("replay_refs") or []) if isinstance(plan.get("replay_refs"), list) else 0
        chain_required = bool(isinstance(materialized, dict) and not refused)
        has_chain_evidence = bool(route_chain) or bool(plan_evidence_count)
        route_chain_complete = route_chain_check["complete"] if route_chain else bool(plan_evidence_count or not chain_required)
        chain_ok = isinstance(evidence_chain, list) and (has_chain_evidence or not chain_required) and route_chain_complete
        route_paths = _paths_from_items(route_chain)
        if isinstance(materialized, dict):
            route_paths.extend(_string_list(materialized.get("selected_paths") or materialized.get("paths")))
        excluded_paths = _string_list(
            metadata.get("excluded_paths")
            or (materialized.get("excluded_paths") if isinstance(materialized, dict) else None)
            or (materialized.get("excluded_candidates") if isinstance(materialized, dict) else None)
        )
        path_safety = _path_safety_report(route_paths, excluded_paths)
        ok = not mismatches and not refused and chain_ok and path_safety["ok"]
        return {
            "ok": ok,
            "mismatches": mismatches,
            "materializer_refused": refused,
            "strategy": materialized.get("strategy") if isinstance(materialized, dict) else None,
            "evidence_chain_count": (len(evidence_chain) if isinstance(evidence_chain, list) else 0) + plan_evidence_count,
            "evidence_chain_required": chain_required,
            "route_evidence_chain_count": len(route_chain),
            "route_evidence_chain_complete": route_chain_complete,
            "missing_route_evidence_refs": route_chain_check["missing_refs"],
            "runtime_evidence_refs": route_chain_check["required_refs"],
            "route_evidence_refs": route_chain_check["chain_refs"],
            "route_unsafe_paths": path_safety["unsafe_paths"],
            "route_protected_paths": path_safety["protected_paths"],
            "excluded_unsafe_paths": path_safety["excluded_unsafe_paths"],
            "excluded_protected_paths": path_safety["excluded_protected_paths"],
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
    def _check_agent_loop_repair_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        loop, loop_source = _agent_loop_mapping(metadata)
        if loop is None:
            return None

        trace = _agent_loop_trace(loop)
        final_verified, final_verify_step = _agent_loop_final_verified(loop, trace)
        failed_actions = [_agent_loop_step_summary(step) for step in trace if _agent_loop_step_failed(step)]
        recovered_failures = [
            failure
            for failure in failed_actions
            if final_verified and final_verify_step is not None and failure["step"] < final_verify_step
        ]
        unresolved_failures = [
            failure
            for failure in failed_actions
            if not final_verified or final_verify_step is None or failure["step"] >= final_verify_step
        ]
        first_failure_step = failed_actions[0]["step"] if failed_actions else None
        repair_attempts = [
            _agent_loop_step_summary(step)
            for step in trace
            if first_failure_step is not None
            and step["step"] > first_failure_step
            and (final_verify_step is None or step["step"] <= final_verify_step)
            and step["action"] in {"patch", "write", "run", "shell"}
        ]
        return {
            "ok": bool(trace) and final_verified and not unresolved_failures,
            "loop_source": loop_source,
            "status": loop.get("status") or loop.get("final_status"),
            "trace_count": len(trace),
            "final_verified": final_verified,
            "final_verify_step": final_verify_step,
            "repair_attempted": bool(repair_attempts),
            "repair_attempt_count": len(repair_attempts),
            "repair_attempts": repair_attempts,
            "failed_action_count": len(failed_actions),
            "failed_actions_recovered": bool(recovered_failures) and not unresolved_failures,
            "recovered_failures": recovered_failures,
            "unresolved_failures": unresolved_failures,
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
        if not isinstance(evidence_chain, list):
            evidence_chain = product_route.get("route_evidence_chain")
        route_chain = evidence_chain if isinstance(evidence_chain, list) else []
        route_reasons = product_route.get("route_reasons") or []
        if not _is_non_string_sequence(route_reasons):
            route_reasons = []
        method = product_route.get("method")
        raw_product_evidence = product_route.get("evidence") or []
        product_evidence = list(raw_product_evidence) if _is_non_string_sequence(raw_product_evidence) else []
        selected_paths = _string_list(product_route.get("selected_paths"))
        combined_evidence = [item for item in [*evidence, *product_evidence] if isinstance(item, dict)]
        route_chain_check = _route_evidence_chain_check(combined_evidence, route_chain)
        excluded_paths = _string_list(product_route.get("excluded_paths") or product_route.get("excluded_candidates"))
        path_safety = _path_safety_report(
            [*_string_list(selected_paths), *_paths_from_items(product_evidence), *_paths_from_items(route_chain)],
            excluded_paths,
        )
        ok = (
            method == capability
            and bool(product_route.get("methodology"))
            and bool(product_route.get("capability"))
            and bool(product_route.get("proof_rig"))
            and len(product_evidence) >= len(evidence)
            and (not evidence or isinstance(evidence_chain, list))
            and route_chain_check["complete"]
            and path_safety["ok"]
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
            "route_evidence_chain_complete": route_chain_check["complete"],
            "missing_route_evidence_refs": route_chain_check["missing_refs"],
            "runtime_evidence_refs": route_chain_check["required_refs"],
            "route_evidence_refs": route_chain_check["chain_refs"],
            "selected_paths": selected_paths,
            "selected_unsafe_paths": path_safety["unsafe_paths"],
            "selected_protected_paths": path_safety["protected_paths"],
            "excluded_unsafe_paths": path_safety["excluded_unsafe_paths"],
            "excluded_protected_paths": path_safety["excluded_protected_paths"],
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


def _is_present_value(value: Any) -> bool:
    return value is not None and value != ""


def _first_present_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and _is_present_value(mapping.get(key)):
            return mapping.get(key)
    return None


def _temporal_occurrence_entries(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    occurrence_keys = (
        "ordinal",
        "ordinal_index",
        "ordinal_rank",
        "occurrence",
        "occurrence_index",
        "occurrence_id",
        "occurrence_rank",
    )
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        for key in occurrence_keys:
            value = item.get(key)
            if value is not None:
                entries.append(
                    {
                        "index": index,
                        "key": key,
                        "value": value,
                        "artifact_id": item.get("artifact_id"),
                    }
                )
    return entries


def _ordinal_value_matches(requested: Any, actual: Any, evidence_count: int) -> bool:
    return bool(_ordinal_tokens(requested, evidence_count) & _ordinal_tokens(actual, evidence_count))


def _ordinal_tokens(value: Any, evidence_count: int) -> set[Any]:
    if not _is_present_value(value):
        return set()
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    tokens: set[Any] = {text}
    numeric = _maybe_int(value)
    if numeric is not None:
        tokens.update({numeric, str(numeric)})

    aliases: dict[str, set[Any]] = {
        "latest": {"latest", "last", "most_recent", "newest", 0, "0"},
        "last": {"latest", "last", "most_recent", "newest", 0, "0"},
        "most_recent": {"latest", "last", "most_recent", "newest", 0, "0"},
        "newest": {"latest", "last", "most_recent", "newest", 0, "0"},
        "first": {"first", 1, "1"},
        "1st": {"first", 1, "1"},
        "second": {"second", 2, "2"},
        "2nd": {"second", 2, "2"},
        "third": {"third", 3, "3"},
        "3rd": {"third", 3, "3"},
    }
    if evidence_count == 1 and text in {"latest", "last", "most_recent", "newest"}:
        tokens.update({"only", "single"})
    tokens.update(aliases.get(text, set()))
    return tokens


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected_hop_value(item: Any) -> str:
    if isinstance(item, Mapping):
        for key in ("hop", "chain_hop", "symbol", "artifact_id", "id"):
            value = item.get(key)
            if value is not None:
                return str(value)
    return str(item)


def _chain_hop_record(index: int, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "hop": _chain_hop_id(item),
        "order": _chain_order_value(item),
        "has_order": _chain_has_order_metadata(item),
        "link_values": _chain_link_values(item),
        "chain_id": item.get("chain_id"),
        "has_provenance": bool(item.get("provenance") or item.get("source") or item.get("metadata")),
    }


def _chain_hop_id(item: Mapping[str, Any]) -> str | None:
    for key in ("chain_hop", "symbol", "artifact_id", "id", "hop"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return None


def _chain_order_value(item: Mapping[str, Any]) -> Any:
    for key in ("ordinal", "hop_index", "chain_index", "position", "sequence"):
        value = item.get(key)
        if value is not None:
            return value
    hop = item.get("hop")
    return hop if _maybe_int(hop) is not None else None


def _chain_has_order_metadata(item: Mapping[str, Any]) -> bool:
    return any(item.get(key) is not None for key in ("ordinal", "hop_index", "chain_index", "position", "sequence", "hop"))


def _chain_link_values(item: Mapping[str, Any]) -> list[str]:
    links: list[str] = []
    for key in ("depends_on", "links_to", "previous_hop", "previous", "parent_hop"):
        value = item.get(key)
        if value is None:
            continue
        if _is_non_string_sequence(value):
            links.extend(str(entry) for entry in value if entry is not None)
        else:
            links.append(str(value))
    return links


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _chain_order_errors(records: list[dict[str, Any]], expected: list[str]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    previous_record: dict[str, Any] | None = None
    previous_key: tuple[int, Any] | None = None
    for record in records:
        if record["order"] is None:
            continue
        key = _order_sort_key(record["order"])
        if previous_key is not None and key <= previous_key:
            errors.append(
                {
                    "reason": "non_increasing_declared_order",
                    "index": record["index"],
                    "hop": record["hop"],
                    "order": record["order"],
                    "previous_hop": previous_record["hop"] if previous_record else None,
                    "previous_order": previous_record["order"] if previous_record else None,
                }
            )
        previous_key = key
        previous_record = record

    if expected:
        expected_rank = {hop: index for index, hop in enumerate(expected)}
        observed = [record["hop"] for record in records if record["hop"] in expected_rank]
        observed_ranks = [expected_rank[str(hop)] for hop in observed]
        if observed_ranks != sorted(observed_ranks):
            errors.append(
                {
                    "reason": "expected_hop_order_mismatch",
                    "expected_order": expected,
                    "observed_order": observed,
                }
            )
    return errors


def _order_sort_key(value: Any) -> tuple[int, Any]:
    numeric = _maybe_int(value)
    if numeric is not None:
        return (0, numeric)
    return (1, str(value))


def _missing_chain_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for record in records:
        fields: list[str] = []
        if record["hop"] is None:
            fields.append("hop")
        if not record["has_order"]:
            fields.append("order")
        if not record["has_provenance"]:
            fields.append("provenance")
        if fields:
            missing.append({"index": record["index"], "hop": record["hop"], "missing": fields})
    return missing


def _chain_link_gaps(records: list[dict[str, Any]], expected: list[str]) -> list[dict[str, Any]]:
    if not expected:
        return []
    by_hop: dict[str, dict[str, Any]] = {}
    for record in records:
        hop = record["hop"]
        if hop is not None and hop not in by_hop:
            by_hop[hop] = record

    gaps: list[dict[str, Any]] = []
    for index, hop in enumerate(expected[1:], start=1):
        record = by_hop.get(hop)
        if record is None:
            continue
        previous_hop = expected[index - 1]
        links = set(record["link_values"])
        if previous_hop not in links:
            gaps.append(
                {
                    "hop": hop,
                    "expected_depends_on": previous_hop,
                    "links": sorted(links),
                    "reason": "missing_link" if not links else "previous_hop_not_linked",
                }
            )
    return gaps


def _route_evidence_chain_check(
    evidence: list[dict[str, Any]],
    route_chain: list[Any],
) -> dict[str, Any]:
    chain_refs: set[str] = set()
    for item in route_chain:
        if isinstance(item, Mapping):
            chain_refs.update(_evidence_ref_alternatives(item))

    required_refs: set[str] = set()
    missing_refs: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        refs = _evidence_ref_alternatives(item)
        if not refs:
            continue
        required_refs.update(refs)
        if not refs & chain_refs:
            missing_refs.append({"index": index, "refs": sorted(refs)})

    return {
        "complete": not missing_refs,
        "missing_refs": missing_refs,
        "required_refs": sorted(required_refs),
        "chain_refs": sorted(chain_refs),
    }


def _evidence_ref_alternatives(item: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("artifact_id", "id"):
        value = item.get(key)
        if value:
            refs.update({f"artifact_id:{value}", str(value)})
    for key in ("path", "file", "target_path"):
        value = item.get(key)
        if value:
            refs.update({f"path:{value}", f"file:{value}", f"target_path:{value}", str(value)})
    if not refs and item.get("symbol"):
        refs.add(f"symbol:{item['symbol']}")
    return refs


def _paths_from_items(items: Any) -> list[str]:
    if not _is_non_string_sequence(items):
        return []
    paths: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key in ("path", "file", "target_path"):
            value = item.get(key)
            if value:
                paths.append(str(value))
        for value in item.get("selected_paths", []) or []:
            paths.append(str(value))
        for value in item.get("selected_tests", []) or []:
            paths.append(str(value))
    return list(dict.fromkeys(paths))


def _path_safety_report(included_paths: list[str], excluded_paths: list[str]) -> dict[str, Any]:
    included = list(dict.fromkeys(str(path) for path in included_paths if _is_present_value(path)))
    excluded = list(dict.fromkeys(str(path) for path in excluded_paths if _is_present_value(path)))
    unsafe = [path for path in included if not Verifier._is_safe_patch_path(path)]
    protected = [path for path in included if classify_path(path).is_protected]
    excluded_unsafe = [path for path in excluded if not Verifier._is_safe_patch_path(path)]
    excluded_protected = [path for path in excluded if classify_path(path).is_protected]
    return {
        "ok": not unsafe and not protected,
        "paths": included,
        "unsafe_paths": unsafe,
        "protected_paths": protected,
        "excluded_paths": excluded,
        "excluded_unsafe_paths": excluded_unsafe,
        "excluded_protected_paths": excluded_protected,
    }


def _agent_loop_mapping(metadata: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    for source, value in (
        ("loop", metadata.get("loop")),
        ("agent_loop", metadata.get("agent_loop")),
    ):
        if isinstance(value, Mapping):
            return dict(value), source

    for writeback_key in ("writeback", "memory_writeback", "agent_loop_writeback"):
        writeback = metadata.get(writeback_key)
        if not isinstance(writeback, Mapping):
            continue
        for source, value in (
            (f"{writeback_key}.loop", writeback.get("loop")),
            (f"{writeback_key}.agent_loop", writeback.get("agent_loop")),
        ):
            if isinstance(value, Mapping):
                return dict(value), source
        writeback_metadata = writeback.get("metadata")
        if isinstance(writeback_metadata, Mapping):
            for source, value in (
                (f"{writeback_key}.metadata.loop", writeback_metadata.get("loop")),
                (f"{writeback_key}.metadata.agent_loop", writeback_metadata.get("agent_loop")),
            ):
                if isinstance(value, Mapping):
                    return dict(value), source

    loop_trace = metadata.get("loop_trace") or metadata.get("agent_loop_trace")
    if _is_non_string_sequence(loop_trace):
        return {
            "trace": list(loop_trace),
            "status": metadata.get("loop_status") or metadata.get("status"),
            "verified": metadata.get("final_verified") or metadata.get("verified"),
        }, "loop_trace"
    return None, None


def _agent_loop_trace(loop: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_trace = loop.get("trace") or loop.get("loop_trace") or loop.get("steps") or []
    if not _is_non_string_sequence(raw_trace):
        return []

    trace: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_trace, start=1):
        if not isinstance(raw_step, Mapping):
            continue
        observation = raw_step.get("observation")
        if not isinstance(observation, Mapping):
            observation = {}
        step = _coerce_int(raw_step.get("step"), index)
        action = str(raw_step.get("action") or raw_step.get("tool") or "").strip().lower()
        ok = raw_step.get("ok")
        if ok is None and "passed" in raw_step:
            ok = raw_step.get("passed")
        if ok is None and action == "verify":
            ok = observation.get("passed")
        if ok is None and action == "patch":
            ok = observation.get("ok")
        trace.append(
            {
                "step": step,
                "action": action,
                "ok": ok if isinstance(ok, bool) else None,
                "observation": dict(observation),
                "reason": raw_step.get("reason") or observation.get("reason") or observation.get("error"),
            }
        )
    return trace


def _agent_loop_final_verified(
    loop: Mapping[str, Any],
    trace: list[dict[str, Any]],
) -> tuple[bool, int | None]:
    status = str(loop.get("status") or loop.get("final_status") or "").strip().lower()
    verified = loop.get("verified")
    if verified is None:
        verified = loop.get("final_verified")
    successful_verify_steps = [
        step["step"]
        for step in trace
        if step["action"] == "verify" and _agent_loop_step_succeeded(step)
    ]
    last_step = trace[-1] if trace else None
    last_step_verified = bool(
        last_step
        and last_step["action"] == "verify"
        and _agent_loop_step_succeeded(last_step)
    )
    final_verified = (
        last_step_verified
        or status in {"verified", "success", "succeeded", "passed"}
        or verified is True
    ) and status not in {"refused", "verify_failed", "failed", "error", "max_steps"}
    if not final_verified:
        return False, None
    if last_step_verified:
        return True, int(last_step["step"])
    if successful_verify_steps:
        return True, max(successful_verify_steps)
    if trace:
        return True, int(trace[-1]["step"])
    return True, None


def _agent_loop_step_failed(step: Mapping[str, Any]) -> bool:
    if step.get("action") in {"refuse", "error"}:
        return True
    ok = step.get("ok")
    if isinstance(ok, bool):
        return not ok
    observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
    if step.get("action") == "verify" and isinstance(observation.get("passed"), bool):
        return not bool(observation["passed"])
    if step.get("action") == "patch" and isinstance(observation.get("ok"), bool):
        return not bool(observation["ok"])
    return False


def _agent_loop_step_succeeded(step: Mapping[str, Any]) -> bool:
    ok = step.get("ok")
    if isinstance(ok, bool):
        return ok
    observation = step.get("observation") if isinstance(step.get("observation"), Mapping) else {}
    if step.get("action") == "verify" and isinstance(observation.get("passed"), bool):
        return bool(observation["passed"])
    if step.get("action") == "patch" and isinstance(observation.get("ok"), bool):
        return bool(observation["ok"])
    return False


def _agent_loop_step_summary(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": step.get("step"),
        "action": step.get("action"),
        "ok": step.get("ok"),
        "reason": step.get("reason"),
    }


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
