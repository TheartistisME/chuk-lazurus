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
        return checks

    @staticmethod
    def _check_temporal_ordinal(evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        requested = metadata.get("requested_ordinal") or metadata.get("ordinal")
        ordinals = [item.get("ordinal") for item in evidence if item.get("ordinal") is not None]
        timestamps = [item.get("timestamp") for item in evidence if item.get("timestamp")]
        ids = [item.get("artifact_id") for item in evidence if item.get("artifact_id")]
        ok = bool(evidence and ordinals and timestamps and ids)
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "requested_ordinal": requested,
            "has_ordinal": bool(ordinals),
            "has_timestamp": bool(timestamps),
            "has_artifact_id": bool(ids),
            "ordinals": ordinals,
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
        missing = [hop for hop in expected if hop not in hop_values]
        complete = not missing if expected else len(evidence) >= 2 and bool(ordered)
        ok = bool(evidence) and complete
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "expected_hops": expected,
            "missing_hops": missing,
            "ordered_hops": len(ordered),
            "linked_hops": len(linked),
        }

    def _check_patch_targets(self, evidence: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
        paths = self._candidate_paths(evidence, metadata)
        unsafe = [path for path in paths if not self._is_safe_patch_path(path)]
        protected = [path for path in paths if classify_path(path).is_protected]
        ok = bool(evidence and paths) and not unsafe and not protected
        return {
            "ok": ok,
            "evidence_count": len(evidence),
            "paths": paths,
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
        ok = not mismatches and not refused
        return {
            "ok": ok,
            "mismatches": mismatches,
            "materializer_refused": refused,
            "strategy": materialized.get("strategy") if isinstance(materialized, dict) else None,
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
        return {
            "ok": not missing and family_ok,
            "missing": missing,
            "expected_family": expected_family,
            "actual_family": writeback.get("family"),
            "kind": writeback.get("kind"),
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
