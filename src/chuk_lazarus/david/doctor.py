"""Filesystem-only readiness checks for David model boot."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .model_artifacts import VindexArtifactMetadata, inspect_vindex_artifact, is_vindex_artifact_path
from .model_validation import (
    GET_MODEL_CONFIG_RELATIVE_PATH,
    VALIDATE_MODEL_CONFIG_RELATIVE_PATH,
    ValidationReportDiscovery,
    ValidationReportSummary,
    discover_validation_report,
    summarize_validation_report,
    workspace_auto_model_report_paths,
)

ATTESTATION_SCHEMA_NAMES = (
    "david.model_attestation",
    "lazarus.model_attestation",
    "david.manual_model_attestation",
)
ATTESTATION_FILENAMES = (
    "model_attestation.json",
    "david_model_attestation.json",
    "manual_model_attestation.json",
)
WORKSPACE_ATTESTATION_PATHS = (
    Path(".david") / "model_attestation.json",
    Path(".david") / "model" / "model_attestation.json",
    Path(".david") / "model_validation" / "model_attestation.json",
)
STANDARD_DECODE_CAPABILITIES = frozenset(("standard_decode", "decode", "text_generation"))
UNSAFE_REPLAY_CAPABILITIES = frozenset(
    (
        "tensor_replay",
        "kv_replay",
        "kv_direct",
        "kv_direct_tensor_replay",
        "residual_replay",
        "residual_direct",
        "residual_tensor_replay",
    )
)
DEFAULT_WSL_PROBE_TIMEOUT_SECONDS = 20.0
WSL_PROBE_TIMEOUT_ENV = "DAVID_WSL_PROBE_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class DoctorCheck:
    """One readiness check from the production doctor command."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class AttestationDiscovery:
    """Result of looking for a manual-reviewed model attestation."""

    path: Path | None
    checked_paths: tuple[Path, ...]
    explicit: bool = False


@dataclass(frozen=True)
class ModelAttestationSummary:
    """Display-only summary of an operator-reviewed model attestation."""

    path: Path | None
    readable: bool
    attestation_status: str | None
    reviewer: str | None
    rationale: str | None
    expires_at: str | None
    model_identity: str | None
    tokenizer_identity: str | None
    model_revision: str | None
    validation_report_digest: str | None
    selected_adapter_config_id: str | None
    allowed_capabilities: tuple[str, ...]
    requested_replay_capabilities: tuple[str, ...]
    refused_replay_capabilities: tuple[str, ...]
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    error: str | None = None

    @property
    def standard_decode_allowed(self) -> bool:
        return any(
            capability in STANDARD_DECODE_CAPABILITIES
            for capability in self.allowed_capabilities
        )

    @property
    def tensor_replay_effective(self) -> bool:
        return False

    @property
    def standard_decode_only(self) -> bool:
        return self.standard_decode_allowed and not self.tensor_replay_effective

    def compact_detail(self) -> str:
        reasons = "; ".join(self.rejection_reasons) if self.rejection_reasons else "none"
        refused = (
            ", ".join(self.refused_replay_capabilities)
            if self.refused_replay_capabilities
            else "none"
        )
        requested = (
            ", ".join(self.requested_replay_capabilities)
            if self.requested_replay_capabilities
            else "none"
        )
        capabilities = ", ".join(self.allowed_capabilities) if self.allowed_capabilities else "none"
        return (
            f"attestation_status={self.attestation_status or 'unknown'}; "
            f"reviewer={self.reviewer or 'unknown'}; "
            f"standard_decode_allowed={self.standard_decode_allowed}; "
            f"standard_decode_only={self.standard_decode_only}; "
            f"tensor_replay_effective={self.tensor_replay_effective}; "
            f"allowed_capabilities={capabilities}; "
            f"requested_replay_capabilities={requested}; "
            f"refused_replay_capabilities={refused}; "
            f"rejection_reasons={reasons}"
        )


@dataclass(frozen=True)
class DavidDoctorReport:
    """Complete filesystem/package readiness report."""

    model: str | None
    workspace_path: Path
    checks: tuple[DoctorCheck, ...]
    validation_discovery: ValidationReportDiscovery
    validation_summary: ValidationReportSummary | None = None
    attestation_discovery: AttestationDiscovery | None = None
    attestation_summary: ModelAttestationSummary | None = None

    @property
    def ready(self) -> bool:
        ignored_checks = {".vindex.ple artifact"}
        for check in self.checks:
            if check.name in ignored_checks:
                continue
            if check.name == "validation report" and self._standard_decode_attested_ready():
                continue
            if check.status in {"blocked", "missing"}:
                return False
        return True

    def _standard_decode_attested_ready(self) -> bool:
        validation = self.validation_summary
        attestation = self.attestation_summary
        if self.validation_discovery.path is None or validation is None or attestation is None:
            return False
        if validation.validation_status != "needs_review" or validation.auto_load_allowed is not False:
            return False
        return (
            attestation.readable
            and attestation.standard_decode_allowed
            and attestation.standard_decode_only
            and not attestation.rejection_reasons
            and not attestation.error
            and not attestation.requested_replay_capabilities
        )


def run_doctor(
    *,
    model: str | None,
    workspace_path: Path,
    validation_report: str | None = None,
    attestation_path: str | None = None,
    auto_validate_model: bool = False,
    repo_root: Path | None = None,
) -> DavidDoctorReport:
    """Inspect local boot blockers without downloading or loading a model."""

    workspace_root = workspace_path.expanduser().resolve()
    root = _repo_root(repo_root)
    discovery = _discover_report(
        model=model,
        workspace_path=workspace_root,
        validation_report=validation_report,
    )
    validation_summary = (
        summarize_validation_report(discovery.path) if discovery.path is not None else None
    )
    attestation_discovery = _discover_attestation(
        model=model,
        workspace_path=workspace_root,
        validation_report_path=discovery.path,
        attestation_path=attestation_path,
    )
    attestation_summary = (
        summarize_model_attestation(attestation_discovery.path)
        if attestation_discovery.path is not None
        else None
    )
    checks = [
        _workspace_check(workspace_root),
        _model_location_check(model),
        _hf_snapshot_check(model),
        _wsl_hf_snapshot_check(model),
        _vindex_check(workspace_root, model),
        _validation_report_check(discovery, validation_summary),
        _attestation_check(attestation_discovery, attestation_summary),
        *_optional_package_checks(),
        _torch_cuda_check_compat(discovery.path),
        _wsl_python_packages_check(),
        _auto_validate_check(
            model=model,
            workspace_path=workspace_root,
            discovery=discovery,
            auto_validate_model=auto_validate_model,
            repo_root=root,
        ),
        _wsl_tooling_check(),
    ]
    return DavidDoctorReport(
        model=model,
        workspace_path=workspace_root,
        checks=tuple(checks),
        validation_discovery=discovery,
        validation_summary=validation_summary,
        attestation_discovery=attestation_discovery,
        attestation_summary=attestation_summary,
    )


def format_doctor_report(report: DavidDoctorReport) -> str:
    """Render a stable, grep-friendly doctor report for terminal users."""

    lines = [
        "David model doctor",
        f"- workspace: {report.workspace_path}",
        f"- model: {report.model or '(not provided)'}",
    ]
    for check in report.checks:
        lines.append(f"- {check.name}: {check.status}: {check.detail}")
    if report.validation_summary is not None:
        summary = report.validation_summary
        warnings = "; ".join(summary.warnings) if summary.warnings else "none"
        reasons = (
            "; ".join(summary.rejection_reasons) if summary.rejection_reasons else "none"
        )
        lines.append("- validation report summary:")
        lines.append(f"  - validation_status: {summary.validation_status or 'unknown'}")
        lines.append(f"  - confidence: {summary.confidence or 'unknown'}")
        lines.append(f"  - auto_load_allowed: {summary.auto_load_allowed}")
        lines.append(f"  - harness_load_policy: {summary.harness_load_policy or 'unknown'}")
        lines.append(f"  - selected_config_layer_scope: {summary.layer_scope_text()}")
        lines.append(
            f"  - selected_adapter_config_id: "
            f"{summary.selected_adapter_config_id or 'unknown'}"
        )
        lines.append(f"  - selected_insertion_family: {summary.selected_insertion_family or 'unknown'}")
        lines.append(f"  - warnings: {warnings}")
        lines.append(f"  - rejection_reasons: {reasons}")
    if report.attestation_summary is not None:
        summary = report.attestation_summary
        warnings = "; ".join(summary.warnings) if summary.warnings else "none"
        reasons = (
            "; ".join(summary.rejection_reasons) if summary.rejection_reasons else "none"
        )
        refused = (
            ", ".join(summary.refused_replay_capabilities)
            if summary.refused_replay_capabilities
            else "none"
        )
        requested = (
            ", ".join(summary.requested_replay_capabilities)
            if summary.requested_replay_capabilities
            else "none"
        )
        capabilities = (
            ", ".join(summary.allowed_capabilities) if summary.allowed_capabilities else "none"
        )
        lines.append("- model attestation summary:")
        lines.append(f"  - attestation_status: {summary.attestation_status or 'unknown'}")
        lines.append(f"  - reviewer: {summary.reviewer or 'unknown'}")
        lines.append(f"  - rationale: {summary.rationale or 'unknown'}")
        lines.append(f"  - expires_at: {summary.expires_at or 'unknown'}")
        lines.append(f"  - model_identity: {summary.model_identity or 'unknown'}")
        lines.append(f"  - tokenizer_identity: {summary.tokenizer_identity or 'unknown'}")
        lines.append(f"  - model_revision: {summary.model_revision or 'unknown'}")
        lines.append(f"  - validation_report_digest: {summary.validation_report_digest or 'unknown'}")
        lines.append(f"  - selected_adapter_config_id: {summary.selected_adapter_config_id or 'unknown'}")
        lines.append(f"  - allowed_capabilities: {capabilities}")
        lines.append(f"  - standard_decode_only: {summary.standard_decode_only}")
        lines.append(f"  - tensor_replay_effective: {summary.tensor_replay_effective}")
        lines.append(f"  - requested_replay_capabilities: {requested}")
        lines.append(f"  - refused_replay_capabilities: {refused}")
        lines.append(f"  - warnings: {warnings}")
        lines.append(f"  - rejection_reasons: {reasons}")
    if report.validation_discovery.checked_paths:
        lines.append("- checked validation report paths:")
        lines.extend(f"  - {path}" for path in report.validation_discovery.checked_paths)
    if report.attestation_discovery and report.attestation_discovery.checked_paths:
        lines.append("- checked attestation paths:")
        lines.extend(f"  - {path}" for path in report.attestation_discovery.checked_paths)
    return "\n".join(lines) + "\n"


def summarize_model_attestation(path: str | Path) -> ModelAttestationSummary:
    """Read a manual-reviewed attestation without changing boot policy."""

    attestation_path = Path(path).expanduser()
    try:
        with attestation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        return _unreadable_attestation_summary(attestation_path, f"could not read attestation: {exc}")
    except json.JSONDecodeError as exc:
        return _unreadable_attestation_summary(attestation_path, f"invalid JSON: {exc}")
    return summarize_model_attestation_data(payload, path=attestation_path)


def summarize_model_attestation_data(
    payload: object,
    *,
    path: str | Path | None = None,
) -> ModelAttestationSummary:
    """Summarize an operator attestation while refusing replay capabilities."""

    attestation_path = Path(path).expanduser() if path is not None else None
    if not isinstance(payload, dict):
        return _unreadable_attestation_summary(
            attestation_path,
            "model attestation must be a JSON object",
        )

    schema_name = _optional_str(payload.get("schema_name"))
    status = _optional_str(payload.get("attestation_status") or payload.get("status"))
    if status is None and schema_name == "david.manual_model_attestation":
        status = "manual_reviewed"
    reviewer = _optional_str(payload.get("reviewer") or payload.get("reviewed_by"))
    rationale = _optional_str(payload.get("rationale") or payload.get("review_rationale"))
    expires_at = _optional_str(payload.get("expires_at"))
    allowed = tuple(_unique_strings(_string_list(payload.get("allowed_capabilities"))))
    declared_refused = tuple(
        capability
        for capability in _string_list(payload.get("refused_capabilities"))
        if capability in UNSAFE_REPLAY_CAPABILITIES
    )
    requested_replay = tuple(
        capability for capability in allowed if capability in UNSAFE_REPLAY_CAPABILITIES
    )
    refused = tuple(_unique_strings([*requested_replay, *declared_refused]))
    selected_config = payload.get("selected_config")
    selected_config_map = selected_config if isinstance(selected_config, dict) else None
    selected_config_sha256 = _optional_str(
        payload.get("selected_config_sha256") or payload.get("selected_config_hash")
    )
    model_identity = _identity_text(payload, "model_identity", ("model_id", "model_name"))
    tokenizer_identity = _identity_text(payload, "tokenizer_identity", ("tokenizer_id",))
    validation_report_digest = _validation_report_digest(payload)
    model_revision = _model_revision_text(payload)
    warnings = _unique_strings(_string_list(payload.get("warnings")))

    rejection_reasons: list[str] = []
    if schema_name not in ATTESTATION_SCHEMA_NAMES:
        rejection_reasons.append(f"unsupported schema {schema_name!r}")
    if status not in {"manual_reviewed", "operator_reviewed", "approved"}:
        rejection_reasons.append(
            f"attestation_status is {status!r}, expected 'manual_reviewed'"
        )
    if not reviewer:
        rejection_reasons.append("reviewer is missing")
    if not rationale:
        rejection_reasons.append("rationale is missing")
    if not model_identity:
        rejection_reasons.append("model_identity is missing")
    if not tokenizer_identity:
        rejection_reasons.append("tokenizer_identity is missing")
    if not model_revision:
        rejection_reasons.append("model_revision is missing")
    if not validation_report_digest:
        rejection_reasons.append("validation_report_digest is missing")
    if selected_config_map is None and not selected_config_sha256:
        rejection_reasons.append("selected_config or selected_config_sha256 is missing")
    if not any(capability in STANDARD_DECODE_CAPABILITIES for capability in allowed):
        rejection_reasons.append("standard_decode is not allowed by attestation")

    return ModelAttestationSummary(
        path=attestation_path,
        readable=True,
        attestation_status=status,
        reviewer=reviewer,
        rationale=rationale,
        expires_at=expires_at,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        model_revision=model_revision,
        validation_report_digest=validation_report_digest,
        selected_adapter_config_id=(
            _optional_str(selected_config_map.get("adapter_config_id"))
            if selected_config_map is not None
            else selected_config_sha256
        ),
        allowed_capabilities=allowed,
        requested_replay_capabilities=requested_replay,
        refused_replay_capabilities=refused,
        warnings=tuple(warnings),
        rejection_reasons=tuple(_unique_strings(rejection_reasons)),
    )


def _discover_report(
    *,
    model: str | None,
    workspace_path: Path,
    validation_report: str | None,
) -> ValidationReportDiscovery:
    if validation_report:
        report = Path(validation_report).expanduser()
        return ValidationReportDiscovery(path=report if report.is_file() else None, checked_paths=(report,))
    return discover_validation_report(model_path=model, workspace_path=workspace_path)


def _discover_attestation(
    *,
    model: str | None,
    workspace_path: Path,
    validation_report_path: Path | None,
    attestation_path: str | None,
) -> AttestationDiscovery:
    if attestation_path:
        candidate = Path(attestation_path).expanduser()
        return AttestationDiscovery(
            path=candidate if candidate.is_file() else None,
            checked_paths=(candidate,),
            explicit=True,
        )

    checked_paths: list[Path] = []
    for candidate in _candidate_attestation_paths(
        model=model,
        workspace_path=workspace_path,
        validation_report_path=validation_report_path,
    ):
        checked_paths.append(candidate)
        if candidate.is_file():
            return AttestationDiscovery(path=candidate, checked_paths=tuple(checked_paths))
    return AttestationDiscovery(path=None, checked_paths=tuple(checked_paths))


def _candidate_attestation_paths(
    *,
    model: str | None,
    workspace_path: Path,
    validation_report_path: Path | None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if model:
        model_path = Path(model).expanduser()
        if model_path.exists() or model_path.is_absolute():
            model_root = model_path if model_path.is_dir() else model_path.parent
            candidates.extend(model_root / name for name in ATTESTATION_FILENAMES)

    if validation_report_path is not None:
        candidates.extend(validation_report_path.parent / name for name in ATTESTATION_FILENAMES)

    workspace_root = workspace_path.expanduser().resolve()
    candidates.extend(workspace_root / path for path in WORKSPACE_ATTESTATION_PATHS)
    return tuple(dict.fromkeys(candidates))


def _workspace_check(workspace_root: Path) -> DoctorCheck:
    if workspace_root.is_dir():
        return DoctorCheck("workspace", "ready", str(workspace_root))
    if workspace_root.exists():
        return DoctorCheck("workspace", "blocked", "path exists but is not a directory")
    return DoctorCheck("workspace", "missing", "workspace directory does not exist")


def _model_location_check(model: str | None) -> DoctorCheck:
    if not model:
        return DoctorCheck("model location", "missing", "--model was not provided")
    model_path = Path(model).expanduser()
    if _looks_like_hf_model_id(model):
        return DoctorCheck("model location", "review", "HF model id; checking local cache only")
    if _looks_like_unix_absolute_path(model) and not model_path.exists():
        return _wsl_model_location_check(model)
    if not model_path.exists():
        return DoctorCheck("model location", "missing", "local model path does not exist")
    if not model_path.is_dir():
        return DoctorCheck("model location", "blocked", "local model path is not a directory")
    if is_vindex_artifact_path(model_path):
        metadata = inspect_vindex_artifact(model_path)
        if metadata.available:
            return DoctorCheck(
                "model location",
                "review",
                ".vindex.ple artifact path; supports evidence/materialization only; "
                f"direct generation unsupported; {_format_vindex_artifact_detail(metadata)}",
            )
        return DoctorCheck(
            "model location",
            "blocked",
            ".vindex.ple artifact path is incomplete; "
            f"{_format_vindex_artifact_detail(metadata)}; direct generation unsupported",
        )
    complete, detail = _local_model_completeness(model_path)
    if complete:
        return DoctorCheck("model location", "ready", f"local HF-style model directory; {detail}")
    return DoctorCheck("model location", "review", f"local directory exists but HF files are incomplete; {detail}")


def _hf_snapshot_check(model: str | None) -> DoctorCheck:
    if not model:
        return DoctorCheck("HF snapshot", "missing", "--model was not provided")
    model_path = Path(model).expanduser()
    if model_path.exists():
        if is_vindex_artifact_path(model_path):
            metadata = inspect_vindex_artifact(model_path)
            source = f"; source={metadata.source_hf_path}" if metadata.source_hf_path else ""
            return DoctorCheck(
                "HF snapshot",
                "review",
                "not applicable for direct .vindex.ple artifact path"
                f"{source}; use a local HF checkpoint for direct generation",
            )
        complete, detail = _local_model_completeness(model_path)
        status = "ready" if complete else "review"
        return DoctorCheck("HF snapshot", status, detail)
    if _looks_like_unix_absolute_path(model):
        return DoctorCheck("HF snapshot", "review", "not applicable for WSL local model path")
    if not _looks_like_hf_model_id(model):
        return DoctorCheck("HF snapshot", "missing", "model is neither an existing path nor an HF id")

    cache_root = _huggingface_cache_root()
    model_cache = cache_root / f"models--{model.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if not snapshots.is_dir():
        return DoctorCheck("HF snapshot", "missing", f"no local HF cache snapshots under {snapshots}")

    complete = []
    incomplete = []
    for snapshot in sorted(path for path in snapshots.iterdir() if path.is_dir()):
        is_complete, detail = _local_model_completeness(snapshot)
        if is_complete:
            complete.append((snapshot, detail))
        else:
            incomplete.append(snapshot)
    if complete:
        snapshot, completeness = complete[-1]
        return DoctorCheck("HF snapshot", "ready", f"{snapshot}; {completeness}")
    return DoctorCheck(
        "HF snapshot",
        "blocked",
        f"{len(incomplete)} snapshot(s) found but none had config, tokenizer, and weights",
    )


def _wsl_hf_snapshot_check(model: str | None) -> DoctorCheck:
    name = "WSL HF snapshot"
    if not model:
        return DoctorCheck(name, "review", "--model was not provided")
    if not _looks_like_hf_model_id(model):
        return DoctorCheck(name, "review", "not applicable; model is not an HF id")

    model_cache = f"models--{model.replace('/', '--')}"
    script = f"""
import json
from pathlib import Path

model_cache = {model_cache!r}
snapshots = Path.home() / ".cache" / "huggingface" / "hub" / model_cache / "snapshots"

def completeness(path):
    has_config = (path / "config.json").is_file()
    has_tokenizer = any(
        (path / item).is_file()
        for item in ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json")
    )
    has_weights = any(path.glob(pattern) for pattern in ("*.safetensors", "*.bin", "*.gguf"))
    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_tokenizer:
        missing.append("tokenizer files")
    if not has_weights:
        missing.append("weight files")
    return not missing, ", ".join(missing)

payload = {{"cache": str(snapshots), "snapshot_count": 0, "complete": False}}
if snapshots.is_dir():
    snapshot_dirs = sorted(path for path in snapshots.iterdir() if path.is_dir())[:64]
    payload["snapshot_count"] = len(snapshot_dirs)
    incomplete = []
    for snapshot in snapshot_dirs:
        complete, missing = completeness(snapshot)
        if complete:
            payload.update({{
                "complete": True,
                "snapshot": str(snapshot),
                "detail": "config, tokenizer, and weights present",
            }})
            break
        incomplete.append({{"path": str(snapshot), "missing": missing}})
    if not payload["complete"]:
        payload["incomplete"] = incomplete[:5]

print(json.dumps(payload))
"""
    payload, error = _run_wsl_python_json(script)
    if error is not None:
        return DoctorCheck(name, "review", error)
    if payload is None:
        return DoctorCheck(name, "review", "wsl probe returned no data")

    if payload.get("complete"):
        snapshot = payload.get("snapshot", "(unknown snapshot)")
        detail = payload.get("detail", "config, tokenizer, and weights present")
        return DoctorCheck(name, "ready", f"{snapshot}; {detail}")

    cache = payload.get("cache", f"/home/<user>/.cache/huggingface/hub/{model_cache}/snapshots")
    snapshot_count = int(payload.get("snapshot_count") or 0)
    if snapshot_count:
        incomplete = payload.get("incomplete")
        example = ""
        if isinstance(incomplete, list) and incomplete:
            first = incomplete[0]
            if isinstance(first, dict):
                example = f"; first incomplete snapshot missing {first.get('missing', 'unknown files')}"
        return DoctorCheck(
            name,
            "review",
            f"{snapshot_count} WSL snapshot(s) found under {cache}, none complete{example}",
        )
    return DoctorCheck(name, "review", f"no local WSL HF cache snapshots under {cache}")


def _vindex_check(workspace_root: Path, model: str | None) -> DoctorCheck:
    candidates: list[Path] = []
    if workspace_root.is_dir():
        candidates.extend(workspace_root.glob("*.vindex.ple"))
    if model:
        model_path = Path(model).expanduser()
        if model_path.exists():
            if is_vindex_artifact_path(model_path):
                candidates.append(model_path)
            candidates.extend(model_path.parent.glob("*.vindex.ple"))
            adjacent = model_path.with_suffix(model_path.suffix + ".vindex.ple")
            if adjacent.exists():
                candidates.append(adjacent)
    unique = tuple(dict.fromkeys(path.resolve() for path in candidates if path.exists()))
    if unique:
        metadata = [inspect_vindex_artifact(path) for path in unique[:3]]
        names = ", ".join(_format_vindex_artifact_detail(item) for item in metadata)
        suffix = "" if len(unique) <= 3 else f" (+{len(unique) - 3} more)"
        if any(item.available for item in metadata):
            return DoctorCheck(".vindex.ple artifact", "ready", f"{names}{suffix}")
        return DoctorCheck(".vindex.ple artifact", "blocked", f"{names}{suffix}")
    return DoctorCheck(".vindex.ple artifact", "missing", "no model vector index artifact found nearby")


def _validation_report_check(
    discovery: ValidationReportDiscovery,
    summary: ValidationReportSummary | None,
) -> DoctorCheck:
    if discovery.path is not None:
        if summary is None:
            return DoctorCheck("validation report", "ready", str(discovery.path))
        if summary.can_auto_load:
            return DoctorCheck(
                "validation report",
                "ready",
                f"{discovery.path}; {summary.compact_detail()}",
            )
        status = "blocked" if summary.error or summary.rejection_reasons else "review"
        return DoctorCheck(
            "validation report",
            status,
            f"{discovery.path}; {summary.compact_detail()}",
        )
    return DoctorCheck("validation report", "missing", "no boot-safe validation report discovered")


def _attestation_check(
    discovery: AttestationDiscovery,
    summary: ModelAttestationSummary | None,
) -> DoctorCheck:
    name = "model attestation"
    if discovery.path is None:
        if discovery.explicit:
            return DoctorCheck(name, "missing", "explicit attestation path was not found")
        return DoctorCheck(
            name,
            "review",
            "no manual-reviewed attestation discovered; accepted validator report remains the auto-load path",
        )
    if summary is None:
        status = "blocked" if discovery.explicit else "review"
        return DoctorCheck(name, status, f"{discovery.path}; attestation could not be summarized")

    if summary.error or summary.rejection_reasons:
        status = "blocked" if discovery.explicit else "review"
    elif summary.requested_replay_capabilities:
        status = "review"
    else:
        status = "ready"

    return DoctorCheck(
        name,
        status,
        f"{discovery.path}; {summary.compact_detail()}; "
        "manual attestation permits standard decode only; unsafe replay capabilities refused",
    )


def _optional_package_checks() -> tuple[DoctorCheck, ...]:
    packages = (
        "torch",
        "transformers",
        "pydantic",
        "safetensors",
        "accelerate",
        "huggingface_hub",
    )
    checks = []
    for package in packages:
        status = "ready" if importlib.util.find_spec(package) is not None else "missing"
        detail = "importable" if status == "ready" else "not installed in this Python environment"
        checks.append(DoctorCheck(f"package {package}", status, detail))
    return tuple(checks)


def _torch_cuda_check_compat(validation_report_path: Path | None) -> DoctorCheck:
    try:
        return _torch_cuda_check(validation_report_path=validation_report_path)
    except TypeError:
        # Older focused tests monkeypatch _torch_cuda_check with a zero-arg stub.
        return _torch_cuda_check()  # type: ignore[call-arg]


def _torch_cuda_check(*, validation_report_path: Path | None = None) -> DoctorCheck:
    if importlib.util.find_spec("torch") is None:
        return DoctorCheck("torch CUDA", "missing", "torch is not installed")
    try:
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - defensive for broken installs.
        return DoctorCheck("torch CUDA", "blocked", f"torch import failed: {exc}")

    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return DoctorCheck("torch CUDA", "blocked", "torch.cuda API is unavailable")
    estimate = _validation_report_model_memory_estimate(validation_report_path)
    if not is_available():
        detail = _torch_runtime_detail(
            cuda_available=False,
            device_count=0,
            selected_device="cpu",
            devices=(),
            dtype_support=_torch_dtype_support(torch, cuda, cuda_available=False),
            estimate=estimate,
        )
        return DoctorCheck(
            "torch CUDA",
            "review",
            f"CPU-only torch; real Gemma boot may be too slow or impossible; {detail}",
        )

    count = _safe_int_call(getattr(cuda, "device_count", None))
    selected_index = _safe_int_call(getattr(cuda, "current_device", None))
    if selected_index is None and count:
        selected_index = 0
    devices = _torch_cuda_devices(cuda, count or 0)
    selected = next(
        (device for device in devices if device.get("index") == selected_index),
        devices[0] if devices else None,
    )
    feasible = _selected_device_feasibility(selected, estimate)
    status = "review" if feasible.startswith("blocked") else "ready"
    detail = _torch_runtime_detail(
        cuda_available=True,
        device_count=count,
        selected_device=f"cuda:{selected_index}" if selected_index is not None else "cuda:unknown",
        devices=devices,
        dtype_support=_torch_dtype_support(torch, cuda, cuda_available=True),
        estimate=estimate,
        selected_feasibility=feasible,
    )
    return DoctorCheck("torch CUDA", status, f"CUDA available; {detail}")


def _torch_cuda_devices(cuda: object, count: int) -> tuple[dict[str, object], ...]:
    devices: list[dict[str, object]] = []
    for index in range(max(count, 0)):
        device: dict[str, object] = {"index": index}
        get_name = getattr(cuda, "get_device_name", None)
        if callable(get_name):
            try:
                device["name"] = str(get_name(index))
            except Exception:
                device["name"] = "unknown"
        props = None
        get_props = getattr(cuda, "get_device_properties", None)
        if callable(get_props):
            try:
                props = get_props(index)
            except Exception:
                props = None
        total_memory = getattr(props, "total_memory", None) if props is not None else None
        if isinstance(total_memory, int):
            device["total_vram_mib"] = round(total_memory / (1024**2))
        capability = _device_capability(cuda, index)
        if capability is not None:
            device["capability"] = capability
        free_memory = _device_free_memory(cuda, index)
        if free_memory is not None:
            device["free_vram_mib"] = round(free_memory / (1024**2))
        devices.append(device)
    return tuple(devices)


def _torch_dtype_support(torch: object, cuda: object, *, cuda_available: bool) -> dict[str, object]:
    support: dict[str, object] = {
        "float16": "cuda" if cuda_available and hasattr(torch, "float16") else "unknown",
        "bfloat16": "unknown",
    }
    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            support["bfloat16"] = bool(is_bf16_supported())
        except Exception:
            support["bfloat16"] = "unknown"
    elif cuda_available:
        capability = _device_capability(cuda, 0)
        if capability is not None:
            major = int(capability.split(".", 1)[0])
            support["bfloat16"] = major >= 8
    return support


def _torch_runtime_detail(
    *,
    cuda_available: bool,
    device_count: int | None,
    selected_device: str,
    devices: tuple[dict[str, object], ...],
    dtype_support: dict[str, object],
    estimate: dict[str, object] | None,
    selected_feasibility: str | None = None,
) -> str:
    parts = [
        f"cuda_available={cuda_available}",
        f"devices={device_count if device_count is not None else 'unknown'}",
        f"selected_device={selected_device}",
        f"selected_device_feasibility={selected_feasibility or 'unknown'}",
        f"float16_support={dtype_support.get('float16', 'unknown')}",
        f"bfloat16_support={dtype_support.get('bfloat16', 'unknown')}",
    ]
    device_text = _format_cuda_devices(devices)
    if device_text:
        parts.append(f"available_devices=[{device_text}]")
    if estimate is not None:
        parts.append(
            "rough_model_memory_estimate="
            f"{estimate.get('estimated_weight_mib')}MiB"
            f"/{estimate.get('dtype', 'unknown')}"
            f"; estimate_source={estimate.get('source', 'unknown')}"
        )
        evidence = estimate.get("evidence")
        if evidence:
            parts.append(f"estimate_evidence={evidence}")
    return "; ".join(parts)


def _format_cuda_devices(devices: tuple[dict[str, object], ...]) -> str:
    formatted = []
    for device in devices:
        fields = [f"cuda:{device.get('index', '?')}"]
        if device.get("name"):
            fields.append(str(device["name"]))
        if device.get("total_vram_mib") is not None:
            fields.append(f"total={device['total_vram_mib']}MiB")
        if device.get("free_vram_mib") is not None:
            fields.append(f"free={device['free_vram_mib']}MiB")
        if device.get("capability"):
            fields.append(f"cc={device['capability']}")
        formatted.append("(" + ", ".join(fields) + ")")
    return ", ".join(formatted)


def _selected_device_feasibility(
    selected: dict[str, object] | None,
    estimate: dict[str, object] | None,
) -> str:
    if selected is None:
        return "unknown_no_selected_cuda_device"
    if estimate is None:
        return "unknown_no_model_estimate"
    needed_mib = estimate.get("estimated_weight_mib")
    if not isinstance(needed_mib, int):
        return "unknown_no_model_estimate"
    available_mib = selected.get("free_vram_mib") or selected.get("total_vram_mib")
    if not isinstance(available_mib, int):
        return "unknown_no_vram"
    if needed_mib <= int(available_mib * 0.85):
        return "likely_feasible_for_weights"
    return "blocked_vram_below_weight_estimate"


def _validation_report_model_memory_estimate(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, Mapping):
        return None

    model = _mapping_or_none(report.get("model"))
    topology = _mapping_or_none(report.get("topology"))
    selected_config = _mapping_or_none(report.get("selected_config"))
    if selected_config is None:
        selected_config = _mapping_or_none(report.get("adapter_config_candidate"))
    provenance = _mapping_or_none(report.get("provenance"))
    loader_options = _mapping_or_none(provenance.get("loader_options")) if provenance else None

    hidden_size = _first_int(
        model,
        topology,
        selected_config,
        keys=("hidden_size", "route_dimension", "d_model", "n_embd"),
    )
    num_layers = _first_int(model, topology, keys=("num_layers", "num_hidden_layers", "n_layer"))
    vocab_size = _first_int(model, topology, keys=("vocab_size",))
    parameter_count = _first_int(model, topology, keys=("parameter_count", "num_parameters"))
    dtype = (
        _first_text(model, loader_options, keys=("dtype", "requested_dtype"))
        or "float16"
    )
    bytes_per_param = _dtype_bytes(dtype)
    if parameter_count is not None:
        estimated_mib = round(parameter_count * bytes_per_param / (1024**2))
        return {
            "estimated_weight_mib": estimated_mib,
            "dtype": dtype,
            "source": "validation_report_parameter_count",
            "evidence": f"parameter_count={parameter_count}",
        }
    if hidden_size is None or num_layers is None:
        return None

    rough_params = num_layers * hidden_size * hidden_size * 12
    evidence = f"hidden_size={hidden_size}, num_layers={num_layers}"
    if vocab_size is not None:
        rough_params += vocab_size * hidden_size
        evidence = f"{evidence}, vocab_size={vocab_size}"
    estimated_mib = round(rough_params * bytes_per_param / (1024**2))
    return {
        "estimated_weight_mib": estimated_mib,
        "dtype": dtype,
        "source": "validation_report_topology_rough",
        "evidence": evidence,
    }


def _mapping_or_none(value: object) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _first_int(*sources: Mapping[str, Any] | None, keys: tuple[str, ...]) -> int | None:
    for source in sources:
        if source is None:
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                continue
            try:
                return int(value) if value is not None else None
            except (TypeError, ValueError):
                continue
    return None


def _first_text(*sources: Mapping[str, Any] | None, keys: tuple[str, ...]) -> str | None:
    for source in sources:
        if source is None:
            continue
        for key in keys:
            text = _optional_str(source.get(key))
            if text:
                return text
    return None


def _dtype_bytes(dtype: str) -> int:
    normalized = dtype.lower().replace("torch.", "")
    if normalized in {"float16", "bfloat16", "fp16", "bf16", "half", "f16"}:
        return 2
    if normalized in {"float32", "fp32", "f32", "single"}:
        return 4
    if normalized in {"int8", "uint8", "i8", "u8"}:
        return 1
    return 2


def _safe_int_call(callable_value: object) -> int | None:
    if not callable(callable_value):
        return None
    try:
        value = callable_value()
    except Exception:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _device_capability(cuda: object, index: int) -> str | None:
    get_capability = getattr(cuda, "get_device_capability", None)
    if not callable(get_capability):
        return None
    try:
        capability = get_capability(index)
    except Exception:
        return None
    if isinstance(capability, tuple) and len(capability) >= 2:
        return f"{capability[0]}.{capability[1]}"
    return None


def _device_free_memory(cuda: object, index: int) -> int | None:
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if not callable(mem_get_info):
        return None
    try:
        memory = mem_get_info(index)
    except TypeError:
        try:
            memory = mem_get_info()
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(memory, tuple) and memory:
        try:
            return int(memory[0])
        except (TypeError, ValueError):
            return None
    return None


def _wsl_python_packages_check() -> DoctorCheck:
    name = "WSL Python packages"
    script = """
import importlib.util
import json

packages = ("torch", "transformers", "pydantic", "accelerate")
available = {package: importlib.util.find_spec(package) is not None for package in packages}
missing = [package for package, present in available.items() if not present]
cuda_status = "not_checked"
cuda_detail = "torch is not importable"
if available.get("torch"):
    try:
        import torch

        cuda = getattr(torch, "cuda", None)
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available) and is_available():
            device_count = getattr(cuda, "device_count", lambda: "?")()
            cuda_status = "ready"
            cuda_detail = f"CUDA available; devices={device_count}"
        elif callable(is_available):
            cuda_status = "cpu_only"
            cuda_detail = "torch importable; CUDA unavailable"
        else:
            cuda_status = "review"
            cuda_detail = "torch.cuda API is unavailable"
    except Exception as exc:
        cuda_status = "review"
        cuda_detail = f"torch import failed: {exc}"

print(json.dumps({"missing": missing, "cuda_status": cuda_status, "cuda_detail": cuda_detail}))
"""
    payload, error = _run_wsl_python_json(script)
    if error is not None:
        return DoctorCheck(name, "review", error)
    if payload is None:
        return DoctorCheck(name, "review", "wsl probe returned no data")

    missing = payload.get("missing")
    missing_packages = tuple(str(item) for item in missing) if isinstance(missing, list) else ()
    cuda_status = str(payload.get("cuda_status") or "not_checked")
    cuda_detail = str(payload.get("cuda_detail") or "CUDA status unavailable")
    if missing_packages:
        return DoctorCheck(name, "review", f"missing packages: {', '.join(missing_packages)}; {cuda_detail}")
    status = "ready" if cuda_status == "ready" else "review"
    return DoctorCheck(name, status, f"torch, transformers, pydantic, accelerate importable; {cuda_detail}")


def _auto_validate_check(
    *,
    model: str | None,
    workspace_path: Path,
    discovery: ValidationReportDiscovery,
    auto_validate_model: bool,
    repo_root: Path,
) -> DoctorCheck:
    scan_report, validation_report = workspace_auto_model_report_paths(workspace_path)
    getter = repo_root / GET_MODEL_CONFIG_RELATIVE_PATH
    validator = repo_root / VALIDATE_MODEL_CONFIG_RELATIVE_PATH
    blockers = []
    if not model:
        blockers.append("--model missing")
    if not getter.is_file():
        blockers.append(f"missing {getter}")
    if not validator.is_file():
        blockers.append(f"missing {validator}")
    for package in ("torch", "transformers"):
        if importlib.util.find_spec(package) is None:
            blockers.append(f"missing package {package}")

    if blockers:
        return DoctorCheck("--auto-validate-model", "blocked", "; ".join(blockers))
    if discovery.path is not None:
        return DoctorCheck("--auto-validate-model", "ready", "not needed; validation report already discovered")
    if model and is_vindex_artifact_path(Path(model).expanduser()):
        status = "blocked" if auto_validate_model else "review"
        return DoctorCheck(
            "--auto-validate-model",
            status,
            "not applicable for direct .vindex.ple artifact path; "
            "run model scan/validate against the source HF checkpoint",
        )
    prefix = "requested; " if auto_validate_model else "available; "
    return DoctorCheck(
        "--auto-validate-model",
        "ready",
        f"{prefix}would write {scan_report} and {validation_report}",
    )


def _wsl_tooling_check() -> DoctorCheck:
    wsl = shutil.which("wsl")
    if wsl is None:
        return DoctorCheck("WSL/tooling", "review", "wsl.exe is not on PATH")
    tinydex = Path("C:/Users/jehma/Desktop/TinyTool/bin/tinydex")
    if tinydex.exists():
        return DoctorCheck("WSL/tooling", "ready", f"wsl.exe found; tinydex found at {tinydex}")
    return DoctorCheck("WSL/tooling", "review", "wsl.exe found; tinydex path not found")


def _run_wsl_python_json(
    script: str,
    *,
    timeout_seconds: float | None = None,
) -> tuple[dict[str, object] | None, str | None]:
    wsl = shutil.which("wsl")
    if wsl is None:
        return None, "wsl.exe is not on PATH"
    timeout = timeout_seconds if timeout_seconds is not None else _wsl_probe_timeout_seconds()
    command = (wsl, "bash", "-lc", f"python3 - <<'PY'\n{script}\nPY")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"wsl probe timed out after {timeout:g}s"
    except OSError as exc:
        return None, f"wsl probe failed to start: {exc}"

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        detail = stderr or stdout or "no output"
        return None, f"wsl probe failed rc={result.returncode}: {detail}"
    if not stdout:
        return None, "wsl probe returned empty output"
    try:
        return json.loads(stdout.splitlines()[-1]), None
    except json.JSONDecodeError as exc:
        return None, f"wsl probe returned invalid JSON: {exc}"


def _wsl_probe_timeout_seconds() -> float:
    raw = os.environ.get(WSL_PROBE_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_WSL_PROBE_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        return DEFAULT_WSL_PROBE_TIMEOUT_SECONDS
    if timeout <= 0:
        return DEFAULT_WSL_PROBE_TIMEOUT_SECONDS
    return timeout


def _format_vindex_artifact_detail(metadata: VindexArtifactMetadata) -> str:
    missing = ", ".join(metadata.missing_files) if metadata.missing_files else "none"
    missing_binary = ", ".join(metadata.missing_binary_files) if metadata.missing_binary_files else "none"
    source = metadata.source_hf_path or "unknown"
    return (
        f"{metadata.path}; available={metadata.available}; family={metadata.family or 'unknown'}; "
        f"layers={metadata.layers or 'unknown'}; hidden_size={metadata.hidden_size or 'unknown'}; "
        f"total_bytes={metadata.total_bytes}; source={source}; "
        f"manifest_files={len(metadata.manifest_declared_files)}; "
        f"missing={missing}; missing_binary={missing_binary}"
    )


def _local_model_completeness(model_path: Path) -> tuple[bool, str]:
    has_config = (model_path / "config.json").is_file()
    has_tokenizer = any(
        (model_path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json")
    )
    has_weights = any(model_path.glob(pattern) for pattern in ("*.safetensors", "*.bin", "*.gguf"))
    missing = []
    if not has_config:
        missing.append("config.json")
    if not has_tokenizer:
        missing.append("tokenizer files")
    if not has_weights:
        missing.append("weight files")
    if missing:
        return False, f"missing {', '.join(missing)}"
    return True, "config, tokenizer, and weights present"


def _wsl_model_location_check(model: str) -> DoctorCheck:
    script = f"""
import json
from pathlib import Path

model_path = Path({model!r}).expanduser()
payload = {{"path": str(model_path), "exists": model_path.exists()}}
if payload["exists"]:
    payload["is_dir"] = model_path.is_dir()
    if payload["is_dir"]:
        has_config = (model_path / "config.json").is_file()
        has_tokenizer = any(
            (model_path / name).is_file()
            for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json")
        )
        has_weights = any(model_path.glob(pattern) for pattern in ("*.safetensors", "*.bin", "*.gguf"))
        missing = []
        if not has_config:
            missing.append("config.json")
        if not has_tokenizer:
            missing.append("tokenizer files")
        if not has_weights:
            missing.append("weight files")
        payload["complete"] = not missing
        payload["detail"] = (
            "config, tokenizer, and weights present"
            if not missing
            else "missing " + ", ".join(missing)
        )

print(json.dumps(payload))
"""
    payload, error = _run_wsl_python_json(script)
    if error is not None:
        return DoctorCheck("model location", "review", f"WSL local model path; {error}")
    if payload is None:
        return DoctorCheck("model location", "review", "WSL local model path; wsl probe returned no data")

    path = str(payload.get("path") or model)
    if not payload.get("exists"):
        return DoctorCheck("model location", "missing", f"WSL local model path does not exist: {path}")
    if not payload.get("is_dir"):
        return DoctorCheck("model location", "blocked", f"WSL local model path is not a directory: {path}")

    detail = str(payload.get("detail") or "HF file completeness unavailable")
    if payload.get("complete"):
        return DoctorCheck("model location", "ready", f"WSL local HF-style model directory: {path}; {detail}")
    return DoctorCheck("model location", "review", f"WSL local directory exists: {path}; {detail}")


def _looks_like_unix_absolute_path(model: str) -> bool:
    return model.startswith("/")


def _looks_like_hf_model_id(model: str) -> bool:
    return (
        "/" in model
        and not _looks_like_unix_absolute_path(model)
        and not any(sep in model for sep in ("\\", "./", "../"))
        and not Path(model).exists()
    )


def _huggingface_cache_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return (Path.home() / ".cache" / "huggingface" / "hub").expanduser()


def _repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _unreadable_attestation_summary(
    path: Path | None,
    reason: str,
) -> ModelAttestationSummary:
    return ModelAttestationSummary(
        path=path,
        readable=False,
        attestation_status=None,
        reviewer=None,
        rationale=None,
        expires_at=None,
        model_identity=None,
        tokenizer_identity=None,
        model_revision=None,
        validation_report_digest=None,
        selected_adapter_config_id=None,
        allowed_capabilities=(),
        requested_replay_capabilities=(),
        refused_replay_capabilities=(),
        warnings=(),
        rejection_reasons=(reason,),
        error=reason,
    )


def _identity_text(
    payload: Mapping[str, Any],
    mapping_key: str,
    fallback_keys: tuple[str, ...],
) -> str | None:
    value = payload.get(mapping_key)
    if isinstance(value, Mapping):
        for key in ("id", "model_id", "tokenizer_id", "name", "repo_id"):
            text = _optional_str(value.get(key))
            if text:
                return text
    text = _optional_str(value)
    if text:
        return text
    for key in fallback_keys:
        text = _optional_str(payload.get(key))
        if text:
            return text
    return None


def _model_revision_text(payload: Mapping[str, Any]) -> str | None:
    for key in ("model_revision_or_hash", "model_revision", "revision", "model_hash", "commit_hash"):
        text = _optional_str(payload.get(key))
        if text:
            return text
    value = payload.get("model_identity")
    if isinstance(value, Mapping):
        for key in ("revision", "model_revision", "hash", "sha256", "commit_hash"):
            text = _optional_str(value.get(key))
            if text:
                return text
    return None


def _validation_report_digest(payload: Mapping[str, Any]) -> str | None:
    for key in ("validation_report_digest", "validation_report_sha256", "report_sha256"):
        text = _optional_str(payload.get(key))
        if text:
            return text
    value = payload.get("validation_report")
    if isinstance(value, Mapping):
        for key in ("sha256", "digest", "report_sha256"):
            text = _optional_str(value.get(key))
            if text:
                return text
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _unique_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
