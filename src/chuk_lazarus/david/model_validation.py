"""Validation-report discovery for the David CLI boot lifecycle."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MODEL_REPORT_FILENAMES = ("validation_report.json", "model_validation_report.json")
WORKSPACE_REPORT_PATHS = (Path(".david") / "model_validation_report.json",)
WORKSPACE_AUTO_SCAN_REPORT_PATH = Path(".david") / "model" / "model_config_report.json"
WORKSPACE_AUTO_VALIDATION_REPORT_PATH = (
    Path(".david") / "model_validation" / "model_validation_report.json"
)
GET_MODEL_CONFIG_RELATIVE_PATH = Path("David") / "get_model_config.py"
VALIDATE_MODEL_CONFIG_RELATIVE_PATH = Path("David") / "validate_model_config.py"
VALIDATION_REPORT_SCHEMA = "lazarus.model_config_validation_report"
ACCEPTED_STATUS = "accepted"


@dataclass(frozen=True)
class ValidationReportSummary:
    """Human-readable startup summary of a validator JSON report."""

    path: Path | None
    readable: bool
    validation_status: str | None
    confidence: str | None
    auto_load_allowed: bool | None
    harness_load_policy: str | None
    selected_config_layer_scope: dict[str, int | None]
    selected_adapter_config_id: str | None
    selected_insertion_family: str | None
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    error: str | None = None

    @property
    def can_auto_load(self) -> bool:
        return (
            self.readable
            and self.validation_status == ACCEPTED_STATUS
            and self.auto_load_allowed is True
            and not self.rejection_reasons
        )

    def layer_scope_text(self) -> str:
        if not self.selected_config_layer_scope:
            return "none"
        parts = []
        for name in (
            "route_layer",
            "boundary_layer",
            "residual_capture_layer",
            "kv_source_layer",
            "kv_target_layer",
            "injection_layer",
        ):
            value = self.selected_config_layer_scope.get(name)
            if value is not None:
                parts.append(f"{name}={value}")
        return ", ".join(parts) if parts else "none"

    def compact_detail(self) -> str:
        reasons = "; ".join(self.rejection_reasons) if self.rejection_reasons else "none"
        warnings = "; ".join(self.warnings[:3]) if self.warnings else "none"
        if len(self.warnings) > 3:
            warnings = f"{warnings}; +{len(self.warnings) - 3} more"
        return (
            f"validation_status={self.validation_status or 'unknown'}; "
            f"confidence={self.confidence or 'unknown'}; "
            f"auto_load_allowed={self.auto_load_allowed}; "
            f"harness_load_policy={self.harness_load_policy or 'unknown'}; "
            f"layer_scope={self.layer_scope_text()}; "
            f"warnings={warnings}; "
            f"rejection_reasons={reasons}"
        )


@dataclass(frozen=True)
class ValidationReportDiscovery:
    """Result of looking for a boot-safe model validation report."""

    path: Path | None
    checked_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ModelCommandResult:
    """Result of running an explicit standalone model-report command."""

    returncode: int
    command: tuple[str, ...]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class AutoModelValidationResult:
    """Result of the explicit scan-then-validate boot flow."""

    scan_report_path: Path
    validation_report_path: Path
    scan_result: ModelCommandResult
    validation_result: ModelCommandResult | None

    @property
    def returncode(self) -> int:
        if self.scan_result.returncode != 0:
            return self.scan_result.returncode
        if self.validation_result is None:
            return self.scan_result.returncode
        return self.validation_result.returncode


def summarize_validation_report(path: str | Path) -> ValidationReportSummary:
    """Read an existing validator report into a display-only boot summary."""

    report_path = Path(path).expanduser()
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except OSError as exc:
        return _unreadable_validation_summary(report_path, f"could not read report: {exc}")
    except json.JSONDecodeError as exc:
        return _unreadable_validation_summary(report_path, f"invalid JSON: {exc}")
    return summarize_validation_report_data(report, path=report_path)


def summarize_validation_report_data(
    report: Mapping[str, Any] | object,
    *,
    path: str | Path | None = None,
) -> ValidationReportSummary:
    """Summarize validator JSON without changing boot acceptance rules."""

    report_path = Path(path).expanduser() if path is not None else None
    if not isinstance(report, Mapping):
        return _unreadable_validation_summary(report_path, "validation report must be a JSON object")

    schema_name = _optional_str(report.get("schema_name"))
    validation_status = _optional_str(report.get("validation_status"))
    confidence = _optional_str(report.get("confidence"))
    auto_load_allowed = _optional_bool(report.get("auto_load_allowed"))
    harness_load_policy = _optional_str(report.get("harness_load_policy"))
    selected_config = _mapping_or_none(report.get("selected_config"))
    warnings = _unique_strings(_string_list(report.get("warnings")) + _gate_warnings(report))
    layer_scope = _selected_config_layer_scope(selected_config)

    rejection_reasons = []
    if schema_name != VALIDATION_REPORT_SCHEMA:
        rejection_reasons.append(f"unsupported schema {schema_name!r}")
    rejection_reasons.extend(
        _boot_failure_reasons(
            validation_status=validation_status,
            auto_load_allowed=auto_load_allowed is True,
            selected_config=selected_config,
        )
    )
    rejection_reasons.extend(_failure_reasons(report))

    return ValidationReportSummary(
        path=report_path,
        readable=True,
        validation_status=validation_status,
        confidence=confidence,
        auto_load_allowed=auto_load_allowed,
        harness_load_policy=harness_load_policy,
        selected_config_layer_scope=layer_scope,
        selected_adapter_config_id=_optional_str(selected_config.get("adapter_config_id"))
        if selected_config is not None
        else None,
        selected_insertion_family=_optional_str(selected_config.get("insertion_family"))
        if selected_config is not None
        else None,
        warnings=tuple(warnings),
        rejection_reasons=tuple(_unique_strings(rejection_reasons)),
    )


def discover_validation_report(
    *,
    model_path: str | None,
    workspace_path: Path,
) -> ValidationReportDiscovery:
    """Find the first obvious local validation report without scanning a model."""

    checked_paths: list[Path] = []
    for candidate in _candidate_report_paths(model_path=model_path, workspace_path=workspace_path):
        checked_paths.append(candidate)
        if candidate.is_file():
            return ValidationReportDiscovery(path=candidate, checked_paths=tuple(checked_paths))
    return ValidationReportDiscovery(path=None, checked_paths=tuple(checked_paths))


def run_model_scan(
    *,
    model: str,
    output: Path,
    repo_root: Path | None = None,
    extra_args: Sequence[str] = (),
) -> ModelCommandResult:
    """Run David/get_model_config.py explicitly for a user-requested model scan."""

    root = _repo_root(repo_root)
    script = root / GET_MODEL_CONFIG_RELATIVE_PATH
    command = (
        sys.executable,
        str(script),
        "--model",
        model,
        "--json-out",
        str(output),
        *tuple(extra_args),
    )
    return _run_standalone_model_command(command=command, script=script, cwd=root)


def run_model_validate(
    *,
    report: Path,
    output: Path,
    model: str | None = None,
    repo_root: Path | None = None,
    extra_args: Sequence[str] = (),
) -> ModelCommandResult:
    """Run David/validate_model_config.py explicitly for a user-requested validation."""

    root = _repo_root(repo_root)
    script = root / VALIDATE_MODEL_CONFIG_RELATIVE_PATH
    command = [
        sys.executable,
        str(script),
        "--config-report",
        str(report),
        "--json-out",
        str(output),
    ]
    if model:
        command.extend(("--model", model))
    command.extend(extra_args)
    return _run_standalone_model_command(command=tuple(command), script=script, cwd=root)


def workspace_auto_model_report_paths(workspace_path: Path) -> tuple[Path, Path]:
    """Return deterministic workspace artifact paths for auto model boot."""

    workspace_root = workspace_path.expanduser().resolve()
    return (
        workspace_root / WORKSPACE_AUTO_SCAN_REPORT_PATH,
        workspace_root / WORKSPACE_AUTO_VALIDATION_REPORT_PATH,
    )


def run_auto_model_validation(
    *,
    model: str,
    workspace_path: Path,
    repo_root: Path | None = None,
) -> AutoModelValidationResult:
    """Run scanner then validator for an explicitly requested boot validation."""

    scan_report, validation_report = workspace_auto_model_report_paths(workspace_path)
    scan_report.parent.mkdir(parents=True, exist_ok=True)
    validation_report.parent.mkdir(parents=True, exist_ok=True)

    scan_result = run_model_scan(model=model, output=scan_report, repo_root=repo_root)
    validation_result: ModelCommandResult | None = None
    if scan_result.returncode == 0:
        validation_result = run_model_validate(
            report=scan_report,
            output=validation_report,
            model=model,
            repo_root=repo_root,
        )
    return AutoModelValidationResult(
        scan_report_path=scan_report,
        validation_report_path=validation_report,
        scan_result=scan_result,
        validation_result=validation_result,
    )


def _candidate_report_paths(*, model_path: str | None, workspace_path: Path) -> tuple[Path, ...]:
    workspace_root = workspace_path.expanduser().resolve()
    candidates: list[Path] = []

    if model_path:
        model_root = Path(model_path).expanduser()
        if _looks_like_local_model_path(model_root):
            if not model_root.is_absolute():
                model_root = model_root.resolve()
            candidates.extend(model_root / name for name in MODEL_REPORT_FILENAMES)

    candidates.extend(workspace_root / path for path in WORKSPACE_REPORT_PATHS)
    candidates.append(workspace_root / WORKSPACE_AUTO_VALIDATION_REPORT_PATH)
    return tuple(candidates)


def _looks_like_local_model_path(model_root: Path) -> bool:
    if model_root.exists() or model_root.is_absolute():
        return True
    text = str(model_root)
    return text.startswith(".") or "\\" in text or "/" in text


def _repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _run_standalone_model_command(
    *,
    command: Sequence[str],
    script: Path,
    cwd: Path,
) -> ModelCommandResult:
    command_tuple = tuple(str(part) for part in command)
    if not script.is_file():
        return ModelCommandResult(
            returncode=2,
            command=command_tuple,
            stdout="",
            stderr=f"Standalone David model helper not found: {script}",
        )

    try:
        completed = subprocess.run(
            command_tuple,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return ModelCommandResult(
            returncode=2,
            command=command_tuple,
            stdout="",
            stderr=f"Failed to start standalone David model helper: {exc}",
        )

    return ModelCommandResult(
        returncode=completed.returncode,
        command=command_tuple,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _unreadable_validation_summary(
    path: Path | None,
    reason: str,
) -> ValidationReportSummary:
    return ValidationReportSummary(
        path=path,
        readable=False,
        validation_status=None,
        confidence=None,
        auto_load_allowed=None,
        harness_load_policy=None,
        selected_config_layer_scope={},
        selected_adapter_config_id=None,
        selected_insertion_family=None,
        warnings=(),
        rejection_reasons=(reason,),
        error=reason,
    )


def _boot_failure_reasons(
    *,
    validation_status: str | None,
    auto_load_allowed: bool,
    selected_config: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if validation_status != ACCEPTED_STATUS:
        reasons.append(f"validation_status is {validation_status!r}, expected 'accepted'")
    if not auto_load_allowed:
        reasons.append("auto_load_allowed is false")
    if selected_config is None:
        reasons.append("selected_config is missing")
    else:
        for field_name in (
            "route_layer",
            "boundary_layer",
            "kv_source_layer",
            "kv_target_layer",
            "insertion_family",
        ):
            if selected_config.get(field_name) is None:
                reasons.append(f"selected_config.{field_name} is missing")
    return reasons


def _selected_config_layer_scope(
    selected_config: Mapping[str, Any] | None,
) -> dict[str, int | None]:
    if selected_config is None:
        return {}
    return {
        "route_layer": _optional_int(selected_config.get("route_layer")),
        "boundary_layer": _optional_int(selected_config.get("boundary_layer")),
        "residual_capture_layer": _optional_int(selected_config.get("residual_capture_layer")),
        "kv_source_layer": _optional_int(selected_config.get("kv_source_layer")),
        "kv_target_layer": _optional_int(selected_config.get("kv_target_layer")),
        "injection_layer": _optional_int(selected_config.get("injection_layer")),
    }


def _gate_warnings(report: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    for gate_name in (
        "report_integrity",
        "topology_gate",
        "model_identity_gate",
        "projection_gate",
        "behavior_gate",
        "failure",
    ):
        gate = report.get(gate_name)
        if not isinstance(gate, Mapping):
            continue
        warnings.extend(_string_list(gate.get("warnings")))
        reason = _optional_str(gate.get("reason"))
        if reason:
            warnings.append(f"{gate_name}: {reason}")
    return warnings


def _failure_reasons(report: Mapping[str, Any]) -> list[str]:
    failure = report.get("failure")
    if not isinstance(failure, Mapping):
        return []
    reasons = []
    for key in ("reason", "message", "error", "error_type"):
        value = _optional_str(failure.get(key))
        if value:
            reasons.append(f"failure.{key}: {value}")
    return reasons


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
