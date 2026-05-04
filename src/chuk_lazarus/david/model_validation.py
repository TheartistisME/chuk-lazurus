"""Validation-report discovery for the David CLI boot lifecycle."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MODEL_REPORT_FILENAMES = ("validation_report.json", "model_validation_report.json")
WORKSPACE_REPORT_PATHS = (Path(".david") / "model_validation_report.json",)
WORKSPACE_AUTO_SCAN_REPORT_PATH = Path(".david") / "model" / "model_config_report.json"
WORKSPACE_AUTO_VALIDATION_REPORT_PATH = (
    Path(".david") / "model_validation" / "model_validation_report.json"
)
GET_MODEL_CONFIG_RELATIVE_PATH = Path("David") / "get_model_config.py"
VALIDATE_MODEL_CONFIG_RELATIVE_PATH = Path("David") / "validate_model_config.py"


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
