"""Validation-report discovery for the David CLI boot lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_REPORT_FILENAMES = ("validation_report.json", "model_validation_report.json")
WORKSPACE_REPORT_PATHS = (Path(".david") / "model_validation_report.json",)


@dataclass(frozen=True)
class ValidationReportDiscovery:
    """Result of looking for a boot-safe model validation report."""

    path: Path | None
    checked_paths: tuple[Path, ...]


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
    return tuple(candidates)


def _looks_like_local_model_path(model_root: Path) -> bool:
    if model_root.exists() or model_root.is_absolute():
        return True
    text = str(model_root)
    return text.startswith(".") or "\\" in text or "/" in text
