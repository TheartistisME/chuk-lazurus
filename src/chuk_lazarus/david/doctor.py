"""Filesystem-only readiness checks for David model boot."""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .model_artifacts import VindexArtifactMetadata, inspect_vindex_artifact, is_vindex_artifact_path
from .model_validation import (
    GET_MODEL_CONFIG_RELATIVE_PATH,
    VALIDATE_MODEL_CONFIG_RELATIVE_PATH,
    ValidationReportDiscovery,
    discover_validation_report,
    workspace_auto_model_report_paths,
)


@dataclass(frozen=True)
class DoctorCheck:
    """One readiness check from the production doctor command."""

    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class DavidDoctorReport:
    """Complete filesystem/package readiness report."""

    model: str | None
    workspace_path: Path
    checks: tuple[DoctorCheck, ...]
    validation_discovery: ValidationReportDiscovery

    @property
    def ready(self) -> bool:
        return all(check.status not in {"blocked", "missing"} for check in self.checks)


def run_doctor(
    *,
    model: str | None,
    workspace_path: Path,
    validation_report: str | None = None,
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
    checks = [
        _workspace_check(workspace_root),
        _model_location_check(model),
        _hf_snapshot_check(model),
        _vindex_check(workspace_root, model),
        _validation_report_check(discovery),
        *_optional_package_checks(),
        _torch_cuda_check(),
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
    if report.validation_discovery.checked_paths:
        lines.append("- checked validation report paths:")
        lines.extend(f"  - {path}" for path in report.validation_discovery.checked_paths)
    return "\n".join(lines) + "\n"


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


def _validation_report_check(discovery: ValidationReportDiscovery) -> DoctorCheck:
    if discovery.path is not None:
        return DoctorCheck("validation report", "ready", str(discovery.path))
    return DoctorCheck("validation report", "missing", "no boot-safe validation report discovered")


def _optional_package_checks() -> tuple[DoctorCheck, ...]:
    packages = ("torch", "transformers", "safetensors", "accelerate", "huggingface_hub")
    checks = []
    for package in packages:
        status = "ready" if importlib.util.find_spec(package) is not None else "missing"
        detail = "importable" if status == "ready" else "not installed in this Python environment"
        checks.append(DoctorCheck(f"package {package}", status, detail))
    return tuple(checks)


def _torch_cuda_check() -> DoctorCheck:
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
    if is_available():
        count = getattr(cuda, "device_count", lambda: "?")()
        return DoctorCheck("torch CUDA", "ready", f"CUDA available; devices={count}")
    return DoctorCheck("torch CUDA", "review", "CPU-only torch; real Gemma boot may be too slow or impossible")


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


def _looks_like_hf_model_id(model: str) -> bool:
    return "/" in model and not any(sep in model for sep in ("\\", "./", "../")) and not Path(model).exists()


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
