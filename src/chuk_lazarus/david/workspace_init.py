"""Core workspace initialization for the David terminal-agent harness."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import (
    CONFIG_SCHEMA_NAME,
    CONFIG_SCHEMA_VERSION,
    DEFAULT_MODEL_BACKEND,
    DEFAULT_MODEL_DTYPE,
    DEFAULT_MODEL_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ATTESTATION_PATH,
    DEFAULT_VALIDATION_REPORT_PATH,
    coerce_workspace_operator_config,
)

DAVID_DIR = Path(".david")
CONFIG_PATH = DAVID_DIR / "config.json"
NEXT_STEPS_PATH = DAVID_DIR / "NEXT_STEPS.md"
LAYOUT_DIRS = (
    DAVID_DIR,
    DAVID_DIR / "memory",
    DAVID_DIR / "indexes",
    DAVID_DIR / "model_validation",
    DAVID_DIR / "sessions",
)


@dataclass(frozen=True)
class WorkspaceInitResult:
    """Result of creating or refreshing a David workspace layout."""

    workspace_root: Path
    david_dir: Path
    config_path: Path
    next_steps_path: Path
    config: dict[str, Any]
    created_paths: tuple[Path, ...]
    existing_paths: tuple[Path, ...]
    overwritten_paths: tuple[Path, ...]
    next_steps_summary: str

    @property
    def changed(self) -> bool:
        return bool(self.created_paths or self.overwritten_paths)


class WorkspaceInitError(ValueError):
    """Raised when init would escape or corrupt the requested workspace."""


def initialize_workspace(
    workspace_path: str | Path,
    *,
    force: bool = False,
    model: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    max_new_tokens: int | None = None,
    auto_jit_index: bool = False,
) -> WorkspaceInitResult:
    """Create the David workspace state layout without indexing or loading a model.

    The initializer writes only deterministic paths under ``workspace_path``. Existing
    files are preserved unless ``force`` is true, making repeated calls idempotent.
    """

    workspace_root = Path(workspace_path).expanduser().resolve()
    if workspace_root.exists() and not workspace_root.is_dir():
        raise WorkspaceInitError(f"workspace path exists but is not a directory: {workspace_root}")
    if not workspace_root.exists():
        if not workspace_root.parent.is_dir():
            raise WorkspaceInitError(
                f"workspace parent does not exist or is not a directory: {workspace_root.parent}"
            )
        workspace_root.mkdir()

    targets = _workspace_targets(workspace_root)
    created: list[Path] = []
    existing: list[Path] = []
    overwritten: list[Path] = []

    for relative_dir in LAYOUT_DIRS:
        directory = targets[relative_dir]
        if directory.exists():
            if not directory.is_dir():
                raise WorkspaceInitError(f"David layout path exists but is not a directory: {directory}")
            existing.append(directory)
            continue
        directory.mkdir()
        created.append(directory)

    config = default_workspace_config(
        model=model,
        backend=backend,
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        auto_jit_index=auto_jit_index,
    )
    config_path = targets[CONFIG_PATH]
    _write_json_file(
        config_path,
        config,
        force=force,
        created=created,
        existing=existing,
        overwritten=overwritten,
    )

    summary = format_next_steps_summary(config)
    next_steps_path = targets[NEXT_STEPS_PATH]
    _write_text_file(
        next_steps_path,
        summary,
        force=force,
        created=created,
        existing=existing,
        overwritten=overwritten,
    )

    return WorkspaceInitResult(
        workspace_root=workspace_root,
        david_dir=targets[DAVID_DIR],
        config_path=config_path,
        next_steps_path=next_steps_path,
        config=config,
        created_paths=tuple(created),
        existing_paths=tuple(existing),
        overwritten_paths=tuple(overwritten),
        next_steps_summary=summary,
    )


def init_workspace(*args: Any, **kwargs: Any) -> WorkspaceInitResult:
    """Compatibility alias for the eventual ``david init`` command."""

    return initialize_workspace(*args, **kwargs)


def default_workspace_config(
    *,
    model: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    max_new_tokens: int | None = None,
    auto_jit_index: bool = False,
) -> dict[str, Any]:
    """Return CLI-compatible defaults using relative workspace paths."""

    return coerce_workspace_operator_config(
        {
            "schema_name": CONFIG_SCHEMA_NAME,
            "schema_version": CONFIG_SCHEMA_VERSION,
            "paths": {
                "state_dir": ".david",
                "memory_dir": ".david/memory",
                "indexes_dir": ".david/indexes",
                "model_validation_dir": ".david/model_validation",
                "sessions_dir": ".david/sessions",
            },
            "model": _clean_optional_string(model),
            "validation_report": DEFAULT_VALIDATION_REPORT_PATH,
            "model_attestation": DEFAULT_MODEL_ATTESTATION_PATH,
            "model_backend": _clean_optional_string(backend) or DEFAULT_MODEL_BACKEND,
            "model_device": _clean_optional_string(device),
            "model_dtype": _clean_optional_string(dtype) or DEFAULT_MODEL_DTYPE,
            "model_max_new_tokens": _positive_int_or_default(
                max_new_tokens,
                DEFAULT_MODEL_MAX_NEW_TOKENS,
            ),
            "auto_jit_index": bool(auto_jit_index),
        }
    )


def format_next_steps_summary(config: dict[str, Any]) -> str:
    """Build the human-readable handoff written by workspace init."""

    model = config.get("model") or "set a model with --model or edit .david/config.json"
    validation_report = config.get("validation_report")
    return "\n".join(
        [
            "# David Workspace",
            "",
            "David workspace state has been initialized.",
            "",
            "Layout:",
            "- config: .david/config.json",
            "- memory: .david/memory",
            "- indexes: .david/indexes",
            "- model validation: .david/model_validation",
            "- sessions: .david/sessions",
            "",
            "Next steps:",
            f"- Model default: {model}",
            f"- Validation report path: {validation_report}",
            "- Run model scanning and validation before booting a model-backed session.",
            "- Build or refresh the workspace index only after model validation is accepted.",
            "",
        ]
    )


def _workspace_targets(workspace_root: Path) -> dict[Path, Path]:
    targets = {
        relative: _safe_workspace_child(workspace_root, relative)
        for relative in (*LAYOUT_DIRS, CONFIG_PATH, NEXT_STEPS_PATH)
    }
    return targets


def _safe_workspace_child(workspace_root: Path, relative_path: Path) -> Path:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise WorkspaceInitError(f"David init target must be workspace-relative: {relative_path}")

    target = workspace_root / relative_path
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(workspace_root):
        raise WorkspaceInitError(f"David init target escapes workspace: {target}")
    return target


def _write_json_file(
    path: Path,
    payload: dict[str, Any],
    *,
    force: bool,
    created: list[Path],
    existing: list[Path],
    overwritten: list[Path],
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_text_file(
        path,
        text,
        force=force,
        created=created,
        existing=existing,
        overwritten=overwritten,
    )


def _write_text_file(
    path: Path,
    text: str,
    *,
    force: bool,
    created: list[Path],
    existing: list[Path],
    overwritten: list[Path],
) -> None:
    if path.exists() and path.is_dir():
        raise WorkspaceInitError(f"David managed file path exists but is a directory: {path}")
    if path.exists() and not force:
        existing.append(path)
        return
    if path.exists():
        overwritten.append(path)
    else:
        created.append(path)
    path.write_text(text, encoding="utf-8")


def _clean_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _positive_int_or_default(value: int | None, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
