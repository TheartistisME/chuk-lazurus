"""Workspace JIT-index readiness helpers for the David product harness.

This module owns David's product metadata path under a workspace:

``<workspace>/.chuk_lazarus/david/``

It never writes benchmark fixtures or benchmark result artifacts. The helpers
only check or create David product manifests and memory directories that are
scoped to a model/tokenizer/adapter tuple.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DAVID_PRODUCT_DIR = ".chuk_lazarus"
DAVID_DIR = "david"
INDEX_DIR = "index"
TASK_MEMORY_DIR = "task_memory"
SESSION_DIR = "sessions"
MANIFEST_FILENAME = "manifest.json"
WORKSPACE_INDEX_SCHEMA_NAME = "chuk_lazarus.david.workspace_index_manifest"
WORKSPACE_INDEX_SCHEMA_VERSION = 1

_DEFAULT_MEMORY_FAMILIES = ("task", "code")
_BENCHMARK_ARTIFACT_PARTS = {
    "benchmark_artifacts",
    "benchmark_results",
    "benchmarks_artifacts",
    "benchmarks_results",
}


@dataclass(frozen=True)
class DavidModelScope:
    """Model/tokenizer/adapter compatibility scope for product memory."""

    model_identity: str | None = None
    tokenizer_identity: str | None = None
    adapter_config_id: str | None = None
    adapter_family: str | None = None

    @property
    def complete(self) -> bool:
        return bool(
            self.model_identity
            and self.tokenizer_identity
            and self.adapter_config_id
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DavidMemoryPaths:
    """Resolved David product paths for user, task, index, and resume data."""

    workspace_path: str
    david_root: str
    index_root: str
    index_manifest_path: str
    task_memory_path: str
    user_memory_path: str
    sessions_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DavidWorkspaceIndexManifest:
    """Manifest proving a workspace index belongs to one model scope."""

    index_id: str
    workspace_id: str
    workspace_path: str
    model_identity: str | None
    tokenizer_identity: str | None
    adapter_config_id: str | None
    adapter_family: str | None = None
    schema_name: str = WORKSPACE_INDEX_SCHEMA_NAME
    schema_version: int = WORKSPACE_INDEX_SCHEMA_VERSION
    memory_families: tuple[str, ...] = _DEFAULT_MEMORY_FAMILIES
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    artifacts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["memory_families"] = list(self.memory_families)
        data.setdefault("provenance", {})
        data["provenance"] = {
            "source": "chuk_lazarus.david.indexing",
            "product_artifact": True,
            "benchmark_artifact": False,
            **dict(self.provenance),
        }
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DavidWorkspaceIndexManifest:
        scope = extract_model_scope(values)
        return cls(
            index_id=str(values.get("index_id") or ""),
            workspace_id=str(values.get("workspace_id") or ""),
            workspace_path=str(values.get("workspace_path") or ""),
            model_identity=scope.model_identity,
            tokenizer_identity=scope.tokenizer_identity,
            adapter_config_id=scope.adapter_config_id,
            adapter_family=scope.adapter_family,
            schema_name=str(values.get("schema_name") or WORKSPACE_INDEX_SCHEMA_NAME),
            schema_version=int(values.get("schema_version") or WORKSPACE_INDEX_SCHEMA_VERSION),
            memory_families=tuple(
                str(item) for item in (values.get("memory_families") or _DEFAULT_MEMORY_FAMILIES)
            ),
            created_at=str(values.get("created_at") or _utc_now()),
            updated_at=str(values.get("updated_at") or _utc_now()),
            artifacts=dict(values.get("artifacts") or {}),
            provenance=dict(values.get("provenance") or {}),
        )


@dataclass(frozen=True)
class DavidIndexReadiness:
    """Readiness check result and planned JIT actions for one workspace."""

    workspace_path: str
    david_root: str
    manifest_path: str
    state: str
    ready: bool
    jit_required: bool
    planned_actions: tuple[str, ...] = ()
    missing_reasons: tuple[str, ...] = ()
    mismatch_reasons: tuple[str, ...] = ()
    manifest: DavidWorkspaceIndexManifest | None = None
    memory_paths: DavidMemoryPaths | None = None
    checked_at: str = field(default_factory=lambda: _utc_now())
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def jit_actions(self) -> tuple[str, ...]:
        return self.planned_actions

    @property
    def compatible(self) -> bool:
        return self.ready and not self.mismatch_reasons

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["jit_actions"] = list(self.jit_actions)
        data["planned_actions"] = list(self.planned_actions)
        data["missing_reasons"] = list(self.missing_reasons)
        data["mismatch_reasons"] = list(self.mismatch_reasons)
        return data


def resolve_memory_paths(
    workspace_path: str | Path,
    *,
    user_memory_root: str | Path | None = None,
) -> DavidMemoryPaths:
    """Return David product paths without creating them."""

    workspace = Path(workspace_path).expanduser().resolve()
    david_root = workspace / DAVID_PRODUCT_DIR / DAVID_DIR
    index_root = david_root / INDEX_DIR
    user_root = _resolve_user_memory_path(user_memory_root)
    return DavidMemoryPaths(
        workspace_path=str(workspace),
        david_root=str(david_root),
        index_root=str(index_root),
        index_manifest_path=str(index_root / MANIFEST_FILENAME),
        task_memory_path=str(david_root / TASK_MEMORY_DIR),
        user_memory_path=str(user_root),
        sessions_path=str(david_root / SESSION_DIR),
    )


def check_workspace_index(
    workspace_path: str | Path,
    model_scope: DavidModelScope | Mapping[str, Any] | Any | None = None,
    *,
    model_report: Any | None = None,
    model_identity: str | None = None,
    tokenizer_identity: str | None = None,
    adapter_config_id: str | None = None,
    adapter_family: str | None = None,
    user_memory_root: str | Path | None = None,
) -> DavidIndexReadiness:
    """Check whether the workspace index manifest matches the active model."""

    paths = resolve_memory_paths(workspace_path, user_memory_root=user_memory_root)
    expected = extract_model_scope(
        model_scope or model_report,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        adapter_config_id=adapter_config_id,
        adapter_family=adapter_family,
    )
    david_root = Path(paths.david_root)
    index_root = Path(paths.index_root)
    manifest_path = Path(paths.index_manifest_path)

    actions: list[str] = []
    missing: list[str] = []
    mismatches: list[str] = []
    manifest: DavidWorkspaceIndexManifest | None = None

    if not david_root.exists():
        actions.append("create_david_workspace_root")
        missing.append("david_workspace_root_missing")
    if not Path(paths.task_memory_path).exists():
        actions.append("create_workspace_task_memory_root")
        missing.append("workspace_task_memory_root_missing")
    if not index_root.exists():
        actions.append("create_workspace_index_root")
        missing.append("workspace_index_root_missing")
    if not manifest_path.exists():
        actions.extend(("jit_index_workspace", "write_workspace_index_manifest"))
        missing.append("workspace_index_manifest_missing")
        return _readiness(
            paths,
            state="missing",
            actions=actions,
            missing=missing,
            mismatches=mismatches,
            manifest=manifest,
        )

    try:
        manifest = load_workspace_index_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        actions.extend(("refresh_workspace_index_for_model", "write_workspace_index_manifest"))
        mismatches.append(f"workspace_index_manifest_invalid:{exc}")
        return _readiness(
            paths,
            state="incompatible",
            actions=actions,
            missing=missing,
            mismatches=mismatches,
            manifest=None,
        )

    if manifest.schema_name != WORKSPACE_INDEX_SCHEMA_NAME:
        mismatches.append("schema_name_mismatch")
    if manifest.schema_version != WORKSPACE_INDEX_SCHEMA_VERSION:
        mismatches.append("schema_version_mismatch")
    mismatches.extend(_scope_mismatches(manifest, expected))

    if mismatches:
        actions.extend(("refresh_workspace_index_for_model", "write_workspace_index_manifest"))
        return _readiness(
            paths,
            state="incompatible",
            actions=actions,
            missing=missing,
            mismatches=mismatches,
            manifest=manifest,
        )

    state = "ready" if not actions else "missing"
    return _readiness(
        paths,
        state=state,
        actions=actions,
        missing=missing,
        mismatches=mismatches,
        manifest=manifest,
    )


def build_workspace_index_manifest(
    workspace_path: str | Path,
    model_scope: DavidModelScope | Mapping[str, Any] | Any | None = None,
    *,
    model_report: Any | None = None,
    model_identity: str | None = None,
    tokenizer_identity: str | None = None,
    adapter_config_id: str | None = None,
    adapter_family: str | None = None,
    user_memory_root: str | Path | None = None,
    memory_families: tuple[str, ...] | list[str] = _DEFAULT_MEMORY_FAMILIES,
    artifacts: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> DavidWorkspaceIndexManifest:
    """Build a manifest payload without writing it."""

    paths = resolve_memory_paths(workspace_path, user_memory_root=user_memory_root)
    scope = extract_model_scope(
        model_scope or model_report,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        adapter_config_id=adapter_config_id,
        adapter_family=adapter_family,
    )
    workspace_id = workspace_id_for(paths.workspace_path)
    index_id = index_id_for(workspace_id, scope)
    now = _utc_now()
    return DavidWorkspaceIndexManifest(
        index_id=index_id,
        workspace_id=workspace_id,
        workspace_path=paths.workspace_path,
        model_identity=scope.model_identity,
        tokenizer_identity=scope.tokenizer_identity,
        adapter_config_id=scope.adapter_config_id,
        adapter_family=scope.adapter_family,
        memory_families=tuple(str(item) for item in memory_families),
        created_at=now,
        updated_at=now,
        artifacts=dict(artifacts or {}),
        provenance={
            "manifest_path": paths.index_manifest_path,
            "task_memory_path": paths.task_memory_path,
            "user_memory_path": paths.user_memory_path,
            **dict(provenance or {}),
        },
    )


def write_workspace_index_manifest(
    workspace_path: str | Path,
    manifest: DavidWorkspaceIndexManifest | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Path:
    """Atomically write the workspace index manifest and return its path."""

    paths = resolve_memory_paths(
        workspace_path,
        user_memory_root=kwargs.get("user_memory_root"),
    )
    final_path = Path(paths.index_manifest_path)
    ensure_product_artifact_path(paths.workspace_path, final_path)

    if manifest is None:
        payload = build_workspace_index_manifest(workspace_path, **kwargs).to_dict()
    elif isinstance(manifest, DavidWorkspaceIndexManifest):
        payload = manifest.to_dict()
    elif isinstance(manifest, Mapping):
        if _looks_like_manifest_mapping(manifest):
            payload = DavidWorkspaceIndexManifest.from_dict(manifest).to_dict()
        else:
            payload = build_workspace_index_manifest(workspace_path, manifest, **kwargs).to_dict()
    else:
        raise TypeError("manifest must be a DavidWorkspaceIndexManifest, mapping, or None")

    Path(paths.david_root).mkdir(parents=True, exist_ok=True)
    Path(paths.task_memory_path).mkdir(parents=True, exist_ok=True)
    Path(paths.index_root).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(final_path, payload)
    return final_path


def ensure_workspace_index_manifest(
    workspace_path: str | Path,
    model_scope: DavidModelScope | Mapping[str, Any] | Any | None = None,
    **kwargs: Any,
) -> DavidIndexReadiness:
    """Create a compatible manifest when missing or mismatched, then re-check."""

    readiness = check_workspace_index(workspace_path, model_scope, **kwargs)
    if readiness.ready:
        return readiness
    write_workspace_index_manifest(workspace_path, model_scope=model_scope, **kwargs)
    return check_workspace_index(workspace_path, model_scope, **kwargs)


def load_workspace_index_manifest(path_or_workspace: str | Path) -> DavidWorkspaceIndexManifest:
    """Load a manifest from a manifest path, index root, or workspace root."""

    path = Path(path_or_workspace).expanduser()
    if path.is_dir():
        if path.name == INDEX_DIR:
            path = path / MANIFEST_FILENAME
        else:
            path = path / DAVID_PRODUCT_DIR / DAVID_DIR / INDEX_DIR / MANIFEST_FILENAME
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, Mapping):
        raise ValueError("workspace index manifest must be a JSON object")
    return DavidWorkspaceIndexManifest.from_dict(loaded)


def extract_model_scope(
    source: DavidModelScope | Mapping[str, Any] | Any | None = None,
    *,
    model_identity: str | None = None,
    tokenizer_identity: str | None = None,
    adapter_config_id: str | None = None,
    adapter_family: str | None = None,
) -> DavidModelScope:
    """Extract a scope from mappings, validation reports, or explicit fields."""

    if isinstance(source, DavidModelScope):
        base = source
    else:
        base = DavidModelScope(
            model_identity=_optional_str(
                _value_from_source(source, "model_identity", "model_id", "identity")
            ),
            tokenizer_identity=_optional_str(
                _value_from_source(source, "tokenizer_identity", "tokenizer_id")
            ),
            adapter_config_id=_optional_str(
                _value_from_source(source, "adapter_config_id", "adapter_config")
            ),
            adapter_family=_optional_str(
                _value_from_source(source, "adapter_family", "family")
            ),
        )

    return DavidModelScope(
        model_identity=_optional_str(model_identity) or base.model_identity,
        tokenizer_identity=_optional_str(tokenizer_identity) or base.tokenizer_identity,
        adapter_config_id=_optional_str(adapter_config_id) or base.adapter_config_id,
        adapter_family=_optional_str(adapter_family) or base.adapter_family,
    )


def workspace_id_for(workspace_path: str | Path) -> str:
    workspace = Path(workspace_path).expanduser().resolve()
    digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()[:16]
    name = workspace.name or "workspace"
    return f"{_safe_token(name)}-{digest}"


def index_id_for(workspace_id: str, scope: DavidModelScope) -> str:
    payload = "|".join(
        (
            workspace_id,
            scope.model_identity or "",
            scope.tokenizer_identity or "",
            scope.adapter_config_id or "",
            scope.adapter_family or "",
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"david-index-{digest}"


def ensure_product_artifact_path(workspace_path: str | Path, path: str | Path) -> Path:
    """Return ``path`` if it is under the David product metadata root."""

    workspace = Path(workspace_path).expanduser().resolve()
    product_root = (workspace / DAVID_PRODUCT_DIR / DAVID_DIR).resolve()
    candidate = Path(path).expanduser().resolve()
    if not _is_relative_to(candidate, product_root):
        raise ValueError(
            "David product metadata writes must stay under "
            f"{product_root}"
        )
    relative_parts = {part.lower() for part in candidate.relative_to(product_root).parts}
    if relative_parts & _BENCHMARK_ARTIFACT_PARTS:
        raise ValueError("David product helpers refuse benchmark artifact paths")
    return candidate


def _readiness(
    paths: DavidMemoryPaths,
    *,
    state: str,
    actions: list[str],
    missing: list[str],
    mismatches: list[str],
    manifest: DavidWorkspaceIndexManifest | None,
) -> DavidIndexReadiness:
    planned = tuple(_dedupe(actions))
    return DavidIndexReadiness(
        workspace_path=paths.workspace_path,
        david_root=paths.david_root,
        manifest_path=paths.index_manifest_path,
        state=state,
        ready=state == "ready" and not planned and not mismatches,
        jit_required=bool(planned or missing or mismatches),
        planned_actions=planned,
        missing_reasons=tuple(_dedupe(missing)),
        mismatch_reasons=tuple(_dedupe(mismatches)),
        manifest=manifest,
        memory_paths=paths,
        provenance={
            "source": "chuk_lazarus.david.indexing",
            "product_artifact": True,
            "benchmark_artifact": False,
        },
    )


def _scope_mismatches(
    manifest: DavidWorkspaceIndexManifest,
    expected: DavidModelScope,
) -> list[str]:
    mismatches: list[str] = []
    for field_name in ("model_identity", "tokenizer_identity", "adapter_config_id"):
        expected_value = getattr(expected, field_name)
        if expected_value is None:
            continue
        actual_value = getattr(manifest, field_name)
        if actual_value is None:
            mismatches.append(f"{field_name}_missing")
        elif str(actual_value) != str(expected_value):
            mismatches.append(f"{field_name}_mismatch")
    if expected.adapter_family and manifest.adapter_family:
        if str(expected.adapter_family) != str(manifest.adapter_family):
            mismatches.append("adapter_family_mismatch")
    return mismatches


def _value_from_source(source: Any | None, *names: str) -> Any | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if source.get(name) is not None:
                return source[name]
        for nested_name in (
            "scope",
            "model_scope",
            "selected_config",
            "selected_adapter_config",
            "source_report_summary",
            "model_identity_gate",
            "adapter",
        ):
            nested = source.get(nested_name)
            if isinstance(nested, Mapping):
                value = _value_from_source(nested, *names)
                if value is not None:
                    return value
        return None

    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value

    for nested_name in (
        "scope",
        "model_scope",
        "selected_config",
        "selected_adapter_config",
        "source_report_summary",
        "adapter",
    ):
        nested = getattr(source, nested_name, None)
        value = _value_from_source(nested, *names)
        if value is not None:
            return value
    return None


def _looks_like_manifest_mapping(values: Mapping[str, Any]) -> bool:
    return any(key in values for key in ("schema_name", "schema_version", "index_id", "workspace_id"))


def _atomic_write_json(final_path: Path, payload: Mapping[str, Any]) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _resolve_user_memory_path(user_memory_root: str | Path | None) -> Path:
    configured = user_memory_root
    if configured is None:
        configured = os.environ.get("CHUK_LAZARUS_DAVID_USER_MEMORY_ROOT")
    if configured is None:
        configured = os.environ.get("CHUK_LAZARUS_USER_MEMORY_ROOT")
    if configured is None:
        configured = Path.home() / DAVID_PRODUCT_DIR / DAVID_DIR / "user_memory"
    return Path(configured).expanduser().resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _optional_str(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_token(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return safe.strip("._-") or "workspace"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


WorkspaceIndexManifest = DavidWorkspaceIndexManifest
WorkspaceIndexReadiness = DavidIndexReadiness
WorkspaceMemoryPaths = DavidMemoryPaths
ModelScope = DavidModelScope

ensure_workspace_index = ensure_workspace_index_manifest
create_workspace_index_manifest = write_workspace_index_manifest
read_workspace_index_manifest = load_workspace_index_manifest


__all__ = [
    "DAVID_DIR",
    "DAVID_PRODUCT_DIR",
    "INDEX_DIR",
    "MANIFEST_FILENAME",
    "TASK_MEMORY_DIR",
    "WORKSPACE_INDEX_SCHEMA_NAME",
    "WORKSPACE_INDEX_SCHEMA_VERSION",
    "DavidIndexReadiness",
    "DavidMemoryPaths",
    "DavidModelScope",
    "DavidWorkspaceIndexManifest",
    "ModelScope",
    "WorkspaceIndexManifest",
    "WorkspaceIndexReadiness",
    "WorkspaceMemoryPaths",
    "build_workspace_index_manifest",
    "check_workspace_index",
    "create_workspace_index_manifest",
    "ensure_product_artifact_path",
    "ensure_workspace_index",
    "ensure_workspace_index_manifest",
    "extract_model_scope",
    "index_id_for",
    "load_workspace_index_manifest",
    "read_workspace_index_manifest",
    "resolve_memory_paths",
    "workspace_id_for",
    "write_workspace_index_manifest",
]
