"""Resumable David session metadata snapshots.

Snapshots are metadata only. They are written atomically under the workspace
David product root and carry separate user-memory and task-memory paths so
person-in-time memory is not collapsed into workspace/task memory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from chuk_lazarus.david.indexing import (
    DavidModelScope,
    ensure_product_artifact_path,
    extract_model_scope,
    resolve_memory_paths,
    workspace_id_for,
)

SESSION_SNAPSHOT_SCHEMA_NAME = "chuk_lazarus.david.session_snapshot"
SESSION_SNAPSHOT_SCHEMA_VERSION = 1
LATEST_SESSION_FILENAME = "latest.json"


@dataclass(frozen=True)
class DavidSessionSnapshot:
    """Serializable resume metadata for one David terminal-agent session."""

    session_id: str
    workspace_id: str
    workspace_path: str
    user_memory_path: str
    task_memory_path: str
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    adapter_config_id: str | None = None
    adapter_family: str | None = None
    user_id: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    selected_methodology: str | None = None
    turn_index: int = 0
    status: str = "active"
    schema_name: str = SESSION_SNAPSHOT_SCHEMA_NAME
    schema_version: int = SESSION_SNAPSHOT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def model_scope(self) -> DavidModelScope:
        return DavidModelScope(
            model_identity=self.model_identity,
            tokenizer_identity=self.tokenizer_identity,
            adapter_config_id=self.adapter_config_id,
            adapter_family=self.adapter_family,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = {
            "source": "chuk_lazarus.david.resume",
            "product_artifact": True,
            "benchmark_artifact": False,
            **dict(self.provenance),
        }
        return _jsonable_dict(data)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DavidSessionSnapshot:
        scope = extract_model_scope(values)
        return cls(
            session_id=str(values.get("session_id") or ""),
            workspace_id=str(values.get("workspace_id") or ""),
            workspace_path=str(values.get("workspace_path") or ""),
            user_memory_path=str(values.get("user_memory_path") or ""),
            task_memory_path=str(values.get("task_memory_path") or ""),
            model_identity=scope.model_identity,
            tokenizer_identity=scope.tokenizer_identity,
            adapter_config_id=scope.adapter_config_id,
            adapter_family=scope.adapter_family,
            user_id=_optional_str(values.get("user_id")),
            task_id=_optional_str(values.get("task_id")),
            task_type=_optional_str(values.get("task_type")),
            selected_methodology=_optional_str(values.get("selected_methodology")),
            turn_index=int(values.get("turn_index") or 0),
            status=str(values.get("status") or "active"),
            schema_name=str(values.get("schema_name") or SESSION_SNAPSHOT_SCHEMA_NAME),
            schema_version=int(values.get("schema_version") or SESSION_SNAPSHOT_SCHEMA_VERSION),
            created_at=str(values.get("created_at") or _utc_now()),
            updated_at=str(values.get("updated_at") or _utc_now()),
            metadata=dict(values.get("metadata") or {}),
            provenance=dict(values.get("provenance") or {}),
        )


def snapshot_from_session(
    session: Any | None = None,
    *,
    workspace_path: str | Path | None = None,
    user_memory_root: str | Path | None = None,
    session_id: str | None = None,
    model_identity: str | None = None,
    tokenizer_identity: str | None = None,
    adapter_config_id: str | None = None,
    adapter_family: str | None = None,
    user_id: str | None = None,
    task_id: str | None = None,
    task_type: str | None = None,
    selected_methodology: str | None = None,
    turn_index: int | None = None,
    status: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DavidSessionSnapshot:
    """Create a snapshot from a harness session, mapping, or explicit fields."""

    source = session or {}
    workspace_value = (
        workspace_path
        or _value_from_source(source, "workspace_path", "workspace")
        or "."
    )
    source_user_memory_path = _value_from_source(
        source,
        "user_memory_path",
        "user_memory_root",
    )
    paths = resolve_memory_paths(
        workspace_value,
        user_memory_root=user_memory_root or source_user_memory_path,
    )
    scope = extract_model_scope(
        source,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        adapter_config_id=adapter_config_id,
        adapter_family=adapter_family,
    )
    now = _utc_now()
    source_metadata = _value_from_source(source, "metadata")
    snapshot_metadata = dict(source_metadata) if isinstance(source_metadata, Mapping) else {}
    snapshot_metadata.update(dict(metadata or {}))
    return DavidSessionSnapshot(
        session_id=(
            _optional_str(session_id)
            or _optional_str(_value_from_source(source, "session_id", "id"))
            or f"session-{uuid4().hex}"
        ),
        workspace_id=_optional_str(_value_from_source(source, "workspace_id"))
        or workspace_id_for(paths.workspace_path),
        workspace_path=paths.workspace_path,
        user_memory_path=paths.user_memory_path,
        task_memory_path=paths.task_memory_path,
        model_identity=scope.model_identity,
        tokenizer_identity=scope.tokenizer_identity,
        adapter_config_id=scope.adapter_config_id,
        adapter_family=scope.adapter_family,
        user_id=_optional_str(user_id) or _optional_str(_value_from_source(source, "user_id")),
        task_id=_optional_str(task_id) or _optional_str(_value_from_source(source, "task_id")),
        task_type=_optional_str(task_type) or _optional_str(_value_from_source(source, "task_type")),
        selected_methodology=_optional_str(selected_methodology)
        or _optional_str(_value_from_source(source, "selected_methodology", "methodology")),
        turn_index=int(turn_index if turn_index is not None else (_value_from_source(source, "turn_index") or 0)),
        status=_optional_str(status) or _optional_str(_value_from_source(source, "status")) or "active",
        created_at=_optional_str(_value_from_source(source, "created_at")) or now,
        updated_at=now,
        metadata=snapshot_metadata,
        provenance={
            "workspace_product_root": str(Path(paths.sessions_path).parent),
            "snapshot_family": "resume_metadata",
        },
    )


def write_session_snapshot(
    workspace_or_snapshot: str | Path | DavidSessionSnapshot | Mapping[str, Any] | Any,
    snapshot: DavidSessionSnapshot | Mapping[str, Any] | Any | None = None,
    **overrides: Any,
) -> Path:
    """Atomically write a session snapshot and update the latest pointer.

    Supported call styles:
    ``write_session_snapshot(workspace_path, snapshot)``
    ``write_session_snapshot(snapshot)``
    ``write_session_snapshot(workspace_path, session_id="...")``
    """

    workspace_path, source = _coerce_write_args(workspace_or_snapshot, snapshot, overrides)
    snapshot_obj = coerce_session_snapshot(source, workspace_path=workspace_path, **overrides)
    paths = resolve_memory_paths(
        snapshot_obj.workspace_path,
        user_memory_root=snapshot_obj.user_memory_path,
    )
    sessions_path = Path(paths.sessions_path)
    sessions_path.mkdir(parents=True, exist_ok=True)
    Path(paths.task_memory_path).mkdir(parents=True, exist_ok=True)

    final_path = session_snapshot_path(
        snapshot_obj.workspace_path,
        snapshot_obj.session_id,
        user_memory_root=snapshot_obj.user_memory_path,
    )
    ensure_product_artifact_path(snapshot_obj.workspace_path, final_path)
    _atomic_write_json(final_path, snapshot_obj.to_dict())

    latest_path = sessions_path / LATEST_SESSION_FILENAME
    ensure_product_artifact_path(snapshot_obj.workspace_path, latest_path)
    _atomic_write_json(
        latest_path,
        {
            "schema_name": "chuk_lazarus.david.latest_session_pointer",
            "schema_version": 1,
            "session_id": snapshot_obj.session_id,
            "snapshot_file": final_path.name,
            "updated_at": snapshot_obj.updated_at,
            "benchmark_artifact": False,
        },
    )
    return final_path


def coerce_session_snapshot(
    source: DavidSessionSnapshot | Mapping[str, Any] | Any | None = None,
    **overrides: Any,
) -> DavidSessionSnapshot:
    """Return a ``DavidSessionSnapshot`` from supported source shapes."""

    if isinstance(source, DavidSessionSnapshot) and not overrides:
        return source
    if isinstance(source, DavidSessionSnapshot):
        data = source.to_dict()
        data.update(overrides)
        return snapshot_from_session(data)
    if isinstance(source, Mapping):
        data = dict(source)
        data.update({key: value for key, value in overrides.items() if value is not None})
        return snapshot_from_session(data)
    return snapshot_from_session(source, **overrides)


def session_snapshot_path(
    workspace_path: str | Path,
    session_id: str,
    *,
    user_memory_root: str | Path | None = None,
) -> Path:
    paths = resolve_memory_paths(workspace_path, user_memory_root=user_memory_root)
    return Path(paths.sessions_path) / f"{_safe_file_stem(session_id)}.json"


def load_session_snapshot(
    workspace_path: str | Path,
    session_id: str,
    *,
    user_memory_root: str | Path | None = None,
) -> DavidSessionSnapshot | None:
    path = session_snapshot_path(
        workspace_path,
        session_id,
        user_memory_root=user_memory_root,
    )
    if not path.exists():
        return None
    return _load_snapshot_file(path)


def load_latest_session_snapshot(
    workspace_path: str | Path,
    *,
    user_memory_root: str | Path | None = None,
) -> DavidSessionSnapshot | None:
    """Load the latest snapshot for a workspace, if one exists."""

    paths = resolve_memory_paths(workspace_path, user_memory_root=user_memory_root)
    sessions_path = Path(paths.sessions_path)
    latest_path = sessions_path / LATEST_SESSION_FILENAME
    if latest_path.exists():
        try:
            pointer = _load_json(latest_path)
            snapshot_file = pointer.get("snapshot_file")
            if isinstance(snapshot_file, str) and snapshot_file:
                candidate = sessions_path / snapshot_file
                ensure_product_artifact_path(paths.workspace_path, candidate)
                if candidate.exists():
                    return _load_snapshot_file(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    snapshots = []
    for candidate in sessions_path.glob("*.json"):
        if candidate.name == LATEST_SESSION_FILENAME or candidate.name.endswith(".tmp"):
            continue
        try:
            snapshots.append(_load_snapshot_file(candidate))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not snapshots:
        return None
    return max(snapshots, key=lambda item: (item.updated_at, item.session_id))


def _coerce_write_args(
    workspace_or_snapshot: str | Path | DavidSessionSnapshot | Mapping[str, Any] | Any,
    snapshot: DavidSessionSnapshot | Mapping[str, Any] | Any | None,
    overrides: Mapping[str, Any],
) -> tuple[str | Path | None, Any | None]:
    if _looks_like_path(workspace_or_snapshot):
        return workspace_or_snapshot, snapshot
    workspace_path = overrides.get("workspace_path")
    if workspace_path is None:
        workspace_path = _value_from_source(workspace_or_snapshot, "workspace_path", "workspace")
    return workspace_path, workspace_or_snapshot


def _load_snapshot_file(path: Path) -> DavidSessionSnapshot:
    loaded = _load_json(path)
    if not isinstance(loaded, Mapping):
        raise ValueError("session snapshot must be a JSON object")
    snapshot = DavidSessionSnapshot.from_dict(loaded)
    if snapshot.schema_name != SESSION_SNAPSHOT_SCHEMA_NAME:
        raise ValueError("session snapshot schema mismatch")
    if snapshot.schema_version != SESSION_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("session snapshot schema version mismatch")
    return snapshot


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(final_path: Path, payload: Mapping[str, Any]) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable_dict(dict(payload)), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _value_from_source(source: Any | None, *names: str) -> Any | None:
    if source is None:
        return None
    if isinstance(source, Mapping):
        for name in names:
            if source.get(name) is not None:
                return source[name]
        for nested_name in ("metadata", "selected_config", "selected_adapter_config"):
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
    for nested_name in ("metadata", "selected_config", "selected_adapter_config"):
        nested = getattr(source, nested_name, None)
        value = _value_from_source(nested, *names)
        if value is not None:
            return value
    return None


def _looks_like_path(value: Any) -> bool:
    return isinstance(value, (str, Path))


def _safe_file_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("._-") or f"session-{uuid4().hex}"


def _jsonable_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(dict(values), default=str)
    loaded = json.loads(encoded)
    return loaded if isinstance(loaded, dict) else {}


def _optional_str(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


SessionSnapshot = DavidSessionSnapshot

save_session_snapshot = write_session_snapshot
load_latest_session = load_latest_session_snapshot
load_session = load_session_snapshot


__all__ = [
    "LATEST_SESSION_FILENAME",
    "SESSION_SNAPSHOT_SCHEMA_NAME",
    "SESSION_SNAPSHOT_SCHEMA_VERSION",
    "DavidSessionSnapshot",
    "SessionSnapshot",
    "coerce_session_snapshot",
    "load_latest_session",
    "load_latest_session_snapshot",
    "load_session",
    "load_session_snapshot",
    "save_session_snapshot",
    "session_snapshot_path",
    "snapshot_from_session",
    "write_session_snapshot",
]
