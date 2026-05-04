"""Live incremental source indexing for David workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .patch_routing import is_protected_path, normalize_path
from .source_index import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_FILES,
    DEFAULT_PRUNED_DIRS,
    DEFAULT_SUFFIXES,
    SourceFileRecord,
    SourceIndexManifest,
    index_source_file,
    save_source_index,
)


SCHEMA = "david.live_index.v1"
REFRESH_HANDLE_SCHEMA = "david.live_index_refresh.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class LiveFileState:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    record: SourceFileRecord

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "record": self.record.to_json(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LiveFileState":
        record = SourceFileRecord.from_json(dict(data["record"]))
        return cls(
            path=str(data["path"]),
            size_bytes=int(data["size_bytes"]),
            mtime_ns=int(data["mtime_ns"]),
            sha256=str(data["sha256"]),
            record=record,
        )


@dataclass(frozen=True)
class LiveIndexState:
    workspace_root: str
    adapter_scope: dict[str, Any]
    files: dict[str, LiveFileState]
    pruned_dirs: list[str]
    max_files: int = DEFAULT_MAX_FILES
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    truncated: bool = False
    schema: str = SCHEMA
    indexed_at: str = field(default_factory=_utc_now)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "indexed_at": self.indexed_at,
            "workspace_root": self.workspace_root,
            "adapter_scope": self.adapter_scope,
            "max_files": self.max_files,
            "max_file_bytes": self.max_file_bytes,
            "pruned_dirs": self.pruned_dirs,
            "truncated": self.truncated,
            "file_count": len(self.files),
            "files": [state.to_json() for _, state in sorted(self.files.items())],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "LiveIndexState":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported live index schema: {data.get('schema')!r}")
        files = {
            state.path: state
            for state in (LiveFileState.from_json(item) for item in data.get("files", []))
        }
        return cls(
            workspace_root=str(data["workspace_root"]),
            adapter_scope=dict(data.get("adapter_scope") or {}),
            files=files,
            pruned_dirs=[str(item) for item in data.get("pruned_dirs", [])],
            max_files=int(data.get("max_files") or DEFAULT_MAX_FILES),
            max_file_bytes=int(data.get("max_file_bytes") or DEFAULT_MAX_FILE_BYTES),
            truncated=bool(data.get("truncated")),
            indexed_at=str(data.get("indexed_at") or ""),
        )

    def to_source_manifest(self) -> SourceIndexManifest:
        return SourceIndexManifest(
            workspace_root=self.workspace_root,
            adapter_scope=dict(self.adapter_scope),
            files=[state.record for _, state in sorted(self.files.items())],
            pruned_dirs=list(self.pruned_dirs),
            max_files=self.max_files,
            max_file_bytes=self.max_file_bytes,
            truncated=self.truncated,
            indexed_at=self.indexed_at,
        )


@dataclass(frozen=True)
class LiveIndexRefresh:
    workspace_root: str
    state_path: str
    source_index_path: str
    adapter_scope: dict[str, Any]
    changed_paths: list[str]
    indexed_paths: list[str]
    deleted_paths: list[str]
    unchanged_paths: list[str]
    skipped_paths: list[str]
    file_count: int
    truncated: bool
    manifest_sha256: str
    schema: str = REFRESH_HANDLE_SCHEMA
    refreshed_at: str = field(default_factory=_utc_now)

    @property
    def handle(self) -> dict[str, Any]:
        return self.to_json()

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "refreshed_at": self.refreshed_at,
            "workspace_root": self.workspace_root,
            "state_path": self.state_path,
            "source_index_path": self.source_index_path,
            "adapter_scope": self.adapter_scope,
            "changed_paths": self.changed_paths,
            "indexed_paths": self.indexed_paths,
            "deleted_paths": self.deleted_paths,
            "unchanged_paths": self.unchanged_paths,
            "skipped_paths": self.skipped_paths,
            "changed_count": len(self.changed_paths),
            "indexed_count": len(self.indexed_paths),
            "deleted_count": len(self.deleted_paths),
            "unchanged_count": len(self.unchanged_paths),
            "skipped_count": len(self.skipped_paths),
            "file_count": self.file_count,
            "truncated": self.truncated,
            "manifest_sha256": self.manifest_sha256,
        }

    def to_session_refresh_handle(self) -> dict[str, Any]:
        return {
            "kind": "david_live_source_index",
            "schema": self.schema,
            "workspace_root": self.workspace_root,
            "source_index_path": self.source_index_path,
            "changed_paths": self.changed_paths,
            "deleted_paths": self.deleted_paths,
            "changed_count": len(self.changed_paths),
            "indexed_count": len(self.indexed_paths),
            "deleted_count": len(self.deleted_paths),
            "file_count": self.file_count,
            "manifest_sha256": self.manifest_sha256,
            "adapter_scope": self.adapter_scope,
        }


class LiveIndexer:
    """Incrementally refresh a bounded source index for one workspace."""

    def __init__(
        self,
        workspace_root: Path,
        adapter_scope: dict[str, Any],
        *,
        state_path: Path | None = None,
        source_index_path: Path | None = None,
        max_files: int = DEFAULT_MAX_FILES,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        pruned_dirs: Iterable[str] = DEFAULT_PRUNED_DIRS,
        suffixes: Iterable[str] = DEFAULT_SUFFIXES,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.adapter_scope = dict(adapter_scope)
        default_dir = self.workspace_root / ".david" / "indexes"
        self.state_path = Path(state_path) if state_path is not None else default_dir / "live-source-state.json"
        self.source_index_path = (
            Path(source_index_path) if source_index_path is not None else default_dir / "live-source.json"
        )
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.pruned_dirs = set(pruned_dirs)
        self.suffixes = {suffix.lower() for suffix in suffixes}

    def refresh(self) -> LiveIndexRefresh:
        previous = self._load_state()
        if previous is not None and previous.adapter_scope != self.adapter_scope:
            previous = None

        previous_files = previous.files if previous is not None else {}
        next_files: dict[str, LiveFileState] = {}
        changed_paths: list[str] = []
        indexed_paths: list[str] = []
        unchanged_paths: list[str] = []
        skipped_paths: list[str] = []
        truncated = False

        for relative_path, absolute_path, stat in self._iter_candidate_files():
            if len(next_files) >= self.max_files:
                truncated = True
                break
            previous_state = previous_files.get(relative_path)
            if (
                previous_state is not None
                and previous_state.size_bytes == stat.st_size
                and previous_state.mtime_ns == stat.st_mtime_ns
            ):
                next_files[relative_path] = previous_state
                unchanged_paths.append(relative_path)
                continue

            record = index_source_file(self.workspace_root, absolute_path)
            if record is None:
                skipped_paths.append(relative_path)
                continue

            indexed_paths.append(relative_path)
            state = LiveFileState(
                path=relative_path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                sha256=record.sha256,
                record=record,
            )
            next_files[relative_path] = state

            if previous_state is None or previous_state.sha256 != record.sha256:
                changed_paths.append(relative_path)
            else:
                unchanged_paths.append(relative_path)

        deleted_paths = sorted(set(previous_files) - set(next_files))
        state = LiveIndexState(
            workspace_root=str(self.workspace_root),
            adapter_scope=dict(self.adapter_scope),
            files=next_files,
            pruned_dirs=sorted(self.pruned_dirs),
            max_files=self.max_files,
            max_file_bytes=self.max_file_bytes,
            truncated=truncated,
        )
        manifest = state.to_source_manifest()
        save_source_index(self.source_index_path, manifest)
        self._save_state(state)
        manifest_hash = _json_sha256(manifest.to_json())

        return LiveIndexRefresh(
            workspace_root=str(self.workspace_root),
            state_path=str(self.state_path),
            source_index_path=str(self.source_index_path),
            adapter_scope=dict(self.adapter_scope),
            changed_paths=sorted(changed_paths),
            indexed_paths=sorted(indexed_paths),
            deleted_paths=deleted_paths,
            unchanged_paths=sorted(unchanged_paths),
            skipped_paths=sorted(skipped_paths),
            file_count=len(next_files),
            truncated=truncated,
            manifest_sha256=manifest_hash,
        )

    def _iter_candidate_files(self) -> Iterable[tuple[str, Path, os.stat_result]]:
        for current, dirs, files in os.walk(self.workspace_root):
            dirs[:] = sorted(dirname for dirname in dirs if dirname not in self.pruned_dirs)
            for filename in sorted(files):
                path = Path(current) / filename
                if path.suffix.lower() not in self.suffixes:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = normalize_path(path.relative_to(self.workspace_root).as_posix())
                if is_protected_path(relative):
                    continue
                if stat.st_size > self.max_file_bytes:
                    continue
                yield relative, path, stat

    def _load_state(self) -> LiveIndexState | None:
        try:
            return LiveIndexState.from_json(json.loads(self.state_path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None

    def _save_state(self, state: LiveIndexState) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def load_live_index_state(path: Path) -> LiveIndexState:
    return LiveIndexState.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def _json_sha256(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
