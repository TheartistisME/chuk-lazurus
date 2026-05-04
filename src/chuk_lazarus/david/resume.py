"""Session resume snapshots for David terminal-agent runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA = "david.resume.v1"


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    workspace: str
    adapter_scope: dict[str, Any]
    memory_paths: dict[str, str]
    last_result_summary: str
    schema: str = SCHEMA
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "updated_at": self.updated_at,
            "session_id": self.session_id,
            "workspace": self.workspace,
            "adapter_scope": self.adapter_scope,
            "memory_paths": self.memory_paths,
            "last_result_summary": self.last_result_summary,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SessionSnapshot":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported resume schema: {data.get('schema')!r}")
        return cls(
            session_id=str(data["session_id"]),
            workspace=str(data["workspace"]),
            adapter_scope=dict(data.get("adapter_scope") or {}),
            memory_paths={str(key): str(value) for key, value in dict(data.get("memory_paths") or {}).items()},
            last_result_summary=str(data.get("last_result_summary") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


def default_resume_path(workspace: Path) -> Path:
    return Path(workspace).resolve() / ".david" / "resume.json"


def save_session_snapshot(snapshot: SessionSnapshot, path: Path | None = None) -> SessionSnapshot:
    target = Path(path) if path is not None else default_resume_path(Path(snapshot.workspace))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return snapshot


def load_session_snapshot(path: Path) -> SessionSnapshot | None:
    source = Path(path)
    if not source.exists():
        return None
    return SessionSnapshot.from_json(json.loads(source.read_text(encoding="utf-8")))


def summarize_result(result: Any, *, max_chars: int = 500) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        summary = result
    elif hasattr(result, "to_json"):
        data = result.to_json()
        summary = str(data.get("answer") or data)
    else:
        summary = str(result)
    return summary[:max_chars]
