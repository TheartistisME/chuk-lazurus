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


@dataclass(frozen=True)
class ResumeLoadStatus:
    path: Path
    state: str
    snapshot: SessionSnapshot | None = None
    message: str = ""

    @property
    def is_valid(self) -> bool:
        return self.state == "valid" and self.snapshot is not None


def default_resume_path(workspace: Path) -> Path:
    return Path(workspace).resolve() / ".david" / "resume.json"


def save_session_snapshot(snapshot: SessionSnapshot, path: Path | None = None) -> SessionSnapshot:
    target = Path(path) if path is not None else default_resume_path(Path(snapshot.workspace))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    return snapshot


def load_session_snapshot(path: Path) -> SessionSnapshot | None:
    status = load_session_snapshot_status(path)
    if status.is_valid:
        return status.snapshot
    return None


def load_session_snapshot_status(path: Path) -> ResumeLoadStatus:
    source = Path(path)
    try:
        exists = source.exists()
    except OSError as exc:
        return ResumeLoadStatus(path=source, state="unreadable", message=str(exc))
    if not exists:
        return ResumeLoadStatus(path=source, state="missing", message="no saved session")

    try:
        raw = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return ResumeLoadStatus(
            path=source,
            state="unreadable",
            message=f"resume is not valid UTF-8: {exc.reason}",
        )
    except OSError as exc:
        return ResumeLoadStatus(path=source, state="unreadable", message=str(exc))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ResumeLoadStatus(
            path=source,
            state="corrupt",
            message=f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )
    if not isinstance(data, dict):
        return ResumeLoadStatus(
            path=source,
            state="corrupt",
            message=f"resume root must be a JSON object, got {type(data).__name__}",
        )

    try:
        snapshot = SessionSnapshot.from_json(data)
    except (KeyError, TypeError, ValueError) as exc:
        return ResumeLoadStatus(path=source, state="corrupt", message=str(exc))
    return ResumeLoadStatus(
        path=source,
        state="valid",
        snapshot=snapshot,
        message="saved session ready",
    )


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


def compact_summary(text: str, *, max_chars: int = 240) -> str:
    summary = " ".join(str(text or "").split())
    if len(summary) <= max_chars:
        return summary
    return f"{summary[: max_chars - 3].rstrip()}..."


def format_resume_snapshot(snapshot: SessionSnapshot | None, *, max_summary_chars: int = 240) -> str:
    if snapshot is None:
        return "David resume\n- status: no saved session"

    summary = compact_summary(snapshot.last_result_summary, max_chars=max_summary_chars)
    return "\n".join(
        [
            "David resume",
            f"- session: {snapshot.session_id or 'unknown'}",
            f"- workspace: {snapshot.workspace or 'unknown'}",
            f"- updated: {snapshot.updated_at or 'unknown'}",
            f"- last result: {summary or 'none'}",
        ]
    )


def format_resume_status(status: ResumeLoadStatus, *, max_summary_chars: int = 240) -> str:
    if status.is_valid:
        return format_resume_snapshot(status.snapshot, max_summary_chars=max_summary_chars)
    if status.state == "missing":
        return format_resume_snapshot(None, max_summary_chars=max_summary_chars)

    label = "unreadable saved session" if status.state == "unreadable" else "corrupt saved session"
    detail = compact_summary(status.message or "unknown resume error", max_chars=500)
    return "\n".join(
        [
            "David resume",
            f"- status: {label}",
            f"- path: {status.path}",
            f"- detail: {detail}",
        ]
    )
