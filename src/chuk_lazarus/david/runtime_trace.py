"""Structured runtime tracing and state-file locking for David."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import socket
from typing import Any
from uuid import uuid4


DEFAULT_LOCK_STALE_AFTER_SECONDS = 30 * 60
MAX_TRACE_STRING_CHARS = 2_000
MAX_TRACE_ITEMS = 24
MAX_TRACE_DEPTH = 4


class RuntimeStateLockError(RuntimeError):
    """Raised when David refuses to write shared state under an active lock."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_name(value: str) -> str:
    safe = []
    for char in str(value or "default"):
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("-")
    return "".join(safe).strip(".-") or "default"


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _bounded(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_TRACE_DEPTH:
        return _bounded_scalar(value)
    if isinstance(value, dict):
        bounded: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_TRACE_ITEMS:
                bounded["__truncated_items__"] = len(value) - MAX_TRACE_ITEMS
                break
            bounded[str(key)] = _bounded(item, depth=depth + 1)
        return bounded
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        bounded_items = [_bounded(item, depth=depth + 1) for item in items[:MAX_TRACE_ITEMS]]
        if len(items) > MAX_TRACE_ITEMS:
            bounded_items.append({"__truncated_items__": len(items) - MAX_TRACE_ITEMS})
        return bounded_items
    return _bounded_scalar(value)


def _bounded_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if len(text) <= MAX_TRACE_STRING_CHARS:
        return text
    return f"{text[:MAX_TRACE_STRING_CHARS]}...[truncated {len(text) - MAX_TRACE_STRING_CHARS} chars]"


def _route_summary(route: Any | None) -> dict[str, Any] | None:
    if route is None:
        return None
    summary: dict[str, Any] = {}
    for name in (
        "method",
        "methodology",
        "capability",
        "proof_rig",
        "memory_family",
        "session_id",
        "tier",
        "route_reason",
        "token_cost",
        "activation_score",
        "lexical_score",
        "ordinal_score",
        "recency_score",
        "residual_available",
        "kv_ready",
    ):
        if hasattr(route, name):
            value = getattr(route, name)
            if value not in (None, "", [], {}):
                summary[name] = value
    for name in (
        "selected_paths",
        "selected_tests",
        "selected_symbols",
        "route_reasons",
    ):
        if hasattr(route, name):
            values = list(getattr(route, name) or ())
            if values:
                summary[name] = values[:MAX_TRACE_ITEMS]
                if len(values) > MAX_TRACE_ITEMS:
                    summary[f"{name}_truncated"] = len(values) - MAX_TRACE_ITEMS
    evidence = getattr(route, "evidence", None)
    if evidence is not None:
        summary["evidence_count"] = len(evidence)
    return _bounded(summary) if summary else None


class RuntimeTraceWriter:
    """Append-only JSONL trace writer scoped to a David session."""

    def __init__(self, state_dir: Path, session_id: str) -> None:
        self.state_dir = Path(state_dir)
        self.session_id = str(session_id or "default")
        self.path = self.state_dir / "sessions" / f"{_safe_name(self.session_id)}-runtime.jsonl"

    def write(
        self,
        event_type: str,
        *,
        method: str | None = None,
        route: Any | None = None,
        ok: bool | None = None,
        error: BaseException | str | None = None,
        metadata: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_type": str(event_type),
            "session_id": self.session_id,
            "timestamp": utc_now_iso(),
        }
        if method:
            event["method"] = str(method)
        route_data = _route_summary(route)
        if route_data:
            event["route"] = route_data
        if ok is not None:
            event["ok"] = bool(ok)
        if error is not None:
            event["error"] = self._error_json(error)
        if metadata:
            event["metadata"] = _bounded(metadata)
        if payload:
            event["payload"] = _bounded(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        return event

    @staticmethod
    def _error_json(error: BaseException | str) -> dict[str, Any]:
        if isinstance(error, BaseException):
            return {
                "type": type(error).__name__,
                "message": _bounded_scalar(error),
            }
        return {
            "type": "error",
            "message": _bounded_scalar(error),
        }


@dataclass
class WorkspaceStateLock:
    """Best-effort exclusive-create lock for David state write sections."""

    state_dir: Path
    session_id: str = "default"
    reason: str = "state_write"
    stale_after_seconds: float = DEFAULT_LOCK_STALE_AFTER_SECONDS
    lock_name: str = "workspace-state.lock"
    owner_token: str = field(default_factory=lambda: uuid4().hex)
    path: Path = field(init=False)
    acquired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.state_dir = Path(self.state_dir)
        self.path = self.state_dir / "locks" / self.lock_name

    def __enter__(self) -> "WorkspaceStateLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.release()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._exclusive_create()
        except FileExistsError as exc:
            if not self._break_stale_lock_if_safe():
                raise RuntimeStateLockError(self._refusal_message()) from exc
            try:
                self._exclusive_create()
            except FileExistsError as retry_exc:
                raise RuntimeStateLockError(self._refusal_message()) from retry_exc
        self.acquired = True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            data = self._read_lock_json()
            if data.get("owner_token") == self.owner_token:
                self.path.unlink()
        except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
            pass
        finally:
            self.acquired = False

    def to_json(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "reason": self.reason,
            "owner_token": self.owner_token,
            "acquired": self.acquired,
        }

    def _exclusive_create(self) -> None:
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self._metadata(), indent=2, sort_keys=True) + "\n")
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema": "david.workspace_state_lock.v1",
            "created_at": utc_now_iso(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "session_id": self.session_id,
            "reason": self.reason,
            "owner_token": self.owner_token,
        }

    def _read_lock_json(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _break_stale_lock_if_safe(self) -> bool:
        try:
            data = self._read_lock_json()
        except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        created_at = _parse_utc(data.get("created_at"))
        if created_at is None or self.stale_after_seconds < 0:
            return False
        age = (datetime.now(timezone.utc) - created_at).total_seconds()
        if age <= self.stale_after_seconds:
            return False
        pid = data.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if _pid_is_running(pid):
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return True

    def _refusal_message(self) -> str:
        return f"David state is locked at {self.path}; refusing {self.reason}"


def _pid_is_running(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_pid_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


def _windows_pid_is_running(pid: int) -> bool:
    try:
        import ctypes
    except ImportError:
        return True
    kernel32 = ctypes.windll.kernel32
    process_query_limited_information = 0x1000
    synchronize = 0x00100000
    handle = kernel32.OpenProcess(process_query_limited_information | synchronize, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        still_active = 259
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)
