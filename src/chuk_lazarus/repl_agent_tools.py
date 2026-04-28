"""Local coding-agent tools for the interactive memory REPL.

Tool traces are append-only JSONL records. The chat transcript and tool traces
are the source of truth; any memory index built from them is derived state.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOL_SYSTEM_PROMPT = """You can request local coding tools when you need repository context or edits.
Emit one or more tool calls, then wait for TOOL_RESULT before answering normally.
Use either:
<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>
or a fenced block:
```tool_call
{"name":"read_file","arguments":{"path":"README.md"}}
```
Available tools: list_dir, read_file, search, shell, apply_patch.
Tool JSON must contain only JSON, with no prose inside it. Use "arguments" or "args"."""


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    output: str
    error: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


_XML_TOOL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FENCED_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _bounded(text: Any, limit: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n[truncated {len(value) - limit} chars]"


def _coerce_tool_call(obj: Any) -> list[ToolCall]:
    if isinstance(obj, list):
        calls: list[ToolCall] = []
        for item in obj:
            calls.extend(_coerce_tool_call(item))
        return calls
    if not isinstance(obj, dict):
        return []
    name = obj.get("name")
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(name, str) or not name.strip():
        return []
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return []
    call_id = obj.get("call_id") or obj.get("id") or uuid.uuid4().hex
    return [ToolCall(name=name.strip(), arguments=dict(args), call_id=str(call_id))]


def _parse_json_tool_calls(payload: str) -> list[ToolCall]:
    try:
        parsed = json.loads(payload.strip())
    except json.JSONDecodeError:
        return []
    return _coerce_tool_call(parsed)


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Extract XML-ish or fenced JSON tool calls from model text.

    Invalid JSON is ignored so a malformed request never crashes the REPL.
    """
    calls: list[ToolCall] = []
    for match in _XML_TOOL_RE.finditer(text or ""):
        calls.extend(_parse_json_tool_calls(match.group(1)))

    for match in _FENCED_RE.finditer(text or ""):
        info = " ".join(match.group(1).strip().lower().split())
        payload = match.group(2)
        if "tool_call" in info:
            calls.extend(_parse_json_tool_calls(payload))
            continue
        if info == "json":
            candidate_calls = _parse_json_tool_calls(payload)
            if candidate_calls:
                calls.extend(candidate_calls)
    return calls


class LocalCodingToolRunner:
    """Execute bounded local tools rooted inside a workspace."""

    def __init__(
        self,
        workspace_root: Path,
        trace_root: Path,
        timeout_seconds: int = 30,
        max_output_chars: int = 12000,
        allow_shell: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.trace_root = Path(trace_root).resolve()
        self.timeout_seconds = int(timeout_seconds)
        self.max_output_chars = int(max_output_chars)
        self.allow_shell = bool(allow_shell)
        self.trace_root.mkdir(parents=True, exist_ok=True)

    def _resolve_inside_workspace(self, raw_path: Any = ".") -> Path:
        path_text = "." if raw_path in (None, "") else str(raw_path)
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = self.workspace_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"path is outside workspace: {path_text}") from exc
        return resolved

    def _result(
        self,
        call: ToolCall,
        *,
        ok: bool,
        output: str = "",
        error: str = "",
        started_at: str,
        finished_at: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=ok,
            output=_bounded(output, self.max_output_chars),
            error=_bounded(error, self.max_output_chars),
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed_seconds,
            metadata=metadata or {},
        )

    def _list_dir(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        path = self._resolve_inside_workspace(call.arguments.get("path", "."))
        if not path.exists():
            return False, "", f"path does not exist: {path}", {}
        if not path.is_dir():
            return False, "", f"path is not a directory: {path}", {}
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        capped = entries[:500]
        output = "\n".join(capped)
        if len(entries) > len(capped):
            output += f"\n[truncated {len(entries) - len(capped)} entries]"
        return True, output, "", {"path": str(path), "count": len(entries)}

    def _read_file(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        path = self._resolve_inside_workspace(call.arguments.get("path", "."))
        start_line = max(1, int(call.arguments.get("start_line", 1) or 1))
        limit = max(1, int(call.arguments.get("limit", 200) or 200))
        if not path.exists():
            return False, "", f"path does not exist: {path}", {}
        if not path.is_file():
            return False, "", f"path is not a file: {path}", {}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        start_index = start_line - 1
        selected = lines[start_index : start_index + limit]
        output_lines = [f"{idx}: {line}" for idx, line in enumerate(selected, start=start_line)]
        if start_index + limit < len(lines):
            output_lines.append(f"[truncated after line {start_index + limit} of {len(lines)}]")
        return True, "\n".join(output_lines), "", {
            "path": str(path),
            "start_line": start_line,
            "limit": limit,
            "total_lines": len(lines),
        }

    def _search(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        pattern = str(call.arguments.get("pattern", ""))
        if not pattern:
            return False, "", "search requires a pattern", {}
        path = self._resolve_inside_workspace(call.arguments.get("path", "."))
        if not path.exists():
            return False, "", f"path does not exist: {path}", {}
        max_results = max(1, int(call.arguments.get("max_results", 100) or 100))
        glob_value = call.arguments.get("glob")
        rg = self._run_rg_search(pattern, path, glob_value, max_results)
        if rg is not None:
            return rg
        return self._python_search(pattern, path, glob_value, max_results)

    def _run_rg_search(
        self,
        pattern: str,
        path: Path,
        glob_value: Any,
        max_results: int,
    ) -> tuple[bool, str, str, dict[str, Any]] | None:
        args = [
            "rg",
            "-n",
            "--hidden",
            "--glob",
            "!.git/**",
            "--max-count",
            str(max_results),
        ]
        if glob_value:
            args.extend(["--glob", str(glob_value)])
        args.extend(["--", pattern, str(path)])
        try:
            completed = subprocess.run(
                args,
                cwd=self.workspace_root,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        output = completed.stdout
        lines = output.splitlines()[:max_results]
        return True, "\n".join(lines), completed.stderr, {
            "engine": "rg",
            "exit_code": completed.returncode,
            "path": str(path),
        }

    def _python_search(
        self,
        pattern: str,
        path: Path,
        glob_value: Any,
        max_results: int,
    ) -> tuple[bool, str, str, dict[str, Any]]:
        try:
            regex = re.compile(pattern)
            use_regex = True
        except re.error:
            regex = None
            use_regex = False
        paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        results: list[str] = []
        for file_path in paths:
            if ".git" in file_path.parts:
                continue
            if glob_value and not file_path.match(str(glob_value)):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                matched = bool(regex.search(line)) if use_regex and regex is not None else pattern in line
                if not matched:
                    continue
                try:
                    rel_path = file_path.relative_to(self.workspace_root)
                except ValueError:
                    rel_path = file_path
                results.append(f"{rel_path}:{line_number}:{line}")
                if len(results) >= max_results:
                    return True, "\n".join(results), "", {"engine": "python", "truncated": True}
        return True, "\n".join(results), "", {"engine": "python", "truncated": False}

    def _shell(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        if not self.allow_shell:
            return False, "", "shell tool is disabled", {}
        command = str(call.arguments.get("command", ""))
        if not command.strip():
            return False, "", "shell requires a command", {}
        requested_timeout = call.arguments.get("timeout_seconds")
        timeout = self.timeout_seconds
        if requested_timeout is not None:
            timeout = min(timeout, max(1, int(requested_timeout)))
        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=self.workspace_root,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return False, str(stdout), f"command timed out after {timeout}s\n{stderr}", {
                "timeout_seconds": timeout,
            }
        output = completed.stdout
        error = completed.stderr
        if completed.returncode != 0:
            error = (error + f"\nexit_code={completed.returncode}").strip()
        return completed.returncode == 0, output, error, {
            "exit_code": completed.returncode,
            "timeout_seconds": timeout,
            "command": command,
        }

    def _apply_patch(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        patch = str(call.arguments.get("patch", ""))
        if not patch.strip():
            return False, "", "apply_patch requires a patch", {}
        completed = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch,
            cwd=self.workspace_root,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        error = completed.stderr
        if completed.returncode != 0:
            error = (error + f"\nexit_code={completed.returncode}").strip()
        return completed.returncode == 0, completed.stdout, error, {
            "exit_code": completed.returncode,
        }

    def _dispatch(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        if call.name == "list_dir":
            return self._list_dir(call)
        if call.name == "read_file":
            return self._read_file(call)
        if call.name == "search":
            return self._search(call)
        if call.name == "shell":
            return self._shell(call)
        if call.name == "apply_patch":
            return self._apply_patch(call)
        return False, "", f"unknown tool: {call.name}", {}

    def execute(
        self,
        call: ToolCall,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        started_at = _utc_now()
        t0 = time.perf_counter()
        try:
            ok, output, error, metadata = self._dispatch(call)
        except Exception as exc:  # noqa: BLE001 - tool errors must be data
            ok, output, error, metadata = False, "", str(exc), {"exception": type(exc).__name__}
        elapsed = time.perf_counter() - t0
        finished_at = _utc_now()
        result = self._result(
            call,
            ok=ok,
            output=output,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            metadata=metadata,
        )
        self._write_trace(call, result, session_id=session_id, turn_index=turn_index)
        return result

    def _write_trace(
        self,
        call: ToolCall,
        result: ToolResult,
        *,
        session_id: str | None,
        turn_index: int | None,
    ) -> None:
        trace_session = session_id or "manual"
        safe_session = re.sub(r"[^A-Za-z0-9_.-]+", "_", trace_session) or "manual"
        record = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "cwd": str(self.workspace_root),
            "workspace": str(self.workspace_root),
            "session_id": session_id,
            "turn_index": turn_index,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "elapsed_seconds": result.elapsed_seconds,
            "metadata": result.metadata,
        }
        trace_path = self.trace_root / f"{safe_session}.jsonl"
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results as a bounded user-message payload."""
    payload: list[dict[str, Any]] = []
    for result in results:
        item = asdict(result)
        item["output"] = _bounded(item.get("output", ""), 6000)
        item["error"] = _bounded(item.get("error", ""), 2000)
        payload.append(item)
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if len(text) > 20000:
        compact_payload = []
        for item in payload[:20]:
            compact = dict(item)
            compact["output"] = _bounded(compact.get("output", ""), 1000)
            compact["error"] = _bounded(compact.get("error", ""), 500)
            compact_payload.append(compact)
        if len(payload) > len(compact_payload):
            compact_payload.append({"ok": False, "error": "tool result list truncated"})
        text = json.dumps(compact_payload, separators=(",", ":"), ensure_ascii=False)
        if len(text) > 20000:
            text = json.dumps(
                [{"ok": False, "error": "tool results omitted because payload was too large"}],
                separators=(",", ":"),
            )
    return "TOOL_RESULT\n" + text
