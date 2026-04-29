"""Local coding-agent tools for the interactive memory REPL.

Tool traces are append-only JSONL records. The chat transcript and tool traces
are the source of truth; any memory index built from them is derived state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
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
Available tools: list_dir, read_file, write_file, save_custom_tool,
read_custom_tool, list_custom_tools, search, shell, apply_patch, plus loaded
custom tools.
write_file creates new files by default; set overwrite=true or append=true for existing files.
save_custom_tool persists a Python or bash tool for this and future runs. The
script receives tool arguments as JSON on stdin, runs from the workspace root,
prints useful output to stdout, and should exit nonzero on failure.
Tool JSON must contain only JSON, with no prose inside it. Use "arguments" or "args"."""

_BUILTIN_TOOL_NAMES = frozenset(
    {
        "list_dir",
        "read_file",
        "write_file",
        "save_custom_tool",
        "read_custom_tool",
        "list_custom_tools",
        "search",
        "shell",
        "apply_patch",
    }
)
_CUSTOM_TOOL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_CUSTOM_TOOL_RUNTIMES = frozenset({"python", "bash"})


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


@dataclass
class CustomTool:
    name: str
    description: str
    runtime: str
    entrypoint: str
    input_schema: dict[str, Any]
    manifest_path: Path
    script_path: Path


_XML_TOOL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_FENCED_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _bounded(text: Any, limit: int) -> str:
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n[truncated {len(value) - limit} chars]"


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(content, encoding=encoding)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


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
        custom_tools_root: Path | None = None,
        allow_custom_tools: bool = True,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.trace_root = Path(trace_root).resolve()
        self.timeout_seconds = int(timeout_seconds)
        self.max_output_chars = int(max_output_chars)
        self.allow_shell = bool(allow_shell)
        self.custom_tools_root = Path(
            custom_tools_root or self.trace_root.parent / "custom_tools"
        ).resolve()
        self.allow_custom_tools = bool(allow_custom_tools)
        self.custom_tools: dict[str, CustomTool] = {}
        self.trace_root.mkdir(parents=True, exist_ok=True)
        if self.allow_custom_tools:
            self.custom_tools_root.mkdir(parents=True, exist_ok=True)
            self.reload_custom_tools()

    def _normalize_custom_tool_name(self, raw_name: Any) -> str:
        name = str(raw_name or "").strip()
        if not _CUSTOM_TOOL_NAME_RE.fullmatch(name):
            raise ValueError(
                "custom tool name must match [A-Za-z_][A-Za-z0-9_]{0,63}"
            )
        return name

    def _resolve_inside_custom_tools(self, raw_path: Any) -> Path:
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = self.custom_tools_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.custom_tools_root)
        except ValueError as exc:
            raise ValueError(f"path is outside custom tools root: {raw_path}") from exc
        return resolved

    def reload_custom_tools(self) -> dict[str, CustomTool]:
        self.custom_tools = {}
        if not self.allow_custom_tools or not self.custom_tools_root.exists():
            return self.custom_tools
        for manifest_path in sorted(self.custom_tools_root.glob("*.json")):
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                name = self._normalize_custom_tool_name(data.get("name"))
                if name in _BUILTIN_TOOL_NAMES:
                    continue
                runtime = str(data.get("runtime", "")).strip().lower()
                if runtime not in _CUSTOM_TOOL_RUNTIMES:
                    continue
                entrypoint = str(data.get("entrypoint", "")).strip()
                if not entrypoint:
                    continue
                script_path = self._resolve_inside_custom_tools(entrypoint)
                if not script_path.is_file():
                    continue
                input_schema = data.get("input_schema", {})
                if not isinstance(input_schema, dict):
                    input_schema = {}
                self.custom_tools[name] = CustomTool(
                    name=name,
                    description=str(data.get("description", "")).strip(),
                    runtime=runtime,
                    entrypoint=entrypoint,
                    input_schema=input_schema,
                    manifest_path=manifest_path,
                    script_path=script_path,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return self.custom_tools

    def available_tool_names(self) -> list[str]:
        self.reload_custom_tools()
        return sorted(_BUILTIN_TOOL_NAMES | set(self.custom_tools))

    def tool_system_prompt(self) -> str:
        self.reload_custom_tools()
        prompt = TOOL_SYSTEM_PROMPT + f"\nCustom tool storage: {self.custom_tools_root}"
        if not self.custom_tools:
            return prompt + "\nLoaded custom tools: none."
        lines = ["Loaded custom tools:"]
        for tool in sorted(self.custom_tools.values(), key=lambda item: item.name)[:50]:
            desc = _bounded(tool.description, 220).replace("\n", " ")
            lines.append(f"- {tool.name} ({tool.runtime}): {desc or 'no description'}")
        if len(self.custom_tools) > 50:
            lines.append(f"- [truncated {len(self.custom_tools) - 50} custom tools]")
        return prompt + "\n" + "\n".join(lines)

    def describe_custom_tools(self) -> str:
        self.reload_custom_tools()
        lines = [f"custom tool storage: {self.custom_tools_root}"]
        if not self.custom_tools:
            lines.append("(no custom tools installed)")
        for tool in sorted(self.custom_tools.values(), key=lambda item: item.name):
            lines.append(
                f"- {tool.name} ({tool.runtime}): "
                f"{_bounded(tool.description, 300).replace(chr(10), ' ')}"
            )
        return "\n".join(lines)

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

    def _write_file(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        raw_path = call.arguments.get("path")
        if raw_path in (None, ""):
            return False, "", "write_file requires a path", {}
        if "content" not in call.arguments:
            return False, "", "write_file requires content", {}

        path = self._resolve_inside_workspace(raw_path)
        content = str(call.arguments.get("content", ""))
        encoding = str(call.arguments.get("encoding", "utf-8") or "utf-8")
        overwrite = bool(call.arguments.get("overwrite", False))
        append = bool(call.arguments.get("append", False))
        make_dirs = bool(call.arguments.get("make_dirs", False))
        expected_sha256 = call.arguments.get("expected_sha256")
        if overwrite and append:
            return False, "", "write_file cannot use both overwrite and append", {}
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                return False, "", "expected_sha256 must be a 64-character hex digest", {}

        parent = path.parent
        try:
            parent.relative_to(self.workspace_root)
        except ValueError:
            return False, "", f"path parent is outside workspace: {raw_path}", {}
        if not parent.exists():
            if make_dirs:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                return False, "", f"parent directory does not exist: {parent}", {}
        if not parent.is_dir():
            return False, "", f"path parent is not a directory: {parent}", {}

        existed = path.exists()
        if existed and not path.is_file():
            return False, "", f"path is not a file: {path}", {}
        if existed and not overwrite and not append:
            return False, "", "path exists; set overwrite=true or append=true", {}
        if expected_sha256 is not None:
            if not existed:
                return False, "", "expected_sha256 was provided but file does not exist", {}
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != expected_sha256:
                return False, "", "expected_sha256 did not match current file", {
                    "actual_sha256": actual_sha256,
                }

        mode = "append" if append else ("overwrite" if existed else "create")
        write_content = content
        if append and existed:
            write_content = path.read_text(encoding=encoding, errors="replace") + content

        _atomic_write_text(path, write_content, encoding=encoding)

        written_bytes = len(content.encode(encoding))
        return True, f"wrote {written_bytes} bytes to {path.relative_to(self.workspace_root)}", "", {
            "path": str(path),
            "mode": mode,
            "existed": existed,
            "bytes_written": written_bytes,
            "encoding": encoding,
        }

    def _custom_tool_manifest(self, tool: CustomTool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "runtime": tool.runtime,
            "entrypoint": tool.entrypoint,
            "input_schema": tool.input_schema,
        }

    def _save_custom_tool(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        if not self.allow_custom_tools:
            return False, "", "custom tools are disabled", {}
        try:
            name = self._normalize_custom_tool_name(call.arguments.get("name"))
        except ValueError as exc:
            return False, "", str(exc), {}
        if name in _BUILTIN_TOOL_NAMES:
            return False, "", f"custom tool cannot shadow built-in tool: {name}", {}
        runtime = str(call.arguments.get("runtime", "python") or "python").strip().lower()
        if runtime not in _CUSTOM_TOOL_RUNTIMES:
            return False, "", "custom tool runtime must be one of: bash, python", {}
        if "content" not in call.arguments:
            return False, "", "save_custom_tool requires content", {}
        description = str(call.arguments.get("description", "")).strip()
        if not description:
            return False, "", "save_custom_tool requires description", {}
        input_schema = call.arguments.get("input_schema", {})
        if input_schema is None:
            input_schema = {}
        if not isinstance(input_schema, dict):
            return False, "", "input_schema must be a JSON object", {}

        overwrite = bool(call.arguments.get("overwrite", False))
        extension = ".py" if runtime == "python" else ".sh"
        entrypoint = f"{name}{extension}"
        script_path = self.custom_tools_root / entrypoint
        manifest_path = self.custom_tools_root / f"{name}.json"
        if (script_path.exists() or manifest_path.exists()) and not overwrite:
            return False, "", "custom tool exists; set overwrite=true to replace it", {}

        self.custom_tools_root.mkdir(parents=True, exist_ok=True)
        content = str(call.arguments.get("content", ""))
        _atomic_write_text(script_path, content)
        if runtime == "bash":
            script_path.chmod(script_path.stat().st_mode | 0o700)
        manifest = {
            "name": name,
            "description": description,
            "runtime": runtime,
            "entrypoint": entrypoint,
            "input_schema": input_schema,
        }
        _atomic_write_text(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        self.reload_custom_tools()
        return True, f"saved custom tool {name}; call it as tool name {name}", "", {
            "custom_tool": True,
            "name": name,
            "runtime": runtime,
            "manifest_path": str(manifest_path),
            "script_path": str(script_path),
        }

    def _read_custom_tool(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        if not self.allow_custom_tools:
            return False, "", "custom tools are disabled", {}
        try:
            name = self._normalize_custom_tool_name(call.arguments.get("name"))
        except ValueError as exc:
            return False, "", str(exc), {}
        self.reload_custom_tools()
        tool = self.custom_tools.get(name)
        if tool is None:
            return False, "", f"custom tool not found: {name}", {}
        payload = {
            "manifest": self._custom_tool_manifest(tool),
            "content": tool.script_path.read_text(encoding="utf-8", errors="replace"),
        }
        return True, json.dumps(payload, indent=2, sort_keys=True), "", {
            "custom_tool": True,
            "name": name,
            "script_path": str(tool.script_path),
        }

    def _list_custom_tools(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        del call
        if not self.allow_custom_tools:
            return False, "", "custom tools are disabled", {}
        output = self.describe_custom_tools()
        return True, output, "", {
            "custom_tool_count": len(self.custom_tools),
            "custom_tools_root": str(self.custom_tools_root),
            "custom_tool_names": sorted(self.custom_tools),
        }

    def _run_custom_tool(self, call: ToolCall) -> tuple[bool, str, str, dict[str, Any]]:
        if not self.allow_custom_tools:
            return False, "", "custom tools are disabled", {}
        self.reload_custom_tools()
        tool = self.custom_tools.get(call.name)
        if tool is None:
            return False, "", f"custom tool not found: {call.name}", {}
        if tool.runtime == "python":
            command = [sys.executable, str(tool.script_path)]
        elif tool.runtime == "bash":
            command = ["bash", str(tool.script_path)]
        else:
            return False, "", f"unsupported custom tool runtime: {tool.runtime}", {}

        env = os.environ.copy()
        env.update(
            {
                "CUSTOM_TOOL_DIR": str(self.custom_tools_root),
                "CUSTOM_TOOL_NAME": tool.name,
                "WORKSPACE_ROOT": str(self.workspace_root),
            }
        )
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(call.arguments, separators=(",", ":")),
                cwd=self.workspace_root,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return False, str(stdout), f"custom tool timed out\n{stderr}", {
                "custom_tool": True,
                "name": tool.name,
                "timeout_seconds": self.timeout_seconds,
            }
        error = completed.stderr
        if completed.returncode != 0:
            error = (error + f"\nexit_code={completed.returncode}").strip()
        return completed.returncode == 0, completed.stdout, error, {
            "custom_tool": True,
            "name": tool.name,
            "runtime": tool.runtime,
            "exit_code": completed.returncode,
            "script_path": str(tool.script_path),
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
        if call.name == "write_file":
            return self._write_file(call)
        if call.name == "save_custom_tool":
            return self._save_custom_tool(call)
        if call.name == "read_custom_tool":
            return self._read_custom_tool(call)
        if call.name == "list_custom_tools":
            return self._list_custom_tools(call)
        if call.name == "search":
            return self._search(call)
        if call.name == "shell":
            return self._shell(call)
        if call.name == "apply_patch":
            return self._apply_patch(call)
        self.reload_custom_tools()
        if call.name in self.custom_tools:
            return self._run_custom_tool(call)
        return False, "", f"unknown tool: {call.name}; available: {self.available_tool_names()}", {}

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
