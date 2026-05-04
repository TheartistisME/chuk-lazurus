"""Plain ANSI terminal UI for the David agent surface."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .resume import SessionSnapshot, format_resume_snapshot, load_session_snapshot


HELP_TEXT = """David commands:
  prompt text              Send a one-shot prompt to the runtime
  /status, /readiness      Show model validation, index, and memory readiness
  /resume                  Show last saved session summary
  /memory                  Show user/task memory readiness and artifact details
  /index [jit|build]       Show or refresh the workspace index
  /verify [cmd]            Run the verifier or a workspace command as the gate
  /shell, /run <cmd>       Run a local shell command in the workspace
  /agent, /loop <action>   Run a deterministic agent tool/action loop
  /read <path>             Read a workspace-local file
  /write <path> <text>     Write a workspace-local file
  /patch, /apply <patch>   Apply strict search/replace or unified diff text
  /help                    Show this help
  /exit, /quit             Exit David"""


@dataclass(frozen=True)
class CommandResult:
    text: str
    should_exit: bool = False


class DavidTui:
    """A testable terminal UI without curses or rich dependencies."""

    def __init__(
        self,
        runtime: Any,
        *,
        color: bool = True,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.runtime = runtime
        self.color = color
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout

    def run(self, once: str | None = None) -> int:
        self._write_startup()
        if once is not None:
            result = self.dispatch(once)
            if result.text:
                self._write_block(result.text)
            return 0

        self._write_block("Type /help for commands.")
        while True:
            self._write("david> ")
            line = self.input_stream.readline()
            if line == "":
                self._write("\n")
                return 0
            result = self.dispatch(line.strip())
            if result.text:
                self._write_block(result.text)
            if result.should_exit:
                return 0

    def dispatch(self, text: str) -> CommandResult:
        if not text:
            return CommandResult("")
        if not text.startswith("/"):
            return CommandResult(self._runtime_call(("respond", "ask", "prompt", "run_once"), text))

        command, _, arg = text.partition(" ")
        command = command.lower()
        arg = arg.strip() or None

        if command in {"/exit", "/quit"}:
            return CommandResult("bye", should_exit=True)
        if command == "/help":
            return CommandResult(HELP_TEXT)
        if command in {"/status", "/readiness"}:
            return CommandResult(self.format_readiness())
        if command == "/resume":
            return CommandResult(self.format_resume())
        if command == "/memory":
            fallback = f"memory: {self._readiness().get('memory', 'unknown')}"
            return CommandResult(self._runtime_call(("memory_status",), fallback=fallback))
        if command == "/index":
            return CommandResult(self._index_command(arg))
        if command == "/verify":
            if hasattr(self.runtime, "verify"):
                return CommandResult(self._runtime_call(("verify",), arg))
            prompt = f"/verify {arg}" if arg else "/verify"
            return CommandResult(self._runtime_call(("run_once",), prompt))
        if command in {"/run", "/shell"}:
            if not arg:
                return CommandResult("shell: missing command")
            return CommandResult(self._runtime_call(("run_shell", "run"), arg))
        if command in {"/agent", "/loop"}:
            if not arg:
                return CommandResult("agent: missing action payload")
            return CommandResult(self._agent_loop_command(arg))
        if command == "/read":
            if not arg:
                return CommandResult("read: missing path")
            return CommandResult(self._runtime_call(("read_file", "read"), arg))
        if command == "/write":
            if not arg:
                return CommandResult("write: missing path and text")
            return CommandResult(self._runtime_call(("write_file", "write"), arg))
        if command in {"/apply", "/patch"}:
            if not arg:
                return CommandResult("patch: missing patch text")
            patch_text = arg.replace("\\n", "\n")
            return CommandResult(self._runtime_call(("apply_patch",), patch_text))
        return CommandResult(f"unknown command: {command}\n{HELP_TEXT}")

    def _agent_loop_command(self, arg: str) -> str:
        for name in ("run_agent_loop", "agent_loop"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                return self._format_agent_loop_result(method(arg))
        return "agent: runtime loop hook unavailable"

    def _format_agent_loop_result(self, value: Any) -> str:
        loop = getattr(value, "loop", None)
        if loop is None and isinstance(value, dict):
            loop = value.get("loop")
        if loop is None:
            return self._stringify_runtime_value(value)

        if isinstance(loop, dict):
            status = str(loop.get("status", "unknown"))
            steps = loop.get("steps", "?")
            verified = loop.get("verified", False)
            reason = str(loop.get("reason", ""))
            trace = loop.get("trace") or []
        else:
            status = str(getattr(loop, "status", "unknown"))
            steps = getattr(loop, "steps", "?")
            verified = getattr(loop, "verified", False)
            reason = str(getattr(loop, "reason", ""))
            trace = [step.to_dict() for step in getattr(loop, "trace", ())]

        lines = [f"agent loop: {status} steps={steps} verified={verified}"]
        if reason:
            lines.append(f"reason: {reason}")
        for item in trace:
            if not isinstance(item, dict):
                continue
            action = item.get("action", "unknown")
            ok = item.get("ok", False)
            observation = item.get("observation") or {}
            detail = self._agent_loop_observation_summary(observation)
            suffix = f" {detail}" if detail else ""
            lines.append(f"- {item.get('step', '?')}: {action} ok={ok}{suffix}")
        return "\n".join(lines)

    def _agent_loop_observation_summary(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return ""
        if "path" in observation:
            if "bytes" in observation:
                return f"path={observation['path']} bytes={observation['bytes']}"
            return f"path={observation['path']}"
        if "returncode" in observation:
            return f"rc={observation['returncode']}"
        if "passed" in observation:
            return f"passed={observation['passed']}"
        if "error" in observation:
            return f"error={observation['error']}"
        if "reason" in observation:
            return f"reason={observation['reason']}"
        return ""

    def format_readiness(self) -> str:
        readiness = self._readiness()
        lines = ["David startup readiness"]
        for name in ("model validation", "index", "memory"):
            lines.append(f"- {name}: {readiness.get(name, 'unknown')}")
        for name, value in readiness.items():
            if name not in {"model validation", "index", "memory"}:
                lines.append(f"- {name}: {value}")
        return "\n".join(lines)

    def format_resume(self) -> str:
        direct = self._runtime_call(("resume_status", "resume_summary"), fallback=None)
        if direct != "runtime hook unavailable":
            return self._compact_multiline(direct)
        return format_resume_snapshot(self._resume_snapshot())

    def _write_startup(self) -> None:
        title = self._style("David terminal agent", "bold")
        self._write_block(title)
        self._write_block(self.format_readiness())
        snapshot = self._resume_snapshot()
        if snapshot is not None:
            self._write_block(format_resume_snapshot(snapshot))

    def _readiness(self) -> dict[str, str]:
        for name in ("readiness", "startup_status", "status"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                value = method()
                return self._normalize_readiness(value)
        index = getattr(self.runtime, "index", None)
        index_check = getattr(index, "check", None)
        index_value = None
        if callable(index_check):
            checked = index_check()
            ready = getattr(checked, "ready", None)
            reason = getattr(checked, "reason", None)
            index_value = "ready" if ready else f"missing: {reason or 'JIT required'}"
        return {
            "model validation": self._model_validation_status(),
            "index": index_value or "unknown",
            "memory": self._memory_status_for_startup(),
        }

    def _normalize_readiness(self, value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            return {str(key): str(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return {
                str(index): str(item)
                for index, item in enumerate(value, start=1)
            }
        return {"status": str(value)}

    def _index_command(self, arg: str | None) -> str:
        action = (arg or "status").strip().lower()
        if action in {"status", "check"}:
            fallback = f"index: {self._readiness().get('index', 'unknown')}"
            return self._runtime_call(("index_status",), fallback=fallback)
        if action not in {"jit", "build", "refresh"}:
            return "index: expected /index [jit|build]"

        direct = self._runtime_call(("jit_index", "build_index", "refresh_index"), fallback=None)
        if direct != "runtime hook unavailable":
            return direct

        index = getattr(self.runtime, "index", None)
        jit = getattr(index, "jit", None)
        if callable(jit):
            value = jit()
            summary = self._stringify_runtime_value(value)
            status = self._runtime_call(("index_status",), fallback="")
            if summary and status:
                return f"{summary}\n{status}"
            return summary or status or "index: refreshed"
        return "index: JIT hook unavailable"

    def _runtime_call(
        self,
        names: tuple[str, ...],
        *args: Any,
        fallback: str | None = None,
    ) -> str:
        for name in names:
            method = getattr(self.runtime, name, None)
            if callable(method):
                value = method(*args)
                return self._stringify_runtime_value(value)
        return fallback or "runtime hook unavailable"

    def _stringify_runtime_value(self, value: Any) -> str:
        if value is None:
            return ""
        answer = getattr(value, "answer", None)
        if answer is not None:
            return str(answer)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return str(to_dict())
        to_json = getattr(value, "to_json", None)
        if callable(to_json):
            return str(to_json())
        return str(value)

    def _resume_snapshot(self) -> SessionSnapshot | None:
        snapshot = getattr(self.runtime, "resume_snapshot", None)
        if isinstance(snapshot, SessionSnapshot):
            return snapshot
        if isinstance(snapshot, dict):
            try:
                return SessionSnapshot.from_json(snapshot)
            except (KeyError, TypeError, ValueError):
                return None

        path = getattr(self.runtime, "resume_path", None)
        if path is not None:
            try:
                return load_session_snapshot(Path(path))
            except (OSError, ValueError):
                return None
        return None

    def _compact_multiline(self, text: str, *, max_chars: int = 500) -> str:
        lines = []
        for line in str(text).splitlines() or [""]:
            clean = " ".join(line.split())
            if clean:
                lines.append(clean)
        compacted = "\n".join(lines)
        if len(compacted) <= max_chars:
            return compacted
        return f"{compacted[: max_chars - 3].rstrip()}..."

    def _model_validation_status(self) -> str:
        config = getattr(self.runtime, "config", None)
        adapter = getattr(config, "adapter", None)
        if adapter is not None:
            family = getattr(adapter, "adapter_family", "adapter")
            model = getattr(adapter, "model_id", "unknown")
            return f"ready ({family}:{model})"
        return "unknown"

    def _memory_status_for_startup(self) -> str:
        memory = getattr(self.runtime, "memory", None)
        if memory is not None:
            return "ready"
        return "unknown"

    def _write_block(self, text: str) -> None:
        self._write(f"{text.rstrip()}\n")

    def _write(self, text: str) -> None:
        self.output_stream.write(text)
        self.output_stream.flush()

    def _style(self, text: str, style: str) -> str:
        if not self.color:
            return text
        if style == "bold":
            return f"\033[1m{text}\033[0m"
        return text
