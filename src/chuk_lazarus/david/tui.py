"""Plain ANSI terminal UI for the David agent surface."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, TextIO


HELP_TEXT = """Commands:
  /status          Show model validation, index, and memory readiness
  /memory          Show memory readiness/details
  /index           Show workspace index readiness
  /verify [cmd]    Run the verifier or configured verify command
  /run <cmd>       Run a local shell command in the workspace
  /read <path>     Read a workspace-local file
  /write <path> <text>
                   Write a workspace-local file
  /apply <patch>   Apply a strict search/replace or unified diff patch
  /help            Show this help
  /exit            Exit David"""


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

        if command == "/exit":
            return CommandResult("bye", should_exit=True)
        if command == "/help":
            return CommandResult(HELP_TEXT)
        if command == "/status":
            return CommandResult(self.format_readiness())
        if command == "/memory":
            fallback = f"memory: {self._readiness().get('memory', 'unknown')}"
            return CommandResult(self._runtime_call(("memory_status",), fallback=fallback))
        if command == "/index":
            fallback = f"index: {self._readiness().get('index', 'unknown')}"
            return CommandResult(self._runtime_call(("index_status",), fallback=fallback))
        if command == "/verify":
            if hasattr(self.runtime, "verify"):
                return CommandResult(self._runtime_call(("verify",), arg))
            prompt = f"/verify {arg}" if arg else "/verify"
            return CommandResult(self._runtime_call(("run_once",), prompt))
        if command == "/run":
            if not arg:
                return CommandResult("run: missing command")
            return CommandResult(self._runtime_call(("run_shell", "run"), arg))
        if command == "/read":
            if not arg:
                return CommandResult("read: missing path")
            return CommandResult(self._runtime_call(("read_file", "read"), arg))
        if command == "/write":
            if not arg:
                return CommandResult("write: missing path and text")
            return CommandResult(self._runtime_call(("write_file", "write"), arg))
        if command == "/apply":
            if not arg:
                return CommandResult("apply: missing patch text")
            patch_text = arg.replace("\\n", "\n")
            return CommandResult(self._runtime_call(("apply_patch",), patch_text))
        return CommandResult(f"unknown command: {command}\n{HELP_TEXT}")

    def format_readiness(self) -> str:
        readiness = self._readiness()
        lines = ["David startup readiness"]
        for name in ("model validation", "index", "memory"):
            lines.append(f"- {name}: {readiness.get(name, 'unknown')}")
        for name, value in readiness.items():
            if name not in {"model validation", "index", "memory"}:
                lines.append(f"- {name}: {value}")
        return "\n".join(lines)

    def _write_startup(self) -> None:
        title = self._style("David terminal agent", "bold")
        self._write_block(title)
        self._write_block(self.format_readiness())

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
        return str(value)

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
