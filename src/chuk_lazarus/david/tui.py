"""Stream-friendly terminal UI for the David coding agent.

The TUI is deliberately runtime-agnostic. It accepts an injected runtime object
and talks to it through duck-typed methods/properties; model backends stay on
the other side of the runtime boundary.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import shlex
import sys
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, TextIO

from .theme import Theme


SUCCESS = 0
KEYBOARD_INTERRUPT = 130
RUNTIME_ERROR = 1

PROMPT_METHODS = ("respond", "ask", "chat", "complete", "generate", "run_prompt", "run_once", "run")
RUN_METHODS = ("run_command", "run_shell", "execute_shell", "shell")
SEARCH_METHODS = ("search", "find", "lookup")
READ_METHODS = ("read", "read_file", "open_file")
VERIFY_METHODS = ("verify", "run_verification", "verification_report")
STATUS_METHODS = ("status_summary", "status", "get_status", "describe_status")
BOOT_METHODS = ("boot_summary", "describe_boot", "summary")
ROADMAP_METHODS = ("roadmap_summary", "roadmap", "describe_roadmap")
TOOL_RUNNER_ATTRS = ("tool_runner", "tools_runner", "local_tools", "coding_tools")
TOOL_LIST_METHODS = ("available_tools", "available_tool_names", "tools", "list_tools")


HELP_TEXT = """Commands:
  /help                 Show this help.
  /status               Show runtime and harness readiness.
  /verify [command]     Run configured verification, or one verification command.
  /run <command>        Run a workspace command through the runtime/tool runner.
  /search <pattern> [path]
                        Search the workspace. Quote patterns with spaces.
  /read <path> [start] [limit]
                        Read a file through the runtime/tool runner.
  /tools                List available runtime/local tools.
  /roadmap              Show David's productization spine.
  /exit                 Leave the TUI.

Type a normal prompt to send it to the runtime.
"""


ROADMAP_TEXT = """David terminal-agent roadmap:
  1. Boot: scan and validate model config before auto-load.
  2. Readiness: check user/task memory and workspace indexes.
  3. Detect: choose methodology by task shape, not benchmark name.
  4. Route: select files, symbols, memories, and evidence.
  5. Materialize: use boundary/residual/KV context only when compatible.
  6. Decode: constrain generation for paths, symbols, formats, and priors.
  7. Verify: prove the result, then write durable user/task memory.
"""


@dataclass
class _ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class _OptionsRuntimeAdapter:
    """Fallback runtime used when tests or older CLI code pass options directly."""

    def __init__(self, options: Any) -> None:
        workspace = getattr(options, "workspace", getattr(options, "workspace_path", "."))
        model = getattr(options, "model", getattr(options, "model_path", None))
        self.options = options
        self.workspace_path = Path(workspace).expanduser().resolve()
        self.model_identity = model or "offline/no-model"
        self.validation_report_path = getattr(
            options,
            "validation_report",
            getattr(options, "validation_report_path", None),
        )
        self.allow_unvalidated = bool(getattr(options, "allow_unvalidated", False))
        self.allow_shell = bool(getattr(options, "allow_shell", True))
        self.tool_error: str | None = None
        self.tool_runner = self._build_tool_runner()

    def initialize(self) -> Any | None:
        return None

    boot = initialize
    start = initialize

    def boot_summary(self) -> dict[str, Any]:
        return {
            "mode": "offline_shell" if self.model_identity == "offline/no-model" else "runtime_options",
            "workspace_path": str(self.workspace_path),
            "model_identity": self.model_identity,
            "validation_report_path": str(self.validation_report_path)
            if self.validation_report_path
            else None,
            "allow_unvalidated": self.allow_unvalidated,
            "local_tools_ready": self.tool_runner is not None,
            "tool_error": self.tool_error,
        }

    def status(self) -> dict[str, Any]:
        return self.boot_summary()

    def verify(self, commands: list[str] | None = None) -> dict[str, Any] | list[Any]:
        if commands:
            return [self.execute_tool("shell", {"command": command}) for command in commands]
        return {
            "workspace_exists": self.workspace_path.exists(),
            "workspace_is_dir": self.workspace_path.is_dir(),
            "local_tools_ready": self.tool_runner is not None,
            "model_backend_loaded": False,
            "notes": "offline TUI adapter; inject a model runtime for agent responses",
        }

    def run_shell(self, command: str) -> Any:
        return self.execute_tool("shell", {"command": command})

    def execute_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        if self.tool_runner is None:
            return {"ok": False, "error": self.tool_error or "local tools are unavailable"}
        return self.tool_runner.execute(_ToolCall(name=name, arguments=dict(arguments or {})))

    def available_tools(self) -> list[str]:
        if self.tool_runner is None:
            return []
        available = getattr(self.tool_runner, "available_tool_names", None)
        return list(available()) if callable(available) else []

    def roadmap_summary(self) -> dict[str, Any]:
        return {
            "product": "David terminal coding agent",
            "mode": "offline shell adapter",
            "runtime_boundary": "inject a model runtime for natural-language turns",
        }

    def respond(self, prompt: str) -> str:
        del prompt
        return (
            "offline shell mode is ready. Use /read, /search, /run, /tools, "
            "or inject a model runtime for natural-language agent turns."
        )

    def _build_tool_runner(self) -> Any | None:
        try:
            from chuk_lazarus.repl_agent_tools import LocalCodingToolRunner
        except Exception as exc:  # noqa: BLE001 - user-facing status should explain.
            self.tool_error = f"LocalCodingToolRunner unavailable: {exc}"
            return None

        try:
            return LocalCodingToolRunner(
                workspace_root=self.workspace_path,
                trace_root=self.workspace_path / ".chuk_lazarus" / "david" / "tool_traces",
                allow_shell=self.allow_shell,
                allow_custom_tools=True,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing status should explain.
            self.tool_error = f"LocalCodingToolRunner failed to start: {exc}"
            return None


class DavidTUI:
    """A small injectable REPL for the David runtime."""

    def __init__(
        self,
        runtime: Any,
        *,
        session: Any | None = None,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        error_stream: TextIO | None = None,
        use_color: bool | None = None,
        show_banner: bool = True,
    ) -> None:
        self.runtime = runtime
        self.session = session
        self.input = input_stream or sys.stdin
        self.output = output_stream or sys.stdout
        self.error = error_stream or self.output
        self.theme = Theme.for_stream(self.output, use_color=use_color)
        self.show_banner = bool(show_banner)
        self.turn_index = 0

    def run(self, once_prompt: str | None = None) -> int:
        self._print_startup()
        if once_prompt is not None:
            self._handle_line(once_prompt)
            return SUCCESS

        while True:
            line = self._readline(self.theme.prompt())
            if line is None:
                self._print()
                return SUCCESS
            if not self._handle_line(line):
                return SUCCESS

    def once(self, prompt: str) -> int:
        return self.run(once_prompt=prompt)

    def _print_startup(self) -> None:
        if self.show_banner:
            self._print(
                self.theme.banner(
                    "DAVID TERMINAL AGENT",
                    "open model harness | memory runtime | local coding tools",
                )
            )
        self._print(self.theme.section("boot"))
        self._print(self._boot_summary())
        self._print(self.theme.badge("ready", ok=True) + " type /help for commands")

    def _handle_line(self, raw_line: str) -> bool:
        text = raw_line.strip()
        if not text:
            return True
        if text.lower() in {"exit", "quit"}:
            text = "/exit"
        if text.startswith("/"):
            return self._handle_command(text)
        self._handle_prompt(text)
        return True

    def _handle_command(self, text: str) -> bool:
        command, _, raw_args = text.partition(" ")
        command = command.lower()
        args = raw_args.strip()

        if command in {"/exit", "/quit"}:
            self._print("session closed")
            return False
        if command == "/help":
            self._print(HELP_TEXT.rstrip())
            return True
        if command == "/status":
            self._print(self.theme.section("status"))
            self._print(self._status())
            return True
        if command == "/verify":
            self._print(self.theme.section("verify"))
            self._print(self._verify(args))
            return True
        if command == "/run":
            self._print(self.theme.section("run"))
            self._print(self._run_command(args))
            return True
        if command == "/search":
            self._print(self.theme.section("search"))
            self._print(self._search(args))
            return True
        if command == "/read":
            self._print(self.theme.section("read"))
            self._print(self._read(args))
            return True
        if command == "/tools":
            self._print(self.theme.section("tools"))
            self._print(self._tools())
            return True
        if command == "/roadmap":
            self._print(self.theme.section("roadmap"))
            self._print(self._roadmap())
            return True

        self._print_error(f"unknown command: {command}")
        self._print("type /help for commands")
        return True

    def _handle_prompt(self, prompt: str) -> None:
        found, result = self._call_runtime(
            PROMPT_METHODS,
            [
                ((prompt,), {}),
                ((), {"prompt": prompt}),
                ((), {"message": prompt}),
            ],
        )
        if not found:
            self._print_error("runtime has no prompt method")
            return
        self.turn_index += 1
        self._print(self._render(result))

    def _status(self) -> str:
        found, result = self._call_runtime(STATUS_METHODS, [((), {})], include_properties=True)
        if found and result not in (None, ""):
            return self._render(result)
        if self.session is not None:
            return self._render(self.session)
        return self._boot_summary()

    def _verify(self, args: str) -> str:
        if not args:
            found, result = self._call_runtime(VERIFY_METHODS, [((), {})])
            return self._render(result) if found else "no verify hook exposed by runtime"

        for name in VERIFY_METHODS:
            method = self._safe_getattr(self.runtime, name)
            if not callable(method):
                continue
            variants = self._verify_variants(method, args)
            found, result = self._call_callable(method, variants)
            if found:
                return self._render(result)
        result = self._execute_tool("shell", {"command": args, "purpose": "verification"})
        return self._render(result) if result is not None else "no verify hook exposed by runtime"

    def _run_command(self, args: str) -> str:
        if not args:
            return "usage: /run <command>"
        found, result = self._call_runtime(
            RUN_METHODS,
            [((args,), {}), ((), {"command": args})],
        )
        if found:
            return self._render(result)
        result = self._execute_tool("shell", {"command": args})
        return self._render(result) if result is not None else "no command runner exposed by runtime"

    def _search(self, args: str) -> str:
        tokens = self._split_args(args)
        if not tokens:
            return "usage: /search <pattern> [path]"
        pattern = tokens[0]
        path = tokens[1] if len(tokens) > 1 else "."
        found, result = self._call_runtime(
            SEARCH_METHODS,
            [
                ((pattern,), {"path": path}),
                ((pattern, path), {}),
                ((), {"pattern": pattern, "path": path}),
            ],
        )
        if found:
            return self._render(result)
        result = self._execute_tool("search", {"pattern": pattern, "path": path, "max_results": 100})
        return self._render(result) if result is not None else "no search hook exposed by runtime"

    def _read(self, args: str) -> str:
        tokens = self._split_args(args)
        if not tokens:
            return "usage: /read <path> [start] [limit]"
        path = tokens[0]
        start_line = self._parse_int(tokens[1], default=1) if len(tokens) > 1 else 1
        limit = self._parse_int(tokens[2], default=200) if len(tokens) > 2 else 200
        found, result = self._call_runtime(
            READ_METHODS,
            [
                ((path,), {"start_line": start_line, "limit": limit}),
                ((path, start_line, limit), {}),
                ((), {"path": path, "start_line": start_line, "limit": limit}),
            ],
        )
        if found:
            return self._render(result)
        result = self._execute_tool(
            "read_file",
            {"path": path, "start_line": start_line, "limit": limit},
        )
        return self._render(result) if result is not None else "no read hook exposed by runtime"

    def _tools(self) -> str:
        lines: list[str] = []
        found, result = self._call_runtime(TOOL_LIST_METHODS, [((), {})], include_properties=True)
        if found and result not in (None, ""):
            lines.append(self._render(result))

        runner = self._tool_runner()
        if runner is not None:
            available = self._call_runner(runner, "available_tool_names")
            if available is not None:
                lines.append("local tools: " + ", ".join(str(item) for item in available))
            custom = self._call_runner(runner, "describe_custom_tools")
            if custom:
                lines.append(str(custom))
        return "\n\n".join(lines) if lines else "no tools exposed by runtime"

    def _roadmap(self) -> str:
        found, result = self._call_runtime(ROADMAP_METHODS, [((), {})], include_properties=True)
        return self._render(result) if found and result not in (None, "") else ROADMAP_TEXT.rstrip()

    def _boot_summary(self) -> str:
        found, result = self._call_runtime(BOOT_METHODS, [((), {})], include_properties=True)
        if found and result not in (None, ""):
            return self._render(result)
        if self.session is not None:
            return self._render(self.session)
        values = {
            "runtime": type(self.runtime).__name__,
            "workspace": self._first_attr("workspace_path", "workspace", "workspace_root"),
            "model": self._first_attr("model_identity", "model_path", "model"),
        }
        lines = [
            self.theme.key_value(key, value)
            for key, value in values.items()
            if value not in (None, "")
        ]
        return "\n".join(lines) if lines else "runtime: unavailable"

    def _execute_tool(self, name: str, arguments: Mapping[str, Any]) -> Any | None:
        execute_tool = self._safe_getattr(self.runtime, "execute_tool")
        if callable(execute_tool):
            found, result = self._call_callable(
                execute_tool,
                [
                    ((name, arguments), {"turn_index": self.turn_index}),
                    ((name, arguments), {}),
                    ((), {"name": name, "arguments": dict(arguments)}),
                ],
            )
            if found:
                return result

        runner = self._tool_runner()
        execute = self._safe_getattr(runner, "execute") if runner is not None else None
        if not callable(execute):
            return None
        call = _ToolCall(name=name, arguments=dict(arguments))
        found, result = self._call_callable(
            execute,
            [
                ((call,), {"turn_index": self.turn_index}),
                ((call,), {}),
            ],
        )
        return result if found else None

    def _tool_runner(self) -> Any | None:
        return self._first_attr(*TOOL_RUNNER_ATTRS)

    def _call_runtime(
        self,
        names: tuple[str, ...],
        variants: list[tuple[tuple[Any, ...], dict[str, Any]]],
        *,
        include_properties: bool = False,
    ) -> tuple[bool, Any]:
        for name in names:
            attr = self._safe_getattr(self.runtime, name)
            if attr is None:
                continue
            if callable(attr):
                found, result = self._call_callable(attr, variants)
                if found:
                    return True, result
            elif include_properties:
                return True, attr
        return False, None

    def _call_callable(
        self,
        method: Any,
        variants: list[tuple[tuple[Any, ...], dict[str, Any]]],
    ) -> tuple[bool, Any]:
        last_type_error: TypeError | None = None
        for args, kwargs in variants:
            try:
                return True, self._settle(method(*args, **kwargs))
            except TypeError as exc:
                last_type_error = exc
        if last_type_error is not None:
            return True, f"{getattr(method, '__name__', 'callable')} could not be called: {last_type_error}"
        return False, None

    def _verify_variants(
        self, method: Any, target: str
    ) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "target" in parameters:
            return [((), {"target": target}), ((target,), {})]
        if "command" in parameters:
            return [((), {"command": target}), ((target,), {})]
        if "commands" in parameters:
            return [((), {"commands": [target]}), (([target],), {})]
        return [((target,), {}), ((), {"target": target}), (([target],), {})]

    def _settle(self, result: Any) -> Any:
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            raise RuntimeError("async runtime method returned while an event loop is already running")
        return result

    def _render(self, value: Any) -> str:
        if value is None:
            return "(no output)"
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if self._looks_like_tool_result(value):
            return self._render_tool_result(value)
        if hasattr(value, "answer") or hasattr(value, "verification_summary"):
            return self._render_once_result(value)

        converted = self._to_plain(value)
        if isinstance(converted, str):
            return converted
        if isinstance(converted, (Mapping, list, tuple)):
            try:
                return json.dumps(converted, indent=2, sort_keys=True, default=str)
            except TypeError:
                return pformat(converted, width=100, sort_dicts=True)
        return str(converted)

    def _render_tool_result(self, result: Any) -> str:
        ok = bool(self._safe_getattr(result, "ok"))
        label = self.theme.badge("ok" if ok else "failed", ok=ok)
        name = self._safe_getattr(result, "name") or "tool"
        output = str(self._safe_getattr(result, "output") or "").rstrip()
        error = str(self._safe_getattr(result, "error") or "").rstrip()
        lines = [f"{label} {name}"]
        if output:
            lines.append(output)
        if error:
            lines.append(self.theme.key_value("error", error))
        if len(lines) == 1:
            lines.append("(no output)")
        return "\n".join(lines)

    def _render_once_result(self, result: Any) -> str:
        lines: list[str] = []
        answer = self._safe_getattr(result, "answer")
        verification = self._safe_getattr(result, "verification_summary")
        exit_code = self._safe_getattr(result, "exit_code")
        if answer:
            lines.append(str(answer))
        if verification:
            lines.append(self._render(verification))
        if exit_code not in (None, 0):
            lines.append(self.theme.key_value("exit_code", exit_code))
        return "\n".join(lines) if lines else self._render(self._to_plain(result))

    def _looks_like_tool_result(self, value: Any) -> bool:
        return all(hasattr(value, name) for name in ("ok", "output", "error"))

    def _to_plain(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        to_dict = self._safe_getattr(value, "to_dict")
        if callable(to_dict):
            try:
                return to_dict()
            except TypeError:
                pass
        return value

    def _first_attr(self, *names: str) -> Any | None:
        for name in names:
            value = self._safe_getattr(self.runtime, name)
            if value is not None:
                return value
        return None

    def _safe_getattr(self, obj: Any, name: str) -> Any | None:
        if obj is None:
            return None
        try:
            return getattr(obj, name)
        except Exception:
            return None

    def _call_runner(self, runner: Any, method_name: str) -> Any | None:
        method = self._safe_getattr(runner, method_name)
        if not callable(method):
            return None
        try:
            return self._settle(method())
        except Exception:
            return None

    def _split_args(self, args: str) -> list[str]:
        try:
            return shlex.split(args)
        except ValueError as exc:
            self._print_error(f"argument parse error: {exc}")
            return []

    def _parse_int(self, raw: str, *, default: int) -> int:
        try:
            return int(raw)
        except ValueError:
            return default

    def _readline(self, prompt: str) -> str | None:
        self.output.write(prompt)
        self.output.flush()
        line = self.input.readline()
        if line == "":
            return None
        return line.rstrip("\n")

    def _print(self, text: object = "", *, stream: TextIO | None = None) -> None:
        target = stream or self.output
        target.write(str(text))
        target.write("\n")
        target.flush()

    def _print_error(self, text: object) -> None:
        self._print(self.theme.style(text, "yellow"), stream=self.error)


def run_tui(
    runtime_or_options: Any = None,
    *,
    runtime: Any = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
    once_prompt: str | None = None,
    use_color: bool | None = None,
) -> int:
    """Run David's terminal UI for an injected runtime or options object."""

    candidate = runtime if runtime is not None else runtime_or_options
    if once_prompt is None and bool(getattr(candidate, "once", False)):
        once_prompt = getattr(candidate, "prompt", None)
    if use_color is None and bool(getattr(candidate, "no_color", False)):
        use_color = False

    runtime_obj = _coerce_runtime(candidate)
    session = _initialize_runtime(runtime_obj)
    return DavidTUI(
        runtime_obj,
        session=session,
        input_stream=input_stream or stdin,
        output_stream=output_stream or stdout,
        error_stream=error_stream or stderr,
        use_color=use_color,
    ).run(once_prompt=once_prompt)


def run_repl(
    runtime: Any,
    *,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
    once_prompt: str | None = None,
    use_color: bool | None = None,
) -> int:
    return run_tui(
        runtime,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
        once_prompt=once_prompt,
        use_color=use_color,
    )


def run_once(
    runtime: Any,
    prompt: str,
    *,
    output_stream: TextIO | None = None,
    error_stream: TextIO | None = None,
    use_color: bool | None = None,
) -> int:
    return run_tui(
        runtime,
        output_stream=output_stream,
        error_stream=error_stream,
        once_prompt=prompt,
        use_color=use_color,
    )


def _initialize_runtime(runtime: Any) -> Any | None:
    for name in ("initialize", "boot", "start"):
        method = getattr(runtime, name, None)
        if not callable(method):
            continue
        result = method()
        if result is not None and result is not runtime:
            return result
        return getattr(runtime, "harness_session", getattr(runtime, "session", None))
    return getattr(runtime, "harness_session", getattr(runtime, "session", None))


def _coerce_runtime(candidate: Any) -> Any:
    if _looks_like_runtime(candidate):
        return candidate
    if _looks_like_cli_options(candidate):
        return _OptionsRuntimeAdapter(candidate)
    return candidate


def _looks_like_runtime(candidate: Any) -> bool:
    for name in (
        PROMPT_METHODS
        + STATUS_METHODS
        + READ_METHODS
        + SEARCH_METHODS
        + RUN_METHODS
        + ("execute_tool", "initialize", "boot", "start")
    ):
        if callable(getattr(candidate, name, None)):
            return True
    return any(getattr(candidate, name, None) is not None for name in TOOL_RUNNER_ATTRS)


def _looks_like_cli_options(candidate: Any) -> bool:
    has_workspace = hasattr(candidate, "workspace") or hasattr(candidate, "workspace_path")
    return has_workspace and hasattr(candidate, "no_color") and hasattr(candidate, "allow_unvalidated")


__all__ = ["DavidTUI", "run_once", "run_repl", "run_tui"]
