"""Plain ANSI terminal UI for the David agent surface."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .agent_loop import plan_natural_language_fallback_actions
from .resume import SessionSnapshot, format_resume_snapshot, load_session_snapshot


HELP_TEXT = """David commands:
  prompt text              Send a prompt; obvious safe file writes may agent-run
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
            if self._plain_prompt_can_agent_run(text):
                return CommandResult(self._agent_loop_command(text))
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
            return CommandResult(self._verify_command(arg))
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

    def _plain_prompt_can_agent_run(self, prompt: str) -> bool:
        return bool(plan_natural_language_fallback_actions(prompt))

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
        lines.extend(self._format_repair_summary(self._repair_summary_from_runtime_value(value)))
        return "\n".join(lines)

    def _agent_loop_observation_summary(self, observation: Any) -> str:
        if not isinstance(observation, dict):
            return ""
        patch_detail = self._agent_loop_patch_observation_summary(observation)
        if patch_detail:
            return patch_detail
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

    def _agent_loop_patch_observation_summary(self, observation: dict[str, Any]) -> str:
        if "mode" not in observation or (
            "failures" not in observation
            and "protected_paths" not in observation
            and "details" not in observation
        ):
            return ""
        failures = self._as_string_list(observation.get("failures"))
        protected_paths = self._as_string_list(observation.get("protected_paths"))
        details = observation.get("details")
        if not failures and not protected_paths and not details:
            return ""

        parts = [f"mode={observation.get('mode', 'unknown')}"]
        if protected_paths:
            parts.append(f"protected={self._join_limited(protected_paths)}")
        if failures:
            parts.append(f"failures={self._join_limited(failures)}")
        if isinstance(details, dict) and details:
            parts.append(f"details={self._compact_mapping(details)}")
        elif details:
            parts.append(f"details={self._compact_value(details)}")
        return " ".join(parts)

    def _as_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item)]
        return [str(value)]

    def _join_limited(self, values: list[str], *, limit: int = 2, max_chars: int = 160) -> str:
        clipped = [self._compact_value(value, max_chars=max_chars) for value in values[:limit]]
        if len(values) > limit:
            clipped.append(f"+{len(values) - limit} more")
        return "; ".join(clipped)

    def _compact_mapping(self, value: dict[Any, Any], *, limit: int = 3, max_chars: int = 160) -> str:
        items = list(value.items())
        parts = [
            f"{self._compact_value(key, max_chars=40)}={self._compact_value(item, max_chars=max_chars)}"
            for key, item in items[:limit]
        ]
        if len(items) > limit:
            parts.append(f"+{len(items) - limit} more")
        return ", ".join(parts)

    def _compact_value(self, value: Any, *, max_chars: int = 160) -> str:
        text = " ".join(str(value).split())
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    def format_readiness(self) -> str:
        readiness = self._readiness()
        lines = ["David startup readiness"]
        for name in ("model validation", "index", "memory"):
            lines.append(self._format_readiness_line(name, readiness.get(name, "unknown")))
        for name, value in readiness.items():
            if name not in {"model validation", "index", "memory"}:
                lines.append(self._format_readiness_line(name, value))
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

    def _verify_command(self, arg: str | None) -> str:
        verify = getattr(self.runtime, "verify", None)
        if callable(verify):
            return self._format_verify_result(verify(arg), command=arg)
        prompt = f"/verify {arg}" if arg else "/verify"
        return self._runtime_call(("run_once",), prompt)

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

    def _format_readiness_line(self, name: str, value: Any) -> str:
        text = self._compact_value(value, max_chars=320)
        return f"- {name}: {text} [state={self._readiness_state(text)}]"

    def _readiness_state(self, value: str) -> str:
        text = str(value).strip().lower()
        head = text.split(":", 1)[0].strip()
        if not text:
            return "unknown"
        if head in {"blocked", "rejected", "invalid", "failed", "error", "refused"}:
            return "blocked"
        if head in {"missing", "absent"}:
            return "missing"
        if head in {"degraded", "needs_review", "manual_reviewed"}:
            return "degraded"
        if text.startswith(("ready", "accepted", "loaded", "hot", "warm")):
            if any(marker in text for marker in ("degraded", "missing", "stale", "truncated", "unreadable", "warning")):
                return "degraded"
            return "ready"
        if text in {"unknown", "none"}:
            return "unknown"
        if any(marker in text for marker in ("blocked", "rejected", "invalid", "failed", "error", "refused")):
            return "blocked"
        if any(marker in text for marker in ("missing", "not found", "jit required")):
            return "missing"
        if any(
            marker in text
            for marker in (
                "degraded",
                "needs_review",
                "manual_reviewed",
                "manual review",
                "offline shell mode",
                "not loaded",
                "stale",
            )
        ):
            return "degraded"
        return "info"

    def _format_verify_result(self, value: Any, *, command: str | None) -> str:
        data = self._mapping_from_runtime_value(value)
        if data is not None:
            return self._format_verify_mapping(data, command=command)

        text = str(value or "").strip()
        if text.startswith("David verification"):
            return text

        mode = "command" if command else "candidate discovery"
        lines = ["David verification", f"mode: {mode}"]
        if command:
            lines.append(f"command: {command}")
        if text:
            lines.extend(("output:", text))
        else:
            lines.append("output: (none)")
        return "\n".join(lines)

    def _format_verify_mapping(self, data: Mapping[str, Any], *, command: str | None) -> str:
        command_result = self._verify_command_result(data)
        quality_gate = self._verify_quality_gate_check(data)
        mode = str(data.get("mode") or ("command" if command or command_result else "candidate discovery"))
        reason = data.get("reason") or data.get("run_reason")
        ok = self._verify_ok(data, command_result)

        lines = ["David verification", f"mode: {mode}"]
        display_command = command or self._display_command_value(
            command_result.get("command") if isinstance(command_result, Mapping) else data.get("command")
        )
        if display_command:
            lines.append(f"command: {display_command}")
        if ok is not None:
            lines.append(f"result: {'passed' if ok else 'failed'}")
        if reason:
            lines.append(f"reason: {self._compact_value(reason)}")

        discovered = self._verify_discovered_candidates(data, quality_gate)
        selected = self._verify_selected_commands(data, quality_gate)
        if discovered or selected or quality_gate is not None:
            lines.append("selected commands:")
            lines.extend(f"- {item}" for item in (selected or ["none"]))
            lines.append("discovered candidate gates:")
            lines.extend(self._format_verify_candidate(item) for item in discovered) if discovered else lines.append("- none")
            lines.append("commands run:")
            commands_run = quality_gate.get("commands_run", 0) if isinstance(quality_gate, Mapping) else 0
            lines.append(f"- {commands_run}" if commands_run else "- none")

        if isinstance(command_result, Mapping):
            lines.extend(self._format_command_result(command_result))

        metadata = self._verify_metadata_lines(data, quality_gate, command_result)
        if metadata:
            lines.append("metadata:")
            lines.extend(metadata)
        return "\n".join(lines)

    def _verify_command_result(self, data: Mapping[str, Any]) -> Mapping[str, Any] | None:
        command_result = data.get("command_result") or data.get("result")
        if isinstance(command_result, Mapping):
            return command_result
        if any(key in data for key in ("returncode", "stdout", "stderr")):
            return data
        return None

    def _verify_quality_gate_check(self, data: Mapping[str, Any]) -> Mapping[str, Any] | None:
        checks = data.get("checks")
        if isinstance(checks, Mapping):
            gate = checks.get("candidate_quality_gates")
            if isinstance(gate, Mapping):
                return gate
        gate = data.get("candidate_quality_gates") or data.get("quality_gates")
        return gate if isinstance(gate, Mapping) else None

    def _verify_ok(self, data: Mapping[str, Any], command_result: Mapping[str, Any] | None) -> bool | None:
        for key in ("ok", "passed", "success"):
            value = data.get(key)
            if isinstance(value, bool):
                return value
        if isinstance(command_result, Mapping):
            if isinstance(command_result.get("passed"), bool):
                return bool(command_result["passed"])
            if "returncode" in command_result:
                try:
                    return int(command_result["returncode"]) == 0
                except (TypeError, ValueError):
                    return None
        return None

    def _verify_discovered_candidates(
        self,
        data: Mapping[str, Any],
        quality_gate: Mapping[str, Any] | None,
    ) -> list[Any]:
        candidates = data.get("candidates") or data.get("discovered_candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes, bytearray)):
            return list(candidates)
        evidence = data.get("evidence")
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
            candidate_evidence = [
                item
                for item in evidence
                if isinstance(item, Mapping) and item.get("kind") == "quality_gate_candidate"
            ]
            if candidate_evidence:
                return candidate_evidence
        if isinstance(quality_gate, Mapping):
            commands = quality_gate.get("discovered_commands")
            if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes, bytearray)):
                return list(commands)
        return []

    def _verify_selected_commands(
        self,
        data: Mapping[str, Any],
        quality_gate: Mapping[str, Any] | None,
    ) -> list[str]:
        selected = data.get("selected_commands")
        if isinstance(quality_gate, Mapping):
            selected = selected or quality_gate.get("selected_commands")
        if isinstance(selected, Sequence) and not isinstance(selected, (str, bytes, bytearray)):
            return [self._display_command_value(item) for item in selected if self._display_command_value(item)]
        return []

    def _format_verify_candidate(self, item: Any) -> str:
        if isinstance(item, Mapping):
            command = self._display_command_value(item.get("display_command") or item.get("command"))
            name = item.get("name")
            reason = item.get("reason")
            label = f"{name}: {command}" if name and command else command or self._compact_value(item)
            if reason:
                return f"- {label} ({self._compact_value(reason)})"
            return f"- {label}"
        return f"- {self._display_command_value(item) or self._compact_value(item)}"

    def _format_command_result(self, command_result: Mapping[str, Any]) -> list[str]:
        lines: list[str] = []
        if "returncode" in command_result:
            lines.append(f"return code: {command_result['returncode']}")
        stdout = str(command_result.get("stdout") or "").rstrip()
        stderr = str(command_result.get("stderr") or "").rstrip()
        if stdout:
            lines.extend(("stdout:", stdout))
        if stderr:
            lines.extend(("stderr:", stderr))
        return lines

    def _verify_metadata_lines(
        self,
        data: Mapping[str, Any],
        quality_gate: Mapping[str, Any] | None,
        command_result: Mapping[str, Any] | None,
    ) -> list[str]:
        lines: list[str] = []
        for key in ("capability",):
            if data.get(key):
                lines.append(f"- {key}: {data[key]}")
        if isinstance(quality_gate, Mapping):
            for key in ("candidate_count", "selected_count"):
                if key in quality_gate:
                    lines.append(f"- {key}: {quality_gate[key]}")
        if isinstance(command_result, Mapping) and "command" in command_result:
            command = self._display_command_value(command_result["command"])
            if command:
                lines.append(f"- executed: {command}")
        return lines

    def _display_command_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            return " ".join(str(item) for item in value)
        return str(value)

    def _repair_summary_from_runtime_value(self, value: Any) -> Mapping[str, Any] | None:
        for attr in ("repair_summary", "failure_recovery", "recovery_summary"):
            summary = getattr(value, attr, None)
            if isinstance(summary, Mapping):
                return summary
        data = self._mapping_from_runtime_value(value)
        if data is None:
            return None
        for key in ("repair_summary", "failure_recovery", "recovery_summary"):
            summary = data.get(key)
            if isinstance(summary, Mapping):
                return summary
        writeback = data.get("writeback")
        if isinstance(writeback, Mapping):
            metadata = writeback.get("metadata")
            if isinstance(metadata, Mapping):
                for key in ("repair_summary", "failure_recovery", "recovery_summary"):
                    summary = metadata.get(key)
                    if isinstance(summary, Mapping):
                        return summary
        return None

    def _format_repair_summary(self, summary: Mapping[str, Any] | None) -> list[str]:
        if summary is None:
            return []
        failure_count = self._nonnegative_int(summary.get("failure_count") or summary.get("failed_step_count"))
        repair_count = self._nonnegative_int(summary.get("repair_attempt_count") or summary.get("repair_count"))
        retry_count = self._nonnegative_int(summary.get("retry_count") or summary.get("retries"))
        verification = summary.get("verification") if isinstance(summary.get("verification"), Mapping) else {}
        verification_outcome = verification.get("outcome") or summary.get("verification_outcome")
        recovery_outcome = str(summary.get("recovery_outcome") or "unknown")
        interesting = any((failure_count, repair_count, retry_count)) or recovery_outcome not in {
            "unknown",
            "no_failure",
            "none",
        }
        if not interesting:
            return []

        lines = [
            "repair summary: "
            f"failures={failure_count} repairs={repair_count} retries={retry_count} "
            f"recovery={recovery_outcome} verification={verification_outcome or 'unknown'}"
        ]
        failed_steps = self._mapping_list(summary.get("failed_steps") or summary.get("recovered_failures"))
        repair_steps = self._mapping_list(summary.get("repair_steps") or summary.get("repair_attempts"))
        if failed_steps:
            lines.append(f"- failed: {self._format_step_references(failed_steps)}")
        if repair_steps:
            lines.append(f"- repair: {self._format_step_references(repair_steps)}")
        return lines

    def _format_step_references(self, steps: list[Mapping[str, Any]], *, limit: int = 2) -> str:
        formatted = []
        for step in steps[:limit]:
            bits = [f"step={step.get('step', '?')}", f"action={step.get('action', 'unknown')}"]
            for key in ("error", "reason", "returncode"):
                if step.get(key) not in (None, ""):
                    bits.append(f"{key}={self._compact_value(step[key], max_chars=80)}")
                    break
            formatted.append(" ".join(bits))
        if len(steps) > limit:
            formatted.append(f"+{len(steps) - limit} more")
        return "; ".join(formatted)

    def _mapping_list(self, value: Any) -> list[Mapping[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        return [item for item in value if isinstance(item, Mapping)]

    def _mapping_from_runtime_value(self, value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        for name in ("to_json", "to_dict"):
            method = getattr(value, name, None)
            if callable(method):
                mapped = method()
                if isinstance(mapped, Mapping):
                    return mapped
        return None

    def _nonnegative_int(self, value: Any) -> int:
        if isinstance(value, bool):
            return 0
        if isinstance(value, int) and value >= 0:
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0

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
