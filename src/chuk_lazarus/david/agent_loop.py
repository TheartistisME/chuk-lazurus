"""Deterministic multi-step coding-agent loop core for David."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
import re
import subprocess
from typing import Any, Callable

from .tools import LocalTools, PathSafetyError


ActionPayload = Mapping[str, Any] | str | None
ModelStep = Callable[["AgentLoopState"], ActionPayload]


@dataclass(frozen=True)
class AgentAction:
    """A conservative tool request parsed from model text or direct input."""

    action: str
    path: str | None = None
    content: str | None = None
    command: tuple[str, ...] = ()
    cwd: str = "."
    timeout: int = 30
    passed: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["command"] = list(self.command)
        return data


@dataclass(frozen=True)
class AgentStepTrace:
    step: int
    action: str
    ok: bool
    observation: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "ok": self.ok,
            "observation": dict(self.observation),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class AgentLoopState:
    workspace_root: str
    step: int
    max_steps: int
    trace: tuple[AgentStepTrace, ...] = ()

    @property
    def last_observation(self) -> Mapping[str, Any]:
        if not self.trace:
            return {}
        return self.trace[-1].observation


@dataclass(frozen=True)
class AgentLoopResult:
    status: str
    steps: int
    trace: tuple[AgentStepTrace, ...]
    verified: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "verified" and self.verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steps": self.steps,
            "verified": self.verified,
            "reason": self.reason,
            "trace": [step.to_dict() for step in self.trace],
        }


class AgentLoopError(ValueError):
    pass


JSON_BLOCK_RE = re.compile(r"```(?:json|tool|action)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
ALLOWED_ACTIONS = {"plan", "read", "write", "run", "shell", "verify", "done", "none", "no_action"}


def run_agent_loop(
    model_step: ModelStep | Sequence[ActionPayload],
    tools: LocalTools,
    *,
    max_steps: int = 8,
) -> AgentLoopResult:
    """Run a bounded action loop using model text or direct tool requests."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    trace: list[AgentStepTrace] = []
    provider = _step_provider(model_step)
    for index in range(1, max_steps + 1):
        state = AgentLoopState(
            workspace_root=str(tools.workspace_root),
            step=index,
            max_steps=max_steps,
            trace=tuple(trace),
        )
        raw = provider(state)
        try:
            action = parse_agent_action(raw)
        except AgentLoopError as exc:
            trace.append(_trace(index, "refuse", False, {"error": str(exc)}, {"raw": raw}))
            return AgentLoopResult("refused", index, tuple(trace), reason=str(exc))

        if action is None or action.action in {"done", "none", "no_action"}:
            trace.append(_trace(index, "no_action", True, {"reason": "no action requested"}, {"raw": raw}))
            return AgentLoopResult("no_action", index, tuple(trace), reason="no action requested")

        step_trace = execute_agent_action(action, tools, step=index, raw=raw)
        trace.append(step_trace)

        if not step_trace.ok and step_trace.action == "refuse":
            return AgentLoopResult("refused", index, tuple(trace), reason=str(step_trace.observation.get("error", "")))
        if step_trace.action == "verify":
            passed = bool(step_trace.observation.get("passed"))
            if passed:
                return AgentLoopResult("verified", index, tuple(trace), verified=True, reason="verify passed")
            return AgentLoopResult("verify_failed", index, tuple(trace), verified=False, reason="verify failed")

    return AgentLoopResult("max_steps", max_steps, tuple(trace), reason="max steps reached")


def parse_agent_action(raw: ActionPayload) -> AgentAction | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return _action_from_mapping(raw)
    if not isinstance(raw, str):
        raise AgentLoopError(f"unsupported action payload: {type(raw).__name__}")

    text = raw.strip()
    if not text:
        return None
    parsed = _parse_json_object(text)
    if parsed is None:
        match = JSON_BLOCK_RE.search(text)
        if match:
            parsed = _parse_json_object(match.group(1))
    if parsed is None:
        return None
    return _action_from_mapping(parsed)


def execute_agent_action(
    action: AgentAction,
    tools: LocalTools,
    *,
    step: int = 1,
    raw: ActionPayload = None,
) -> AgentStepTrace:
    try:
        if action.action == "plan":
            return _trace(step, "plan", True, {"reason": action.reason}, {"raw": raw, "action": action.to_dict()})
        if action.action == "read":
            if not action.path:
                raise AgentLoopError("read requires path")
            content = tools.read(action.path)
            return _trace(
                step,
                "read",
                True,
                {"path": action.path, "content": content},
                {"raw": raw, "action": action.to_dict()},
            )
        if action.action == "write":
            if not action.path:
                raise AgentLoopError("write requires path")
            if action.content is None:
                raise AgentLoopError("write requires content")
            target = tools.write(action.path, action.content)
            return _trace(
                step,
                "write",
                True,
                {"path": action.path, "bytes": len(action.content.encode("utf-8")), "resolved": str(target)},
                {"raw": raw, "action": action.to_dict()},
            )
        if action.action in {"run", "shell"}:
            if not action.command:
                raise AgentLoopError(f"{action.action} requires command")
            observation = tools.run(action.command, cwd=action.cwd, timeout=action.timeout)
            return _trace(
                step,
                "run",
                int(observation["returncode"]) == 0,
                observation,
                {"raw": raw, "action": action.to_dict()},
            )
        if action.action == "verify":
            observation = _verify(action, tools)
            return _trace(step, "verify", bool(observation["passed"]), observation, {"raw": raw, "action": action.to_dict()})
    except (AgentLoopError, PathSafetyError, OSError, TypeError, subprocess.TimeoutExpired) as exc:
        return _trace(
            step,
            "refuse",
            False,
            {"error": str(exc), "requested_action": action.action},
            {"raw": raw, "action": action.to_dict()},
        )
    return _trace(
        step,
        "refuse",
        False,
        {"error": f"unknown action: {action.action}", "requested_action": action.action},
        {"raw": raw, "action": action.to_dict()},
    )


def _verify(action: AgentAction, tools: LocalTools) -> dict[str, Any]:
    if action.passed is not None:
        return {"passed": action.passed, "reason": action.reason}
    if not action.command:
        raise AgentLoopError("verify requires passed or command")
    run_result = tools.run(action.command, cwd=action.cwd, timeout=action.timeout)
    return {
        "passed": int(run_result["returncode"]) == 0,
        "command": run_result["command"],
        "cwd": run_result["cwd"],
        "returncode": run_result["returncode"],
        "stdout": run_result["stdout"],
        "stderr": run_result["stderr"],
    }


def _action_from_mapping(data: Mapping[str, Any]) -> AgentAction:
    action_name = str(data.get("action", data.get("tool", ""))).strip().lower()
    if not action_name:
        raise AgentLoopError("action is required")
    if action_name not in ALLOWED_ACTIONS:
        raise AgentLoopError(f"unknown action: {action_name}")

    command = data.get("command", ())
    if isinstance(command, str):
        raise AgentLoopError("command must be a list of arguments")
    if command is None:
        command = ()
    if not isinstance(command, Sequence):
        raise AgentLoopError("command must be a list of arguments")

    try:
        timeout = int(data.get("timeout", 30))
    except (TypeError, ValueError) as exc:
        raise AgentLoopError("timeout must be an integer") from exc
    if timeout < 1:
        raise AgentLoopError("timeout must be at least 1")

    passed = data.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise AgentLoopError("passed must be a boolean")

    content = data.get("content")
    if content is not None and not isinstance(content, str):
        raise AgentLoopError("content must be a string")

    path = data.get("path")
    if path is not None and not isinstance(path, str):
        raise AgentLoopError("path must be a string")

    cwd = data.get("cwd", ".")
    if not isinstance(cwd, str):
        raise AgentLoopError("cwd must be a string")

    return AgentAction(
        action=action_name,
        path=path,
        content=content,
        command=tuple(str(part) for part in command),
        cwd=cwd,
        timeout=timeout,
        passed=passed,
        reason=str(data.get("reason", "")),
    )


def _parse_json_object(text: str) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, Mapping):
        raise AgentLoopError("action JSON must be an object")
    return parsed


def _step_provider(model_step: ModelStep | Sequence[ActionPayload]) -> ModelStep:
    if callable(model_step):
        return model_step
    requests = list(model_step)

    def next_request(state: AgentLoopState) -> ActionPayload:
        offset = state.step - 1
        if offset >= len(requests):
            return None
        return requests[offset]

    return next_request


def _trace(
    step: int,
    action: str,
    ok: bool,
    observation: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> AgentStepTrace:
    return AgentStepTrace(
        step=step,
        action=action,
        ok=ok,
        observation=dict(observation),
        provenance=dict(provenance),
    )
