"""Deterministic multi-step coding-agent loop core for David."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
from pathlib import PurePosixPath, PureWindowsPath
import re
import shlex
import subprocess
import sys
from typing import Any, Callable

from .patching import apply_patch_candidate
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
    objective: str = ""
    mode: str = "explicit"

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
ACTION_JSON_SENTINEL = "David action JSON:"
ALLOWED_ACTIONS = {"plan", "read", "patch", "write", "run", "shell", "verify", "done", "none", "no_action"}
WRITE_CONTENT_RE = re.compile(r"\b(?:that\s+says|with\s+content|containing|contents?\s*:)\b", re.IGNORECASE)
OPTIONAL_RUN_RE = re.compile(r"\b(?:then|and)\s+run\s+(.+)$", re.IGNORECASE | re.DOTALL)
QUOTED_OR_TOKEN_RE = r"`[^`]+`|\"[^\"]+\"|'[^']+'|[^\s]+"
PROTECTED_FALLBACK_PATHS = {
    "scripts/run_swebench_pro_parity.py",
    "scripts/run_locobench_benchmark.py",
    "scripts/run_ruler_benchmark.py",
    "scripts/run_mrcr_benchmark.py",
    "scripts/interactive_memory_chat.py",
    "scripts/benchmark_jit_indexing.py",
    "David/central router.py",
    "David/decoding_prior_store.py",
}


def run_agent_loop(
    model_step: ModelStep | Sequence[ActionPayload],
    tools: LocalTools,
    *,
    max_steps: int = 8,
    objective: str = "",
    mode: str = "explicit",
) -> AgentLoopResult:
    """Run a bounded action loop using model text or direct tool requests."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    trace: list[AgentStepTrace] = []
    provider = _step_provider(model_step)
    fallback_requests: tuple[ActionPayload, ...] | None = None
    fallback_reason = ""
    for index in range(1, max_steps + 1):
        state = AgentLoopState(
            workspace_root=str(tools.workspace_root),
            step=index,
            max_steps=max_steps,
            trace=tuple(trace),
            objective=objective,
            mode=mode,
        )
        if fallback_requests is not None:
            raw = _fallback_payload_for_step(fallback_requests, index=index, reason=fallback_reason)
            if raw is None:
                trace.append(
                    _trace(
                        index,
                        "no_action",
                        True,
                        {"reason": "natural-language fallback exhausted"},
                        {"provenance": "david.agent_loop.natural_language_fallback"},
                    )
                )
                return AgentLoopResult("no_action", index, tuple(trace), reason="natural-language fallback exhausted")
        else:
            try:
                raw = provider(state)
            except Exception as exc:
                fallback_requests = _fallback_plan_for_state(state, mode=mode)
                if fallback_requests:
                    fallback_reason = f"provider failed: {type(exc).__name__}: {exc}"
                    raw = _fallback_payload_for_step(fallback_requests, index=index, reason=fallback_reason)
                else:
                    trace.append(
                        _trace(
                            index,
                            "refuse",
                            False,
                            {"error": f"model step failed: {type(exc).__name__}: {exc}"},
                            {"provenance": "david.agent_loop.provider"},
                        )
                    )
                    return AgentLoopResult("refused", index, tuple(trace), reason=str(trace[-1].observation["error"]))
        try:
            action = parse_agent_action(raw)
        except AgentLoopError as exc:
            fallback_requests = _fallback_plan_for_state(state, mode=mode)
            if fallback_requests:
                fallback_reason = f"unparseable model action: {exc}"
                raw = _fallback_payload_for_step(fallback_requests, index=index, reason=fallback_reason)
                action = parse_agent_action(raw)
            else:
                trace.append(_trace(index, "refuse", False, {"error": str(exc)}, {"raw": raw}))
                return AgentLoopResult("refused", index, tuple(trace), reason=str(exc))

        if action is None or action.action in {"none", "no_action"}:
            fallback_requests = _fallback_plan_for_state(state, mode=mode)
            if fallback_requests:
                fallback_reason = "model returned no usable action"
                raw = _fallback_payload_for_step(fallback_requests, index=index, reason=fallback_reason)
                action = parse_agent_action(raw)
            else:
                trace.append(_trace(index, "no_action", True, {"reason": "no action requested"}, {"raw": raw}))
                return AgentLoopResult("no_action", index, tuple(trace), reason="no action requested")
        if action is None or action.action in {"none", "no_action"}:
            trace.append(_trace(index, "no_action", True, {"reason": "no action requested"}, {"raw": raw}))
            return AgentLoopResult("no_action", index, tuple(trace), reason="no action requested")
        if action.action == "done":
            reason = action.reason or "done"
            trace.append(_trace(index, "done", True, {"reason": reason}, {"raw": raw, "action": action.to_dict()}))
            return AgentLoopResult("done", index, tuple(trace), reason=reason)

        step_trace = execute_agent_action(action, tools, step=index, raw=raw)
        trace.append(step_trace)

        if not step_trace.ok and step_trace.action == "refuse":
            return AgentLoopResult("refused", index, tuple(trace), reason=str(step_trace.observation.get("error", "")))
        if not step_trace.ok and step_trace.action == "patch":
            return AgentLoopResult("refused", index, tuple(trace), reason="patch failed")
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
        parsed = _extract_single_json_object(text)
    if parsed is None:
        return None
    return _action_from_mapping(parsed)


def plan_natural_language_fallback_actions(objective: str) -> tuple[ActionPayload, ...]:
    """Infer a tiny, fail-closed action plan for obvious write-file requests.

    This is deliberately not a general planner. It only covers plain-English
    prompts such as "create a file named hello.txt that says hello" when the
    model backend cannot provide a JSON action. If the path, content, or an
    optional test command cannot be inferred safely, it returns no plan.
    """

    parsed = _parse_write_file_request(objective)
    if parsed is None:
        return ()

    path, content, command = parsed
    actions: list[ActionPayload] = [
        {
            "action": "write",
            "path": path,
            "content": content,
            "reason": "natural-language fallback write",
        }
    ]
    if command:
        actions.append(
            {
                "action": "verify",
                "command": list(command),
                "reason": "natural-language fallback explicit verification command",
            }
        )
    else:
        actions.append(
            {
                "action": "verify",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import sys; "
                        "actual = Path(sys.argv[1]).read_text(encoding='utf-8'); "
                        "raise SystemExit(0 if actual == sys.argv[2] else 1)"
                    ),
                    path,
                    content,
                ],
                "reason": "natural-language fallback verified written content",
            }
        )
    return tuple(actions)


def render_model_action_prompt(state: AgentLoopState) -> str:
    """Render the bounded JSON-action contract for a model-driven step."""

    observations = [
        {
            "step": item.step,
            "action": item.action,
            "ok": item.ok,
            "observation": _clip_observation(item.observation),
        }
        for item in state.trace
    ]
    return (
        "You are David's terminal coding-agent action planner.\n"
        f"Return exactly one JSON object after the label `{ACTION_JSON_SENTINEL}` and no prose.\n"
        "Allowed actions: plan, read, patch, write, run, shell, verify, done, none.\n"
        "Use run/shell only with command as an array of arguments, never a shell string.\n"
        "Use relative workspace paths only. Do not request deletion or path escapes.\n"
        "For repo fixes, prefer read -> patch -> verify. Use patch content for strict search/replace blocks or unified diff text.\n"
        "Prefer read/plan before write. Verify with passed or a command array when work is complete.\n"
        f"{ACTION_JSON_SENTINEL} "
        '{"action":"plan|read|patch|write|run|verify|done|none","path":"relative/path",'
        '"content":"text","command":["program","arg"],"cwd":".","timeout":30,'
        '"passed":true,"reason":"short reason"}\n'
        f"Workspace: {state.workspace_root}\n"
        f"Objective: {state.objective}\n"
        f"Step: {state.step} of {state.max_steps}\n"
        f"Prior observations: {json.dumps(observations, sort_keys=True)}"
    )


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
        if action.action == "patch":
            if action.content is None or not action.content.strip():
                raise AgentLoopError("patch requires non-empty content")
            diagnostic = apply_patch_candidate(tools.workspace_root, action.content)
            return _trace(
                step,
                "patch",
                diagnostic.ok,
                diagnostic.to_dict(),
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


def _extract_single_json_object(text: str) -> Mapping[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        end = _balanced_json_object_end(text, start)
        if end is None:
            index = start + 1
            continue
        parsed = _parse_json_object(text[start:end])
        if parsed is not None:
            candidates.append(parsed)
        index = end

    if len(candidates) > 1:
        raise AgentLoopError("multiple action JSON objects are not allowed")
    if not candidates:
        return None
    return candidates[0]


def _balanced_json_object_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
            if depth < 0:
                return None
    return None


def _fallback_plan_for_state(state: AgentLoopState, *, mode: str) -> tuple[ActionPayload, ...]:
    if mode != "model_driven" or state.trace:
        return ()
    return plan_natural_language_fallback_actions(state.objective)


def _fallback_payload_for_step(
    requests: Sequence[ActionPayload],
    *,
    index: int,
    reason: str,
) -> ActionPayload:
    offset = index - 1
    if offset >= len(requests):
        return None
    request = requests[offset]
    if isinstance(request, Mapping):
        payload = dict(request)
        payload["_fallback_provenance"] = {
            "provenance": "david.agent_loop.natural_language_fallback",
            "reason": reason,
            "step": index,
        }
        return payload
    return request


def _parse_write_file_request(objective: str) -> tuple[str, str, tuple[str, ...] | None] | None:
    text = " ".join(objective.strip().split())
    if not text:
        return None
    delimiter = WRITE_CONTENT_RE.search(text)
    if delimiter is None:
        return None

    left = text[: delimiter.start()].strip()
    content_text = text[delimiter.end() :].strip()
    path = _extract_write_path(left)
    if path is None:
        return None

    content, command = _split_content_and_optional_command(content_text)
    if content is None:
        return None
    return path, content, command


def _extract_write_path(left: str) -> str | None:
    patterns = [
        rf"\b(?:named|called|at|path)\s+(?P<path>{QUOTED_OR_TOKEN_RE})\s*$",
        rf"\b(?:create|write|make)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?P<path>{QUOTED_OR_TOKEN_RE})\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, left, re.IGNORECASE)
        if not match:
            continue
        path = _clean_quoted_value(match.group("path")).rstrip(".,;:")
        if _is_safe_fallback_path(path):
            return path
    return None


def _split_content_and_optional_command(content_text: str) -> tuple[str | None, tuple[str, ...] | None]:
    command: tuple[str, ...] | None = None
    content = content_text.strip()
    command_match = OPTIONAL_RUN_RE.search(content)
    if command_match:
        command_text = command_match.group(1).strip()
        content = content[: command_match.start()].strip()
        command = _parse_simple_test_command(command_text)
        if command is None:
            return None, None
    content = _clean_quoted_value(content)
    if not content:
        return None, None
    return content, command


def _parse_simple_test_command(command_text: str) -> tuple[str, ...] | None:
    cleaned = command_text.strip().rstrip(".")
    if not cleaned or re.search(r"[;&|><`$()\{\}\n\r]", cleaned):
        return None
    try:
        parts = shlex.split(cleaned)
    except ValueError:
        return None
    if not parts or any(not _is_safe_command_arg(part) for part in parts):
        return None

    head = parts[0].lower()
    if head == "pytest":
        return tuple(parts)
    if head in {"python", "python3", "py"} and len(parts) >= 3 and parts[1] == "-m":
        module = parts[2]
        if module in {"pytest", "py_compile"}:
            return (sys.executable, "-m", *parts[2:])
    if parts in (["npm", "test"], ["pnpm", "test"], ["yarn", "test"]):
        return tuple(parts)
    if len(parts) == 3 and parts[0] in {"npm", "pnpm", "yarn"} and parts[1] == "run" and parts[2] == "test":
        return tuple(parts)
    return None


def _clean_quoted_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"', "`"}:
        return cleaned[1:-1]
    return cleaned


def _is_safe_fallback_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not normalized or normalized.lower() in {"file", "new", "a"}:
        return False
    if "\x00" in normalized or "\n" in normalized or "\r" in normalized:
        return False
    if "://" in normalized or normalized.startswith("~"):
        return False
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(path).is_absolute():
        return False
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        return False
    if any(char in normalized for char in "*?[]{}"):
        return False
    return normalized not in PROTECTED_FALLBACK_PATHS


def _is_safe_command_arg(arg: str) -> bool:
    if not arg or "\x00" in arg or "\n" in arg or "\r" in arg:
        return False
    if ".." in arg.replace("\\", "/").split("/"):
        return False
    return re.fullmatch(r"[\w./\\:=@+,-]+", arg) is not None


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


def _clip_observation(observation: Mapping[str, Any], *, max_chars: int = 1_500) -> dict[str, Any]:
    clipped: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, str) and len(value) > max_chars:
            clipped[key] = value[:max_chars] + "...[truncated]"
        else:
            clipped[key] = value
    return clipped
