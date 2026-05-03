"""David terminal-agent runtime facade.

DavidRuntime is product orchestration glue. It boots the validated harness when
there is a model, keeps offline/no-model shell work separate, and exposes
small helpers that a CLI/TUI can call without importing model backends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from chuk_lazarus.david.config import DavidConfig, DavidMode, DavidRuntimeConfig
from chuk_lazarus.harness.boot import boot_harness as default_boot_harness
from chuk_lazarus.repl_agent_tools import (
    LocalCodingToolRunner,
    ToolCall,
    ToolResult,
    extract_tool_calls,
    format_tool_results,
)


@dataclass(frozen=True)
class DavidTaskMethodology:
    """Detected task shape and the production methodology David should use."""

    task_type: str
    methodology: str
    proof_rig: str
    capabilities: tuple[str, ...] = ()
    route_reason: str = ""
    verification_hints: tuple[str, ...] = ()
    confidence: float = 0.0

    @property
    def capability_mode(self) -> str:
        return {
            "repo_patch": "patch_target",
            "source_dependency_reasoning": "dependency_source",
            "symbolic_multi_hop": "symbolic_chain",
            "temporal_recall": "temporal_ordinal",
            "user_continuity": "durable_chat_memory",
            "terminal_triage": "general_recall",
        }.get(self.task_type, self.task_type)

    @property
    def selected_methodology(self) -> str:
        return self.capability_mode

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        data["verification_hints"] = list(self.verification_hints)
        data["capability_mode"] = self.capability_mode
        data["selected_methodology"] = self.selected_methodology
        return data


@dataclass(frozen=True)
class DavidRuntimeStatus:
    """Serializable summary of David's current boot/tool state."""

    mode: str
    workspace_path: str
    model_path: str | None
    validation_report_path: str | None
    model_requested: bool
    model_boot_attempted: bool
    harness_booted: bool
    model_validated: bool
    model_load_allowed: bool
    fail_closed: bool
    local_tools_available: bool
    shell_available: bool
    user_memory_ready: bool | None
    workspace_memory_ready: bool | None
    workspace_index_ready: bool | None
    jit_required: bool | None
    jit_actions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["jit_actions"] = list(self.jit_actions)
        data["errors"] = list(self.errors)
        data["warnings"] = list(self.warnings)
        return data


_METHODOLOGY_CATALOG: tuple[DavidTaskMethodology, ...] = (
    DavidTaskMethodology(
        task_type="repo_patch",
        methodology="patch_targeting",
        proof_rig="patch-target proof rig",
        capabilities=(
            "repo patch-target routing",
            "source/dependency routing",
            "constrained edit decoding",
        ),
        route_reason="The task asks for code changes, tests, diffs, or patchable repo work.",
        verification_hints=("run focused tests", "check changed files", "inspect git diff"),
        confidence=0.75,
    ),
    DavidTaskMethodology(
        task_type="source_dependency_reasoning",
        methodology="dependency_routing",
        proof_rig="dependency proof rig",
        capabilities=("source/dependency routing", "symbol lookup", "evidence windows"),
        route_reason="The task asks how source files, imports, symbols, or dependencies connect.",
        verification_hints=("cite file spans", "confirm dependency chain"),
        confidence=0.7,
    ),
    DavidTaskMethodology(
        task_type="symbolic_multi_hop",
        methodology="symbolic_chain_routing",
        proof_rig="symbolic-chain proof rig",
        capabilities=("symbolic chain routing", "exact ID recall", "multi-hop task routing"),
        route_reason="The task requires chained symbolic facts, IDs, or ordered constraints.",
        verification_hints=("show each hop", "check requested exact values"),
        confidence=0.68,
    ),
    DavidTaskMethodology(
        task_type="temporal_recall",
        methodology="temporal_ordinal_recall",
        proof_rig="temporal proof rig",
        capabilities=("temporal ordinal recall", "entity mention recall", "recency scoring"),
        route_reason="The task asks about earlier/later events, dates, deadlines, or prior turns.",
        verification_hints=("use concrete dates", "distinguish similar occurrences"),
        confidence=0.68,
    ),
    DavidTaskMethodology(
        task_type="user_continuity",
        methodology="durable_user_memory",
        proof_rig="interactive memory proof rig",
        capabilities=("person-in-time memory", "staleness handling", "preference recall"),
        route_reason="The task asks David to remember preferences, decisions, or user context.",
        verification_hints=("respect stale memories", "surface uncertainty"),
        confidence=0.64,
    ),
    DavidTaskMethodology(
        task_type="terminal_triage",
        methodology="terminal_agent_triage",
        proof_rig="Harness smoke",
        capabilities=("topical search", "tool execution", "verification command execution"),
        route_reason="No stronger specialized methodology matched.",
        verification_hints=("gather context", "run the smallest useful check"),
        confidence=0.45,
    ),
)


class DavidRuntime:
    """Product runtime facade for a runnable David terminal coding agent."""

    def __init__(
        self,
        config: DavidRuntimeConfig | Mapping[str, Any] | None = None,
        *,
        streams: Any | None = None,
        boot_harness: Any = default_boot_harness,
        model_loader: Any | None = None,
        decoder: Any | None = None,
        tool_runner_factory: Any | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            self.config = DavidRuntimeConfig.from_mapping(overrides)
        elif isinstance(config, DavidRuntimeConfig):
            self.config = config.with_overrides(**overrides)
        elif isinstance(config, Mapping):
            self.config = DavidRuntimeConfig.from_mapping(config, **overrides)
        else:
            raise TypeError("config must be DavidRuntimeConfig, a mapping, or None")

        self.streams = streams or SimpleNamespace(stdin=None, stdout=None, stderr=None)
        self._boot_harness = boot_harness
        self._model_loader = model_loader
        self.decoder = decoder

        timeout_seconds = max(
            int(self.config.tool_timeout_seconds),
            int(self.config.verification_timeout_seconds),
        )
        runner_factory = tool_runner_factory or LocalCodingToolRunner
        self.tool_runner = runner_factory(
            self.config.resolved_workspace_path,
            self.config.resolved_trace_root,
            timeout_seconds=timeout_seconds,
            max_output_chars=self.config.max_tool_output_chars,
            allow_shell=self.config.allow_shell,
            custom_tools_root=self.config.resolved_custom_tools_root,
            allow_custom_tools=self.config.allow_custom_tools,
        )
        self.harness_session: Any | None = None
        self.session: Any | None = None
        self.model: Any | None = None
        self._boot_attempted = False
        self._boot_errors: list[str] = []
        self._boot_warnings: list[str] = []

    @classmethod
    def from_config(
        cls,
        config: DavidRuntimeConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> "DavidRuntime":
        return cls(config, **overrides)

    def boot(self) -> "DavidRuntime":
        """Boot model validation/memory readiness when model mode is requested."""

        self._boot_attempted = True
        self._boot_errors = list(self.config.validation_errors())
        self._boot_warnings = []
        self.harness_session = None

        if self._boot_errors:
            return self
        if self.config.mode is DavidMode.OFFLINE_SHELL:
            self._boot_warnings.append("offline_no_model_shell_mode")
            return self

        model_path = self.config.resolved_model_path
        if model_path is None:
            self._boot_warnings.append("model_path_missing")
            return self

        try:
            self.harness_session = self._boot_harness(
                model_path=str(model_path),
                workspace_path=str(self.config.resolved_workspace_path),
                validation_report_path=_path_or_none(self.config.resolved_validation_report_path),
                require_validated_model=self.config.require_validated_model,
            )
            self.session = self.harness_session
        except Exception as exc:  # noqa: BLE001 - fail-closed state belongs in status
            self._boot_errors.append(f"model_boot_failed: {exc}")
            self.harness_session = None
            self.session = None
        return self

    start = boot

    def initialize(self) -> Any | None:
        """Boot the harness and only load model weights when explicitly safe."""

        self.boot()
        if (
            self.model_load_allowed
            and not self.config.dry_run
            and self._model_loader is not None
        ):
            self.model = self._model_loader(self.harness_session)
        return self.harness_session

    @property
    def model_validated(self) -> bool:
        session = self.harness_session
        if session is None:
            return False
        if bool(getattr(session, "can_auto_load", False)):
            return True
        metadata = getattr(session, "metadata", None)
        if isinstance(metadata, Mapping) and metadata.get("model_validated") is not None:
            return bool(metadata["model_validated"])
        return bool(getattr(session, "model_validated", False))

    @property
    def model_load_allowed(self) -> bool:
        """Whether a backend loader may safely instantiate the configured model."""

        if self.harness_session is None or self._boot_errors:
            return False
        return self.model_validated

    @property
    def fail_closed(self) -> bool:
        return bool(
            self._boot_attempted
            and self.config.model_requested
            and not self.model_load_allowed
        )

    @property
    def local_tools_available(self) -> bool:
        if self.config.allow_no_model_shell:
            return True
        if self.config.model_requested and not self._boot_attempted:
            self.boot()
        return self.model_load_allowed

    @property
    def shell_available(self) -> bool:
        return bool(self.config.allow_shell and self.local_tools_available)

    def require_model_load_allowed(self) -> Any:
        """Return the validated harness session or raise before backend loading."""

        if not self._boot_attempted:
            self.boot()
        if not self.model_load_allowed:
            raise RuntimeError(
                "DavidRuntime refuses to load a model without an accepted, "
                "auto-load-safe validation report."
            )
        return self.harness_session

    load_model = require_model_load_allowed

    def status(self) -> DavidRuntimeStatus:
        session = self.harness_session
        errors = tuple(_dedupe((*self.config.validation_errors(), *self._boot_errors)))
        warnings = tuple(_dedupe((*self._boot_warnings, *_session_warnings(session))))
        return DavidRuntimeStatus(
            mode=self.config.mode.value,
            workspace_path=str(self.config.resolved_workspace_path),
            model_path=_path_or_none(self.config.resolved_model_path),
            validation_report_path=_path_or_none(self.config.resolved_validation_report_path),
            model_requested=self.config.model_requested,
            model_boot_attempted=self._boot_attempted,
            harness_booted=session is not None,
            model_validated=self.model_validated,
            model_load_allowed=self.model_load_allowed,
            fail_closed=self.fail_closed,
            local_tools_available=self.local_tools_available,
            shell_available=self.shell_available,
            user_memory_ready=_session_bool(session, "user_memory_ready", "user_memory"),
            workspace_memory_ready=_session_bool(
                session, "workspace_memory_ready", "task_memory"
            ),
            workspace_index_ready=_session_bool(
                session, "workspace_index_ready", "index_readiness"
            ),
            jit_required=_session_scalar(session, "jit_required"),
            jit_actions=tuple(_session_sequence(session, "jit_actions")),
            errors=errors,
            warnings=warnings,
            session_id=getattr(session, "session_id", None),
        )

    def status_summary(self) -> dict[str, Any]:
        """Return a compact serializable status payload for CLI/TUI display."""

        return self.status().to_dict()

    def roadmap_summary(self) -> dict[str, Any]:
        """Describe how this runtime maps proof rigs to product capabilities."""

        return {
            "product": "David terminal coding agent",
            "active_mode": self.config.mode.value,
            "runtime_contract": [
                "validate model config before backend model loading",
                "allow offline/no-model local coding tools as a separate mode",
                "route tasks by methodology, not benchmark module",
                "execute verification commands through traced local tools",
            ],
            "proof_rigs": {
                "MRCR": "temporal ordinal recall",
                "RULER": "symbolic multi-hop task memory",
                "LoCoBench": "source/dependency routing",
                "SWE-bench": "repo patch-target routing",
                "Chat": "durable user/task memory",
            },
            "implemented_here": [
                "harness boot gate",
                "fail-closed model load guard",
                "offline shell/tool runner",
                "status summary",
                "methodology detector",
                "verification command helper",
            ],
            "handoff_boundaries": [
                "CLI/TUI wiring lives outside this runtime slice",
                "model adapters and routers remain owned by harness/router slices",
                "benchmark scripts stay proof rigs, not product modules",
            ],
        }

    def detect_task_methodology(
        self, task: str, *, paths: Sequence[str] = ()
    ) -> DavidTaskMethodology:
        """Detect the methodology David should use for a user task."""

        text = f"{task or ''} {' '.join(paths)}".lower()
        scores = {item.task_type: 0 for item in _METHODOLOGY_CATALOG}

        if _contains_any(
            text,
            (
                "fix ",
                "bug",
                "patch",
                "implement",
                "edit",
                "failing test",
                "pytest",
                "diff",
                ".py",
                ".ts",
                ".js",
            ),
        ):
            scores["repo_patch"] += 4
        if _contains_any(
            text,
            ("import", "dependency", "depends", "call graph", "symbol", "source", "span"),
        ):
            scores["source_dependency_reasoning"] += 3
        if _contains_any(
            text,
            ("multi-hop", "chain", "derive", "exact id", " id:", "constraint", "prove"),
        ):
            scores["symbolic_multi_hop"] += 3
        if _contains_any(
            text,
            (
                "first time",
                "second time",
                "third time",
                "fourth time",
                "previous",
                "earlier",
                "later",
                "last time",
                "yesterday",
                "today",
                "tomorrow",
                "deadline",
                "when did",
            ),
        ):
            scores["temporal_recall"] += 3
        if _contains_any(
            text,
            ("remember", "preference", "prefer", "decision", "user context", "my goal"),
        ):
            scores["user_continuity"] += 3

        best_type = max(scores, key=scores.get)
        if scores[best_type] <= 0:
            best_type = "terminal_triage"

        for methodology in _METHODOLOGY_CATALOG:
            if methodology.task_type == best_type:
                if best_type == "terminal_triage":
                    return methodology
                confidence = min(0.95, methodology.confidence + scores[best_type] / 20)
                return DavidTaskMethodology(
                    task_type=methodology.task_type,
                    methodology=methodology.methodology,
                    proof_rig=methodology.proof_rig,
                    capabilities=methodology.capabilities,
                    route_reason=methodology.route_reason,
                    verification_hints=methodology.verification_hints,
                    confidence=confidence,
                )
        return _METHODOLOGY_CATALOG[-1]

    def methodology_summary(self, task: str, *, paths: Sequence[str] = ()) -> dict[str, Any]:
        return self.detect_task_methodology(task, paths=paths).to_dict()

    def available_tools(self) -> list[str]:
        return self.tool_runner.available_tool_names()

    available_tool_names = available_tools

    def tool_system_prompt(self) -> str:
        return self.tool_runner.tool_system_prompt()

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        return extract_tool_calls(text)

    def execute_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        return self.execute_tool_call(
            ToolCall(name=name, arguments=dict(arguments or {})),
            session_id=session_id,
            turn_index=turn_index,
        )

    def execute_tool_call(
        self,
        call: ToolCall | Mapping[str, Any],
        *,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        tool_call = _coerce_tool_call(call)
        if not self.local_tools_available:
            return _blocked_tool_result(
                tool_call,
                "local tools are unavailable because model validation failed "
                "and offline/no-model shell mode is disabled",
            )
        if tool_call.name == "shell" and not self.shell_available:
            return _blocked_tool_result(tool_call, "shell tool is disabled")
        return self.tool_runner.execute(
            tool_call,
            session_id=session_id or self.config.session_id,
            turn_index=turn_index,
        )

    def execute_tools(
        self,
        calls: Iterable[ToolCall | Mapping[str, Any]],
        *,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> list[ToolResult]:
        return [
            self.execute_tool_call(call, session_id=session_id, turn_index=turn_index)
            for call in calls
        ]

    def format_tool_results(self, results: list[ToolResult]) -> str:
        return format_tool_results(results)

    def run_shell(
        self,
        command: str,
        *,
        timeout_seconds: int | None = None,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        args: dict[str, Any] = {"command": command}
        if timeout_seconds is not None:
            args["timeout_seconds"] = timeout_seconds
        return self.execute_tool(
            "shell",
            args,
            session_id=session_id,
            turn_index=turn_index,
        )

    def search(
        self,
        pattern: str,
        *,
        path: str = ".",
        max_results: int = 100,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        return self.execute_tool(
            "search",
            {
                "pattern": pattern,
                "path": path,
                "max_results": max_results,
            },
            session_id=session_id,
            turn_index=turn_index,
        )

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        limit: int = 120,
        session_id: str | None = None,
        turn_index: int | None = None,
    ) -> ToolResult:
        return self.execute_tool(
            "read_file",
            {
                "path": path,
                "start_line": start_line,
                "limit": limit,
            },
            session_id=session_id,
            turn_index=turn_index,
        )

    def run_verification_command(
        self,
        command: str,
        *,
        timeout_seconds: int | None = None,
        session_id: str | None = None,
    ) -> ToolResult:
        timeout = timeout_seconds or self.config.verification_timeout_seconds
        return self.execute_tool(
            "shell",
            {
                "command": command,
                "timeout_seconds": timeout,
                "purpose": "verification",
            },
            session_id=session_id or self.config.session_id or "verification",
        )

    def run_verification(
        self,
        commands: Sequence[str] | None = None,
        *,
        timeout_seconds: int | None = None,
        session_id: str | None = None,
    ) -> list[ToolResult]:
        selected = tuple(commands or self.config.verification_commands)
        return [
            self.run_verification_command(
                command,
                timeout_seconds=timeout_seconds,
                session_id=session_id,
            )
            for command in selected
        ]

    verify = run_verification

    def run_once(self, prompt: str) -> SimpleNamespace:
        """Run one agent turn, including a simple local tool-call loop."""

        if not self._boot_attempted:
            self.initialize()

        if self.decoder is None:
            methodology = self.detect_task_methodology(prompt)
            return SimpleNamespace(
                answer=(
                    f"Task routed as {methodology.capability_mode}. "
                    "Model decode is not configured yet; use local tools or attach a decoder."
                ),
                verification_summary="not requested",
                exit_code=0,
                tool_results=[],
                events=[
                    {
                        "type": "methodology",
                        "capability_mode": methodology.capability_mode,
                    }
                ],
            )

        first = _complete_with_decoder(self.decoder, prompt)
        calls = self.parse_tool_calls(first)
        if not calls:
            return SimpleNamespace(
                answer=first,
                verification_summary="not requested",
                exit_code=0,
                tool_results=[],
                events=[],
            )

        results = self.execute_tools(
            calls,
            session_id=getattr(self.harness_session, "session_id", self.config.session_id),
            turn_index=1,
        )
        tool_payload = self.format_tool_results(results)
        second = _complete_with_decoder(self.decoder, prompt, tool_payload)
        return SimpleNamespace(
            answer=second,
            verification_summary="verified: local tool trace captured",
            exit_code=0 if all(result.ok for result in results) else 1,
            tool_results=results,
            events=[
                {
                    "type": "tool_call",
                    "name": result.name,
                    "ok": result.ok,
                }
                for result in results
            ],
        )

    @staticmethod
    def verification_summary(results: Sequence[ToolResult]) -> dict[str, Any]:
        return {
            "passed": all(result.ok for result in results),
            "command_count": len(results),
            "failed_commands": [
                {
                    "name": result.name,
                    "error": result.error,
                    "output": result.output,
                    "metadata": result.metadata,
                }
                for result in results
                if not result.ok
            ],
        }


def detect_task_methodology(task: str, *, paths: Sequence[str] = ()) -> DavidTaskMethodology:
    """Benchmark-name-free product task detector."""

    return DavidRuntime.detect_task_methodology(None, task, paths=paths)


def _complete_with_decoder(decoder: Any, *args: Any, **kwargs: Any) -> str:
    for name in ("complete", "generate"):
        method = getattr(decoder, name, None)
        if callable(method):
            return str(method(*args, **kwargs))
    if callable(decoder):
        return str(decoder(*args, **kwargs))
    raise TypeError("decoder must expose complete(), generate(), or be callable")


def _coerce_tool_call(call: ToolCall | Mapping[str, Any]) -> ToolCall:
    if isinstance(call, ToolCall):
        return call
    name = call.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call mapping requires a non-empty name")
    args = call.get("arguments", call.get("args", {}))
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ValueError("tool call arguments must be a mapping")
    call_id = call.get("call_id", call.get("id"))
    if call_id is None:
        return ToolCall(name=name, arguments=dict(args))
    return ToolCall(name=name, arguments=dict(args), call_id=str(call_id))


def _blocked_tool_result(call: ToolCall, error: str) -> ToolResult:
    now = _utc_now()
    return ToolResult(
        call_id=call.call_id,
        name=call.name,
        ok=False,
        output="",
        error=error,
        started_at=now,
        finished_at=now,
        elapsed_seconds=0.0,
        metadata={"blocked": True},
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _path_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _session_warnings(session: Any | None) -> list[str]:
    if session is None:
        return []
    warnings: list[str] = []
    for warning in getattr(session, "warnings", ()) or ():
        code = getattr(warning, "code", None)
        warnings.append(str(code if code is not None else warning))
    metadata = getattr(session, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("boot_warnings", "warnings"):
            raw = metadata.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, str):
                warnings.extend(str(item) for item in raw)
    return warnings


def _session_bool(session: Any | None, metadata_name: str, component_name: str) -> bool | None:
    if session is None:
        return None
    direct = getattr(session, metadata_name, None)
    if direct is not None:
        return bool(direct)
    metadata = getattr(session, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get(metadata_name) is not None:
        return bool(metadata[metadata_name])
    component = getattr(session, component_name, None)
    state = getattr(component, "state", None)
    if state is None:
        return None
    return str(getattr(state, "value", state)) == "ready"


def _session_scalar(session: Any | None, name: str) -> Any | None:
    if session is None:
        return None
    direct = getattr(session, name, None)
    if direct is not None:
        return direct
    metadata = getattr(session, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return None


def _session_sequence(session: Any | None, name: str) -> tuple[str, ...]:
    value = _session_scalar(session, name)
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def run_once(options: Any) -> int:
    """Minimal non-interactive CLI backend used by ``david --once``."""

    runtime = DavidRuntime(
        {
            "workspace": getattr(options, "workspace", "."),
            "model": getattr(options, "model", None),
            "validation_report": getattr(options, "validation_report", None),
            "allow_unvalidated": getattr(options, "allow_unvalidated", False),
        }
    ).boot()
    status = runtime.status_summary()
    if status["errors"]:
        for error in status["errors"]:
            print(f"david: {error}")
        return 1
    print("david: runtime ready")
    print(f"david: mode={status['mode']}")
    return 0


__all__ = [
    "DavidConfig",
    "DavidMode",
    "DavidRuntime",
    "DavidRuntimeConfig",
    "DavidRuntimeStatus",
    "DavidTaskMethodology",
    "detect_task_methodology",
    "run_once",
]
