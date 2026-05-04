"""Command-line entry point for the David terminal agent."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .doctor import format_doctor_report, run_doctor
from .model_validation import (
    AutoModelValidationResult,
    ModelCommandResult,
    ValidationReportDiscovery,
    discover_validation_report,
    run_auto_model_validation,
    run_model_scan,
    run_model_validate,
)
from .model_onboarding import ModelOnboardingResult, onboard_model
from .tui import DavidTui
from .workspace_init import WorkspaceInitError, WorkspaceInitResult, initialize_workspace

try:  # Prefer the real harness objects when another worker has provided them.
    from .config import DavidConfig as _RuntimeDavidConfig
    from .runtime import DavidRuntime as _RuntimeDavidRuntime
except ImportError:  # pragma: no cover - exercised by isolated CLI installs.
    _RuntimeDavidConfig = None
    _RuntimeDavidRuntime = None


@dataclass(frozen=True)
class _FallbackDavidConfig:
    """CLI-owned boot settings for the David terminal surface."""

    workspace_path: Path
    model_path: str | None = None
    validation_report_path: str | None = None
    model_attestation_path: str | None = None
    require_validated_model: bool = True
    color: bool = True
    once: str | None = None
    verify_command: str | None = None
    command_timeout_seconds: int | None = None
    auto_jit_index: bool = False
    model_backend: str | None = None
    model_device: str | None = None
    model_dtype: str | None = None
    model_max_new_tokens: int | None = None

    @classmethod
    def from_values(
        cls,
        *,
        workspace_path: Path,
        model_path: str | None,
        validation_report_path: str | None,
        require_validated_model: bool,
        color: bool,
        once: str | None,
        verify_command: str | None,
        command_timeout_seconds: int | None,
        auto_jit_index: bool = False,
        model_backend: str | None = None,
        model_device: str | None = None,
        model_dtype: str | None = None,
        model_max_new_tokens: int | None = None,
        model_attestation_path: str | None = None,
    ) -> "_FallbackDavidConfig":
        return cls(
            workspace_path=workspace_path.expanduser().resolve(),
            model_path=model_path,
            validation_report_path=validation_report_path,
            model_attestation_path=model_attestation_path,
            require_validated_model=require_validated_model,
            color=color,
            once=once,
            verify_command=verify_command,
            command_timeout_seconds=command_timeout_seconds,
            auto_jit_index=auto_jit_index,
            model_backend=model_backend.strip().lower() if model_backend is not None else None,
            model_device=model_device,
            model_dtype=model_dtype,
            model_max_new_tokens=model_max_new_tokens,
        )


class _FallbackDavidRuntime:
    """Small CLI adapter until the full harness runtime is wired in."""

    def __init__(self, config: Any) -> None:
        self.config = config

    @classmethod
    def create(cls, config: Any) -> "_FallbackDavidRuntime":
        return cls(config)

    def readiness(self) -> dict[str, str]:
        validation = "ready"
        if self.config.validation_report_path:
            report = Path(self.config.validation_report_path).expanduser()
            validation = "ready" if report.exists() else "missing report"
        elif self.config.require_validated_model:
            validation = "blocked: no validated model report"
        else:
            validation = "review: unvalidated model allowed"

        return {
            "model validation": validation,
            "index": self._workspace_state((".lazarus", ".david", ".index")),
            "memory": self._workspace_state((".lazarus_memory", ".memory", ".david_memory")),
        }

    def respond(self, prompt: str) -> str:
        return (
            "David harness runtime is not connected yet.\n"
            f"Captured one-shot prompt for {self.config.workspace_path}: {prompt}"
        )

    def memory_status(self) -> str:
        return f"memory: {self.readiness()['memory']}"

    def index_status(self) -> str:
        return f"index: {self.readiness()['index']}"

    def jit_index(self) -> str:
        return self.index_status()

    def build_index(self) -> str:
        return self.jit_index()

    def refresh_index(self) -> str:
        return self.jit_index()

    def verify(self, command: str | None = None) -> str:
        command = command or self.config.verify_command
        if not command:
            return "verify: no command configured"
        return self.run_shell(command)

    def run_shell(self, command: str) -> str:
        timeout = self.config.command_timeout_seconds
        completed = subprocess.run(
            command,
            cwd=self.config.workspace_path,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (completed.stdout + completed.stderr).strip()
        if not output:
            output = "(no output)"
        return f"$ {command}\nrc={completed.returncode}\n{output}"

    def _workspace_state(self, names: tuple[str, ...]) -> str:
        if any((self.config.workspace_path / name).exists() for name in names):
            return "ready"
        return "missing: JIT required"


DavidConfig = _RuntimeDavidConfig or _FallbackDavidConfig
DavidRuntime = _RuntimeDavidRuntime or _FallbackDavidRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="david",
        description="David: local terminal coding-agent harness.",
    )
    _add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command")
    init = subparsers.add_parser(
        "init",
        help="Initialize David workspace state without model load or indexing",
    )
    init.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite David-managed config and next-steps files",
    )
    _add_common_options(init, suppress_defaults=True)
    code = subparsers.add_parser("code", help="Open David in a workspace")
    code.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    _add_common_options(code, suppress_defaults=True)
    verify = subparsers.add_parser(
        "verify",
        help="Run David verification once without opening the TUI",
    )
    verify.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    verify.add_argument(
        "--cmd",
        default=None,
        help="Workspace shell command to verify through runtime.verify(command)",
    )
    verify.add_argument(
        "--patch",
        action="store_true",
        help="Run David's built-in verification path when --cmd is not supplied",
    )
    _add_common_options(verify, suppress_defaults=True)
    index = subparsers.add_parser(
        "index",
        help="Inspect or build David workspace indexes without opening the TUI",
    )
    index.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    index_action = index.add_mutually_exclusive_group()
    index_action.add_argument("--status", action="store_true", help="Print index readiness")
    index_action.add_argument("--build", action="store_true", help="Build the workspace index")
    index_action.add_argument("--refresh", action="store_true", help="Refresh the workspace index")
    _add_common_options(index, suppress_defaults=True)
    memory = subparsers.add_parser(
        "memory",
        help="Print David user/task memory artifact status without opening the TUI",
    )
    memory.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    _add_common_options(memory, suppress_defaults=True)
    resume = subparsers.add_parser(
        "resume",
        help="Print the saved David resume summary without opening the TUI",
    )
    resume.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    _add_common_options(resume, suppress_defaults=True)
    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect local model boot readiness without downloads or model load",
    )
    doctor.add_argument("--workspace", default=".", help="Workspace path")
    doctor.add_argument("--model", default=argparse.SUPPRESS, help="Path or HF id for the open model")
    doctor.add_argument(
        "--validation-report",
        default=argparse.SUPPRESS,
        help="Path to a validator JSON report accepted by the harness boot gate",
    )
    doctor.add_argument(
        "--model-attestation",
        default=argparse.SUPPRESS,
        help="Path to a manual-reviewed attestation JSON for standard decode of needs-review reports",
    )
    doctor.add_argument(
        "--auto-validate-model",
        action="store_true",
        help="Report whether David can run the standalone scanner and validator",
    )
    subparsers.add_parser(
        "capabilities",
        help="Print wired, guarded, and TODO David product surfaces without model load",
    )
    model = subparsers.add_parser("model", help="Explicit model scan and validation commands")
    model_subparsers = model.add_subparsers(dest="model_command", required=True)
    scan = model_subparsers.add_parser(
        "scan",
        help="Run the standalone David model-config scanner",
        description="Run David/get_model_config.py explicitly.",
    )
    scan.add_argument("model", help="HF model id or local model path to scan")
    scan.add_argument("--output", required=True, help="Path for the generated scan report JSON")
    validate = model_subparsers.add_parser(
        "validate",
        help="Run the standalone David model-config validator",
        description="Run David/validate_model_config.py explicitly.",
    )
    validate.add_argument("report", help="Path to a model scan report JSON")
    validate.add_argument("--output", required=True, help="Path for the validation report JSON")
    validate.add_argument("--model", default=None, help="Optional model path/id override")
    onboard = model_subparsers.add_parser(
        "onboard",
        help="Plan or execute safe scanner/validator onboarding for a model",
    )
    onboard.add_argument("model", help="HF model id or local model path to onboard")
    onboard.add_argument("--workspace", required=True, help="Workspace path for onboarding artifacts")
    onboard.add_argument(
        "--execute",
        action="store_true",
        help="Run scan/validate wrappers; does not create or accept attestations",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_list)
    if getattr(args, "command", None) == "init":
        explicit_args = _explicit_common_args(argv_list)
        _apply_operator_defaults(args, explicit_args=explicit_args)
        return _run_init_command(args)
    if getattr(args, "command", None) == "doctor":
        explicit_args = _explicit_common_args(argv_list)
        workspace_path = Path(args.workspace)
        workspace_defaults = _load_workspace_config_defaults(workspace_path)
        if isinstance(workspace_defaults, WorkspaceConfigError):
            _write_workspace_config_error(workspace_defaults)
            return 2
        _apply_workspace_config_defaults(args, workspace_path, workspace_defaults, explicit_args)
        _apply_operator_defaults(args, explicit_args=explicit_args)
        return _run_doctor_command(args)
    if getattr(args, "command", None) == "model":
        return _run_model_command(args)
    if getattr(args, "command", None) == "capabilities":
        return _run_capabilities_command()

    workspace = getattr(args, "workspace", ".")
    direct_commands = {"verify", "index", "memory", "resume"}
    if getattr(args, "command", None) not in (None, "code", *direct_commands):
        parser.error(f"unknown command: {args.command}")

    workspace_path = Path(workspace)
    explicit_args = _explicit_common_args(argv_list)
    workspace_defaults = _load_workspace_config_defaults(workspace_path)
    if isinstance(workspace_defaults, WorkspaceConfigError):
        _write_workspace_config_error(workspace_defaults)
        return 2
    _apply_workspace_config_defaults(args, workspace_path, workspace_defaults, explicit_args)
    _apply_operator_defaults(args, explicit_args=explicit_args)
    discovery = _resolve_validation_report(args, workspace_path)
    if _should_auto_validate_model(args, discovery):
        auto_result = run_auto_model_validation(model=args.model, workspace_path=workspace_path)
        if auto_result.returncode != 0:
            _write_auto_validation_failure(auto_result)
            return auto_result.returncode
        discovery = ValidationReportDiscovery(
            path=auto_result.validation_report_path,
            checked_paths=discovery.checked_paths + (auto_result.validation_report_path,),
        )
    if _missing_required_validation_report(args, discovery):
        _write_missing_validation_report_status(args, workspace_path, discovery)
        return 2

    args.validation_report = str(discovery.path) if discovery.path is not None else args.validation_report
    config = _build_config(args, workspace_path)
    runtime = DavidRuntime.create(config)
    if _startup_rejected_validated_model(args, runtime):
        _write_rejected_validation_report_status(args, workspace_path, runtime)
        return 2
    if getattr(args, "command", None) == "verify":
        return _run_verify_command(args, runtime)
    if getattr(args, "command", None) == "index":
        return _run_index_command(args, runtime)
    if getattr(args, "command", None) == "memory":
        return _run_memory_command(runtime)
    if getattr(args, "command", None) == "resume":
        return _run_resume_command(args, runtime)
    tui = DavidTui(runtime, color=config.color)
    return tui.run(once=config.once)


def _run_model_command(args: argparse.Namespace) -> int:
    if args.model_command == "scan":
        result = run_model_scan(
            model=args.model,
            output=Path(args.output).expanduser(),
        )
        return _write_model_command_result(
            result,
            success_message=f"Model scan report written to {Path(args.output).expanduser()}",
            failure_label="Model scan failed",
        )

    if args.model_command == "validate":
        result = run_model_validate(
            report=Path(args.report).expanduser(),
            output=Path(args.output).expanduser(),
            model=args.model,
        )
        return _write_model_command_result(
            result,
            success_message=f"Model validation report written to {Path(args.output).expanduser()}",
            failure_label="Model validation failed",
        )

    if args.model_command == "onboard":
        result = onboard_model(
            model=args.model,
            workspace_path=Path(args.workspace).expanduser(),
            execute=args.execute,
        )
        _write_model_onboarding_result(result)
        return 0 if result.status in {"planned", "accepted", "needs_review"} else 2

    raise AssertionError(f"Unhandled model command: {args.model_command}")


def _run_init_command(args: argparse.Namespace) -> int:
    workspace_path = Path(args.workspace)
    try:
        result = initialize_workspace(
            workspace_path,
            force=args.force,
            model=getattr(args, "model", None),
            backend=getattr(args, "model_backend", None),
            device=getattr(args, "model_device", None),
            dtype=getattr(args, "model_dtype", None),
            max_new_tokens=getattr(args, "model_max_new_tokens", None),
            auto_jit_index=bool(getattr(args, "auto_jit_index", False)),
        )
    except WorkspaceInitError as exc:
        sys.stderr.write("David init failed\n")
        sys.stderr.write(f"{exc}\n")
        return 2

    _write_workspace_init_result(result)
    return 0


def _run_doctor_command(args: argparse.Namespace) -> int:
    report = run_doctor(
        model=args.model,
        workspace_path=Path(args.workspace),
        validation_report=args.validation_report,
        attestation_path=getattr(args, "model_attestation", None),
        auto_validate_model=args.auto_validate_model,
    )
    sys.stdout.write(format_doctor_report(report))
    return 0 if report.ready else 2


def _run_capabilities_command() -> int:
    sys.stdout.write(format_capabilities_status())
    return 0


def format_capabilities_status() -> str:
    sections = (
        (
            "WIRED",
            (
                "CLI/TUI terminal surface",
                "boot harness and startup readiness gates",
                "validation and model attestation checks",
                "torch standard decode path guarded by validation/attestation",
                "doctor readiness surface with WSL path checks when available",
                "user/task memory artifact status",
                "workspace index status and build hooks",
                "capability router surface",
                "materializer metadata and compatibility guards",
                "residual-sidecar replay metadata status",
                "decoder prior store surface",
                "verification command surface",
                "operator tools surface",
                "safe agent loop shell",
            ),
        ),
        (
            "GUARDED/PARTIAL",
            (
                "model-driven repo autonomy remains guarded",
                "repo patching is partially wired through guarded read -> patch -> verify",
                "tensor replay is fail-closed behind compatibility evidence",
                "residual-sidecar replay is guarded and experimental",
                "live steering hooks are limited",
            ),
        ),
        (
            "TODO",
            (
                "real Gemma CUDA smoke and guided model onboarding",
                "full central router product wiring",
                "real activation/residual/KV indexes",
                "adapter-safe production KV/direct residual materialization",
                "live logit hooks and steering",
                "stronger semantic, chain, temporal, and behavioral verification",
                "broad multi-step autonomous repo patching",
            ),
        ),
    )
    lines = ["David capabilities"]
    for label, items in sections:
        lines.append(f"{label}:")
        lines.extend(f"- {item}" for item in items)
    return "\n".join(lines) + "\n"


def _run_verify_command(args: argparse.Namespace, runtime: Any) -> int:
    command = args.cmd
    text = runtime.verify(command) if command else _run_builtin_verify(runtime)
    if command is None and text and "David verification" not in text:
        text = f"David verification\n{text}"
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    if command:
        return _returncode_from_verify_text(text)
    return 0


def _run_index_command(args: argparse.Namespace, runtime: Any) -> int:
    if getattr(args, "build", False):
        text = _runtime_call(runtime, ("build_index", "jit_index", "refresh_index"))
    elif getattr(args, "refresh", False):
        text = _runtime_call(runtime, ("refresh_index", "jit_index", "build_index"))
    else:
        fallback = f"index: {_runtime_readiness_value(runtime, 'index')}"
        text = _runtime_call(runtime, ("index_status",), fallback=fallback)
    _write_command_text(text)
    return 0


def _run_memory_command(runtime: Any) -> int:
    fallback = f"memory: {_runtime_readiness_value(runtime, 'memory')}"
    _write_command_text(_runtime_call(runtime, ("memory_status",), fallback=fallback))
    return 0


def _run_resume_command(args: argparse.Namespace, runtime: Any) -> int:
    text = DavidTui(runtime, color=not args.no_color).format_resume()
    _write_command_text(text)
    return 0


def _run_builtin_verify(runtime: Any) -> str:
    run_once = getattr(runtime, "run_once", None)
    if callable(run_once):
        result = run_once("Verify quality gate")
        answer = getattr(result, "answer", None)
        if answer is not None:
            return str(answer)
        return str(result)
    return runtime.verify(None)


def _runtime_call(
    runtime: Any,
    names: tuple[str, ...],
    *args: Any,
    fallback: str | None = None,
) -> str:
    for name in names:
        method = getattr(runtime, name, None)
        if callable(method):
            return _stringify_runtime_value(method(*args))
    return fallback or "runtime hook unavailable"


def _runtime_readiness_value(runtime: Any, key: str) -> str:
    readiness = getattr(runtime, "readiness", None)
    if callable(readiness):
        value = readiness()
        if isinstance(value, dict):
            return str(value.get(key, "unknown"))
    return "unknown"


def _stringify_runtime_value(value: Any) -> str:
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


def _write_command_text(text: str) -> None:
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def _returncode_from_verify_text(text: str) -> int:
    for line in text.splitlines():
        if not line.startswith("rc="):
            continue
        value = line.partition("=")[2].strip()
        try:
            return int(value)
        except ValueError:
            return 1
    return 1


def _write_workspace_init_result(result: WorkspaceInitResult) -> None:
    lines = [
        "David init",
        f"workspace: {result.workspace_root}",
        f"config: {result.config_path}",
        f"next steps: {result.next_steps_path}",
    ]
    lines.extend(_format_path_group("created", result.created_paths))
    lines.extend(_format_path_group("skipped", result.existing_paths))
    lines.extend(_format_path_group("overwritten", result.overwritten_paths))
    lines.extend(
        [
            "next actions:",
            f"- review {result.config_path}",
            "- run david model onboard <model> --workspace <workspace> before execution",
            "- add --execute only when you want scanner/validator wrappers to run",
        ]
    )
    sys.stdout.write("\n".join(lines) + "\n")


def _write_model_onboarding_result(result: ModelOnboardingResult) -> None:
    plan = result.plan
    lines = [
        "David model onboarding",
        f"status: {result.status}",
        f"summary: {result.summary}",
        f"model: {plan.model}",
        f"workspace: {plan.workspace_path}",
        f"scan report: {plan.scan_report_path}",
        f"validation report: {plan.validation_report_path}",
        f"execute: {str(result.execute).lower()}",
        "attestation: not created or accepted by this command",
    ]
    if result.scan_result is not None:
        lines.append(f"scan rc: {result.scan_result.returncode}")
    if result.validation_result is not None:
        lines.append(f"validation rc: {result.validation_result.returncode}")
    if result.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in result.errors)
    lines.append("next actions:")
    lines.extend(f"- {action}" for action in result.next_actions)
    sys.stdout.write("\n".join(lines) + "\n")


def _format_path_group(label: str, paths: tuple[Path, ...]) -> list[str]:
    lines = [f"{label} paths:"]
    if not paths:
        lines.append("- none")
        return lines
    lines.extend(f"- {path}" for path in paths)
    return lines


def _write_model_command_result(
    result: ModelCommandResult,
    *,
    success_message: str,
    failure_label: str,
) -> int:
    if result.returncode == 0:
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stdout.write(f"{success_message}\n")
        return 0

    _write_model_command_failure(result, failure_label=failure_label)
    return result.returncode


def _build_config(args: argparse.Namespace, workspace_path: Path) -> Any:
    values = {
        "workspace_path": workspace_path,
        "workspace_root": workspace_path,
        "model_path": args.model,
        "validation_report_path": args.validation_report,
        "model_attestation_path": args.model_attestation,
        "require_validated_model": not args.allow_unvalidated,
        "color": not args.no_color,
        "once": args.once,
        "verify_command": args.verify_command,
        "command_timeout_seconds": args.timeout,
        "auto_jit_index": args.auto_jit_index,
        "model_backend": args.model_backend,
        "model_device": args.model_device,
        "model_dtype": args.model_dtype,
        "model_max_new_tokens": args.model_max_new_tokens,
    }

    from_values = getattr(DavidConfig, "from_values", None)
    if callable(from_values):
        return from_values(
            workspace_path=workspace_path,
            model_path=args.model,
            validation_report_path=args.validation_report,
            require_validated_model=not args.allow_unvalidated,
            color=not args.no_color,
            once=args.once,
            verify_command=args.verify_command,
            command_timeout_seconds=args.timeout,
            auto_jit_index=args.auto_jit_index,
            model_backend=args.model_backend,
            model_device=args.model_device,
            model_dtype=args.model_dtype,
            model_max_new_tokens=args.model_max_new_tokens,
            model_attestation_path=args.model_attestation,
        )

    signature = inspect.signature(DavidConfig)
    kwargs = {
        name: value
        for name, value in values.items()
        if name in signature.parameters
    }
    config = DavidConfig(**kwargs)
    for name in (
        "model_path",
        "validation_report_path",
        "model_attestation_path",
        "require_validated_model",
        "color",
        "once",
        "verify_command",
        "command_timeout_seconds",
        "auto_jit_index",
        "model_backend",
        "model_device",
        "model_dtype",
        "model_max_new_tokens",
    ):
        if not hasattr(config, name):
            object.__setattr__(config, name, values[name])
    return config


@dataclass(frozen=True)
class WorkspaceConfigError:
    path: Path
    message: str


_WORKSPACE_CONFIG_ARG_NAMES = {
    "model",
    "validation_report",
    "model_attestation",
    "model_backend",
    "model_device",
    "model_dtype",
    "model_max_new_tokens",
    "auto_jit_index",
}

_COMMON_OPTION_DESTS = {
    "--model": "model",
    "--validation-report": "validation_report",
    "--model-attestation": "model_attestation",
    "--model-backend": "model_backend",
    "--model-device": "model_device",
    "--model-dtype": "model_dtype",
    "--model-max-new-tokens": "model_max_new_tokens",
    "--auto-jit-index": "auto_jit_index",
}


def _apply_operator_defaults(
    args: argparse.Namespace,
    *,
    explicit_args: set[str] | None = None,
) -> None:
    defaults = _operator_env_defaults()
    for arg_name, value in defaults.items():
        if explicit_args is None:
            should_apply = _arg_is_absent(args, arg_name)
        else:
            should_apply = arg_name not in explicit_args
        if value is not None and should_apply:
            setattr(args, arg_name, value)


def _operator_env_defaults() -> dict[str, str | int | None]:
    return {
        "model": _env_optional_string("DAVID_MODEL"),
        "validation_report": _env_optional_string("DAVID_VALIDATION_REPORT"),
        "model_attestation": _env_optional_string("DAVID_MODEL_ATTESTATION"),
        "model_backend": _env_optional_string("DAVID_MODEL_BACKEND"),
        "model_device": _env_optional_string("DAVID_MODEL_DEVICE"),
        "model_dtype": _env_optional_string("DAVID_MODEL_DTYPE"),
        "model_max_new_tokens": _env_optional_int("DAVID_MODEL_MAX_NEW_TOKENS"),
    }


def _explicit_common_args(argv: list[str]) -> set[str]:
    explicit: set[str] = set()
    options_with_values = {
        "--model",
        "--validation-report",
        "--model-attestation",
        "--model-backend",
        "--model-device",
        "--model-dtype",
        "--model-max-new-tokens",
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        option, has_inline_value = (token.split("=", 1)[0], "=" in token)
        dest = _COMMON_OPTION_DESTS.get(option)
        if dest is not None:
            explicit.add(dest)
            if option in options_with_values and not has_inline_value:
                index += 1
        index += 1
    return explicit


def _load_workspace_config_defaults(workspace_path: Path) -> dict[str, Any] | WorkspaceConfigError:
    config_path = workspace_path.expanduser() / ".david" / "config.json"
    if not config_path.exists():
        return {}
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return WorkspaceConfigError(
            path=config_path,
            message=f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        )
    except OSError as exc:
        return WorkspaceConfigError(path=config_path, message=str(exc))
    if not isinstance(loaded, dict):
        return WorkspaceConfigError(path=config_path, message="config root must be a JSON object")
    return loaded


def _apply_workspace_config_defaults(
    args: argparse.Namespace,
    workspace_path: Path,
    defaults: dict[str, Any],
    explicit_args: set[str],
) -> None:
    for arg_name in _WORKSPACE_CONFIG_ARG_NAMES:
        if arg_name in explicit_args or arg_name not in defaults:
            continue
        value = _coerce_workspace_config_value(arg_name, defaults[arg_name], workspace_path)
        if value is not None:
            setattr(args, arg_name, value)


def _coerce_workspace_config_value(arg_name: str, value: Any, workspace_path: Path) -> str | int | bool | None:
    if arg_name == "auto_jit_index":
        return value if isinstance(value, bool) else None
    if arg_name == "model_max_new_tokens":
        if isinstance(value, bool):
            return None
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value:
        return None
    if arg_name in {"validation_report", "model_attestation"}:
        return str(_resolve_workspace_path(value, workspace_path))
    if arg_name == "model":
        return str(_resolve_workspace_model(value, workspace_path))
    return value


def _resolve_workspace_path(value: str, workspace_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return workspace_path.expanduser() / path


def _resolve_workspace_model(value: str, workspace_path: Path) -> str | Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if value.startswith("."):
        return workspace_path.expanduser() / path
    candidate = workspace_path.expanduser() / path
    if candidate.exists():
        return candidate
    return value


def _write_workspace_config_error(error: WorkspaceConfigError) -> None:
    sys.stderr.write("David workspace config error\n")
    sys.stderr.write(f"config: {error.path}\n")
    sys.stderr.write(f"{error.message}\n")
    sys.stderr.write("Fix or remove .david/config.json before booting a model-backed David session.\n")


def _arg_is_absent(args: argparse.Namespace, name: str) -> bool:
    return not hasattr(args, name) or getattr(args, name) is None


def _env_optional_string(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_optional_int(name: str) -> int | None:
    value = _env_optional_string(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return max(1, parsed)


def _resolve_validation_report(
    args: argparse.Namespace,
    workspace_path: Path,
) -> ValidationReportDiscovery:
    if args.validation_report:
        return ValidationReportDiscovery(path=Path(args.validation_report).expanduser(), checked_paths=())
    return discover_validation_report(model_path=args.model, workspace_path=workspace_path)


def _missing_required_validation_report(
    args: argparse.Namespace,
    discovery: ValidationReportDiscovery,
) -> bool:
    return bool(args.model and not args.allow_unvalidated and discovery.path is None)


def _should_auto_validate_model(
    args: argparse.Namespace,
    discovery: ValidationReportDiscovery,
) -> bool:
    return bool(
        args.model
        and args.auto_validate_model
        and not args.allow_unvalidated
        and discovery.path is None
    )


def _write_auto_validation_failure(result: AutoModelValidationResult) -> None:
    if result.scan_result.returncode != 0:
        _write_model_command_failure(
            result.scan_result,
            failure_label="Auto model scan failed",
        )
        return
    if result.validation_result is None:
        sys.stderr.write("Auto model validation failed: validator did not run\n")
        return
    _write_model_command_failure(
        result.validation_result,
        failure_label="Auto model validation failed",
    )


def _write_missing_validation_report_status(
    args: argparse.Namespace,
    workspace_path: Path,
    discovery: ValidationReportDiscovery,
) -> None:
    checked = "\n".join(f"  - {path}" for path in discovery.checked_paths)
    if not checked:
        checked = "  - no local validation-report paths were applicable"
    sys.stdout.write(
        "\n".join(
            (
                "David startup readiness",
                "- model validation: blocked: no boot-safe validation report found",
                f"- model: {args.model}",
                f"- workspace: {workspace_path.expanduser().resolve()}",
                "- checked validation report paths:",
                checked,
                "Use --validation-report <path>, --auto-validate-model, or --allow-unvalidated to open offline shell mode.",
                "",
            )
        )
    )


def _startup_rejected_validated_model(args: argparse.Namespace, runtime: Any) -> bool:
    return bool(
        args.model
        and not args.allow_unvalidated
        and _runtime_boot_errors(runtime)
    )


def _runtime_boot_errors(runtime: Any) -> tuple[str, ...]:
    errors = getattr(runtime, "boot_errors", ())
    if errors is None:
        return ()
    if isinstance(errors, str):
        return (errors,) if errors else ()
    try:
        return tuple(str(error) for error in errors if str(error))
    except TypeError:
        return (str(errors),) if str(errors) else ()


def _write_rejected_validation_report_status(
    args: argparse.Namespace,
    workspace_path: Path,
    runtime: Any,
) -> None:
    boot_errors = _runtime_boot_errors(runtime)
    error_text = "; ".join(boot_errors) if boot_errors else "validation rejected by harness boot"
    report = args.validation_report or "not supplied"
    sys.stdout.write(
        "\n".join(
            (
                "David startup readiness",
                f"- model validation: blocked: {error_text}",
                f"- model: {args.model}",
                f"- validation report: {report}",
                f"- workspace: {workspace_path.expanduser().resolve()}",
                "Harness boot rejected this validation report, so David did not open the TUI or run any workspace command.",
                "Use --allow-unvalidated to open offline shell mode with model decode disabled.",
                "",
            )
        )
    )


def _add_common_options(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    absent_default: Any = argparse.SUPPRESS if suppress_defaults else None
    absent_false: Any = argparse.SUPPRESS if suppress_defaults else False
    parser.add_argument("--model", default=absent_default, help="Path or HF id for the open model")
    parser.add_argument(
        "--validation-report",
        default=absent_default,
        help="Path to a validator JSON report accepted by the harness boot gate",
    )
    parser.add_argument(
        "--model-attestation",
        default=absent_default,
        help="Path to a manual-reviewed attestation JSON for standard decode of needs-review reports",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        default=absent_false,
        help="Open without a boot-safe validation report. Model decode remains disabled.",
    )
    parser.add_argument("--no-color", action="store_true", default=absent_false, help="Disable ANSI color")
    parser.add_argument("--once", default=absent_default, help="Run one prompt/command and exit")
    parser.add_argument(
        "--verify-command",
        default=absent_default,
        help="Default command used by /verify",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=absent_default,
        help="Local tool command timeout in seconds",
    )
    parser.add_argument(
        "--auto-jit-index",
        action="store_true",
        default=absent_false,
        help="Build a workspace index at startup when one is required and missing",
    )
    parser.add_argument(
        "--auto-validate-model",
        action="store_true",
        default=absent_false,
        help="Run David's standalone scanner and validator into workspace .david artifacts before boot",
    )
    parser.add_argument(
        "--model-backend",
        default=absent_default,
        help="Model backend selector for validated live decode, such as transformers or torch-runtime",
    )
    parser.add_argument(
        "--model-device",
        default=absent_default,
        help="Device requested for validated live model decode, such as auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--model-dtype",
        default=absent_default,
        help="Torch dtype requested for validated live model decode: auto, float16, bfloat16, float32, or none",
    )
    parser.add_argument(
        "--model-max-new-tokens",
        type=int,
        default=absent_default,
        help="Maximum new tokens for validated live model generation",
    )


def _write_model_command_failure(
    result: ModelCommandResult,
    *,
    failure_label: str,
) -> None:
    command_text = " ".join(result.command)
    sys.stderr.write(f"{failure_label} (rc={result.returncode})\n")
    sys.stderr.write(f"command: {command_text}\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")
    if result.stdout:
        sys.stderr.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stderr.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
