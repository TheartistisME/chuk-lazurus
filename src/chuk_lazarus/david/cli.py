"""Command-line entry point for the David terminal agent."""

from __future__ import annotations

import argparse
import inspect
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
from .tui import DavidTui

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
    require_validated_model: bool = True
    color: bool = True
    once: str | None = None
    verify_command: str | None = None
    command_timeout_seconds: int | None = None
    auto_jit_index: bool = False

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
    ) -> "_FallbackDavidConfig":
        return cls(
            workspace_path=workspace_path.expanduser().resolve(),
            model_path=model_path,
            validation_report_path=validation_report_path,
            require_validated_model=require_validated_model,
            color=color,
            once=once,
            verify_command=verify_command,
            command_timeout_seconds=command_timeout_seconds,
            auto_jit_index=auto_jit_index,
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
    code = subparsers.add_parser("code", help="Open David in a workspace")
    code.add_argument("workspace", nargs="?", default=".", help="Workspace path")
    _add_common_options(code, suppress_defaults=True)
    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect local model boot readiness without downloads or model load",
    )
    doctor.add_argument("--workspace", default=".", help="Workspace path")
    doctor.add_argument("--model", default=None, help="Path or HF id for the open model")
    doctor.add_argument(
        "--validation-report",
        default=None,
        help="Path to a validator JSON report accepted by the harness boot gate",
    )
    doctor.add_argument(
        "--auto-validate-model",
        action="store_true",
        help="Report whether David can run the standalone scanner and validator",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "doctor":
        return _run_doctor_command(args)
    if getattr(args, "command", None) == "model":
        return _run_model_command(args)

    workspace = getattr(args, "workspace", ".")
    if getattr(args, "command", None) not in (None, "code"):
        parser.error(f"unknown command: {args.command}")

    workspace_path = Path(workspace)
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

    raise AssertionError(f"Unhandled model command: {args.model_command}")


def _run_doctor_command(args: argparse.Namespace) -> int:
    report = run_doctor(
        model=args.model,
        workspace_path=Path(args.workspace),
        validation_report=args.validation_report,
        auto_validate_model=args.auto_validate_model,
    )
    sys.stdout.write(format_doctor_report(report))
    return 0 if report.ready else 2


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
        "require_validated_model": not args.allow_unvalidated,
        "color": not args.no_color,
        "once": args.once,
        "verify_command": args.verify_command,
        "command_timeout_seconds": args.timeout,
        "auto_jit_index": args.auto_jit_index,
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
        "require_validated_model",
        "color",
        "once",
        "verify_command",
        "command_timeout_seconds",
        "auto_jit_index",
    ):
        if not hasattr(config, name):
            object.__setattr__(config, name, values[name])
    return config


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
