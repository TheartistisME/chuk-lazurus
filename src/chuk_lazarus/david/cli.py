"""Command-line entry point for the David terminal agent."""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    _add_common_options(code)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace = getattr(args, "workspace", ".")
    if getattr(args, "command", None) not in (None, "code"):
        parser.error(f"unknown command: {args.command}")

    config = _build_config(args, Path(workspace))
    runtime = DavidRuntime.create(config)
    tui = DavidTui(runtime, color=config.color)
    return tui.run(once=config.once)


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
        "auto_jit_index": False,
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
        )

    signature = inspect.signature(DavidConfig)
    kwargs = {
        name: value
        for name, value in values.items()
        if name in signature.parameters
    }
    config = DavidConfig(**kwargs)
    for name in ("color", "once", "verify_command", "command_timeout_seconds"):
        if not hasattr(config, name):
            object.__setattr__(config, name, values[name])
    return config


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Path or HF id for the open model")
    parser.add_argument(
        "--validation-report",
        default=None,
        help="Path to a validator JSON report accepted by the harness boot gate",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Open without a boot-safe validation report. Model decode remains disabled.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color")
    parser.add_argument("--once", default=None, help="Run one prompt/command and exit")
    parser.add_argument(
        "--verify-command",
        default=None,
        help="Default command used by /verify",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Local tool command timeout in seconds",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
