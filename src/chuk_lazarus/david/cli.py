"""Command-line entry point for the David terminal coding agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence, TextIO

from .config import DavidRuntimeConfig
from .runtime import DavidRuntime


SUCCESS = 0
RUNTIME_ERROR = 1
KEYBOARD_INTERRUPT = 130


class DavidCliError(RuntimeError):
    """User-facing CLI/runtime error."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="david",
        description="Open the David terminal coding agent.",
        allow_abbrev=False,
    )
    parser.add_argument("command", nargs="?", help="'code', 'doctor', or a workspace path")
    parser.add_argument("workspace_arg", nargs="?", help="Workspace path for 'david code [path]'")
    parser.add_argument("--workspace", default=None, help="Workspace path")
    parser.add_argument("--model", default=None, help="Model identifier or local model path")
    parser.add_argument(
        "--validation-report",
        default=None,
        help="Path to an accepted model validation report",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Open local tool shell mode without loading model weights",
    )
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="Do not require a boot-safe validation report. Decode remains guarded.",
    )
    parser.add_argument("--no-shell", action="store_true", help="Disable shell tool access")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    parser.add_argument(
        "--once",
        nargs="?",
        const="",
        default=None,
        help="Run a single prompt/command and exit",
    )
    parser.add_argument(
        "--verify-command",
        action="append",
        default=None,
        help="Verification command available to the runtime",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    runtime_factory: Callable[[DavidRuntimeConfig, Any], Any] | None = None,
) -> int:
    streams = SimpleNamespace(
        stdin=stdin or sys.stdin,
        stdout=stdout or sys.stdout,
        stderr=stderr or sys.stderr,
    )
    try:
        options = _parse_options(argv)
        _preflight(options)
        factory = runtime_factory or (lambda config, streams=None: DavidRuntime(config, streams=streams))
        runtime = factory(_config_from_options(options), streams)
        session = runtime.initialize()
        if options.command == "doctor":
            _print_doctor(options, session, streams.stdout)
            return SUCCESS
        if options.once is not None:
            if options.once.strip().startswith("/"):
                from .tui import run_tui

                return int(
                    run_tui(
                        runtime=runtime,
                        stdin=streams.stdin,
                        stdout=streams.stdout,
                        stderr=streams.stderr,
                        once_prompt=options.once,
                        use_color=not bool(options.no_color),
                    )
                )
            result = runtime.run_once(options.once)
            _print_doctor(options, session, streams.stdout)
            _print_run_result(result, streams.stdout)
            return int(getattr(result, "exit_code", SUCCESS))

        from .tui import run_tui

        return int(run_tui(runtime=runtime, stdin=streams.stdin, stdout=streams.stdout, stderr=streams.stderr))
    except KeyboardInterrupt:
        print("david: interrupted", file=streams.stderr)
        return KEYBOARD_INTERRUPT
    except SystemExit as exc:
        code = exc.code
        if code in (None, 0):
            return SUCCESS
        print("david: invalid arguments", file=streams.stderr)
        return int(code) if isinstance(code, int) else RUNTIME_ERROR
    except DavidCliError as exc:
        message = str(exc)
        print(f"david: {message}", file=streams.stderr)
        return RUNTIME_ERROR
    except Exception as exc:  # noqa: BLE001 - CLI must return a process code
        print(f"david: {exc}", file=streams.stderr)
        return RUNTIME_ERROR


def _parse_options(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = args.command
    workspace = args.workspace
    if command in (None, ""):
        command = "code"
        workspace = workspace or "."
    elif command == "code":
        workspace = workspace or args.workspace_arg or "."
    elif command == "doctor":
        workspace = workspace or args.workspace_arg or "."
    elif args.workspace_arg is None and args.workspace is None:
        workspace = command
        command = "code"
    else:
        raise DavidCliError(f"unknown command: {command}")

    args.command = command
    args.workspace_path = Path(workspace).expanduser()
    return args


def _preflight(options: argparse.Namespace) -> None:
    workspace = options.workspace_path
    if not workspace.exists():
        raise DavidCliError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise DavidCliError(f"workspace is not a directory: {workspace}")


def _config_from_options(options: argparse.Namespace) -> DavidRuntimeConfig:
    return DavidRuntimeConfig(
        workspace_path=options.workspace_path.resolve(),
        model_path=Path(options.model).expanduser() if options.model else None,
        validation_report_path=(
            Path(options.validation_report).expanduser()
            if options.validation_report
            else None
        ),
        require_validated_model=not bool(options.allow_unvalidated),
        offline=bool(options.offline or options.model is None),
        dry_run=bool(options.offline),
        allow_shell=not bool(options.no_shell),
        verification_commands=tuple(options.verify_command or ()),
    )


def _print_doctor(options: argparse.Namespace, session: Any, stdout: TextIO) -> None:
    workspace = Path(options.workspace_path).resolve()
    print("David readiness", file=stdout)
    print(f"workspace: {workspace}", file=stdout)
    if session is None:
        print("validation_status: offline", file=stdout)
        return
    print(f"validation_status: {getattr(session, 'validation_status', 'unknown')}", file=stdout)
    print(f"session_id: {getattr(session, 'session_id', 'unknown')}", file=stdout)
    print(f"jit_required: {getattr(session, 'jit_required', False)}", file=stdout)
    actions = ", ".join(getattr(session, "jit_actions", ()) or ()) or "none"
    print(f"jit_actions: {actions}", file=stdout)
    index = getattr(session, "index_readiness", None)
    state = getattr(index, "state", None)
    if state is not None:
        print(f"index_state: {getattr(state, 'value', state)}", file=stdout)


def _print_run_result(result: Any, stdout: TextIO) -> None:
    answer = getattr(result, "answer", None)
    if answer:
        print(answer, file=stdout)
    verification = getattr(result, "verification_summary", None)
    if verification:
        print(verification, file=stdout)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
