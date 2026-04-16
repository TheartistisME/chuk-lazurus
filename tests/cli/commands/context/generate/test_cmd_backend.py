"""Shared backend regression coverage for context generate + calibrate.

Exercises the real argparse registration for the shared backend flags while
keeping command-module checks at the AST level so the file stays independent of
runtime ``mlx`` availability.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

from chuk_lazarus.cli._parsers._context_generate import (
    register_context_calibrate_parser,
    register_context_generate_parser,
)
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES

REPO_ROOT = Path(__file__).resolve().parents[5]
GENERATE_CMD_PATH = REPO_ROOT / "src/chuk_lazarus/cli/commands/context/generate/_cmd.py"
CALIBRATE_CMD_PATH = REPO_ROOT / "src/chuk_lazarus/cli/commands/context/calibrate_frames.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus context")
    subparsers = parser.add_subparsers(dest="command")
    register_context_generate_parser(subparsers)
    register_context_calibrate_parser(subparsers)
    return parser


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    return subparsers.choices[name]


def _ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize(
    ("subcommand", "extra_argv"),
    [
        ("generate", ["--prompt", "hi"]),
        ("calibrate-frames", ["--output", "frames.npz"]),
    ],
)
def test_context_subparsers_accept_backend_and_device(
    subcommand: str,
    extra_argv: list[str],
) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [subcommand, "--model", "m", *extra_argv, "--backend", "torch", "--device", "cuda:0"]
    )
    assert args.backend == "torch"
    assert args.device == "cuda:0"
    assert callable(args.func)


@pytest.mark.parametrize("subcommand", ["generate", "calibrate-frames"])
def test_context_subparsers_backend_flags_match_shared_helper(subcommand: str) -> None:
    parser = _build_parser()
    subparser = _subparser(parser, subcommand)
    backend_action = next(action for action in subparser._actions if action.dest == "backend")
    assert tuple(backend_action.choices) == BACKEND_CHOICES
    assert any(action.dest == "device" for action in subparser._actions)


@pytest.mark.parametrize(
    ("subcommand", "extra_argv"),
    [
        ("generate", ["--prompt", "hi"]),
        ("calibrate-frames", ["--output", "frames.npz"]),
    ],
)
def test_context_subparsers_reject_unknown_backend(
    subcommand: str,
    extra_argv: list[str],
) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([subcommand, "--model", "m", *extra_argv, "--backend", "jax"])


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (GENERATE_CMD_PATH, "generate"),
        (CALIBRATE_CMD_PATH, "calibrate-frames"),
    ],
)
def test_context_command_modules_avoid_top_level_mlx_imports(
    path: Path,
    label: str,
) -> None:
    imports = _top_level_imports(_ast(path))
    assert all(not name.startswith("mlx") for name in imports), (
        f"{label} must keep mlx imports out of module scope"
    )
