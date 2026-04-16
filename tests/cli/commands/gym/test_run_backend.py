"""EWS-14 backend-flag regression for ``gym run``.

Asserts the shared ``add_backend_flags`` helper (EWS-0 single source of truth)
is wired onto the gym run subparser with the Epic 2 flag contract, and that
``GymRunConfig`` surfaces the resolved backend/device fields.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import pytest

from chuk_lazarus.cli._parsers._gym import register_gym_parsers
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES
from chuk_lazarus.cli.commands.gym._types import GymRunConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_gym_parsers(sub)
    return parser


def _subparser(parser: argparse.ArgumentParser, *path: str) -> argparse.ArgumentParser:
    current = parser
    for name in path:
        subs = next(a for a in current._actions if isinstance(a, argparse._SubParsersAction))
        current = subs.choices[name]
    return current


def test_backend_flag_registered_on_run() -> None:
    parser = _build_parser()
    run = _subparser(parser, "gym", "run")
    action = next(a for a in run._actions if a.dest == "backend")
    assert tuple(action.choices) == BACKEND_CHOICES


def test_device_flag_registered_on_run() -> None:
    parser = _build_parser()
    run = _subparser(parser, "gym", "run")
    assert any(a.dest == "device" for a in run._actions)


def test_run_accepts_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "gym",
            "run",
            "--tokenizer",
            "gpt2",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
        ]
    )
    assert args.backend == "torch"
    assert args.device == "cuda:0"


def test_run_rejects_unknown_backend() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["gym", "run", "--tokenizer", "gpt2", "--backend", "jax"])


def test_run_backend_defaults_to_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(["gym", "run", "--tokenizer", "gpt2"])
    assert args.backend is None
    assert args.device is None


def test_gym_run_config_from_args_includes_backend() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "gym",
            "run",
            "--tokenizer",
            "gpt2",
            "--backend",
            "mlx",
            "--device",
            "mps",
        ]
    )
    cfg = GymRunConfig.from_args(args)
    assert cfg.backend == "mlx"
    assert cfg.device == "mps"


def test_run_module_has_no_top_level_mlx() -> None:
    """EWS-14: gym.run must not import mlx at top level."""
    for mod in list(sys.modules):
        if mod.startswith("mlx"):
            sys.modules.pop(mod, None)
    sys.modules.pop("chuk_lazarus.cli.commands.gym.run", None)
    importlib.import_module("chuk_lazarus.cli.commands.gym.run")
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
