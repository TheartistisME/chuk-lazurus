"""EWS-14 backend-flag regression for ``gym info``."""

from __future__ import annotations

import argparse
import importlib
import sys

from chuk_lazarus.cli._parsers._gym import register_gym_parsers
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES
from chuk_lazarus.cli.commands.gym.info import gym_info, gym_info_cmd


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


def test_info_backend_flag_registered() -> None:
    parser = _build_parser()
    info = _subparser(parser, "gym", "info")
    action = next(a for a in info._actions if a.dest == "backend")
    assert tuple(action.choices) == BACKEND_CHOICES


def test_info_device_flag_registered() -> None:
    parser = _build_parser()
    info = _subparser(parser, "gym", "info")
    assert any(a.dest == "device" for a in info._actions)


def test_info_accepts_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(["gym", "info", "--backend", "torch", "--device", "cuda"])
    assert args.backend == "torch"
    assert args.device == "cuda"


def test_info_signature_accepts_backend_kwargs() -> None:
    """gym_info must accept backend/device kwargs for CLI forwarding."""
    import inspect

    sig = inspect.signature(gym_info)
    assert "backend" in sig.parameters
    assert "device" in sig.parameters
    # Also verify the cmd-level entrypoint pulls both off args
    cmd_src = inspect.getsource(gym_info_cmd)
    assert "backend" in cmd_src and "device" in cmd_src


def test_info_module_has_no_top_level_mlx() -> None:
    for mod in list(sys.modules):
        if mod.startswith("mlx"):
            sys.modules.pop(mod, None)
    sys.modules.pop("chuk_lazarus.cli.commands.gym.info", None)
    importlib.import_module("chuk_lazarus.cli.commands.gym.info")
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
