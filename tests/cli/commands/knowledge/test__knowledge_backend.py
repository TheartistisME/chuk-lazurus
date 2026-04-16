"""EWS-4 backend-flag regression for ``knowledge {build,query,chat}``.

Asserts the shared ``add_backend_flags`` helper (EWS-0 single source of truth)
is wired onto every knowledge subparser with the Epic 2 flag contract.
"""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._knowledge import register_knowledge_parsers
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_knowledge_parsers(sub)
    return parser


def _subparser(parser: argparse.ArgumentParser, *path: str) -> argparse.ArgumentParser:
    current = parser
    for name in path:
        subs = next(a for a in current._actions if isinstance(a, argparse._SubParsersAction))
        current = subs.choices[name]
    return current


@pytest.mark.parametrize("sub", ["build", "query", "chat"])
def test_backend_flag_registered(sub: str) -> None:
    parser = _build_parser()
    kn_sub = _subparser(parser, "knowledge", sub)
    action = next(a for a in kn_sub._actions if a.dest == "backend")
    assert tuple(action.choices) == BACKEND_CHOICES


@pytest.mark.parametrize("sub", ["build", "query", "chat"])
def test_device_flag_registered(sub: str) -> None:
    parser = _build_parser()
    kn_sub = _subparser(parser, "knowledge", sub)
    assert any(a.dest == "device" for a in kn_sub._actions)


def test_build_accepts_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "knowledge",
            "build",
            "--model",
            "m",
            "--input",
            "in.txt",
            "--output",
            "out",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
        ]
    )
    assert args.backend == "torch"
    assert args.device == "cuda:0"


def test_query_accepts_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "knowledge",
            "query",
            "--model",
            "m",
            "--prompt",
            "hi",
            "--backend",
            "mlx",
            "--device",
            "mps",
        ]
    )
    assert args.backend == "mlx"
    assert args.device == "mps"


def test_chat_accepts_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "knowledge",
            "chat",
            "--model",
            "m",
            "--store",
            "store-dir",
            "--backend",
            "torch",
            "--device",
            "cuda",
        ]
    )
    assert args.backend == "torch"
    assert args.device == "cuda"


@pytest.mark.parametrize("sub", ["build", "query", "chat"])
def test_rejects_unknown_backend(sub: str) -> None:
    parser = _build_parser()
    argv = ["knowledge", sub, "--model", "m", "--backend", "jax"]
    if sub == "build":
        argv += ["--input", "x", "--output", "y"]
    elif sub == "query":
        argv += ["--prompt", "hi"]
    elif sub == "chat":
        argv += ["--store", "s"]
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_backend_defaults_to_none() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["knowledge", "build", "--model", "m", "--input", "in.txt", "--output", "out"]
    )
    assert args.backend is None
    assert args.device is None
