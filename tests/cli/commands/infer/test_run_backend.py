"""EWS-1 (Wave 0.5) backend-flag regression tests for the ``infer run`` parser.

Validates that the post-EWS-0 parser registers ``--backend``/``--device`` via
the shared ``add_backend_flags`` helper — the single source of truth mandated
by Epic 2 §EWS-0.  Also pins the precedence contract (flag > CHUK_BACKEND >
auto) via ``InferenceConfig.from_args``.
"""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._infer_run import register_infer_parser
from chuk_lazarus.cli.commands._base import BACKEND_CHOICES
from chuk_lazarus.cli.commands.infer._types import InferenceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_infer_parser(sub)
    return parser


def test_parser_exposes_backend_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["infer", "--model", "m", "--prompt", "hi", "--backend", "torch"])
    assert args.backend == "torch"


def test_parser_exposes_device_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["infer", "--model", "m", "--prompt", "hi", "--device", "cuda:0"]
    )
    assert args.device == "cuda:0"


def test_parser_rejects_unknown_backend() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--model", "m", "--backend", "jax"])


def test_backend_choices_match_shared_helper() -> None:
    parser = _build_parser()
    backend_action = next(
        a for a in parser._subparsers._actions if False  # dummy, replaced below
    ) if False else None
    # Walk the infer subparser's actions directly.
    infer_sub = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    infer_parser = infer_sub.choices["infer"]
    backend_action = next(a for a in infer_parser._actions if a.dest == "backend")
    assert tuple(backend_action.choices) == BACKEND_CHOICES


def test_config_threads_backend_and_device() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--model",
            "m",
            "--prompt",
            "hi",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
        ]
    )
    cfg = InferenceConfig.from_args(args)
    assert cfg.backend == "torch"
    assert cfg.device == "cuda:0"


def test_config_defaults_leave_backend_unset() -> None:
    parser = _build_parser()
    args = parser.parse_args(["infer", "--model", "m", "--prompt", "hi"])
    cfg = InferenceConfig.from_args(args)
    assert cfg.backend is None
    assert cfg.device is None
