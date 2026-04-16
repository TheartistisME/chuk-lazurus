"""WS-6: CLI wiring tests for --backend / --device flags on `infer`."""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._infer import register_infer_parser
from chuk_lazarus.cli.commands.infer._types import InferenceConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    subparsers = parser.add_subparsers(dest="command")
    register_infer_parser(subparsers)
    return parser


def test_backend_and_device_flags_propagate_to_config():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--model",
            "test-model",
            "--prompt",
            "hi",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
        ]
    )
    assert args.backend == "torch"
    assert args.device == "cuda:0"

    config = InferenceConfig.from_args(args)
    assert config.backend == "torch"
    assert config.device == "cuda:0"


def test_backend_and_device_default_to_none():
    parser = _build_parser()
    args = parser.parse_args(
        ["infer", "--model", "test-model", "--prompt", "hi"]
    )
    assert args.backend is None
    assert args.device is None

    config = InferenceConfig.from_args(args)
    assert config.backend is None
    assert config.device is None


def test_backend_rejects_legacy_cuda_choice():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["infer", "--model", "m", "--prompt", "p", "--backend", "cuda"]
        )


def test_backend_accepts_mlx():
    parser = _build_parser()
    args = parser.parse_args(
        ["infer", "--model", "m", "--prompt", "p", "--backend", "mlx"]
    )
    config = InferenceConfig.from_args(args)
    assert config.backend == "mlx"
    assert config.device is None
