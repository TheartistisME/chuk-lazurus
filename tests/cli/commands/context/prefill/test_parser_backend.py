"""EWS-2: context prefill parser honours --backend / --device."""

from __future__ import annotations

import argparse

import pytest

from chuk_lazarus.cli._parsers._context_prefill import register_context_prefill_parser


@pytest.fixture
def prefill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ctx_sub = parser.add_subparsers(dest="ctx_cmd")
    register_context_prefill_parser(ctx_sub)
    return parser


BASE_ARGS = [
    "prefill",
    "--model",
    "m",
    "--input",
    "/tmp/in.txt",
    "--checkpoint",
    "/tmp/out",
]


def test_backend_flag_accepts_torch(prefill_parser):
    args = prefill_parser.parse_args(BASE_ARGS + ["--backend", "torch"])
    assert args.backend == "torch"


def test_backend_flag_accepts_mlx(prefill_parser):
    args = prefill_parser.parse_args(BASE_ARGS + ["--backend", "mlx"])
    assert args.backend == "mlx"


def test_backend_rejects_unknown(prefill_parser):
    with pytest.raises(SystemExit):
        prefill_parser.parse_args(BASE_ARGS + ["--backend", "jax"])


def test_device_flag(prefill_parser):
    args = prefill_parser.parse_args(BASE_ARGS + ["--device", "cuda:0"])
    assert args.device == "cuda:0"


def test_backend_defaults_to_none(prefill_parser):
    args = prefill_parser.parse_args(BASE_ARGS)
    assert args.backend is None
    assert args.device is None


def test_prefill_config_threads_backend_device():
    from chuk_lazarus.cli.commands.context._types import PrefillConfig

    ns = argparse.Namespace(
        model="m",
        input="/tmp/in.txt",
        checkpoint="/tmp/out",
        window_size=None,
        max_tokens=None,
        no_resume=False,
        name=None,
        residual_mode="interval",
        frame_bank=None,
        store_pages=False,
        store_kv_full=False,
        phases="all",
        compass_layer=None,
        mode="standard",
        backend="torch",
        device="cuda",
    )
    cfg = PrefillConfig.from_args(ns)
    assert cfg.backend == "torch"
    assert cfg.device == "cuda"
