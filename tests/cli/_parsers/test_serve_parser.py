"""EWS-8: serve parser exposes ``--backend`` / ``--device`` on the subcommand."""

from __future__ import annotations

import argparse

from chuk_lazarus.cli._parsers._serve import register_serve_parser


def _root_with_serve() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register_serve_parser(sub)
    return parser


def test_serve_subcommand_accepts_backend_and_device():
    parser = _root_with_serve()
    ns = parser.parse_args(
        ["serve", "--model", "m", "--backend", "torch", "--device", "cuda:0"]
    )
    assert ns.backend == "torch"
    assert ns.device == "cuda:0"


def test_serve_subcommand_accepts_mlx_backend():
    parser = _root_with_serve()
    ns = parser.parse_args(["serve", "--model", "m", "--backend", "mlx"])
    assert ns.backend == "mlx"
    assert ns.device is None


def test_serve_subcommand_rejects_unknown_backend():
    parser = _root_with_serve()
    import pytest

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model", "m", "--backend", "tpu"])
