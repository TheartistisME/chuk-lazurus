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
        [
            "serve",
            "--model",
            "m",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
            "--engine",
            "bounded_kv_direct",
            "--hot-budget-mib",
            "1024",
        ]
    )
    assert ns.backend == "torch"
    assert ns.device == "cuda:0"
    assert ns.engine == "bounded_kv_direct"
    assert ns.hot_budget_mib == 1024


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


def test_serve_subcommand_accepts_residual_bounded_kv_direct():
    """Qwen residual-backed bounded KV-direct must be a first-class --engine choice."""
    parser = _root_with_serve()
    ns = parser.parse_args(
        [
            "serve",
            "--model",
            "m",
            "--backend",
            "torch",
            "--engine",
            "residual_bounded_kv_direct",
            "--hot-budget-mib",
            "150",
        ]
    )
    assert ns.engine == "residual_bounded_kv_direct"
    assert ns.hot_budget_mib == 150


def test_serve_subcommand_accepts_session_cache_size():
    """``--session-cache-size`` sizes the WARM residual session cache."""
    parser = _root_with_serve()
    ns = parser.parse_args(
        [
            "serve",
            "--model",
            "m",
            "--engine",
            "residual_bounded_kv_direct",
            "--session-cache-size",
            "12",
        ]
    )
    assert ns.session_cache_size == 12
