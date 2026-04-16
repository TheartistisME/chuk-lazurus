"""EWS-8: server.cli wires --backend/--device into ModelEngine.load."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.server.cli import _add_serve_args, _serve_async, add_serve_parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_serve_args(parser)
    return parser


def test_add_serve_args_registers_backend_flags():
    parser = _build_parser()
    dests = {a.dest for a in parser._actions}
    assert "backend" in dests
    assert "device" in dests


def test_add_serve_parser_subcommand_has_backend():
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command")
    add_serve_parser(sub)
    ns = root.parse_args(
        ["serve", "--model", "m", "--backend", "torch", "--device", "cuda:0"]
    )
    assert ns.backend == "torch"
    assert ns.device == "cuda:0"


def test_backend_choices_enforced():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", "--backend", "jax"])


def test_serve_async_threads_backend_into_engine_load():
    parser = _build_parser()
    ns = parser.parse_args(
        ["--model", "fake/m", "--backend", "torch", "--device", "cuda:0", "--port", "0"]
    )

    fake_engine = MagicMock()
    fake_app = MagicMock()

    fake_server = MagicMock()
    fake_server.serve = AsyncMock()

    with (
        patch(
            "chuk_lazarus.server.engine.ModelEngine.load",
            new=AsyncMock(return_value=fake_engine),
        ) as load_mock,
        patch("chuk_lazarus.server.app.create_app", return_value=fake_app),
        patch("uvicorn.Config"),
        patch("uvicorn.Server", return_value=fake_server),
    ):
        import asyncio

        asyncio.run(_serve_async(ns))

    load_mock.assert_awaited_once()
    _, kwargs = load_mock.call_args
    assert kwargs["backend"] == "torch"
    assert kwargs["device"] == "cuda:0"
