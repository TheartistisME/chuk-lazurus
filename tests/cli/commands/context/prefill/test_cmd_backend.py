"""EWS-2: ``context prefill`` command propagates backend/device to env
and bails cleanly on non-MLX backends (current engine is MLX-only)."""

from __future__ import annotations

import asyncio
import os
from argparse import Namespace

import pytest


def _make_args(**overrides) -> Namespace:
    base = dict(
        model="m",
        input="/nonexistent/input.txt",
        checkpoint="/tmp/ckpt",
        window_size=None,
        max_tokens=None,
        no_resume=True,
        name=None,
        residual_mode="interval",
        frame_bank=None,
        store_pages=False,
        store_kv_full=False,
        phases="windows",
        compass_layer=None,
        mode="standard",
        backend=None,
        device=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_cmd_exports_backend_and_device_env(monkeypatch, capsys):
    from chuk_lazarus.cli.commands.context.prefill._cmd import context_prefill_cmd

    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    class _StubBackend:
        name = "torch"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )

    asyncio.run(context_prefill_cmd(_make_args(backend="torch", device="cuda")))
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda"
    captured = capsys.readouterr()
    assert "MLX backend only" in captured.err


def test_cmd_no_override_when_flags_absent(monkeypatch):
    from chuk_lazarus.cli.commands.context.prefill._cmd import context_prefill_cmd

    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    class _StubBackend:
        name = "torch"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )

    asyncio.run(context_prefill_cmd(_make_args()))
    assert "CHUK_BACKEND" not in os.environ
    assert "CHUK_DEVICE" not in os.environ


@pytest.mark.parametrize(
    "module",
    [
        "chuk_lazarus.cli.commands.context.prefill._cmd",
        "chuk_lazarus.cli.commands.context.prefill._sparse",
        "chuk_lazarus.cli.commands.context.prefill._surprise",
        "chuk_lazarus.cli.commands.context.prefill._compass",
        "chuk_lazarus.cli.commands.context.prefill._darkspace",
        "chuk_lazarus.cli.commands.context.prefill._interval",
        "chuk_lazarus.cli.commands.context.prefill._kv_route",
        "chuk_lazarus.cli.commands.context.prefill._pages",
        "chuk_lazarus.cli.commands.context.prefill._save",
        "chuk_lazarus.cli.commands.context.prefill._vec_inject",
    ],
)
def test_prefill_modules_importable(module):
    import importlib

    importlib.import_module(module)
