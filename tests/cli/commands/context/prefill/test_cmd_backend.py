"""Backend regression tests for the maintained torch prefill sidecar path."""

from __future__ import annotations

import asyncio
import os
from argparse import Namespace
from types import SimpleNamespace

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


def test_cmd_exports_backend_and_device_env(monkeypatch, capsys, tmp_path):
    from chuk_lazarus.cli.commands.context.prefill._cmd import context_prefill_cmd

    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    input_path = tmp_path / "apollo.txt"
    input_path.write_text("Apollo launch transcript", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"

    class _StubBackend:
        name = "torch"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )
    seen: dict[str, object] = {}

    def _fake_run_torch_prefill(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(num_tokens=7, num_windows=1, reused=False)

    monkeypatch.setattr(
        "chuk_lazarus.cli.commands.context.prefill._torch_sidecar.run_torch_prefill_sidecar",
        _fake_run_torch_prefill,
    )

    asyncio.run(
        context_prefill_cmd(
            _make_args(
                input=str(input_path),
                checkpoint=str(checkpoint),
                backend="torch",
                device="cuda",
            )
        )
    )
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda"
    assert seen["requested_device"] == "cuda"
    assert seen["checkpoint_path"] == checkpoint
    assert seen["raw_text"] == "Apollo launch transcript"
    assert seen["name"] == "Apollo"
    assert seen["requested_phases"] == ["windows"]
    assert seen["resume"] is False
    captured = capsys.readouterr()
    assert "torch sidecar contract only" in captured.err
    assert "Prefill Complete" in captured.out


def test_cmd_no_override_when_flags_absent(monkeypatch, tmp_path):
    from chuk_lazarus.cli.commands.context.prefill._cmd import context_prefill_cmd

    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    input_path = tmp_path / "apollo.txt"
    input_path.write_text("Apollo launch transcript", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"

    class _StubBackend:
        name = "torch"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )
    seen: dict[str, object] = {}

    def _fake_run_torch_prefill(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(num_tokens=5, num_windows=1, reused=True)

    monkeypatch.setattr(
        "chuk_lazarus.cli.commands.context.prefill._torch_sidecar.run_torch_prefill_sidecar",
        _fake_run_torch_prefill,
    )

    asyncio.run(
        context_prefill_cmd(
            _make_args(
                input=str(input_path),
                checkpoint=str(checkpoint),
            )
        )
    )
    assert "CHUK_BACKEND" not in os.environ
    assert "CHUK_DEVICE" not in os.environ
    assert seen["requested_device"] is None
    assert seen["name"] == "Apollo"


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
        "chuk_lazarus.cli.commands.context.prefill._torch_sidecar",
        "chuk_lazarus.cli.commands.context.prefill._vec_inject",
    ],
)
def test_prefill_modules_importable(module):
    import importlib

    importlib.import_module(module)
