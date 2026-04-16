"""Tests for the shared ``add_backend_flags`` helper (EWS-0)."""

from __future__ import annotations

import argparse
import os

import pytest

from chuk_lazarus.cli.commands._base import BACKEND_CHOICES, add_backend_flags
from chuk_lazarus.cli.main import _propagate_top_level_backend_flags, app


def _fresh_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def test_add_backend_flags_registers_both_flags():
    parser = _fresh_parser()
    add_backend_flags(parser)
    ns = parser.parse_args(["--backend", "torch", "--device", "cuda:0"])
    assert ns.backend == "torch"
    assert ns.device == "cuda:0"


def test_add_backend_flags_defaults_to_none():
    parser = _fresh_parser()
    add_backend_flags(parser)
    ns = parser.parse_args([])
    assert ns.backend is None
    assert ns.device is None


def test_add_backend_flags_rejects_unknown_backend():
    parser = _fresh_parser()
    add_backend_flags(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "jax"])


def test_add_backend_flags_is_idempotent():
    parser = _fresh_parser()
    add_backend_flags(parser)
    # Second call must not raise argparse.ArgumentError.
    add_backend_flags(parser)
    ns = parser.parse_args(["--backend", "mlx"])
    assert ns.backend == "mlx"


def test_backend_choices_matches_flag_choices():
    parser = _fresh_parser()
    add_backend_flags(parser)
    backend_action = next(a for a in parser._actions if a.dest == "backend")
    assert tuple(backend_action.choices) == BACKEND_CHOICES


def test_top_level_flags_on_root_parser():
    parser = app()
    # --help must not crash; parse the flags alone (no subcommand) and check
    # they land on the dedicated dest names.
    ns, _ = parser.parse_known_args(["--backend", "torch", "--device", "cuda:0"])
    assert ns._top_backend == "torch"
    assert ns._top_device == "cuda:0"


def test_propagate_top_level_sets_env(monkeypatch):
    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)
    ns = argparse.Namespace(_top_backend="torch", _top_device="cuda:0")
    _propagate_top_level_backend_flags(ns)
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda:0"


def test_propagate_top_level_does_not_overwrite(monkeypatch):
    monkeypatch.setenv("CHUK_BACKEND", "mlx")
    monkeypatch.setenv("CHUK_DEVICE", "mps")
    ns = argparse.Namespace(_top_backend="torch", _top_device="cuda:0")
    _propagate_top_level_backend_flags(ns)
    assert os.environ["CHUK_BACKEND"] == "mlx"
    assert os.environ["CHUK_DEVICE"] == "mps"
