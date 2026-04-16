"""EWS-15 — ``bench`` CLI accepts ``--backend`` / ``--device``.

EWS-15 owns only ``cli/_parsers/_bench.py``; the handler lives in
``cli/commands/gym/benchmark.py`` (EWS-14).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

for name in ("mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx.optimizers"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["mlx.nn"].Module = MagicMock
sys.modules["mlx.nn"].Linear = MagicMock

import argparse  # noqa: E402

import pytest  # noqa: E402

from chuk_lazarus.cli._parsers._bench import register_bench_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    root = argparse.ArgumentParser(prog="lazarus")
    subparsers = root.add_subparsers(dest="command")
    register_bench_parser(subparsers)
    return root.parse_args(argv)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_bench_accepts_backend_flag(backend):
    ns = _parse(["bench", "--backend", backend])
    assert ns.backend == backend


def test_bench_accepts_device_flag():
    ns = _parse(["bench", "--backend", "torch", "--device", "cuda:0"])
    assert ns.device == "cuda:0"


def test_bench_rejects_unknown_backend():
    with pytest.raises(SystemExit):
        _parse(["bench", "--backend", "jax"])


def test_bench_backend_defaults_to_none():
    ns = _parse(["bench"])
    assert ns.backend is None
    assert ns.device is None
