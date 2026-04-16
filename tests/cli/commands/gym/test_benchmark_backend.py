"""EWS-14 backend-wiring regression for ``gym benchmark``.

The bench subparser is registered by EWS-15 in ``_parsers/_bench.py`` — this
test drives the ``BenchmarkConfig`` / handler side owned by EWS-14.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from chuk_lazarus.cli.commands.gym._types import BenchmarkConfig


def test_benchmark_config_accepts_backend_and_device() -> None:
    ns = argparse.Namespace(backend="torch", device="cuda:0")
    cfg = BenchmarkConfig.from_args(ns)
    assert cfg.backend == "torch"
    assert cfg.device == "cuda:0"


def test_benchmark_config_defaults_to_none() -> None:
    ns = argparse.Namespace()
    cfg = BenchmarkConfig.from_args(ns)
    assert cfg.backend is None
    assert cfg.device is None


def test_benchmark_module_has_no_top_level_mlx() -> None:
    for mod in list(sys.modules):
        if mod.startswith("mlx"):
            sys.modules.pop(mod, None)
    sys.modules.pop("chuk_lazarus.cli.commands.gym.benchmark", None)
    importlib.import_module("chuk_lazarus.cli.commands.gym.benchmark")
    assert not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules)
