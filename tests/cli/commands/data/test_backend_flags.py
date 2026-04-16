"""EWS-12: Backend-flag wiring and lazy-import assertions for data bucket.

Verifies that every owned data subcommand accepts ``--backend``/``--device``
via ``add_backend_flags`` and that importing the owned modules under
``CHUK_BACKEND=torch`` does not pull MLX at module-import time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest

from chuk_lazarus.cli._parsers._data import register_data_parsers


OWNED_SUBCOMMANDS: list[tuple[str, ...]] = [
    ("data", "lengths", "build"),
    ("data", "lengths", "stats"),
    ("data", "batchplan", "build"),
    ("data", "batchplan", "info"),
    ("data", "batchplan", "verify"),
    ("data", "batchplan", "shard"),
    ("data", "batching", "analyze"),
    ("data", "batching", "histogram"),
    ("data", "batching", "suggest"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chuk-lazarus")
    sub = parser.add_subparsers(dest="command")
    register_data_parsers(sub)
    return parser


@pytest.mark.parametrize("path", OWNED_SUBCOMMANDS)
def test_backend_flags_registered(path: tuple[str, ...]) -> None:
    """Each owned subcommand must accept --backend and --device."""
    parser = _build_parser()
    # Build the minimum required argv, filling required flags with dummies.
    required_dummies = {
        "--dataset": "d", "--tokenizer": "t", "--output": "o",
        "--cache": "c", "--lengths": "l", "--plan": "p",
        "--world-size": "2",
    }
    # Use a targeted argv per command:
    argv_map = {
        ("data", "lengths", "build"): ["--dataset", "d", "--tokenizer", "t", "--output", "o"],
        ("data", "lengths", "stats"): ["--cache", "c"],
        ("data", "batchplan", "build"): ["--lengths", "l", "--output", "o"],
        ("data", "batchplan", "info"): ["--plan", "p"],
        ("data", "batchplan", "verify"): ["--plan", "p", "--lengths", "l"],
        ("data", "batchplan", "shard"): ["--plan", "p", "--world-size", "2", "--output", "o"],
        ("data", "batching", "analyze"): ["--cache", "c"],
        ("data", "batching", "histogram"): ["--cache", "c"],
        ("data", "batching", "suggest"): ["--cache", "c"],
    }
    extra = argv_map[path]
    argv = list(path) + extra + ["--backend", "torch", "--device", "cpu"]
    args = parser.parse_args(argv)
    assert args.backend == "torch"
    assert args.device == "cpu"


OWNED_MODULES: list[str] = [
    "chuk_lazarus.cli._parsers._data",
    "chuk_lazarus.cli.commands.data",
    "chuk_lazarus.cli.commands.data.batchplan",
    "chuk_lazarus.cli.commands.data.batchplan.build",
    "chuk_lazarus.cli.commands.data.batchplan.info",
    "chuk_lazarus.cli.commands.data.batchplan.shard",
    "chuk_lazarus.cli.commands.data.batchplan.verify",
    "chuk_lazarus.cli.commands.data.batching.analyze",
    "chuk_lazarus.cli.commands.data.batching.histogram",
    "chuk_lazarus.cli.commands.data.batching.suggest",
    "chuk_lazarus.cli.commands.data.lengths.build",
    "chuk_lazarus.cli.commands.data.lengths.stats",
    "chuk_lazarus.data.batching",
    "chuk_lazarus.data.samples.schema",
]


@pytest.mark.parametrize("module", OWNED_MODULES)
def test_import_under_torch_does_not_pull_mlx(module: str) -> None:
    """Lazy-import assertion: CHUK_BACKEND=torch must not load MLX."""
    code = (
        "import sys\n"
        f"import {module}  # noqa: F401\n"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if m.startswith('mlx'))\n"
        "assert 'mlx_lm' not in sys.modules\n"
    )
    repo_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
    )
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([repo_src, existing_pp]) if existing_pp else repo_src
    env = {**os.environ, "CHUK_BACKEND": "torch", "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
