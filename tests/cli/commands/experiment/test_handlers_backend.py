"""EWS-15 — backend-flag plumbing for the ``experiment`` CLI."""

from __future__ import annotations

# --- stub mlx for hosts where libmlx.so is missing (pre-existing Epic 1 ---
# eager import at ``models_v2/adapters/lora.py:15``; out of EWS-15 scope).
import sys
import types
from unittest.mock import MagicMock

for name in ("mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx.optimizers"):
    mod = types.ModuleType(name)
    sys.modules.setdefault(name, mod)
sys.modules["mlx.nn"].Module = MagicMock
sys.modules["mlx.nn"].Linear = MagicMock
# ------------------------------------------------------------------------

import ast  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

import pytest  # noqa: E402

from chuk_lazarus.cli.commands.experiment import handlers  # noqa: E402
from chuk_lazarus.experiments.base import ExperimentResult  # noqa: E402


@pytest.fixture
def _result():
    return ExperimentResult(
        experiment_name="demo",
        status="success",
        started_at="2024-01-01T00:00:00",
        finished_at="2024-01-01T00:00:01",
        duration_seconds=1.0,
        run_results={},
        eval_results={},
        config={},
        error=None,
    )


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_experiment_run_forwards_backend(_result, backend):
    with patch(
        "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
        return_value=_result,
    ) as mock_run:
        handlers.experiment_run(
            "demo",
            experiments_dir="/tmp/exp",
            backend=backend,
            device="cuda:0" if backend == "torch" else None,
        )
    kwargs = mock_run.call_args[1]
    assert kwargs["backend"] == backend
    assert kwargs["device"] == ("cuda:0" if backend == "torch" else None)


def test_experiment_run_defaults_backend_none(_result):
    with patch(
        "chuk_lazarus.cli.commands.experiment.handlers._run_experiment",
        return_value=_result,
    ) as mock_run:
        handlers.experiment_run("demo", experiments_dir="/tmp/exp")
    kwargs = mock_run.call_args[1]
    assert kwargs["backend"] is None
    assert kwargs["device"] is None


def test_handlers_module_has_no_top_level_mlx():
    src = Path(handlers.__file__).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", [])]
            assert not mod.startswith("mlx"), f"top-level mlx import: {mod}"
            for n in names:
                assert not n.startswith("mlx"), f"top-level mlx import: {n}"
