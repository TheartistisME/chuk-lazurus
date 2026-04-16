"""EWS-15 — runner honours ``backend`` / ``device`` overrides."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock as _MM

for name in ("mlx", "mlx.core", "mlx.nn", "mlx.utils", "mlx.optimizers"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["mlx.nn"].Module = _MM
sys.modules["mlx.nn"].Linear = _MM

import ast  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

import chuk_lazarus.experiments.runner as runner_mod
from chuk_lazarus.experiments.base import ExperimentConfig


def _fake_config(tmp_path, backend=None, device=None):
    return ExperimentConfig(
        name="demo",
        description="t",
        backend=backend,
        device=device,
        experiment_dir=tmp_path,
        data_dir=tmp_path / "d",
        checkpoint_dir=tmp_path / "c",
        results_dir=tmp_path / "r",
    )


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_run_experiment_dry_run_resolves_backend(tmp_path, backend):
    cfg = _fake_config(tmp_path)

    with patch.object(runner_mod, "load_config", return_value=cfg):
        fake_backend = MagicMock()
        with patch(
            "chuk_lazarus.models_v2.core.backend.get_backend",
            return_value=fake_backend,
        ) as mock_gb:
            result = runner_mod.run_experiment(
                "demo",
                experiments_dir=tmp_path,
                dry_run=True,
                backend=backend,
                device="cuda:0" if backend == "torch" else None,
            )

    assert result.status == "dry_run"
    mock_gb.assert_called_once()
    args, kwargs = mock_gb.call_args
    assert args[0] == backend
    assert kwargs.get("device") == ("cuda:0" if backend == "torch" else None)
    assert cfg.backend == backend


def test_run_experiment_no_flag_does_not_call_get_backend(tmp_path):
    cfg = _fake_config(tmp_path)

    with patch.object(runner_mod, "load_config", return_value=cfg):
        with patch("chuk_lazarus.models_v2.core.backend.get_backend") as mock_gb:
            runner_mod.run_experiment(
                "demo", experiments_dir=tmp_path, dry_run=True
            )
    mock_gb.assert_not_called()


def test_config_backend_used_when_flag_absent(tmp_path):
    cfg = _fake_config(tmp_path, backend="mlx")

    with patch.object(runner_mod, "load_config", return_value=cfg):
        with patch(
            "chuk_lazarus.models_v2.core.backend.get_backend"
        ) as mock_gb:
            runner_mod.run_experiment(
                "demo", experiments_dir=tmp_path, dry_run=True
            )
    mock_gb.assert_called_once()
    assert mock_gb.call_args[0][0] == "mlx"


def test_runner_module_has_no_top_level_mlx():
    src = Path(runner_mod.__file__).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", [])]
            assert not mod.startswith("mlx"), f"top-level mlx import: {mod}"
            for n in names:
                assert not n.startswith("mlx"), f"top-level mlx import: {n}"
