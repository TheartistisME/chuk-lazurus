"""EWS-0.3b — focused gate for ``introspection/hooks.py`` lazy-mlx contract.

These tests complement ``tests/ci/test_no_top_level_mlx.py`` by asserting
the public API surface remains usable on a torch/CUDA-only host without
``libmlx.so``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_probe(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "module",
    [
        "chuk_lazarus.introspection.hooks",
        "chuk_lazarus.introspection",
    ],
)
def test_import_does_not_pull_mlx(module: str) -> None:
    """Importing the module (or the package root) must not load mlx/mlx_lm."""
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k.startswith('mlx.') "
        "or k == 'mlx_lm' or k.startswith('mlx_lm.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    res = _run_probe(code)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_enums_and_config_resolvable_without_mlx() -> None:
    """``LayerSelection``/``PositionSelection``/``CaptureConfig`` / ``CapturedState``
    must resolve on a torch-only host without importing mlx."""
    code = (
        "import sys\n"
        "from chuk_lazarus.introspection.hooks import (\n"
        "    CaptureConfig, CapturedState, LayerSelection, PositionSelection\n"
        ")\n"
        "assert LayerSelection.ALL.value == 'all'\n"
        "assert PositionSelection.LAST.value == 'last'\n"
        "cfg = CaptureConfig()\n"
        "assert cfg.capture_hidden_states is True\n"
        "assert CapturedState().num_layers_captured == 0\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k.startswith('mlx.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    res = _run_probe(code)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_modelhooks_class_resolvable_without_mlx() -> None:
    """``ModelHooks`` must be resolvable as a class object from the lazy
    package root without triggering mlx import; only method bodies touch mlx."""
    code = (
        "import sys\n"
        "from chuk_lazarus.introspection import ModelHooks\n"
        "assert ModelHooks.__name__ == 'ModelHooks'\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k.startswith('mlx.')]\n"
        "assert not bad, bad\n"
        "print('ok')\n"
    )
    res = _run_probe(code)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
