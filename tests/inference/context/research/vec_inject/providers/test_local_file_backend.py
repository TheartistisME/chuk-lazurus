"""EWS-2: LocalVecInjectProvider module is mlx-free at import time."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_local_file_provider_import_is_mlx_free():
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        "m = importlib.import_module("
        "'chuk_lazarus.inference.context.research.vec_inject.providers._local_file')\n"
        "assert hasattr(m, 'LocalVecInjectProvider')\n"
        "assert 'mlx' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_providers_package_import_is_mlx_free():
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        "importlib.import_module("
        "'chuk_lazarus.inference.context.research.vec_inject.providers')\n"
        "assert 'mlx' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
