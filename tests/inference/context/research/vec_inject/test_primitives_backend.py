"""EWS-2: vec_inject primitives lazy-load mlx and expose backend-agnostic API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_primitives_import_is_mlx_free():
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        "importlib.import_module("
        "'chuk_lazarus.inference.context.research.vec_inject._primitives')\n"
        "assert 'mlx' not in sys.modules\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_primitives_exports_injection_functions():
    from chuk_lazarus.inference.context.research.vec_inject import _primitives

    assert hasattr(_primitives, "vec_inject")
    assert hasattr(_primitives, "vec_inject_all")
