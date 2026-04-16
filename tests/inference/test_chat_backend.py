"""EWS-1 (Wave 0.5) lazy-import regression test for ``inference.chat``.

Per Epic 2 §EWS-1 MLX-preservation rule::

    python -c "import chuk_lazarus.inference.chat; import sys;
               assert 'mlx' not in sys.modules"

must pass when ``CHUK_BACKEND=torch``.  This test runs the same assertion in
a clean subprocess so the check holds in CI on non-macOS hosts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(module: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "bad = sorted(k for k in sys.modules if k == 'mlx' or k == 'mlx_lm')\n"
        "assert not bad, bad\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_chat_module_is_lazy_under_torch_backend() -> None:
    res = _run("chuk_lazarus.inference.chat")
    assert res.returncode == 0, (
        f"inference.chat pulls mlx at import time\nstdout={res.stdout}\nstderr={res.stderr}"
    )


def test_generation_module_is_lazy_under_torch_backend() -> None:
    res = _run("chuk_lazarus.inference.generation")
    assert res.returncode == 0, (
        f"inference.generation pulls mlx at import time\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
