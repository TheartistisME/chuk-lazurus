from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )


def test_top_level_package_import_is_lazy_runtime_clean():
    res = _run(
        "import sys\n"
        "import chuk_lazarus  # noqa: F401\n"
        "bad = [name for name in sys.modules if name == 'mlx' or name == 'mlx_lm']\n"
        "assert not bad, bad\n"
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_top_level_package_lazy_exports_resolve_without_mlx():
    res = _run(
        "import sys\n"
        "from chuk_lazarus import LoRAConfig, apply_lora\n"
        "assert LoRAConfig.__module__ == 'chuk_lazarus.models_v2.adapters.lora'\n"
        "assert apply_lora.__module__ == 'chuk_lazarus.models_v2.adapters.lora'\n"
        "bad = [name for name in sys.modules if name == 'mlx' or name == 'mlx_lm']\n"
        "assert not bad, bad\n"
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
