from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
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


def test_analyzer_package_import_is_lazy_runtime_clean() -> None:
    res = _run(
        "import sys\n"
        "import chuk_lazarus.introspection.analyzer as analyzer\n"
        "assert 'ModelAnalyzer' in analyzer.__all__\n"
        "bad = [name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')]\n"
        "assert not bad, bad\n"
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_analyzer_lazy_exports_resolve_without_mlx() -> None:
    res = _run(
        "import sys\n"
        "from chuk_lazarus.introspection.analyzer import (\n"
        "    ModelAnalyzer,\n"
        "    _load_model_sync,\n"
        "    compute_entropy,\n"
        "    get_layers_to_capture,\n"
        ")\n"
        "assert ModelAnalyzer.__module__ == 'chuk_lazarus.introspection.analyzer.core'\n"
        "assert _load_model_sync.__module__ == 'chuk_lazarus.introspection.analyzer.loader'\n"
        "assert compute_entropy.__module__ == 'chuk_lazarus.introspection.analyzer.utils'\n"
        "assert get_layers_to_capture([1].__len__(), 'all') == [0]\n"
        "bad = [name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')]\n"
        "assert not bad, bad\n"
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"


def test_analyzer_submodules_import_without_mlx() -> None:
    res = _run(
        "import importlib, sys\n"
        "importlib.import_module('chuk_lazarus.introspection.analyzer.core')\n"
        "importlib.import_module('chuk_lazarus.introspection.analyzer.utils')\n"
        "bad = [name for name in sys.modules if name == 'mlx' or name.startswith('mlx.')]\n"
        "assert not bad, bad\n"
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
