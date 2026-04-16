"""Smoke test: importing prefill._vec_inject under CHUK_BACKEND=torch must not pull MLX."""

from __future__ import annotations

import os
import subprocess
import sys


def test_import_vec_inject_with_torch_backend_does_not_load_mlx():
    code = (
        "import sys\n"
        "import chuk_lazarus.cli.commands.context.prefill._vec_inject  # noqa: F401\n"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if m.startswith('mlx'))\n"
        "assert 'mlx_lm' not in sys.modules\n"
    )
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([os.path.abspath(repo_src), existing_pp]) if existing_pp else os.path.abspath(repo_src)
    env = {**os.environ, "CHUK_BACKEND": "torch", "PYTHONPATH": pythonpath}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
