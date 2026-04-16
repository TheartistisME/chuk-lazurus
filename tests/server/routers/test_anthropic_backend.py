"""EWS-8: lazy-import assertion for server/routers/anthropic.py."""

from __future__ import annotations

import subprocess
import sys


def test_anthropic_router_does_not_import_mlx():
    code = (
        "import sys, importlib\n"
        "importlib.import_module('chuk_lazarus.server.routers.anthropic')\n"
        "bad = [k for k in sys.modules if k in ('mlx', 'mlx_lm')]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr
