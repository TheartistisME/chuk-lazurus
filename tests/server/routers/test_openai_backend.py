"""EWS-8: lazy-import assertion for server/routers/openai.py."""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_openai_router_does_not_import_mlx():
    code = (
        "import sys, importlib\n"
        "importlib.import_module('chuk_lazarus.server.routers.openai')\n"
        "bad = [k for k in sys.modules if k in ('mlx', 'mlx_lm')]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_openai_router_exports_router():
    pytest.importorskip(
        "fastapi",
        reason="route export assertions require the optional server dependency",
    )

    from chuk_lazarus.server.routers.openai import router

    assert router is not None
    paths = {r.path for r in router.routes}
    assert any("chat/completions" in p for p in paths)
