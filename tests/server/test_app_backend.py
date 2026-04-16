"""EWS-8: server/app.py lazy-import assertion."""

from __future__ import annotations

import subprocess
import sys


def test_app_import_does_not_load_mlx():
    code = (
        "import sys, importlib\n"
        "importlib.import_module('chuk_lazarus.server.app')\n"
        "bad = [k for k in sys.modules if k in ('mlx', 'mlx_lm')]\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


def test_create_app_signature_unchanged():
    from chuk_lazarus.server.app import Protocol, create_app

    assert Protocol.OPENAI.value == "openai"
    # Signature still accepts engine + protocols kwarg.
    import inspect

    params = inspect.signature(create_app).parameters
    assert "engine" in params
    assert "protocols" in params
