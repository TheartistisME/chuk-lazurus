"""Tests for ``introspection._backend_dispatch`` (EWS-5).

Covers:
- Lazy-import: importing ``_backend_dispatch`` under torch backend does not
  pull MLX into ``sys.modules``.
- ``to_backend_tensor`` / ``from_backend_tensor`` round-trip for torch.
- ``backend_matmul`` parity with torch matmul.
- ``register_hook`` torch arm detaches cleanly.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest


def _run_subprocess(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join([repo_src, existing_pp]) if existing_pp else repo_src
    full_env = {**os.environ, **env, "PYTHONPATH": pythonpath}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=full_env,
        capture_output=True,
        text=True,
    )


def test_import_does_not_load_mlx_under_torch_backend():
    code = (
        "import sys\n"
        "from chuk_lazarus.introspection import _backend_dispatch  # noqa: F401\n"
        "assert 'mlx' not in sys.modules, sorted(m for m in sys.modules if m.startswith('mlx'))\n"
    )
    result = _run_subprocess(code, {"CHUK_BACKEND": "torch"})
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"


def test_public_surface():
    from chuk_lazarus.introspection import _backend_dispatch as bd

    assert hasattr(bd, "to_backend_tensor")
    assert hasattr(bd, "from_backend_tensor")
    assert hasattr(bd, "backend_matmul")
    assert hasattr(bd, "register_hook")


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_roundtrip_torch(monkeypatch):
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    from chuk_lazarus.models_v2.core.backend import get_backend

    # Reset backend singleton
    import chuk_lazarus.models_v2.core.backend as backend_mod

    if hasattr(backend_mod, "_current_backend"):
        backend_mod._current_backend = None

    be = get_backend()
    if be.name != "torch":
        pytest.skip(f"backend resolved to {be.name}, not torch")

    from chuk_lazarus.introspection._backend_dispatch import (
        backend_matmul,
        from_backend_tensor,
        to_backend_tensor,
    )

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = to_backend_tensor(arr, backend=be)
    back = from_backend_tensor(t, backend=be)
    np.testing.assert_allclose(arr, back)

    out = backend_matmul(arr, arr, backend=be)
    expected = arr @ arr
    np.testing.assert_allclose(from_backend_tensor(out, backend=be), expected, rtol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_register_hook_torch_detaches():
    import torch
    import torch.nn as tnn

    from chuk_lazarus.introspection._backend_dispatch import register_hook

    module = tnn.Linear(4, 4)
    hits: list = []

    def _hook(_m, _inp, out):
        hits.append(out.shape)

    detach = register_hook(backend="torch", module=module, hook=_hook)
    module(torch.zeros(1, 4))
    assert len(hits) == 1

    detach()
    module(torch.zeros(1, 4))
    assert len(hits) == 1


def test_register_hook_unsupported_backend_raises():
    from chuk_lazarus.introspection._backend_dispatch import register_hook

    with pytest.raises(NotImplementedError):
        register_hook(backend="jax", module=object(), hook=lambda *a, **k: None)
