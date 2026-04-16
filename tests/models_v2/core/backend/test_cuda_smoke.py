"""
Opt-in CUDA smoke test for TorchBackend.

Skipped unless CHUK_CUDA_SMOKE=1 is set; exercises real hardware paths:
device selection, capability detection, dtype preference, and a minimal
matmul round-trip on the live GPU.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CHUK_CUDA_SMOKE") != "1",
    reason="CUDA smoke opt-in (set CHUK_CUDA_SMOKE=1)",
)


def test_cuda_smoke_real_hardware():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is False on this host")

    from chuk_lazarus.models_v2.core.backend import TorchBackend

    backend = TorchBackend()
    assert backend.device.startswith("cuda")

    major, _ = torch.cuda.get_device_capability(
        backend.device if ":" in backend.device else "cuda:0"
    )
    expected = torch.bfloat16 if major >= 8 else torch.float16
    assert backend.dtype == expected

    import numpy as np

    a = backend.from_numpy(np.eye(4, dtype=np.float32))
    b = backend.from_numpy(np.eye(4, dtype=np.float32) * 2.0)
    c = backend.matmul(a, b)
    result = backend.to_numpy(c)
    assert np.allclose(result, np.eye(4) * 2.0)
