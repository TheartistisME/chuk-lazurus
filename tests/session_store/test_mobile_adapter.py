"""Unit tests for :class:`chuk_lazarus.session_store.mobile_adapter.MobileAdapter`.

Mobile-path proof-of-viability — no CUDA, no real model. These tests exercise
the four-method ``Adapter`` Protocol surface that ``MobileAdapter`` satisfies,
plus the external ``pause()``/``resume()`` API and the static ``dequantize``
helper. The int8 round-trip, the ``.npz``-at-``.npy``-path atomic write, the
flush cadence (N windows AND idle seconds), and the desktop-path non-regression
(adapter=None still works, module import does not mutate global torch state)
are all asserted here.

The final test in this file drives a full :class:`LiveIndexer` session with a
``MobileAdapter`` plugged in and a monkeypatched capture; it guards against
CUDA-stream construction on the mobile path and validates that per-window
boundaries round-trip through int8 with <=2% cosine drift end-to-end.
"""

from __future__ import annotations

import inspect
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chuk_lazarus.session_store.live_indexer import Adapter
from chuk_lazarus.session_store.mobile_adapter import MobileAdapter


# ---------------------------------------------------------------------------
# (A) Protocol conformance.
# ---------------------------------------------------------------------------


def test_mobile_adapter_satisfies_adapter_protocol() -> None:
    """runtime_checkable Protocol: MobileAdapter instance must pass isinstance."""
    adapter = MobileAdapter(apply_torch_threads=False)
    assert isinstance(adapter, Adapter)


def test_mobile_adapter_has_all_protocol_methods() -> None:
    """Explicit method presence + callability for each Protocol surface method."""
    adapter = MobileAdapter(apply_torch_threads=False)
    for name in ("residual_finalize", "write_boundary", "should_flush_manifest", "check_pause"):
        assert hasattr(adapter, name), f"missing Protocol method: {name}"
        assert callable(getattr(adapter, name)), f"Protocol attr not callable: {name}"


# ---------------------------------------------------------------------------
# (B) Int8 round-trip correctness (cosine drift <=2%).
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_residual_finalize_int8_round_trip_within_2pct_cosine_drift() -> None:
    """Deterministic fp32 (2304,) vector: int8 round-trip keeps cos >= 0.98."""
    rng = np.random.default_rng(seed=0)
    t_original = rng.standard_normal(2304).astype(np.float32)

    adapter = MobileAdapter(apply_torch_threads=False)
    finalized = adapter.residual_finalize(t_original)

    assert finalized["int8"].dtype == np.int8
    assert finalized["int8"].shape == (2304,)
    assert isinstance(finalized["scale"], float)
    assert finalized["scale"] > 0
    assert finalized["shape"] == (2304,)

    reconstructed = MobileAdapter.dequantize(finalized)
    cos = _cosine(t_original, reconstructed)
    assert cos >= 0.98, f"cosine drift too large: cos={cos}"


def test_residual_finalize_accepts_torch_tensor() -> None:
    """Torch tensor input path: same int8 + <=2% drift contract."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    t = torch.randn(2304, dtype=torch.float32)

    finalized = MobileAdapter(apply_torch_threads=False).residual_finalize(t)
    assert finalized["int8"].dtype == np.int8
    assert finalized["int8"].shape == (2304,)

    reconstructed = MobileAdapter.dequantize(finalized)
    cos = _cosine(t.detach().cpu().numpy(), reconstructed)
    assert cos >= 0.98, f"cosine drift too large (torch path): cos={cos}"


def test_residual_finalize_zero_tensor_scale_one() -> None:
    """All-zero residual: scale defaults to 1.0 and payload stays zero."""
    t = np.zeros((16,), dtype=np.float32)
    finalized = MobileAdapter(apply_torch_threads=False).residual_finalize(t)
    assert finalized["scale"] == 1.0
    assert np.all(finalized["int8"] == 0)


# ---------------------------------------------------------------------------
# (C) Atomic boundary write (.npz-at-.npy-path).
# ---------------------------------------------------------------------------


def test_write_boundary_atomic_npz_at_npy_path(tmp_path: Path) -> None:
    """write_boundary writes np.savez bytes at a .npy-suffixed path atomically."""
    adapter = MobileAdapter(apply_torch_threads=False)
    finalized = adapter.residual_finalize(np.arange(2304, dtype=np.float32))
    path = tmp_path / "window_000.npy"

    adapter.write_boundary(path, finalized)

    assert path.exists()
    # The tmp sibling should have been atomically renamed into place.
    assert not path.with_name(path.name + ".tmp").exists()

    loaded = np.load(path, allow_pickle=False)
    assert set(loaded.files) == {"int8", "scale", "shape"}
    np.testing.assert_array_equal(loaded["int8"], finalized["int8"])
    assert float(loaded["scale"]) == pytest.approx(finalized["scale"])
    np.testing.assert_array_equal(
        loaded["shape"], np.asarray(finalized["shape"], dtype=np.int64)
    )


def test_write_boundary_rejects_non_dict(tmp_path: Path) -> None:
    """Passing an ndarray (the default-path output) must raise TypeError."""
    adapter = MobileAdapter(apply_torch_threads=False)
    with pytest.raises(TypeError):
        adapter.write_boundary(tmp_path / "x.npy", np.zeros(4, dtype=np.float32))


def test_write_boundary_rejects_incomplete_dict(tmp_path: Path) -> None:
    """Dict missing required keys (scale/shape) must raise ValueError."""
    adapter = MobileAdapter(apply_torch_threads=False)
    with pytest.raises(ValueError):
        adapter.write_boundary(
            tmp_path / "x.npy", {"int8": np.zeros(4, dtype=np.int8)}
        )


# ---------------------------------------------------------------------------
# (D) Flush cadence (N windows AND idle seconds).
# ---------------------------------------------------------------------------


def test_should_flush_manifest_fires_every_n_windows() -> None:
    """Deterministic clock pinned at 0: flushes fire exactly on the Nth call."""
    t = [0.0]
    clk = lambda: t[0]
    adapter = MobileAdapter(
        flush_every_n_windows=8,
        flush_idle_seconds=60.0,
        clock=clk,
        apply_torch_threads=False,
    )
    # 7 calls should NOT flush; the 8th should.
    for _ in range(7):
        assert adapter.should_flush_manifest() is False
    assert adapter.should_flush_manifest() is True
    # After a flush, the counter resets; the next 7 are False and the 8th True.
    for _ in range(7):
        assert adapter.should_flush_manifest() is False
    assert adapter.should_flush_manifest() is True


def test_should_flush_manifest_fires_on_idle_timeout() -> None:
    """With N huge, the idle-seconds branch alone must trip a flush."""
    t = [0.0]
    clk = lambda: t[0]
    adapter = MobileAdapter(
        flush_every_n_windows=1000,
        flush_idle_seconds=60.0,
        clock=clk,
        apply_torch_threads=False,
    )
    # First call: counter=1, elapsed=0 -> False.
    assert adapter.should_flush_manifest() is False
    # Advance the clock past the idle threshold; the next call flushes.
    t[0] = 60.5
    assert adapter.should_flush_manifest() is True


def test_should_flush_manifest_thread_safe_under_contention() -> None:
    """4 threads * 100 calls = 400 calls; with N=8 exactly 50 are flushes."""
    adapter = MobileAdapter(
        flush_every_n_windows=8,
        flush_idle_seconds=1e9,
        apply_torch_threads=False,
    )
    hits: list[bool] = []
    hits_lock = threading.Lock()

    def worker() -> None:
        local: list[bool] = []
        for _ in range(100):
            local.append(adapter.should_flush_manifest())
        with hits_lock:
            hits.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(hits) == 400
    assert sum(1 for h in hits if h) == 400 // 8


# ---------------------------------------------------------------------------
# (E) pause() / resume() semantics.
# ---------------------------------------------------------------------------


def test_check_pause_blocks_until_resume() -> None:
    """check_pause blocks after pause(); resume() releases the blocked thread."""
    adapter = MobileAdapter(apply_torch_threads=False)
    adapter.pause()
    done = threading.Event()

    def worker() -> None:
        adapter.check_pause()
        done.set()

    th = threading.Thread(target=worker)
    th.start()
    # Give the worker time to enter check_pause and block.
    assert not done.wait(timeout=0.2), "worker returned from check_pause while paused"

    adapter.resume()
    assert done.wait(timeout=2.0), "resume() did not release the blocked worker"
    th.join(timeout=2.0)
    assert not th.is_alive()


def test_check_pause_noop_when_running() -> None:
    """With no prior pause(), check_pause must return (near-)immediately."""
    adapter = MobileAdapter(apply_torch_threads=False)
    t0 = time.monotonic()
    adapter.check_pause()
    assert time.monotonic() - t0 < 0.1


# ---------------------------------------------------------------------------
# (F) Desktop-path non-regression.
# ---------------------------------------------------------------------------


def test_module_import_does_not_mutate_torch_threads() -> None:
    """Constructing with apply_torch_threads=False must leave torch state alone."""
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    _ = MobileAdapter(apply_torch_threads=False)
    after = torch.get_num_threads()
    assert before == after


def test_live_indexer_still_accepts_adapter_none() -> None:
    """Sanity: the ``adapter`` param default remains ``None`` (desktop path)."""
    from chuk_lazarus.session_store.live_indexer import LiveIndexer

    sig = inspect.signature(LiveIndexer.__init__)
    assert sig.parameters["adapter"].default is None


def test_live_indexer_end_to_end_with_mobile_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration: MobileAdapter drives a full LiveIndexer session.

    Monkeypatches ``capture_post_crystal_boundary`` to avoid needing a real
    model. Asserts: (1) no CUDA stream constructed, (2) per-window .npy
    files exist at the expected paths (even though they are .npz-formatted
    bytes), (3) round-trip dequantization stays within 2% cosine drift
    versus the synthetic input residual.
    """
    torch = pytest.importorskip("torch")
    from chuk_lazarus.session_store import live_indexer as _mod
    from chuk_lazarus.session_store.live_indexer import LiveIndexer

    rng = np.random.default_rng(seed=42)
    expected: dict[int, np.ndarray] = {}
    call_counter = {"n": 0}

    def fake_capture(model, token_ids, *, crystal_layer, device=None, initial_residual=None):
        wid = call_counter["n"]
        call_counter["n"] += 1
        arr = rng.standard_normal(128).astype(np.float32)
        expected[wid] = arr.copy()
        return torch.from_numpy(arr)

    monkeypatch.setattr(_mod, "capture_post_crystal_boundary", fake_capture)

    # Fail loudly if any CUDA stream gets constructed on the mobile path.
    class _ExplodingStream:
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise AssertionError("MobileAdapter path must not construct a CUDA stream")

    if hasattr(torch, "cuda"):
        monkeypatch.setattr(torch.cuda, "Stream", _ExplodingStream, raising=False)

    arch_config = {
        "retrieval_layer": 4,
        "query_head": 0,
        "injection_layer": 6,
        "hidden_dim": 128,
        "head_dim": 64,
        "crystal_layer": 5,
        "window_size": 4,
    }
    adapter = MobileAdapter(
        flush_every_n_windows=2,
        flush_idle_seconds=1e9,
        apply_torch_threads=False,
    )

    class _DummyModel:
        def parameters(self):
            return iter(())

    indexer = LiveIndexer(
        model=_DummyModel(),
        tokenizer=None,
        checkpoint_dir=tmp_path / "store",
        crystal_layer=5,
        window_size=4,
        arch_config=arch_config,
        adapter=adapter,
    )
    for wid in range(4):
        indexer.enqueue(wid, [1, 2, 3, 4], ["k"], {"turn": wid})
    indexer.flush_and_close()

    boundaries_dir = tmp_path / "store" / "boundaries"
    for wid in range(4):
        p = boundaries_dir / f"window_{wid:03d}.npy"
        assert p.exists(), f"missing boundary for window {wid}"
        loaded = np.load(p, allow_pickle=False)
        assert set(loaded.files) == {"int8", "scale", "shape"}
        recon = loaded["int8"].astype(np.float32) * float(loaded["scale"])
        orig = expected[wid]
        cos = float(
            np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon))
        )
        assert cos >= 0.98, f"window {wid} cosine drift too large: {cos}"
