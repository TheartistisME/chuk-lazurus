"""Unit tests for :class:`chuk_lazarus.session_store.live_indexer.LiveIndexer`.

These tests never touch a real model or CUDA. They monkeypatch the
locally-bound ``capture_post_crystal_boundary`` reference inside the
``live_indexer`` module so the worker only exercises LiveIndexer's queue /
threading / atomic-write / compaction code paths.

The on-disk contract is gated by
:class:`chuk_lazarus.inference.context.knowledge.torch_store.TorchKnowledgeStore`
load — if ``TorchKnowledgeStore.load`` accepts the directory, axis-3 is
honouring the baseline store schema.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
from chuk_lazarus.session_store import live_indexer as _live_indexer_mod
from chuk_lazarus.session_store.live_indexer import (
    Adapter,
    LiveIndexer,
    STORE_VERSION,
)


# ---------------------------------------------------------------------------
# Local constants (folder-isolated; we do not import from session_retrieval).
# ---------------------------------------------------------------------------

HIDDEN_SIZE: int = 1536

ARCH_CONFIG_E2B: dict[str, Any] = {
    "retrieval_layer": 28,
    "query_head": 7,
    "injection_layer": 29,
    "hidden_dim": 1536,
    "head_dim": 256,
    "crystal_layer": 29,
    "window_size": 512,
}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _fake_capture_factory(constant: float = 0.0):
    """Build a ``capture_post_crystal_boundary`` stand-in returning a fp32 tensor.

    The returned tensor is ``torch.full((HIDDEN_SIZE,), constant)``; callers
    that care about per-window uniqueness can monkeypatch their own.
    """

    def _fake_capture(model, token_ids, *, crystal_layer, device=None, initial_residual=None):
        return torch.full((HIDDEN_SIZE,), float(constant), dtype=torch.float32)

    return _fake_capture


def _patch_capture(monkeypatch: pytest.MonkeyPatch, fn=None) -> None:
    """Monkeypatch the *locally-bound* capture reference inside live_indexer."""
    monkeypatch.setattr(
        "chuk_lazarus.session_store.live_indexer.capture_post_crystal_boundary",
        fn if fn is not None else _fake_capture_factory(0.0),
    )


def make_indexer(tmp_path: Path, **overrides: Any) -> LiveIndexer:
    """Construct a default LiveIndexer rooted at ``tmp_path / "store"``."""
    store_dir = tmp_path / "store"
    kwargs: dict[str, Any] = dict(
        model=object(),  # sentinel; capture is monkeypatched so model is never used
        tokenizer=object(),
        checkpoint_dir=store_dir,
        crystal_layer=29,
        window_size=512,
        arch_config=ARCH_CONFIG_E2B,
    )
    kwargs.update(overrides)
    return LiveIndexer(**kwargs)


def wait_for_boundary(path: Path, timeout: float = 3.0) -> bool:
    """Poll up to ``timeout`` seconds for a boundary ``.npy`` to appear."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return False


def wait_for(condition, timeout: float = 3.0) -> bool:
    """Generic poll: wait for ``condition()`` to return truthy."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# 1. test_imports.
# ---------------------------------------------------------------------------


def test_imports() -> None:
    """Public surface is exported and ``STORE_VERSION == 12``."""
    from chuk_lazarus.session_store.live_indexer import (
        Adapter as _Adapter,
        LiveIndexer as _LiveIndexer,
        STORE_VERSION as _STORE_VERSION,
    )

    assert _STORE_VERSION == 12
    assert _LiveIndexer is LiveIndexer
    assert _Adapter is Adapter

    class _DuckAdapter:
        def residual_finalize(self, tensor):
            return tensor

        def write_boundary(self, path, array):
            return None

        def should_flush_manifest(self) -> bool:
            return False

        def check_pause(self) -> None:
            return None

    # runtime_checkable Protocol: duck-type instance must pass isinstance.
    assert isinstance(_DuckAdapter(), Adapter)


# ---------------------------------------------------------------------------
# 2. test_arch_config_validation_rejects_missing_keys (parametrized).
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = [
    "retrieval_layer",
    "query_head",
    "injection_layer",
    "hidden_dim",
    "head_dim",
    "crystal_layer",
    "window_size",
]


@pytest.mark.parametrize("missing_key", _REQUIRED_KEYS)
def test_arch_config_validation_rejects_missing_keys(
    tmp_path: Path, missing_key: str
) -> None:
    """Dropping any required arch_config key must raise ValueError."""
    bad_cfg = {k: v for k, v in ARCH_CONFIG_E2B.items() if k != missing_key}
    with pytest.raises(ValueError):
        LiveIndexer(
            model=object(),
            tokenizer=object(),
            checkpoint_dir=tmp_path / "store",
            crystal_layer=29,
            window_size=512,
            arch_config=bad_cfg,
        )


# ---------------------------------------------------------------------------
# 3. test_enqueue_starts_worker_and_processes_window.
# ---------------------------------------------------------------------------


def test_enqueue_starts_worker_and_processes_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single enqueue produces a boundary.npy and both shard JSONL files."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.0))
    indexer = make_indexer(tmp_path)
    try:
        indexer.enqueue(0, [10, 11, 12], ["alice"], {"clause_id": "x.0.0"})

        boundary_path = tmp_path / "store" / "boundaries" / "window_000.npy"
        assert wait_for_boundary(boundary_path, timeout=3.0), "worker never wrote boundary"

        arr = np.load(boundary_path)
        assert arr.shape == (HIDDEN_SIZE,)
        assert arr.dtype == np.float32

        shards_dir = tmp_path / "store" / "shards"
        tokens_shards = list(shards_dir.glob("tokens_*.jsonl"))
        keywords_shards = list(shards_dir.glob("keywords_*.jsonl"))
        assert len(tokens_shards) == 1
        assert len(keywords_shards) == 1

        tokens_records = [
            json.loads(line)
            for line in tokens_shards[0].read_text("utf-8").splitlines()
            if line.strip()
        ]
        keywords_records = [
            json.loads(line)
            for line in keywords_shards[0].read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert len(tokens_records) == 1
        assert tokens_records[0]["window_id"] == 0
        assert tokens_records[0]["token_ids"] == [10, 11, 12]
        assert len(keywords_records) == 1
        assert keywords_records[0]["window_id"] == 0
        assert keywords_records[0]["keywords"] == ["alice"]
    finally:
        indexer.stop()


# ---------------------------------------------------------------------------
# 4. test_flush_and_close_produces_TorchKnowledgeStore_loadable_store.
# ---------------------------------------------------------------------------


def test_flush_and_close_produces_TorchKnowledgeStore_loadable_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After flush_and_close, the directory loads via TorchKnowledgeStore.load."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.5))
    indexer = make_indexer(tmp_path)
    try:
        for wid in range(3):
            indexer.enqueue(wid, [10 + wid, 11 + wid], [f"kw{wid}"], {"clause_id": f"x.{wid}.0"})
        indexer.flush_and_close()
    finally:
        # flush_and_close already closed; stop is a no-op on closed indexer.
        pass

    store_dir = tmp_path / "store"

    # Compaction should have removed the shards/ directory.
    assert not (store_dir / "shards").exists(), "shards/ should be removed after compaction"

    # Manifest presence + shape.
    manifest = json.loads((store_dir / "manifest.json").read_text("utf-8"))
    assert manifest["clause_aligned"] is True
    assert manifest["num_windows"] == 3
    assert manifest["num_entries"] == 3
    for k in _REQUIRED_KEYS:
        assert k in manifest["arch_config"], f"missing arch_config key: {k}"

    # window_token_lists.npz present with three keys "0","1","2".
    with np.load(store_dir / "window_token_lists.npz") as npz:
        assert sorted(npz.files) == ["0", "1", "2"]

    # Three fp32 (1536,) boundary files.
    for wid in range(3):
        bpath = store_dir / "boundaries" / f"window_{wid:03d}.npy"
        assert bpath.exists()
        arr = np.load(bpath)
        assert arr.shape == (HIDDEN_SIZE,)
        assert arr.dtype == np.float32

    # TorchKnowledgeStore.load must accept the directory.
    store = TorchKnowledgeStore.load(store_dir)
    assert store.num_windows == 3


# ---------------------------------------------------------------------------
# 5. test_semaphore_serializes_model_forward_calls.
# ---------------------------------------------------------------------------


def test_semaphore_serializes_model_forward_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a Semaphore(1), max concurrent entrants into capture must be 1."""
    sem = threading.Semaphore(1)

    state_lock = threading.Lock()
    counter = {"current": 0, "max": 0}

    def _serialized_capture(model, token_ids, *, crystal_layer, device=None, initial_residual=None):
        with state_lock:
            counter["current"] += 1
            if counter["current"] > counter["max"]:
                counter["max"] = counter["current"]
        try:
            time.sleep(0.03)
            return torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
        finally:
            with state_lock:
                counter["current"] -= 1

    _patch_capture(monkeypatch, _serialized_capture)
    indexer = make_indexer(tmp_path, forward_semaphore=sem)
    try:
        for wid in range(5):
            indexer.enqueue(wid, [10 + wid], [], {"clause_id": f"x.{wid}.0"})
        # Drain: wait for all five boundaries.
        for wid in range(5):
            bp = tmp_path / "store" / "boundaries" / f"window_{wid:03d}.npy"
            assert wait_for_boundary(bp, timeout=3.0), f"window {wid} never written"
    finally:
        indexer.stop()

    assert counter["max"] == 1, f"expected max 1 concurrent capture, saw {counter['max']}"


# ---------------------------------------------------------------------------
# 6. test_pause_blocks_worker_resume_releases_it.
# ---------------------------------------------------------------------------


def test_pause_blocks_worker_resume_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pause() before first enqueue stalls the worker; resume() releases it."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.0))
    indexer = make_indexer(tmp_path)
    try:
        indexer.pause()
        indexer.enqueue(0, [10, 11, 12], ["alice"], {"clause_id": "x.0.0"})

        boundary_path = tmp_path / "store" / "boundaries" / "window_000.npy"
        # Worker should be paused. Give it plenty of time to (wrongly) process.
        time.sleep(0.15)
        assert not boundary_path.exists(), "worker wrote while paused"

        indexer.resume()
        assert wait_for_boundary(boundary_path, timeout=0.5), (
            "resume() did not release the worker"
        )
    finally:
        indexer.stop()


# ---------------------------------------------------------------------------
# 7. test_atomic_write_crash_safety_no_partial_final_files.
# ---------------------------------------------------------------------------


def test_atomic_write_crash_safety_no_partial_final_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulated crash inside _atomic_save_npy leaves no final boundary file."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.0))

    def _crashing_save(final_path: Path, array: np.ndarray) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(
        "chuk_lazarus.session_store.live_indexer._atomic_save_npy",
        _crashing_save,
    )

    indexer = make_indexer(tmp_path)
    try:
        indexer.enqueue(0, [10, 11, 12], ["alice"], {"clause_id": "x.0.0"})
        # Wait for the worker to attempt + fail.
        assert wait_for(lambda: indexer._failed_q.qsize() > 0, timeout=3.0), (
            "worker did not record a failure"
        )
    finally:
        indexer.stop()

    boundary_path = tmp_path / "store" / "boundaries" / "window_000.npy"
    assert not boundary_path.exists(), "final boundary file must not exist after crash"
    # Internal state peek for crash-safety testing.
    assert indexer._failed_q.qsize() > 0


# ---------------------------------------------------------------------------
# 8. test_compaction_tolerates_torn_jsonl_tail.
# ---------------------------------------------------------------------------


def test_compaction_tolerates_torn_jsonl_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn tail line in a token shard is silently discarded on compaction."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.0))
    indexer = make_indexer(tmp_path)

    # Pre-create shards dir + torn shard before enqueue kicks off the worker.
    shards_dir = tmp_path / "store" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    torn_path = shards_dir / "tokens_00099.jsonl"
    torn_path.write_text(
        '{"window_id":99,"token_ids":[77,78]}\n'
        '{"window_id":99,"token_ids":[1,2',  # torn tail — no trailing bracket/newline
        encoding="utf-8",
    )

    try:
        indexer.enqueue(0, [10, 11, 12], ["alice"], {"clause_id": "x.0.0"})
        indexer.flush_and_close()
    finally:
        pass

    store_dir = tmp_path / "store"
    manifest = json.loads((store_dir / "manifest.json").read_text("utf-8"))
    # Torn tail is silently discarded; the first valid line (wid=99) + our wid=0
    # survive. num_entries counts unique token_list keys: {0, 99} => 2.
    assert manifest["num_entries"] == 2
    # num_windows derives from max(wid) + 1 per the compaction code.
    assert manifest["num_windows"] == 100

    # The valid first line of the torn shard was retained.
    with np.load(store_dir / "window_token_lists.npz") as npz:
        assert "99" in npz.files
        assert list(npz["99"]) == [77, 78]
        assert "0" in npz.files


# ---------------------------------------------------------------------------
# 9. test_adapter_injection_delegates_finalize_and_write.
# ---------------------------------------------------------------------------


class _RecordingAdapter:
    """Adapter stand-in that records every protocol call."""

    def __init__(self) -> None:
        self.residual_finalize_calls: list[Any] = []
        self.write_boundary_calls: list[tuple[Path, Any]] = []
        self.should_flush_manifest_calls: int = 0
        self.check_pause_calls: int = 0

    def residual_finalize(self, tensor: Any) -> Any:
        self.residual_finalize_calls.append(tensor)
        # Hand back a numpy array the default write-path would have produced.
        return np.asarray(tensor.detach().to("cpu", dtype=torch.float32).numpy())

    def write_boundary(self, path: Path, array: Any) -> None:
        self.write_boundary_calls.append((Path(path), array))
        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.save(fh, np.asarray(array, dtype=np.float32), allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)

    def should_flush_manifest(self) -> bool:
        self.should_flush_manifest_calls += 1
        return True

    def check_pause(self) -> None:
        self.check_pause_calls += 1


def test_adapter_injection_delegates_finalize_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adapter's 4 Protocol methods are all called on the hot path."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.25))
    recorder = _RecordingAdapter()
    indexer = make_indexer(tmp_path, adapter=recorder)
    try:
        for wid in range(2):
            indexer.enqueue(wid, [10 + wid, 11 + wid], [f"kw{wid}"], {"clause_id": f"x.{wid}.0"})
        indexer.flush_and_close()
    finally:
        pass

    assert len(recorder.residual_finalize_calls) == 2
    assert len(recorder.write_boundary_calls) == 2

    # Expected paths per-window.
    expected_paths = {
        tmp_path / "store" / "boundaries" / "window_000.npy",
        tmp_path / "store" / "boundaries" / "window_001.npy",
    }
    assert {p for p, _ in recorder.write_boundary_calls} == expected_paths

    assert recorder.should_flush_manifest_calls >= 2
    assert recorder.check_pause_calls >= 2


# ---------------------------------------------------------------------------
# 10. test_stop_then_reenqueue_raises_or_is_noop.
# ---------------------------------------------------------------------------


def test_stop_then_reenqueue_raises_or_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After stop(), enqueue() raises RuntimeError (per the shipped code)."""
    _patch_capture(monkeypatch, _fake_capture_factory(0.0))
    indexer = make_indexer(tmp_path)
    indexer.stop()
    with pytest.raises(RuntimeError):
        indexer.enqueue(0, [1, 2, 3], [], {"clause_id": "x.0.0"})


# ---------------------------------------------------------------------------
# 11. test_benchmark_live_indexer_overhead_in_isolation (Q3 ceiling).
# ---------------------------------------------------------------------------


@pytest.mark.benchmark_like
def test_benchmark_live_indexer_overhead_in_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure queue/threading/fsync/atomic-write/compaction overhead only.

    THRESHOLD_MS default 150 ms/window; override via ``LIVE_INDEXER_BENCH_MS_PER_WINDOW``
    env var so the lead can tune without editing tests.
    """
    threshold_ms_str = os.environ.get("LIVE_INDEXER_BENCH_MS_PER_WINDOW", "150")
    try:
        THRESHOLD_MS = float(threshold_ms_str)
    except ValueError:
        THRESHOLD_MS = 150.0

    # Instant capture — no sleep, no torch ops beyond a zero alloc.
    def _instant_capture(model, token_ids, *, crystal_layer, device=None, initial_residual=None):
        return torch.zeros(HIDDEN_SIZE, dtype=torch.float32)

    _patch_capture(monkeypatch, _instant_capture)
    indexer = make_indexer(tmp_path)

    N = 20
    t0 = time.perf_counter()
    for wid in range(N):
        indexer.enqueue(
            wid,
            [10 + (wid % 7), 11 + (wid % 5)],
            [f"kw{wid}"],
            {"clause_id": f"x.{wid}.0"},
        )
    indexer.flush_and_close()
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    avg_ms = elapsed_ms / N
    print(
        f"[live-indexer-bench] N={N} avg={avg_ms:.2f}ms/window "
        f"threshold={THRESHOLD_MS}ms"
    )
    assert avg_ms < THRESHOLD_MS, (
        f"LiveIndexer overhead exceeded ceiling: {avg_ms:.2f}ms/window >= {THRESHOLD_MS}ms"
    )


# ---------------------------------------------------------------------------
# Sanity: the monkeypatch target exists on the module we targeted.
# ---------------------------------------------------------------------------


def test_monkeypatch_target_exists() -> None:
    """Guardrail: the local binding we patch must live on live_indexer."""
    assert hasattr(_live_indexer_mod, "capture_post_crystal_boundary")
    assert hasattr(_live_indexer_mod, "_atomic_save_npy")
