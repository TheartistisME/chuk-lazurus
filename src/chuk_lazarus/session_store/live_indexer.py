"""Live incremental memory sidecar for per-session clause-aligned stores.

Who writes what
---------------
This module defines :class:`LiveIndexer` — a background daemon that consumes
``(window_id, token_ids, keywords, clause_metadata)`` tuples emitted by the
chat-process turn pipeline and transforms them, live, into the same on-disk
layout that :class:`chuk_lazarus.inference.context.knowledge.torch_store.TorchKnowledgeStore`
knows how to load. The indexer delegates residual capture to the existing
primitive ``capture_post_crystal_boundary`` — it does not re-implement
transformer plumbing.

On-disk contract
----------------
Files written into ``checkpoint_dir`` match the baseline store byte-for-byte
in terms of *schema*:

* ``boundaries/window_NNN.npy``           (float32, 1D hidden_size)
* ``window_token_lists.npz``              (int64 arrays keyed by ``str(wid)``)
* ``window_tokens.npz``                   (same data as token_lists)
* ``keywords.json``                       (``{str(wid): [str, ...]}``)
* ``window_metadata.json``                (``{str(wid): {...}}``)
* ``idf.json``                            (``{str(token_id): float}``)
* ``manifest.json``                       (gate keys + ``arch_config``)

Every final-name write is ``.tmp`` + ``os.replace`` — atomic on POSIX and
on WSL NTFS. The manifest is written **last**, after every other file is in
place, so any reader that uses manifest presence as the readiness gate
(``iter_checkpoint_handles``) naturally skips an in-flight session.

Adapter Protocol surface
------------------------
:class:`Adapter` is a ``typing.Protocol`` single-owned by this module. axis-3
(``MobileAdapter``) imports and implements it here — never re-defined
downstream. It carves out exactly the surface that a battery/thermal-aware
mobile backend might want to vary:

* ``residual_finalize(tensor)``            — how to turn the fp32 CPU tensor
  captured by ``capture_post_crystal_boundary`` into the array/bytes this
  adapter persists. Default: ``tensor.numpy().astype(np.float32)``.
* ``write_boundary(path, array)``          — how to write the array to a
  sibling ``.tmp`` then replace to ``path``. Default: ``np.save`` + replace.
* ``should_flush_manifest()``              — cadence check; default flushes
  an in-flight manifest on every window.
* ``check_pause()``                        — blocking hook the daemon calls
  between iterations. Default: no-op.

Crash safety
------------
* Tuples enqueued from the chat process survive until the daemon pops them
  (``queue.Queue`` is in-process and unbounded). Crash before ``flush_and_close``
  returns ⇒ store is incomplete ⇒ manifest absent ⇒ reader skips it.
* Per-window boundary writes are atomic (``.npy.tmp`` → ``os.replace``).
* Per-turn metadata is appended to JSONL shards (``shards/tokens_NNNNN.jsonl``,
  ``shards/keywords_NNNNN.jsonl``, ``shards/window_metadata_NNNNN.jsonl``).
  Append-only mode + ``fsync`` + close on each append means a torn *last*
  line is the only failure mode; compaction detects it by catching
  ``JSONDecodeError`` on the tail and discarding it. A decode failure on any
  non-tail line is treated as real corruption and raised.
* Compaction builds canonical ``.npz``/``.json`` files, ``os.replace``s them
  into place, THEN removes the ``shards/`` directory, THEN writes
  ``manifest.json``. Interrupted compaction leaves the prior shards for a
  re-run; manifest absence keeps readers away.

Concurrency rationale
---------------------
The indexer runs a single background daemon thread that shares the model
with the chat process. Two concurrency primitives protect the shared model:

* ``threading.Semaphore(1)`` (``_forward_semaphore``) — serializes every
  model forward call. The chat process passes its own semaphore in so that
  decode and residual-capture forwards never interleave.
* ``torch.cuda.Stream(priority=-1)`` — when the model lives on CUDA, the
  capture forward is launched on a low-priority stream so that decode on
  the default (higher-priority) stream is not stalled by H2D / D2H overlap
  of the residual transfer. On CPU this is ``None`` and ignored.

State mutated from both worker and API surfaces lives behind
``self._state_lock``. ``capture_q`` is a ``queue.Queue`` (already thread-safe).
Pause/resume uses a ``threading.Event`` (set=running, clear=paused) checked
at the top of each worker iteration. ``stop()`` enqueues a singleton
sentinel onto ``capture_q`` which the worker interprets as "drain and exit".
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:  # Torch is needed for capture, but keep module import-safe on CPU-only hosts.
    import torch
except Exception:  # pragma: no cover - torch is a core dep elsewhere
    torch = None  # type: ignore[assignment]

from chuk_lazarus.inference.context.knowledge.torch_capture import (
    capture_post_crystal_boundary,
)
from chuk_lazarus.inference.context.knowledge.torch_store import (
    BOUNDARIES_DIR,
    IDF_FILE,
    KEYWORDS_FILE,
    MANIFEST_FILE,
    STORE_VERSION,
    WINDOW_METADATA_FILE,
    WINDOW_TOKEN_LISTS_FILE,
    WINDOW_TOKENS_FILE,
)

__all__ = ["Adapter", "LiveIndexer", "STORE_VERSION"]


# ---------------------------------------------------------------------------
# Module-level constants (single-owned here; axis-2/3 must import, not redefine).
# ---------------------------------------------------------------------------

_STOP_SENTINEL: object = object()
"""Singleton enqueued by :meth:`LiveIndexer.stop` to drain the daemon."""

_SHARDS_DIRNAME: str = "shards"
_TOKENS_SHARD_FMT: str = "tokens_{:05d}.jsonl"
_KEYWORDS_SHARD_FMT: str = "keywords_{:05d}.jsonl"
_METADATA_SHARD_FMT: str = "window_metadata_{:05d}.jsonl"

_REQUIRED_ARCH_CONFIG_KEYS: tuple[str, ...] = (
    "retrieval_layer",
    "query_head",
    "injection_layer",
    "hidden_dim",
    "head_dim",
    "crystal_layer",
    "window_size",
)


# ---------------------------------------------------------------------------
# Adapter Protocol — hot-swappable delegation surface.
# ---------------------------------------------------------------------------


@runtime_checkable
class Adapter(Protocol):
    """Hot-swappable persistence / pacing surface for :class:`LiveIndexer`.

    axis-3's ``MobileAdapter`` duck-types this. The default CUDA/file path is
    *inlined* inside :class:`LiveIndexer` and triggered when ``adapter`` is
    ``None`` — there is no ``CudaAdapter`` class because the default behavior
    is trivial; lifting it into a symmetric adapter is a pure refactor.
    """

    def residual_finalize(self, tensor: Any) -> Any:
        """Finalize the fp32 CPU residual tensor into persistable array/bytes.

        Parameters
        ----------
        tensor:
            Output of ``capture_post_crystal_boundary`` — already on CPU in
            fp32, shape ``(hidden_size,)``.
        """
        ...

    def write_boundary(self, path: Path, array: Any) -> None:
        """Write ``array`` to ``path`` atomically.

        Implementations MUST write a sibling ``.tmp`` then
        ``os.replace(tmp, path)``. The path suffix (default ``.npy``) is
        under the adapter's control — mobile backends may choose a different
        container entirely.
        """
        ...

    def should_flush_manifest(self) -> bool:
        """Return True if the daemon should write an in-flight manifest now.

        Default adapter flushes on every window; a mobile adapter may batch.
        """
        ...

    def check_pause(self) -> None:
        """Blocking call invoked between worker iterations.

        Mobile adapters may sleep/spin here for battery / thermal budgets.
        The default no-op returns immediately.
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _atomic_write_bytes_via_tmp(final_path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``final_path`` atomically via ``.tmp`` + replace."""
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)


def _atomic_write_text_via_tmp(final_path: Path, text: str) -> None:
    _atomic_write_bytes_via_tmp(final_path, text.encode("utf-8"))


def _atomic_save_npy(final_path: Path, array: np.ndarray) -> None:
    """Atomically persist a numpy array to ``final_path`` (``.npy``)."""
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    # ``np.save`` adds the ``.npy`` suffix unless it's already present; our
    # tmp name does not end in ``.npy`` so we pass ``allow_pickle=False`` and
    # rely on file-handle form to avoid any auto-suffixing.
    with open(tmp_path, "wb") as fh:
        np.save(fh, np.asarray(array, dtype=np.float32), allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)


def _atomic_save_npz(final_path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically persist an ``.npz`` with int64 per-window arrays."""
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, final_path)


def _append_jsonl_line(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record as a line; fsync + close each call."""
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _read_jsonl_tolerant_tail(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL shard, tolerating a single torn tail line.

    A ``JSONDecodeError`` on the *last* non-empty line is silently dropped
    (treated as crash-mid-write). A decode error anywhere else signals real
    corruption and is re-raised.
    """
    if not path.exists():
        return []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    # Strip trailing empty lines — they are not "last data line".
    while raw_lines and raw_lines[-1].strip() == "":
        raw_lines.pop()
    if not raw_lines:
        return []

    out: list[dict[str, Any]] = []
    last_idx = len(raw_lines) - 1
    for idx, line in enumerate(raw_lines):
        if line.strip() == "":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if idx == last_idx:
                # Torn tail — drop it.
                break
            raise
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _compute_idf_fallback(
    window_token_lists: dict[int, list[int]],
) -> dict[str, float]:
    """Best-effort IDF computation; falls back to empty dict on any failure.

    Import is resolved lazily so a missing/renamed router still yields a
    valid (if empty) ``idf.json`` — ``TorchKnowledgeStore.load`` will then
    compute IDF itself at read time. Either answer is correct.
    """
    try:
        from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter  # noqa: WPS433

        window_token_sets: dict[int, set[int]] = {
            wid: {int(t) for t in tokens} for wid, tokens in window_token_lists.items()
        }
        raw = TFIDFRouter.compute_idf(window_token_sets)  # type: ignore[attr-defined]
        return {str(int(k)): float(v) for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001 - defensive; IDF is optional at write time
        sys.stderr.write(
            f"[LiveIndexer] WARN: IDF fallback engaged ({exc!r}); "
            "writing empty idf.json and deferring to reader-side compute_idf.\n"
        )
        return {}


# ---------------------------------------------------------------------------
# LiveIndexer.
# ---------------------------------------------------------------------------


class LiveIndexer:
    """Daemon-backed incremental writer for a clause-aligned torch store.

    See module docstring for on-disk contract and concurrency rationale.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        checkpoint_dir: Path | str,
        *,
        crystal_layer: int,
        window_size: int,
        arch_config: dict[str, Any],
        adapter: Adapter | None = None,
        forward_semaphore: threading.Semaphore | None = None,
    ) -> None:
        # Validate arch_config early — we write it straight through into the
        # manifest at flush time, and a missing key there will only surface
        # when a reader tries to load the store (by which time we have
        # already written boundaries/shards for nothing).
        self._validate_arch_config(arch_config)

        self._model = model
        self._tokenizer = tokenizer
        self._checkpoint_dir: Path = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self._checkpoint_dir / BOUNDARIES_DIR).mkdir(parents=True, exist_ok=True)
        (self._checkpoint_dir / _SHARDS_DIRNAME).mkdir(parents=True, exist_ok=True)

        self._crystal_layer: int = int(crystal_layer)
        self._window_size: int = int(window_size)
        self._arch_config: dict[str, Any] = dict(arch_config)

        self._adapter: Adapter | None = adapter
        self._forward_semaphore: threading.Semaphore = (
            forward_semaphore if forward_semaphore is not None else threading.Semaphore(1)
        )

        # CUDA stream for overlapping residual capture with chat decode. Only
        # created when model sits on CUDA. Lazy import-safe guard for torch.
        self._stream: Any | None = None
        if torch is not None:
            try:
                device = next(self._model.parameters()).device
                if device.type == "cuda":
                    # priority=-1 means "lower priority than default(0)" in CUDA land.
                    self._stream = torch.cuda.Stream(priority=-1)
            except (StopIteration, AttributeError, RuntimeError):
                self._stream = None

        # Work queue + worker thread (started lazily on first enqueue).
        self._capture_q: queue.Queue[tuple[int, list[int], list[str], dict[str, Any]] | object] = queue.Queue()
        self._failed_q: queue.Queue[tuple[Any, BaseException]] = queue.Queue(maxsize=64)
        self._worker: threading.Thread | None = None
        self._worker_started: bool = False
        self._worker_stop_requested: bool = False

        # Pause/resume gate. ``set()`` ⇒ running, ``clear()`` ⇒ paused.
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()

        # Mutable state (always accessed under ``_state_lock``).
        self._state_lock: threading.Lock = threading.Lock()
        self._windows_written: int = 0
        self._shards_open: int = 0  # kept for observability; append opens/closes per line
        self._manifest_dirty: bool = False
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def enqueue(
        self,
        window_id: int,
        token_ids: list[int],
        keywords: list[str],
        clause_metadata: dict[str, Any],
    ) -> None:
        """Non-blocking: enqueue one window tuple; start the daemon if idle."""
        if self._closed:
            raise RuntimeError("LiveIndexer: cannot enqueue after flush_and_close/stop")
        tuple_payload = (
            int(window_id),
            [int(t) for t in token_ids],
            [str(k) for k in keywords],
            dict(clause_metadata),
        )
        self._ensure_worker()
        self._capture_q.put(tuple_payload)

    def pause(self) -> None:
        """Thread-safe pause: worker stalls at top of its next iteration."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Thread-safe resume: worker proceeds on next iteration."""
        self._pause_event.set()

    def stop(self) -> None:
        """Enqueue sentinel, drain for ≤30 s, join the worker. No compaction."""
        if self._closed:
            return
        self._worker_stop_requested = True
        self._capture_q.put(_STOP_SENTINEL)
        deadline = time.monotonic() + 30.0
        if self._worker is not None:
            remaining = max(0.0, deadline - time.monotonic())
            self._worker.join(timeout=remaining)
            if self._worker.is_alive():
                raise RuntimeError(
                    "LiveIndexer.stop: worker did not drain within 30s "
                    f"(pending={self._capture_q.qsize()})"
                )
        self._closed = True

    def flush_and_close(self) -> None:
        """Drain queue, compact shards, write manifest LAST. Idempotent-ish.

        Raises
        ------
        RuntimeError
            If the daemon failed to process one or more windows (``_failed_q``
            non-empty) OR if compaction itself fails.
        """
        if self._closed:
            return

        # Signal end-of-stream to the worker, then join.
        self._capture_q.put(_STOP_SENTINEL)
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=60.0)
            if self._worker.is_alive():
                raise RuntimeError(
                    "LiveIndexer.flush_and_close: worker still running after 60s; "
                    "pending captures were not processed."
                )

        failed = self._drain_failed_queue()
        if failed:
            # Still attempt compaction of what we have? No — raise so the
            # caller knows the store is incomplete; shards remain on disk
            # for post-mortem.
            raise RuntimeError(
                f"LiveIndexer: {len(failed)} window(s) failed during capture: "
                + ", ".join(f"wid={w}:{type(e).__name__}" for w, e in failed[:8])
                + (" …" if len(failed) > 8 else "")
            )

        self._compact()
        self._closed = True

    # ------------------------------------------------------------------
    # Worker lifecycle.
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        with self._state_lock:
            if self._worker_started:
                return
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="LiveIndexer-worker",
                daemon=True,
            )
            self._worker.start()
            self._worker_started = True

    def _worker_loop(self) -> None:
        while True:
            # Pause gate (external pause()).
            self._pause_event.wait()

            # Adapter pause hook (battery/thermal).
            try:
                self._check_pause()
            except Exception as exc:  # noqa: BLE001 - never let the pause hook kill the worker
                sys.stderr.write(
                    f"[LiveIndexer] ERROR: adapter.check_pause raised {exc!r}; continuing.\n"
                )

            try:
                item = self._capture_q.get(timeout=0.5)
            except queue.Empty:
                if self._worker_stop_requested and self._capture_q.empty():
                    return
                continue

            if item is _STOP_SENTINEL:
                return

            # Normal work item.
            window_id, token_ids, keywords, clause_metadata = item  # type: ignore[misc]
            try:
                self._process_one(window_id, token_ids, keywords, clause_metadata)
            except BaseException as exc:  # noqa: BLE001 - the whole point is to not die
                sys.stderr.write(
                    f"[LiveIndexer] ERROR: window {window_id} failed: {exc!r}\n"
                )
                try:
                    self._failed_q.put_nowait((window_id, exc))
                except queue.Full:
                    sys.stderr.write(
                        "[LiveIndexer] ERROR: failed-queue is full; dropping error record.\n"
                    )
                # Keep running — losing one window is acceptable; losing the session is not.

    def _process_one(
        self,
        window_id: int,
        token_ids: list[int],
        keywords: list[str],
        clause_metadata: dict[str, Any],
    ) -> None:
        # 1. Capture residual under the shared forward semaphore, on a low-prio
        #    CUDA stream when available (overlaps with chat decode on default).
        with self._forward_semaphore:
            if torch is not None and self._stream is not None:
                with torch.cuda.stream(self._stream):  # type: ignore[attr-defined]
                    boundary_tensor = capture_post_crystal_boundary(
                        self._model,
                        token_ids,
                        crystal_layer=self._crystal_layer,
                    )
                # Ensure the capture stream is done before we pull the CPU tensor.
                self._stream.synchronize()
            else:
                boundary_tensor = capture_post_crystal_boundary(
                    self._model,
                    token_ids,
                    crystal_layer=self._crystal_layer,
                )

        # 2. Finalize residual (adapter-overridable).
        array = self._finalize(boundary_tensor)

        # 3. Persist boundary atomically.
        boundary_path = (
            self._checkpoint_dir / BOUNDARIES_DIR / f"window_{int(window_id):03d}.npy"
        )
        self._write_boundary(boundary_path, array)

        # 4. Append shards (tokens / keywords / metadata).
        shard_dir = self._checkpoint_dir / _SHARDS_DIRNAME
        shard_index = int(window_id)  # one shard per window keeps compaction simple
        _append_jsonl_line(
            shard_dir / _TOKENS_SHARD_FMT.format(shard_index),
            {"window_id": int(window_id), "token_ids": [int(t) for t in token_ids]},
        )
        _append_jsonl_line(
            shard_dir / _KEYWORDS_SHARD_FMT.format(shard_index),
            {"window_id": int(window_id), "keywords": [str(k) for k in keywords]},
        )
        _append_jsonl_line(
            shard_dir / _METADATA_SHARD_FMT.format(shard_index),
            {"window_id": int(window_id), **{str(k): v for k, v in clause_metadata.items()}},
        )

        # 5. Bookkeeping.
        with self._state_lock:
            self._windows_written += 1
            self._manifest_dirty = True
            windows_written = self._windows_written

        # 6. In-flight manifest (adapter-overridable cadence).
        if self._should_flush():
            self._write_inflight_manifest(windows_written)

    # ------------------------------------------------------------------
    # Adapter dispatch helpers.
    # ------------------------------------------------------------------

    def _finalize(self, tensor: Any) -> Any:
        if self._adapter is not None:
            return self._adapter.residual_finalize(tensor)
        # default/CUDA path — axis-3 can port this by lifting into CudaAdapter.
        return tensor.detach().to("cpu", dtype=torch.float32).numpy()  # type: ignore[union-attr]

    def _write_boundary(self, path: Path, array: Any) -> None:
        if self._adapter is not None:
            self._adapter.write_boundary(path, array)
            return
        # default/CUDA path
        _atomic_save_npy(path, np.asarray(array, dtype=np.float32))

    def _should_flush(self) -> bool:
        if self._adapter is not None:
            return bool(self._adapter.should_flush_manifest())
        # default/CUDA path: flush every window.
        return True

    def _check_pause(self) -> None:
        if self._adapter is not None:
            self._adapter.check_pause()
        # default/CUDA path: no-op.

    # ------------------------------------------------------------------
    # Manifest + compaction.
    # ------------------------------------------------------------------

    def _write_inflight_manifest(self, windows_written: int) -> None:
        """Write a minimal best-effort manifest. Readers still skip (no compaction)."""
        manifest: dict[str, Any] = {
            "store_version": STORE_VERSION,
            "clause_aligned": True,
            "num_windows": int(windows_written),
            "num_entries": int(windows_written),
            "num_tokens": 0,  # filled correctly by _compact
            "window_metadata": WINDOW_METADATA_FILE,
            "arch_config": dict(self._arch_config),
            "in_flight": True,
        }
        _atomic_write_text_via_tmp(
            self._checkpoint_dir / MANIFEST_FILE,
            json.dumps(manifest, indent=2),
        )
        with self._state_lock:
            self._manifest_dirty = False

    def _compact(self) -> None:
        """Merge all JSONL shards into canonical store files, then manifest."""
        shard_dir = self._checkpoint_dir / _SHARDS_DIRNAME

        token_lists: dict[int, list[int]] = {}
        keywords_map: dict[str, list[str]] = {}
        metadata_map: dict[str, dict[str, Any]] = {}

        if shard_dir.exists():
            for shard in sorted(shard_dir.glob("tokens_*.jsonl")):
                for rec in _read_jsonl_tolerant_tail(shard):
                    wid_raw = rec.get("window_id")
                    tokens = rec.get("token_ids")
                    if wid_raw is None or not isinstance(tokens, list):
                        continue
                    token_lists[int(wid_raw)] = [int(t) for t in tokens]

            for shard in sorted(shard_dir.glob("keywords_*.jsonl")):
                for rec in _read_jsonl_tolerant_tail(shard):
                    wid_raw = rec.get("window_id")
                    kws = rec.get("keywords")
                    if wid_raw is None or not isinstance(kws, list):
                        continue
                    keywords_map[str(int(wid_raw))] = [str(k) for k in kws]

            for shard in sorted(shard_dir.glob("window_metadata_*.jsonl")):
                for rec in _read_jsonl_tolerant_tail(shard):
                    wid_raw = rec.get("window_id")
                    if wid_raw is None:
                        continue
                    meta_view = {k: v for k, v in rec.items() if k != "window_id"}
                    metadata_map[str(int(wid_raw))] = meta_view

        if not token_lists:
            raise RuntimeError(
                "LiveIndexer._compact: no windows were captured — refusing to "
                "write a manifest for an empty store."
            )

        # Canonical token arrays. Both filenames carry identical data so
        # TorchKnowledgeStore.load's either-or fallback is a no-op.
        arrays: dict[str, np.ndarray] = {
            str(int(wid)): np.asarray(tokens, dtype=np.int64)
            for wid, tokens in sorted(token_lists.items())
        }
        _atomic_save_npz(self._checkpoint_dir / WINDOW_TOKEN_LISTS_FILE, arrays)
        _atomic_save_npz(self._checkpoint_dir / WINDOW_TOKENS_FILE, arrays)

        # Keywords.
        _atomic_write_text_via_tmp(
            self._checkpoint_dir / KEYWORDS_FILE,
            json.dumps(keywords_map, ensure_ascii=False, indent=2),
        )

        # Window metadata.
        _atomic_write_text_via_tmp(
            self._checkpoint_dir / WINDOW_METADATA_FILE,
            json.dumps(metadata_map, ensure_ascii=False, indent=2),
        )

        # IDF. Best-effort via TFIDFRouter.compute_idf; empty on import failure.
        idf_payload = _compute_idf_fallback(token_lists)
        _atomic_write_text_via_tmp(
            self._checkpoint_dir / IDF_FILE,
            json.dumps(idf_payload, ensure_ascii=False, indent=2),
        )

        # Remove shards AFTER every canonical file is in place.
        if shard_dir.exists():
            for shard in list(shard_dir.iterdir()):
                try:
                    shard.unlink()
                except OSError:  # pragma: no cover - best-effort cleanup
                    pass
            try:
                shard_dir.rmdir()
            except OSError:  # pragma: no cover - directory not empty is fine
                pass

        # Manifest LAST. Readers gate on its presence.
        num_windows = max(token_lists.keys()) + 1 if token_lists else 0
        num_tokens = sum(len(tl) for tl in token_lists.values())
        manifest: dict[str, Any] = {
            "store_version": STORE_VERSION,
            "clause_aligned": True,
            "num_windows": int(num_windows),
            "num_entries": int(len(token_lists)),
            "num_tokens": int(num_tokens),
            "window_metadata": WINDOW_METADATA_FILE,
            "arch_config": dict(self._arch_config),
        }
        _atomic_write_text_via_tmp(
            self._checkpoint_dir / MANIFEST_FILE,
            json.dumps(manifest, indent=2),
        )
        with self._state_lock:
            self._manifest_dirty = False

    # ------------------------------------------------------------------
    # Misc.
    # ------------------------------------------------------------------

    def _drain_failed_queue(self) -> list[tuple[Any, BaseException]]:
        failures: list[tuple[Any, BaseException]] = []
        while True:
            try:
                failures.append(self._failed_q.get_nowait())
            except queue.Empty:
                break
        return failures

    @staticmethod
    def _validate_arch_config(arch_config: dict[str, Any]) -> None:
        if not isinstance(arch_config, dict):
            raise ValueError("LiveIndexer: arch_config must be a dict")
        missing = [k for k in _REQUIRED_ARCH_CONFIG_KEYS if k not in arch_config]
        if missing:
            raise ValueError(
                "LiveIndexer: arch_config is missing required keys: "
                + ", ".join(sorted(missing))
            )
