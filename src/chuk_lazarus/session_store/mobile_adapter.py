"""Mobile / battery-aware adapter for :class:`LiveIndexer`.

Proof-of-viability scope
------------------------
This module is a *desktop-runnable* implementation of the ``Adapter`` Protocol
single-owned by :mod:`chuk_lazarus.session_store.live_indexer`. It is not a
phone binary — it has no ExecuTorch / TorchScript / Android glue. Its purpose
is to demonstrate that the four-method Protocol surface is expressive enough
to accommodate a battery/thermal-aware backend without any change to
``LiveIndexer`` or the Protocol itself.

What it varies vs. the default (inlined) CUDA/file path
-------------------------------------------------------
* **Residual compression** — int8 symmetric per-tensor quantization of the
  fp32 CPU residual. For a hidden_size of ``H = 2304`` this takes a boundary
  from ~9 KB (``2304 × 4 B`` fp32) to ~2.4 KB (``2304 × 1 B`` int8 plus a
  single fp32 scale). Empirically this keeps cosine drift versus the fp32
  original at or below ~2 %; topical routing is TF-IDF and therefore
  unaffected by residual compression.
* **Persistence container** — ``residual_finalize`` returns a ``dict`` rather
  than an ``ndarray``. The Protocol docstring explicitly permits this
  ("persistable array/bytes") because ``write_boundary`` is the only consumer
  and it is symmetric with ``residual_finalize`` — both live on the adapter.
  The dict carries the int8 payload, the fp32 scale, and the original shape.
* **On-disk bytes** — ``write_boundary`` is handed a path whose suffix is
  ``.npy`` (``LiveIndexer`` hard-codes ``window_NNN.npy``) but writes
  ``np.savez``-formatted bytes into it. The Protocol docstring licenses this:
  *"The path suffix (default .npy) is under the adapter's control — mobile
  backends may choose a different container entirely."* ``LiveIndexer`` never
  reads boundaries back; mobile consumers own the read side of the contract.
* **Manifest cadence** — batched. The in-flight manifest is flushed once
  every ``N = 8`` windows **or** every ``60 s`` of wall-clock idle, whichever
  comes first. A crash before compaction loses at most ``N`` windows' worth
  of manifest visibility; the per-window shard writes themselves remain
  atomic so captured boundaries survive.
* **Pause / resume** — ``pause()`` / ``resume()`` expose a cooperative gate
  on top of a :class:`threading.Event`. ``check_pause()`` blocks on that
  gate, so a screen-off / wake-lock-yield orchestrator can stall the indexer
  between iterations without tearing down the worker.

Desktop-path non-regression
---------------------------
``MobileAdapter`` is strictly opt-in: callers wire it explicitly via
``LiveIndexer(..., adapter=MobileAdapter(...))``. Passing ``adapter=None``
continues to exercise the default CUDA/file path inlined in
``LiveIndexer``. Module import is side-effect-free; the only call that
touches global torch state (``torch.set_num_threads``) is made inside
``__init__`` and is guarded by the ``apply_torch_threads`` flag plus a
``try/except`` so a restricted thermal / sandboxed environment that rejects
the call cannot break the constructor.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:  # Torch is optional at import time — fp32 CPU tensors or ndarrays both work.
    import torch
except Exception:  # pragma: no cover - torch is a core dep elsewhere
    torch = None  # type: ignore[assignment]

from chuk_lazarus.session_store.live_indexer import Adapter  # noqa: F401  # Protocol consumption only.

__all__ = ["MobileAdapter"]


# ---------------------------------------------------------------------------
# Module-level constants.
# ---------------------------------------------------------------------------

_INT8_MAX: int = 127
"""Positive bound for symmetric int8 quantization (the ``-128`` bucket is unused)."""


# ---------------------------------------------------------------------------
# MobileAdapter.
# ---------------------------------------------------------------------------


class MobileAdapter:
    """Battery/thermal-aware adapter implementing :class:`Adapter`.

    See module docstring for the rationale of int8 residual quantization,
    the ``.npz``-at-``.npy``-path container choice, the ``N=8`` / ``60 s``
    manifest cadence, and the pause/resume semantics.

    Parameters
    ----------
    flush_every_n_windows:
        Manifest is flushed on the Nth call to :meth:`should_flush_manifest`
        since the previous flush. Default ``8``.
    flush_idle_seconds:
        Manifest is also flushed if at least this many seconds of wall-clock
        have elapsed since the previous flush, regardless of window count.
        Default ``60.0``.
    torch_num_threads:
        Value to pass to :func:`torch.set_num_threads` during ``__init__``.
        If ``None``, defaults to ``max(1, (os.cpu_count() or 2) - 1)`` so
        that the chat process retains at least one core.
    clock:
        Test seam. A zero-arg callable returning seconds as ``float``. If
        ``None``, :func:`time.monotonic` is used.
    apply_torch_threads:
        If ``False``, ``__init__`` does not call ``torch.set_num_threads``.
        Tests set this to avoid mutating the process-global torch state.
    """

    def __init__(
        self,
        *,
        flush_every_n_windows: int = 8,
        flush_idle_seconds: float = 60.0,
        torch_num_threads: int | None = None,
        clock: Callable[[], float] | None = None,
        apply_torch_threads: bool = True,
    ) -> None:
        if flush_every_n_windows <= 0:
            raise ValueError(
                "MobileAdapter: flush_every_n_windows must be positive, "
                f"got {flush_every_n_windows!r}"
            )
        if flush_idle_seconds <= 0:
            raise ValueError(
                "MobileAdapter: flush_idle_seconds must be positive, "
                f"got {flush_idle_seconds!r}"
            )

        self._flush_every_n_windows: int = int(flush_every_n_windows)
        self._flush_idle_seconds: float = float(flush_idle_seconds)

        # Resolve thread budget; leave at least one core for the chat process.
        if torch_num_threads is None:
            resolved_threads = max(1, (os.cpu_count() or 2) - 1)
        else:
            resolved_threads = max(1, int(torch_num_threads))
        self._resolved_num_threads: int = resolved_threads

        # Module-level torch state mutation is quarantined here, guarded
        # twice: behind ``apply_torch_threads`` (tests can disable) and
        # behind a try/except (restricted thermal/sandboxed environments
        # may reject the call).
        if apply_torch_threads and torch is not None:
            try:
                torch.set_num_threads(self._resolved_num_threads)
            except Exception:  # noqa: BLE001 - cannot let env rejection break __init__
                pass

        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

        # Pause/resume gate. ``set()`` ⇒ running, ``clear()`` ⇒ paused.
        self._pause_event: threading.Event = threading.Event()
        self._pause_event.set()

        # Flush-cadence state (mutated under ``_lock``).
        self._lock: threading.Lock = threading.Lock()
        self._windows_since_flush: int = 0
        self._last_flush_time: float = self._clock()

    # ------------------------------------------------------------------
    # Adapter Protocol surface.
    # ------------------------------------------------------------------

    def residual_finalize(self, tensor: Any) -> dict[str, Any]:
        """Int8 symmetric per-tensor quantization of the fp32 CPU residual.

        Returns a dict (``{"int8", "scale", "shape", "dtype_out"}``) rather
        than an ndarray because the lossless dequantization needs both the
        payload and the scalar scale. The Protocol docstring explicitly
        allows "array/bytes" — the dict is consumed by :meth:`write_boundary`
        which is single-owned by this same adapter, so the asymmetry never
        leaks out.
        """
        if torch is not None and isinstance(tensor, torch.Tensor):
            t = tensor.detach().to("cpu", dtype=torch.float32).numpy()
        else:
            t = np.asarray(tensor, dtype=np.float32)

        amax = float(np.abs(t).max()) if t.size > 0 else 0.0
        if amax == 0.0:
            scale = 1.0
        else:
            scale = amax / float(_INT8_MAX)

        q = np.round(t / scale).clip(-_INT8_MAX, _INT8_MAX).astype(np.int8)

        return {
            "int8": q,
            "scale": float(scale),
            "shape": tuple(int(d) for d in t.shape),
            "dtype_out": "int8",
        }

    def write_boundary(self, path: Path, array: Any) -> None:
        """Atomically persist the quantized residual at ``path``.

        The bytes written are ``np.savez`` format (int8 payload + fp32 scale
        + int64 shape); the path suffix is whatever ``LiveIndexer`` handed us
        (``.npy`` today). The Protocol docstring explicitly licenses this
        divergence. The atomic-write pattern matches
        :func:`chuk_lazarus.session_store.live_indexer._atomic_save_npy`
        exactly: sibling ``.tmp`` → ``flush`` → ``fsync`` → ``os.replace``.
        """
        if not isinstance(array, dict):
            raise TypeError(
                "MobileAdapter.write_boundary: expected dict from residual_finalize, "
                f"got {type(array).__name__}"
            )
        if "int8" not in array or "scale" not in array or "shape" not in array:
            raise ValueError(
                "MobileAdapter.write_boundary: finalized dict missing required keys "
                "(expected 'int8', 'scale', 'shape')"
            )

        tmp_path = path.with_name(path.name + ".tmp")
        with open(tmp_path, "wb") as fh:
            np.savez(
                fh,
                int8=np.asarray(array["int8"], dtype=np.int8),
                scale=np.float32(array["scale"]),
                shape=np.asarray(array["shape"], dtype=np.int64),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)

    def should_flush_manifest(self) -> bool:
        """Flush on every Nth window OR after ``flush_idle_seconds`` elapsed.

        Each call counts as one window's worth of "time has passed". When
        either threshold trips, both counters are reset atomically under
        ``self._lock`` so the two conditions cannot double-fire.
        """
        with self._lock:
            self._windows_since_flush += 1
            now = self._clock()
            elapsed = now - self._last_flush_time
            if (
                self._windows_since_flush >= self._flush_every_n_windows
                or elapsed >= self._flush_idle_seconds
            ):
                self._windows_since_flush = 0
                self._last_flush_time = now
                return True
            return False

    def check_pause(self) -> None:
        """Block until :meth:`resume` is called (no-op when already running)."""
        self._pause_event.wait()

    # ------------------------------------------------------------------
    # External pause/resume API (not part of :class:`Adapter`).
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Signal the indexer to stall at its next ``check_pause`` call."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Release any thread currently blocked inside :meth:`check_pause`."""
        self._pause_event.set()

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def dequantize(finalized: dict[str, Any]) -> np.ndarray:
        """Reconstruct the fp32 residual from a :meth:`residual_finalize` output.

        Used by tests to verify cosine drift against the pre-quantization
        tensor. Mobile consumers that read boundaries back implement their
        own reader; this helper is not on the hot path of
        :class:`LiveIndexer`, which does not read boundaries at all.
        """
        q = np.asarray(finalized["int8"], dtype=np.int8)
        scale = float(finalized["scale"])
        return q.astype(np.float32) * scale
