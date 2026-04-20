"""MLX regression snapshot helpers shared across Epic 2 tests.

Two layers of support are provided:

* The legacy ``save_snapshot`` / ``load_snapshot`` / ``assert_snapshot``
  functions preserve the existing .npy-based golden workflow for tests
  that already depend on it.
* ``mlx_snapshot_assert`` is the parity oracle used by the dual-backend
  parity corpus. It records a JSON *residual entry* per op under
  ``tests/_helpers/mlx_snapshots/residuals.json`` and asserts the two
  arms fall inside the configured tolerance. The residual log is the
  committed artifact expected by the M-x1 deliverable (axis-3 parity
  baseline at ve-ins-0mo7014jk000083602f).

Residual schema (one entry appended per call)::

    {
      "op": "sft_loss",
      "max_abs_err": 2.384e-07,
      "rtol_violation_count": 0,
      "shape": [2, 5],
      "dtype": "float32",
      "input_hash": "b4f3...",
      "atol": 1e-05,
      "rtol": 1e-04,
      "timestamp": "2026-04-20T10:00:00Z"
    }
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "mlx_snapshots"
RESIDUAL_LOG = SNAPSHOT_ROOT / "residuals.json"
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4


def snapshot_path(name: str) -> Path:
    """Return the canonical on-disk path for an MLX regression snapshot."""

    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    if not name.endswith(".npy"):
        name = f"{name}.npy"
    return SNAPSHOT_ROOT / name


def save_snapshot(name: str, array: Any) -> Path:
    """Persist *array* as the golden snapshot for *name*."""

    path = snapshot_path(name)
    np.save(path, np.asarray(array))
    return path


def load_snapshot(name: str) -> np.ndarray:
    """Load the golden snapshot for *name*.  Raises ``FileNotFoundError`` if
    the snapshot has not been captured yet."""

    path = snapshot_path(name)
    if not path.exists():
        raise FileNotFoundError(f"MLX snapshot missing: {path}")
    return np.load(path)


def assert_snapshot(
    name: str,
    array: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    """Assert *array* matches the golden snapshot for *name* within tolerance.

    If the snapshot does not exist yet, save it (seed mode) so the next run
    performs a real comparison.  Tests relying on this helper should assume
    that the first run on a new host seeds the snapshot.
    """

    arr = np.asarray(array)
    path = snapshot_path(name)
    if not path.exists():
        np.save(path, arr)
        return
    expected = np.load(path)
    if expected.shape != arr.shape:
        raise AssertionError(
            f"MLX snapshot {name!r} shape mismatch: expected {expected.shape}, got {arr.shape}"
        )
    np.testing.assert_allclose(arr, expected, atol=atol, rtol=rtol)


def _to_numpy(x: Any) -> np.ndarray:
    """Coerce torch / mlx / numpy arrays to a contiguous numpy array."""

    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _hash_inputs(inputs: Iterable[Any] | None) -> str:
    if inputs is None:
        return ""
    h = hashlib.sha256()
    for item in inputs:
        arr = _to_numpy(item)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _residual_entry(
    op: str,
    mlx_values: Any,
    torch_values: Any,
    atol: float,
    rtol: float,
    inputs: Iterable[Any] | None,
) -> dict[str, Any]:
    a = _to_numpy(mlx_values)
    b = _to_numpy(torch_values)
    if a.shape != b.shape:
        max_abs_err = float("inf")
        rtol_violations = -1
    else:
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs_err = float(diff.max()) if diff.size else 0.0
        allowed = atol + rtol * np.abs(b.astype(np.float64))
        rtol_violations = int((diff > allowed).sum())
    return {
        "op": op,
        "max_abs_err": max_abs_err,
        "rtol_violation_count": rtol_violations,
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "input_hash": _hash_inputs(inputs),
        "atol": atol,
        "rtol": rtol,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _write_residual(entry: dict[str, Any]) -> None:
    """Append *entry* to the residual log file atomically-enough for tests."""

    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if RESIDUAL_LOG.exists():
        try:
            existing = json.loads(RESIDUAL_LOG.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    existing.append(entry)
    tmp = RESIDUAL_LOG.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, indent=2, sort_keys=True))
    os.replace(tmp, RESIDUAL_LOG)


def mlx_snapshot_assert(
    op: str,
    mlx_values: Any,
    torch_values: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    inputs: Iterable[Any] | None = None,
    err_msg: str | None = None,
) -> dict[str, Any]:
    """Compare MLX arm vs torch arm for *op* and persist the residual.

    Parameters
    ----------
    op:
        Canonical operation name (e.g. ``"sft_loss"`` or
        ``"circuit.edge_attribution"``). Used as the JSON key.
    mlx_values, torch_values:
        Arrays from each backend. Accepts numpy / torch / mlx arrays.
    atol, rtol:
        Tolerances forwarded to ``np.testing.assert_allclose``.
    inputs:
        Iterable of input tensors whose bytes are hashed into the
        residual entry so reviewers can tell when inputs changed.
    err_msg:
        Optional extra context prepended to the assertion message.

    Returns
    -------
    dict
        The residual entry that was written to disk.
    """

    entry = _residual_entry(op, mlx_values, torch_values, atol, rtol, inputs)
    _write_residual(entry)

    label = err_msg or op
    a = _to_numpy(mlx_values)
    b = _to_numpy(torch_values)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, err_msg=label)
    return entry


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "RESIDUAL_LOG",
    "SNAPSHOT_ROOT",
    "assert_snapshot",
    "load_snapshot",
    "mlx_snapshot_assert",
    "save_snapshot",
    "snapshot_path",
]
