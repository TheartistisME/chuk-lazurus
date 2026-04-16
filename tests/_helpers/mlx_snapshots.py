"""MLX regression snapshot helpers shared across Epic 2 tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_ROOT = Path(__file__).resolve().parent / "mlx_snapshots"
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


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "SNAPSHOT_ROOT",
    "assert_snapshot",
    "load_snapshot",
    "save_snapshot",
    "snapshot_path",
]
