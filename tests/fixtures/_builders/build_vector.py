"""Build ``v.npy`` — unit-norm float32 steering vector."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build(out_dir: Path, *, model: str | None = None, seed: int = 0, hidden: int = 64) -> Path:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((hidden,), dtype=np.float32)
    v = v / np.linalg.norm(v)
    out_path = out_dir / "v.npy"
    np.save(out_path, v.astype(np.float32))
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
