"""Build ``vec.npz`` — float32 vectors + int32 token ids (Appendix A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build(out_dir: Path, *, model: str | None = None, seed: int = 0, hidden: int = 64) -> Path:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((32, hidden), dtype=np.float32)
    token_ids = rng.integers(0, 32_000, size=(32,), dtype=np.int32)
    out_path = out_dir / "vec.npz"
    np.savez(out_path, vectors=vectors, token_ids=token_ids)
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
