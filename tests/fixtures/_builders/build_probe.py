"""Build ``probe.jsonl`` — balanced binary-labeled short texts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_path = out_dir / "probe.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(128):
            label = int(i % 2)
            noise = rng.integers(0, 1000)
            text = f"probe-{label}-{noise}"
            fh.write(json.dumps({"text": text, "label": label}) + "\n")
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
