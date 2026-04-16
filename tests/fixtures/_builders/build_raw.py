"""Build ``raw.jsonl`` — pre-prep raw text corpus."""

from __future__ import annotations

import json
from pathlib import Path


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    out_path = out_dir / "raw.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(256):
            fh.write(json.dumps({"text": f"raw sample {i} — seed {seed}."}) + "\n")
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
