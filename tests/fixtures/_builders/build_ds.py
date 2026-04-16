"""Build ``ds.jsonl`` — positive/negative direction pairs."""

from __future__ import annotations

import json
from pathlib import Path


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    out_path = out_dir / "ds.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(64):
            fh.write(
                json.dumps(
                    {
                        "positive": f"I feel happy about item {i}.",
                        "negative": f"I feel sad about item {i}.",
                    }
                )
                + "\n"
            )
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
