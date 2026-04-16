"""Build ``grpo.jsonl`` — GRPO reward-fn binding rows."""

from __future__ import annotations

import json
from pathlib import Path


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    out_path = out_dir / "grpo.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(16):
            fh.write(
                json.dumps({"prompt": f"Solve problem {i}.", "reward_fn": "noop_reward"}) + "\n"
            )
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
