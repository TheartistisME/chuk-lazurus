"""Build ``corpus.jsonl`` — 64 short deterministic documents."""

from __future__ import annotations

import json
from pathlib import Path


_SEED_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Large language models learn from sequences of tokens.",
    "MLX runs natively on Apple Silicon GPUs via Metal.",
    "PyTorch supports CUDA kernels on NVIDIA accelerators.",
    "Attention mechanisms compute weighted sums of values.",
    "Tokenization breaks text into subword units.",
    "Backpropagation computes gradients via the chain rule.",
    "Layer normalization stabilizes transformer training.",
]


def build(out_dir: Path, *, model: str | None = None, seed: int = 0) -> Path:
    out_path = out_dir / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(64):
            text = _SEED_SENTENCES[i % len(_SEED_SENTENCES)] + f" (doc {i})"
            fh.write(json.dumps({"id": f"doc-{i:03d}", "text": text}) + "\n")
    return out_path


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
