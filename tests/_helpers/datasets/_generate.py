"""Deterministic generator for EWS-9 tiny fixtures.

Run (from repo root):

    uv run python -m tests._helpers.datasets._generate

Writes / overwrites three JSONL files in this directory from a single numpy
seed so the committed fixtures are byte-stable.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20240915
VOCAB = list(string.ascii_lowercase + " .,!?")


def _sample_tokens(rng: np.random.Generator, approx_len: int) -> str:
    length = max(4, int(rng.normal(approx_len, approx_len * 0.1)))
    idx = rng.integers(0, len(VOCAB), size=length)
    return "".join(VOCAB[i] for i in idx).strip() or "a"


def build_sft(rng: np.random.Generator, n: int = 200, avg_tokens: int = 64) -> list[dict]:
    rows = []
    for i in range(n):
        prompt = _sample_tokens(rng, max(8, avg_tokens // 4))
        response = _sample_tokens(rng, max(8, 3 * avg_tokens // 4))
        rows.append({"id": i, "prompt": prompt, "response": response})
    return rows


def build_pref(rng: np.random.Generator, n: int = 200) -> list[dict]:
    rows = []
    for i in range(n):
        prompt = _sample_tokens(rng, 16)
        chosen = _sample_tokens(rng, 48)
        rejected = _sample_tokens(rng, 48)
        rows.append({"id": i, "prompt": prompt, "chosen": chosen, "rejected": rejected})
    return rows


def build_prompts(rng: np.random.Generator, n: int = 100) -> list[dict]:
    return [{"id": i, "prompt": _sample_tokens(rng, 32)} for i in range(n)]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    _write_jsonl(HERE / "toy_sft_tiny.jsonl", build_sft(rng))
    _write_jsonl(HERE / "toy_pref_tiny.jsonl", build_pref(rng))
    _write_jsonl(HERE / "toy_prompts_tiny.jsonl", build_prompts(rng))


if __name__ == "__main__":
    main()
