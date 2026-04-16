from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore


class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.inverse = {v: k for k, v in vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = []
        for word in text.lower().replace("-", " ").replace(",", " ").split():
            token_id = self.vocab.get(word)
            if token_id is not None:
                tokens.append(token_id)
        if add_special_tokens:
            tokens = [999] + tokens + [998]
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = True):
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in {998, 999}:
                continue
            words.append(self.inverse.get(int(token_id), f"<{token_id}>"))
        return " ".join(words)


def _write_npz_mapping(path: Path, mapping: dict[int, list[int] | set[int]]) -> None:
    arrays = {str(key): np.array(sorted(values), dtype=np.uint32) for key, values in mapping.items()}
    np.savez(str(path), **arrays)


def _write_store(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 12,
                "num_entries": 0,
                "num_windows": 3,
                "num_tokens": 9,
                "entries_per_window": 8,
                "crystal_layer": 30,
                "window_size": 512,
                "arch_config": {
                    "retrieval_layer": 29,
                    "query_head": 4,
                    "injection_layer": 30,
                },
                "has_residuals": False,
                "window_metadata": "window_metadata.json",
            },
            indent=2,
        )
        + "\n"
    )

    _write_npz_mapping(
        root / "window_tokens.npz",
        {
            0: {1, 7},
            1: {1, 2},
            2: {3, 4, 5, 6},
        },
    )
    np.savez(
        str(root / "window_token_lists.npz"),
        **{
            "0": np.array([1, 7], dtype=np.uint32),
            "1": np.array([1, 2], dtype=np.uint32),
            "2": np.array([3, 4, 5, 6], dtype=np.uint32),
        },
    )
    (root / "idf.json").write_text(json.dumps({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}, indent=2) + "\n")
    (root / "keywords.json").write_text(
        json.dumps({0: ["accessible"], 1: ["accessible", "readily"], 2: ["rcd"]}, indent=2) + "\n"
    )
    (root / "window_metadata.json").write_text(
        json.dumps(
            {
                "0": {"clause_id": "1.4.2", "clause_title": "Accessible", "part_index": 1, "part_count": 1},
                "1": {
                    "clause_id": "1.4.3",
                    "clause_title": "Accessible, readily",
                    "part_index": 1,
                    "part_count": 1,
                },
                "2": {
                    "clause_id": "1.4.102",
                    "clause_title": "Residual current device (RCD)",
                    "part_index": 1,
                    "part_count": 1,
                },
            },
            indent=2,
        )
        + "\n"
    )
    return root


def test_exact_clause_id_and_normalized_title_routing(tmp_path):
    store = TorchKnowledgeStore.load(_write_store(tmp_path / "clause_store"))
    tokenizer = SimpleTokenizer(
        {
            "accessible": 1,
            "readily": 2,
            "residual": 3,
            "current": 4,
            "device": 5,
            "rcd": 6,
            "difference": 7,
            "between": 8,
        }
    )

    assert store.route("1.4.2", tokenizer=tokenizer, method="auto") == 0
    assert store.route("Accessible", tokenizer=tokenizer, method="auto") == 0
    assert store.route("Residual current device (RCD)", tokenizer=tokenizer, method="auto") == 2


def test_comparison_prompt_returns_all_exact_primary_windows(tmp_path):
    store = TorchKnowledgeStore.load(_write_store(tmp_path / "clause_store"))
    tokenizer = SimpleTokenizer(
        {
            "accessible": 1,
            "readily": 2,
            "difference": 7,
            "between": 8,
        }
    )

    routed = store.route_top_k("What is the difference between Accessible and Accessible, readily?", tokenizer, k=1)
    assert routed == [0, 1]


def test_exact_routing_beats_tf_idf_ties(tmp_path):
    store = TorchKnowledgeStore.load(_write_store(tmp_path / "clause_store"))
    tokenizer = SimpleTokenizer({"accessible": 1})

    # TF-IDF would prefer window 1 on a tie because it has the higher id, but
    # the exact title route must pin the prompt to the clause 1.4.2 window.
    assert store.route("Accessible", tokenizer=tokenizer, method="auto") == 0

