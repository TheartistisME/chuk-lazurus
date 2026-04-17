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

    manifest = {
        "version": 12,
        "num_entries": 0,
        "num_windows": 9,
        "num_tokens": 26,
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
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _write_npz_mapping(
        root / "window_tokens.npz",
        {
            0: {1, 7},
            1: {1, 2},
            2: {3, 4, 5, 6},
            3: {10, 11, 30, 31},
            4: {10, 11, 12, 13, 14, 15},
            5: {20, 21, 22, 23, 24, 25},
            6: {11},
            7: {23},
            8: {24},
        },
    )
    np.savez(
        str(root / "window_token_lists.npz"),
        **{
            "0": np.array([1, 7], dtype=np.uint32),
            "1": np.array([1, 2], dtype=np.uint32),
            "2": np.array([3, 4, 5, 6], dtype=np.uint32),
            "3": np.array([10, 11, 30, 31], dtype=np.uint32),
            "4": np.array([10, 11, 12, 13, 14, 15], dtype=np.uint32),
            "5": np.array([20, 21, 22, 23, 24, 25], dtype=np.uint32),
            "6": np.array([11], dtype=np.uint32),
            "7": np.array([23], dtype=np.uint32),
            "8": np.array([24], dtype=np.uint32),
        },
    )
    (root / "idf.json").write_text(
        json.dumps(
            {
                1: 1.0,
                2: 1.0,
                3: 1.0,
                4: 1.0,
                5: 1.0,
                6: 1.0,
                7: 1.0,
                10: 1.0,
                11: 1.0,
                12: 1.0,
                13: 1.0,
                14: 1.0,
                15: 1.0,
                20: 1.0,
                21: 1.0,
                22: 1.0,
                23: 1.0,
                24: 1.0,
                25: 1.0,
                30: 1.0,
                31: 1.0,
            },
            indent=2,
        )
        + "\n"
    )
    (root / "keywords.json").write_text(
        json.dumps(
            {
                0: ["accessible"],
                1: ["accessible", "readily"],
                2: ["rcd"],
                3: ["basic", "protection"],
                4: ["basic", "protection"],
                5: ["faults", "live", "conductors"],
                6: ["protection"],
                7: ["live"],
                8: ["conductors"],
            },
            indent=2,
        )
        + "\n"
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
                "3": {
                    "clause_id": "1.5.4",
                    "clause_title": "Basic protection",
                    "part_index": 1,
                    "part_count": 1,
                },
                "4": {
                    "clause_id": "1.5.6.1",
                    "clause_title": "Basic protection",
                    "part_index": 1,
                    "part_count": 1,
                },
                "5": {
                    "clause_id": "2.6.1",
                    "clause_title": "General",
                    "part_index": 1,
                    "part_count": 1,
                },
                "6": {
                    "clause_id": "4.10.2",
                    "clause_title": "PROTECTION",
                    "part_index": 1,
                    "part_count": 1,
                },
                "7": {
                    "clause_id": "1.4.78",
                    "clause_title": "Live",
                    "part_index": 1,
                    "part_count": 1,
                },
                "8": {
                    "clause_id": "5.5.6.1",
                    "clause_title": "Conductors",
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


def test_comparison_prompt_returns_all_primary_windows(tmp_path):
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

    assert store.route("Accessible", tokenizer=tokenizer, method="auto") == 0


def test_rcd_prompts_fall_through_to_stronger_clause_evidence(tmp_path):
    store = TorchKnowledgeStore.load(_write_store(tmp_path / "clause_store"))
    tokenizer = SimpleTokenizer(
        {
            "accessible": 1,
            "readily": 2,
            "residual": 3,
            "current": 4,
            "device": 5,
            "rcd": 6,
            "rcds": 6,
            "basic": 10,
            "protection": 11,
            "sole": 12,
            "means": 13,
            "normal": 14,
            "service": 15,
            "recognized": 16,
            "provide": 20,
            "faults": 21,
            "between": 22,
            "live": 23,
            "conductors": 24,
        }
    )

    assert store.route(
        "Are RCDs recognized as a sole means of basic protection in normal service?",
        tokenizer=tokenizer,
        method="auto",
    ) == 4
    assert store.route(
        "Do RCDs provide protection against faults between live conductors?",
        tokenizer=tokenizer,
        method="auto",
    ) == 5


def test_clause_id_matches_ignore_ambiguous_titles_and_generic_aliases(tmp_path):
    store = TorchKnowledgeStore.load(_write_store(tmp_path / "clause_store"))
    tokenizer = SimpleTokenizer(
        {
            "rcd": 6,
            "rcds": 6,
            "basic": 10,
            "protection": 11,
            "sole": 12,
            "means": 13,
            "normal": 14,
            "service": 15,
            "recognized": 16,
        }
    )

    assert store.route(
        "Does clause 1.5.6.1 recognize RCDs as a sole means of basic protection in normal service?",
        tokenizer=tokenizer,
        method="auto",
    ) == 4
