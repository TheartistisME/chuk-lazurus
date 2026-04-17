from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore

REAL_STORE = Path("/tmp/apollo-demo/apollo11_store")


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


def _write_window_metadata(path: Path, metadata: dict[int, dict[str, object]]) -> None:
    payload = {str(window_id): value for window_id, value in metadata.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_synthetic_store(root: Path, window_metadata: dict[int, dict[str, object]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "boundaries").mkdir(exist_ok=True)

    manifest = {
        "version": 12,
        "num_entries": 0,
        "num_windows": 2,
        "num_tokens": 6,
        "entries_per_window": 8,
        "crystal_layer": 30,
        "window_size": 512,
        "arch_config": {
            "retrieval_layer": 29,
            "query_head": 4,
            "injection_layer": 30,
        },
        "has_residuals": False,
    }
    if window_metadata is not None:
        manifest["num_windows"] = max(manifest["num_windows"], max(window_metadata.keys(), default=-1) + 1)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    _write_npz_mapping(
        root / "window_tokens.npz",
        {
            0: {1, 2, 3, 4},
            1: {5, 6},
        },
    )
    np.savez(
        str(root / "window_token_lists.npz"),
        **{
            "0": np.array([1, 2, 3, 4], dtype=np.uint32),
            "1": np.array([5, 6], dtype=np.uint32),
        },
    )
    (root / "idf.json").write_text(
        json.dumps({1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0}, indent=2) + "\n"
    )
    (root / "keywords.json").write_text(
        json.dumps({0: ["apollo", "launch"], 1: ["coyle", "transcript"]}, indent=2) + "\n"
    )
    if window_metadata is not None:
        _write_window_metadata(root / "window_metadata.json", window_metadata)
    np.savez(
        str(root / "entries.npz"),
        entries=np.array(
            [],
            dtype=[
                ("token_id", np.uint32),
                ("coefficient", np.float32),
                ("window_id", np.uint16),
                ("position_in_window", np.uint16),
                ("fact_id", np.uint16),
            ],
        ),
    )
    np.save(str(root / "boundaries" / "window_000.npy"), np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32))
    np.save(str(root / "boundaries" / "window_001.npy"), np.array([[[5.0, 6.0, 7.0, 8.0]]], dtype=np.float32))
    np.save(str(root / "boundary_residual.npy"), np.array([[[9.0, 10.0, 11.0, 12.0]]], dtype=np.float32))
    return root


def test_module_does_not_import_mlx_at_module_scope():
    import chuk_lazarus.inference.context.knowledge.torch_store as torch_store_module

    assert not hasattr(torch_store_module, "mx")


def test_load_real_apollo_store_if_available():
    if not REAL_STORE.exists():
        pytest.skip("Apollo store not present")

    store = TorchKnowledgeStore.load(REAL_STORE)

    assert store.num_windows == 176
    assert store.num_tokens == 90000
    assert store.config.retrieval_layer == 29
    assert store.config.query_head == 4
    assert store.config.crystal_layer == 30
    assert len(store.window_tokens) == 176
    assert len(store.window_token_lists) == 176
    assert len(store.keywords) == 176
    assert len(store.window_token_lists[0]) == 512

    boundary = store.load_boundary(0)
    assert torch.is_tensor(boundary)
    assert boundary.dtype == torch.float32
    assert boundary.ndim == 1
    assert boundary.shape[0] == 2560
    assert boundary.device.type == "cpu"

    if torch.cuda.is_available():
        boundary_cuda = store.load_boundary(0, device="cuda")
        assert boundary_cuda.device.type == "cuda"
        assert boundary_cuda.shape == boundary.shape


def test_synthetic_store_routes_and_loads_boundaries(tmp_path):
    store = TorchKnowledgeStore.load(_make_synthetic_store(tmp_path / "apollo_store"))
    tokenizer = SimpleTokenizer(
        {
            "apollo": 1,
            "launch": 2,
            "coyle": 5,
            "transcript": 6,
            "discussion": 7,
        }
    )

    assert store.route("apollo launch", tokenizer=tokenizer, method="tfidf") == 0
    assert store.route("coyle discussion", method="keyword") == 1
    assert store.get_window_text(1, tokenizer) == "coyle transcript"
    assert store.route_top_k("apollo launch", tokenizer, k=2, expansion_ids=[5, 6]) == [0, 1]
    assert store.window_metadata == {}

    boundary = store.load_boundary(0)
    assert boundary.shape == (4,)
    assert boundary.dtype == torch.float32
    assert float(boundary[0]) == 1.0

    if torch.cuda.is_available():
        boundary_cuda = store.load_boundary(1, device="cuda")
        assert boundary_cuda.device.type == "cuda"
        assert boundary_cuda.shape == (4,)


def test_log_stats_reports_torch_store(capsys, tmp_path):
    store = TorchKnowledgeStore.load(_make_synthetic_store(tmp_path / "apollo_store"))

    store.log_stats(file=sys.stdout)

    captured = capsys.readouterr()
    assert "TorchKnowledgeStore v12" in captured.out
    assert "2 windows" in captured.out


def test_loads_window_metadata_and_routes_exact_clauses(tmp_path):
    metadata = {
        0: {"clause_id": "1.4.2", "clause_title": "Accessible", "part_index": 1, "part_count": 1},
        1: {
            "clause_id": "1.4.3",
            "clause_title": "Accessible, readily",
            "part_index": 1,
            "part_count": 1,
        },
        2: {
            "clause_id": "1.4.102",
            "clause_title": "Residual current device (RCD)",
            "part_index": 1,
            "part_count": 1,
        },
    }
    store = TorchKnowledgeStore.load(_make_synthetic_store(tmp_path / "clause_store", metadata))
    tokenizer = SimpleTokenizer(
        {
            "accessible": 1,
            "readily": 2,
            "residual": 3,
            "current": 4,
            "device": 5,
            "rcd": 6,
        }
    )

    assert store.window_metadata[0]["clause_id"] == "1.4.2"
    assert store.route("1.4.2", tokenizer=tokenizer, method="auto") == 0
    assert store.route("Accessible", tokenizer=tokenizer, method="auto") == 0
    assert store.route("Residual current device (RCD)", tokenizer=tokenizer, method="auto") == 2
    assert store.route_top_k("Accessible and Accessible, readily", tokenizer, k=1) == [0, 1]
