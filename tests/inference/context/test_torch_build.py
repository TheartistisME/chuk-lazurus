from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig
from chuk_lazarus.inference.context.knowledge.torch_build import (
    _fill_entity_gaps,
    _generate_topic_expansion_ids,
    _select_targets,
    build_knowledge_store_torch,
)
from chuk_lazarus.inference.context.knowledge.torch_capture import (
    capture_post_crystal_boundary,
    capture_window_boundaries,
)
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore


class DummyTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.inverse = {idx: tok for tok, idx in vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False):
        tokens = []
        for word in text.lower().replace(",", " ").replace(".", " ").split():
            token_id = self.vocab.get(word)
            if token_id is not None:
                tokens.append(token_id)
        if add_special_tokens:
            return [999] + tokens + [998]
        return tokens

    def decode(self, token_ids, skip_special_tokens: bool = True):
        words = []
        for token_id in token_ids:
            if skip_special_tokens and int(token_id) in {998, 999}:
                continue
            words.append(self.inverse.get(int(token_id), f"<{int(token_id)}>"))
        return " ".join(words)


class DummyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)]
        )
        self.config = SimpleNamespace(model_type="gemma", num_hidden_layers=num_layers)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids, attention_mask=None, use_cache: bool = False):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return hidden


class DummyLMWithLogits(DummyCausalLM):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
        self.logits_proj = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache: bool = False):
        hidden = super().forward(input_ids, attention_mask=attention_mask, use_cache=use_cache)
        return SimpleNamespace(logits=self.logits_proj(hidden))


def test_build_modules_do_not_import_mlx_at_module_scope():
    torch_build_module = importlib.import_module(
        "chuk_lazarus.inference.context.knowledge.torch_build"
    )
    torch_capture_module = importlib.import_module(
        "chuk_lazarus.inference.context.knowledge.torch_capture"
    )

    assert torch_build_module.__dict__.get("mx") is None
    assert torch_capture_module.__dict__.get("mx") is None


def test_target_selection_keeps_gap_filling_semantics():
    chunk_ids = [101, 201, 202, 203, 204]
    idf = {201: 3.0, 203: 3.0, 202: 0.0, 204: 0.0}

    selected = _select_targets(chunk_ids, idf, min_k=2)

    assert selected == [(201, 1), (202, 2), (203, 3)]
    assert _fill_entity_gaps({1, 3}, chunk_ids, max_gap=2) == {1, 2, 3}


def test_capture_post_crystal_boundary_returns_cpu_float32_vector():
    tokenizer = DummyTokenizer({"apollo": 1, "launch": 2, "moon": 3})
    model = DummyCausalLM(vocab_size=8, hidden_size=6, num_layers=3)

    boundary = capture_post_crystal_boundary(model, tokenizer.encode("apollo launch moon"), crystal_layer=1)

    assert torch.is_tensor(boundary)
    assert boundary.dtype == torch.float32
    assert boundary.device.type == "cpu"
    assert boundary.ndim == 1
    assert boundary.shape[0] == 6


def test_capture_window_boundaries_chains_residual_between_windows():
    model = DummyCausalLM(vocab_size=16, hidden_size=6, num_layers=3)
    windows = [[1, 2, 3], [4, 5, 6]]

    boundaries, final_boundary = capture_window_boundaries(model, windows, crystal_layer=1)
    standalone_second = capture_post_crystal_boundary(model, windows[1], crystal_layer=1)

    assert set(boundaries) == {0, 1}
    assert torch.is_tensor(final_boundary)
    assert not torch.allclose(boundaries[1], standalone_second)


def test_build_knowledge_store_torch_writes_v12_layout_and_loads(tmp_path):
    tokenizer = DummyTokenizer(
        {
            "apollo": 1,
            "launch": 2,
            "moon": 3,
            "transcript": 4,
            "capsule": 5,
            "communicator": 6,
            "lunar": 7,
            "landing": 8,
            "window": 9,
            "two": 10,
        }
    )
    model = DummyCausalLM(vocab_size=32, hidden_size=8, num_layers=4)
    config = ArchitectureConfig(
        retrieval_layer=2,
        query_head=0,
        injection_layer=3,
        crystal_layer=2,
        window_size=6,
        entries_per_window=2,
    )

    corpus = (
        "Apollo launch moon transcript capsule communicator lunar landing "
        "window two Apollo launch moon transcript capsule communicator lunar landing"
    )
    output = tmp_path / "apollo_store"

    loaded = build_knowledge_store_torch(model, tokenizer, corpus, config, output)

    assert isinstance(loaded, TorchKnowledgeStore)
    assert loaded.num_windows == 3
    assert loaded.num_tokens == len(tokenizer.encode(corpus))
    assert loaded.config.crystal_layer == 2
    assert loaded.config.window_size == 6
    assert len(loaded.window_tokens) == 3
    assert len(loaded.window_token_lists) == 3
    assert loaded.route("apollo moon transcript", tokenizer=tokenizer, method="tfidf") == 0
    assert loaded.get_window_text(0, tokenizer).startswith("apollo launch moon")

    boundary = loaded.load_boundary(0)
    assert torch.is_tensor(boundary)
    assert boundary.dtype == torch.float32
    assert boundary.device.type == "cpu"
    assert boundary.shape == (8,)

    expected_files = {
        "manifest.json",
        "entries.npz",
        "window_tokens.npz",
        "window_token_lists.npz",
        "idf.json",
        "keywords.json",
        "boundary_residual.npy",
    }
    assert expected_files.issubset({path.name for path in output.iterdir() if path.is_file()})
    assert (output / "boundaries" / "window_000.npy").exists()
    assert (output / "boundaries" / "window_001.npy").exists()
    assert (output / "activation_routes.npy").exists()

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["version"] == 12
    assert manifest["num_windows"] == 3
    assert manifest["arch_config"]["crystal_layer"] == 2
    assert manifest["activation_routes"]["path"] == "activation_routes.npy"
    assert manifest["activation_routes"]["layer"] == 2
    assert manifest["activation_routes"]["shape"] == [3, 8]
    assert manifest["activation_routes"]["count"] == 3

    boundary_disk = np.load(str(output / "boundaries" / "window_000.npy"), allow_pickle=False)
    assert boundary_disk.dtype == np.float32
    assert boundary_disk.shape == (8,)

    routes_disk = np.load(str(output / "activation_routes.npy"), allow_pickle=False)
    assert routes_disk.dtype == np.float16
    assert routes_disk.shape == (3, 8)


def test_generate_topic_expansion_ids_uses_model_logits():
    tokenizer = DummyTokenizer(
        {
            "apollo": 1,
            "launch": 2,
            "topics:": 3,
            "booster": 4,
        }
    )
    model = DummyLMWithLogits(vocab_size=8, hidden_size=6, num_layers=2)
    with torch.no_grad():
        model.embed_tokens.weight.fill_(1.0)
        for layer in model.layers:
            layer.weight.copy_(torch.eye(layer.weight.shape[0]))
        model.logits_proj.weight.zero_()
        model.logits_proj.weight[4].fill_(1.0)

    expansion_ids = _generate_topic_expansion_ids(
        model,
        tokenizer,
        tokenizer.encode("apollo launch"),
        n_tokens=2,
    )

    assert expansion_ids == [4, 4]


def test_build_knowledge_store_torch_expands_routing_terms(tmp_path, monkeypatch):
    tokenizer = DummyTokenizer(
        {
            "apollo": 1,
            "launch": 2,
            "moon": 3,
            "landing": 4,
            "booster": 5,
            "capsule": 6,
        }
    )
    model = DummyCausalLM(vocab_size=16, hidden_size=6, num_layers=2)
    config = ArchitectureConfig(
        retrieval_layer=0,
        query_head=0,
        injection_layer=1,
        crystal_layer=0,
        window_size=2,
        entries_per_window=1,
    )
    def _expand_booster_for_second_window(_model, _tokenizer, window_token_ids, **_kwargs):
        return [5] if list(window_token_ids) == [3, 4] else []

    monkeypatch.setattr(
        "chuk_lazarus.inference.context.knowledge.torch_build._generate_topic_expansion_ids",
        _expand_booster_for_second_window,
    )

    store = build_knowledge_store_torch(
        model,
        tokenizer,
        "apollo launch moon landing",
        config,
        tmp_path / "store",
    )

    assert store.route("booster", tokenizer=tokenizer, method="tfidf") == 1
    assert store.route("booster", method="keyword") == 1
    assert store.idf[5] == pytest.approx(0.6931471805599453)


def test_build_knowledge_store_torch_can_use_max_tokens(tmp_path):
    tokenizer = DummyTokenizer({"apollo": 1, "launch": 2, "moon": 3, "transcript": 4})
    model = DummyCausalLM(vocab_size=16, hidden_size=4, num_layers=2)
    config = ArchitectureConfig(
        retrieval_layer=0,
        query_head=0,
        injection_layer=1,
        crystal_layer=0,
        window_size=4,
        entries_per_window=1,
    )

    store = build_knowledge_store_torch(
        model,
        tokenizer,
        "Apollo launch moon transcript Apollo launch moon transcript",
        config,
        tmp_path / "store",
        max_tokens=4,
    )

    assert store.num_windows == 1
    assert store.num_tokens == 4
