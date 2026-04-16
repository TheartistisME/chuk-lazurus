from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "inference" / "build_knowledge_store_torch.py"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "apollo18_tiny.txt"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_knowledge_store_torch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    eos_token = "</s>"
    pad_token = None

    def __init__(self):
        self.texts = []

    def encode(self, text: str, add_special_tokens: bool = False):
        self.texts.append((text, add_special_tokens))
        return [11, 12, 13, 14]


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(model_type="gemma", num_hidden_layers=26)
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self


class FakeStore:
    def __init__(self):
        self.num_windows = 2
        self.num_tokens = 4
        self.config = SimpleNamespace(
            retrieval_layer=17,
            query_head=0,
            injection_layer=18,
            crystal_layer=18,
            window_size=512,
            entries_per_window=8,
        )


def test_example_builds_store_from_fixture_without_external_downloads(monkeypatch, tmp_path, capsys):
    module = _load_module()
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    fake_store = FakeStore()
    seen = {}

    def fake_load_model_and_tokenizer(model_id, device):
        fake_model.to("cpu").eval()
        return fake_model, fake_tokenizer, "cpu"

    monkeypatch.setattr(module, "_load_model_and_tokenizer", fake_load_model_and_tokenizer)
    monkeypatch.setattr(module, "_resolve_architecture_config", lambda model_config, **kwargs: SimpleNamespace(**{
        "retrieval_layer": 17,
        "query_head": 0,
        "injection_layer": 18,
        "crystal_layer": 18,
        "window_size": kwargs["window_size"] or 256,
        "entries_per_window": kwargs["entries_per_window"] or 4,
    }))

    def fake_build_knowledge_store_torch(*, model, tokenizer, raw_text, config, output_path, max_tokens):
        seen["model"] = model
        seen["tokenizer"] = tokenizer
        seen["raw_text"] = raw_text
        seen["config"] = config
        seen["output_path"] = Path(output_path)
        seen["max_tokens"] = max_tokens
        seen["output_path"].mkdir(parents=True, exist_ok=True)
        (seen["output_path"] / "manifest.json").write_text("{}\n")
        return fake_store

    monkeypatch.setattr(module, "build_knowledge_store_torch", fake_build_knowledge_store_torch)

    output_path = tmp_path / "apollo18_store"
    exit_code = module.main(
        [
            "--model",
            "google/gemma-3-1b-it",
            "--input",
            str(FIXTURE_PATH),
            "--output",
            str(output_path),
            "--device",
            "cuda",
            "--max-tokens",
            "32",
            "--window-size",
            "256",
            "--entries-per-window",
            "4",
        ]
    )

    captured = capsys.readouterr()
    expected_text = FIXTURE_PATH.read_text(encoding="utf-8")

    assert exit_code == 0
    assert seen["model"] is fake_model
    assert seen["tokenizer"] is fake_tokenizer
    assert seen["raw_text"] == expected_text
    assert seen["output_path"] == output_path
    assert seen["max_tokens"] == 32
    assert seen["config"].window_size == 256
    assert seen["config"].entries_per_window == 4
    assert fake_model.device == "cpu"
    assert fake_model.evaluated is True
    assert "Torch knowledge-store build" in captured.out
    assert "Model: google/gemma-3-1b-it" in captured.out
    assert f"Output: {output_path}" in captured.out
    assert "Store version: 12" in captured.out
    assert "Store ready for the read-only torch runner contract" in captured.out
