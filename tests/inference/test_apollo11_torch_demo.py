from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "inference" / "demo_c_apollo11_torch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("demo_c_apollo11_torch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    eos_token = "</s>"
    pad_token = None

    def __init__(self):
        self.apply_chat_template_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.apply_chat_template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        parts = [message["content"] for message in messages]
        return " || ".join(parts) + (" || <model>" if add_generation_prompt else "")


class FakeStore:
    def __init__(self, window_keywords=None, window_text="Apollo 11 transcript excerpt"):
        self.route_top_k_calls = []
        self.route_calls = []
        self.load_boundary_calls = []
        self.window_token_lists = {7: [11, 12, 13]}
        self.num_tokens = 999
        self.config = SimpleNamespace(crystal_layer=23, window_size=512)
        self.keywords = {7: window_keywords or ["capsule", "communicator", "sound"]}
        self._window_text = window_text

    def route_top_k(self, question, tokenizer, k=1):
        self.route_top_k_calls.append((question, tokenizer, k))
        return [7]

    def route(self, question, tokenizer=None, method="keyword"):
        self.route_calls.append((question, tokenizer, method))
        return 7

    def get_window_text(self, window_id, tokenizer):
        assert window_id == 7
        assert tokenizer is fake_tokenizer
        return self._window_text

    def load_boundary(self, window_id, device="cpu"):
        self.load_boundary_calls.append((window_id, device))
        return [1.0, 2.0, 3.0]


class FakeStats:
    summary = "Generated 3 tokens in 0.01s (300.0 tok/s)"


class FakeResult:
    def __init__(self, text):
        self.text = text
        self.stats = FakeStats()


class FakeRuntime:
    backend = SimpleNamespace(value="cuda")

    def __init__(self, hidden_size=3, layer_count=24):
        self.calls = []
        self._model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size))
        self._layer_count = layer_count

    def _resolve_layers(self):
        return [object() for _ in range(self._layer_count)]

    def generate_with_residual(self, prompt, residual_state, config):
        self.calls.append(
            {
                "mode": "residual",
                "prompt": prompt,
                "residual_state": residual_state,
                "config": config,
            }
        )
        return FakeResult("The capsule communicator sounded like a living room.")

    def generate(self, prompt, config):
        self.calls.append(
            {
                "mode": "prompt-context",
                "prompt": prompt,
                "config": config,
            }
        )
        return FakeResult("The capsule communicator sounded like a living room.")


fake_tokenizer = FakeTokenizer()


def test_run_apollo_demo_routes_and_injects(monkeypatch, capsys):
    module = _load_module()
    store = FakeStore()
    runtime = FakeRuntime()
    boundary_tensor = SimpleNamespace(shape=(1, 3), dtype="float32")

    monkeypatch.setattr(module, "load_apollo_store", lambda store_path: store)
    monkeypatch.setattr(module, "_load_torch_runtime", lambda model_id, device: (runtime, fake_tokenizer))
    monkeypatch.setattr(module, "_as_cpu_torch_tensor", lambda value: boundary_tensor)

    exit_code = module.run_apollo_demo(
        store_path="/tmp/apollo-demo/apollo11_store",
        model_id="/models/local-gemma4-e2b",
        device="cuda",
        question="During the early launch phase of Apollo 11, what did the commander say about how the capsule communicator sounded?",
        max_new_tokens=64,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Apollo 11 torch/CUDA demo" in captured.out
    assert "Mode: residual" in captured.out
    assert "Routed window: 7" in captured.out
    assert "living room" in captured.out

    assert store.route_top_k_calls == []
    assert store.route_calls == [
        (
            "During the early launch phase of Apollo 11, what did the commander say about how the capsule communicator sounded?",
            None,
            "keyword",
        )
    ]
    assert store.load_boundary_calls == [(7, "cpu")]

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["mode"] == "residual"
    assert "Apollo 11 transcript excerpt" in call["prompt"]
    assert call["residual_state"].backend.value == "cuda"
    assert call["residual_state"].layer_index == 23
    assert call["residual_state"].device == "cpu"
    assert call["residual_state"].hidden_size == 3
    assert call["config"].max_new_tokens == 64
    assert call["config"].temperature == 0.0


def test_run_apollo_demo_falls_back_to_prompt_context(monkeypatch, capsys):
    module = _load_module()
    store = FakeStore(window_keywords=["capsule", "communicator"], window_text="Apollo window excerpt")
    runtime = FakeRuntime(hidden_size=4, layer_count=24)
    boundary_tensor = SimpleNamespace(shape=(1, 3), dtype="float32")

    monkeypatch.setattr(module, "load_apollo_store", lambda store_path: store)
    monkeypatch.setattr(module, "_load_torch_runtime", lambda model_id, device: (runtime, fake_tokenizer))
    monkeypatch.setattr(module, "_as_cpu_torch_tensor", lambda value: boundary_tensor)

    exit_code = module.run_apollo_demo(
        store_path="/tmp/apollo-demo/apollo11_store",
        model_id="/models/local-mlp-or-causal-lm",
        device="cuda",
        question="During the early launch phase of Apollo 11, what did the commander say about how the capsule communicator sounded?",
        max_new_tokens=32,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Mode: prompt-context" in captured.out

    assert store.route_calls == [
        (
            "During the early launch phase of Apollo 11, what did the commander say about how the capsule communicator sounded?",
            None,
            "keyword",
        )
    ]
    assert store.load_boundary_calls == [(7, "cpu")]
    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    assert call["mode"] == "prompt-context"
    assert "Apollo window excerpt" in call["prompt"]
    assert "Apollo store keywords: capsule, communicator" in call["prompt"]
    assert call["config"].max_new_tokens == 32


def test_build_parser_accepts_local_model_path():
    module = _load_module()
    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--question",
            "What did Bruce McCandless hear?",
            "--model",
            "/models/local-gemma4-e2b",
        ]
    )

    assert args.store == "/tmp/apollo-demo/apollo11_store"
    assert args.model == "/models/local-gemma4-e2b"
    assert args.device == "cuda"
    assert args.max_new_tokens == 64


def test_help_does_not_require_runtime(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "load_apollo_store", lambda *_args, **_kwargs: None)
    parser = module.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
