from __future__ import annotations

import importlib.util
import sys
import types

from chuk_lazarus.david import model_backend
from chuk_lazarus.david.model_backend import OfflineModelBackend, TransformersCausalLMBackend


def test_offline_backend_is_deterministic_and_applies_stop() -> None:
    backend = OfflineModelBackend(prefix="test")

    first = backend.generate("alpha beta gamma", max_new_tokens=2)
    second = backend.generate("alpha beta gamma", max_new_tokens=2)
    stopped = backend.generate("alpha STOP beta", stop=["STOP"])

    assert backend.status().loaded is True
    assert first == second
    assert first.text == "test: alpha beta"
    assert stopped.text == "test: alpha "


def test_transformers_backend_reports_missing_optional_packages(monkeypatch) -> None:
    def fake_find_spec(name: str):
        if name in {"torch", "transformers"}:
            return None
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    backend = TransformersCausalLMBackend("example/local-model")

    status = backend.status()
    result = backend.generate("hello")

    assert status.available is False
    assert "missing optional packages" in status.reason
    assert result.ok is False
    assert result.text == ""
    assert result.metadata["local_files_only"] is True


def test_transformers_backend_never_downloads_by_default() -> None:
    backend = TransformersCausalLMBackend("example/local-model")

    assert backend.local_files_only is True
    assert backend.status().metadata["local_files_only"] is True


def test_transformers_backend_loads_and_generates_with_fake_local_modules(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeTensor:
        def __init__(self) -> None:
            self.moved_to: str | None = None

        def to(self, device: str) -> "FakeTensor":
            self.moved_to = device
            return self

    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, FakeTensor]:
            calls.append(("tokenize", prompt, {"return_tensors": return_tensors}))
            return {"input_ids": FakeTensor()}

        def decode(self, output_ids: list[int], *, skip_special_tokens: bool) -> str:
            calls.append(("decode", repr(output_ids), {"skip_special_tokens": skip_special_tokens}))
            return "prompt answer STOP trailing"

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False
            self.device: str | None = None

        def to(self, device: str) -> "FakeModel":
            self.device = device
            calls.append(("model.to", device, {}))
            return self

        def eval(self) -> None:
            self.eval_called = True
            calls.append(("model.eval", "", {}))

        def generate(self, **kwargs: object) -> list[list[int]]:
            calls.append(("generate", "", kwargs))
            return [[1, 2, 3]]

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    class FakeNoGrad:
        def __enter__(self) -> None:
            calls.append(("no_grad.enter", "", {}))

        def __exit__(self, *exc: object) -> None:
            calls.append(("no_grad.exit", "", {}))

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerFactory
    fake_transformers.AutoModelForCausalLM = FakeModelFactory

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = FakeNoGrad

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(model_backend, "_missing_optional_packages", lambda *names: [])

    backend = TransformersCausalLMBackend(
        "local/test-model",
        device="cpu",
        torch_dtype="float16",
    )

    load_status = backend.load()
    fake_processor = object()
    result = backend.generate("prompt", max_new_tokens=7, stop=["STOP"], logits_processor=fake_processor)

    assert load_status.available is True
    assert load_status.loaded is True
    assert backend.tokenizer is not None
    assert result.ok is True
    assert result.text == "prompt answer "
    assert result.metadata["model_id"] == "local/test-model"
    assert result.metadata["local_files_only"] is True
    assert result.metadata["max_new_tokens"] == 7
    assert ("tokenizer", "local/test-model", {"local_files_only": True}) in calls
    assert ("model", "local/test-model", {"local_files_only": True, "torch_dtype": "float16"}) in calls
    assert ("model.to", "cpu", {}) in calls
    generate_call = next(call for call in calls if call[0] == "generate")
    assert generate_call[2]["max_new_tokens"] == 7
    assert generate_call[2]["input_ids"].moved_to == "cpu"
    assert generate_call[2]["logits_processor"] == [fake_processor]
    assert result.metadata["logits_processor_count"] == 1


def test_transformers_backend_reports_local_asset_load_errors(monkeypatch) -> None:
    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            raise OSError(f"no local files for {model_id}; local_files_only={kwargs.get('local_files_only')}")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerFactory
    fake_transformers.AutoModelForCausalLM = object

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(model_backend, "_missing_optional_packages", lambda *names: [])

    backend = TransformersCausalLMBackend("local/missing-model")
    result = backend.generate("hello")

    assert result.ok is False
    assert result.text == ""
    assert result.error is not None
    assert "OSError: no local files for local/missing-model" in result.error
    assert "local_files_only=True" in result.error
    assert result.metadata["local_files_only"] is True


def test_transformers_backend_reports_generation_errors(monkeypatch) -> None:
    class FakeTokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, object]:
            return {"input_ids": object()}

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            return FakeTokenizer()

    class FakeModel:
        def eval(self) -> None:
            pass

        def generate(self, **kwargs: object) -> list[list[int]]:
            raise RuntimeError("decode failed")

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            return FakeModel()

    class FakeNoGrad:
        def __enter__(self) -> None:
            pass

        def __exit__(self, *exc: object) -> None:
            pass

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerFactory
    fake_transformers.AutoModelForCausalLM = FakeModelFactory

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = FakeNoGrad

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(model_backend, "_missing_optional_packages", lambda *names: [])

    backend = TransformersCausalLMBackend("local/test-model")
    result = backend.generate("hello")

    assert result.ok is False
    assert result.text == ""
    assert result.error == "RuntimeError: decode failed"
    assert result.metadata["model_id"] == "local/test-model"
