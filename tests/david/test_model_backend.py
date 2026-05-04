from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types

from chuk_lazarus.david import model_backend
from chuk_lazarus.david.config import DavidConfig
from chuk_lazarus.david.model_backend import OfflineModelBackend, TransformersCausalLMBackend, VindexArtifactBackend
from chuk_lazarus.david.runtime import DavidRuntime


def test_offline_backend_is_deterministic_and_applies_stop() -> None:
    backend = OfflineModelBackend(prefix="test")

    first = backend.generate("alpha beta gamma", max_new_tokens=2)
    second = backend.generate("alpha beta gamma", max_new_tokens=2)
    stopped = backend.generate("alpha STOP beta", stop=["STOP"])

    assert backend.status().loaded is True
    assert first == second
    assert first.text == "test: alpha beta"
    assert stopped.text == "test: alpha "


def test_offline_backend_records_replay_plan_as_refused_metadata() -> None:
    backend = OfflineModelBackend(prefix="test")
    plan = {
        "version": 1,
        "strategy": "kv_sidecar",
        "requested_strategy": "kv_sidecar",
        "requires_runtime_replay": True,
        "memory_family": "task",
        "tier": "hot",
        "session_id": "s1",
        "runtime_replay": {
            "strategy": "kv_sidecar",
            "requires_runtime_replay": True,
            "required_capability": "materialization.replay.kv_cache.v1",
            "adapter_scope": {
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
            },
            "memory_family": "task",
        },
        "sidecars": [{"artifact_id": "hot-window-1"}],
        "replay_refs": [{"kind": "kv_cache"}],
    }

    result = backend.generate("alpha", materialization_plan=plan)

    replay = result.metadata["materialization_replay"]
    assert result.ok is True
    assert replay["attempted"] is True
    assert replay["refused"] is True
    assert replay["ignored"] is True
    assert replay["applied"] is False
    assert replay["tensor_replay"] is False
    assert replay["required_capability"] == "materialization.replay.kv_cache.v1"
    assert replay["plan"]["sidecar_count"] == 1
    assert "runtime replay consumer required for kv_sidecar" in replay["reason"]
    assert "offline-deterministic has no tensor replay hook for kv_sidecar" in replay["reason"]


def test_offline_backend_records_replay_consumer_without_plan() -> None:
    backend = OfflineModelBackend(prefix="test")

    result = backend.generate("alpha", replay_consumer={"consumer_id": "metadata-only"})

    replay = result.metadata["materialization_replay"]
    assert replay["attempted"] is False
    assert replay["refused"] is False
    assert replay["ignored"] is True
    assert replay["consumer"]["consumer_id"] == "metadata-only"
    assert replay["reason"] == "no runtime replay required"


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
    fake_torch.float16 = "torch.float16"

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
    materialization_plan = {
        "version": 1,
        "strategy": "residual_sidecar",
        "requested_strategy": "residual_sidecar",
        "requires_runtime_replay": True,
        "memory_family": "task",
        "runtime_replay": {
            "strategy": "residual_sidecar",
            "requires_runtime_replay": True,
            "required_capability": "materialization.replay.residual_stream.v1",
            "adapter_scope": {
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
            },
            "memory_family": "task",
        },
    }
    replay_consumer = {
        "consumer_id": "compatible-replay-hook",
        "capabilities": ["materialization.replay.residual_stream.v1"],
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_families": ["kv_direct"],
        "memory_families": ["task"],
    }
    result = backend.generate(
        "prompt",
        max_new_tokens=7,
        stop=["STOP"],
        logits_processor=fake_processor,
        materialization_plan=materialization_plan,
        replay_consumer=replay_consumer,
    )

    assert load_status.available is True
    assert load_status.loaded is True
    assert backend.tokenizer is not None
    assert result.ok is True
    assert result.text == "prompt answer "
    assert result.metadata["model_id"] == "local/test-model"
    assert result.metadata["local_files_only"] is True
    assert result.metadata["device"] == "cpu"
    assert result.metadata["requested_dtype"] == "float16"
    assert result.metadata["resolved_dtype"] == "float16"
    assert result.metadata["max_new_tokens"] == 7
    assert ("tokenizer", "local/test-model", {"local_files_only": True}) in calls
    assert ("model", "local/test-model", {"local_files_only": True, "torch_dtype": "torch.float16"}) in calls
    assert ("model.to", "cpu", {}) in calls
    generate_call = next(call for call in calls if call[0] == "generate")
    assert generate_call[2]["max_new_tokens"] == 7
    assert generate_call[2]["input_ids"].moved_to == "cpu"
    assert generate_call[2]["logits_processor"] == [fake_processor]
    assert "materialization_plan" not in generate_call[2]
    assert "replay_consumer" not in generate_call[2]
    assert result.metadata["logits_processor_count"] == 1
    replay = result.metadata["materialization_replay"]
    assert replay["backend"] == "transformers-causal-lm"
    assert replay["required_capability"] == "materialization.replay.residual_stream.v1"
    assert replay["consumer"]["consumer_id"] == "compatible-replay-hook"
    assert replay["refused"] is True
    assert replay["applied"] is False
    assert replay["tensor_replay"] is False
    assert replay["refusal_reasons"] == [
        "transformers-causal-lm has no tensor replay hook for residual_sidecar"
    ]


def test_transformers_backend_rejects_invalid_dtype_without_loading(monkeypatch) -> None:
    monkeypatch.setattr(model_backend, "_missing_optional_packages", lambda *names: [])
    backend = TransformersCausalLMBackend("local/test-model", torch_dtype="float64")

    status = backend.status()
    result = backend.generate("hello")

    assert status.available is False
    assert status.loaded is False
    assert "invalid torch dtype 'float64'" in status.reason
    assert status.metadata["requested_dtype"] == "float64"
    assert status.metadata["device"] is None
    assert status.metadata["local_files_only"] is True
    assert result.ok is False
    assert result.error == status.reason


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


def test_vindex_artifact_backend_reports_available_but_not_decodable(tmp_path: Path) -> None:
    artifact = _write_vindex_artifact(tmp_path)
    backend = VindexArtifactBackend(str(artifact))

    status = backend.status()
    result = backend.generate("hello")

    assert status.name == "vindex-artifact"
    assert status.available is True
    assert status.loaded is False
    assert "not directly loadable by transformers-causal-lm" in status.reason
    assert status.metadata["family"] == "gemma4"
    assert status.metadata["source_hf_path"] == "google/gemma-4-E2B-it"
    assert status.metadata["hidden_size"] == 1536
    assert status.metadata["layers"] == 35
    assert status.metadata["vocab"] == 262144
    assert status.metadata["dtype"] == "f32"
    assert status.metadata["direct_generation"] is False
    assert result.ok is False
    assert result.text == ""
    assert result.backend == "vindex-artifact"
    assert result.metadata["loadable_by_transformers_causal_lm"] is False


def test_runtime_selects_vindex_backend_before_transformers_autoload(tmp_path: Path) -> None:
    artifact = _write_vindex_artifact(tmp_path)
    runtime = DavidRuntime.create(
        DavidConfig(
            workspace_root=tmp_path / "workspace",
            state_dir=tmp_path / "state",
            model_path=str(artifact),
            require_validated_model=False,
        )
    )

    readiness = runtime.readiness()

    assert isinstance(runtime.backend, VindexArtifactBackend)
    assert readiness["backend"].startswith("vindex-artifact: available for methodology/materialization evidence")


def _write_vindex_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "gemma4-e2b.vindex.ple"
    artifact.mkdir()
    (artifact / "index.json").write_text(
        json.dumps(
            {
                "family": "gemma4",
                "source": {"huggingface_repo": "google/gemma-4-E2B-it"},
                "hidden_size": 1536,
                "num_layers": 35,
                "vocab_size": 262144,
                "dtype": "f32",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact / "weight_manifest.json").write_text(
        json.dumps([{"key": "layers.0.self_attn.q_proj.weight", "file": "attn_weights.bin"}]),
        encoding="utf-8",
    )
    (artifact / "attn_weights.bin").write_bytes(b"weights")
    return artifact
