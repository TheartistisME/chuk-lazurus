from __future__ import annotations

from pathlib import Path
import json
import sys
import types

from chuk_lazarus.david import torch_backend
from chuk_lazarus.david.config import AdapterSessionMetadata, DavidConfig
from chuk_lazarus.david.runtime import DavidRuntime
from chuk_lazarus.david.torch_backend import TorchRuntimeModelBackend


def _validation_report() -> dict[str, object]:
    selected_config = {
        "adapter_config_id": "gemma-runtime-layer-23",
        "route_layer": 11,
        "route_query_head": 3,
        "route_dimension": 2048,
        "boundary_layer": 17,
        "residual_capture_layer": 17,
        "kv_source_layer": 21,
        "kv_target_layer": 23,
        "injection_layer": 23,
        "projection_producer_layer": 21,
        "behavior_cache_layer": 21,
        "insertion_family": "kv_direct",
        "kv_layout": "bshd",
        "candidate_role": "behavioral",
    }
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": "accepted",
        "confidence": "high",
        "validation_level": "behavioral",
        "auto_load_allowed": True,
        "harness_load_policy": "auto",
        "selected_config": selected_config,
        "source_report_summary": {
            "model_identity": "gemma-runtime-test",
            "tokenizer_identity": "gemma-runtime-tokenizer",
            "adapter_family": "gemma",
            "model_revision_or_hash": "runtime-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "model_identity_gate": {
            "model_identity": "gemma-runtime-test",
            "tokenizer_identity": "gemma-runtime-tokenizer",
            "adapter_family": "gemma",
            "model_revision_or_hash": "runtime-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "topology_gate": {"accepted": True},
        "projection_gate": {"ranked_candidates": []},
        "behavior_gate": {"accepted": True},
        "report_integrity": {"accepted": True},
        "provenance": {"loader_options": {"model": "gemma-runtime-test"}},
        "warnings": [],
    }


def test_torch_runtime_backend_reports_missing_optional_packages(monkeypatch) -> None:
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: list(names),
    )
    backend = TorchRuntimeModelBackend("local/model")

    status = backend.status()
    result = backend.generate("hello")

    assert status.available is False
    assert status.loaded is False
    assert "missing torch-runtime dependencies: torch, transformers, pydantic" in status.reason
    assert status.metadata["dependency_check"]["missing_packages"] == [
        "torch",
        "transformers",
        "pydantic",
    ]
    assert result.ok is False
    assert result.error == status.reason
    assert result.metadata["local_files_only"] is True
    assert result.metadata["device"] == "cuda"


def test_torch_runtime_backend_reports_local_generation_dependency_errors(monkeypatch) -> None:
    def fake_import_module(name: str) -> types.ModuleType:
        if name == "chuk_lazarus.inference.generation":
            raise ModuleNotFoundError("No module named 'pydantic'", name="pydantic")
        return types.ModuleType(name)

    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: [],
    )
    monkeypatch.setattr(torch_backend.importlib, "import_module", fake_import_module)

    backend = TorchRuntimeModelBackend("local/model", device="cpu")

    status = backend.status()
    result = backend.generate("hello")

    assert status.available is False
    assert status.loaded is False
    assert status.reason == (
        "torch-runtime dependency import failed: "
        "chuk_lazarus.inference.generation: missing dependency pydantic"
    )
    assert status.metadata["dependency_check"]["required_modules"] == [
        "chuk_lazarus.inference.generation",
        "chuk_lazarus.inference.backends.torch_runtime",
    ]
    assert status.metadata["dependency_check"]["import_errors"] == [
        "chuk_lazarus.inference.generation: missing dependency pydantic"
    ]
    assert result.ok is False
    assert result.error == status.reason


def test_torch_runtime_backend_loads_local_model_and_uses_standard_generation_config(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []
    generation_configs: list[dict[str, object]] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda
    fake_torch.bfloat16 = "torch.bfloat16"

    class FakeTokenizer:
        pass

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.device: str | None = None

        def to(self, device: str, **kwargs: object) -> "FakeModel":
            self.device = device
            calls.append(("model.to", device, kwargs))
            return self

        def eval(self) -> None:
            calls.append(("model.eval", "", {}))

    class FakeModelFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeModel:
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTokenizerFactory
    fake_transformers.AutoModelForCausalLM = FakeModelFactory

    class FakeGenerationConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            generation_configs.append(kwargs)

    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")
    fake_generation.GenerationConfig = FakeGenerationConfig

    class FakeStats:
        def model_dump(self) -> dict[str, object]:
            return {"input_tokens": 3, "output_tokens": 2}

    class FakeGenerationResult:
        text = "torch says hello STOP trailing"
        stats = FakeStats()
        stop_reason = "max_tokens"

    class FakeTorchInferenceRuntime:
        def __init__(self, model: object, tokenizer: object, **kwargs: object) -> None:
            calls.append(("runtime", type(model).__name__, kwargs))
            self.last_generation_path = None

        def generate(self, prompt: str, config: object) -> FakeGenerationResult:
            calls.append(("runtime.generate", prompt, {"config": config}))
            self.last_generation_path = "torch.generate.standard"
            return FakeGenerationResult()

    fake_runtime = types.ModuleType("chuk_lazarus.inference.backends.torch_runtime")
    fake_runtime.TorchInferenceRuntime = FakeTorchInferenceRuntime

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "chuk_lazarus.inference.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "chuk_lazarus.inference.backends.torch_runtime", fake_runtime)
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: [],
    )

    backend = TorchRuntimeModelBackend(
        "local/test-model",
        device="cuda:0",
        torch_dtype="bfloat16",
    )
    load_status = backend.load()
    result = backend.generate(
        "prompt",
        max_new_tokens=9,
        stop=["STOP"],
        logits_processor=object(),
        materialization_plan={
            "version": 1,
            "strategy": "kv_sidecar",
            "requested_strategy": "kv_sidecar",
            "requires_runtime_replay": True,
            "runtime_replay": {
                "strategy": "kv_sidecar",
                "requires_runtime_replay": True,
                "required_capability": "materialization.replay.kv_cache.v1",
            },
        },
    )

    assert load_status.available is True
    assert load_status.loaded is True
    assert result.ok is True
    assert result.text == "torch says hello "
    assert generation_configs == [{"max_new_tokens": 9, "temperature": 0.0, "use_plugins": False}]
    assert ("tokenizer", "local/test-model", {"local_files_only": True, "trust_remote_code": False}) in calls
    assert (
        "model",
        "local/test-model",
        {
            "local_files_only": True,
            "trust_remote_code": False,
            "torch_dtype": "torch.bfloat16",
        },
    ) in calls
    assert ("model.to", "cuda:0", {"non_blocking": True}) in calls
    assert ("runtime", "FakeModel", {"device": "cuda:0", "engine": "standard"}) in calls
    assert result.backend == "torch-runtime"
    assert result.metadata["max_new_tokens"] == 9
    assert result.metadata["temperature"] == 0.0
    assert result.metadata["use_plugins"] is False
    assert result.metadata["generation_path"] == "torch.generate.standard"
    assert result.metadata["logits_processor_count"] == 1
    assert result.metadata["logits_processor_applied"] is False
    assert result.metadata["stats"] == {"input_tokens": 3, "output_tokens": 2}
    replay = result.metadata["materialization_replay"]
    assert replay["backend"] == "torch-runtime"
    assert replay["tensor_replay"] is False
    assert replay["applied"] is False
    assert replay["refused"] is True


def test_torch_runtime_backend_fails_closed_when_cuda_requested_but_unavailable(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: [],
    )

    backend = TorchRuntimeModelBackend("local/model", device="cuda")
    result = backend.generate("hello")

    assert result.ok is False
    assert result.error == "CUDA device requested but torch.cuda.is_available() is false"
    assert result.metadata["device"] == "cuda"


def test_torch_runtime_backend_reports_no_tensor_replay_capabilities() -> None:
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        insertion_family="kv_direct",
    )
    backend = TorchRuntimeModelBackend("local/model")

    capabilities = backend.replay_consumer_capabilities(adapter)

    assert capabilities is not None
    assert capabilities.consumer_id == "torch-runtime:no-tensor-replay"
    assert capabilities.capabilities == ()
    assert capabilities.strategies == ()
    assert capabilities.model_id == "model-a"
    assert capabilities.tokenizer_id == "tokenizer-a"
    assert capabilities.model_revision == "rev-a"
    assert capabilities.adapter_family == "family-a"
    assert capabilities.insertion_families == ("kv_direct",)
    assert capabilities.metadata["supports_tensor_replay"] is False


def test_runtime_selects_torch_runtime_backend_when_requested_after_validated_boot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model_root = tmp_path / "model"
    model_root.mkdir()
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(_validation_report()), encoding="utf-8")

    runtime = DavidRuntime.create(
        DavidConfig(
            workspace_root=workspace,
            state_dir=tmp_path / "state",
            model_path=str(model_root),
            validation_report_path=str(report_path),
            require_validated_model=True,
            model_backend="torch-runtime",
            model_device="cpu",
            model_dtype="float32",
        )
    )

    assert isinstance(runtime.backend, TorchRuntimeModelBackend)
    assert runtime.backend.model_id == str(model_root)
    assert runtime.backend.device == "cpu"
    assert runtime.backend.requested_dtype == "float32"
    assert runtime.readiness()["backend"].startswith("torch-runtime:")


def test_runtime_rejects_unknown_backend_selector_after_validated_boot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model_root = tmp_path / "model"
    model_root.mkdir()
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(_validation_report()), encoding="utf-8")

    runtime = DavidRuntime.create(
        DavidConfig(
            workspace_root=workspace,
            state_dir=tmp_path / "state",
            model_path=str(model_root),
            validation_report_path=str(report_path),
            require_validated_model=True,
            model_backend="mystery",
        )
    )

    assert runtime.backend.name == "offline-deterministic"
    assert runtime.boot_errors == ["unsupported model backend selector: mystery"]
