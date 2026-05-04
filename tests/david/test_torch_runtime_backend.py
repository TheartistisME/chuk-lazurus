from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            calls.append(
                (
                    "chat_template",
                    repr(messages),
                    {"tokenize": tokenize, "add_generation_prompt": add_generation_prompt},
                )
            )
            assert messages == [{"role": "user", "content": "prompt"}]
            return "<start_of_turn>user\nprompt<end_of_turn>\n<start_of_turn>model\n"

    class FakeTokenizerFactory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls.append(("tokenizer", model_id, kwargs))
            return FakeTokenizer()

    class FakeModel:
        def __init__(self) -> None:
            self.device: str | None = None

        def to(self, device: str, **kwargs: object) -> FakeModel:
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
    runtime_generate_call = next(call for call in calls if call[0] == "runtime.generate")
    assert runtime_generate_call[1] == "<start_of_turn>user\nprompt<end_of_turn>\n<start_of_turn>model\n"
    assert result.backend == "torch-runtime"
    assert result.metadata["max_new_tokens"] == 9
    assert result.metadata["prompt_format"] == "chat_template"
    assert result.metadata["prompt_format_source"] == "tokenizer.apply_chat_template"
    assert result.metadata["temperature"] == 0.0
    assert result.metadata["use_plugins"] is False
    assert result.metadata["generation_path"] == "torch.generate.standard"
    assert result.metadata["logits_processor_count"] == 1
    assert result.metadata["logits_processor_applied"] is False
    assert result.metadata["processors_refused"] is True
    assert result.metadata["processors_refusal_reason"] == (
        "torch-runtime standard decode backend cannot apply decoder logits processors"
    )
    assert result.metadata["steering_applied"] is False
    assert result.metadata["steering_refused_reason"] == (
        "torch-runtime standard decode backend cannot apply decoder logits processors"
    )
    assert result.metadata["stats"] == {"input_tokens": 3, "output_tokens": 2}
    replay = result.metadata["materialization_replay"]
    assert replay["backend"] == "torch-runtime"
    assert replay["tensor_replay"] is False
    assert replay["applied"] is False
    assert replay["refused"] is True


def test_torch_runtime_backend_falls_back_to_raw_prompt_when_chat_template_fails(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda

    class FakeTokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            raise ValueError("no template")

    class FakeGenerationConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")
    fake_generation.GenerationConfig = FakeGenerationConfig

    class FakeGenerationResult:
        text = "fallback"
        stats = None
        stop_reason = None

    class FakeRuntime:
        last_generation_path = "torch.generate.standard"

        def generate(self, prompt: str, config: object) -> FakeGenerationResult:
            calls.append(("runtime.generate", prompt, {"config": config}))
            return FakeGenerationResult()

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "chuk_lazarus.inference.generation", fake_generation)
    monkeypatch.setattr(torch_backend.importlib, "import_module", lambda name: types.ModuleType(name))
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: [],
    )

    backend = TorchRuntimeModelBackend("local/test-model", device="cpu")
    backend._tokenizer = FakeTokenizer()
    backend._runtime = FakeRuntime()

    result = backend.generate("raw prompt")

    assert result.ok is True
    assert result.text == "fallback"
    assert result.metadata["prompt_format"] == "raw"
    assert result.metadata["prompt_format_fallback_reason"] == "ValueError: no template"
    assert calls[0][1] == "raw prompt"
    assert result.metadata["logits_processor_count"] == 0
    assert result.metadata["processors_refused"] is False
    assert result.metadata["processors_refusal_reason"] is None
    assert result.metadata["steering_applied"] is False
    assert result.metadata["steering_refused_reason"] is None


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


def test_torch_runtime_backend_refuses_processors_when_runtime_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: list(names),
    )
    backend = TorchRuntimeModelBackend("local/model")

    result = backend.generate("hello", logits_processor=[object(), object()])

    assert result.ok is False
    assert result.metadata["logits_processor_count"] == 2
    assert result.metadata["logits_processor_applied"] is False
    assert result.metadata["processors_refused"] is True
    assert result.metadata["processors_refusal_reason"] == (
        "torch-runtime standard decode backend cannot apply decoder logits processors"
    )
    assert result.metadata["steering_applied"] is False
    assert result.metadata["steering_refused_reason"] == (
        "torch-runtime standard decode backend cannot apply decoder logits processors"
    )


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


def test_torch_runtime_backend_advertises_residual_sidecar_only_when_enabled() -> None:
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        insertion_family="kv_direct",
    )
    backend = TorchRuntimeModelBackend("local/model", enable_residual_sidecar_replay=True)

    capabilities = backend.replay_consumer_capabilities(adapter)

    assert capabilities is not None
    assert capabilities.consumer_id == "torch-runtime:residual-sidecar"
    assert capabilities.strategies == ("residual_sidecar",)
    assert capabilities.capabilities == ("materialization.replay.residual_stream.v1",)
    assert capabilities.metadata["supports_tensor_replay"] is True


def test_torch_runtime_backend_applies_single_boundary_residual_sidecar(monkeypatch) -> None:
    calls: list[tuple[str, object, object]] = []

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert messages == [{"role": "user", "content": "prompt"}]
            return "formatted prompt"

    class FakeGenerationConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")
    fake_generation.GenerationConfig = FakeGenerationConfig

    class FakeStats:
        def model_dump(self) -> dict[str, object]:
            return {"input_tokens": 3, "output_tokens": 2}

    class FakeGenerationResult:
        text = "seeded answer STOP trailing"
        stats = FakeStats()
        stop_reason = "max_tokens"

    class FakeRuntime:
        last_generation_path = None

        def generate(self, prompt: str, config: object) -> object:
            calls.append(("standard", prompt, config))
            raise AssertionError("standard path must not run for residual sidecar")

        def generate_with_residual_seeded_at_layer(
            self,
            prompt: str,
            residual_state: object,
            config: object,
        ) -> FakeGenerationResult:
            calls.append(("residual", prompt, residual_state))
            assert prompt == "formatted prompt"
            assert residual_state.layer_index == 7
            assert residual_state.hidden_size == 4
            assert tuple(residual_state.tensor.shape) == (1, 4)
            assert config.kwargs == {"max_new_tokens": 5, "temperature": 0.0, "use_plugins": False}
            self.last_generation_path = "torch.generate.residual_seeded_at_layer"
            return FakeGenerationResult()

    _install_loaded_fake_torch_backend(monkeypatch, fake_generation)
    backend = TorchRuntimeModelBackend("local/test-model", device="cpu")
    backend._tokenizer = FakeTokenizer()
    backend._runtime = FakeRuntime()

    result = backend.generate(
        "prompt",
        max_new_tokens=5,
        stop=["STOP"],
        materialization_plan=_residual_sidecar_plan(),
        replay_consumer=_residual_sidecar_consumer(),
    )

    assert result.ok is True
    assert result.text == "seeded answer "
    assert calls and calls[0][0] == "residual"
    assert result.metadata["engine"] == "residual_sidecar"
    assert result.metadata["generation_path"] == "torch.generate.residual_seeded_at_layer"
    replay = result.metadata["materialization_replay"]
    assert replay["applied"] is True
    assert replay["tensor_replay"] is True
    sidecar_replay = result.metadata["residual_sidecar_replay"]
    assert sidecar_replay["accepted"] is True
    assert sidecar_replay["tensor_replay_applied"] is True
    assert sidecar_replay["tensors_loaded"] is True
    assert sidecar_replay["loaded_tensor"]["shape"] == [1, 4]
    assert sidecar_replay["loaded_tensor"]["layer"] == 7
    assert result.metadata["stats"] == {"input_tokens": 3, "output_tokens": 2}


def test_torch_runtime_backend_refuses_residual_sidecar_mismatch_without_runtime_call(monkeypatch) -> None:
    calls: list[tuple[str, object, object]] = []
    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")

    class FakeRuntime:
        def generate(self, prompt: str, config: object) -> object:
            calls.append(("standard", prompt, config))
            raise AssertionError("standard path must not run after residual mismatch")

        def generate_with_residual_seeded_at_layer(self, *args: object) -> object:
            calls.append(("residual", args, {}))
            raise AssertionError("residual path must not run after mismatch")

    _install_loaded_fake_torch_backend(monkeypatch, fake_generation)
    backend = TorchRuntimeModelBackend("local/test-model", device="cpu")
    backend._tokenizer = _RawTokenizer()
    backend._runtime = FakeRuntime()

    plan = _residual_sidecar_plan(sidecar_scope_overrides={"model_id": "model-b"})
    result = backend.generate(
        "prompt",
        materialization_plan=plan,
        replay_consumer=_residual_sidecar_consumer(),
    )

    assert result.ok is False
    assert "model_id mismatch" in result.error
    assert calls == []
    assert result.metadata["materialization_replay"]["refused"] is True
    assert "model_id mismatch" in result.metadata["materialization_replay"]["reason"]


def test_torch_runtime_backend_refuses_multi_row_residual_stream_without_runtime_call(monkeypatch) -> None:
    calls: list[tuple[str, object, object]] = []
    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")

    class FakeRuntime:
        def generate_with_residual_seeded_at_layer(self, *args: object) -> object:
            calls.append(("residual", args, {}))
            raise AssertionError("residual path must not run for multi-row stream")

    _install_loaded_fake_torch_backend(monkeypatch, fake_generation)
    backend = TorchRuntimeModelBackend("local/test-model", device="cpu")
    backend._tokenizer = _RawTokenizer()
    backend._runtime = FakeRuntime()

    result = backend.generate(
        "prompt",
        materialization_plan=_residual_sidecar_plan(
            ref={
                "kind": "residual_stream",
                "layer": 7,
                "dtype": "float32",
                "shape": [2, 4],
                "inline_values": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                "metadata": {},
            },
            sidecar_scope_overrides={"residual_layer": 7},
        ),
        replay_consumer=_residual_sidecar_consumer(),
    )

    assert result.ok is False
    assert result.error == "multi-row residual_stream replay is not supported"
    assert calls == []


def test_torch_runtime_backend_refuses_logits_processors_on_residual_sidecar_path(monkeypatch) -> None:
    calls: list[tuple[str, object, object]] = []
    fake_generation = types.ModuleType("chuk_lazarus.inference.generation")

    class FakeRuntime:
        def generate_with_residual_seeded_at_layer(self, *args: object) -> object:
            calls.append(("residual", args, {}))
            raise AssertionError("residual path must not run with logits processors")

    _install_loaded_fake_torch_backend(monkeypatch, fake_generation)
    backend = TorchRuntimeModelBackend("local/test-model", device="cpu")
    backend._tokenizer = _RawTokenizer()
    backend._runtime = FakeRuntime()

    result = backend.generate(
        "prompt",
        logits_processor=object(),
        materialization_plan=_residual_sidecar_plan(),
        replay_consumer=_residual_sidecar_consumer(),
    )

    assert result.ok is False
    assert calls == []
    assert result.metadata["logits_processor_count"] == 1
    assert result.metadata["processors_refused"] is True
    assert result.metadata["materialization_replay"]["refused"] is True
    assert (
        "torch-runtime residual-sidecar path cannot apply decoder logits processors"
        in result.metadata["materialization_replay"]["reason"]
    )


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


class _RawTokenizer:
    def apply_chat_template(self, *args: object, **kwargs: object) -> str:
        raise ValueError("no template")


def _install_loaded_fake_torch_backend(monkeypatch, fake_generation: types.ModuleType) -> None:
    class FakeLazarusBackend:
        CUDA = "cuda"

    class FakeResidualState:
        def __init__(
            self,
            *,
            backend: object,
            layer_index: int,
            tensor: object,
            sequence_length: int,
            hidden_size: int,
            dtype: str,
            device: str,
        ) -> None:
            self.backend = backend
            self.layer_index = layer_index
            self.tensor = tensor
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.dtype = dtype
            self.device = device

    fake_types = types.ModuleType("chuk_lazarus.inference.backends.types")
    fake_types.LazarusBackend = FakeLazarusBackend
    fake_types.ResidualState = FakeResidualState

    monkeypatch.setitem(sys.modules, "chuk_lazarus.inference.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "chuk_lazarus.inference.backends.types", fake_types)
    monkeypatch.setattr(torch_backend.importlib, "import_module", lambda name, *args, **kwargs: types.ModuleType(name))
    monkeypatch.setattr(
        "chuk_lazarus.david.torch_backend._missing_optional_packages",
        lambda *names: [],
    )


def _residual_sidecar_plan(
    *,
    ref: dict[str, object] | None = None,
    adapter_scope_overrides: dict[str, object] | None = None,
    sidecar_scope_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    adapter_scope: dict[str, object] = {
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_family": "kv_direct",
        "boundary_layer": 7,
        "hidden_size": 4,
    }
    adapter_scope.update(adapter_scope_overrides or {})
    sidecar_scope = {
        **adapter_scope,
        "residual_layer": None,
        "kv_source_layer": None,
        "kv_target_layer": None,
    }
    sidecar_scope.update(sidecar_scope_overrides or {})
    residual_ref = ref or {
        "kind": "boundary_residual",
        "layer": 7,
        "dtype": "float32",
        "shape": [1, 4],
        "inline_values": [[1.0, 2.0, 3.0, 4.0]],
        "metadata": {"span_id": "hot-1"},
    }
    return {
        "version": 1,
        "strategy": "residual_sidecar",
        "requested_strategy": "residual_sidecar",
        "requires_runtime_replay": True,
        "memory_family": "task",
        "adapter_scope": adapter_scope,
        "runtime_replay": {
            "strategy": "residual_sidecar",
            "requires_runtime_replay": True,
            "required_capability": "materialization.replay.residual_stream.v1",
            "adapter_scope": adapter_scope,
            "memory_family": "task",
        },
        "sidecars": [
            {
                "artifact_id": "hot-window-1",
                "memory_family": "task",
                "scope": sidecar_scope,
                "refs": [residual_ref],
                "provenance": {"source": "unit"},
            }
        ],
        "replay_refs": [residual_ref],
    }


def _residual_sidecar_consumer() -> dict[str, object]:
    return {
        "consumer_id": "explicit-residual-sidecar",
        "strategies": ["residual_sidecar"],
        "capabilities": ["materialization.replay.residual_stream.v1"],
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_families": ["kv_direct"],
        "memory_families": ["task"],
    }
