from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david import runtime as runtime_module
from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.model_backend import (
    ModelBackendResult,
    ModelBackendStatus,
    TransformersCausalLMBackend,
)
from chuk_lazarus.david.routing import RoutePacket


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


def test_missing_index_requires_jit_plan(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Inspect source dependencies for this workspace")

    assert result.index.required is True
    assert result.index.jit_plan is not None
    assert result.index.jit_plan["action"] == "jit_index_workspace"
    assert "jit_required" in result.answer
    assert result.product_route is not None
    assert result.product_route.method == "source_dependency"
    assert result.decoder_prior is not None
    assert (tmp_path / "state" / "decoder_priors.json").exists()
    assert result.resume_snapshot is not None
    assert (tmp_path / "state" / "resume.json").exists()
    assert result.product_route.provenance["router"] == "david.central_router.full"
    assert result.product_route.provenance["proof_router_available"] is True


def test_runtime_falls_back_when_full_central_router_wrapper_unavailable(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    def unavailable() -> object:
        raise RuntimeError("wrapper load failed")

    monkeypatch.setattr(
        runtime_module.CentralRouterAdapter,
        "from_stable_wrapper_if_available",
        unavailable,
    )
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Inspect source dependencies for this workspace")

    assert runtime.product_router_errors == ["RuntimeError: wrapper load failed"]
    assert result.product_route is not None
    assert result.product_route.provenance["router"] == "david.central_router.offline"
    assert result.product_route.provenance["proof_router_available"] is False


def test_repo_patch_routing_and_task_writeback(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("def broken_session():\n    return None\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Fix the repo bug by patching src/example.py")

    assert result.method == "repo_patch"
    assert result.route.memory_family == "task"
    assert result.product_route is not None
    assert result.product_route.method == "repo_patch"
    assert result.product_route.methodology == "patch_target"
    assert "src/example.py" in result.product_route.selected_paths
    assert any(item.get("path") == "src/example.py" for item in result.route.evidence)
    assert "patch_compatible_edits" in result.decoder.constraints["bias"]
    assert result.writeback["family"] == "task"
    assert result.writeback_verification is not None
    assert result.writeback_verification.ok is True
    assert result.writeback_verification.checks["memory_writeback"]["kind"] == "repo_patch"
    assert result.writeback_verification.checks["product_route"]["route_reason_count"] >= 1
    assert result.writeback["metadata"]["route_evidence_chain"]
    assert result.writeback["metadata"]["route"]["evidence"] == result.route.evidence
    assert result.model_result is not None
    steering = result.model_result.metadata["decoder_steering"]
    assert steering["attempted"] is True
    assert steering["applied"] is False
    assert steering["processor_count"] == 0
    assert steering["refused_reason"] == "backend does not expose tokenizer for live steering"
    assert result.materialized.materialization_plan["strategy"] == "boundary_text"
    replay = result.model_result.metadata["materialization_replay"]
    assert replay["attempted"] is True
    assert replay["applied"] is False
    assert replay["tensor_replay"] is False
    assert result.writeback["metadata"]["materialized"]["materialization_replay"] == replay
    assert (tmp_path / "state" / "memory" / "task-default.jsonl").exists()
    assert not (tmp_path / "state" / "memory" / "user-default.jsonl").exists()


def test_runtime_surfaces_unsafe_materialization_in_writeback_verification(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("def broken_session():\n    return None\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    original_route = runtime.router.route

    def route_with_mismatch(**kwargs: object) -> RoutePacket:
        packet = original_route(**kwargs)
        return RoutePacket(
            **{
                **packet.to_json(),
                "provenance": {
                    **packet.provenance,
                    "materialization_scope": {
                        "model_id": "other-model",
                        "tokenizer_id": runtime.adapter.tokenizer_id,
                    },
                },
            }
        )

    runtime.router.route = route_with_mismatch  # type: ignore[method-assign]

    result = runtime.run_once("Fix the repo bug by patching src/example.py")

    assert result.materialized.refused is True
    assert "model_id mismatch" in result.materialized.reason
    assert result.writeback_verification is not None
    assert result.writeback_verification.ok is False
    assert result.writeback_verification.checks["adapter_materialization_compatibility"]["materializer_refused"] is True
    assert result.writeback["metadata"]["materialized"]["refused"] is True


def test_runtime_applies_live_decoder_steering_for_transformers_backend(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.js"
    source.parent.mkdir()
    source.write_text("const value = require('x')\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    captured: dict[str, object] = {}

    class FakeTokenizer:
        tokenizer_id = "offline-tokenizer"

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return {
                "def ": [2],
                "class ": [3],
                "self": [4],
                "None": [5],
                "True": [6],
                "False": [7],
            }.get(text, [])

    class LiveBackend(TransformersCausalLMBackend):
        def __init__(self) -> None:
            super().__init__("offline-deterministic")
            self._tokenizer = FakeTokenizer()

        def load(self) -> ModelBackendStatus:
            return ModelBackendStatus(name=self.name, available=True, loaded=True)

        def generate(self, prompt: str, **kwargs: object) -> ModelBackendResult:
            captured.update(kwargs)
            processors = kwargs.get("logits_processor")
            processor_count = len(processors) if isinstance(processors, list) else 0
            return ModelBackendResult(
                text="live answer",
                backend=self.name,
                metadata={"logits_processor_count": processor_count, "prompt": prompt},
            )

    runtime.backend = LiveBackend()

    result = runtime.run_once("Fix the JavaScript repo bug by patching src/app.js")

    assert result.model_result is not None
    assert result.method == "repo_patch"
    assert isinstance(captured["logits_processor"], list)
    assert captured["materialization_plan"] == result.materialized.materialization_plan
    assert result.model_result.metadata["logits_processor_count"] == 1
    steering = result.model_result.metadata["decoder_steering"]
    assert steering["attempted"] is True
    assert steering["applied"] is True
    assert steering["processor_count"] == 1
    assert steering["refused_reason"] is None
    assert steering["forbidden_token_count"] >= 1


def test_runtime_refuses_live_steering_when_transformers_backend_cannot_load(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app.js"
    source.parent.mkdir()
    source.write_text("const value = 1\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    captured: dict[str, object] = {}

    class UnloadedBackend(TransformersCausalLMBackend):
        def load(self) -> ModelBackendStatus:
            return ModelBackendStatus(name=self.name, available=True, loaded=False, reason="no local model")

        def generate(self, prompt: str, **kwargs: object) -> ModelBackendResult:
            captured.update(kwargs)
            return ModelBackendResult(text="", backend=self.name, ok=False, error="no local model")

    runtime.backend = UnloadedBackend("local/missing")

    result = runtime.run_once("Fix the JavaScript repo bug by patching src/app.js")

    assert captured["logits_processor"] is None
    assert result.model_result is not None
    steering = result.model_result.metadata["decoder_steering"]
    assert steering["attempted"] is True
    assert steering["applied"] is False
    assert steering["processor_count"] == 0
    assert steering["refused_reason"] == "backend not loaded: no local model"


def test_verification_command_behavior(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Verify quality gate", verify_command=["python", "-c", "print('ok')"])

    assert result.method == "verify"
    assert result.verification.ok is True
    assert result.verification.command_result is not None
    assert result.verification.command_result["returncode"] == 0
    assert "ok" in result.verification.command_result["stdout"]


def test_runtime_status_and_shell_commands_are_available_to_tui(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    readiness = runtime.readiness()
    shell = runtime.run_shell("python -c \"print('hello')\"")

    assert readiness["model validation"].startswith("offline shell mode")
    assert readiness["backend"].startswith("offline-deterministic")
    assert "index" in readiness
    assert "source index" in readiness
    assert "resume" in readiness
    assert "rc=0" in shell
    assert "hello" in shell


def test_runtime_agent_loop_executes_actions_and_persists_trace(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_agent_loop(
        [
            {"action": "write", "path": "src/generated.py", "content": "VALUE = 7\n"},
            {"action": "verify", "passed": True, "reason": "deterministic pass"},
        ]
    )

    assert result.ok is True
    assert result.loop.status == "verified"
    assert result.loop.steps == 2
    assert result.writeback is not None
    assert result.writeback["family"] == "task"
    assert result.writeback["kind"] == "agent_loop"
    assert result.writeback["metadata"]["loop"]["status"] == "verified"
    assert result.resume_snapshot is not None
    assert "agent_loop status=verified" in result.resume_snapshot["last_result_summary"]
    assert (tmp_path / "src" / "generated.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    assert (tmp_path / "state" / "memory" / "task-default.jsonl").exists()
    assert (tmp_path / "state" / "resume.json").exists()


def test_runtime_agent_loop_uses_backend_for_natural_language_prompt(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))
    prompts: list[str] = []

    class ActionBackend:
        name = "fake-action-backend"

        def status(self) -> ModelBackendStatus:
            return ModelBackendStatus(name=self.name, available=True, loaded=True)

        def generate(self, prompt: str, **kwargs: object) -> ModelBackendResult:
            del kwargs
            prompts.append(prompt)
            if len(prompts) == 1:
                return ModelBackendResult(
                    text=json.dumps(
                        {"action": "write", "path": "src/generated.py", "content": "VALUE = 11\n"}
                    ),
                    backend=self.name,
                    metadata={"step": 1},
                )
            assert '"bytes": 11' in prompt
            return ModelBackendResult(
                text='```json\n{"action": "verify", "passed": true, "reason": "observed write"}\n```',
                backend=self.name,
                metadata={"step": 2},
            )

    runtime.backend = ActionBackend()

    result = runtime.run_agent_loop("Create src/generated.py with VALUE = 11")

    assert result.loop.status == "verified"
    assert result.loop.steps == 2
    assert len(prompts) == 2
    assert "Return exactly one JSON object" in prompts[0]
    assert result.loop.trace[0].provenance["raw"]["_model_provenance"]["backend"] == "fake-action-backend"
    assert (tmp_path / "src" / "generated.py").read_text(encoding="utf-8") == "VALUE = 11\n"


def test_runtime_auto_jit_builds_bounded_source_index(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("import os\n\ndef boot_agent():\n    return os.getcwd()\n", encoding="utf-8")
    runtime = DavidRuntime.create(
        DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state", auto_jit_index=True)
    )

    result = runtime.run_once("Inspect source dependencies for boot_agent")

    assert result.index.ready is True
    assert result.source_index is not None
    assert result.source_index["file_count"] == 1
    assert result.source_index["files"][0]["path"] == "src/agent.py"
    assert "boot_agent" in result.source_index["files"][0]["symbols"]
    assert result.live_index_refresh is not None
    assert result.live_index_refresh["changed_count"] == 1
    assert result.live_index_refresh["indexed_count"] == 1
    assert result.live_index_refresh["deleted_count"] == 0
    assert len(result.live_index_refresh["manifest_sha256"]) == 64
    assert result.to_json()["live_index_refresh"] == result.live_index_refresh
    assert result.resume_snapshot is not None
    assert result.resume_snapshot["memory_paths"]["live_index_state"].endswith("-live-source-state.json")
    assert result.product_route is not None
    assert "src/agent.py" in result.product_route.source_index_paths
    assert "boot_agent" in result.product_route.selected_symbols


def test_runtime_jit_index_hook_refreshes_manifest_source_index_and_live_state(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def boot_agent():\n    return 'ready'\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    summary = runtime.jit_index()

    assert "index: ready" in summary
    assert "source index: 1 files" in summary
    assert "changed=1 indexed=1 deleted=0" in summary
    assert "manifest_sha256=" in summary
    assert runtime.index.check().ready is True
    assert runtime._loaded_source_index() is not None
    assert runtime.latest_live_index_refresh is not None
    assert Path(runtime.latest_live_index_refresh.state_path).exists()
    status = runtime.index_status()
    assert "live indexed_at=" in status
    assert "manifest_sha256=" in status

    second_summary = runtime.refresh_index()

    assert "changed=0 indexed=0 deleted=0" in second_summary


def test_runtime_uses_validated_harness_adapter_for_prior_scope(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    report_path = tmp_path / "validation.json"
    report_path.write_text(json.dumps(_validation_report()), encoding="utf-8")
    runtime = DavidRuntime.create(
        DavidConfig(
            workspace_root=tmp_path / "workspace",
            state_dir=tmp_path / "state",
            model_path=str(model_root),
            validation_report_path=str(report_path),
            require_validated_model=True,
        )
    )

    result = runtime.run_once("Remember my JavaScript preference for this workspace")

    assert runtime.adapter.model_id == "gemma-runtime-test"
    assert runtime.adapter.kv_target_layer == 23
    assert result.harness_session is not None
    assert result.harness_session["validation_status"] == "accepted"
    assert result.decoder_prior is not None
    assert result.decoder_prior["scope"]["layer"] == 23
    assert result.decoder_prior["scope"]["model_id"] == "gemma-runtime-test"


def test_runtime_passes_model_execution_controls_to_validated_backend(tmp_path: Path) -> None:
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
            model_device="cuda:0",
            model_dtype="bfloat16",
            model_max_new_tokens=37,
        )
    )
    captured: dict[str, object] = {}

    assert isinstance(runtime.backend, TransformersCausalLMBackend)
    assert runtime.backend.device == "cuda:0"
    assert runtime.backend.requested_dtype == "bfloat16"

    class CaptureBackend:
        name = "capture"

        def status(self) -> ModelBackendStatus:
            return ModelBackendStatus(name=self.name, available=True, loaded=True)

        def generate(self, prompt: str, **kwargs: object) -> ModelBackendResult:
            captured.update(kwargs)
            return ModelBackendResult(text="controlled answer", backend=self.name, metadata={"prompt": prompt})

    runtime.backend = CaptureBackend()

    result = runtime.run_once("Inspect source dependencies")

    assert captured["max_new_tokens"] == 37
    assert result.model_result is not None
    assert result.model_result.text == "controlled answer"
