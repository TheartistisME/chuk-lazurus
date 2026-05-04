from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
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
    assert result.product_route is not None
    assert "src/agent.py" in result.product_route.source_index_paths
    assert "boot_agent" in result.product_route.selected_symbols


def test_runtime_jit_index_hook_refreshes_manifest_and_source_index(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def boot_agent():\n    return 'ready'\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    summary = runtime.jit_index()

    assert "index: ready" in summary
    assert "source index: 1 files" in summary
    assert runtime.index.check().ready is True
    assert runtime._loaded_source_index() is not None


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
