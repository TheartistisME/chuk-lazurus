from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.decoder import DecoderController
from chuk_lazarus.david.materialization_replay import replay_generation_metadata
from chuk_lazarus.david.materializer import Materializer
from chuk_lazarus.david.routing import RoutePacket


def test_materializer_compatibility_comes_from_adapter_metadata(tmp_path: Path) -> None:
    adapter = AdapterSessionMetadata(
        model_id="test/model",
        tokenizer_id="test/tokenizer",
        model_revision="abc",
        adapter_family="test-family",
        boundary_layer=4,
        kv_source_layer=5,
        kv_target_layer=6,
        insertion_family="full_attention",
    )
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state", adapter=adapter))

    result = runtime.run_once("Fix repo patch target")

    assert result.materialized.compatibility["model_id"] == "test/model"
    assert result.materialized.compatibility["kv_source_layer"] == 5
    assert result.decoder.prior_scope["kv_target_layer"] == 6
    assert result.decoder.prior_scope["session_id"] == "default"


def test_decoder_constraints_for_temporal_recall(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("What was the latest thing I asked you to remember?")

    assert result.method == "temporal_recall"
    assert result.decoder.constraints["ordinal"] == "exact_occurrence_required"
    assert result.decoder.prior_scope["model_id"] == "offline-deterministic"


def test_materializer_refuses_kv_when_adapter_layers_are_missing() -> None:
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["hot span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        kv_ready=True,
    )

    materialized = Materializer().materialize(route, AdapterSessionMetadata())

    assert materialized.refused is True
    assert materialized.strategy == "refuse"
    assert "kv_source_layer" in materialized.reason
    assert "kv_target_layer" in materialized.reason
    assert materialized.compatibility["materialization_safe"] is False


def test_materializer_refuses_cross_model_scope_mixing() -> None:
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        adapter_family="family-a",
        kv_source_layer=4,
        kv_target_layer=5,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["hot span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        kv_ready=True,
        provenance={"materialization_scope": {"model_id": "model-b", "tokenizer_id": "tokenizer-a"}},
    )

    materialized = Materializer().materialize(route, adapter)

    assert materialized.refused is True
    assert "model_id mismatch" in materialized.reason


def test_materializer_selects_boundary_residual_when_safe() -> None:
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        adapter_family="family-a",
        boundary_layer=7,
    )
    route = RoutePacket(
        method="source_dependency",
        selected_windows=["source span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        residual_available=True,
        provenance={"materialization_scope": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"}},
    )

    materialized = Materializer().materialize(route, adapter)

    assert materialized.refused is False
    assert materialized.strategy == "boundary_residual"
    assert materialized.text_context == "source span"


def test_materializer_emits_loadable_residual_sidecar_plan(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sidecar.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "hot-window-1",
                "memory_family": "task",
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
                "boundary_layer": 7,
                "residual_layer": 7,
                "hidden_size": 4,
                "refs": [
                    {
                        "kind": "boundary_residual",
                        "layer": 7,
                        "dtype": "float32",
                        "shape": [1, 4],
                        "inline_values": [[0.1, 0.2, 0.3, 0.4]],
                    }
                ],
                "provenance": {"capture": "unit-test"},
            }
        ),
        encoding="utf-8",
    )
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        hidden_size=4,
        boundary_layer=7,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="source_dependency",
        selected_windows=["source span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[{"sidecar_manifest": str(manifest_path)}],
        token_cost=2,
        residual_available=True,
    )

    replay_consumer = {
        "consumer_id": "unit-replay-hook",
        "capabilities": ["materialization.replay.residual_stream.v1"],
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_families": ["kv_direct"],
        "memory_families": ["task"],
    }

    materialized = Materializer().materialize(route, adapter, replay_consumer=replay_consumer)

    assert materialized.refused is False
    assert materialized.strategy == "residual_sidecar"
    assert materialized.materialization_plan["replay_family"] == "residual_sidecar"
    assert materialized.materialization_plan["replay_status"] == "guarded"
    assert materialized.materialization_plan["guarded"] is True
    assert materialized.materialization_plan["tensor_replay_applied"] is False
    assert materialized.materialization_plan["runtime_replay"]["replay_family"] == "residual_sidecar"
    assert materialized.materialization_plan["requires_runtime_replay"] is True
    assert (
        materialized.materialization_plan["runtime_replay"]["required_capability"]
        == "materialization.replay.residual_stream.v1"
    )
    assert materialized.materialization_plan["runtime_replay"]["consumer"]["consumer_id"] == "unit-replay-hook"
    assert materialized.materialization_plan["replay_refs"][0]["kind"] == "boundary_residual"
    assert materialized.materialization_plan["replay_refs"][0]["inline_value_count"] == 4
    assert materialized.materialization_plan["sidecars"][0]["artifact_id"] == "hot-window-1"
    assert materialized.compatibility["sidecar_artifact_ids"] == ["hot-window-1"]


def test_materializer_refuses_sidecar_replay_without_consumer(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sidecar.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "kv-window-1",
                "memory_family": "task",
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
                "kv_source_layer": 6,
                "kv_target_layer": 7,
                "refs": [{"kind": "kv_cache", "layer": 7, "dtype": "float16", "shape": [2, 1, 8]}],
            }
        ),
        encoding="utf-8",
    )
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        kv_source_layer=6,
        kv_target_layer=7,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["hot span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        kv_ready=True,
        provenance={"sidecar_manifest": str(manifest_path)},
    )

    materialized = Materializer().materialize(route, adapter)

    assert materialized.refused is True
    assert materialized.strategy == "refuse"
    assert materialized.materialization_plan["requested_strategy"] == "kv_sidecar"
    assert materialized.materialization_plan["requires_runtime_replay"] is True
    assert "runtime replay consumer required for kv_sidecar" in materialized.reason


def test_replay_metadata_distinguishes_residual_sidecar_guarded_applied_refused() -> None:
    plan = {
        "version": 1,
        "strategy": "residual_sidecar",
        "requested_strategy": "residual_sidecar",
        "requires_runtime_replay": True,
        "memory_family": "task",
        "tier": "hot",
        "session_id": "s1",
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
        "sidecars": [{"artifact_id": "hot-window-1"}],
        "replay_refs": [{"kind": "boundary_residual"}, {"kind": "residual_stream"}],
    }
    consumer = {
        "consumer_id": "residual-sidecar-hook",
        "capabilities": ["materialization.replay.residual_stream.v1"],
        "model_id": "model-a",
        "tokenizer_id": "tokenizer-a",
        "model_revision": "rev-a",
        "adapter_family": "family-a",
        "insertion_families": ["kv_direct"],
        "memory_families": ["task"],
    }

    guarded = replay_generation_metadata(
        plan,
        consumer,
        backend_name="unit-backend",
        supports_tensor_replay=False,
    )
    applied = replay_generation_metadata(
        plan,
        consumer,
        backend_name="unit-backend",
        supports_tensor_replay=True,
    )
    refused = replay_generation_metadata(
        plan,
        {**consumer, "model_id": "model-b"},
        backend_name="unit-backend",
        supports_tensor_replay=True,
    )

    assert guarded is not None
    assert guarded["replay_family"] == "residual_sidecar"
    assert guarded["replay_status"] == "guarded"
    assert guarded["guarded"] is True
    assert guarded["refused"] is False
    assert guarded["applied"] is False
    assert guarded["contract_refused"] is False
    assert guarded["state"] == {
        "family": "residual_sidecar",
        "status": "guarded",
        "guarded": True,
        "applied": False,
        "refused": False,
        "tensor_replay": False,
    }
    assert guarded["plan"]["replay_ref_kinds"] == {"boundary_residual": 1, "residual_stream": 1}
    assert guarded["guard_reasons"] == ["unit-backend has no tensor replay hook for residual_sidecar"]

    assert applied is not None
    assert applied["replay_status"] == "applied"
    assert applied["applied"] is True
    assert applied["tensor_replay_applied"] is True
    assert applied["state"]["tensor_replay"] is True

    assert refused is not None
    assert refused["replay_status"] == "refused"
    assert refused["contract_refused"] is True
    assert refused["state"]["refused"] is True
    assert any("model_id mismatch" in reason for reason in refused["contract_refusal_reasons"])


def test_replay_metadata_keeps_kv_sidecar_family_distinct_from_residual_sidecar() -> None:
    metadata = replay_generation_metadata(
        {
            "version": 1,
            "strategy": "kv_sidecar",
            "requested_strategy": "kv_sidecar",
            "requires_runtime_replay": True,
            "memory_family": "task",
            "runtime_replay": {
                "strategy": "kv_sidecar",
                "requires_runtime_replay": True,
                "required_capability": "materialization.replay.kv_cache.v1",
                "memory_family": "task",
            },
            "replay_refs": [{"kind": "kv_cache"}],
        },
        {"consumer_id": "metadata-only"},
        backend_name="unit-backend",
        supports_tensor_replay=False,
    )

    assert metadata is not None
    assert metadata["replay_family"] == "kv_sidecar"
    assert metadata["replay_status"] == "refused"
    assert metadata["state"]["family"] == "kv_sidecar"
    assert metadata["plan"]["replay_ref_kinds"] == {"kv_cache": 1}
    assert metadata["required_capability"] == "materialization.replay.kv_cache.v1"


def test_materializer_refuses_incompatible_replay_consumer(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sidecar.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "kv-window-1",
                "memory_family": "task",
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
                "kv_source_layer": 6,
                "kv_target_layer": 7,
                "refs": [{"kind": "kv_cache", "layer": 7, "dtype": "float16", "shape": [2, 1, 8]}],
            }
        ),
        encoding="utf-8",
    )
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        kv_source_layer=6,
        kv_target_layer=7,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["hot span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        kv_ready=True,
        provenance={"sidecar_manifest": str(manifest_path)},
    )

    materialized = Materializer().materialize(
        route,
        adapter,
        replay_consumer={
            "consumer_id": "text-only-hook",
            "capabilities": ["materialization.replay.residual_stream.v1"],
            "model_id": "model-b",
        },
    )

    assert materialized.refused is True
    assert materialized.strategy == "refuse"
    assert "lacks materialization.replay.kv_cache.v1" in materialized.reason
    assert "model_id mismatch" in materialized.reason


def test_materializer_refuses_sidecar_scope_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad-sidecar.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "wrong-tokenizer",
                "memory_family": "task",
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-b",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
                "kv_source_layer": 6,
                "kv_target_layer": 7,
                "refs": [{"kind": "kv_cache", "layer": 7, "dtype": "float16", "shape": [2, 1, 8]}],
            }
        ),
        encoding="utf-8",
    )
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        model_revision="rev-a",
        adapter_family="family-a",
        kv_source_layer=6,
        kv_target_layer=7,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["hot span"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
        kv_ready=True,
        provenance={"sidecar_manifest": str(manifest_path)},
    )

    materialized = Materializer().materialize(route, adapter)

    assert materialized.refused is True
    assert materialized.strategy == "refuse"
    assert "tokenizer_id mismatch" in materialized.reason
    assert materialized.compatibility["materialization_plan"]["refused"] is True


def test_decoder_plan_includes_steering_constraints_and_scope_metadata() -> None:
    adapter = AdapterSessionMetadata(
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        adapter_family="family-a",
        boundary_layer=7,
        kv_target_layer=8,
        insertion_family="kv_direct",
    )
    route = RoutePacket(
        method="repo_patch",
        selected_windows=["src/tool.py def run_tool(): return None"],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[],
        token_cost=2,
    )

    plan = DecoderController().plan(route=route, adapter=adapter, session_id="s1", prompt="Fix the Python patch task")

    assert plan.constraints["steering"]["target_language"] == "python"
    assert "javascript_declarations" in plan.constraints["steering"]["forbidden_token_families"]
    assert plan.prior_scope["task_type"] == "code_patch"
    assert plan.prior_scope["steering_version"] == "david-decoder-steering-v1"
    assert plan.prior_scope["target_language"] == "python"
