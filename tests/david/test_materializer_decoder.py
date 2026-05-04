from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.decoder import DecoderController
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

    materialized = Materializer().materialize(route, adapter)

    assert materialized.refused is False
    assert materialized.strategy == "residual_sidecar"
    assert materialized.materialization_plan["requires_runtime_replay"] is True
    assert materialized.materialization_plan["sidecars"][0]["artifact_id"] == "hot-window-1"
    assert materialized.compatibility["sidecar_artifact_ids"] == ["hot-window-1"]


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
