from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.harness import (
    HarnessSession,
    MemoryFamily,
    ReadinessState,
)
from chuk_lazarus.harness.boot import boot_harness


def _validation_report(
    *,
    status: str = "accepted",
    auto_load_allowed: bool = True,
    adapter_config_id: str = "gemma-e2b-l23",
    model_identity: str = "gemma-e2b-test",
    tokenizer_identity: str = "gemma-tokenizer-test",
    route_layer: int = 11,
    boundary_layer: int = 17,
    kv_source_layer: int = 21,
    kv_target_layer: int = 23,
) -> dict[str, object]:
    selected_config = {
        "adapter_config_id": adapter_config_id,
        "route_layer": route_layer,
        "route_query_head": 3,
        "route_dimension": 2048,
        "boundary_layer": boundary_layer,
        "residual_capture_layer": boundary_layer,
        "kv_source_layer": kv_source_layer,
        "kv_target_layer": kv_target_layer,
        "injection_layer": kv_target_layer,
        "projection_producer_layer": kv_source_layer,
        "behavior_cache_layer": kv_source_layer,
        "insertion_family": "kv_direct",
        "kv_layout": "bshd",
        "candidate_role": "behavioral",
    }
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": status,
        "confidence": "high",
        "validation_level": "behavioral",
        "auto_load_allowed": auto_load_allowed,
        "harness_load_policy": "auto",
        "selected_config": selected_config,
        "source_report_summary": {
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "adapter_family": "gemma",
            "model_revision_or_hash": "synthetic-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "model_identity_gate": {
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "adapter_family": "gemma",
            "model_revision_or_hash": "synthetic-rev",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "topology_gate": {"accepted": status == "accepted"},
        "projection_gate": {
            "ranked_candidates": [
                {
                    "kv_source_layer": kv_source_layer,
                    "kv_target_layer": kv_target_layer,
                    "candidate_role": "behavioral",
                    "adapter_family": "gemma",
                    "projection_aliases": {"k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj"},
                    "kv_layout": "bshd",
                }
            ]
        },
        "behavior_gate": {"accepted": status == "accepted"},
        "report_integrity": {"accepted": True},
        "provenance": {"loader_options": {"model": model_identity}},
        "warnings": [],
    }


def _write_report(path: Path, report: dict[str, object]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture()
def boot_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    user_memory = tmp_path / "user-memory"
    task_memory = tmp_path / "workspace" / ".chuk_lazarus" / "task_memory"
    user_memory.mkdir(parents=True)
    task_memory.mkdir(parents=True)
    monkeypatch.setenv("CHUK_LAZARUS_USER_MEMORY_ROOT", str(user_memory))
    monkeypatch.setenv("CHUK_LAZARUS_WORKSPACE_MEMORY_ROOT", str(task_memory))
    return {
        "model": tmp_path / "model",
        "workspace": tmp_path / "workspace",
        "user_memory": user_memory,
        "task_memory": task_memory,
    }


def test_accepted_validation_report_populates_harness_session(
    tmp_path: Path, boot_roots: dict[str, Path]
) -> None:
    boot_roots["model"].mkdir()
    report_path = _write_report(tmp_path / "validation.json", _validation_report())

    session = boot_harness(
        model_path=str(boot_roots["model"]),
        workspace_path=str(boot_roots["workspace"]),
        validation_report_path=str(report_path),
        require_validated_model=True,
    )

    assert isinstance(session, HarnessSession)
    assert session.session_id
    assert session.workspace_path == str(boot_roots["workspace"])
    assert session.memory_root == str(boot_roots["task_memory"].parent)
    assert session.model_identity == "gemma-e2b-test"
    assert session.tokenizer_identity == "gemma-tokenizer-test"
    assert session.validation_status == "accepted"
    assert session.selected_adapter_config is not None
    assert session.selected_adapter_config["kv_target_layer"] == 23
    assert session.model_adapter is not None
    assert session.model_adapter.model_identity == "gemma-e2b-test"
    assert session.model_adapter.tokenizer_identity == "gemma-tokenizer-test"
    assert session.model_adapter.adapter_family == "gemma"
    assert session.model_adapter.route_dimension == 2048
    assert session.model_adapter.boundary_layer_candidate == 17
    assert session.model_adapter.kv_source_layer_candidate == 21
    assert session.model_adapter.kv_target_layer_candidate == 23
    assert session.materialization_compatibility is not None
    assert session.index_readiness is not None
    assert session.user_memory is not None
    assert session.task_memory is not None


@pytest.mark.parametrize("status", ["rejected", "review_required"])
def test_rejected_or_review_required_config_fails_when_validation_required(
    tmp_path: Path, boot_roots: dict[str, Path], status: str
) -> None:
    boot_roots["model"].mkdir()
    report_path = _write_report(
        tmp_path / f"{status}.json",
        _validation_report(status=status, auto_load_allowed=False),
    )

    with pytest.raises(Exception, match="validation|boot-safe|review|required|rejected"):
        boot_harness(
            model_path=str(boot_roots["model"]),
            workspace_path=str(boot_roots["workspace"]),
            validation_report_path=str(report_path),
            require_validated_model=True,
        )


def test_decoder_prior_scope_uses_selected_adapter_metadata_not_hardcoded_layer_14(
    tmp_path: Path, boot_roots: dict[str, Path]
) -> None:
    boot_roots["model"].mkdir()
    report_path = _write_report(
        tmp_path / "validation.json",
        _validation_report(
            adapter_config_id="adapter-selected-layer-23",
            route_layer=9,
            boundary_layer=16,
            kv_source_layer=20,
            kv_target_layer=23,
        ),
    )

    session = boot_harness(
        model_path=str(boot_roots["model"]),
        workspace_path=str(boot_roots["workspace"]),
        validation_report_path=str(report_path),
        require_validated_model=True,
    )

    assert session.decoder_prior_scope is not None
    assert session.decoder_prior_scope.model_identity == "gemma-e2b-test"
    assert session.decoder_prior_scope.tokenizer_identity == "gemma-tokenizer-test"
    assert session.decoder_prior_scope.adapter_config_id == "adapter-selected-layer-23"
    assert session.decoder_prior_scope.layer_scope == (23,)
    assert 14 not in session.decoder_prior_scope.layer_scope


@pytest.mark.parametrize(
    ("manifest", "expected_action"),
    [
        (None, "jit_index_workspace"),
        ({}, "refresh_workspace_index_for_model"),
        (
            {"model_identity": "other-model", "tokenizer_identity": "gemma-tokenizer-test"},
            "refresh_workspace_index_for_model",
        ),
        (
            {"model_identity": "gemma-e2b-test", "tokenizer_identity": "other-tokenizer"},
            "refresh_workspace_index_for_model",
        ),
    ],
)
def test_missing_or_mismatched_workspace_index_requires_jit_with_planned_actions(
    tmp_path: Path,
    boot_roots: dict[str, Path],
    manifest: dict[str, str] | None,
    expected_action: str,
) -> None:
    boot_roots["model"].mkdir()
    index_root = boot_roots["task_memory"] / "index"
    if manifest is not None:
        index_root.mkdir(parents=True)
        (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report_path = _write_report(tmp_path / "validation.json", _validation_report())

    session = boot_harness(
        model_path=str(boot_roots["model"]),
        workspace_path=str(boot_roots["workspace"]),
        validation_report_path=str(report_path),
        require_validated_model=True,
    )

    assert session.index_readiness is not None
    assert session.index_readiness.state is ReadinessState.NEEDS_JIT
    assert session.metadata["jit_required"] is True
    assert expected_action in session.metadata["planned_actions"]


def test_user_memory_and_task_workspace_memory_statuses_are_separate(
    tmp_path: Path, boot_roots: dict[str, Path]
) -> None:
    boot_roots["model"].mkdir()
    index_root = boot_roots["task_memory"] / "index"
    index_root.mkdir(parents=True)
    (index_root / "manifest.json").write_text(
        json.dumps(
            {
                "model_identity": "gemma-e2b-test",
                "tokenizer_identity": "gemma-tokenizer-test",
            }
        ),
        encoding="utf-8",
    )
    report_path = _write_report(tmp_path / "validation.json", _validation_report())

    session = boot_harness(
        model_path=str(boot_roots["model"]),
        workspace_path=str(boot_roots["workspace"]),
        validation_report_path=str(report_path),
        require_validated_model=True,
    )

    assert session.user_memory is not None
    assert session.task_memory is not None
    assert session.user_memory is not session.task_memory
    assert session.user_memory.state is ReadinessState.READY
    assert session.task_memory.state is ReadinessState.READY
    assert MemoryFamily.USER in session.index_readiness.memory_families
    assert MemoryFamily.TASK in session.index_readiness.memory_families
    assert session.user_memory.store_id == str(boot_roots["user_memory"])
    assert session.task_memory.store_id == str(boot_roots["task_memory"])
