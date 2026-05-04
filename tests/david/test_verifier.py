from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david.tools import LocalTools
from chuk_lazarus.david.verifier import Verifier


def _verifier(tmp_path: Path) -> Verifier:
    return Verifier(LocalTools(tmp_path))


def test_temporal_recall_requires_ordinal_timestamp_and_artifact(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        "temporal_recall",
        [
            {
                "artifact_id": "user-1",
                "family": "user",
                "text": "Earlier preference",
                "timestamp": "2026-05-04T01:00:00+00:00",
                "ordinal": 0,
            }
        ],
        metadata={"requested_ordinal": "latest"},
    )

    assert result.ok is True
    assert result.checks["temporal_ordinal"]["requested_ordinal"] == "latest"
    assert result.checks["temporal_ordinal"]["has_timestamp"] is True


def test_temporal_recall_rejects_similar_text_without_occurrence_metadata(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify("temporal_recall", [{"text": "Similar remembered words"}])

    assert result.ok is False
    assert result.reason == "failed checks: temporal_ordinal"
    assert result.checks["temporal_ordinal"]["has_ordinal"] is False


def test_symbolic_chain_reports_missing_expected_hops(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="symbolic_multi_hop",
        evidence=[
            {"artifact_id": "parse", "text": "parser calls lexer", "ordinal": 0, "provenance": "jsonl"},
            {"artifact_id": "lexer", "text": "lexer reads tokens", "ordinal": 1, "provenance": "jsonl"},
        ],
        metadata={"expected_hops": ["parse", "lexer", "emit"]},
    )

    assert result.ok is False
    assert result.checks["symbolic_chain"]["missing_hops"] == ["emit"]


def test_symbolic_chain_accepts_complete_ordered_hops(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="symbolic_multi_hop",
        evidence=[
            {"artifact_id": "parse", "text": "parser calls lexer", "ordinal": 0, "provenance": "jsonl"},
            {
                "artifact_id": "lexer",
                "text": "lexer emits tokens",
                "ordinal": 1,
                "depends_on": "parse",
                "provenance": "jsonl",
            },
            {
                "artifact_id": "emit",
                "text": "emitter formats output",
                "ordinal": 2,
                "depends_on": "lexer",
                "provenance": "jsonl",
            },
        ],
        metadata={"expected_hops": ["parse", "lexer", "emit"]},
    )

    assert result.ok is True
    assert result.checks["symbolic_chain"]["ordered_hops"] == 3
    assert result.checks["symbolic_chain"]["provenance_count"] == 3


def test_symbolic_chain_rejects_single_or_unprovenanced_hop(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="symbolic_multi_hop",
        evidence=[
            {"artifact_id": "parse", "text": "parser calls lexer", "ordinal": 0, "provenance": "jsonl"},
            {"artifact_id": "lexer", "text": "lexer emits tokens", "ordinal": 1},
        ],
    )

    assert result.ok is False
    assert result.checks["symbolic_chain"]["provenance_count"] == 1
    assert result.checks["symbolic_chain"]["requires_multi_hop"] is True


def test_repo_patch_verifies_safe_workspace_paths_and_writeback(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="repo_patch",
        evidence=[
            {
                "artifact_id": "workspace:src/chuk_lazarus/david/verifier.py",
                "kind": "patch_target",
                "path": "src/chuk_lazarus/david/verifier.py",
                "text": "verifier target",
                "selected_tests": ["tests/david/test_verifier.py"],
            }
        ],
        metadata={
            "writeback": {
                "artifact_id": "task-1",
                "family": "task",
                "kind": "repo_patch",
                "text": "verified patch route",
                "timestamp": "2026-05-04T02:00:00+00:00",
            },
            "route_evidence_chain": [{"artifact_id": "workspace:src/chuk_lazarus/david/verifier.py"}],
        },
    )

    assert result.ok is True
    assert result.checks["patch_targets"]["unsafe_paths"] == []
    assert result.checks["patch_targets"]["target_paths"] == ["src/chuk_lazarus/david/verifier.py"]
    assert result.checks["memory_writeback"]["expected_family"] == "task"


def test_repo_patch_requires_selected_paths_or_patch_target_evidence(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="repo_patch",
        evidence=[{"kind": "source", "path": "src/example.py", "text": "plain source evidence"}],
    )

    assert result.ok is False
    assert result.checks["patch_targets"]["paths"] == ["src/example.py"]
    assert result.checks["patch_targets"]["target_paths"] == []
    assert result.checks["patch_targets"]["requires_selected_or_patch_target_evidence"] is True


def test_repo_patch_rejects_escaping_absolute_and_protected_paths(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="repo_patch",
        evidence=[
            {"path": "../outside.py", "text": "escape"},
            {"path": "/tmp/outside.py", "text": "absolute"},
            {"path": "C:/temp/outside.py", "text": "windows absolute"},
            {"path": "scripts/run_swebench_pro_parity.py", "text": "protected proof rig"},
        ],
    )

    assert result.ok is False
    assert "../outside.py" in result.checks["patch_targets"]["unsafe_paths"]
    assert "/tmp/outside.py" in result.checks["patch_targets"]["unsafe_paths"]
    assert "C:/temp/outside.py" in result.checks["patch_targets"]["unsafe_paths"]
    assert result.checks["patch_targets"]["protected_paths"] == ["scripts/run_swebench_pro_parity.py"]


def test_adapter_materialization_compatibility_metadata_is_checked_when_present(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "symbol": "boot", "text": "def boot(): ..."}],
        metadata={
            "adapter": {
                "model_id": "gemma-e2b",
                "tokenizer_id": "gemma-tokenizer",
                "kv_target_layer": 23,
                "insertion_family": "kv_direct",
            },
            "materialized": {
                "strategy": "kv_direct",
                "refused": False,
                "compatibility": {
                    "model_id": "gemma-e2b",
                    "tokenizer_id": "gemma-tokenizer",
                    "kv_target_layer": 23,
                    "insertion_family": "kv_direct",
                },
            },
            "route_evidence_chain": [{"artifact_id": "source-1", "path": "src/example.py"}],
        },
    )

    assert result.ok is True
    assert result.checks["adapter_materialization_compatibility"]["strategy"] == "kv_direct"
    assert result.checks["adapter_materialization_compatibility"]["evidence_chain_count"] == 1


def test_adapter_materialization_mismatch_fails_with_details(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter_scope": {"model_id": "gemma-e2b", "kv_target_layer": 23},
            "compatibility": {"model_id": "other-model", "kv_target_layer": 23},
        },
    )

    assert result.ok is False
    assert result.checks["adapter_materialization_compatibility"]["mismatches"] == {
        "model_id": {"adapter": "gemma-e2b", "materialized": "other-model"}
    }


def test_materialization_acceptance_requires_evidence_chain_when_not_refused(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter": {"model_id": "gemma-e2b", "tokenizer_id": "gemma-tokenizer"},
            "materialized": {
                "strategy": "boundary_residual",
                "refused": False,
                "compatibility": {"model_id": "gemma-e2b", "tokenizer_id": "gemma-tokenizer"},
            },
        },
    )

    assert result.ok is False
    check = result.checks["adapter_materialization_compatibility"]
    assert check["evidence_chain_required"] is True
    assert check["evidence_chain_count"] == 0


def test_task_writeback_requires_route_evidence_chain(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="repo_patch",
        evidence=[{"kind": "patch_target", "path": "src/example.py", "text": "target"}],
        metadata={
            "writeback": {
                "artifact_id": "task-1",
                "family": "task",
                "kind": "repo_patch",
                "text": "verified patch route",
                "timestamp": "2026-05-04T02:00:00+00:00",
            }
        },
    )

    assert result.ok is False
    assert result.checks["memory_writeback"]["evidence_chain_required"] is True
    assert result.checks["memory_writeback"]["evidence_chain_count"] == 0


def test_agent_loop_repair_metadata_reports_recovered_verify_failure(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)
    route_evidence_chain = [{"artifact_id": "workspace:src/service.py", "path": "src/service.py"}]

    result = verifier.verify(
        capability="repo_patch",
        evidence=[
            {
                "kind": "patch_target",
                "path": "src/service.py",
                "text": "service patch target",
            }
        ],
        metadata={
            "route_evidence_chain": route_evidence_chain,
            "writeback": {
                "artifact_id": "task-loop-1",
                "family": "task",
                "kind": "agent_loop",
                "text": "Agent loop repaired verify failure",
                "timestamp": "2026-05-04T02:30:00+00:00",
                "metadata": {
                    "route_evidence_chain": route_evidence_chain,
                    "loop": {
                        "status": "verified",
                        "steps": 4,
                        "verified": True,
                        "reason": "verify passed after repair",
                        "trace": [
                            {
                                "step": 1,
                                "action": "patch",
                                "ok": True,
                                "observation": {"changed_paths": ["src/service.py"]},
                            },
                            {
                                "step": 2,
                                "action": "verify",
                                "ok": False,
                                "observation": {"passed": False, "reason": "test failed"},
                            },
                            {
                                "step": 3,
                                "action": "patch",
                                "ok": True,
                                "observation": {"changed_paths": ["src/service.py"]},
                            },
                            {
                                "step": 4,
                                "action": "verify",
                                "ok": True,
                                "observation": {"passed": True, "reason": "tests passed"},
                            },
                        ],
                    },
                },
            },
        },
    )

    assert result.ok is True
    check = result.checks["agent_loop_repair"]
    assert check["loop_source"] == "writeback.metadata.loop"
    assert check["final_verified"] is True
    assert check["repair_attempted"] is True
    assert check["repair_attempt_count"] == 1
    assert check["failed_actions_recovered"] is True
    assert check["recovered_failures"][0]["action"] == "verify"
    assert check["unresolved_failures"] == []


def test_agent_loop_repair_metadata_reports_unresolved_patch_failure(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="repo_patch",
        evidence=[
            {
                "kind": "patch_target",
                "path": "src/service.py",
                "text": "service patch target",
            }
        ],
        metadata={
            "loop": {
                "status": "refused",
                "steps": 1,
                "verified": False,
                "reason": "patch failed",
                "trace": [
                    {
                        "step": 1,
                        "action": "patch",
                        "ok": False,
                        "observation": {
                            "ok": False,
                            "reason": "search block did not match",
                            "changed_paths": [],
                        },
                    }
                ],
            }
        },
    )

    assert result.ok is False
    assert result.reason == "failed checks: agent_loop_repair"
    check = result.checks["agent_loop_repair"]
    assert check["final_verified"] is False
    assert check["failed_actions_recovered"] is False
    assert check["repair_attempted"] is False
    assert check["unresolved_failures"][0]["action"] == "patch"
    assert check["unresolved_failures"][0]["reason"] == "search block did not match"


def test_agent_loop_verified_metadata_does_not_override_materialization_failure(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter": {"model_id": "gemma-e2b", "tokenizer_id": "gemma-tokenizer"},
            "materialized": {
                "strategy": "refuse",
                "refused": True,
                "compatibility": {"model_id": "gemma-e2b", "tokenizer_id": "gemma-tokenizer"},
            },
            "loop": {
                "status": "verified",
                "verified": True,
                "trace": [
                    {
                        "step": 1,
                        "action": "verify",
                        "ok": True,
                        "observation": {"passed": True, "reason": "loop says verified"},
                    }
                ],
            },
        },
    )

    assert result.ok is False
    assert result.checks["agent_loop_repair"]["final_verified"] is True
    assert result.checks["agent_loop_repair"]["ok"] is True
    assert result.checks["adapter_materialization_compatibility"]["materializer_refused"] is True
    assert result.checks["adapter_materialization_compatibility"]["ok"] is False


def test_writeback_policy_and_user_lifecycle_metadata_are_verified_when_present(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="temporal_recall",
        evidence=[
            {
                "artifact_id": "user-source-1",
                "timestamp": "2026-05-04T01:00:00+00:00",
                "ordinal": 0,
                "text": "user preference",
            }
        ],
        metadata={
            "write_back_policy": {"family": "user", "targets": ["chat_user_memory"]},
            "writeback": {
                "artifact_id": "user-memory-1",
                "family": "user",
                "kind": "temporal_recall",
                "text": "verified user memory",
                "timestamp": "2026-05-04T02:00:00+00:00",
                "metadata": {
                    "expires_at": "2026-06-04T02:00:00+00:00",
                    "supersedes": ["user-memory-0"],
                    "supersession_status": "active",
                },
            },
        },
    )

    assert result.ok is True
    assert result.checks["memory_writeback"]["policy"]["target_ok"] is True
    assert result.checks["memory_writeback"]["user_lifecycle"]["expiry_ok"] is True
    assert result.checks["memory_writeback"]["user_lifecycle"]["supersedes"] == ["user-memory-0"]


def test_writeback_policy_and_user_lifecycle_mismatch_fail_with_details(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="temporal_recall",
        evidence=[
            {
                "artifact_id": "user-source-1",
                "timestamp": "2026-05-04T01:00:00+00:00",
                "ordinal": 0,
                "text": "user preference",
            }
        ],
        metadata={
            "write_back_policy": {"family": "task", "targets": ["code_task_memory"]},
            "writeback": {
                "artifact_id": "user-memory-1",
                "family": "user",
                "kind": "temporal_recall",
                "text": "verified user memory",
                "timestamp": "2026-05-04T02:00:00+00:00",
                "metadata": {
                    "expires_at": "2026-05-04T01:59:00+00:00",
                    "supersedes": ["user-memory-1"],
                    "supersession_status": "unknown",
                },
            },
        },
    )

    assert result.ok is False
    assert result.checks["memory_writeback"]["policy"]["family_ok"] is False
    lifecycle = result.checks["memory_writeback"]["user_lifecycle"]
    assert lifecycle["expiry_ok"] is False
    assert lifecycle["supersedes_ok"] is False
    assert lifecycle["supersession_status_ok"] is False


def test_sidecar_catalog_compatibility_metadata_is_verified_when_present(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "symbol": "boot", "text": "def boot(): ..."}],
        metadata={
            "adapter": {
                "model_id": "model-a",
                "tokenizer_id": "tokenizer-a",
                "model_revision": "rev-a",
                "adapter_family": "family-a",
                "insertion_family": "kv_direct",
                "kv_target_layer": 7,
            },
            "materialized": {
                "strategy": "kv_sidecar",
                "refused": False,
                "compatibility": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
                "materialization_plan": {
                    "strategy": "kv_sidecar",
                    "requested_strategy": "kv_sidecar",
                    "memory_family": "task",
                    "adapter_scope": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
                    "sidecars": [
                        {
                            "artifact_id": "kv-window-1",
                            "memory_family": "task",
                            "scope": {
                                "model_id": "model-a",
                                "tokenizer_id": "tokenizer-a",
                                "model_revision": "rev-a",
                                "adapter_family": "family-a",
                                "insertion_family": "kv_direct",
                                "kv_target_layer": 7,
                            },
                            "refs": [{"kind": "kv_cache", "layer": 7}],
                        }
                    ],
                    "replay_refs": [{"artifact_id": "kv-window-1", "kind": "kv_cache"}],
                },
            },
            "sidecar_catalog": {"sidecar_count": 1},
        },
    )

    assert result.ok is True
    assert result.checks["sidecar_catalog_compatibility"]["sidecar_artifact_ids"] == ["kv-window-1"]
    assert result.checks["sidecar_catalog_compatibility"]["replay_ref_count"] == 1


def test_sidecar_catalog_mismatch_fails_with_scope_details(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter": {"model_id": "model-a", "tokenizer_id": "tokenizer-a", "kv_target_layer": 7},
            "materialized": {
                "strategy": "kv_sidecar",
                "refused": False,
                "compatibility": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
                "materialization_plan": {
                    "strategy": "kv_sidecar",
                    "requested_strategy": "kv_sidecar",
                    "memory_family": "task",
                    "sidecars": [
                        {
                            "artifact_id": "kv-window-1",
                            "memory_family": "user",
                            "scope": {
                                "model_id": "other-model",
                                "tokenizer_id": "tokenizer-a",
                                "kv_target_layer": 9,
                            },
                            "refs": [{"kind": "kv_cache", "layer": 9}],
                        }
                    ],
                    "replay_refs": [{"artifact_id": "kv-window-1", "kind": "kv_cache"}],
                },
            },
            "sidecar_catalog": {"sidecar_count": 2},
        },
    )

    assert result.ok is False
    check = result.checks["sidecar_catalog_compatibility"]
    assert check["catalog_count_ok"] is False
    assert {"artifact_id": "kv-window-1", "key": "model_id", "adapter": "model-a", "sidecar": "other-model"} in check[
        "mismatches"
    ]
    assert {
        "artifact_id": "kv-window-1",
        "key": "memory_family",
        "route": "task",
        "sidecar": "user",
    } in check["mismatches"]


def test_residual_sidecar_replay_requires_sidecar_evidence_chain(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
            "materialized": {
                "strategy": "residual_sidecar",
                "refused": False,
                "compatibility": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
                "materialization_replay": {
                    "replay_family": "residual_sidecar",
                    "applied": True,
                    "tensor_replay_applied": True,
                    "refused": False,
                },
                "materialization_plan": {
                    "strategy": "residual_sidecar",
                    "requested_strategy": "residual_sidecar",
                    "sidecars": [{"artifact_id": "residual-window-1", "refs": [{"kind": "residual_stream"}]}],
                    "replay_refs": [{"artifact_id": "residual-window-1", "kind": "residual_stream"}],
                },
            },
        },
    )

    assert result.ok is True
    assert result.checks["sidecar_replay_evidence"]["applied"] is True
    assert result.checks["sidecar_replay_evidence"]["evidence_chain_count"] == 2


def test_residual_sidecar_refusal_requires_reason_and_chain(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)

    result = verifier.verify(
        capability="source_dependency",
        evidence=[{"path": "src/example.py", "text": "source evidence"}],
        metadata={
            "adapter": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
            "materialized": {
                "strategy": "refuse",
                "refused": True,
                "reason": "runtime replay consumer required for residual_sidecar",
                "compatibility": {"model_id": "model-a", "tokenizer_id": "tokenizer-a"},
                "materialization_replay": {
                    "replay_family": "residual_sidecar",
                    "applied": False,
                    "refused": True,
                    "refusal_reasons": ["runtime replay consumer required for residual_sidecar"],
                },
                "materialization_plan": {
                    "strategy": "refuse",
                    "requested_strategy": "residual_sidecar",
                    "refused": True,
                    "reason": "runtime replay consumer required for residual_sidecar",
                    "sidecars": [{"artifact_id": "residual-window-1", "refs": [{"kind": "residual_stream"}]}],
                    "replay_refs": [{"artifact_id": "residual-window-1", "kind": "residual_stream"}],
                },
            },
        },
    )

    assert result.ok is False
    replay_check = result.checks["sidecar_replay_evidence"]
    assert replay_check["refused"] is True
    assert replay_check["refusal_reason_count"] >= 1
    assert replay_check["evidence_chain_count"] == 2


def test_verifier_checks_decoder_product_route_and_backend_metadata(tmp_path: Path) -> None:
    verifier = _verifier(tmp_path)
    evidence = [{"path": "src/example.py", "symbol": "boot", "text": "def boot(): ..."}]

    result = verifier.verify(
        capability="source_dependency",
        evidence=evidence,
        metadata={
            "adapter": {
                "model_id": "gemma-e2b",
                "tokenizer_id": "gemma-tokenizer",
                "adapter_family": "gemma",
                "model_revision": "rev-a",
                "insertion_family": "kv_direct",
            },
            "decoder_prior_scope": {
                "model_id": "gemma-e2b",
                "tokenizer_id": "gemma-tokenizer",
                "adapter_family": "gemma",
                "model_revision": "rev-a",
                "insertion_family": "kv_direct",
                "method": "source_dependency",
            },
            "decoder_prior": {
                "scope": {
                    "model_id": "gemma-e2b",
                    "tokenizer_id": "gemma-tokenizer",
                    "adapter_family": "gemma",
                    "model_revision": "rev-a",
                    "insertion_family": "kv_direct",
                    "method": "source_dependency",
                }
            },
            "product_route": {
                "method": "source_dependency",
                "methodology": "dependency_source",
                "capability": "source/dependency routing",
                "proof_rig": "LoCoBench source/dependency routing",
                "route_reasons": ["source_dependency methodology selected"],
                "evidence": evidence,
            },
            "route_evidence_chain": [{"path": "src/example.py", "kind": "source_index_record"}],
            "backend": {"name": "offline-deterministic", "ok": True, "metadata": {"deterministic": True}},
        },
    )

    assert result.ok is True
    assert result.checks["decoder_prior_scope"]["has_persisted_prior"] is True
    assert result.checks["product_route"]["route_evidence_chain_count"] == 1
    assert result.checks["backend"]["name"] == "offline-deterministic"
