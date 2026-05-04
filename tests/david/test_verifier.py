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
            {"artifact_id": "parse", "text": "parser calls lexer", "ordinal": 0},
            {"artifact_id": "lexer", "text": "lexer reads tokens", "ordinal": 1},
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
            {"artifact_id": "parse", "text": "parser calls lexer", "ordinal": 0},
            {"artifact_id": "lexer", "text": "lexer emits tokens", "ordinal": 1},
            {"artifact_id": "emit", "text": "emitter formats output", "ordinal": 2},
        ],
        metadata={"expected_hops": ["parse", "lexer", "emit"]},
    )

    assert result.ok is True
    assert result.checks["symbolic_chain"]["ordered_hops"] == 3


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
            }
        },
    )

    assert result.ok is True
    assert result.checks["patch_targets"]["unsafe_paths"] == []
    assert result.checks["memory_writeback"]["expected_family"] == "task"


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
        },
    )

    assert result.ok is True
    assert result.checks["adapter_materialization_compatibility"]["strategy"] == "kv_direct"


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
