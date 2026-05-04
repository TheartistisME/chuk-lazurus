from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from chuk_lazarus.david.model_attestation import (
    ATTESTATION_SCHEMA_NAME,
    selected_config_sha256,
    sha256_bytes,
    verify_manual_model_attestation,
)


NOW = datetime(2026, 5, 4, tzinfo=timezone.utc)


def test_manual_attestation_accepts_reviewed_needs_review_for_standard_decode(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    report_hash = sha256_bytes(report_path.read_bytes())
    attestation_path = _write_json(
        tmp_path / "attestation.json",
        _attestation_payload(
            validation_report_sha256=report_hash,
            selected_config_sha256=selected_config_sha256(
                _validation_report_payload()["selected_config"]
            ),
        ),
    )

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is True
    assert result.reasons == ()
    assert result.source_validation_status == "needs_review"
    assert result.source_auto_load_allowed is False
    assert result.selected_config["kv_target_layer"] == 29
    assert result.allowed_capabilities == ("standard_decode",)
    assert result.adapter_metadata is not None
    assert result.adapter_metadata.model_id == "google/gemma-4-E2B-it"
    assert result.adapter_metadata.tokenizer_id == "google/gemma-4-E2B-it"
    assert result.adapter_metadata.model_revision == "b4a601102c3d"
    assert result.adapter_metadata.adapter_family == "gemma4"
    assert result.adapter_metadata.kv_source_layer == 28
    assert result.adapter_metadata.kv_target_layer == 29
    assert result.adapter_scope()["insertion_family"] == "kv_direct"


def test_manual_attestation_accepts_explicit_selected_config_binding(tmp_path):
    report = _validation_report_payload()
    report_path = _write_json(tmp_path / "validation_report.json", report)
    attestation = _attestation_payload(
        validation_report_sha256=sha256_bytes(report_path.read_bytes()),
        selected_config=report["selected_config"],
    )
    attestation_path = _write_json(tmp_path / "attestation.json", attestation)

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is True
    assert result.selected_config["adapter_config_id"] == "gemma4-e2b-reviewed"


def test_manual_attestation_rejects_validation_report_digest_mismatch(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    attestation_path = _write_json(
        tmp_path / "attestation.json",
        _attestation_payload(
            validation_report_sha256="0" * 64,
            selected_config_sha256=selected_config_sha256(
                _validation_report_payload()["selected_config"]
            ),
        ),
    )

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert "validation_report_sha256 does not match validation report" in result.reasons
    assert result.adapter_metadata is None


def test_manual_attestation_rejects_selected_config_mismatch(tmp_path):
    report = _validation_report_payload()
    report_path = _write_json(tmp_path / "validation_report.json", report)
    mismatched_config = dict(report["selected_config"])
    mismatched_config["kv_target_layer"] = 31
    attestation_path = _write_json(
        tmp_path / "attestation.json",
        _attestation_payload(
            validation_report_sha256=sha256_bytes(report_path.read_bytes()),
            selected_config=mismatched_config,
        ),
    )

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert "selected_config does not match validation report" in result.reasons


def test_manual_attestation_rejects_identity_scope_mismatch(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    attestation = _attestation_payload(
        validation_report_sha256=sha256_bytes(report_path.read_bytes()),
        selected_config_sha256=selected_config_sha256(
            _validation_report_payload()["selected_config"]
        ),
    )
    attestation["model_revision_or_hash"] = "different-revision"
    attestation_path = _write_json(tmp_path / "attestation.json", attestation)

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert (
        "model_revision_or_hash mismatch: attestation='different-revision', "
        "validation_report='b4a601102c3d'"
    ) in result.reasons


def test_manual_attestation_rejects_expired_attestation(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    attestation_path = _write_json(
        tmp_path / "attestation.json",
        _attestation_payload(
            validation_report_sha256=sha256_bytes(report_path.read_bytes()),
            selected_config_sha256=selected_config_sha256(
                _validation_report_payload()["selected_config"]
            ),
            expires_at="2026-05-03T00:00:00Z",
        ),
    )

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert "attestation is expired" in result.reasons


def test_manual_attestation_rejects_missing_reviewer_and_rationale(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    attestation = _attestation_payload(
        validation_report_sha256=sha256_bytes(report_path.read_bytes()),
        selected_config_sha256=selected_config_sha256(
            _validation_report_payload()["selected_config"]
        ),
    )
    attestation.pop("reviewer")
    attestation["rationale"] = ""
    attestation_path = _write_json(tmp_path / "attestation.json", attestation)

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert "reviewer is missing" in result.reasons
    assert "rationale is missing" in result.reasons


def test_manual_attestation_rejects_dangerous_capabilities(tmp_path):
    report_path = _write_json(tmp_path / "validation_report.json", _validation_report_payload())
    attestation_path = _write_json(
        tmp_path / "attestation.json",
        _attestation_payload(
            validation_report_sha256=sha256_bytes(report_path.read_bytes()),
            selected_config_sha256=selected_config_sha256(
                _validation_report_payload()["selected_config"]
            ),
            allowed_capabilities=["standard_decode", "kv_replay"],
        ),
    )

    result = verify_manual_model_attestation(
        attestation_path=attestation_path,
        validation_report_path=report_path,
        now=NOW,
    )

    assert result.accepted is False
    assert "dangerous capabilities are not permitted: kv_replay" in result.reasons
    assert "allowed_capabilities must be exactly ['standard_decode'] for now" in result.reasons


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _attestation_payload(
    *,
    validation_report_sha256: str,
    selected_config_sha256: str | None = None,
    selected_config: dict[str, object] | None = None,
    allowed_capabilities: list[str] | None = None,
    expires_at: str | None = "2026-06-01T00:00:00Z",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_name": ATTESTATION_SCHEMA_NAME,
        "schema_version": 1,
        "validation_report_sha256": validation_report_sha256,
        "model_identity": "google/gemma-4-E2B-it",
        "tokenizer_identity": "google/gemma-4-E2B-it",
        "model_revision_or_hash": "b4a601102c3d",
        "adapter_family": "gemma4",
        "allowed_capabilities": allowed_capabilities or ["standard_decode"],
        "reviewer": "jehma",
        "reviewed_at": "2026-05-04T07:00:00Z",
        "rationale": "Operator reviewed the tied Gemma adapter candidate for standard decode only.",
    }
    if selected_config_sha256 is not None:
        payload["selected_config_sha256"] = selected_config_sha256
    if selected_config is not None:
        payload["selected_config"] = selected_config
    if expires_at is not None:
        payload["expires_at"] = expires_at
    return payload


def _validation_report_payload() -> dict[str, object]:
    selected_config = {
        "adapter_config_id": "gemma4-e2b-reviewed",
        "route_layer": 14,
        "route_query_head": 3,
        "route_dimension": 1536,
        "boundary_layer": 21,
        "residual_capture_layer": 21,
        "kv_source_layer": 28,
        "kv_target_layer": 29,
        "injection_layer": 29,
        "projection_producer_layer": 28,
        "behavior_cache_layer": 14,
        "insertion_family": "kv_direct",
        "kv_layout": "bshd",
        "candidate_role": "manual_reviewed",
        "hidden_size": 1536,
    }
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": "needs_review",
        "confidence": "low",
        "validation_level": "topology_projection_behavior",
        "auto_load_allowed": False,
        "harness_load_policy": "manual_review_required",
        "model_identity_gate": {
            "status": "ok",
            "model": "google/gemma-4-E2B-it",
            "model_identity": "google/gemma-4-E2B-it",
            "tokenizer_identity": "google/gemma-4-E2B-it",
            "model_revision_or_hash": "b4a601102c3d",
            "adapter_family": "gemma4",
            "hidden_size": 1536,
            "num_attention_heads": 12,
            "num_key_value_heads": 4,
        },
        "source_report_summary": {
            "model": "google/gemma-4-E2B-it",
            "tokenizer_identity": "google/gemma-4-E2B-it",
            "model_revision_or_hash": "b4a601102c3d",
            "adapter_family": "gemma4",
        },
        "selected_config": selected_config,
        "projection_gate": {"status": "needs_review", "ranked_candidates": [selected_config]},
        "behavior_gate": {"status": "needs_review", "warnings": ["candidate tie"]},
        "warnings": ["manual review required"],
    }
