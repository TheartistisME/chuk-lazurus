"""Regression tests for David/get_model_config.py report helper contracts.

These tests intentionally import the script by file path and exercise only
pure data helpers. They must not instantiate or download a model.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_HELPER_PATH = REPO_ROOT / "David" / "get_model_config.py"


def load_config_helper():
    spec = importlib.util.spec_from_file_location(
        "david_get_model_config_for_report_contract", CONFIG_HELPER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_helper(module, name: str):
    assert hasattr(module, name), (
        f"David/get_model_config.py must expose pure helper {name}() for the "
        "production-grade config getter report contract."
    )
    return getattr(module, name)


def test_adapter_family_from_config_identifies_useful_text_model_families():
    module = load_config_helper()
    adapter_family_from_config = require_helper(module, "adapter_family_from_config")

    cases = {
        "gemma4_text": "gemma4",
        "qwen3_5_text": "qwen3",
        "llama": "llama",
    }

    for model_type, expected_family in cases.items():
        config = SimpleNamespace(text_config=SimpleNamespace(model_type=model_type))

        family = adapter_family_from_config(config)

        assert family == expected_family
        assert family != "fallback"


def test_projection_resolver_finds_output_projection_aliases():
    module = load_config_helper()
    resolve_attention_projection = require_helper(module, "resolve_attention_projection")

    sentinel = object()
    for alias in ("o_proj", "out_proj", "dense"):
        attn = SimpleNamespace(**{alias: sentinel})

        resolved, resolved_alias, checked = resolve_attention_projection(
            attn, "o", "fallback"
        )

        assert resolved is sentinel
        assert resolved_alias == alias
        assert alias in checked


def test_projection_diagnostics_report_unavailable_output_projection_alias():
    module = load_config_helper()
    projection_diagnostics = require_helper(module, "projection_diagnostics")

    config = SimpleNamespace(text_config=SimpleNamespace(model_type="gemma4_text"))
    attn = SimpleNamespace(q_proj=object(), k_proj=object(), v_proj=object())

    diagnostics = projection_diagnostics(config, attn, roles=("o",))

    assert diagnostics["adapter_family"] == "gemma4"
    assert diagnostics["resolved_aliases"]["o"] is None
    assert "o" in diagnostics["missing_projections"]
    assert {"o_proj", "out_proj", "dense"}.issubset(
        set(diagnostics["aliases_checked"]["o"])
    )


def test_gemma_like_legacy_kv_disagreement_blocks_ready_adapter_config():
    module = load_config_helper()
    review_recommendation = require_helper(module, "review_recommendation")
    adapter_config_candidate = require_helper(module, "adapter_config_candidate")

    recommendation = {
        "route_layer": 15,
        "boundary_layer": 15,
        "kv_source_layer": 15,
        "kv_target_layer": 16,
        "insertion_family": "full_attention",
        "route_query_head": 3,
        "confidence": "materialization_safe",
    }
    review = review_recommendation(
        recommendation=recommendation,
        legacy={
            "status": "ok",
            "legacy_retrieval_layer": 25,
            "legacy_injection_layer": 26,
            "avg_dot_by_layer": {"25": 0.8, "26": 1.0},
        },
        head_scan={"status": "ok", "best_head": 3, "routing_score": 0.7},
        candidates=[
            {
                "kv_source_layer": 15,
                "kv_target_layer": 16,
                "materialization_safe": True,
                "candidate_score": 10.0,
            },
            {
                "kv_source_layer": 25,
                "kv_target_layer": 26,
                "materialization_safe": True,
                "candidate_score": 7.0,
            },
        ],
        min_score_margin=0.25,
    )

    candidate = adapter_config_candidate(
        model_identity_data={
            "model": "google/gemma-4-4b-it",
            "model_revision_or_hash": "abc123",
            "text_model_type": "gemma4_text",
        },
        recommendation=recommendation,
        review=review,
    )

    assert review["status"] == "needs_review"
    assert review["confidence"] == "review_required"
    assert review["needs_review"] is True
    assert review["meets_minimum_confidence"] is False
    assert any("legacy" in warning.lower() for warning in review["warnings"])
    assert candidate["status"] == "review_required"


def test_ready_adapter_config_allowed_when_legacy_matches_and_margin_strong_without_head_scan():
    module = load_config_helper()
    review_recommendation = require_helper(module, "review_recommendation")
    adapter_config_candidate = require_helper(module, "adapter_config_candidate")

    recommendation = {
        "route_layer": 15,
        "boundary_layer": 15,
        "kv_source_layer": 15,
        "kv_target_layer": 16,
        "insertion_family": "full_attention",
        "route_query_head": None,
        "confidence": "materialization_safe",
    }
    review = review_recommendation(
        recommendation=recommendation,
        legacy={
            "status": "ok",
            "legacy_retrieval_layer": 15,
            "legacy_injection_layer": 16,
            "avg_dot_by_layer": {"15": 0.8, "16": 1.0},
        },
        head_scan={"status": "unavailable", "reason": "inspect-only run"},
        candidates=[
            {
                "kv_source_layer": 15,
                "kv_target_layer": 16,
                "materialization_safe": True,
                "candidate_score": 10.0,
            },
            {
                "kv_source_layer": 16,
                "kv_target_layer": 17,
                "materialization_safe": True,
                "candidate_score": 2.5,
            },
        ],
        min_score_margin=0.25,
    )

    candidate = adapter_config_candidate(
        model_identity_data={
            "model": "google/gemma-4-4b-it",
            "model_revision_or_hash": "abc123",
            "text_model_type": "gemma4_text",
        },
        recommendation=recommendation,
        review=review,
    )

    assert review["confidence"] == "medium"
    assert review["meets_minimum_confidence"] is True
    assert review["needs_review"] is False
    assert candidate["status"] == "ready"


def test_review_flags_legacy_injection_disagrees_with_recommended_target():
    module = load_config_helper()
    review_recommendation = require_helper(module, "review_recommendation")

    review = review_recommendation(
        recommendation={
            "route_layer": 15,
            "kv_source_layer": 15,
            "kv_target_layer": 16,
            "confidence": "materialization_safe",
        },
        legacy={
            "status": "ok",
            "legacy_retrieval_layer": 25,
            "legacy_injection_layer": 26,
            "avg_dot_by_layer": {"25": 0.8, "26": 1.0},
        },
        head_scan={"status": "ok", "best_head": 3, "routing_score": 0.7},
        candidates=[
            {
                "kv_source_layer": 15,
                "kv_target_layer": 16,
                "materialization_safe": True,
                "candidate_score": 10.0,
            },
            {
                "kv_source_layer": 25,
                "kv_target_layer": 26,
                "materialization_safe": True,
                "candidate_score": 9.5,
            },
        ],
        min_score_margin=0.25,
    )

    assert (
        review.get("needs_review") is True
        or review.get("confidence") in {"low", "review_required"}
        or review.get("confidence_level") in {"low", "review_required"}
    )
    assert any("legacy" in warning.lower() for warning in review.get("warnings", []))


def test_adapter_config_candidate_is_harness_ready_and_keeps_injection_alias():
    module = load_config_helper()
    adapter_config_candidate = require_helper(module, "adapter_config_candidate")

    candidate = adapter_config_candidate(
        model_identity_data={
            "model": "google/gemma-3-4b-it",
            "model_revision_or_hash": "abc123",
            "text_model_type": "gemma3_text",
        },
        recommendation={
            "route_layer": 15,
            "boundary_layer": 15,
            "kv_source_layer": 15,
            "kv_target_layer": 16,
            "insertion_family": "full_attention",
            "route_query_head": 3,
            "confidence": "materialization_safe",
        },
        review={"status": "ok", "meets_minimum_confidence": True, "confidence": "high"},
    )

    assert candidate["status"] == "ready"
    assert candidate["route_layer"] == 15
    assert candidate["boundary_layer"] == 15
    assert candidate["kv_source_layer"] == 15
    assert candidate["kv_target_layer"] == 16
    assert candidate["injection_layer"] == 16
    assert candidate["route_query_head"] == 3
    assert candidate["confidence"] == "high"


def test_score_margin_distinguishes_clear_winner_from_tie_or_weak_margin():
    module = load_config_helper()
    score_margin = require_helper(module, "score_margin")

    clear = score_margin(
        [
            {"kv_target_layer": 16, "candidate_score": 10.0},
            {"kv_target_layer": 17, "candidate_score": 2.5},
        ]
    )
    weak = score_margin(
        [
            {"kv_target_layer": 16, "candidate_score": 10.0},
            {"kv_target_layer": 17, "candidate_score": 9.9},
        ]
    )
    tie = score_margin(
        [
            {"kv_target_layer": 16, "candidate_score": 10.0},
            {"kv_target_layer": 17, "candidate_score": 10.0},
        ]
    )

    assert clear["status"] == "ok"
    assert clear["best_kv_target_layer"] == 16
    assert clear["margin"] > 1.0
    assert 0 < weak["margin"] < clear["margin"]
    assert tie["margin"] == 0.0


def test_unavailable_section_with_partial_payload_is_json_serializable():
    module = load_config_helper()
    section_status = require_helper(module, "section_status")

    section = section_status(
        "unavailable",
        "legacy scan unavailable",
        section="head_ablation",
        partial={
            "candidate_layers": [15, 16],
            "scores": {"15": float("nan"), "16": float("inf")},
            "path": CONFIG_HELPER_PATH,
        },
    )

    assert section["status"] == "unavailable"
    assert section["reason"] == "legacy scan unavailable"
    assert "partial" in section
    encoded = json.dumps(module.json_safe(section), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["partial"]["scores"] == {"15": None, "16": None}
    assert decoded["partial"]["path"].endswith("David/get_model_config.py")


def test_failure_report_is_structured_and_harness_blocking():
    module = load_config_helper()
    failure_report = require_helper(module, "failure_report")

    args = module.parse_args(["--model", "missing/local-model", "--local-files-only"])
    report = failure_report(
        args,
        1777710000.0,
        stage="model_load",
        error=RuntimeError("no weights found"),
    )

    assert report["schema_name"] == module.SCHEMA_NAME
    assert report["schema_version"] == module.SCHEMA_VERSION
    assert report["status"] == "failed"
    assert report["failure"]["status"] == "failed"
    assert report["failure"]["stage"] == "model_load"
    assert report["adapter_config_candidate"]["status"] == "review_required"
    assert report["recommendation_review"]["status"] == "unavailable"
    assert report["kv_candidates"] == []

    encoded = json.dumps(module.json_safe(report), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["provenance"]["loader_options"]["model"] == "missing/local-model"
    assert "model_load" in decoded["warnings"][0]
