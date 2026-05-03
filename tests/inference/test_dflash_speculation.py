from __future__ import annotations

import json

import numpy as np
import pytest

from chuk_lazarus.inference.speculative import (
    accept_speculated_prefix,
    build_gemma4_e2b_dflash_plan,
    build_target_layer_ids,
    extract_context_feature,
    load_dflash_calibration,
)


def test_build_target_layer_ids_matches_official_qwen3_8b_config():
    assert build_target_layer_ids(36, 5) == (1, 9, 17, 25, 33)


def test_gemma_plan_prefers_calibrated_layer_13_to_14_seam():
    plan = build_gemma4_e2b_dflash_plan(mask_token_id=123)

    assert plan.target_layer_ids == (4, 9, 13, 14, 29)
    assert plan.calibration.route_layer == 13
    assert plan.calibration.kv_source_layer == 13
    assert plan.calibration.kv_target_layer == 14
    assert plan.mask_token_id == 123
    assert plan.as_config()["dflash_config"]["calibration"]["confidence"] == "high"


def test_gemma_plan_can_emit_uniform_ablation_layers():
    plan = build_gemma4_e2b_dflash_plan(calibrated_context=False)

    assert plan.target_layer_ids == (1, 9, 16, 24, 32)


def test_extract_context_feature_concatenates_hf_decoder_hidden_states():
    hidden_states = [
        np.full((1, 2, 3), float(idx), dtype=np.float32)
        for idx in range(6)
    ]

    feature = extract_context_feature(hidden_states, [0, 2, 4])

    assert feature.shape == (1, 2, 9)
    assert np.array_equal(feature[..., :3], np.full((1, 2, 3), 1.0))
    assert np.array_equal(feature[..., 3:6], np.full((1, 2, 3), 3.0))
    assert np.array_equal(feature[..., 6:], np.full((1, 2, 3), 5.0))


def test_extract_context_feature_rejects_out_of_range_layer():
    hidden_states = [np.zeros((1, 1, 2)) for _ in range(2)]

    with pytest.raises(IndexError):
        extract_context_feature(hidden_states, [3])


def test_accept_speculated_prefix_returns_bonus_token_on_mismatch():
    result = accept_speculated_prefix(
        draft_token_ids=[10, 11, 12, 13],
        posterior_token_ids=[11, 99, 42, 43],
    )

    assert result.accepted_length == 2
    assert result.accepted_token_ids == (10, 11)
    assert result.bonus_token_id == 99


def test_accept_speculated_prefix_handles_full_block_acceptance():
    result = accept_speculated_prefix(
        draft_token_ids=[10, 11, 12],
        posterior_token_ids=[11, 12, 13],
    )

    assert result.accepted_length == 3
    assert result.accepted_token_ids == (10, 11, 12)
    assert result.bonus_token_id == 13


def test_load_dflash_calibration_reads_david_report(tmp_path):
    report = {
        "model": {"model_revision_or_hash": "abc"},
        "adapter_config_candidate": {
            "route_layer": 13,
            "boundary_layer": 13,
            "kv_source_layer": 13,
            "kv_target_layer": 14,
            "insertion_family": "full_attention",
            "confidence": "high",
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    calibration = load_dflash_calibration(path)

    assert calibration.route_layer == 13
    assert calibration.kv_target_layer == 14
    assert calibration.model_revision_or_hash == "abc"
