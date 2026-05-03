from __future__ import annotations

from David.decoding_prior_store import (
    DecoderPriorScope,
    decoder_dialect_prior_plan,
    load_decoder_dialect_prior_store,
    new_decoder_dialect_prior_state,
    save_decoder_dialect_prior_store,
    update_decoder_dialect_prior_state,
)


def test_decoder_prior_store_round_trips_same_dialect_seed(tmp_path) -> None:
    scope = DecoderPriorScope(
        model_key="gemma4-e2b",
        tokenizer_key="gemma4-e2b",
        layer_index=14,
        steering_version="lm_head_logit_lens_v1",
        benchmark="ScaleAI/SWE-bench_Pro",
        task_type="BUILDER",
        runner="run_swebench_pro_parity.py",
    )
    dialect_plan = {
        "paths": {
            "src/app.js": {"target_dialect": "javascript"},
        },
    }
    prior = new_decoder_dialect_prior_state(scope=scope)
    update_decoder_dialect_prior_state(
        prior,
        dialect_plan=dialect_plan,
        steering={
            "hard_masked_candidate_count": 100,
            "masked_but_not_tempting_count": 99,
            "trigger_count": 1,
            "applied_count": 3,
            "alpha": {"peak_by_family": {"python": 0.75}},
            "events": [{"offense_family": "python"}],
        },
        accepted=True,
        case_metadata={"row_index": 2, "repo": "NodeBB/NodeBB", "instance_id": "row2"},
    )

    path = tmp_path / "decoder_prior.json"
    save_meta = save_decoder_dialect_prior_store(path, prior)
    loaded, load_meta = load_decoder_dialect_prior_store(path, scope=scope)
    plan = decoder_dialect_prior_plan(loaded, dialect_plan)

    assert save_meta["saved"] is True
    assert load_meta["loaded"] is True
    assert plan["active"] is True
    assert plan["matched_dialects"] == ["javascript"]
    assert plan["seed_alpha_by_family"] == {"python": 0.18}


def test_decoder_prior_store_rejects_incompatible_model_scope(tmp_path) -> None:
    old_scope = DecoderPriorScope(model_key="gemma4-e2b", tokenizer_key="gemma4-e2b")
    new_scope = DecoderPriorScope(model_key="qwen", tokenizer_key="qwen")
    path = tmp_path / "decoder_prior.json"
    save_decoder_dialect_prior_store(
        path,
        new_decoder_dialect_prior_state(scope=old_scope),
    )

    loaded, metadata = load_decoder_dialect_prior_store(path, scope=new_scope)

    assert metadata["loaded"] is False
    assert metadata["reason"] == "scope_mismatch"
    assert loaded["scope"]["model_key"] == "qwen"
    assert loaded["seen_cases"] == 0
