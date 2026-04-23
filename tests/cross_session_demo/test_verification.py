"""CUDA-gated end-to-end verification: full 5-session demo.

Runs :func:`run_demo` with all five default plans, then asserts:

- token_budget_met is True (>128k gemma tokens)
- all_six_strict_per_query_pass is True (18 True values across three
  QueryResult.strict_assertions dicts)
- all verbatim_recall_per_query are True
- acceptance_passed is True

Plus non-CUDA pure-unit tests for the axis-6 final-impl surface:
``build_kv_direct_execution`` normalisation and the optional-fields
default=None regression guard on :class:`QueryExecution`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


# ─── non-CUDA pure-unit tests (run in default env) ─────────────────────────


def test_query_execution_optional_fields_default_none() -> None:
    """Every axis-6 optional field on ``QueryExecution`` must default to None.

    Regression guard: _build_execution (used by the existing three-path
    demo) constructs a QueryExecution without axis-6 fields — if any new
    field became required, every cross-session demo serialisation would
    break. This test locks the default=None contract.
    """
    from chuk_lazarus.cross_session_demo.verification import (
        QueryExecution,
        SIX_STRICT_KEYS,
    )

    qe = QueryExecution(
        mode="exact",
        query_text="handle.0.0",
        source_session="sess-0001",
        window_id=0,
        routing_score=None,
        generated_answer="some answer",
        strict_assertions={k: True for k in SIX_STRICT_KEYS},
        verbatim_hit=True,
        planted_phrase="planted",
        matched_window_text="matched",
    )
    assert qe.selected_tier is None
    assert qe.mask_penalty_applied is None
    assert qe.kv_direct_active is None
    assert qe.path_a_replay_count is None
    assert qe.hot_budget_mib_observed is None
    assert qe.vram_peak_mib is None
    assert qe.vram_delta_mib is None
    assert qe.tier_counts_selected is None
    assert qe.no_silent_fallback is None


def _fake_kv_result(
    *,
    generated_answer: str = "the saffron sidecar remembers hexagon gullies at sunrise.",
    matched_window_text: str = "window text",
    kv_direct_active: bool = True,
    path_a_replay_count: int = 3,
    hot_budget_mib_observed: int = 32,
    vram_peak_mib: int = 512,
    vram_delta_mib: int = 128,
    tier_counts_selected: int = 2,
    mask_penalty_applied: bool = True,
) -> SimpleNamespace:
    """Construct a minimal SimpleNamespace shaped like a KV-direct QueryResult."""
    return SimpleNamespace(
        routing_mode="kv_direct",
        source_session="sess-0001",
        window_id=7,
        matched_window_text=matched_window_text,
        window_keywords=[],
        generated_answer=generated_answer,
        routing_score=None,
        strict_assertions={
            "kv_direct_active": kv_direct_active,
            "path_a_replay_count": path_a_replay_count,
            "hot_budget_mib_observed": hot_budget_mib_observed,
            "vram_peak_mib": vram_peak_mib,
            "vram_delta_mib": vram_delta_mib,
            "tier_counts_selected": tier_counts_selected,
            "mask_penalty_applied": mask_penalty_applied,
        },
    )


def test_build_kv_direct_execution_populates_axis6_fields() -> None:
    """The helper must surface the real axis-5 strict_assertions across the
    axis-6 fields AND infer the six strict booleans truthfully."""
    from chuk_lazarus.cross_session_demo.verification import (
        SIX_STRICT_KEYS,
        build_kv_direct_execution,
    )

    planted = "the saffron sidecar remembers hexagon gullies at sunrise."
    result = _fake_kv_result(generated_answer=planted)
    qe = build_kv_direct_execution(
        query_text="what was the planted phrase?",
        planted_phrase=planted,
        result=result,
    )

    # axis-6 fields populate from strict_assertions.
    assert qe.kv_direct_active is True
    assert qe.mask_penalty_applied is True
    assert qe.path_a_replay_count == 3
    assert qe.hot_budget_mib_observed == 32
    assert qe.vram_peak_mib == 512
    assert qe.vram_delta_mib == 128
    assert qe.tier_counts_selected == 2
    assert qe.no_silent_fallback is True

    # selected_tier must NOT be the legacy sentinel.
    assert qe.selected_tier != "not-implemented-yet"
    assert qe.selected_tier == "kv_direct"

    # Six strict booleans inferred truthfully for the CUDA KV-direct path.
    for key in SIX_STRICT_KEYS:
        assert qe.strict_assertions.get(key) is True, (
            f"strict key {key} should be True on real KV-direct path"
        )

    # Verbatim hit detection still works.
    assert qe.verbatim_hit is True
    assert qe.mode == "kv_direct"


def test_build_kv_direct_execution_exception_caught_sets_no_silent_fallback_false() -> None:
    """When the caller caught an exception, axis-6 evidence must NOT be
    fabricated: no_silent_fallback=False AND the six strict booleans all
    False to reflect that the runtime did not produce valid evidence."""
    from chuk_lazarus.cross_session_demo.verification import (
        SIX_STRICT_KEYS,
        build_kv_direct_execution,
    )

    planted = "the saffron sidecar remembers hexagon gullies at sunrise."
    result = _fake_kv_result(generated_answer=planted)
    qe = build_kv_direct_execution(
        query_text="what was the planted phrase?",
        planted_phrase=planted,
        result=result,
        exception_caught=True,
    )

    assert qe.no_silent_fallback is False
    for key in SIX_STRICT_KEYS:
        assert qe.strict_assertions.get(key) is False, (
            f"strict key {key} must be False when exception_caught=True"
        )


def test_build_kv_direct_execution_selected_tier_override() -> None:
    """The caller can pass an explicit selected_tier string from tier grouping."""
    from chuk_lazarus.cross_session_demo.verification import (
        build_kv_direct_execution,
    )

    planted = "planted"
    result = _fake_kv_result()
    qe = build_kv_direct_execution(
        query_text="q",
        planted_phrase=planted,
        result=result,
        selected_tier_override="hot+warm",
    )
    assert qe.selected_tier == "hot+warm"


# ─── CUDA-gated end-to-end test ─────────────────────────────────────────────


def _cuda_available() -> bool:
    try:
        import torch  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.cuda
@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA required for cross-session-demo end-to-end",
)
def test_end_to_end_cross_session_demo_passes(tmp_path: Path) -> None:
    from chuk_lazarus.cross_session_demo import (
        SIX_STRICT_KEYS,
        run_demo,
    )

    inputs_root = tmp_path / "inputs"
    checkpoints_root = tmp_path / "checkpoints"
    inputs_root.mkdir()
    checkpoints_root.mkdir()

    demo = run_demo(
        inputs_root=inputs_root,
        checkpoints_root=checkpoints_root,
        device="cuda",
    )
    report = demo.report

    assert report.num_sessions >= 5
    assert report.token_budget_met, (
        f"token_budget not met: total_tokens={report.total_tokens}"
    )
    assert report.all_six_strict_per_query_pass, (
        f"strict assertions failed: {report.query_executions}"
    )
    assert all(report.verbatim_recall_per_query), (
        f"verbatim recall failed: {report.verbatim_recall_per_query}"
    )
    # Each QueryExecution must carry all 6 canonical keys == True.
    for ex in report.query_executions:
        for key in SIX_STRICT_KEYS:
            assert ex.strict_assertions.get(key) is True, (
                f"strict key {key} False in mode={ex.mode}"
            )

    assert report.acceptance_passed, (
        f"acceptance FAILED: zero_mod={report.zero_mod_upstream}"
    )
