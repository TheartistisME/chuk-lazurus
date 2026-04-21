"""CUDA-gated end-to-end verification: full 5-session demo.

Runs :func:`run_demo` with all five default plans, then asserts:

- token_budget_met is True (>128k gemma tokens)
- all_six_strict_per_query_pass is True (18 True values across three
  QueryResult.strict_assertions dicts)
- all verbatim_recall_per_query are True
- acceptance_passed is True
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for cross-session-demo end-to-end",
    ),
]


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
