"""CUDA-gated integration test: verbatim planted-phrase recall.

Asserts that each of the three query paths produces a generated answer
that contains the planted phrase of its target session verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for verbatim-recall test",
    ),
]


def test_verbatim_recall_on_every_query_path(tmp_path: Path) -> None:
    from chuk_lazarus.cross_session_demo import run_demo

    inputs_root = tmp_path / "inputs"
    checkpoints_root = tmp_path / "checkpoints"
    inputs_root.mkdir()
    checkpoints_root.mkdir()

    demo = run_demo(
        inputs_root=inputs_root,
        checkpoints_root=checkpoints_root,
        device="cuda",
    )

    assert len(demo.report.query_executions) == 3
    for ex in demo.report.query_executions:
        assert ex.verbatim_hit, (
            f"mode={ex.mode}: planted phrase {ex.planted_phrase!r} NOT in "
            f"generated answer: {ex.generated_answer!r}"
        )
