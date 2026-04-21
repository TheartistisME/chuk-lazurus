"""CUDA-gated integration test focused on the six strict-mode assertions.

Runs the three query paths and asserts that for each resulting
``QueryResult.strict_assertions`` dict, every one of the six canonical
keys is ``True``. Also asserts the dict is exactly those six keys (no
missing, no extra).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for strict-mode assertions test",
    ),
]


def test_six_strict_assertions_pass_on_every_query(tmp_path: Path) -> None:
    from chuk_lazarus.cross_session_demo import SIX_STRICT_KEYS, run_demo

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
    canonical = set(SIX_STRICT_KEYS)
    for ex in demo.report.query_executions:
        keys = set(ex.strict_assertions.keys())
        assert keys == canonical, (
            f"mode={ex.mode}: keys {keys} != canonical {canonical}"
        )
        for key in SIX_STRICT_KEYS:
            assert ex.strict_assertions[key] is True, (
                f"mode={ex.mode}, key={key} is False"
            )
