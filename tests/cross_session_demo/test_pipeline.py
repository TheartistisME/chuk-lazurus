"""CUDA-gated integration test for ``pipeline.run_session``.

Runs one plan through axes 1-3 and asserts the per-session torch_store
manifest landed on disk with a zero returncode.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for cross-session-demo pipeline",
    ),
]


def test_pipeline_single_session_builds_checkpoint(tmp_path: Path) -> None:
    from chuk_lazarus.cross_session_demo.pipeline import run_session
    from chuk_lazarus.cross_session_demo.session_generator import DEFAULT_PLANS

    inputs_root = tmp_path / "inputs"
    checkpoints_root = tmp_path / "checkpoints"

    plan = DEFAULT_PLANS[0]
    result = run_session(plan, inputs_root, checkpoints_root, device="cuda")

    assert result["returncode"] == 0
    checkpoint = Path(result["checkpoint"])
    manifest_path = checkpoint / "torch_store" / "manifest.json"
    assert manifest_path.is_file(), f"manifest missing at {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest.get("clause_aligned") is True
    assert int(manifest.get("num_windows", 0)) >= 1
    assert int(manifest.get("num_entries", 0)) >= 1

    # Planted handle must round-trip: the session dir under inputs/ has
    # per-turn json files whose clause_id points at the planted turn.
    sid = result["session_id"]
    session_input_dir = inputs_root / sid
    assert session_input_dir.is_dir()
    planted_handle = result["planted_handle"]
    found = False
    for jp in session_input_dir.glob("*.json"):
        data = json.loads(jp.read_text(encoding="utf-8"))
        if data.get("clause_id") == planted_handle:
            found = True
            assert plan.planted_phrase in data.get("clause_content", "")
            break
    assert found, f"no per-turn json carries clause_id={planted_handle}"
