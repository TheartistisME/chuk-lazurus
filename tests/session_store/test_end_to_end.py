"""End-to-end CUDA smoke test for axis-3 session_store.

Runs the real build script against a tiny synthetic session. Skipped when
torch or CUDA is unavailable, which is expected on most CI runners.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]

from chuk_lazarus.session_store import (
    invoke_build,
    verify_checkpoint,
    verify_metadata_preservation,
)


pytestmark = [
    pytest.mark.skipif(
        not _HAS_TORCH or (torch is not None and not torch.cuda.is_available()),
        reason="CUDA+torch required for end-to-end smoke",
    ),
    pytest.mark.slow,
]


def _write_turn_json(
    input_dir: Path,
    *,
    filename: str,
    clause_id: str,
    clause_title: str,
    clause_content: str,
    session_uuid: str,
) -> None:
    payload = {
        "standard_id": "smoke-standard",
        "standard_title": "Smoke Standard",
        "clause_id": clause_id,
        "clause_title": clause_title,
        "clause_content": clause_content,
        # Secondary metadata: all 4 keys so verify_metadata_preservation passes.
        "iso_timestamp": "2026-04-21T00:00:00Z",
        "speaker_role": "user",
        "session_uuid": session_uuid,
        "topic_tags": ["smoke", "test"],
    }
    (input_dir / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_end_to_end_tiny_session(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    checkpoint_root = tmp_path / "out"

    zero_id = "0" * 32
    twenty_words = " ".join(["word"] * 20)

    _write_turn_json(
        input_dir,
        filename="001_0_0.json",
        clause_id=f"{zero_id}.0.0",
        clause_title="Turn One",
        clause_content=twenty_words,
        session_uuid="smoke-uuid-1",
    )
    _write_turn_json(
        input_dir,
        filename="002_0_1.json",
        clause_id=f"{zero_id}.0.1",
        clause_title="Turn Two",
        clause_content=twenty_words,
        session_uuid="smoke-uuid-1",
    )

    result = invoke_build(
        session_id="smoke",
        input_dir=input_dir,
        checkpoint_root=checkpoint_root,
        device="cuda",
    )
    assert result.returncode == 0, (
        f"build script failed (rc={result.returncode}):\n{result.stderr}"
    )

    ck_result = verify_checkpoint(result.checkpoint)
    assert ck_result["ok"] is True

    meta_result = verify_metadata_preservation(input_dir)
    assert meta_result["ok"] is True
