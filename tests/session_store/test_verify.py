"""Unit tests for ``chuk_lazarus.session_store.verify``.

Fixtures are built inline in ``tmp_path`` so each test is fully isolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.session_store.verify import (
    verify_checkpoint,
    verify_metadata_preservation,
)


def _write_checkpoint(
    tmp_path: Path,
    *,
    clause_aligned: bool = True,
    num_windows: int = 3,
    num_entries: int = 7,
    num_tokens: int = 256,
    build_num_windows: int | None = None,
    build_num_entries: int | None = None,
) -> Path:
    """Create a checkpoint directory with healthy (or controllably broken) manifests."""
    checkpoint = tmp_path / "sess-1"
    torch_store = checkpoint / "torch_store"
    torch_store.mkdir(parents=True, exist_ok=True)

    torch_manifest = {
        "clause_aligned": clause_aligned,
        "num_windows": num_windows,
        "num_entries": num_entries,
        "num_tokens": num_tokens,
    }
    (torch_store / "manifest.json").write_text(
        json.dumps(torch_manifest, indent=2), encoding="utf-8"
    )

    build_manifest = {
        "num_windows": num_windows if build_num_windows is None else build_num_windows,
        "num_entries": num_entries if build_num_entries is None else build_num_entries,
    }
    (checkpoint / "clause_aligned_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2), encoding="utf-8"
    )
    return checkpoint


def test_verify_checkpoint_happy_path(tmp_path: Path) -> None:
    ck = _write_checkpoint(tmp_path)
    result = verify_checkpoint(ck)
    assert result["ok"] is True
    assert result["num_windows"] == 3
    assert result["num_entries"] == 7
    assert result["num_tokens"] == 256


def test_verify_checkpoint_rejects_clause_aligned_false(tmp_path: Path) -> None:
    ck = _write_checkpoint(tmp_path, clause_aligned=False)
    with pytest.raises(AssertionError) as exc_info:
        verify_checkpoint(ck)
    assert "clause_aligned" in str(exc_info.value)


def test_verify_checkpoint_rejects_zero_windows(tmp_path: Path) -> None:
    ck = _write_checkpoint(tmp_path, num_windows=0)
    with pytest.raises(AssertionError) as exc_info:
        verify_checkpoint(ck)
    assert "num_windows" in str(exc_info.value)


def test_verify_checkpoint_rejects_zero_entries(tmp_path: Path) -> None:
    ck = _write_checkpoint(tmp_path, num_entries=0)
    with pytest.raises(AssertionError) as exc_info:
        verify_checkpoint(ck)
    assert "num_entries" in str(exc_info.value)


def test_verify_checkpoint_rejects_mismatched_counts(tmp_path: Path) -> None:
    # torch_store says 3 windows; build manifest says 5.
    ck = _write_checkpoint(tmp_path, num_windows=3, build_num_windows=5)
    with pytest.raises(AssertionError) as exc_info:
        verify_checkpoint(ck)
    msg = str(exc_info.value)
    assert "num_windows" in msg
    assert "3" in msg and "5" in msg


def test_verify_checkpoint_missing_manifest_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty-checkpoint"
    empty.mkdir()
    with pytest.raises(AssertionError) as exc_info:
        verify_checkpoint(empty)
    # Either the torch_store manifest or build manifest path should be mentioned.
    msg = str(exc_info.value)
    assert "manifest" in msg.lower() or str(empty) in msg


def test_verify_metadata_preservation_happy_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Spread the 4 secondary keys across 3 files.
    (input_dir / "001_0_0.json").write_text(
        json.dumps({"iso_timestamp": "2026-04-21T00:00:00Z", "speaker_role": "user"}),
        encoding="utf-8",
    )
    (input_dir / "002_0_1.json").write_text(
        json.dumps({"session_uuid": "uuid-xyz"}),
        encoding="utf-8",
    )
    (input_dir / "003_0_2.json").write_text(
        json.dumps({"topic_tags": ["a", "b"]}),
        encoding="utf-8",
    )
    result = verify_metadata_preservation(input_dir)
    assert result["ok"] is True
    assert result["files_checked"] == 3
    assert result["keys_seen"] == sorted(
        ["iso_timestamp", "speaker_role", "session_uuid", "topic_tags"]
    )


def test_verify_metadata_preservation_missing_key_raises(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    # Never mention topic_tags.
    (input_dir / "001_0_0.json").write_text(
        json.dumps(
            {
                "iso_timestamp": "2026-04-21T00:00:00Z",
                "speaker_role": "user",
                "session_uuid": "uuid-xyz",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError) as exc_info:
        verify_metadata_preservation(input_dir)
    assert "topic_tags" in str(exc_info.value)


def test_verify_metadata_preservation_empty_dir_raises(tmp_path: Path) -> None:
    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    with pytest.raises(AssertionError):
        verify_metadata_preservation(input_dir)
