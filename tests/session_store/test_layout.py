"""Unit tests for ``chuk_lazarus.session_store.layout``.

These tests pin the on-disk path conventions that axis-4 (retrieval) relies
on. Any drift here is a contract break.
"""

from __future__ import annotations

from pathlib import Path

from chuk_lazarus.session_store.layout import (
    build_manifest_path,
    log_path,
    manifest_path,
    per_session_checkpoint,
    torch_store_path,
)


def test_per_session_checkpoint_joins_root_and_id(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    result = per_session_checkpoint(root, "abc")
    assert result == root / "abc"


def test_torch_store_path_under_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sess-1"
    result = torch_store_path(checkpoint)
    assert result.name == "torch_store"
    assert result.parent == checkpoint


def test_manifest_path_is_torch_store_manifest_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sess-1"
    result = manifest_path(checkpoint)
    assert result.name == "manifest.json"
    assert result.parent.name == "torch_store"


def test_build_manifest_path_is_clause_aligned_build_manifest_json(
    tmp_path: Path,
) -> None:
    # This MUST be the exact literal emitted by tools/build_clause_aligned_store.py.
    # Some docs referred to ``build_manifest.json``; the script itself writes
    # ``clause_aligned_build_manifest.json``. We match the script.
    checkpoint = tmp_path / "sess-1"
    result = build_manifest_path(checkpoint)
    assert result.name == "clause_aligned_build_manifest.json"
    assert result.parent == checkpoint


def test_log_path_inside_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "sess-1"
    result = log_path(checkpoint)
    assert result.name == "invoke.log"
    assert result.parent == checkpoint
