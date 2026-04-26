"""Tests for :mod:`chuk_lazarus.session_retrieval.enumeration`.

Exercises the validity gate, silent-skip behaviour, and
``load_original_turn_records`` helper without requiring CUDA or any real
HuggingFace model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chuk_lazarus.session_retrieval.enumeration import (
    CheckpointHandle,
    iter_checkpoint_handles,
    load_original_turn_records,
)


def test_empty_checkpoint_root_yields_nothing(tmp_path: Path) -> None:
    """An empty directory produces an empty handle list."""
    root = tmp_path / "checkpoint_root"
    root.mkdir()
    assert list(iter_checkpoint_handles(root)) == []


def test_nonexistent_checkpoint_root_yields_nothing(tmp_path: Path) -> None:
    """A non-existent root is treated as an empty enumerator."""
    root = tmp_path / "does_not_exist"
    assert list(iter_checkpoint_handles(root)) == []


def test_two_valid_sessions_are_sorted(two_sessions_checkpoint_root: dict[str, Path]) -> None:
    """Two valid checkpoints are yielded in sorted session_id order."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    assert len(handles) == 2
    assert handles[0].session_id == two_sessions_checkpoint_root["session_a"]
    assert handles[1].session_id == two_sessions_checkpoint_root["session_b"]
    for handle in handles:
        assert isinstance(handle, CheckpointHandle)
        assert handle.checkpoint_dir.is_dir()
        assert handle.torch_store_dir.is_dir()
        assert isinstance(handle.manifest, dict)


def test_child_without_torch_store_is_silently_skipped(
    tmp_path: Path, two_sessions_checkpoint_root: dict[str, Path]
) -> None:
    """A child directory missing ``torch_store/manifest.json`` is skipped."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    # Create a bare child with no torch_store subdirectory.
    (root / ("c" * 32)).mkdir()
    handles = list(iter_checkpoint_handles(root))
    session_ids = {h.session_id for h in handles}
    assert ("c" * 32) not in session_ids
    assert len(handles) == 2


def test_in_flight_manifest_is_silently_skipped(
    tmp_path: Path, two_sessions_checkpoint_root: dict[str, Path]
) -> None:
    """A live-indexer in-flight manifest is not a retrieval-ready checkpoint."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    session_id = "d" * 32
    torch_store = root / session_id / "torch_store"
    torch_store.mkdir(parents=True)
    (torch_store / "manifest.json").write_text(
        (
            "{\n"
            '  "clause_aligned": true,\n'
            '  "num_windows": 1,\n'
            '  "num_entries": 1,\n'
            '  "num_tokens": 0,\n'
            '  "in_flight": true\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    handles = list(iter_checkpoint_handles(root))
    session_ids = {handle.session_id for handle in handles}

    assert session_id not in session_ids
    assert len(handles) == 2


def test_invalid_clause_aligned_false_raises(
    tmp_path: Path, fake_session_builder: Any
) -> None:
    """``clause_aligned: false`` in manifest raises RuntimeError naming session."""
    root = tmp_path / "checkpoint_root"
    root.mkdir()
    session_id = "d" * 32
    fake_session_builder(
        root,
        session_id,
        windows=[{"token_ids": [1, 2], "keywords": ["x"], "clause_id": "9.0.0"}],
        clause_aligned=False,
    )
    with pytest.raises(RuntimeError) as excinfo:
        list(iter_checkpoint_handles(root))
    assert session_id in str(excinfo.value)
    assert "clause_aligned" in str(excinfo.value)


def test_invalid_num_windows_zero_raises(
    tmp_path: Path, fake_session_builder: Any
) -> None:
    """``num_windows < 1`` raises RuntimeError."""
    root = tmp_path / "checkpoint_root"
    root.mkdir()
    session_id = "e" * 32
    fake_session_builder(
        root,
        session_id,
        windows=[{"token_ids": [1], "keywords": ["x"], "clause_id": "9.0.0"}],
        override_num_windows=0,
    )
    with pytest.raises(RuntimeError) as excinfo:
        list(iter_checkpoint_handles(root))
    assert session_id in str(excinfo.value)
    assert "num_windows" in str(excinfo.value)


def test_invalid_num_entries_zero_raises(
    tmp_path: Path, fake_session_builder: Any
) -> None:
    """``num_entries < 1`` raises RuntimeError."""
    root = tmp_path / "checkpoint_root"
    root.mkdir()
    session_id = "f" * 32
    fake_session_builder(
        root,
        session_id,
        windows=[{"token_ids": [1], "keywords": ["x"], "clause_id": "9.0.0"}],
        override_num_entries=0,
    )
    with pytest.raises(RuntimeError) as excinfo:
        list(iter_checkpoint_handles(root))
    assert session_id in str(excinfo.value)
    assert "num_entries" in str(excinfo.value)


def test_original_input_dir_is_set_per_handle(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """When ``original_input_root`` is provided, handles carry per-session subdir."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    original_root = two_sessions_checkpoint_root["original_input_root"]
    handles = list(iter_checkpoint_handles(root, original_input_root=original_root))
    for handle in handles:
        assert handle.original_input_dir is not None
        assert handle.original_input_dir == original_root / handle.session_id


def test_original_input_dir_is_none_when_root_not_supplied(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Without ``original_input_root`` the handle's ``original_input_dir`` is None."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    for handle in handles:
        assert handle.original_input_dir is None


def test_load_original_turn_records_none_dir_returns_empty(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """``load_original_turn_records`` returns [] when ``original_input_dir`` is None."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(iter_checkpoint_handles(root))
    assert handles[0].original_input_dir is None
    assert load_original_turn_records(handles[0]) == []


def test_load_original_turn_records_missing_dir_returns_empty(
    tmp_path: Path, two_sessions_checkpoint_root: dict[str, Path]
) -> None:
    """Non-existent ``original_input_dir`` returns []."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    handles = list(
        iter_checkpoint_handles(
            root, original_input_root=tmp_path / "nowhere"
        )
    )
    assert handles[0].original_input_dir is not None
    assert not handles[0].original_input_dir.exists()
    assert load_original_turn_records(handles[0]) == []


def test_load_original_turn_records_reads_sorted(
    two_sessions_checkpoint_root: dict[str, Path],
) -> None:
    """Real records round-trip as sorted list of dicts with expected keys."""
    root = two_sessions_checkpoint_root["checkpoint_root"]
    original_root = two_sessions_checkpoint_root["original_input_root"]
    handles = list(iter_checkpoint_handles(root, original_input_root=original_root))
    session_a_handle = handles[0]
    records = load_original_turn_records(session_a_handle)
    assert len(records) == 2
    # Sorted filename => record for clause 1.0.0 first, then 1.0.1.
    assert records[0]["clause_id"] == "1.0.0"
    assert records[1]["clause_id"] == "1.0.1"
    for record in records:
        for required_key in ("clause_id", "iso_timestamp", "topic_tags"):
            assert required_key in record
