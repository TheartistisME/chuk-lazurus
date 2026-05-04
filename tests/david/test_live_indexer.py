from __future__ import annotations

import os
from pathlib import Path

from chuk_lazarus.david.live_indexer import LiveIndexer, load_live_index_state
from chuk_lazarus.david.source_index import load_source_index


def test_live_indexer_initial_refresh_writes_source_index_and_handle(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("import json\n\ndef boot_agent():\n    return json.dumps({})\n", encoding="utf-8")
    indexer = LiveIndexer(tmp_path, {"model_id": "offline"}, max_files=10)

    refresh = indexer.refresh()

    assert refresh.changed_paths == ["src/agent.py"]
    assert refresh.indexed_paths == ["src/agent.py"]
    assert refresh.deleted_paths == []
    assert refresh.file_count == 1
    assert len(refresh.manifest_sha256) == 64
    assert refresh.to_session_refresh_handle() == {
        "kind": "david_live_source_index",
        "schema": "david.live_index_refresh.v1",
        "workspace_root": str(tmp_path.resolve()),
        "source_index_path": str(tmp_path.resolve() / ".david" / "indexes" / "live-source.json"),
        "changed_paths": ["src/agent.py"],
        "deleted_paths": [],
        "changed_count": 1,
        "indexed_count": 1,
        "deleted_count": 0,
        "file_count": 1,
        "manifest_sha256": refresh.manifest_sha256,
        "adapter_scope": {"model_id": "offline"},
    }

    manifest = load_source_index(Path(refresh.source_index_path))
    assert manifest.files[0].path == "src/agent.py"
    assert manifest.files[0].symbols == ["boot_agent"]
    state = load_live_index_state(Path(refresh.state_path))
    assert state.files["src/agent.py"].sha256 == manifest.files[0].sha256


def test_live_indexer_second_refresh_reuses_unchanged_file_state(tmp_path: Path) -> None:
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def boot_agent():\n    return 1\n", encoding="utf-8")
    indexer = LiveIndexer(tmp_path, {}, max_files=10)

    first = indexer.refresh()
    second = indexer.refresh()

    assert first.changed_paths == ["src/agent.py"]
    assert second.changed_paths == []
    assert second.indexed_paths == []
    assert second.unchanged_paths == ["src/agent.py"]
    assert second.manifest_sha256 != ""


def test_live_indexer_updates_changed_added_and_deleted_files(tmp_path: Path) -> None:
    keep = tmp_path / "src" / "keep.py"
    remove = tmp_path / "src" / "remove.py"
    keep.parent.mkdir()
    keep.write_text("def keep():\n    return 1\n", encoding="utf-8")
    remove.write_text("def remove():\n    return 1\n", encoding="utf-8")
    indexer = LiveIndexer(tmp_path, {}, max_files=10)
    indexer.refresh()

    _rewrite_with_new_mtime(keep, "def keep():\n    return 2\n")
    remove.unlink()
    added = tmp_path / "src" / "added.py"
    added.write_text("def added():\n    return 3\n", encoding="utf-8")

    refresh = indexer.refresh()

    assert refresh.changed_paths == ["src/added.py", "src/keep.py"]
    assert refresh.indexed_paths == ["src/added.py", "src/keep.py"]
    assert refresh.deleted_paths == ["src/remove.py"]
    manifest = load_source_index(Path(refresh.source_index_path))
    assert [record.path for record in manifest.files] == ["src/added.py", "src/keep.py"]


def test_live_indexer_bounds_workspace_and_skips_unsafe_files(tmp_path: Path) -> None:
    protected = tmp_path / "scripts" / "run_swebench_pro_parity.py"
    protected.parent.mkdir()
    protected.write_text("def proof_rig(): pass\n", encoding="utf-8")
    (tmp_path / "large.py").write_text("x" * 50, encoding="utf-8")
    (tmp_path / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b(): pass\n", encoding="utf-8")
    indexer = LiveIndexer(tmp_path, {}, max_files=1, max_file_bytes=20)

    refresh = indexer.refresh()

    assert refresh.truncated is True
    assert refresh.changed_paths == ["a.py"]
    assert refresh.skipped_paths == []
    manifest = load_source_index(Path(refresh.source_index_path))
    assert [record.path for record in manifest.files] == ["a.py"]


def _rewrite_with_new_mtime(path: Path, text: str) -> None:
    before = path.stat().st_mtime_ns
    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(before + 1_000_000_000, before + 1_000_000_000))
