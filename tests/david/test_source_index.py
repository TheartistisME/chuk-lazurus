from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david.source_index import build_source_index, load_source_index, save_source_index


def test_source_index_records_symbols_imports_and_adapter_scope(tmp_path: Path) -> None:
    source = tmp_path / "src" / "service.py"
    source.parent.mkdir()
    source.write_text(
        "import json\nfrom pathlib import Path\n\nclass Worker:\n    pass\n\ndef run_task():\n    return Path\n",
        encoding="utf-8",
    )

    manifest = build_source_index(tmp_path, {"model_id": "offline"}, max_files=10)

    assert manifest.adapter_scope == {"model_id": "offline"}
    assert len(manifest.files) == 1
    record = manifest.files[0]
    assert record.path == "src/service.py"
    assert record.language == "python"
    assert record.symbols == ["Worker", "run_task"]
    assert "json" in record.import_tokens
    assert "pathlib" in record.import_tokens


def test_source_index_prunes_dirs_skips_large_files_and_truncates(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.py").write_text("def ignored(): pass\n", encoding="utf-8")
    (tmp_path / "large.py").write_text("x" * 50, encoding="utf-8")
    (tmp_path / "a.py").write_text("def a(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def b(): pass\n", encoding="utf-8")

    manifest = build_source_index(tmp_path, {}, max_files=1, max_file_bytes=20)

    assert manifest.truncated is True
    assert len(manifest.files) == 1
    assert manifest.files[0].path == "a.py"
    assert all(record.path != "node_modules/ignored.py" for record in manifest.files)
    assert all(record.path != "large.py" for record in manifest.files)
    assert "node_modules" in manifest.pruned_dirs


def test_source_index_excludes_protected_proof_rig_files(tmp_path: Path) -> None:
    protected = tmp_path / "scripts" / "run_swebench_pro_parity.py"
    protected.parent.mkdir()
    protected.write_text("def proof_rig(): pass\n", encoding="utf-8")
    source = tmp_path / "src" / "agent.py"
    source.parent.mkdir()
    source.write_text("def boot_agent(): pass\n", encoding="utf-8")

    manifest = build_source_index(tmp_path, {}, max_files=10)

    paths = {record.path for record in manifest.files}
    assert "src/agent.py" in paths
    assert "scripts/run_swebench_pro_parity.py" not in paths


def test_source_index_round_trips_manifest(tmp_path: Path) -> None:
    (tmp_path / "main.rs").write_text("use std::path::Path;\npub fn main_task() {}\n", encoding="utf-8")
    manifest = build_source_index(tmp_path, {"adapter_family": "offline"})
    path = tmp_path / ".david" / "source-index.json"

    save_source_index(path, manifest)
    loaded = load_source_index(path)

    assert loaded.to_json() == manifest.to_json()
    assert loaded.files[0].symbols == ["main_task"]
    assert "std::path::Path" in loaded.files[0].import_tokens
