from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david.cli import main as david_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tiny_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "tiny-david-workspace"
    package = workspace / "src"
    tests = workspace / "tests"
    package.mkdir(parents=True)
    tests.mkdir(parents=True)
    (package / "app.py").write_text("def hello():\n    return 'hello'\n", encoding="utf-8")
    (tests / "test_app.py").write_text(
        "from src.app import hello\n\n"
        "def test_hello():\n"
        "    assert hello() == 'hello'\n",
        encoding="utf-8",
    )
    return workspace


def _assert_not_project_root(workspace: Path) -> None:
    assert workspace.resolve() != PROJECT_ROOT.resolve()
    assert PROJECT_ROOT.resolve() not in workspace.resolve().parents


def test_status_once_reports_startup_readiness_for_tiny_workspace(tmp_path: Path, capsys) -> None:
    workspace = _tiny_workspace(tmp_path)
    _assert_not_project_root(workspace)

    rc = david_main(["code", str(workspace), "--allow-unvalidated", "--once", "/status", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David terminal agent" in output
    assert "David startup readiness" in output
    assert f"workspace: {workspace.resolve()}" in output
    assert "model validation:" in output
    assert "index:" in output
    assert "memory:" in output
    assert not (workspace / ".david" / "indexes").exists()


def test_agent_create_file_plain_english_writes_and_reports_verified_ok(tmp_path: Path, capsys) -> None:
    workspace = _tiny_workspace(tmp_path)
    _assert_not_project_root(workspace)

    rc = david_main(
        [
            "code",
            str(workspace),
            "--allow-unvalidated",
            "--once",
            "/agent create a file named note.txt that says hello tiny workspace",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=2 verified=True" in output
    assert "- 1: write ok=True path=note.txt" in output
    assert "- 2: verify ok=True rc=0" in output
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "hello tiny workspace"
    assert (workspace / ".david" / "memory" / "task-default.jsonl").exists()


def test_memory_and_index_are_separate_status_surfaces(tmp_path: Path, capsys) -> None:
    workspace = _tiny_workspace(tmp_path)
    _assert_not_project_root(workspace)

    memory_rc = david_main(["code", str(workspace), "--allow-unvalidated", "--once", "/memory", "--no-color"])
    memory_output = capsys.readouterr().out
    index_rc = david_main(["code", str(workspace), "--allow-unvalidated", "--once", "/index", "--no-color"])
    index_output = capsys.readouterr().out

    assert memory_rc == 0
    assert index_rc == 0
    assert "memory:" in memory_output
    assert "user artifact:" in memory_output
    assert "task artifact:" in memory_output
    memory_detail = memory_output.split("\nmemory:\n", maxsplit=1)[1]
    assert "source index:" not in memory_detail
    assert "index:" in index_output
    assert "source index:" in index_output
    assert "user artifact:" not in index_output
    assert "task artifact:" not in index_output
    assert not (workspace / ".david" / "indexes").exists()


def test_index_jit_indexes_only_the_tiny_workspace(tmp_path: Path, capsys) -> None:
    workspace = _tiny_workspace(tmp_path)
    _assert_not_project_root(workspace)

    rc = david_main(["code", str(workspace), "--allow-unvalidated", "--once", "/index jit", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    index_dir = workspace / ".david" / "indexes"
    assert "index: ready" in output
    assert "source index:" in output
    assert str(index_dir) in output
    assert index_dir.exists()
    assert all(path.resolve().is_relative_to(workspace.resolve()) for path in index_dir.rglob("*"))
