from __future__ import annotations

import json
import sys
from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.cli import main as david_main


def _tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "tiny-repo"
    source = repo / "src" / "session.py"
    test = repo / "tests" / "test_session.py"
    proof = repo / "scripts" / "run_swebench_pro_parity.py"
    central_router = repo / "David" / "central router.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    proof.parent.mkdir(parents=True)
    central_router.parent.mkdir(parents=True)
    source.write_text("def session_cleanup(user_id):\n    return None\n", encoding="utf-8")
    test.write_text(
        "from src.session import session_cleanup\n\n"
        "def test_session_cleanup():\n"
        "    assert session_cleanup('david') is None\n",
        encoding="utf-8",
    )
    proof.write_text("# protected SWE proof rig\n", encoding="utf-8")
    central_router.write_text("# protected router proof rig\n", encoding="utf-8")
    return repo


def _runtime(repo: Path) -> DavidRuntime:
    return DavidRuntime.create(DavidConfig(workspace_root=repo, state_dir=repo / ".david"))


def test_david_code_once_status_uses_cli_main_on_tiny_repo(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(["code", str(repo), "--once", "/status", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David terminal agent" in output
    assert "David startup readiness" in output
    assert f"workspace: {repo.resolve()}" in output
    assert "index: missing" in output


def test_index_jit_command_builds_tiny_repo_index_via_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(["code", str(repo), "--once", "/index jit", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "index: ready" in output
    assert "source index:" in output
    assert (repo / ".david" / "indexes").exists()


def test_repo_patch_prompt_routes_to_patch_target_and_selected_file(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path)
    runtime = _runtime(repo)

    result = runtime.run_once("Fix the repo bug by patching session cleanup in src/session.py")

    assert result.method == "repo_patch"
    assert result.product_route is not None
    assert result.product_route.methodology == "patch_target"
    assert result.product_route.capability == "repo patch-target routing"
    assert result.product_route.selected_paths[0] == "src/session.py"
    assert "tests/test_session.py" in result.product_route.selected_tests


def test_verify_command_passes_from_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    command = f'"{sys.executable}" -c "print(\'verify-ok\')"'

    rc = david_main(
        [
            "code",
            str(repo),
            "--once",
            "/verify",
            "--verify-command",
            command,
            "--allow-unvalidated",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "rc=0" in output
    assert "verify-ok" in output


def test_user_memory_prompt_writes_user_artifact(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path)
    runtime = _runtime(repo)

    result = runtime.run_once("Remember my preference: I prefer concise David status summaries")

    user_memory = repo / ".david" / "memory" / "user-default.jsonl"
    assert result.method == "user_continuity"
    assert result.writeback["family"] == "user"
    assert user_memory.exists()
    records = [json.loads(line) for line in user_memory.read_text(encoding="utf-8").splitlines()]
    assert records[-1]["family"] == "user"
    assert records[-1]["kind"] == "user_continuity"
    assert "concise David status summaries" in records[-1]["text"]


def test_resume_command_shows_prior_summary_from_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    runtime = _runtime(repo)
    runtime.run_once("Remember my preference: keep David resume summaries short")

    rc = david_main(["code", str(repo), "--once", "/resume", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David resume" in output
    assert "session: default" in output
    assert "last result:" in output
    assert "keep David resume summaries short" in output


def test_protected_proof_rig_files_are_not_selected_for_patch_target(tmp_path: Path) -> None:
    repo = _tiny_repo(tmp_path)
    runtime = _runtime(repo)

    result = runtime.run_once("Fix the repo bug; do not patch proof rigs, patch session cleanup")

    assert result.product_route is not None
    selected = set(result.product_route.selected_paths)
    assert "src/session.py" in selected
    assert "scripts/run_swebench_pro_parity.py" not in selected
    assert "David/central router.py" not in selected
