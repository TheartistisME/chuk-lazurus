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


def test_david_init_creates_workspace_state_without_indexing_or_model_load(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(["init", str(repo), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David init" in output
    assert f"workspace: {repo.resolve()}" in output
    assert "created paths:" in output
    assert "David terminal agent" not in output
    assert (repo / ".david" / "config.json").exists()
    assert (repo / ".david" / "NEXT_STEPS.md").exists()
    assert (repo / ".david" / "memory").is_dir()
    assert (repo / ".david" / "indexes").is_dir()
    assert (repo / ".david" / "model_validation").is_dir()
    assert (repo / ".david" / "sessions").is_dir()
    assert not any((repo / ".david" / "indexes").iterdir())


def test_verify_discovery_and_patch_mode_do_not_run_candidate_gates(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    marker = repo / "candidate-ran.txt"
    (repo / "tests" / "test_candidate_should_not_run.py").write_text(
        "from pathlib import Path\n"
        "Path('candidate-ran.txt').write_text('ran', encoding='utf-8')\n"
        "def test_candidate_should_not_run():\n"
        "    assert False\n",
        encoding="utf-8",
    )

    rc = david_main(["verify", str(repo), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    discovery_output = capsys.readouterr().out
    assert "David verification" in discovery_output
    assert "mode: candidate discovery" in discovery_output
    assert "selected commands:\n- none" in discovery_output
    assert "discovered candidate gates:" in discovery_output
    assert "python -m pytest -q" in discovery_output
    assert "tests/ directory found" in discovery_output
    assert "commands run:\n- none" in discovery_output
    assert "candidate gates were discovered but not executed without --cmd" in discovery_output
    assert "- capability: verify" in discovery_output
    assert "- candidate_count: 1" in discovery_output
    assert "- selected_count: 0" in discovery_output
    assert "- command_count: 0" in discovery_output
    assert not marker.exists()

    rc = david_main(["verify", str(repo), "--patch", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    patch_output = capsys.readouterr().out
    assert "David verification" in patch_output
    assert "mode: built-in patch verification" in patch_output
    assert "selected commands:" in patch_output
    assert "discovered candidate gates:" in patch_output
    assert "python -m pytest -q" in patch_output
    assert "tests/ directory found" in patch_output
    assert "commands run:\n- none" in patch_output
    assert "selected candidate gates but did not execute them without --cmd" in patch_output
    assert "- capability: repo_patch" in patch_output
    assert "- candidate_count: 1" in patch_output
    assert "- selected_count: 1" in patch_output
    assert "- command_count: 0" in patch_output
    assert not marker.exists()


def test_resume_after_code_once_prompt_reads_saved_session(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(
        [
            "code",
            str(repo),
            "--once",
            "Remember my preference: resume after code once smoke",
            "--allow-unvalidated",
            "--no-color",
        ]
    )

    assert rc == 0
    once_output = capsys.readouterr().out
    assert "David terminal agent" in once_output
    assert "David startup readiness" in once_output
    assert "method=user_continuity" in once_output
    assert "resume after code once smoke" in once_output
    assert (repo / ".david" / "resume.json").exists()
    assert (repo / ".david" / "memory" / "user-default.jsonl").exists()

    rc = david_main(["resume", str(repo), "--allow-unvalidated", "--no-color"])

    assert rc == 0
    resume_output = capsys.readouterr().out
    assert "David resume" in resume_output
    assert "session: default" in resume_output
    assert "last result:" in resume_output
    assert "resume after code once smoke" in resume_output


def test_agent_loop_repairs_after_failed_gate_and_records_summary(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    payload = json.dumps(
        [
            {
                "action": "run",
                "command": [sys.executable, "-c", "raise SystemExit(7)"],
                "reason": "simulate a failed gate before repair",
            },
            {"action": "write", "path": "repair.txt", "content": "repaired\n"},
            {
                "action": "verify",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "raise SystemExit(0 if Path('repair.txt').read_text(encoding='utf-8') == 'repaired\\n' else 1)"
                    ),
                ],
            },
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=3 verified=True" in output
    assert "- 1: run ok=False rc=7" in output
    assert "- 2: write ok=True path=repair.txt bytes=9" in output
    assert "- 3: verify ok=True rc=0" in output
    assert (repo / "repair.txt").read_text(encoding="utf-8") == "repaired\n"
    task_records = [
        json.loads(line)
        for line in (repo / ".david" / "memory" / "task-default.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    repair_summary = task_records[-1]["metadata"]["repair_summary"]
    assert repair_summary["failure_count"] == 1
    assert repair_summary["recovered"] is True
    assert repair_summary["recovery_outcome"] == "verified_after_failure"
    assert repair_summary["verification"]["outcome"] == "passed"


def test_david_code_once_status_uses_cli_main_on_tiny_repo(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(["code", str(repo), "--once", "/status", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David terminal agent" in output
    assert "David startup readiness" in output
    assert f"workspace: {repo.resolve()}" in output
    assert "index: missing" in output


def test_david_bare_once_status_uses_current_workspace(tmp_path: Path, capsys, monkeypatch) -> None:
    repo = _tiny_repo(tmp_path)
    monkeypatch.chdir(repo)

    rc = david_main(["--once", "/status", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "David terminal agent" in output
    assert "David startup readiness" in output
    assert f"workspace: {repo.resolve()}" in output


def test_direct_product_commands_use_tiny_workspace(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    runtime = _runtime(repo)
    runtime.run_once("Remember my preference: direct command smoke")
    command = f'"{sys.executable}" -c "print(\'direct-verify-ok\')"'

    assert david_main(["memory", str(repo), "--allow-unvalidated", "--no-color"]) == 0
    memory_output = capsys.readouterr().out
    assert "memory:" in memory_output
    assert "user-default.jsonl" in memory_output
    assert "task-default.jsonl" in memory_output

    assert david_main(["resume", str(repo), "--allow-unvalidated", "--no-color"]) == 0
    resume_output = capsys.readouterr().out
    assert "David resume" in resume_output
    assert "direct command smoke" in resume_output

    assert david_main(["index", str(repo), "--build", "--allow-unvalidated", "--no-color"]) == 0
    index_output = capsys.readouterr().out
    assert "index: ready" in index_output
    assert "source index:" in index_output
    assert (repo / ".david" / "indexes").exists()

    assert david_main(["verify", str(repo), "--cmd", command, "--allow-unvalidated", "--no-color"]) == 0
    verify_output = capsys.readouterr().out
    assert "rc=0" in verify_output
    assert "direct-verify-ok" in verify_output

    assert david_main(["capabilities"]) == 0
    capabilities_output = capsys.readouterr().out
    assert "David capabilities" in capabilities_output
    assert "WIRED:" in capabilities_output
    assert "quality gate discovery surface" in capabilities_output
    assert "quality gate execution requires explicit --cmd" in capabilities_output


def test_index_jit_command_builds_tiny_repo_index_via_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(["code", str(repo), "--once", "/index jit", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "index: ready" in output
    assert "source index:" in output
    assert (repo / ".david" / "indexes").exists()


def test_agent_loop_command_writes_file_via_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    payload = json.dumps(
        [
            {"action": "write", "path": "src/agent_loop_note.txt", "content": "loop wrote this\n"},
            {"action": "verify", "passed": True, "reason": "smoke pass"},
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=2 verified=True" in output
    assert "path=src/agent_loop_note.txt bytes=16" in output
    assert (repo / "src" / "agent_loop_note.txt").read_text(encoding="utf-8") == "loop wrote this\n"
    assert (repo / ".david" / "memory" / "task-default.jsonl").exists()


def test_agent_loop_strict_patch_then_verify_via_cli_main(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    payload = json.dumps(
        [
            {
                "action": "patch",
                "content": """src/session.py
<<<< SEARCH
def session_cleanup(user_id):
    return None
==== REPLACE
def session_cleanup(user_id):
    return True
>>>>
""",
            },
            {
                "action": "verify",
                "command": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "text = Path('src/session.py').read_text(encoding='utf-8'); "
                        "assert 'return True' in text"
                    ),
                ],
            },
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=2 verified=True" in output
    assert "- 1: patch ok=True" in output
    assert "- 2: verify ok=True rc=0" in output
    assert "return True" in (repo / "src" / "session.py").read_text(encoding="utf-8")
    assert (repo / ".david" / "memory" / "task-default.jsonl").exists()


def test_agent_loop_plain_english_writes_file_without_model_json(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(
        [
            "code",
            str(repo),
            "--once",
            "/agent create a file named hello.txt that says hello",
            "--allow-unvalidated",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=2 verified=True" in output
    assert "path=hello.txt bytes=5" in output
    assert "- 2: verify ok=True rc=0" in output
    assert (repo / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert (repo / ".david" / "memory" / "task-default.jsonl").exists()


def test_agent_loop_codex_patch_failure_prints_patch_diagnostics(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    payload = json.dumps(
        [
            {
                "action": "patch",
                "content": """*** Begin Patch
*** Update File: src/session.py
@@
-def session_cleanup(user_id):
+def session_cleanup(user_id):
*** End Patch
""",
            }
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: refused steps=1 verified=False" in output
    assert "- 1: patch ok=False mode=unified_diff" in output
    assert "failures=unified diff contains no file hunks" in output


def test_agent_loop_protected_patch_failure_prints_patch_diagnostics(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    protected_path = "scripts/run_swebench_pro_parity.py"
    payload = json.dumps(
        [
            {
                "action": "patch",
                "content": f"""{protected_path}
<<<< SEARCH
# protected SWE proof rig
==== REPLACE
# changed
>>>>
""",
            }
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: refused steps=1 verified=False" in output
    assert "- 1: patch ok=False mode=strict_search_replace" in output
    assert f"protected={protected_path}" in output
    assert f"failures=block 1: protected proof-rig path: {protected_path}" in output
    assert (repo / protected_path).read_text(encoding="utf-8") == "# protected SWE proof rig\n"


def test_agent_loop_protected_write_failure_keeps_proof_rig_unchanged(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)
    protected_path = "scripts/run_swebench_pro_parity.py"
    payload = json.dumps(
        [
            {
                "action": "write",
                "path": protected_path,
                "content": "# changed\n",
            }
        ]
    )

    rc = david_main(["code", str(repo), "--once", f"/agent {payload}", "--allow-unvalidated", "--no-color"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: refused steps=1 verified=False" in output
    assert "- 1: refuse ok=False error=protected proof-rig path: scripts/run_swebench_pro_parity.py" in output
    assert (repo / protected_path).read_text(encoding="utf-8") == "# protected SWE proof rig\n"


def test_plain_terminal_safe_write_prompt_runs_agent_loop(tmp_path: Path, capsys) -> None:
    repo = _tiny_repo(tmp_path)

    rc = david_main(
        [
            "code",
            str(repo),
            "--once",
            "create a file named hello.txt that says hello",
            "--allow-unvalidated",
            "--no-color",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "agent loop: verified steps=2 verified=True" in output
    assert "path=hello.txt bytes=5" in output
    assert (repo / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert (repo / ".david" / "memory" / "task-default.jsonl").exists()


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
