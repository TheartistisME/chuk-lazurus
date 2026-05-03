from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime


def test_missing_index_requires_jit_plan(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Inspect source dependencies for this workspace")

    assert result.index.required is True
    assert result.index.jit_plan is not None
    assert result.index.jit_plan["action"] == "jit_index_workspace"
    assert "jit_required" in result.answer


def test_repo_patch_routing_and_task_writeback(tmp_path: Path) -> None:
    source = tmp_path / "src" / "example.py"
    source.parent.mkdir()
    source.write_text("def broken_session():\n    return None\n", encoding="utf-8")
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Fix the repo bug by patching src/example.py")

    assert result.method == "repo_patch"
    assert result.route.memory_family == "task"
    assert any(item.get("path") == "src/example.py" for item in result.route.evidence)
    assert "patch_compatible_edits" in result.decoder.constraints["bias"]
    assert result.writeback["family"] == "task"
    assert (tmp_path / "state" / "memory" / "task-default.jsonl").exists()
    assert not (tmp_path / "state" / "memory" / "user-default.jsonl").exists()


def test_verification_command_behavior(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("Verify quality gate", verify_command=["python", "-c", "print('ok')"])

    assert result.method == "verify"
    assert result.verification.ok is True
    assert result.verification.command_result is not None
    assert result.verification.command_result["returncode"] == 0
    assert "ok" in result.verification.command_result["stdout"]


def test_runtime_status_and_shell_commands_are_available_to_tui(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    readiness = runtime.readiness()
    shell = runtime.run_shell("python -c \"print('hello')\"")

    assert readiness["model validation"].startswith("ready")
    assert "index" in readiness
    assert "rc=0" in shell
    assert "hello" in shell
