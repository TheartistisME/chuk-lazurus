from __future__ import annotations

from pathlib import Path
import sys

from chuk_lazarus.david.agent_loop import parse_agent_action, run_agent_loop
from chuk_lazarus.david.tools import LocalTools


def test_parse_agent_action_from_fenced_model_text() -> None:
    action = parse_agent_action(
        """I will read the file.

```json
{"action": "read", "path": "src/example.py"}
```
"""
    )

    assert action is not None
    assert action.action == "read"
    assert action.path == "src/example.py"


def test_agent_loop_reads_workspace_file(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("remember me\n", encoding="utf-8")

    result = run_agent_loop([{"action": "read", "path": "notes.txt"}], LocalTools(tmp_path))

    assert result.status == "no_action"
    assert result.trace[0].action == "read"
    assert result.trace[0].observation["content"] == "remember me\n"


def test_agent_loop_writes_workspace_file(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "write", "path": "src/new.py", "content": "VALUE = 42\n"}],
        LocalTools(tmp_path),
    )

    assert result.status == "no_action"
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == "VALUE = 42\n"
    assert result.trace[0].observation["bytes"] == len("VALUE = 42\n")


def test_agent_loop_runs_shell_command(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "run", "command": [sys.executable, "-c", "print('ok')"]}],
        LocalTools(tmp_path),
    )

    assert result.status == "no_action"
    assert result.trace[0].action == "run"
    assert result.trace[0].ok is True
    assert result.trace[0].observation["stdout"] == "ok\n"


def test_agent_loop_stops_on_verify_pass(tmp_path: Path) -> None:
    result = run_agent_loop(
        [
            {"action": "plan", "reason": "verify directly"},
            {"action": "verify", "passed": True, "reason": "tests passed"},
            {"action": "write", "path": "should_not_exist.txt", "content": "nope"},
        ],
        LocalTools(tmp_path),
    )

    assert result.status == "verified"
    assert result.ok is True
    assert result.steps == 2
    assert not (tmp_path / "should_not_exist.txt").exists()


def test_agent_loop_reports_verify_failure(tmp_path: Path) -> None:
    result = run_agent_loop(
        [{"action": "verify", "command": [sys.executable, "-c", "raise SystemExit(7)"]}],
        LocalTools(tmp_path),
    )

    assert result.status == "verify_failed"
    assert result.verified is False
    assert result.trace[0].action == "verify"
    assert result.trace[0].observation["returncode"] == 7


def test_agent_loop_refuses_unknown_action(tmp_path: Path) -> None:
    result = run_agent_loop([{"action": "delete", "path": "notes.txt"}], LocalTools(tmp_path))

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert "unknown action" in result.reason


def test_agent_loop_enforces_max_step_cap(tmp_path: Path) -> None:
    result = run_agent_loop(
        [
            {"action": "plan", "reason": "one"},
            {"action": "plan", "reason": "two"},
            {"action": "verify", "passed": True},
        ],
        LocalTools(tmp_path),
        max_steps=2,
    )

    assert result.status == "max_steps"
    assert result.verified is False
    assert len(result.trace) == 2


def test_agent_loop_refuses_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    result = run_agent_loop([{"action": "read", "path": "../outside.txt"}], LocalTools(tmp_path))

    assert result.status == "refused"
    assert result.trace[0].action == "refuse"
    assert "escapes workspace" in result.reason
