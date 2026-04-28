from __future__ import annotations

import json
import subprocess
from pathlib import Path

from IDDIA.meta.signoff import append_signoff
from IDDIA.meta.supervisor import supervise_spawn


def _completed(
    command: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _write_spawn_request(
    state_root: Path,
    *,
    request_id: str = "spawn-1",
    objective: str = "Improve IDDIA trust loop",
    worker_name: str = "iddia-worker",
    created_at: str = "2026-04-28T00:00:00+00:00",
) -> Path:
    request = {
        "schema_version": 1,
        "request_id": request_id,
        "created_at": created_at,
        "objective": objective,
        "worker_name": worker_name,
        "prompt_path": str(state_root / "spawn_requests" / f"{request_id}.prompt.md"),
        "tmux_target": "iddia-meta:agents",
    }
    path = state_root / "spawn_requests" / f"{request_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def test_spawn_without_signoff_fails_when_agent_not_running(tmp_path):
    request_path = _write_spawn_request(tmp_path)

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert "list-panes" in command
        return _completed(command, stdout="")

    result = supervise_spawn(
        request_path,
        state_root=tmp_path,
        runner=runner,
        use_wsl=False,
    )

    assert result.status == "failed_no_signoff"
    payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed_no_signoff"
    assert payload["request"]["worker_name"] == "iddia-worker"
    assert payload["observation"]["available"] is True
    assert payload["observation"]["agent_running"] is False


def test_spawn_with_matching_proof_signoff_completes(tmp_path):
    objective = "Improve IDDIA trust loop"
    request_path = _write_spawn_request(tmp_path, objective=objective)
    signoff_path = append_signoff(
        files_modified=("IDDIA/meta/supervisor.py",),
        agent_objective=objective,
        tldr="Added supervisor validation",
        mandatory_dependencies=("pytest",),
        proof_claim="Supervisor rejects missing signoff.",
        proof_metric="Focused tests pass.",
        proof_evidence=("test_meta_supervisor covers proof signoff completion",),
        proof_verdict="proven",
        state_root=tmp_path,
        timestamp="2026-04-28T00:01:00Z",
    )

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert "list-panes" in command
        return _completed(command, stdout="")

    result = supervise_spawn(
        request_path,
        state_root=tmp_path,
        runner=runner,
        use_wsl=False,
    )

    assert result.status == "completed"
    assert result.signoff_path == signoff_path
    payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["signoff"]["proof_verdict"] == "proven"
    assert payload["signoff"]["proof_complete"] is True


def test_spawn_ignores_proof_signoff_from_before_request(tmp_path):
    objective = "Improve IDDIA trust loop"
    request_path = _write_spawn_request(
        tmp_path,
        objective=objective,
        created_at="2026-04-28T00:05:00+00:00",
    )
    append_signoff(
        files_modified=("IDDIA/meta/supervisor.py",),
        agent_objective=objective,
        tldr="Older signoff must not complete newer request",
        mandatory_dependencies=("pytest",),
        proof_claim="Supervisor rejects stale signoffs.",
        proof_metric="Focused test covers stale signoff rejection.",
        proof_evidence=("this signoff is before the request timestamp",),
        proof_verdict="proven",
        state_root=tmp_path,
        timestamp="2026-04-28T00:01:00Z",
    )

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert "list-panes" in command
        return _completed(command, stdout="")

    result = supervise_spawn(
        request_path,
        state_root=tmp_path,
        runner=runner,
        use_wsl=False,
    )

    assert result.status == "failed_no_signoff"
    payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert payload["signoff"]["path"] is None


def test_tmux_pane_output_is_captured_when_available(tmp_path):
    request_path = _write_spawn_request(tmp_path)

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "list-panes" in command:
            return _completed(command, stdout="%7|1|0|node|iddia-worker\n")
        if "capture-pane" in command:
            return _completed(command, stdout="agent output\nproof attempt pending\n")
        raise AssertionError(f"unexpected tmux command: {command}")

    result = supervise_spawn(
        request_path,
        state_root=tmp_path,
        runner=runner,
        use_wsl=False,
    )

    assert result.status == "running"
    assert result.transcript_path is not None
    transcript = result.transcript_path.read_text(encoding="utf-8")
    assert "tmux pane %7" in transcript
    assert "agent output" in transcript
    payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert payload["observation"]["transcript_path"] == str(result.transcript_path)
    assert payload["observation"]["transcript_bytes"] > 0
