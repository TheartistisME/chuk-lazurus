"""Supervisor checks for spawned IDDIA improvement agents."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import DEFAULT_META_ROOT, read_json, utc_now, write_json

Runner = Callable[..., subprocess.CompletedProcess[str]]
VALID_PROOF_VERDICTS = {"proven", "partially_proven", "not_proven"}


@dataclass(frozen=True)
class SupervisionResult:
    status: str
    status_path: Path
    transcript_path: Path | None
    signoff_path: Path | None
    proof_verdict: str | None

    @property
    def run_path(self) -> Path:
        return self.status_path


def run_metadata(state_root: Path, request_id: str) -> dict[str, str]:
    run_dir = state_root / "spawn_runs" / request_id
    return {
        "spawn_run_dir": str(run_dir),
        "status_path": str(run_dir / "status.json"),
        "transcript_path": str(run_dir / "pane.log"),
    }


def extract_pane_ids(text: str) -> tuple[str, ...]:
    ids = re.findall(r"(?<!\S)(%[0-9]+)(?!\S)", text)
    return tuple(dict.fromkeys(ids))


def _tmux_command(args: list[str], *, use_wsl: bool) -> list[str]:
    if use_wsl:
        script = "tmux " + " ".join(shlex.quote(arg) for arg in args)
        if os.name == "nt":
            return ["wsl.exe", "-e", "sh", "-lc", script]
        return ["sh", "-lc", script]
    return ["tmux", *args]


def _run_tmux(
    args: list[str],
    *,
    runner: Runner,
    use_wsl: bool,
) -> subprocess.CompletedProcess[str]:
    return runner(
        _tmux_command(args, use_wsl=use_wsl),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def list_tmux_panes(
    *,
    request: dict[str, Any],
    runner: Runner = subprocess.run,
    use_wsl: bool = True,
) -> dict[str, Any]:
    target = str(request.get("tmux_target") or "")
    if not target:
        return {"available": False, "error": "request has no tmux_target", "panes": []}
    result = _run_tmux(
        ["list-panes", "-t", target, "-F", "#{pane_id}|#{pane_index}|#{pane_active}|#{pane_current_command}|#{pane_title}"],
        runner=runner,
        use_wsl=use_wsl,
    )
    panes: list[dict[str, str]] = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            parts = line.split("|", maxsplit=4)
            if len(parts) < 2 or not parts[0].startswith("%"):
                continue
            panes.append(
                {
                    "pane_id": parts[0],
                    "pane_index": parts[1],
                    "pane_active": parts[2] if len(parts) > 2 else "",
                    "current_command": parts[3] if len(parts) > 3 else "",
                    "pane_title": parts[4] if len(parts) > 4 else "",
                }
            )
    return {
        "available": result.returncode == 0,
        "error": result.stderr.strip(),
        "panes": panes,
    }


def pane_matches_request(pane: dict[str, str], request: dict[str, Any]) -> bool:
    worker = str(request.get("worker_name") or "").lower()
    if worker and worker in str(pane.get("pane_title") or "").lower():
        return True
    pane_ids = {str(item) for item in request.get("pane_ids", [])}
    pane_ids.update(str(item.get("pane_id")) for item in request.get("spawned_panes", []))
    return bool(pane.get("pane_id") in pane_ids)


def capture_tmux_output(
    *,
    panes: list[dict[str, str]],
    request: dict[str, Any],
    runner: Runner = subprocess.run,
    use_wsl: bool = True,
) -> tuple[str, str]:
    matched = [pane for pane in panes if pane_matches_request(pane, request)]
    if not matched and len(panes) == 1:
        matched = panes
    outputs: list[str] = []
    errors: list[str] = []
    for pane in matched:
        pane_id = str(pane.get("pane_id") or "")
        if not pane_id:
            continue
        result = _run_tmux(
            ["capture-pane", "-p", "-S", "-2000", "-t", pane_id],
            runner=runner,
            use_wsl=use_wsl,
        )
        if result.stdout:
            outputs.append(f"# tmux pane {pane_id}\n{result.stdout.rstrip()}")
        if result.stderr:
            errors.append(f"# tmux pane {pane_id}\n{result.stderr.rstrip()}")
    return "\n\n".join(outputs), "\n\n".join(errors)


def _field_value(text: str, label: str) -> str:
    match = re.search(rf"^\s*-\s*{re.escape(label)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def proof_verdict(text: str) -> str | None:
    verdict = _field_value(text, "Verdict").lower()
    return verdict if verdict in VALID_PROOF_VERDICTS else None


def signoff_has_proof(text: str) -> bool:
    claim = _field_value(text, "Claim")
    metric = _field_value(text, "Metric/rubric")
    verdict = proof_verdict(text)
    evidence_section = text.split("- Evidence:", 1)[1] if "- Evidence:" in text else ""
    evidence_items = [line for line in evidence_section.splitlines() if line.strip().startswith("- ")]
    return bool(
        claim
        and claim.lower() != "not supplied"
        and metric
        and metric.lower() != "not supplied"
        and verdict
        and evidence_items
    )


def _objective_tokens(objective: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{5,}", objective.lower())
    ignored = {"iddia", "based", "grade", "improve", "quality", "system"}
    return {word for word in words if word not in ignored}


def signoff_matches_request(text: str, request: dict[str, Any]) -> bool:
    worker = str(request.get("worker_name") or "").lower()
    if worker and worker in text.lower():
        return True
    objective_tokens = _objective_tokens(str(request.get("objective") or ""))
    if not objective_tokens:
        return True
    matched = {token for token in objective_tokens if token in text.lower()}
    return len(matched) >= min(3, len(objective_tokens))


def find_matching_signoff(
    *,
    request: dict[str, Any],
    state_root: Path = DEFAULT_META_ROOT,
) -> dict[str, Any]:
    signoff_dir = state_root / "signoffs"
    signoffs = sorted(signoff_dir.glob("*.md")) if signoff_dir.exists() else []
    fallback: dict[str, Any] = {
        "path": None,
        "proof_verdict": None,
        "proof_complete": False,
    }
    for path in reversed(signoffs):
        text = path.read_text(encoding="utf-8")
        if not signoff_matches_request(text, request):
            continue
        proof_complete = signoff_has_proof(text)
        item = {
            "path": str(path),
            "proof_verdict": proof_verdict(text),
            "proof_complete": proof_complete,
        }
        if proof_complete:
            return item
        if fallback["path"] is None:
            fallback = item
    return fallback


def write_spawn_run_status(
    *,
    state_root: Path,
    request: dict[str, Any],
    status: str,
    status_reason: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    request_id = str(request.get("request_id") or "spawn-request")
    metadata = run_metadata(state_root, request_id)
    payload = {
        "schema_version": 1,
        "request_id": request_id,
        "recorded_at": utc_now(),
        "status": status,
        "status_reason": status_reason,
        "request": {
            "worker_name": request.get("worker_name"),
            "objective": request.get("objective"),
            "grade_id": request.get("grade_id"),
            "tmux_target": request.get("tmux_target"),
            "request_path": request.get("request_path"),
        },
    }
    if extra:
        payload.update(extra)
    return write_json(Path(metadata["status_path"]), payload)


def supervise_spawn(
    request_path: Path,
    *,
    state_root: Path = DEFAULT_META_ROOT,
    wait_seconds: float = 0,
    runner: Runner = subprocess.run,
    use_wsl: bool | None = None,
) -> SupervisionResult:
    request = read_json(request_path)
    if not isinstance(request, dict):
        raise ValueError(f"{request_path} is not a JSON object")
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    if use_wsl is None:
        use_wsl = os.name == "nt"

    request_id = str(request.get("request_id") or request_path.stem)
    request.setdefault("request_path", str(request_path))
    request.update({key: request.get(key, value) for key, value in run_metadata(state_root, request_id).items()})

    observation = list_tmux_panes(request=request, runner=runner, use_wsl=use_wsl)
    panes = list(observation["panes"])
    agent_running = any(pane_matches_request(pane, request) for pane in panes)
    observation["agent_running"] = agent_running

    transcript_path: Path | None = None
    transcript, capture_errors = capture_tmux_output(
        panes=panes,
        request=request,
        runner=runner,
        use_wsl=use_wsl,
    )
    if transcript or capture_errors:
        transcript_path = Path(str(request["transcript_path"]))
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            "\n\n".join(part for part in (transcript, capture_errors) if part),
            encoding="utf-8",
        )
    observation["transcript_path"] = str(transcript_path) if transcript_path else None
    observation["transcript_bytes"] = transcript_path.stat().st_size if transcript_path else 0
    observation["capture_errors"] = capture_errors

    signoff = find_matching_signoff(request=request, state_root=state_root)
    if signoff["proof_complete"]:
        status = "completed"
        status_reason = "matching signoff with proof fields found"
    elif signoff["path"]:
        status = "failed_no_proof" if not agent_running else "running"
        status_reason = "matching signoff exists but proof fields are incomplete"
    elif agent_running:
        status = "running"
        status_reason = "matching pane is still running and no signoff exists yet"
    else:
        status = "failed_no_signoff"
        status_reason = "agent is not running and no matching signoff exists"

    status_path = write_spawn_run_status(
        state_root=state_root,
        request=request,
        status=status,
        status_reason=status_reason,
        extra={
            "observation": observation,
            "signoff": signoff,
        },
    )
    return SupervisionResult(
        status=status,
        status_path=status_path,
        transcript_path=transcript_path,
        signoff_path=Path(signoff["path"]) if signoff["path"] else None,
        proof_verdict=signoff["proof_verdict"],
    )


def supervise_spawn_request(
    request_path: Path,
    *,
    state_root: Path = DEFAULT_META_ROOT,
    wait_seconds: int = 0,
    runner: Runner = subprocess.run,
) -> SupervisionResult:
    return supervise_spawn(
        request_path,
        state_root=state_root,
        wait_seconds=wait_seconds,
        runner=runner,
    )
