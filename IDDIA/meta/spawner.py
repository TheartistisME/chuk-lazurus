"""Conservative vee-backed improvement-agent spawning with pending fallback."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .state import DEFAULT_META_ROOT, safe_slug, utc_now, utc_stamp, write_json

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SpawnResult:
    status: str
    request_path: Path
    prompt_path: Path
    command: list[str]
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


def windows_path_to_wsl(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return text.replace("\\", "/")


def default_vee_repo() -> Path:
    return Path.home() / "Desktop" / "vee"


def default_vee_agent() -> str:
    return os.environ.get("IDDIA_VEE_AGENT", "claude")


def build_prompt(
    *,
    grade_record: dict[str, Any],
    objective: str,
    helper_command: str,
    worker_name: str,
) -> str:
    criteria_lines = []
    for item in grade_record.get("criteria", []):
        criteria_lines.append(
            f"- {item.get('id')}: {float(item.get('score', 0.0)):.2f} "
            f"({'; '.join(str(note) for note in item.get('notes', []))})"
        )
    return "\n".join(
        [
            "You are an IDDIA improvement agent.",
            "",
            f"Objective: {objective}",
            "",
            "Stay inside the IDDIA write scope. Do not modify non-IDDIA files.",
            "Do not quote DDIA source text in reports or commits.",
            "",
            "Grade summary:",
            f"- Grade id: {grade_record.get('grade_id')}",
            f"- Overall score: {grade_record.get('overall_score')}/5 ({grade_record.get('label')})",
            f"- Recommendation: {grade_record.get('recommendation')}",
            "",
            "Criteria:",
            *(criteria_lines or ["- No criteria recorded."]),
            "",
            "First commands to run:",
            f"- `{helper_command}`",
            "- `python -m pytest IDDIA/tests`",
            "",
            "Your improvement loop:",
            "- Read the grade, read the relevant IDDIA code, and determine how to improve the system.",
            "- Decide whether a new grading criterion should be added for the next grade.",
            "- Make the scoped IDDIA fixes, update or append the changelog, and run focused tests.",
            "- Append an IDDIA meta signoff with files modified, objective, TLDR, and mandatory dependencies/context.",
            f"- After signoff, close your Vee process with `vee agent kill {worker_name}` if that command is available.",
            "",
        ]
    )


def build_wsl_script(
    *,
    repo_root: Path,
    vee_repo: Path,
    prompt_path: Path,
    worker_name: str,
    vee_agent: str,
) -> str:
    repo_wsl = windows_path_to_wsl(repo_root)
    vee_wsl = windows_path_to_wsl(vee_repo)
    prompt_wsl = windows_path_to_wsl(prompt_path)
    spawn_args = (
        f"run_vee agent spawn {shlex.quote(vee_agent)} --name "
        f'{shlex.quote(worker_name)} --then "$(cat {shlex.quote(prompt_wsl)})"'
    )
    lines = [
        "set -eu",
        f"cd {shlex.quote(repo_wsl)}",
        f"VEE_REPO={shlex.quote(vee_wsl)}",
        "run_vee() {",
        '  if command -v vee >/dev/null 2>&1; then vee "$@"; return $?; fi',
        '  if [ -f "$VEE_REPO/dist/cli.js" ]; then node "$VEE_REPO/dist/cli.js" "$@"; return $?; fi',
        '  if [ -x "$VEE_REPO/eve" ]; then "$VEE_REPO/eve" "$@"; return $?; fi',
        "  return 127",
        "}",
        spawn_args,
    ]
    return "\n".join(lines)


def pending_request(
    *,
    state_root: Path,
    request_id: str,
    request: dict[str, Any],
    prompt: str,
    reason: str,
) -> SpawnResult:
    prompt_path = state_root / "spawn_requests" / f"{request_id}.prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    request["status"] = "pending"
    request["pending_reason"] = reason
    request["prompt_path"] = str(prompt_path)
    request_path = state_root / "spawn_requests" / f"{request_id}.json"
    write_json(request_path, request)
    return SpawnResult(
        status="pending",
        request_path=request_path,
        prompt_path=prompt_path,
        command=[],
        stderr=reason,
    )


def spawn_improvement_agent(
    grade_record: dict[str, Any],
    *,
    objective: str,
    state_root: Path = DEFAULT_META_ROOT,
    repo_root: Path | None = None,
    vee_repo: Path | None = None,
    vee_agent: str | None = None,
    worker_name: str | None = None,
    runner: Runner = subprocess.run,
    force_pending: bool = False,
) -> SpawnResult:
    repo_root = repo_root or Path.cwd()
    vee_repo = vee_repo or default_vee_repo()
    vee_agent = vee_agent or default_vee_agent()
    request_id = f"{utc_stamp()}-{safe_slug(str(grade_record.get('grade_id') or 'iddia-improve'))}"
    worker = (
        worker_name or safe_slug(f"iddia-improve-{grade_record.get('grade_id', utc_stamp())}")[:48]
    )
    helper = "python -m IDDIA.meta helper-context"
    prompt = build_prompt(
        grade_record=grade_record,
        objective=objective,
        helper_command=helper,
        worker_name=worker,
    )
    request = {
        "schema_version": 1,
        "request_id": request_id,
        "created_at": utc_now(),
        "grade_id": grade_record.get("grade_id"),
        "objective": objective,
        "worker_name": worker,
        "vee_agent": vee_agent,
        "vee_repo": str(vee_repo),
        "repo_root": str(repo_root),
    }
    if force_pending:
        return pending_request(
            state_root=state_root,
            request_id=request_id,
            request=request,
            prompt=prompt,
            reason="forced pending fallback",
        )

    prompt_path = state_root / "spawn_requests" / f"{request_id}.prompt.md"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")
    request["prompt_path"] = str(prompt_path)

    if os.name == "nt":
        command = [
            "wsl.exe",
            "-e",
            "sh",
            "-lc",
            build_wsl_script(
                repo_root=repo_root,
                vee_repo=vee_repo,
                prompt_path=prompt_path,
                worker_name=worker,
                vee_agent=vee_agent,
            ),
        ]
    else:
        command = [
            "sh",
            "-lc",
            build_wsl_script(
                repo_root=repo_root,
                vee_repo=vee_repo,
                prompt_path=prompt_path,
                worker_name=worker,
                vee_agent=vee_agent,
            ),
        ]
    request["command"] = command
    try:
        result = runner(command, capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        return pending_request(
            state_root=state_root,
            request_id=request_id,
            request=request,
            prompt=prompt,
            reason=f"vee invocation unavailable: {exc}",
        )

    request["stdout"] = result.stdout
    request["stderr"] = result.stderr
    request["returncode"] = result.returncode
    if result.returncode != 0:
        return pending_request(
            state_root=state_root,
            request_id=request_id,
            request=request,
            prompt=prompt,
            reason=f"vee invocation failed with exit code {result.returncode}",
        )

    request["status"] = "spawned"
    request_path = state_root / "spawn_requests" / f"{request_id}.json"
    write_json(request_path, request)
    return SpawnResult(
        status="spawned",
        request_path=request_path,
        prompt_path=prompt_path,
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )
