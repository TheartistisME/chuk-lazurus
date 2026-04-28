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
from .supervisor import extract_pane_ids, run_metadata, write_spawn_run_status

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
    run_status_path: Path | None = None
    transcript_path: Path | None = None


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


def default_tmux_session() -> str:
    return os.environ.get("IDDIA_VEE_TMUX_SESSION", "iddia-meta")


def default_tmux_window() -> str:
    return os.environ.get("IDDIA_VEE_TMUX_WINDOW", "agents")


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
            "- `python3 -m pytest IDDIA/tests`",
            "",
            "Your improvement loop:",
            "- Read the grade, read the relevant IDDIA code, and determine how to improve the system.",
            "- Decide whether a new grading criterion should be added for the next grade.",
            "- Make the scoped IDDIA fixes, update or append the changelog, and run focused tests.",
            "",
            "Proof-before-signoff contract:",
            "- Before changing behavior, state the claim you intend to prove and the fixed metric or rubric that will prove it.",
            "- Use a frozen benchmark/rubric where possible; for retrieval work prefer the existing IDDIA eval package runner or the same meta grade command before and after.",
            "- A passing unit test proves only the implementation contract, not better retrieval quality. Do not claim product improvement from tests alone.",
            "- If the claim cannot be proven in this run, say so explicitly and sign off with verdict `not_proven` or `partially_proven`.",
            "- Your signoff must include proof claim, proof metric, proof evidence, and proof verdict.",
            "- Append an IDDIA meta signoff with files modified, objective, TLDR, mandatory dependencies/context, and proof fields.",
            f"- After signoff, close your Vee process with `vee agent kill {worker_name}` if that command is available.",
            f'- If `vee` is not on PATH, use `node "$VEE_BIN" agent kill {worker_name}`.',
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
    tmux_session: str,
    tmux_window: str,
) -> str:
    repo_wsl = windows_path_to_wsl(repo_root)
    vee_wsl = windows_path_to_wsl(vee_repo)
    prompt_wsl = windows_path_to_wsl(prompt_path)
    tmux_target = f"{tmux_session}:{tmux_window}"
    spawn_args = (
        f"run_vee agent spawn {shlex.quote(vee_agent)} --name "
        f"{shlex.quote(worker_name)} --cwd {shlex.quote(repo_wsl)} "
        f"--target {shlex.quote(tmux_target)}"
    )
    start_args = (
        f'run_vee agent start {shlex.quote(worker_name)} "$(cat {shlex.quote(prompt_wsl)})"'
    )
    lines = [
        "set -eu",
        f"cd {shlex.quote(repo_wsl)}",
        f"VEE_REPO={shlex.quote(vee_wsl)}",
        f"TMUX_SESSION={shlex.quote(tmux_session)}",
        f"TMUX_WINDOW={shlex.quote(tmux_window)}",
        "run_vee() {",
        '  if command -v vee >/dev/null 2>&1; then vee "$@"; return $?; fi',
        '  if [ -f "$VEE_REPO/dist/cli.js" ]; then node "$VEE_REPO/dist/cli.js" "$@"; return $?; fi',
        '  if [ -x "$VEE_REPO/eve" ]; then "$VEE_REPO/eve" "$@"; return $?; fi',
        "  return 127",
        "}",
        'if ! command -v tmux >/dev/null 2>&1; then echo "tmux is required for vee agent spawn" >&2; exit 127; fi',
        'if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then',
        '  tmux new-session -d -s "$TMUX_SESSION" -n "$TMUX_WINDOW" -c ' + shlex.quote(repo_wsl),
        "else",
        '  if ! tmux list-windows -t "$TMUX_SESSION" -F "#{window_name}" | grep -Fx "$TMUX_WINDOW" >/dev/null; then',
        '    tmux new-window -d -t "$TMUX_SESSION" -n "$TMUX_WINDOW" -c ' + shlex.quote(repo_wsl),
        "  fi",
        "fi",
        'tmux set-environment -t "$TMUX_SESSION" VEE_REPO "$VEE_REPO"',
        'tmux set-environment -t "$TMUX_SESSION" VEE_BIN "$VEE_REPO/dist/cli.js"',
        spawn_args,
        "sleep ${IDDIA_VEE_STARTUP_DELAY_SECONDS:-6}",
        start_args,
        "sleep 1",
        'tmux list-panes -t "$TMUX_SESSION:$TMUX_WINDOW" -F "#{session_name}:#{window_name}.#{pane_index} #{pane_id} #{pane_current_command} #{pane_title}"',
    ]
    return "\n".join(lines)


def parse_spawned_panes(output: str) -> list[dict[str, str]]:
    panes: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 2 or not parts[1].startswith("%"):
            continue
        panes.append(
            {
                "tmux_pane_target": parts[0],
                "pane_id": parts[1],
                "current_command": parts[2] if len(parts) > 2 else "",
                "pane_title": parts[3] if len(parts) > 3 else "",
            }
        )
    return panes


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
    request["request_path"] = str(request_path)
    write_json(request_path, request)
    run_status_path = write_spawn_run_status(
        state_root=state_root,
        request=request,
        status="pending",
        status_reason=reason,
        extra={
            "spawn_stdout": request.get("stdout", ""),
            "spawn_stderr": request.get("stderr", ""),
            "spawn_returncode": request.get("returncode"),
        },
    )
    return SpawnResult(
        status="pending",
        request_path=request_path,
        prompt_path=prompt_path,
        command=[],
        stderr=reason,
        run_status_path=run_status_path,
        transcript_path=Path(request["transcript_path"]),
    )


def spawn_improvement_agent(
    grade_record: dict[str, Any],
    *,
    objective: str,
    state_root: Path = DEFAULT_META_ROOT,
    repo_root: Path | None = None,
    vee_repo: Path | None = None,
    vee_agent: str | None = None,
    tmux_session: str | None = None,
    tmux_window: str | None = None,
    worker_name: str | None = None,
    runner: Runner = subprocess.run,
    force_pending: bool = False,
) -> SpawnResult:
    repo_root = repo_root or Path.cwd()
    vee_repo = vee_repo or default_vee_repo()
    vee_agent = vee_agent or default_vee_agent()
    tmux_session = tmux_session or default_tmux_session()
    tmux_window = tmux_window or default_tmux_window()
    request_id = f"{utc_stamp()}-{safe_slug(str(grade_record.get('grade_id') or 'iddia-improve'))}"
    worker = (
        worker_name or safe_slug(f"iddia-improve-{grade_record.get('grade_id', utc_stamp())}")[:48]
    )
    helper = "python3 -m IDDIA.meta helper-context"
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
        "tmux_session": tmux_session,
        "tmux_window": tmux_window,
        "tmux_target": f"{tmux_session}:{tmux_window}",
        "vee_repo": str(vee_repo),
        "repo_root": str(repo_root),
    }
    request.update(run_metadata(state_root, request_id))
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
                tmux_session=tmux_session,
                tmux_window=tmux_window,
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
                tmux_session=tmux_session,
                tmux_window=tmux_window,
            ),
        ]
    request["command"] = command
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
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
    request["spawned_panes"] = parse_spawned_panes(result.stdout)
    request["pane_ids"] = extract_pane_ids(result.stdout)
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
    request["request_path"] = str(request_path)
    write_json(request_path, request)
    run_status_path = write_spawn_run_status(
        state_root=state_root,
        request=request,
        status="spawned",
        status_reason="vee spawn command completed",
        extra={
            "spawn_stdout": result.stdout,
            "spawn_stderr": result.stderr,
            "spawn_returncode": result.returncode,
            "spawned_panes": request["spawned_panes"],
        },
    )
    return SpawnResult(
        status="spawned",
        request_path=request_path,
        prompt_path=prompt_path,
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
        run_status_path=run_status_path,
        transcript_path=Path(request["transcript_path"]),
    )
