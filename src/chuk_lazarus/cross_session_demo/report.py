"""VerificationReport pydantic model + builder + zero-modification probe."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from chuk_lazarus.cross_session_demo.verification import (
    QueryExecution,
    SIX_STRICT_KEYS,
    all_assertions_pass,
)

UPSTREAM_READ_ONLY_PATHS: tuple[str, ...] = (
    "src/chuk_lazarus/chat_loop",
    "src/chuk_lazarus/session_close",
    "src/chuk_lazarus/session_store",
    "src/chuk_lazarus/session_retrieval",
    "src/chuk_lazarus/inference/context/knowledge",
    "tools/build_clause_aligned_store.py",
    "tools/count_tokens.py",
)


class VerificationReport(BaseModel):
    """Terminal-axis acceptance report."""

    num_sessions: int
    session_ids: list[str]
    total_tokens: int
    token_budget_met: bool
    query_executions: list[QueryExecution]
    all_six_strict_per_query_pass: bool
    verbatim_recall_per_query: list[bool]
    zero_mod_upstream: dict[str, str]
    acceptance_passed: bool
    timestamp: str = Field(default_factory=lambda: _utc_now())


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_diff_for_path(repo_root: Path, path: str) -> str:
    """Return "clean" if the path has no staged/unstaged diff, otherwise
    a truncated ``git diff`` excerpt.

    Any git invocation failure (not a git repo, git missing) surfaces as
    a diagnostic string rather than crashing the demo: the outcome is
    captured in the report and downstream accountability is preserved.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--", path],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"<git-probe-error: {exc!r}>"
    if proc.returncode != 0:
        return f"<git-rc={proc.returncode}>{(proc.stderr or '').strip()[:200]}"
    diff_out = (proc.stdout or "").strip()
    if not diff_out:
        return "clean"
    # Truncate to keep report sane on large diffs.
    limited = diff_out[:2000]
    suffix = "" if len(diff_out) <= 2000 else "...<truncated>"
    return limited + suffix


def probe_zero_modification(repo_root: Path) -> dict[str, str]:
    """Return a ``path -> "clean"|<diff-excerpt>`` map for upstream axes."""
    root = Path(repo_root)
    return {path: _git_diff_for_path(root, path) for path in UPSTREAM_READ_ONLY_PATHS}


def build_report(
    *,
    sessions: list[dict],
    total_tokens: int,
    executions: list[QueryExecution],
    repo_root: Path,
) -> VerificationReport:
    """Assemble the terminal-axis VerificationReport."""
    session_ids = [s["session_id"] for s in sessions]
    token_budget_met = total_tokens > 128_000
    strict_pass = all(
        all(ex.strict_assertions.get(k, False) for k in SIX_STRICT_KEYS)
        for ex in executions
    )
    verbatim_recall = [ex.verbatim_hit for ex in executions]

    zero_mod = probe_zero_modification(repo_root)
    zero_mod_clean = all(v == "clean" for v in zero_mod.values())

    acceptance = (
        token_budget_met
        and strict_pass
        and all_assertions_pass(executions)
        and zero_mod_clean
    )

    return VerificationReport(
        num_sessions=len(sessions),
        session_ids=session_ids,
        total_tokens=total_tokens,
        token_budget_met=token_budget_met,
        query_executions=executions,
        all_six_strict_per_query_pass=strict_pass,
        verbatim_recall_per_query=verbatim_recall,
        zero_mod_upstream=zero_mod,
        acceptance_passed=acceptance,
    )


__all__ = [
    "UPSTREAM_READ_ONLY_PATHS",
    "VerificationReport",
    "build_report",
    "probe_zero_modification",
]
