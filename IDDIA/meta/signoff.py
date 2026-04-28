"""IDDIA-local changelog, signoff, and helper-context utilities."""

from __future__ import annotations

from pathlib import Path

from .state import DEFAULT_META_ROOT, append_text, safe_slug, utc_now


def append_changelog(
    message: str,
    *,
    state_root: Path = DEFAULT_META_ROOT,
    timestamp: str | None = None,
) -> Path:
    path = state_root / "changelog.md"
    stamp = timestamp or utc_now()
    append_text(path, f"- {stamp}: {message.strip()}\n")
    return path


def append_signoff(
    *,
    files_modified: tuple[str, ...],
    agent_objective: str,
    tldr: str,
    mandatory_dependencies: tuple[str, ...],
    proof_claim: str = "",
    proof_metric: str = "",
    proof_evidence: tuple[str, ...] = (),
    proof_verdict: str = "",
    state_root: Path = DEFAULT_META_ROOT,
    timestamp: str | None = None,
) -> Path:
    rollup_path = state_root / "signoffs.md"
    stamp = timestamp or utc_now()
    lines = [
        f"## Signoff {stamp}",
        "",
        "- Files modified:",
        *[f"  - {item}" for item in files_modified],
        f"- Agent objective: {agent_objective.strip()}",
        f"- TLDR: {tldr.strip()}",
        "- Mandatory dependencies/context:",
        *[f"  - {item}" for item in mandatory_dependencies],
        "- Proof before signoff:",
        f"  - Claim: {proof_claim.strip() or 'not supplied'}",
        f"  - Metric/rubric: {proof_metric.strip() or 'not supplied'}",
        f"  - Verdict: {proof_verdict.strip() or 'not supplied'}",
        "  - Evidence:",
        *[f"    - {item}" for item in proof_evidence],
        "",
    ]
    text = "\n".join(lines)
    append_text(rollup_path, text)
    signoff_dir = state_root / "signoffs"
    signoff_dir.mkdir(parents=True, exist_ok=True)
    signoff_path = signoff_dir / f"{safe_slug(stamp, 'signoff')}.md"
    signoff_path.write_text(text, encoding="utf-8")
    return signoff_path


def _last_lines(path: Path, count: int) -> str:
    if not path.exists():
        return "(none recorded)"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-count:]) or "(empty)"


def _last_signoff(path: Path, signoff_dir: Path) -> str:
    files = sorted(signoff_dir.glob("*.md")) if signoff_dir.exists() else []
    if files:
        return files[-1].read_text(encoding="utf-8").strip() or "(empty)"
    if not path.exists():
        return "(none recorded)"
    text = path.read_text(encoding="utf-8")
    marker = "\n## Signoff "
    index = text.rfind(marker)
    if index >= 0:
        return text[index + 1 :].strip()
    return text.strip() or "(empty)"


def helper_context(*, state_root: Path = DEFAULT_META_ROOT) -> str:
    changelog_path = state_root / "changelog.md"
    signoff_path = state_root / "signoffs.md"
    signoff_dir = state_root / "signoffs"
    sections = [
        "# IDDIA Meta Helper Context",
        "",
        "## Changelog (last 100 lines)",
        "",
        _last_lines(changelog_path, 100),
        "",
        "## Last Agent Signoff",
        "",
        _last_signoff(signoff_path, signoff_dir),
        "",
        "## Mandatory Dependencies / Context",
        "",
        "- Meta state is IDDIA-local and ignored under `IDDIA/artifacts/meta/`.",
        "- Individual agent signoffs live under `IDDIA/artifacts/meta/signoffs/`.",
        "- Grade tool outputs with `python -m IDDIA.meta grade <package.json>`.",
        "- Append signoffs with `python -m IDDIA.meta signoff append ...`.",
        "- Signoffs must include proof claim, proof metric/rubric, proof evidence, and proof verdict before claiming improvement.",
        "- Improvement agents should run `python -m IDDIA.meta helper-context` first.",
        "- Vee spawning uses WSL, creates/reuses a tmux target, calls `vee agent spawn <agent> --name <name> --target <session:window>`, then sends startup text with `vee agent start <name> <prompt>`.",
        '- Spawned panes receive `VEE_BIN`; if `vee` is not on PATH they can close with `node "$VEE_BIN" agent kill <name>`.',
        "",
    ]
    return "\n".join(sections)
