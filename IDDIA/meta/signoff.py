"""IDDIA-local changelog, signoff, and helper-context utilities."""

from __future__ import annotations

from pathlib import Path

from .state import DEFAULT_META_ROOT, append_text, utc_now


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
    state_root: Path = DEFAULT_META_ROOT,
    timestamp: str | None = None,
) -> Path:
    path = state_root / "signoffs.md"
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
        "",
    ]
    append_text(path, "\n".join(lines))
    return path


def _last_lines(path: Path, count: int) -> str:
    if not path.exists():
        return "(none recorded)"
    return "\n".join(path.read_text(encoding="utf-8").splitlines()[-count:]) or "(empty)"


def _last_signoff(path: Path) -> str:
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
    sections = [
        "# IDDIA Meta Helper Context",
        "",
        "## Changelog (last 100 lines)",
        "",
        _last_lines(changelog_path, 100),
        "",
        "## Last Agent Signoff",
        "",
        _last_signoff(signoff_path),
        "",
        "## Mandatory Dependencies / Context",
        "",
        "- Meta state is IDDIA-local and ignored under `IDDIA/artifacts/meta/`.",
        "- Grade tool outputs with `python -m IDDIA.meta grade <package.json>`.",
        "- Append signoffs with `python -m IDDIA.meta signoff append ...`.",
        "- Improvement agents should run `python -m IDDIA.meta helper-context` first.",
        "- Vee spawning uses WSL and the cadence `vee agent spawn codex --name <name> --then <prompt>`, then `vee agent start <name>` when requested.",
        "",
    ]
    return "\n".join(sections)
