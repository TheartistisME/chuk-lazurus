"""Context helpers for meta-eval supervisor prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SIGNOFF_FIELDS = (
    "files_modified",
    "agent_objective",
    "tldr_of_change",
    "dependencies_mandatory_for_system_understanding",
)


def tail_lines(path: Path, limit: int = 100) -> str:
    """Return the final ``limit`` lines from a text file."""

    if not path.exists():
        return f"[missing changelog: {path}]"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def latest_signoff(signoff_dir: Path) -> dict[str, Any]:
    """Load the newest signoff record from a directory.

    Signoffs may be JSON files or plain text. JSON signoffs are normalized to
    the mandatory field names; text signoffs are returned as ``raw_text``.
    """

    if not signoff_dir.exists():
        return {
            "path": None,
            "missing": True,
            "raw_text": f"[missing signoff directory: {signoff_dir}]",
        }

    files = [path for path in signoff_dir.iterdir() if path.is_file()]
    if not files:
        return {"path": None, "missing": True, "raw_text": "[no signoff records found]"}

    newest = max(files, key=lambda path: (path.stat().st_mtime, path.name))
    text = newest.read_text(encoding="utf-8", errors="replace")
    if newest.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"path": str(newest), "raw_text": text}
        normalized = {"path": str(newest)}
        for field in SIGNOFF_FIELDS:
            normalized[field] = data.get(field) or data.get(field.replace("_", " "))
        normalized["raw_text"] = text
        return normalized
    return {"path": str(newest), "raw_text": text}


def render_context(changelog: Path | None, signoff_dir: Path | None) -> str:
    """Render changelog tail and latest signoff for prompt inclusion."""

    parts: list[str] = []
    if changelog is not None:
        parts.append("## Changelog tail\n" + tail_lines(changelog, 100))
    if signoff_dir is not None:
        signoff = latest_signoff(signoff_dir)
        if signoff.get("raw_text"):
            parts.append("## Latest signoff\n" + str(signoff["raw_text"]))
        else:
            parts.append("## Latest signoff\n" + json.dumps(signoff, indent=2, sort_keys=True))
    return "\n\n".join(parts)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Print meta-eval context.")
    parser.add_argument("--changelog", type=Path, help="Changelog file to tail.")
    parser.add_argument("--signoff-dir", type=Path, help="Directory containing signoff records.")
    args = parser.parse_args()
    print(render_context(args.changelog, args.signoff_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

