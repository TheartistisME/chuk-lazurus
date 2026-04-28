"""Install IDDIA slash command templates into .claude/commands."""

from __future__ import annotations

import shutil
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent
SOURCE_DIR = TOOL_ROOT / "slash" / "agent-context"
TARGET_DIR = REPO_ROOT / ".claude" / "commands"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_DIR.glob("*.md")):
        target = TARGET_DIR / f"agent-context:{source.stem}.md"
        shutil.copy2(source, target)
        print(f"{source.relative_to(REPO_ROOT)} -> {target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
