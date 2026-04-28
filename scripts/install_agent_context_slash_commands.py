"""Install tracked agent-context slash command templates into .claude/commands."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "prompts" / "slash" / "agent-context"
TARGET_DIR = ROOT / ".claude" / "commands"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(SOURCE_DIR.glob("*.md")):
        target = TARGET_DIR / f"agent-context:{source.stem}.md"
        shutil.copy2(source, target)
        print(f"{source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
