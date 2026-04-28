"""Install IDDIA slash command templates into .claude/commands."""

from __future__ import annotations

import shutil
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOL_ROOT.parent
SOURCE_DIR = TOOL_ROOT / "slash" / "agent-context"
TARGET_DIR = REPO_ROOT / ".claude" / "commands"


def install_commands(source_dir: Path = SOURCE_DIR, target_dir: Path = TARGET_DIR) -> list[Path]:
    namespace_dir = target_dir / "agent-context"
    namespace_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    for source in sorted(source_dir.glob("*.md")):
        target = namespace_dir / source.name
        shutil.copy2(source, target)
        installed.append(target)
    return installed


def main() -> None:
    for target in install_commands():
        print(f"{target.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
