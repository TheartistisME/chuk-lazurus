"""Path-safe local tools for the David runtime."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence

from .patch_routing import is_protected_path, normalize_path


class PathSafetyError(ValueError):
    pass


class LocalTools:
    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def resolve(self, path: str | Path = ".") -> Path:
        candidate = (self.workspace_root / Path(path)).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PathSafetyError(f"path escapes workspace: {path}") from exc
        return candidate

    def read(self, path: str | Path) -> str:
        return self.resolve(path).read_text(encoding="utf-8")

    def write(self, path: str | Path, content: str) -> Path:
        normalized = normalize_path(str(path))
        if is_protected_path(normalized):
            raise PathSafetyError(f"protected proof-rig path: {normalized}")
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def list(self, path: str | Path = ".") -> list[str]:
        root = self.resolve(path)
        return sorted(child.name for child in root.iterdir())

    def run(self, command: Sequence[str], *, cwd: str | Path = ".", timeout: int = 30) -> dict[str, object]:
        if isinstance(command, (str, bytes)):
            raise TypeError("command must be a sequence of arguments")
        run_cwd = self.resolve(cwd)
        completed = subprocess.run(
            list(command),
            cwd=run_cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": list(command),
            "cwd": str(run_cwd),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
