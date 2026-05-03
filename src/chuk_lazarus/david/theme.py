"""Small terminal theme helpers for the David TUI.

The module is intentionally standard-library only. Color is optional and is
enabled only for TTY-like streams unless a caller opts in explicitly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TextIO


RESET = "\033[0m"
STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}


def should_use_color(stream: TextIO | None, force: bool | None = None) -> bool:
    """Return whether ANSI color should be emitted for ``stream``."""

    if force is not None:
        return bool(force)
    if os.environ.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    try:
        return bool(isatty and isatty())
    except OSError:
        return False


@dataclass(frozen=True)
class Theme:
    """Formatting primitives for David's terminal UI."""

    use_color: bool = False
    width: int = 78

    @classmethod
    def for_stream(
        cls, stream: TextIO | None, *, use_color: bool | None = None, width: int = 78
    ) -> "Theme":
        return cls(use_color=should_use_color(stream, use_color), width=width)

    def style(self, text: object, *styles: str) -> str:
        value = str(text)
        if not self.use_color:
            return value
        prefix = "".join(STYLES[name] for name in styles if name in STYLES)
        if not prefix:
            return value
        return f"{prefix}{value}{RESET}"

    def banner(self, title: str = "DAVID", subtitle: str | None = None) -> str:
        body_width = max(20, self.width - 4)
        title_line = f" {title.strip()} ".center(body_width, "=")
        lines = [
            "+" + "=" * (body_width + 2) + "+",
            "| " + self.style(title_line, "bold", "cyan") + " |",
        ]
        if subtitle:
            lines.append("| " + subtitle.strip()[:body_width].ljust(body_width) + " |")
        lines.append("+" + "=" * (body_width + 2) + "+")
        return "\n".join(lines)

    def section(self, title: str) -> str:
        label = f"[ {title.strip()} ]"
        fill = max(0, self.width - len(label) - 1)
        return self.style(f"{label}{'-' * fill}", "cyan")

    def prompt(self) -> str:
        return self.style("david> ", "bold", "green")

    def badge(self, label: str, *, ok: bool | None = None) -> str:
        color = "cyan"
        if ok is True:
            color = "green"
        elif ok is False:
            color = "yellow"
        return self.style(f"[{label}]", color)

    def key_value(self, key: str, value: object) -> str:
        return f"{self.style(key + ':', 'bold')} {value}"
