from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus.david.tools import LocalTools, PathSafetyError


def test_local_tools_read_write_list_inside_workspace(tmp_path: Path) -> None:
    tools = LocalTools(tmp_path)

    tools.write("src/example.py", "VALUE = 1\n")

    assert tools.read("src/example.py") == "VALUE = 1\n"
    assert "example.py" in tools.list("src")


def test_local_tools_reject_path_escape(tmp_path: Path) -> None:
    tools = LocalTools(tmp_path)

    with pytest.raises(PathSafetyError):
        tools.read("../outside.txt")


def test_local_tools_run_requires_argument_sequence(tmp_path: Path) -> None:
    tools = LocalTools(tmp_path)

    with pytest.raises(TypeError):
        tools.run("python -c pass")  # type: ignore[arg-type]

