"""Zero-modification architectural guarantee for ``tools/build_clause_aligned_store.py``.

The build script is INVOKE-ONLY. Axis-3 must never patch it (not even a
comment). These tests fail loudly if anyone edits the script, deletes it, or
couples it to the wrapper package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

# tests/session_store/test_zero_modification.py -> four parents up = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT_REL = "tools/build_clause_aligned_store.py"
BUILD_SCRIPT_ABS = REPO_ROOT / BUILD_SCRIPT_REL


def test_build_script_exists() -> None:
    assert BUILD_SCRIPT_ABS.exists(), (
        f"Build script missing at {BUILD_SCRIPT_ABS}; "
        "axis-3 wrapper has nothing to invoke."
    )


def test_build_script_has_zero_uncommitted_changes() -> None:
    proc = subprocess.run(
        ["git", "diff", "HEAD", "--", BUILD_SCRIPT_REL],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    # returncode should be 0 for a clean diff; stdout/stderr should be empty.
    if proc.returncode != 0 or proc.stdout != "" or proc.stderr != "":
        pytest.fail(
            "tools/build_clause_aligned_store.py has uncommitted modifications "
            "or git reported an error. Axis-3 is INVOKE-ONLY; the build script "
            "must remain bit-for-bit unchanged.\n"
            f"returncode={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )


def test_no_python_imports_from_session_store_into_build_script() -> None:
    source = BUILD_SCRIPT_ABS.read_text(encoding="utf-8")
    assert "chuk_lazarus.session_store" not in source, (
        "tools/build_clause_aligned_store.py must not import from "
        "chuk_lazarus.session_store. The coupling direction is strictly "
        "wrapper -> script, never the reverse."
    )
