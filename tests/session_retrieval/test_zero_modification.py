"""Zero-modification guarantee for the read-only primitive knowledge package.

Axis-4 ``session_retrieval`` imports the public surface of
``chuk_lazarus.inference.context.knowledge`` but MUST NEVER edit any file under
that directory. These tests fail loudly if any of
``inject.py``, ``torch_query.py``, ``torch_store.py`` have been modified
relative to HEAD. A second, git-diff-independent signal is provided by
comparing each working-tree file's sha256 against its HEAD blob hash.

Skipped when not inside a git work tree.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

# tests/session_retrieval/test_zero_modification.py -> two parents up = repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
PRIMITIVE_REL: str = "src/chuk_lazarus/inference/context/knowledge"
PINNED_FILES: tuple[str, ...] = (
    f"{PRIMITIVE_REL}/inject.py",
    f"{PRIMITIVE_REL}/torch_query.py",
    f"{PRIMITIVE_REL}/torch_store.py",
)


def _inside_git_work_tree() -> bool:
    """Return True iff ``REPO_ROOT`` is inside a git work tree."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _sha256_bytes(blob: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``blob``."""
    return hashlib.sha256(blob).hexdigest()


def _head_blob_bytes(path_rel: str) -> bytes:
    """Return the bytes of ``path_rel`` as recorded in HEAD."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path_rel}"],
        capture_output=True,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return proc.stdout


# ---------------------------------------------------------------------------
# Test body.
# ---------------------------------------------------------------------------


def test_pinned_files_exist() -> None:
    """Each pinned primitive file must exist in the working tree."""
    for rel in PINNED_FILES:
        abs_path = REPO_ROOT / rel
        assert abs_path.is_file(), f"Primitive file missing at {abs_path}"


def test_primitive_files_have_no_uncommitted_changes() -> None:
    """``git diff HEAD --`` on the three primitive files returns empty output."""
    if not _inside_git_work_tree():
        pytest.skip("Not inside a git work tree; cannot assert zero modification.")

    proc = subprocess.run(
        ["git", "diff", "HEAD", "--", *PINNED_FILES],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0 or proc.stdout != "":
        pytest.fail(
            "Read-only primitive package has uncommitted modifications.\n"
            "Axis-4 session_retrieval is a strict-mode consumer; the primitive "
            "MUST remain bit-for-bit unchanged.\n"
            f"returncode={proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )


def test_primitive_files_match_head_sha256() -> None:
    """Working-tree sha256 equals the sha256 of the HEAD blob for each file.

    This is an independent signal from ``git diff``: even if a filesystem
    timestamp or permission bit drifts, the content must hash to the HEAD
    version exactly.
    """
    if not _inside_git_work_tree():
        pytest.skip("Not inside a git work tree; cannot compare against HEAD blob.")

    for rel in PINNED_FILES:
        abs_path = REPO_ROOT / rel
        working_tree_bytes = abs_path.read_bytes()
        head_bytes = _head_blob_bytes(rel)
        assert _sha256_bytes(working_tree_bytes) == _sha256_bytes(head_bytes), (
            f"SHA-256 drift detected for {rel}. Working-tree contents differ "
            "from the committed HEAD blob, which violates the axis-4 "
            "zero-modification hard constraint."
        )
