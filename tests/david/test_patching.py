from __future__ import annotations

from pathlib import Path

import pytest

from chuk_lazarus.david.patching import apply_patch_candidate, validate_patch_candidate
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


def test_strict_search_replace_patch_applies_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")
    patch = """src/example.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
"""

    diagnostic = apply_patch_candidate(tmp_path, patch)

    assert diagnostic.applied is True
    assert diagnostic.changed_paths == ("src/example.py",)
    assert target.read_text(encoding="utf-8") == "VALUE = 2\n"


def test_strict_search_replace_patch_is_transactional_on_later_failure(tmp_path: Path) -> None:
    first = tmp_path / "src" / "first.py"
    second = tmp_path / "src" / "second.py"
    first.parent.mkdir()
    first.write_text("VALUE = 1\n", encoding="utf-8")
    second.write_text("NAME = 'old'\n", encoding="utf-8")
    patch = """src/first.py
<<<< SEARCH
VALUE = 1
==== REPLACE
VALUE = 2
>>>>
src/second.py
<<<< SEARCH
NAME = 'missing'
==== REPLACE
NAME = 'new'
>>>>
"""

    diagnostic = apply_patch_candidate(tmp_path, patch)

    assert diagnostic.applied is False
    assert diagnostic.changed_paths == ()
    assert any("SEARCH text not found" in failure for failure in diagnostic.failures)
    assert first.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert second.read_text(encoding="utf-8") == "NAME = 'old'\n"


def test_patch_validation_rejects_protected_proof_rig_path() -> None:
    patch = """scripts/run_swebench_pro_parity.py
<<<< SEARCH
old
==== REPLACE
new
>>>>
"""

    diagnostic = validate_patch_candidate(patch)

    assert diagnostic.applied is False
    assert diagnostic.protected_paths == ("scripts/run_swebench_pro_parity.py",)
    assert any("protected proof-rig" in failure for failure in diagnostic.failures)


def test_apply_patch_candidate_rejects_empty_patch_content(tmp_path: Path) -> None:
    diagnostic = apply_patch_candidate(tmp_path, "  \n")

    assert diagnostic.applied is False
    assert diagnostic.failures == ("empty patch content",)


def test_unified_diff_patch_applies_and_rejects_nested_protocol_markers(tmp_path: Path) -> None:
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n", encoding="utf-8")
    patch = """--- a/src/example.py
+++ b/src/example.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 3
"""

    diagnostic = apply_patch_candidate(tmp_path, patch)

    assert diagnostic.applied is True
    assert target.read_text(encoding="utf-8") == "VALUE = 3\n"

    rejected = validate_patch_candidate(
        """--- a/src/example.py
+++ b/src/example.py
@@ -1 +1 @@
-VALUE = 3
+<<<< SEARCH
"""
    )

    assert rejected.applied is False
    assert any("nested protocol marker" in failure for failure in rejected.failures)
