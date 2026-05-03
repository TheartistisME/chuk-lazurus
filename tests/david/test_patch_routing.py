from __future__ import annotations

from chuk_lazarus.david.patch_routing import (
    classify_path,
    is_protected_path,
    route_patch_targets,
)


def test_classifies_patch_targets_without_mutating_proof_rigs() -> None:
    source = classify_path("src/app/socket_handler.js")
    test = classify_path("tests/socket_handler.test.js")
    protected = classify_path("scripts/run_swebench_pro_parity.py")

    assert source.is_source is True
    assert test.is_test is True
    assert protected.is_protected is True
    assert is_protected_path("David/central router.py") is True


def test_route_patch_targets_prefers_source_and_keeps_tests_as_hints() -> None:
    files = {
        "tests/socket_handler.test.js": "session leakage postgres socket",
        "src/socket_handler.js": "function handleSocket(session) {}",
        "docs/socket.md": "socket notes",
        "scripts/run_swebench_pro_parity.py": "proof rig",
    }

    plan = route_patch_targets("Fix session leakage in the Postgres socket handler", files=files)

    assert plan.selected_paths[0] == "src/socket_handler.js"
    assert "tests/socket_handler.test.js" in plan.selected_tests
    assert "scripts/run_swebench_pro_parity.py" not in plan.selected_paths
    assert "scripts/run_swebench_pro_parity.py" in plan.rejected_paths
    assert plan.windows_by_path["src/socket_handler.js"][0]["route_source"] == "file_map"


def test_route_patch_targets_adds_triad_source_hints() -> None:
    files = {
        "src/postgres_adapter.py": "db session state",
        "src/socket_handler.py": "socket entrypoint",
        "src/session_cleanup.py": "cleanup lifecycle",
    }

    plan = route_patch_targets(
        "Fix socket session cleanup in postgres adapter",
        files=files,
        limit=1,
        source_hint_limit=0,
    )

    assert plan.selected_paths == ("src/postgres_adapter.py",)
    assert "src/socket_handler.py" in plan.triad_augmented_paths
    assert "src/session_cleanup.py" in plan.triad_augmented_paths
