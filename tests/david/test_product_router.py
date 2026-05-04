from __future__ import annotations

from typing import Any

from chuk_lazarus.david.product_router import ProductRouter, route_product


def test_product_router_adds_temporal_methodology_metadata() -> None:
    packet = route_product(
        "What was the first deadline I mentioned?",
        evidence=[{"text": "deadline was Monday", "score": 4}],
        session_id="chat-1",
    )

    assert packet.method == "temporal_recall"
    assert packet.methodology == "temporal_ordinal"
    assert packet.capability == "temporal ordinal recall"
    assert packet.proof_rig == "MRCR/chat temporal recall"
    assert packet.provenance["protected_imports"] == "not_imported"
    assert packet.ordinal_score == 1.0


def test_product_router_exposes_patch_paths_tests_and_rejections() -> None:
    packet = route_product(
        "Fix session leakage in the Postgres socket handler",
        files={
            "tests/socket_handler.test.js": "session leakage postgres socket",
            "src/socket_handler.js": "function handleSocket(session) {}",
            "scripts/run_swebench_pro_parity.py": "proof rig",
        },
        methodology="patch_target",
    )

    assert packet.methodology == "patch_target"
    assert packet.selected_paths[0] == "src/socket_handler.js"
    assert "tests/socket_handler.test.js" in packet.selected_tests
    assert "scripts/run_swebench_pro_parity.py" in packet.patch_rejected_paths
    assert any("ranked ahead of tests" in reason for reason in packet.route_reasons)
    assert packet.provenance["patch_plan"]["selected_path_count"] >= 1


def test_product_router_fails_safe_when_underlying_router_errors() -> None:
    class BrokenRouter:
        def route(self, **_: Any) -> Any:
            raise RuntimeError("offline proof router unavailable")

    packet = ProductRouter(router=BrokenRouter()).route(
        "Use dependency routing",
        evidence=[{"text": "src/app.py imports src/db.py", "score": 3}],
        methodology="dependency_source",
    )

    assert packet.methodology == "dependency_source"
    assert packet.route_reason == "fail-safe product routing used after router error"
    assert packet.provenance["router"] == "david.product_router.fail_safe"
    assert packet.selected_windows == ["src/app.py imports src/db.py"]


def test_product_router_supports_all_named_methodologies() -> None:
    expected = {
        "temporal_ordinal": "temporal_recall",
        "symbolic_chain": "symbolic_multi_hop",
        "dependency_source": "source_dependency",
        "patch_target": "repo_patch",
        "durable_chat_memory": "user_continuity",
    }

    for methodology, method in expected.items():
        packet = route_product("task", methodology=methodology)
        assert packet.methodology == methodology
        assert packet.method == method
        assert packet.proof_rig
