from __future__ import annotations

from typing import Any

from chuk_lazarus.david.product_router import ProductRouter, route_product
from chuk_lazarus.david.source_index import SourceFileRecord, SourceIndexManifest


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


def test_product_router_carries_full_adapter_contract_inputs() -> None:
    packet = route_product(
        "Trace dependency path",
        methodology="dependency_source",
        evidence=[{"text": "src/app.py imports src/db.py", "score": 4}],
        adapter_metadata={"model_family": "gemma-e2b", "route_layer": "12"},
        index_readiness_metadata={"index_id": "idx-1", "status": "ready"},
        apollo_residual_readiness={"status": "ready", "manifest_id": "apollo-1"},
        path_hints=["src/app.py"],
        identifiers=["AppService"],
        ordinal=1,
        decode_constraints={"valid_paths": ["src/app.py"]},
        verification_expectations=["right file selected"],
        writeback_targets=["code_task_memory"],
    )

    assert packet.adapter_metadata["model_family"] == "gemma-e2b"
    assert packet.index_readiness_metadata["status"] == "ready"
    assert packet.apollo_residual_readiness["manifest_id"] == "apollo-1"
    assert packet.path_hints == ["src/app.py"]
    assert packet.identifiers == ["AppService"]
    assert packet.ordinal == 1
    assert packet.decode_constraints == {"valid_paths": ["src/app.py"]}
    assert packet.verification_expectations == ["right file selected"]
    assert packet.writeback_targets == ["code_task_memory"]
    assert packet.provenance["product_inputs"]["adapter_metadata"]["route_layer"] == "12"


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


def test_product_router_uses_source_index_records_as_patch_hints() -> None:
    manifest = SourceIndexManifest(
        workspace_root="/repo",
        adapter_scope={"model_id": "offline"},
        files=[
            SourceFileRecord(
                path="src/payments/stripe_gateway.py",
                size_bytes=120,
                sha256="abc",
                language="python",
                symbols=["StripeGateway", "refund_payment"],
                import_tokens=["decimal", "requests"],
            ),
            SourceFileRecord(
                path="tests/payments/test_stripe_gateway.py",
                size_bytes=90,
                sha256="def",
                language="python",
                symbols=["test_refund_payment"],
                import_tokens=["pytest"],
            ),
            SourceFileRecord(
                path="scripts/run_swebench_pro_parity.py",
                size_bytes=30,
                sha256="ghi",
                language="python",
                symbols=["proof_rig"],
                import_tokens=[],
            ),
        ],
        pruned_dirs=[],
    )

    packet = route_product(
        "Fix Stripe refund handling in the payment gateway",
        methodology="patch_target",
        source_index=manifest,
    )

    assert packet.selected_paths[0] == "src/payments/stripe_gateway.py"
    assert "tests/payments/test_stripe_gateway.py" in packet.selected_tests
    assert packet.source_index_paths == [
        "src/payments/stripe_gateway.py",
        "tests/payments/test_stripe_gateway.py",
    ]
    assert "scripts/run_swebench_pro_parity.py" not in packet.source_index_paths
    assert "StripeGateway" in packet.selected_symbols
    assert "requests" in packet.import_tokens
    assert any("imports decimal" in hint for hint in packet.dependency_source_hints)
    assert any(item["kind"] == "source_index_record" for item in packet.evidence)
    assert packet.provenance["source_index"]["record_count"] == 2
    assert "source index records enriched routing evidence and path hints" in packet.route_reasons


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
