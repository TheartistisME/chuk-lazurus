from __future__ import annotations

from tests.david import require_attr, require_module, value_at


def _router_module():
    return require_module("chuk_lazarus.david.router")


def _assert_no_benchmark_primary_names(*values: object) -> None:
    joined = " ".join(str(value) for value in values).lower()
    for benchmark_name in ("swe", "swe-bench", "loco", "locobench", "mrcr", "ruler"):
        assert benchmark_name not in joined


def test_router_contract_exports_product_modes_without_benchmark_names() -> None:
    router = _router_module()
    modes = require_attr(router, "PRIMARY_CAPABILITY_MODES", "product router modes")
    request_type = require_attr(router, "DavidRouteRequest", "router request contract")
    packet_type = require_attr(router, "DavidRoutePacket", "router packet contract")
    evidence_type = require_attr(router, "RouteEvidence", "route evidence contract")

    assert {
        "patch_target",
        "dependency_source",
        "temporal_ordinal",
        "durable_chat_memory",
    }.issubset(set(modes))
    _assert_no_benchmark_primary_names(*modes)
    assert request_type(task="inspect").task == "inspect"
    assert evidence_type(
        evidence_id="e-1",
        source="memory:1",
        summary="summary",
        reason="reason",
        score=0.5,
    ).to_dict()["source"] == "memory:1"
    assert hasattr(packet_type, "to_dict")


def test_patch_and_dependency_modes_route_to_code_task_packets() -> None:
    router = _router_module()
    route_task = require_attr(router, "route_task", "product-callable route facade")

    patch_packet = route_task(
        "Fix the failing pytest by patching the David router facade.",
        task_type="repo_patch",
        windows=[
            {
                "id": "router-source",
                "source": "src/chuk_lazarus/david/router.py",
                "kind": "source",
                "text": "route_task handles patch targets, failing pytest, and .py file spans",
                "tokens": 24,
            },
            {
                "id": "unrelated-note",
                "source": "notes.txt",
                "text": "general planning note",
            },
        ],
        index_ready=True,
        residual_available=True,
    )

    assert value_at(patch_packet, "capability_mode") == "patch_target"
    assert value_at(patch_packet, "task_type") == "repo_patch"
    assert value_at(patch_packet, "memory_family") == "code_task"
    assert value_at(patch_packet, "tier") == "hot"
    assert value_at(patch_packet, "residual_ready") is True
    assert value_at(patch_packet, "selected_windows")[0]["id"] == "router-source"
    assert value_at(patch_packet, "evidence")
    assert value_at(patch_packet, "token_cost") > 0
    _assert_no_benchmark_primary_names(
        value_at(patch_packet, "capability_mode"),
        value_at(patch_packet, "task_type"),
        *value_at(patch_packet, "route_reasons"),
    )

    dependency_packet = route_task(
        "Find the source dependency path for LocalCodingToolRunner imports.",
        capability_mode="source_dependency_reasoning",
        windows=[
            {
                "id": "tool-runner",
                "source": "src/chuk_lazarus/david/runtime.py",
                "kind": "source",
                "text": "LocalCodingToolRunner imports Path and exposes dependency spans",
                "tokens": 18,
            }
        ],
        index_ready=True,
        kv_available=True,
    )

    assert value_at(dependency_packet, "capability_mode") == "dependency_source"
    assert value_at(dependency_packet, "memory_family") == "code_task"
    assert value_at(dependency_packet, "tier") == "hot"
    assert value_at(dependency_packet, "kv_ready") is True
    assert value_at(dependency_packet, "selected_windows")[0]["id"] == "tool-runner"
    assert value_at(dependency_packet, "scores")["lexical_score"] > 0
    _assert_no_benchmark_primary_names(value_at(dependency_packet, "capability_mode"))


def test_temporal_and_user_memory_modes_route_to_chat_user_memory() -> None:
    router = _router_module()
    route_task = require_attr(router, "route_task", "product-callable route facade")

    temporal_packet = route_task(
        "What deadline did I mention yesterday?",
        capability_mode="temporal_recall",
        windows=[
            {
                "id": "turn-3",
                "source": "chat:turn-3",
                "kind": "chat_turn",
                "memory_family": "chat_user_memory",
                "text": "Yesterday the user set a Friday deadline for the router slice.",
                "timestamp": "2026-05-03T13:00:00+08:00",
                "tokens": 16,
            }
        ],
        index_ready=True,
    )

    assert value_at(temporal_packet, "capability_mode") == "temporal_ordinal"
    assert value_at(temporal_packet, "memory_family") == "chat_user_memory"
    assert value_at(temporal_packet, "tier") == "warm"
    assert value_at(temporal_packet, "scores")["ordinal_score"] > 0
    assert value_at(temporal_packet, "selected_windows")[0]["id"] == "turn-3"

    user_packet = route_task(
        "Remember my preference for focused QA summaries.",
        task_type="user_continuity",
        windows=[
            {
                "id": "preference-1",
                "source": "user:preference",
                "kind": "user_memory",
                "memory_family": "chat_user_memory",
                "text": "The user prefers focused QA summaries and concise handoffs.",
                "timestamp": "2026-05-03T14:00:00+08:00",
                "tokens": 14,
            }
        ],
        index_ready=True,
    )

    assert value_at(user_packet, "capability_mode") == "durable_chat_memory"
    assert value_at(user_packet, "task_type") == "user_continuity"
    assert value_at(user_packet, "memory_family") == "chat_user_memory"
    assert value_at(user_packet, "selected_windows")[0]["id"] == "preference-1"
    assert value_at(user_packet, "scores")["recency_score"] > 0
    _assert_no_benchmark_primary_names(
        value_at(temporal_packet, "capability_mode"),
        value_at(user_packet, "capability_mode"),
    )


def test_no_index_jit_readiness_is_stable_and_deterministic() -> None:
    router = _router_module()
    Request = require_attr(router, "DavidRouteRequest", "router request contract")
    route = require_attr(router, "route", "module-level route facade")

    request = Request(
        task="Fix a parser bug without a prepared memory index.",
        capability_mode="patch_target",
        index_ready=False,
        jit_allowed=True,
    )
    first = route(request)
    second = route(request)

    assert first.to_dict() == second.to_dict()
    assert value_at(first, "tier") == "cold"
    assert value_at(first, "readiness_status") == "jit_required"
    assert value_at(first, "index_ready") is False
    assert value_at(first, "jit_required") is True
    assert value_at(first, "jit_ready") is True
    assert value_at(first, "selected_windows") == ()
    assert value_at(first, "evidence") == ()

    ready_packet = route(
        Request(
            task="Fix a parser bug with a prepared index.",
            capability_mode="patch_target",
            index_ready=True,
            jit_allowed=True,
        )
    )

    assert value_at(ready_packet, "readiness_status") == "ready"
    assert value_at(ready_packet, "jit_required") is False
    assert value_at(ready_packet, "jit_ready") is False
