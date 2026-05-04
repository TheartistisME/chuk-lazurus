from __future__ import annotations

import pytest

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.decoder import DecoderController
from chuk_lazarus.david.routing import RoutePacket
from chuk_lazarus.david.steering import (
    DecoderSteeringPolicy,
    IncompatibleSteeringScope,
    STEERING_VERSION,
)


def _route(method: str = "repo_patch", *, text: str = "", provenance: dict[str, object] | None = None) -> RoutePacket:
    return RoutePacket(
        method=method,
        selected_windows=[text] if text else [],
        memory_family="task",
        session_id="s1",
        tier="hot",
        route_reason="test",
        evidence=[{"text": text, "score": 5}] if text else [],
        token_cost=4,
        provenance=provenance or {},
    )


def _adapter() -> AdapterSessionMetadata:
    return AdapterSessionMetadata(
        model_id="gemma-e2b",
        tokenizer_id="gemma-tokenizer",
        model_revision="abc123",
        adapter_family="gemma",
        boundary_layer=11,
        kv_target_layer=12,
        insertion_family="kv_direct",
    )


def test_javascript_patch_policy_suppresses_python_dialect_tokens() -> None:
    plan = DecoderController().plan(
        route=_route(text="src/app.js const value = require('x')"),
        adapter=_adapter(),
        session_id="s1",
        prompt="Patch the JavaScript repo bug without adding Python syntax",
    )

    steering = plan.constraints["steering"]

    assert steering["task_type"] == "code_patch"
    assert steering["target_language"] == "javascript"
    assert steering["logit_lock"] is True
    assert "python_block_syntax" in steering["forbidden_token_families"]
    assert steering["alpha_bounds"] == {"min": 0.0, "max": 0.35}
    assert plan.prior_scope["steering_version"] == STEERING_VERSION
    assert plan.prior_scope["target_language"] == "javascript"


def test_python_patch_policy_suppresses_javascript_dialect_tokens() -> None:
    plan = DecoderController().plan(
        route=_route(text="tests/test_agent.py def test_agent(): pass"),
        adapter=_adapter(),
        session_id="s1",
        prompt="Fix the Python pytest failure",
    )

    steering = plan.constraints["steering"]

    assert steering["task_type"] == "code_patch"
    assert steering["target_language"] == "python"
    assert "javascript_declarations" in steering["forbidden_token_families"]
    assert "javascript_arrow_functions" in steering["forbidden_token_families"]


def test_source_dependency_policy_keeps_code_scope_without_foreign_lock_when_language_unknown() -> None:
    plan = DecoderController().plan(
        route=_route("source_dependency", text="Inspect the dependency graph for the workspace"),
        adapter=_adapter(),
        session_id="s1",
    )

    steering = plan.constraints["steering"]

    assert steering["task_type"] == "source_dependency"
    assert steering["target_language"] == "code"
    assert steering["logit_lock"] is False
    assert steering["forbidden_token_families"] == []
    assert plan.prior_scope["task_type"] == "source_dependency"


def test_steering_scope_compatibility_rejects_cross_model_and_layer_mixing() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route(text="src/app.js const value = 1"),
        adapter=adapter,
        session_id="s1",
        prompt="Patch JavaScript",
    )
    compatible = policy.prior_compatible_fields(adapter=adapter)

    policy.assert_scope_compatible(compatible, adapter=adapter)

    with pytest.raises(IncompatibleSteeringScope, match="model_id"):
        policy.assert_scope_compatible(compatible | {"model_id": "other-model"}, adapter=adapter)

    with pytest.raises(IncompatibleSteeringScope, match="layer"):
        policy.assert_scope_compatible(compatible | {"layer": 13}, adapter=adapter)
