from __future__ import annotations

import pytest

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.david.decoder import DecoderController
from chuk_lazarus.david.routing import RoutePacket
from chuk_lazarus.david.steering import (
    DecoderSteeringLogitsProcessor,
    DecoderSteeringPolicy,
    EmptyForbiddenTokenSet,
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


class FakeTokenizer:
    tokenizer_id = "gemma-tokenizer"

    token_ids = {
        "def ": [2],
        "class ": [3],
        "self": [4],
        "None": [5],
        "True": [6],
        "False": [7],
        "const ": [8],
        "let ": [9],
        "=>": [10],
        "console": [11],
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return self.token_ids.get(text, [])


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


def test_javascript_logits_processor_penalizes_forbidden_python_token_ids() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route(text="src/app.js const value = 1"),
        adapter=adapter,
        session_id="s1",
        prompt="Patch JavaScript",
    )
    processor = DecoderSteeringLogitsProcessor.from_policy(
        policy=policy,
        adapter=adapter,
        tokenizer=FakeTokenizer(),
        alpha=0.2,
        logit_penalty=10.0,
    )

    scores = [[0.0 for _ in range(12)]]
    steered = processor(input_ids=[[1]], scores=scores)

    assert steered[0][2] == -2.0
    assert steered[0][5] == -2.0
    assert steered[0][8] == 0.0
    assert scores[0][2] == 0.0
    assert processor.spec.scope()["tokenizer_id"] == "gemma-tokenizer"
    assert processor.spec.effective_penalty == 2.0


def test_logits_processor_supports_explicit_forbidden_token_ids() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route("verify", text="Check generated output"),
        adapter=adapter,
        session_id="s1",
    )
    processor = DecoderSteeringLogitsProcessor.from_policy(
        policy=policy,
        adapter=adapter,
        tokenizer=FakeTokenizer(),
        alpha=0.1,
        logit_penalty=5.0,
        forbidden_token_ids=(1, 4),
    )

    steered = processor(input_ids=[[0]], scores=[[0.0 for _ in range(6)]])

    assert processor.forbidden_token_ids == (1, 4)
    assert steered[0][1] == -0.5
    assert steered[0][4] == -0.5


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


def test_alpha_is_clamped_to_policy_bounds() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route(text="tests/test_agent.py def test_agent(): pass"),
        adapter=adapter,
        session_id="s1",
        prompt="Fix Python",
    )

    processor = DecoderSteeringLogitsProcessor.from_policy(
        policy=policy,
        adapter=adapter,
        tokenizer=FakeTokenizer(),
        alpha=99.0,
        logit_penalty=10.0,
    )

    assert processor.spec.alpha == 0.35
    assert processor([[1]], [[0.0 for _ in range(12)]])[0][8] == -3.5


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


def test_logits_processor_rejects_tokenizer_scope_mismatch() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route(text="src/app.js const value = 1"),
        adapter=adapter,
        session_id="s1",
        prompt="Patch JavaScript",
    )

    class WrongTokenizer(FakeTokenizer):
        tokenizer_id = "other-tokenizer"

    with pytest.raises(IncompatibleSteeringScope, match="tokenizer_id"):
        DecoderSteeringLogitsProcessor.from_policy(
            policy=policy,
            adapter=adapter,
            tokenizer=WrongTokenizer(),
        )

    with pytest.raises(IncompatibleSteeringScope, match="model_id"):
        DecoderSteeringLogitsProcessor.from_policy(
            policy=policy,
            adapter=adapter,
            tokenizer=FakeTokenizer(),
            scope=policy.prior_compatible_fields(adapter=adapter) | {"model_id": "other-model"},
        )


def test_logits_processor_fails_closed_when_lock_has_no_token_ids() -> None:
    adapter = _adapter()
    policy = DecoderSteeringPolicy.for_route(
        route=_route(text="src/app.js const value = 1"),
        adapter=adapter,
        session_id="s1",
        prompt="Patch JavaScript",
    )

    class EmptyTokenizer(FakeTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return []

    with pytest.raises(EmptyForbiddenTokenSet):
        DecoderSteeringLogitsProcessor.from_policy(
            policy=policy,
            adapter=adapter,
            tokenizer=EmptyTokenizer(),
        )
