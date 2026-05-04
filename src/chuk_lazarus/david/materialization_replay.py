"""Runtime replay capability contract for materialization plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import AdapterSessionMetadata


REPLAY_CONTRACT_VERSION = 1
REPLAY_STRATEGY_CAPABILITIES = {
    "kv_sidecar": "materialization.replay.kv_cache.v1",
    "residual_sidecar": "materialization.replay.residual_stream.v1",
}


@dataclass(frozen=True)
class ReplayConsumerCapabilities:
    """Declared capabilities for a backend/hook that can consume replay plans."""

    consumer_id: str
    strategies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    model_id: str | None = None
    tokenizer_id: str | None = None
    model_revision: str | None = None
    adapter_family: str | None = None
    insertion_families: tuple[str, ...] = ()
    memory_families: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ReplayConsumerCapabilities":
        return cls(
            consumer_id=str(payload.get("consumer_id") or payload.get("id") or "anonymous-replay-consumer"),
            strategies=_string_tuple(payload.get("strategies")),
            capabilities=_string_tuple(payload.get("capabilities")),
            model_id=_optional_string(payload.get("model_id")),
            tokenizer_id=_optional_string(payload.get("tokenizer_id")),
            model_revision=_optional_string(payload.get("model_revision")),
            adapter_family=_optional_string(payload.get("adapter_family")),
            insertion_families=_string_tuple(payload.get("insertion_families")),
            memory_families=_string_tuple(payload.get("memory_families")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "strategies": list(self.strategies),
            "capabilities": list(self.capabilities),
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_revision": self.model_revision,
            "adapter_family": self.adapter_family,
            "insertion_families": list(self.insertion_families),
            "memory_families": list(self.memory_families),
            "metadata": dict(self.metadata),
        }


ReplayConsumerInput = ReplayConsumerCapabilities | Mapping[str, Any] | None
ReplayPlanInput = Mapping[str, Any] | None


def normalize_replay_consumer(consumer: ReplayConsumerInput) -> ReplayConsumerCapabilities | None:
    if consumer is None:
        return None
    if isinstance(consumer, ReplayConsumerCapabilities):
        return consumer
    return ReplayConsumerCapabilities.from_mapping(consumer)


def replay_generation_metadata(
    materialization_plan: ReplayPlanInput = None,
    replay_consumer: ReplayConsumerInput = None,
    *,
    backend_name: str,
    supports_tensor_replay: bool = False,
) -> dict[str, Any] | None:
    """Summarize a backend's fail-closed replay decision for generation.

    This is metadata only. Backends must not claim tensor replay unless they
    explicitly pass supports_tensor_replay after installing a real consumer hook.
    """

    if materialization_plan is None and replay_consumer is None:
        return None

    consumer = normalize_replay_consumer(replay_consumer)
    if materialization_plan is not None and not isinstance(materialization_plan, Mapping):
        return {
            "version": REPLAY_CONTRACT_VERSION,
            "backend": backend_name,
            "attempted": True,
            "refused": True,
            "ignored": True,
            "applied": False,
            "tensor_replay": False,
            "reason": "invalid materialization replay plan",
            "refusal_reasons": ["invalid materialization replay plan"],
            "consumer": None if consumer is None else consumer.to_json(),
        }

    plan = dict(materialization_plan or {})
    runtime_replay = plan.get("runtime_replay")
    runtime_contract = dict(runtime_replay) if isinstance(runtime_replay, Mapping) else {}
    requested_strategy = str(plan.get("requested_strategy") or runtime_contract.get("strategy") or plan.get("strategy") or "none")
    strategy = str(plan.get("strategy") or requested_strategy)
    memory_family = _optional_string(plan.get("memory_family") or runtime_contract.get("memory_family"))
    required_capability = _optional_string(
        runtime_contract.get("required_capability") or REPLAY_STRATEGY_CAPABILITIES.get(requested_strategy)
    )
    requires_runtime_replay = bool(
        plan.get("requires_runtime_replay")
        or runtime_contract.get("requires_runtime_replay")
        or required_capability
    )

    refusal_reasons: list[str] = []
    if bool(plan.get("refused")):
        reason = _optional_string(plan.get("reason")) or "upstream materialization refused"
        refusal_reasons.append(f"upstream materialization refused: {reason}")
    if requires_runtime_replay:
        if consumer is None:
            refusal_reasons.append(f"runtime replay consumer required for {requested_strategy}")
        else:
            refusal_reasons.extend(
                _consumer_contract_refusals(
                    requested_strategy,
                    required_capability=required_capability,
                    adapter_scope=runtime_contract.get("adapter_scope"),
                    memory_family=memory_family,
                    consumer=consumer,
                )
            )
        if not supports_tensor_replay:
            refusal_reasons.append(f"{backend_name} has no tensor replay hook for {requested_strategy}")

    applied = requires_runtime_replay and supports_tensor_replay and not refusal_reasons
    ignored = not applied
    reason = "runtime replay applied" if applied else ("; ".join(refusal_reasons) if refusal_reasons else "no runtime replay required")
    return {
        "version": REPLAY_CONTRACT_VERSION,
        "backend": backend_name,
        "attempted": materialization_plan is not None,
        "strategy": strategy,
        "requested_strategy": requested_strategy,
        "requires_runtime_replay": requires_runtime_replay,
        "required_capability": required_capability,
        "refused": bool(refusal_reasons),
        "ignored": ignored,
        "applied": applied,
        "tensor_replay": applied,
        "reason": reason,
        "refusal_reasons": refusal_reasons,
        "consumer": None if consumer is None else consumer.to_json(),
        "plan": {
            "version": plan.get("version"),
            "memory_family": memory_family,
            "tier": plan.get("tier"),
            "sidecar_count": len(plan.get("sidecars") or ()),
            "replay_ref_count": len(plan.get("replay_refs") or ()),
            "session_id": plan.get("session_id"),
        },
    }


def replay_contract_for_strategy(
    strategy: str,
    *,
    adapter: AdapterSessionMetadata,
    memory_family: str,
    consumer: ReplayConsumerCapabilities | None,
) -> dict[str, Any]:
    required_capability = REPLAY_STRATEGY_CAPABILITIES.get(strategy)
    return {
        "version": REPLAY_CONTRACT_VERSION,
        "strategy": strategy,
        "requires_runtime_replay": required_capability is not None,
        "required_capability": required_capability,
        "adapter_scope": adapter.scope(),
        "memory_family": memory_family,
        "consumer": None if consumer is None else consumer.to_json(),
    }


def replay_consumer_refusals(
    strategy: str,
    *,
    adapter: AdapterSessionMetadata,
    memory_family: str,
    consumer: ReplayConsumerCapabilities | None,
) -> list[str]:
    required_capability = REPLAY_STRATEGY_CAPABILITIES.get(strategy)
    if required_capability is None:
        return []
    if consumer is None:
        return [f"runtime replay consumer required for {strategy}"]

    reasons: list[str] = []
    if strategy not in consumer.strategies and required_capability not in consumer.capabilities:
        reasons.append(
            f"runtime replay consumer {consumer.consumer_id} lacks {required_capability} for {strategy}"
        )

    comparisons = {
        "model_id": (consumer.model_id, adapter.model_id),
        "tokenizer_id": (consumer.tokenizer_id, adapter.tokenizer_id),
        "model_revision": (consumer.model_revision, adapter.model_revision),
        "adapter_family": (consumer.adapter_family, adapter.adapter_family),
    }
    for key, (actual, expected) in comparisons.items():
        if actual is not None and str(actual) != str(expected):
            reasons.append(f"runtime replay consumer {key} mismatch: consumer={actual} adapter={expected}")
    if consumer.insertion_families and adapter.insertion_family not in consumer.insertion_families:
        reasons.append(
            "runtime replay consumer insertion_family mismatch: "
            f"consumer={list(consumer.insertion_families)} adapter={adapter.insertion_family}"
        )
    if consumer.memory_families and memory_family not in consumer.memory_families:
        reasons.append(
            "runtime replay consumer memory_family mismatch: "
            f"consumer={list(consumer.memory_families)} route={memory_family}"
        )
    return reasons


def _consumer_contract_refusals(
    strategy: str,
    *,
    required_capability: str | None,
    adapter_scope: Any,
    memory_family: str | None,
    consumer: ReplayConsumerCapabilities,
) -> list[str]:
    reasons: list[str] = []
    if required_capability is not None and strategy not in consumer.strategies and required_capability not in consumer.capabilities:
        reasons.append(
            f"runtime replay consumer {consumer.consumer_id} lacks {required_capability} for {strategy}"
        )
    if isinstance(adapter_scope, Mapping):
        comparisons = {
            "model_id": (consumer.model_id, adapter_scope.get("model_id")),
            "tokenizer_id": (consumer.tokenizer_id, adapter_scope.get("tokenizer_id")),
            "model_revision": (consumer.model_revision, adapter_scope.get("model_revision")),
            "adapter_family": (consumer.adapter_family, adapter_scope.get("adapter_family")),
        }
        for key, (actual, expected) in comparisons.items():
            if actual is not None and expected is not None and str(actual) != str(expected):
                reasons.append(f"runtime replay consumer {key} mismatch: consumer={actual} adapter={expected}")
        insertion_family = adapter_scope.get("insertion_family")
        if (
            insertion_family is not None
            and consumer.insertion_families
            and str(insertion_family) not in consumer.insertion_families
        ):
            reasons.append(
                "runtime replay consumer insertion_family mismatch: "
                f"consumer={list(consumer.insertion_families)} adapter={insertion_family}"
            )
    if memory_family is not None and consumer.memory_families and memory_family not in consumer.memory_families:
        reasons.append(
            "runtime replay consumer memory_family mismatch: "
            f"consumer={list(consumer.memory_families)} route={memory_family}"
        )
    return reasons


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
