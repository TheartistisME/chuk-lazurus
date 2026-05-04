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


def normalize_replay_consumer(consumer: ReplayConsumerInput) -> ReplayConsumerCapabilities | None:
    if consumer is None:
        return None
    if isinstance(consumer, ReplayConsumerCapabilities):
        return consumer
    return ReplayConsumerCapabilities.from_mapping(consumer)


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
