"""Metadata-only residual sidecar replay bridge for David runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import AdapterSessionMetadata
from .materialization_replay import (
    REPLAY_STRATEGY_CAPABILITIES,
    ReplayConsumerCapabilities,
    ReplayConsumerInput,
    normalize_replay_consumer,
)
from .residual_store import ResidualReference, ResidualSidecarManifest

BRIDGE_CONTRACT_VERSION = 1
RESIDUAL_SIDECAR_STRATEGY = "residual_sidecar"
RESIDUAL_REF_KINDS = {"boundary_residual", "residual_stream"}


@dataclass(frozen=True)
class ResidualReplayBridgeDecision:
    """Fail-closed replay gate result before tensor loading is possible."""

    strategy: str
    accepted: bool
    refusal_reasons: tuple[str, ...] = ()
    adapter_scope: dict[str, Any] = field(default_factory=dict)
    sidecar_scope: dict[str, Any] = field(default_factory=dict)
    consumer: dict[str, Any] | None = None
    residual_refs: tuple[dict[str, Any], ...] = ()

    @property
    def refused(self) -> bool:
        return not self.accepted

    @property
    def reason(self) -> str:
        if self.accepted:
            return "residual_sidecar bridge accepted"
        return "; ".join(self.refusal_reasons) or "residual_sidecar bridge refused"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": BRIDGE_CONTRACT_VERSION,
            "strategy": self.strategy,
            "required_capability": REPLAY_STRATEGY_CAPABILITIES[RESIDUAL_SIDECAR_STRATEGY],
            "accepted": self.accepted,
            "refused": self.refused,
            "reason": self.reason,
            "refusal_reasons": list(self.refusal_reasons),
            "eligible_for_tensor_replay": self.accepted,
            "tensor_replay_advertised": False,
            "tensor_replay_applied": False,
            "tensors_loaded": False,
            "adapter_scope": dict(self.adapter_scope),
            "sidecar_scope": dict(self.sidecar_scope),
            "consumer": None if self.consumer is None else dict(self.consumer),
            "residual_refs": [dict(ref) for ref in self.residual_refs],
        }


def validate_residual_sidecar_bridge(
    *,
    adapter: AdapterSessionMetadata,
    sidecar: ResidualSidecarManifest,
    consumer: ReplayConsumerInput,
    memory_family: str,
    layer: int | None = None,
) -> ResidualReplayBridgeDecision:
    """Validate residual sidecar replay scope without touching tensor data.

    The bridge is deliberately stricter than broad materialization planning:
    missing scope evidence is a refusal, not an unknown to be tolerated.
    """

    normalized_consumer = normalize_replay_consumer(consumer)
    expected_layer = adapter.boundary_layer if layer is None else layer
    adapter_scope = _adapter_scope(adapter, memory_family=memory_family, layer=expected_layer)
    sidecar_scope = _sidecar_scope(sidecar, layer=_sidecar_residual_layer(sidecar))
    residual_refs = tuple(_ref_summary(ref) for ref in sidecar.refs if ref.kind in RESIDUAL_REF_KINDS)

    refusal_reasons: list[str] = []
    refusal_reasons.extend(_required_scope_refusals(adapter_scope, "adapter"))
    refusal_reasons.extend(_required_scope_refusals(sidecar_scope, "sidecar"))
    refusal_reasons.extend(_scope_mismatch_refusals(adapter_scope, sidecar_scope))
    refusal_reasons.extend(_residual_ref_refusals(sidecar, residual_refs, expected_layer))
    refusal_reasons.extend(_consumer_refusals(normalized_consumer))

    return ResidualReplayBridgeDecision(
        strategy=RESIDUAL_SIDECAR_STRATEGY,
        accepted=not refusal_reasons,
        refusal_reasons=tuple(refusal_reasons),
        adapter_scope=adapter_scope,
        sidecar_scope=sidecar_scope,
        consumer=None if normalized_consumer is None else normalized_consumer.to_json(),
        residual_refs=residual_refs,
    )


def validate_residual_sidecar_replay(
    *,
    adapter: AdapterSessionMetadata,
    sidecar: ResidualSidecarManifest,
    consumer: ReplayConsumerInput,
    memory_family: str,
    layer: int | None = None,
) -> ResidualReplayBridgeDecision:
    """Validate a residual sidecar before tensor replay by a runtime hook."""

    return validate_residual_sidecar_bridge(
        adapter=adapter,
        sidecar=sidecar,
        consumer=consumer,
        memory_family=memory_family,
        layer=layer,
    )


def residual_sidecar_metadata(
    *,
    adapter: AdapterSessionMetadata,
    sidecar: ResidualSidecarManifest,
    consumer: ReplayConsumerInput,
    memory_family: str,
    layer: int | None = None,
) -> dict[str, Any]:
    """Return accepted/refused metadata for a residual sidecar replay attempt."""

    return validate_residual_sidecar_bridge(
        adapter=adapter,
        sidecar=sidecar,
        consumer=consumer,
        memory_family=memory_family,
        layer=layer,
    ).to_metadata()


def _adapter_scope(
    adapter: AdapterSessionMetadata,
    *,
    memory_family: str,
    layer: int | None,
) -> dict[str, Any]:
    return {
        "model_id": adapter.model_id,
        "tokenizer_id": adapter.tokenizer_id,
        "model_revision": adapter.model_revision,
        "adapter_family": adapter.adapter_family,
        "insertion_family": adapter.insertion_family,
        "memory_family": memory_family,
        "layer": layer,
        "hidden_size": adapter.hidden_size,
    }


def _sidecar_scope(sidecar: ResidualSidecarManifest, *, layer: int | None) -> dict[str, Any]:
    return {
        "model_id": sidecar.model_id,
        "tokenizer_id": sidecar.tokenizer_id,
        "model_revision": sidecar.model_revision,
        "adapter_family": sidecar.adapter_family,
        "insertion_family": sidecar.insertion_family,
        "memory_family": sidecar.memory_family,
        "layer": layer,
        "hidden_size": sidecar.hidden_size,
    }


def _sidecar_residual_layer(sidecar: ResidualSidecarManifest) -> int | None:
    if sidecar.residual_layer is not None:
        return sidecar.residual_layer
    return sidecar.boundary_layer


def _required_scope_refusals(scope: dict[str, Any], label: str) -> list[str]:
    reasons: list[str] = []
    for key in (
        "model_id",
        "tokenizer_id",
        "model_revision",
        "adapter_family",
        "insertion_family",
        "memory_family",
        "layer",
        "hidden_size",
    ):
        value = scope.get(key)
        if value is None or value == "":
            reasons.append(f"{label} missing {key} evidence")
    return reasons


def _scope_mismatch_refusals(adapter_scope: dict[str, Any], sidecar_scope: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "model_id",
        "tokenizer_id",
        "model_revision",
        "adapter_family",
        "insertion_family",
        "memory_family",
        "layer",
        "hidden_size",
    ):
        expected = adapter_scope.get(key)
        actual = sidecar_scope.get(key)
        if expected is None or actual is None:
            continue
        if str(expected) != str(actual):
            reasons.append(f"{key} mismatch: sidecar={actual} adapter={expected}")
    return reasons


def _residual_ref_refusals(
    sidecar: ResidualSidecarManifest,
    residual_refs: tuple[dict[str, Any], ...],
    expected_layer: int | None,
) -> list[str]:
    if not residual_refs:
        return [f"sidecar {sidecar.artifact_id} missing residual replay refs"]
    if expected_layer is None:
        return []

    reasons: list[str] = []
    for ref in residual_refs:
        if ref["layer"] != expected_layer:
            reasons.append(
                f"sidecar {sidecar.artifact_id} ref layer mismatch: "
                f"ref={ref['layer']} adapter={expected_layer}"
            )
    return reasons


def _consumer_refusals(consumer: ReplayConsumerCapabilities | None) -> list[str]:
    if consumer is None:
        return [f"runtime replay consumer required for {RESIDUAL_SIDECAR_STRATEGY}"]

    required_capability = REPLAY_STRATEGY_CAPABILITIES[RESIDUAL_SIDECAR_STRATEGY]
    if RESIDUAL_SIDECAR_STRATEGY not in consumer.strategies and required_capability not in consumer.capabilities:
        return [f"runtime replay consumer {consumer.consumer_id} lacks {required_capability}"]
    return []


def _ref_summary(ref: ResidualReference) -> dict[str, Any]:
    return {
        "kind": ref.kind,
        "layer": ref.layer,
        "dtype": ref.dtype,
        "shape": list(ref.shape),
        "uri": ref.uri,
        "inline_value_count": _nested_inline_count(ref.inline_values),
        "metadata": dict(ref.metadata),
    }


def _nested_inline_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return sum(_nested_inline_count(item) for item in value)
    return 1 if value is not None else 0
