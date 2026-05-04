"""Product decoder prior store for David.

The store keeps generation-control memory scoped to compatible model,
tokenizer, adapter, layer, task, and steering identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


SCOPE_FIELDS = (
    "model_id",
    "tokenizer_id",
    "adapter_family",
    "layer",
    "task_type",
    "steering_version",
)
OPTIONAL_SCOPE_FIELDS = (
    "model_revision",
    "adapter_config_id",
    "insertion_family",
)


class IncompatibleDecoderPriorScope(ValueError):
    """Raised when decoder prior data crosses incompatible scopes."""


@dataclass(frozen=True)
class DecoderPriorScope:
    model_id: str
    tokenizer_id: str
    adapter_family: str
    layer: int
    task_type: str
    steering_version: str
    model_revision: str = ""
    adapter_config_id: str = ""
    insertion_family: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DecoderPriorScope":
        layer = data.get("layer", data.get("kv_target_layer"))
        return cls(
            model_id=str(data["model_id"]),
            tokenizer_id=str(data["tokenizer_id"]),
            adapter_family=str(data["adapter_family"]),
            layer=int(layer),
            task_type=str(data.get("task_type", data.get("method", ""))),
            steering_version=str(data.get("steering_version", data.get("steering", ""))),
            model_revision=str(data.get("model_revision", "")),
            adapter_config_id=str(data.get("adapter_config_id", "")),
            insertion_family=str(data.get("insertion_family", "")),
        )

    def key(self) -> str:
        payload = {field_name: getattr(self, field_name) for field_name in (*SCOPE_FIELDS, *OPTIONAL_SCOPE_FIELDS)}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def assert_compatible(self, other: "DecoderPriorScope") -> None:
        mismatches = [
            field_name
            for field_name in (*SCOPE_FIELDS, *OPTIONAL_SCOPE_FIELDS)
            if getattr(self, field_name) != getattr(other, field_name)
        ]
        if mismatches:
            joined = ", ".join(mismatches)
            raise IncompatibleDecoderPriorScope(f"decoder prior scope mismatch: {joined}")


@dataclass
class DecoderPriorRecord:
    scope: DecoderPriorScope
    seed_alpha: float = 0.0
    accepted_case_count: int = 0
    successful_streak: int = 0
    temptation_events: int = 0
    steering_applications: int = 0
    rejected_case_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "DecoderPriorRecord":
        return cls(
            scope=DecoderPriorScope.from_mapping(data["scope"]),
            seed_alpha=float(data.get("seed_alpha", 0.0)),
            accepted_case_count=int(data.get("accepted_case_count", 0)),
            successful_streak=int(data.get("successful_streak", 0)),
            temptation_events=int(data.get("temptation_events", 0)),
            steering_applications=int(data.get("steering_applications", 0)),
            rejected_case_count=int(data.get("rejected_case_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_json(),
            "seed_alpha": self.seed_alpha,
            "accepted_case_count": self.accepted_case_count,
            "successful_streak": self.successful_streak,
            "temptation_events": self.temptation_events,
            "steering_applications": self.steering_applications,
            "rejected_case_count": self.rejected_case_count,
            "metadata": dict(self.metadata),
        }

    def update_stats(
        self,
        *,
        accepted: bool,
        temptation_event: bool = False,
        steering_applied: bool = False,
        alpha_delta: float = 0.05,
        decay: float = 0.95,
    ) -> None:
        self.seed_alpha = max(0.0, min(1.0, self.seed_alpha * decay + (alpha_delta if accepted else -alpha_delta)))
        if accepted:
            self.accepted_case_count += 1
            self.successful_streak += 1
        else:
            self.rejected_case_count += 1
            self.successful_streak = 0
        if temptation_event:
            self.temptation_events += 1
        if steering_applied:
            self.steering_applications += 1

    def decay_seed_alpha(self, factor: float) -> None:
        if factor < 0:
            raise ValueError("decay factor must be non-negative")
        self.seed_alpha = max(0.0, min(1.0, self.seed_alpha * factor))


class DecoderPriorProductStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, DecoderPriorRecord] = {}

    @classmethod
    def load(cls, path: str | Path) -> "DecoderPriorProductStore":
        store = cls(path)
        if not store.path.exists():
            return store
        payload = json.loads(store.path.read_text(encoding="utf-8"))
        for item in payload.get("records", []):
            record = DecoderPriorRecord.from_json(item)
            store._records[record.scope.key()] = record
        return store

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": [record.to_json() for record in sorted(self._records.values(), key=lambda item: item.scope.key())],
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def get(self, scope: DecoderPriorScope | Mapping[str, Any]) -> DecoderPriorRecord | None:
        resolved = _coerce_scope(scope)
        return self._records.get(resolved.key())

    def get_or_create(
        self,
        scope: DecoderPriorScope | Mapping[str, Any],
        *,
        seed_alpha: float = 0.0,
    ) -> DecoderPriorRecord:
        resolved = _coerce_scope(scope)
        record = self._records.get(resolved.key())
        if record is None:
            record = DecoderPriorRecord(scope=resolved, seed_alpha=max(0.0, min(1.0, seed_alpha)))
            self._records[resolved.key()] = record
        record.scope.assert_compatible(resolved)
        return record

    def update(
        self,
        scope: DecoderPriorScope | Mapping[str, Any],
        *,
        accepted: bool,
        temptation_event: bool = False,
        steering_applied: bool = False,
        alpha_delta: float = 0.05,
        decay: float = 0.95,
    ) -> DecoderPriorRecord:
        record = self.get_or_create(scope)
        record.update_stats(
            accepted=accepted,
            temptation_event=temptation_event,
            steering_applied=steering_applied,
            alpha_delta=alpha_delta,
            decay=decay,
        )
        return record

    def put(self, record: DecoderPriorRecord) -> None:
        existing = self._records.get(record.scope.key())
        if existing is not None:
            existing.scope.assert_compatible(record.scope)
        self._records[record.scope.key()] = record

    def require_compatible(
        self,
        expected: DecoderPriorScope | Mapping[str, Any],
        candidate: DecoderPriorScope | Mapping[str, Any],
    ) -> None:
        _coerce_scope(expected).assert_compatible(_coerce_scope(candidate))

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "records": [record.to_json() for record in sorted(self._records.values(), key=lambda item: item.scope.key())],
        }


def _coerce_scope(scope: DecoderPriorScope | Mapping[str, Any]) -> DecoderPriorScope:
    if isinstance(scope, DecoderPriorScope):
        return scope
    return DecoderPriorScope.from_mapping(scope)
