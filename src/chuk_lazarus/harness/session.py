"""Serializable boot-session schemas for the production harness.

The types in this module describe what the harness knows at startup. They are
metadata only: no model loading, benchmark execution, or framework imports live
here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _StringEnum(str, Enum):
    """Enum base that serializes naturally as a stable string value."""

    def __str__(self) -> str:
        return self.value


class ReviewStatus(_StringEnum):
    UNKNOWN = "unknown"
    MEASURED = "measured"
    NEEDS_REVIEW = "needs_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


class ReadinessState(_StringEnum):
    UNKNOWN = "unknown"
    READY = "ready"
    MISSING = "missing"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    NEEDS_JIT = "needs_jit"
    FAILED_CLOSED = "failed_closed"


class MemoryFamily(_StringEnum):
    USER = "user"
    TASK = "task"
    CODE = "code"
    DECODER_PRIOR = "decoder_prior"
    RESIDUAL = "residual"
    KV = "kv"


class MaterializationStrategy(_StringEnum):
    NONE = "none"
    BOUNDARY_ONLY = "boundary_only"
    FULL_RESIDUAL_STREAM = "full_residual_stream"
    KV_DIRECT = "kv_direct"
    RECAPTURE_HOT_WINDOW = "recapture_hot_window"


class WarningSeverity(_StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


@dataclass(frozen=True)
class Provenance:
    """Where a boot-time fact came from."""

    source: str
    collected_at: datetime = field(default_factory=_utc_now)
    collector: str | None = None
    revision: str | None = None
    digest: str | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class FailClosedWarning:
    """A warning that can intentionally prevent harness auto-load or recall."""

    code: str
    message: str
    severity: WarningSeverity = WarningSeverity.WARNING
    component: str | None = None
    evidence: tuple[str, ...] = ()

    @property
    def is_blocking(self) -> bool:
        return self.severity in {WarningSeverity.ERROR, WarningSeverity.BLOCKING}


@dataclass(frozen=True)
class ModelAdapterMetadata:
    """Measured model and adapter geometry needed by the boot gate."""

    model_identity: str
    tokenizer_identity: str
    adapter_family: str
    model_revision: str | None = None
    model_hash: str | None = None
    adapter_config_id: str | None = None
    hidden_size: int | None = None
    attention_head_count: int | None = None
    kv_head_count: int | None = None
    projection_aliases: dict[str, str] = field(default_factory=dict)
    layer_topology: tuple[str, ...] = ()
    route_layer_candidate: int | None = None
    route_dimension: int | None = None
    route_query_head_candidate: int | None = None
    boundary_layer_candidate: int | None = None
    residual_capture_layer_candidate: int | None = None
    kv_source_layer_candidate: int | None = None
    kv_target_layer_candidate: int | None = None
    projection_producer_layer_candidate: int | None = None
    behavior_cache_layer_candidate: int | None = None
    insertion_family: str | None = None
    kv_layout: str | None = None
    confidence: float | None = None
    review_status: ReviewStatus = ReviewStatus.UNKNOWN
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class IndexReadinessMetadata:
    """Compatibility and freshness state for a workspace memory index."""

    workspace_id: str
    state: ReadinessState = ReadinessState.UNKNOWN
    index_id: str | None = None
    schema_version: str | None = None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    adapter_config_id: str | None = None
    memory_families: tuple[MemoryFamily, ...] = ()
    built_at: datetime | None = None
    checked_at: datetime = field(default_factory=_utc_now)
    supports_boundary_residuals: bool = False
    supports_full_residual_streams: bool = False
    supports_kv_materialization: bool = False
    missing_reasons: tuple[str, ...] = ()
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class UserMemoryStatusMetadata:
    """Readiness state for durable person-in-time memory."""

    state: ReadinessState = ReadinessState.UNKNOWN
    user_id: str | None = None
    store_id: str | None = None
    latest_turn_at: datetime | None = None
    stale_memory_count: int = 0
    confirmed_preference_count: int = 0
    sensitive_boundary_count: int = 0
    missing_reasons: tuple[str, ...] = ()
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class TaskMemoryStatusMetadata:
    """Readiness state for workspace/task memory."""

    state: ReadinessState = ReadinessState.UNKNOWN
    workspace_id: str | None = None
    task_id: str | None = None
    store_id: str | None = None
    indexed_file_count: int = 0
    indexed_symbol_count: int = 0
    failing_test_count: int = 0
    prior_attempt_count: int = 0
    verification_result_count: int = 0
    missing_reasons: tuple[str, ...] = ()
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class DecoderPriorScope:
    """Scope guard for durable decoding-control memory."""

    store_id: str | None = None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    adapter_config_id: str | None = None
    layer_scope: tuple[int, ...] = ()
    steering_version: str | None = None
    task_type: str | None = None
    target_language: str | None = None
    target_dialect: str | None = None
    forbidden_token_families: tuple[str, ...] = ()
    accepted_case_count: int = 0
    successful_streak: int = 0
    temptation_event_count: int = 0
    bounded_seed_alpha: float | None = None
    cross_case_decay: float | None = None
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class MaterializationCompatibilityMetadata:
    """Whether routed memory can be safely materialized for this session."""

    compatible: bool = False
    strategy: MaterializationStrategy = MaterializationStrategy.NONE
    memory_family: MemoryFamily | None = None
    session_identity: str | None = None
    workspace_id: str | None = None
    source_model_identity: str | None = None
    target_model_identity: str | None = None
    source_tokenizer_identity: str | None = None
    target_tokenizer_identity: str | None = None
    adapter_config_id: str | None = None
    source_layer: int | None = None
    target_layer: int | None = None
    insertion_family: str | None = None
    residual_available: bool = False
    kv_ready: bool = False
    refusal_reasons: tuple[str, ...] = ()
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()


@dataclass(frozen=True)
class HarnessSession:
    """Top-level boot spine shared by terminal-agent harness components."""

    session_id: str
    created_at: datetime = field(default_factory=_utc_now)
    workspace_path: str | None = None
    memory_root: str | None = None
    model_identity: str | None = None
    tokenizer_identity: str | None = None
    validation_status: str | None = None
    selected_adapter_config: dict[str, Any] | None = None
    workspace_id: str | None = None
    user_id: str | None = None
    task_id: str | None = None
    task_type: str | None = None
    selected_methodology: str | None = None
    model_adapter: ModelAdapterMetadata | None = None
    index_readiness: IndexReadinessMetadata | None = None
    user_memory: UserMemoryStatusMetadata | None = None
    task_memory: TaskMemoryStatusMetadata | None = None
    decoder_prior_scope: DecoderPriorScope | None = None
    materialization_compatibility: MaterializationCompatibilityMetadata | None = None
    jit_required: bool = False
    jit_actions: tuple[str, ...] = ()
    provenance: Provenance | None = None
    warnings: tuple[FailClosedWarning, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fail_closed_warnings(self) -> tuple[FailClosedWarning, ...]:
        warnings = list(self.warnings)
        for component in (
            self.model_adapter,
            self.index_readiness,
            self.user_memory,
            self.task_memory,
            self.decoder_prior_scope,
            self.materialization_compatibility,
        ):
            if component is not None:
                warnings.extend(component.warnings)
        return tuple(warnings)

    @property
    def blocking_warnings(self) -> tuple[FailClosedWarning, ...]:
        return tuple(warning for warning in self.fail_closed_warnings if warning.is_blocking)

    @property
    def can_auto_load(self) -> bool:
        if self.model_adapter is None or self.blocking_warnings:
            return False
        return self.model_adapter.review_status in {
            ReviewStatus.MEASURED,
            ReviewStatus.REVIEWED,
        }

    def to_dict(self) -> dict[str, Any]:
        data = _serialize(asdict(self))
        data["fail_closed_warnings"] = _serialize(
            tuple(asdict(warning) for warning in self.fail_closed_warnings)
        )
        data["blocking_warnings"] = _serialize(
            tuple(asdict(warning) for warning in self.blocking_warnings)
        )
        return data


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value
