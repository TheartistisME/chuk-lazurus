"""Safe boot metadata helpers for model config validation reports.

This module intentionally depends only on the Python standard library. It does
not import the standalone David validation scripts because those scripts are
proof tools, while this module is product startup plumbing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


VALIDATION_REPORT_SCHEMA = "lazarus.model_config_validation_report"
ACCEPTED_STATUS = "accepted"


class ModelConfigReportError(ValueError):
    """Raised when a validation report is not safe enough for boot."""


@dataclass(frozen=True)
class AdapterBootMetadata:
    """Adapter fields selected by the validator for harness startup."""

    adapter_config_id: str | None
    adapter_family: str | None
    model: str | None
    model_revision_or_hash: str | None
    tokenizer_identity: str | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    projection_aliases: dict[str, Any]
    route_layer: int | None
    route_dimension: int | None
    route_query_head: int | None
    boundary_layer: int | None
    residual_capture_layer: int | None
    kv_source_layer: int | None
    kv_target_layer: int | None
    injection_layer: int | None
    projection_producer_layer: int | None
    behavior_cache_layer: int | None
    insertion_family: str | None
    kv_layout: str | None
    candidate_role: str | None


@dataclass(frozen=True)
class ModelBootMetadata:
    """Product-safe summary of a model validation report."""

    schema_name: str | None
    schema_version: int | None
    validation_status: str | None
    confidence: str | None
    validation_level: str | None
    auto_load_allowed: bool
    harness_load_policy: str | None
    require_validated_model: bool
    adapter: AdapterBootMetadata | None
    selected_config: dict[str, Any] | None
    source_report_summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_validation_report(path: str | Path) -> dict[str, Any]:
    """Load a validator JSON report from disk."""

    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ModelConfigReportError(f"Validation report must be a JSON object: {report_path}")
    return data


def boot_metadata_from_file(
    path: str | Path, *, require_validated_model: bool = True
) -> ModelBootMetadata:
    """Load and parse a validation report into safe boot metadata."""

    return boot_metadata_from_validation_report(
        load_validation_report(path),
        require_validated_model=require_validated_model,
    )


def parse_model_validation_report(
    report: Mapping[str, Any], *, require_validated_model: bool = True
) -> ModelBootMetadata:
    """Compatibility alias for callers that want an explicit parse verb."""

    return boot_metadata_from_validation_report(
        report,
        require_validated_model=require_validated_model,
    )


def boot_metadata_from_validation_report(
    report: Mapping[str, Any], *, require_validated_model: bool = True
) -> ModelBootMetadata:
    """Parse a model validation report into boot metadata.

    Accepted reports may auto-load only when the validator explicitly allowed
    auto-load. Rejected or review-required reports raise when
    ``require_validated_model`` is true.
    """

    if not isinstance(report, Mapping):
        raise ModelConfigReportError("Validation report must be a mapping.")

    schema_name = _optional_str(report.get("schema_name"))
    if schema_name != VALIDATION_REPORT_SCHEMA:
        raise ModelConfigReportError(
            f"Unsupported model validation report schema: {schema_name!r}"
        )

    validation_status = _optional_str(report.get("validation_status"))
    confidence = _optional_str(report.get("confidence"))
    auto_load_allowed = bool(report.get("auto_load_allowed"))
    selected_config = _mapping_or_none(report.get("selected_config"))
    warnings = _string_list(report.get("warnings"))

    failure_reasons = _boot_failure_reasons(
        validation_status=validation_status,
        auto_load_allowed=auto_load_allowed,
        selected_config=selected_config,
    )
    if failure_reasons and require_validated_model:
        detail = "; ".join(failure_reasons)
        raise ModelConfigReportError(f"Model validation report is not boot-safe: {detail}")

    adapter = None
    if selected_config is not None:
        adapter = _adapter_metadata(report, selected_config)

    return ModelBootMetadata(
        schema_name=schema_name,
        schema_version=_optional_int(report.get("schema_version")),
        validation_status=validation_status,
        confidence=confidence,
        validation_level=_optional_str(report.get("validation_level")),
        auto_load_allowed=auto_load_allowed,
        harness_load_policy=_optional_str(report.get("harness_load_policy")),
        require_validated_model=bool(require_validated_model),
        adapter=adapter,
        selected_config=dict(selected_config) if selected_config is not None else None,
        source_report_summary=dict(_mapping_or_empty(report.get("source_report_summary"))),
        warnings=warnings,
        provenance=dict(_mapping_or_empty(report.get("provenance"))),
        gates={
            "report_integrity": dict(_mapping_or_empty(report.get("report_integrity"))),
            "topology_gate": dict(_mapping_or_empty(report.get("topology_gate"))),
            "model_identity_gate": dict(_mapping_or_empty(report.get("model_identity_gate"))),
            "projection_gate": dict(_mapping_or_empty(report.get("projection_gate"))),
            "behavior_gate": dict(_mapping_or_empty(report.get("behavior_gate"))),
        },
    )


def can_auto_load(report: Mapping[str, Any]) -> bool:
    """Return whether a validation report explicitly permits automatic loading."""

    metadata = boot_metadata_from_validation_report(report, require_validated_model=False)
    return bool(
        metadata.validation_status == ACCEPTED_STATUS
        and metadata.auto_load_allowed
        and metadata.adapter is not None
    )


def require_boot_safe(report: Mapping[str, Any]) -> ModelBootMetadata:
    """Parse a report and raise if it cannot be auto-loaded safely."""

    return boot_metadata_from_validation_report(report, require_validated_model=True)


def _boot_failure_reasons(
    *,
    validation_status: str | None,
    auto_load_allowed: bool,
    selected_config: Mapping[str, Any] | None,
) -> list[str]:
    reasons: list[str] = []
    if validation_status != ACCEPTED_STATUS:
        reasons.append(f"validation_status is {validation_status!r}, expected 'accepted'")
    if not auto_load_allowed:
        reasons.append("auto_load_allowed is false")
    if selected_config is None:
        reasons.append("selected_config is missing")
    else:
        for field_name in (
            "route_layer",
            "boundary_layer",
            "kv_source_layer",
            "kv_target_layer",
            "insertion_family",
        ):
            if selected_config.get(field_name) is None:
                reasons.append(f"selected_config.{field_name} is missing")
    return reasons


def _adapter_metadata(
    report: Mapping[str, Any], selected_config: Mapping[str, Any]
) -> AdapterBootMetadata:
    identity = _mapping_or_empty(report.get("model_identity_gate"))
    provenance = _mapping_or_empty(report.get("provenance"))
    loader_options = _mapping_or_empty(provenance.get("loader_options"))
    source_summary = _mapping_or_empty(report.get("source_report_summary"))
    projection_gate = _mapping_or_empty(report.get("projection_gate"))
    selected_projection = _selected_candidate_from_gate(projection_gate, selected_config)

    hidden_size = _first_int(
        identity.get("hidden_size"),
        source_summary.get("hidden_size"),
        selected_config.get("hidden_size"),
    )
    kv_target_layer = _optional_int(selected_config.get("kv_target_layer"))
    injection_layer = _optional_int(selected_config.get("injection_layer")) or kv_target_layer
    kv_source_layer = _optional_int(selected_config.get("kv_source_layer"))
    boundary_layer = _optional_int(selected_config.get("boundary_layer"))

    return AdapterBootMetadata(
        adapter_config_id=_first_str(
            selected_config.get("adapter_config_id"),
            selected_projection.get("adapter_config_id"),
        ),
        adapter_family=_first_str(
            identity.get("adapter_family"),
            source_summary.get("adapter_family"),
            selected_projection.get("adapter_family"),
        ),
        model=_first_str(
            identity.get("model"),
            source_summary.get("model"),
            source_summary.get("model_identity"),
            loader_options.get("model"),
            provenance.get("model"),
            provenance.get("model_path"),
        ),
        model_revision_or_hash=_first_str(
            identity.get("model_revision_or_hash"),
            source_summary.get("model_revision_or_hash"),
        ),
        tokenizer_identity=_first_str(
            identity.get("tokenizer_identity"),
            source_summary.get("tokenizer_identity"),
            identity.get("tokenizer_id"),
            source_summary.get("tokenizer_id"),
            loader_options.get("model"),
            provenance.get("model"),
            provenance.get("model_path"),
        ),
        hidden_size=hidden_size,
        num_attention_heads=_first_int(
            identity.get("num_attention_heads"),
            source_summary.get("num_attention_heads"),
        ),
        num_key_value_heads=_first_int(
            identity.get("num_key_value_heads"),
            source_summary.get("num_key_value_heads"),
        ),
        projection_aliases=dict(
            _first_mapping(
                selected_projection.get("projection_aliases"),
                identity.get("projection_aliases"),
                source_summary.get("projection_aliases"),
            )
        ),
        route_layer=_optional_int(selected_config.get("route_layer")),
        route_dimension=_first_int(selected_config.get("route_dimension"), hidden_size),
        route_query_head=_optional_int(selected_config.get("route_query_head")),
        boundary_layer=boundary_layer,
        residual_capture_layer=_first_int(
            selected_config.get("residual_capture_layer"),
            boundary_layer,
            kv_source_layer,
        ),
        kv_source_layer=kv_source_layer,
        kv_target_layer=kv_target_layer,
        injection_layer=injection_layer,
        projection_producer_layer=_first_int(
            selected_config.get("projection_producer_layer"),
            selected_projection.get("projection_producer_layer"),
        ),
        behavior_cache_layer=_optional_int(selected_config.get("behavior_cache_layer")),
        insertion_family=_optional_str(selected_config.get("insertion_family")),
        kv_layout=_first_str(selected_config.get("kv_layout"), selected_projection.get("kv_layout")),
        candidate_role=_optional_str(selected_config.get("candidate_role")),
    )


def _selected_candidate_from_gate(
    projection_gate: Mapping[str, Any], selected_config: Mapping[str, Any]
) -> Mapping[str, Any]:
    candidates = projection_gate.get("ranked_candidates") or projection_gate.get("candidates") or []
    if not isinstance(candidates, list):
        return {}
    source = _optional_int(selected_config.get("kv_source_layer"))
    target = _optional_int(selected_config.get("kv_target_layer"))
    role = _optional_str(selected_config.get("candidate_role"))
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        item_source = _optional_int(item.get("kv_source_layer"))
        item_target = _optional_int(item.get("kv_target_layer"))
        item_role = _optional_str(item.get("candidate_role"))
        if item_source == source and item_target == target and (role is None or item_role == role):
            return item
    return {}


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        if isinstance(value, Mapping):
            return value
    return {}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_str(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text is not None:
            return text
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _optional_int(value)
        if number is not None:
            return number
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
