"""Manual-reviewed model attestations for David model startup.

The validator remains the automatic startup gate. This module adds a separate
operator-review artifact that can bind a specific validation report and adapter
scope to a deliberately narrow capability set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from chuk_lazarus.david.config import AdapterSessionMetadata
from chuk_lazarus.harness.model_config import (
    ModelConfigReportError,
    boot_metadata_from_validation_report,
)


ATTESTATION_SCHEMA_NAME = "david.manual_model_attestation"
ATTESTATION_SCHEMA_VERSION = 1
STANDARD_DECODE_CAPABILITY = "standard_decode"
SUPPORTED_CAPABILITIES = frozenset({STANDARD_DECODE_CAPABILITY})
DANGEROUS_CAPABILITIES = frozenset(
    {
        "kv_replay",
        "kv_direct",
        "residual_replay",
        "residual_direct",
        "tensor_replay",
    }
)


@dataclass(frozen=True)
class ManualModelAttestation:
    """Parsed JSON manual-review attestation."""

    schema_name: str | None
    schema_version: int | None
    validation_report_sha256: str | None
    model_identity: str | None
    tokenizer_identity: str | None
    model_revision_or_hash: str | None
    adapter_family: str | None
    selected_config_sha256: str | None
    selected_config: dict[str, Any] | None
    allowed_capabilities: tuple[str, ...]
    reviewer: str | None
    reviewed_at: str | None
    rationale: str | None
    expires_at: str | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class ManualModelAttestationResult:
    """Fail-closed result of binding an attestation to a validation report."""

    accepted: bool
    reasons: tuple[str, ...]
    attestation_path: Path | None
    validation_report_path: Path | None
    validation_report_sha256: str | None
    selected_config: dict[str, Any] | None
    allowed_capabilities: tuple[str, ...]
    reviewer: str | None
    reviewed_at: str | None
    expires_at: str | None
    model_identity: str | None
    tokenizer_identity: str | None
    model_revision_or_hash: str | None
    adapter_family: str | None
    source_validation_status: str | None
    source_auto_load_allowed: bool | None
    adapter_metadata: AdapterSessionMetadata | None

    def adapter_scope(self) -> dict[str, Any] | None:
        """Return the runtime compatibility scope when verification accepted."""

        if self.adapter_metadata is None:
            return None
        return self.adapter_metadata.scope()

    def compact_detail(self) -> str:
        if self.accepted:
            caps = ", ".join(self.allowed_capabilities)
            return f"accepted manual model attestation; capabilities={caps}"
        return "; ".join(self.reasons) if self.reasons else "attestation rejected"


def load_manual_model_attestation(path: str | Path) -> ManualModelAttestation:
    """Load a manual-review attestation JSON object from disk."""

    attestation_path = Path(path).expanduser()
    with attestation_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manual model attestation must be a JSON object: {attestation_path}")
    return manual_model_attestation_from_data(data)


def manual_model_attestation_from_data(
    data: Mapping[str, Any],
) -> ManualModelAttestation:
    """Parse attestation data without binding it to a validation report."""

    selected_config = _mapping_or_none(data.get("selected_config"))
    selected_config_hash = _first_str(
        data.get("selected_config_sha256"),
        data.get("selected_config_hash"),
    )
    return ManualModelAttestation(
        schema_name=_optional_str(data.get("schema_name")),
        schema_version=_optional_int(data.get("schema_version")),
        validation_report_sha256=_normalize_digest(
            _first_str(data.get("validation_report_sha256"), data.get("validation_report_digest"))
        ),
        model_identity=_optional_str(data.get("model_identity")),
        tokenizer_identity=_optional_str(data.get("tokenizer_identity")),
        model_revision_or_hash=_optional_str(data.get("model_revision_or_hash")),
        adapter_family=_optional_str(data.get("adapter_family")),
        selected_config_sha256=_normalize_digest(selected_config_hash),
        selected_config=dict(selected_config) if selected_config is not None else None,
        allowed_capabilities=tuple(_unique_strings(_string_list(data.get("allowed_capabilities")))),
        reviewer=_optional_str(data.get("reviewer")),
        reviewed_at=_optional_str(data.get("reviewed_at")),
        rationale=_optional_str(data.get("rationale")),
        expires_at=_optional_str(data.get("expires_at")),
        raw=dict(data),
    )


def verify_manual_model_attestation(
    *,
    attestation_path: str | Path,
    validation_report_path: str | Path,
    now: datetime | None = None,
) -> ManualModelAttestationResult:
    """Load and verify a manual attestation against a validator report file."""

    attestation_file = Path(attestation_path).expanduser()
    validation_file = Path(validation_report_path).expanduser()

    try:
        attestation = load_manual_model_attestation(attestation_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _rejected_result(
            reasons=(f"could not read attestation: {exc}",),
            attestation_path=attestation_file,
            validation_report_path=validation_file,
        )

    try:
        validation_bytes = validation_file.read_bytes()
    except OSError as exc:
        return _rejected_result(
            reasons=(f"could not read validation report: {exc}",),
            attestation=attestation,
            attestation_path=attestation_file,
            validation_report_path=validation_file,
        )

    try:
        validation_report = json.loads(validation_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _rejected_result(
            reasons=(f"validation report is not valid JSON: {exc}",),
            attestation=attestation,
            validation_report_sha256=sha256_bytes(validation_bytes),
            attestation_path=attestation_file,
            validation_report_path=validation_file,
        )

    if not isinstance(validation_report, Mapping):
        return _rejected_result(
            reasons=("validation report must be a JSON object",),
            attestation=attestation,
            validation_report_sha256=sha256_bytes(validation_bytes),
            attestation_path=attestation_file,
            validation_report_path=validation_file,
        )

    return verify_manual_model_attestation_data(
        attestation.raw,
        validation_report=validation_report,
        validation_report_sha256=sha256_bytes(validation_bytes),
        attestation_path=attestation_file,
        validation_report_path=validation_file,
        now=now,
    )


def verify_manual_model_attestation_data(
    attestation_data: Mapping[str, Any],
    *,
    validation_report: Mapping[str, Any],
    validation_report_sha256: str,
    attestation_path: str | Path | None = None,
    validation_report_path: str | Path | None = None,
    now: datetime | None = None,
) -> ManualModelAttestationResult:
    """Verify attestation data against an already loaded validation report."""

    attestation = manual_model_attestation_from_data(attestation_data)
    validation_digest = _normalize_digest(validation_report_sha256)
    reasons: list[str] = []

    if attestation.schema_name != ATTESTATION_SCHEMA_NAME:
        reasons.append(f"unsupported attestation schema {attestation.schema_name!r}")
    if attestation.schema_version != ATTESTATION_SCHEMA_VERSION:
        reasons.append(
            f"unsupported attestation schema_version {attestation.schema_version!r}"
        )
    if not attestation.validation_report_sha256:
        reasons.append("validation_report_sha256 is missing")
    elif attestation.validation_report_sha256 != validation_digest:
        reasons.append("validation_report_sha256 does not match validation report")

    reasons.extend(_review_reasons(attestation, now=now))
    reasons.extend(_capability_reasons(attestation.allowed_capabilities))

    metadata = None
    try:
        metadata = boot_metadata_from_validation_report(
            validation_report,
            require_validated_model=False,
        )
    except ModelConfigReportError as exc:
        reasons.append(f"validation report cannot be parsed: {exc}")

    selected_config = _mapping_or_none(validation_report.get("selected_config"))
    selected_config_dict = dict(selected_config) if selected_config is not None else None
    if selected_config is None:
        reasons.append("validation report selected_config is missing")
    else:
        reasons.extend(_selected_config_reasons(attestation, selected_config))

    actual_scope = _validation_scope(validation_report, metadata)
    reasons.extend(_scope_reasons(attestation, actual_scope))

    adapter_metadata = None
    if metadata is not None and metadata.adapter is not None:
        adapter_metadata = _adapter_session_metadata(metadata.adapter, actual_scope)
    elif metadata is not None:
        reasons.append("validation report adapter metadata is missing")

    accepted = not reasons
    return ManualModelAttestationResult(
        accepted=accepted,
        reasons=tuple(_unique_strings(reasons)),
        attestation_path=Path(attestation_path).expanduser() if attestation_path else None,
        validation_report_path=Path(validation_report_path).expanduser()
        if validation_report_path
        else None,
        validation_report_sha256=validation_digest,
        selected_config=selected_config_dict,
        allowed_capabilities=attestation.allowed_capabilities,
        reviewer=attestation.reviewer,
        reviewed_at=attestation.reviewed_at,
        expires_at=attestation.expires_at,
        model_identity=actual_scope.get("model_identity"),
        tokenizer_identity=actual_scope.get("tokenizer_identity"),
        model_revision_or_hash=actual_scope.get("model_revision_or_hash"),
        adapter_family=actual_scope.get("adapter_family"),
        source_validation_status=_optional_str(validation_report.get("validation_status")),
        source_auto_load_allowed=_optional_bool(validation_report.get("auto_load_allowed")),
        adapter_metadata=adapter_metadata if accepted else None,
    )


def selected_config_sha256(selected_config: Mapping[str, Any]) -> str:
    """Return the canonical SHA256 digest for a selected adapter config."""

    return sha256_text(_canonical_json(selected_config))


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase hexadecimal SHA256 digest for bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return a lowercase hexadecimal SHA256 digest for UTF-8 text."""

    return sha256_bytes(text.encode("utf-8"))


def _review_reasons(
    attestation: ManualModelAttestation,
    *,
    now: datetime | None,
) -> list[str]:
    reasons: list[str] = []
    if not attestation.reviewer:
        reasons.append("reviewer is missing")
    if not attestation.rationale:
        reasons.append("rationale is missing")

    reviewed_at = _parse_datetime(attestation.reviewed_at)
    if attestation.reviewed_at is None:
        reasons.append("reviewed_at is missing")
    elif reviewed_at is None:
        reasons.append("reviewed_at is not a valid ISO timestamp")

    if attestation.expires_at:
        expires_at = _parse_datetime(attestation.expires_at)
        if expires_at is None:
            reasons.append("expires_at is not a valid ISO timestamp")
        else:
            current_time = _normalize_datetime(now or datetime.now(timezone.utc))
            if expires_at <= current_time:
                reasons.append("attestation is expired")
    return reasons


def _capability_reasons(capabilities: Sequence[str]) -> list[str]:
    reasons: list[str] = []
    capability_set = set(capabilities)
    dangerous = sorted(capability_set & DANGEROUS_CAPABILITIES)
    if dangerous:
        reasons.append(f"dangerous capabilities are not permitted: {', '.join(dangerous)}")
    if capability_set != SUPPORTED_CAPABILITIES:
        reasons.append("allowed_capabilities must be exactly ['standard_decode'] for now")
    return reasons


def _selected_config_reasons(
    attestation: ManualModelAttestation,
    selected_config: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    actual_hash = selected_config_sha256(selected_config)
    if attestation.selected_config_sha256 is None and attestation.selected_config is None:
        reasons.append("selected_config_sha256 or selected_config is required")
    if (
        attestation.selected_config_sha256 is not None
        and attestation.selected_config_sha256 != actual_hash
    ):
        reasons.append("selected_config_sha256 does not match validation report")
    if attestation.selected_config is not None:
        attested_hash = selected_config_sha256(attestation.selected_config)
        if attested_hash != actual_hash:
            reasons.append("selected_config does not match validation report")
    return reasons


def _scope_reasons(
    attestation: ManualModelAttestation,
    actual_scope: Mapping[str, str | None],
) -> list[str]:
    reasons: list[str] = []
    expected = {
        "model_identity": attestation.model_identity,
        "tokenizer_identity": attestation.tokenizer_identity,
        "model_revision_or_hash": attestation.model_revision_or_hash,
        "adapter_family": attestation.adapter_family,
    }
    for field_name, expected_value in expected.items():
        actual_value = actual_scope.get(field_name)
        if not expected_value:
            reasons.append(f"{field_name} is missing")
        elif not actual_value:
            reasons.append(f"validation report {field_name} is missing")
        elif actual_value != expected_value:
            reasons.append(
                f"{field_name} mismatch: attestation={expected_value!r}, "
                f"validation_report={actual_value!r}"
            )
    return reasons


def _validation_scope(
    validation_report: Mapping[str, Any],
    metadata: Any,
) -> dict[str, str | None]:
    identity = _mapping_or_empty(validation_report.get("model_identity_gate"))
    summary = _mapping_or_empty(validation_report.get("source_report_summary"))
    provenance = _mapping_or_empty(validation_report.get("provenance"))
    loader_options = _mapping_or_empty(provenance.get("loader_options"))
    adapter = getattr(metadata, "adapter", None) if metadata is not None else None
    return {
        "model_identity": _first_str(
            identity.get("model_identity"),
            identity.get("model"),
            summary.get("model_identity"),
            summary.get("model"),
            getattr(adapter, "model", None),
            loader_options.get("model"),
            provenance.get("model"),
            provenance.get("model_path"),
        ),
        "tokenizer_identity": _first_str(
            identity.get("tokenizer_identity"),
            identity.get("tokenizer_id"),
            summary.get("tokenizer_identity"),
            summary.get("tokenizer_id"),
            getattr(adapter, "tokenizer_identity", None),
            loader_options.get("model"),
        ),
        "model_revision_or_hash": _first_str(
            identity.get("model_revision_or_hash"),
            identity.get("model_revision"),
            summary.get("model_revision_or_hash"),
            summary.get("model_revision"),
            getattr(adapter, "model_revision_or_hash", None),
        ),
        "adapter_family": _first_str(
            identity.get("adapter_family"),
            summary.get("adapter_family"),
            getattr(adapter, "adapter_family", None),
        ),
    }


def _adapter_session_metadata(adapter: Any, scope: Mapping[str, str | None]) -> AdapterSessionMetadata:
    return AdapterSessionMetadata(
        model_id=scope.get("model_identity") or getattr(adapter, "model", None) or "unknown-model",
        tokenizer_id=scope.get("tokenizer_identity")
        or getattr(adapter, "tokenizer_identity", None)
        or "unknown-tokenizer",
        model_revision=scope.get("model_revision_or_hash")
        or getattr(adapter, "model_revision_or_hash", None)
        or "unknown-revision",
        adapter_family=scope.get("adapter_family")
        or getattr(adapter, "adapter_family", None)
        or "unknown-adapter",
        hidden_size=getattr(adapter, "hidden_size", None),
        route_layer=getattr(adapter, "route_layer", None),
        route_query_head=getattr(adapter, "route_query_head", None),
        boundary_layer=getattr(adapter, "boundary_layer", None),
        kv_source_layer=getattr(adapter, "kv_source_layer", None),
        kv_target_layer=getattr(adapter, "kv_target_layer", None),
        insertion_family=getattr(adapter, "insertion_family", None) or "text_only",
        memory_family="david-runtime",
    )


def _rejected_result(
    *,
    reasons: Sequence[str],
    attestation: ManualModelAttestation | None = None,
    validation_report_sha256: str | None = None,
    attestation_path: Path | None = None,
    validation_report_path: Path | None = None,
) -> ManualModelAttestationResult:
    return ManualModelAttestationResult(
        accepted=False,
        reasons=tuple(_unique_strings(reasons)),
        attestation_path=attestation_path,
        validation_report_path=validation_report_path,
        validation_report_sha256=validation_report_sha256,
        selected_config=attestation.selected_config if attestation is not None else None,
        allowed_capabilities=attestation.allowed_capabilities if attestation is not None else (),
        reviewer=attestation.reviewer if attestation is not None else None,
        reviewed_at=attestation.reviewed_at if attestation is not None else None,
        expires_at=attestation.expires_at if attestation is not None else None,
        model_identity=attestation.model_identity if attestation is not None else None,
        tokenizer_identity=attestation.tokenizer_identity if attestation is not None else None,
        model_revision_or_hash=attestation.model_revision_or_hash
        if attestation is not None
        else None,
        adapter_family=attestation.adapter_family if attestation is not None else None,
        source_validation_status=None,
        source_auto_load_allowed=None,
        adapter_metadata=None,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _normalize_digest(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized.startswith("sha256:"):
        normalized = normalized[len("sha256:") :]
    return normalized or None


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return _normalize_datetime(datetime.fromisoformat(text))
    except ValueError:
        return None


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_str(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text is not None:
            return text
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _unique_strings(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
