"""Boot orchestration for the open model harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4


try:  # Prefer the shared session contract when another harness slice provides it.
    from . import session as _session_types
    from .session import HarnessSession as _SharedHarnessSession
except ImportError:  # pragma: no cover - exercised only until session.py exists.
    _session_types = None
    _SharedHarnessSession = None


try:  # Model config helpers are optional while the harness slices land in parallel.
    from . import model_config as _model_config
except ImportError:  # pragma: no cover - exercised only until model_config.py exists.
    _model_config = None


@dataclass(slots=True)
class _LocalHarnessSession:
    model_path: str
    workspace_path: str
    validation_report_path: str | None
    require_validated_model: bool
    model_report: Any | None = None
    validation: Any | None = None
    model_validated: bool = False
    user_memory_root: str | None = None
    workspace_memory_root: str | None = None
    user_memory_ready: bool = False
    workspace_memory_ready: bool = False
    model_index_ready: bool = False
    workspace_index_ready: bool = False
    jit_required: bool = False
    jit_actions: list[str] = field(default_factory=list)
    boot_errors: list[str] = field(default_factory=list)
    boot_warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


HarnessSession = _SharedHarnessSession or _LocalHarnessSession


def boot_harness(
    model_path: str,
    workspace_path: str,
    validation_report_path: str | None = None,
    require_validated_model: bool = True,
) -> HarnessSession:
    """Prepare a harness session and fail closed when validation is required.

    Boot only orchestrates startup checks. It does not select a benchmark,
    import the central router, or mutate memory/index artifacts.
    """

    model_root = Path(model_path).expanduser()
    workspace_root = Path(workspace_path).expanduser()
    report_path = Path(validation_report_path).expanduser() if validation_report_path else None

    model_report = _load_model_report(model_root, report_path, require_validated_model)
    validation = model_report
    model_validated = _is_validation_success(validation)

    boot_errors: list[str] = []
    boot_warnings: list[str] = []
    if require_validated_model and not model_validated:
        boot_errors.append("model_config_validation_required")
    elif validation is None:
        boot_warnings.append("model_config_validation_unavailable")

    user_memory_root = _memory_root("CHUK_LAZARUS_USER_MEMORY_ROOT", Path.home() / ".chuk_lazarus" / "user_memory")
    workspace_memory_root = _memory_root("CHUK_LAZARUS_WORKSPACE_MEMORY_ROOT", workspace_root / ".chuk_lazarus" / "task_memory")

    user_memory_ready = user_memory_root.exists() and user_memory_root.is_dir()
    workspace_memory_ready = workspace_memory_root.exists() and workspace_memory_root.is_dir()
    model_index_ready = _index_ready(model_root / ".chuk_lazarus" / "model_index", model_report)
    workspace_index = workspace_memory_root / "index"
    workspace_index_ready = _index_ready(workspace_index, model_report)

    jit_actions: list[str] = []
    if not workspace_memory_ready:
        jit_actions.append("create_workspace_task_memory_root")
    if not workspace_index.exists():
        jit_actions.append("jit_index_workspace")
    elif not workspace_index_ready:
        jit_actions.append("refresh_workspace_index_for_model")

    jit_required = bool(jit_actions)
    provenance = {
        "model_path": str(model_root),
        "workspace_path": str(workspace_root),
        "validation_report_path": str(report_path) if report_path else None,
        "model_config_helpers_available": _model_config is not None,
        "memory_families": {
            "user": str(user_memory_root),
            "workspace_task": str(workspace_memory_root),
        },
    }

    session_payload = {
        "model_path": str(model_root),
        "workspace_path": str(workspace_root),
        "memory_root": str(workspace_memory_root.parent),
        "validation_report_path": str(report_path) if report_path else None,
        "require_validated_model": require_validated_model,
        "model_report": model_report,
        "validation": validation,
        "model_validated": model_validated,
        "user_memory_root": str(user_memory_root),
        "workspace_memory_root": str(workspace_memory_root),
        "user_memory_ready": user_memory_ready,
        "workspace_memory_ready": workspace_memory_ready,
        "model_index_ready": model_index_ready,
        "workspace_index_ready": workspace_index_ready,
        "jit_required": jit_required,
        "jit_actions": jit_actions,
        "planned_actions": jit_actions,
        "boot_errors": boot_errors,
        "boot_warnings": boot_warnings,
        "provenance": provenance,
    }

    if boot_errors:
        raise RuntimeError(f"harness boot failed: {', '.join(boot_errors)}")

    return _build_session(session_payload)


def _load_model_report(
    model_root: Path, report_path: Path | None, require_validated_model: bool
) -> Any | None:
    if _model_config is None:
        return None

    if report_path is not None:
        for name in ("boot_metadata_from_file", "require_boot_safe"):
            helper = getattr(_model_config, name, None)
            if callable(helper):
                try:
                    return helper(report_path, require_validated_model=require_validated_model)
                except TypeError:
                    return helper(report_path)

    return _call_model_config_helper(
        ("get_model_config", "scan_model_config", "load_model_report", "read_model_report"),
        model_root,
        report_path,
    )


def _call_model_config_helper(names: tuple[str, ...], *args: Any) -> Any | None:
    if _model_config is None:
        return None

    for name in names:
        helper = getattr(_model_config, name, None)
        if callable(helper):
            return _call_with_supported_arity(helper, args)
    return None


def _call_with_supported_arity(helper: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    last_type_error: TypeError | None = None
    for arity in range(len(args), -1, -1):
        try:
            return helper(*args[:arity])
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    return helper()


def _is_validation_success(validation: Any | None) -> bool:
    if validation is None:
        return False
    if isinstance(validation, bool):
        return validation
    if isinstance(validation, Mapping):
        for key in (
            "valid",
            "validated",
            "ok",
            "safe_to_load",
            "auto_load",
            "auto_load_allowed",
        ):
            if key in validation:
                return bool(validation[key])
        return False
    validation_status = getattr(validation, "validation_status", None)
    auto_load_allowed = getattr(validation, "auto_load_allowed", None)
    adapter = getattr(validation, "adapter", None)
    if validation_status == "accepted" and auto_load_allowed and adapter is not None:
        return True
    for attr in ("valid", "validated", "ok", "safe_to_load", "auto_load", "auto_load_allowed"):
        if hasattr(validation, attr):
            return bool(getattr(validation, attr))
    return False


def _memory_root(env_name: str, default: Path) -> Path:
    import os

    configured = os.environ.get(env_name)
    return Path(configured).expanduser() if configured else default


def _index_ready(index_root: Path, model_report: Any | None) -> bool:
    if not index_root.exists() or not index_root.is_dir():
        return False

    manifest = _load_index_manifest(index_root)
    if manifest is None:
        return False

    expected_model_id = _value_from_report(model_report, "model_identity", "model_id", "identity")
    indexed_model_id = manifest.get("model_identity") or manifest.get("model_id") or manifest.get("identity")
    if expected_model_id and not indexed_model_id:
        return False
    if expected_model_id and indexed_model_id and expected_model_id != indexed_model_id:
        return False

    expected_tokenizer = _value_from_report(model_report, "tokenizer_identity", "tokenizer_id")
    indexed_tokenizer = manifest.get("tokenizer_identity") or manifest.get("tokenizer_id")
    if expected_tokenizer and not indexed_tokenizer:
        return False
    if expected_tokenizer and indexed_tokenizer and expected_tokenizer != indexed_tokenizer:
        return False

    expected_adapter_config_id = _adapter_config_id(model_report)
    indexed_adapter_config_id = manifest.get("adapter_config_id") or manifest.get("adapter_config")
    if (
        expected_adapter_config_id
        and indexed_adapter_config_id
        and expected_adapter_config_id != indexed_adapter_config_id
    ):
        return False

    return True


def _load_index_manifest(index_root: Path) -> dict[str, Any] | None:
    import json

    for name in ("manifest.json", "index_manifest.json", "metadata.json"):
        manifest_path = index_root / name
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else None
    return None


def _value_from_report(report: Any | None, *names: str) -> Any | None:
    if report is None:
        return None
    if isinstance(report, Mapping):
        for name in names:
            if name in report:
                return report[name]
        for nested_name in ("source_report_summary", "model_identity_gate", "selected_config"):
            nested = report.get(nested_name)
            if isinstance(nested, Mapping):
                for name in names:
                    if name in nested:
                        return nested[name]
    for name in names:
        if hasattr(report, name):
            return getattr(report, name)
    adapter = getattr(report, "adapter", None)
    if adapter is not None:
        adapter_aliases = {
            "model_identity": ("model_identity", "model", "model_id"),
            "model_id": ("model_id", "model_identity", "model"),
            "identity": ("model_identity", "model", "model_id"),
            "tokenizer_identity": ("tokenizer_identity", "tokenizer_id"),
            "tokenizer_id": ("tokenizer_id", "tokenizer_identity"),
            "adapter_config_id": ("adapter_config_id",),
            "kv_target_layer": ("kv_target_layer",),
            "injection_layer": ("injection_layer",),
        }
        for name in names:
            for attr in adapter_aliases.get(name, (name,)):
                if hasattr(adapter, attr):
                    value = getattr(adapter, attr)
                    if value is not None:
                        return value
    for nested_name in ("source_report_summary", "gates", "selected_config"):
        nested = getattr(report, nested_name, None)
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested:
                    return nested[name]
            if nested_name == "gates":
                for gate in nested.values():
                    if isinstance(gate, Mapping):
                        for name in names:
                            if name in gate:
                                return gate[name]
    return None


def _build_session(payload: dict[str, Any]) -> HarnessSession:
    if _SharedHarnessSession is not None and _session_types is not None:
        return _build_shared_session(payload)

    try:
        return HarnessSession(**payload)
    except TypeError:
        session = HarnessSession()
        for key, value in payload.items():
            setattr(session, key, value)
        return session


def _build_shared_session(payload: dict[str, Any]) -> HarnessSession:
    readiness = _session_types.ReadinessState
    severity = _session_types.WarningSeverity

    warnings = tuple(
        _session_types.FailClosedWarning(
            code=code,
            message=code.replace("_", " "),
            severity=severity.BLOCKING if code in payload["boot_errors"] else severity.WARNING,
            component="boot",
        )
        for code in (*payload["boot_errors"], *payload["boot_warnings"])
    )
    user_state = readiness.READY if payload["user_memory_ready"] else readiness.MISSING
    workspace_state = readiness.READY if payload["workspace_memory_ready"] else readiness.MISSING
    index_state = readiness.READY if payload["workspace_index_ready"] else readiness.NEEDS_JIT

    return _SharedHarnessSession(
        session_id=f"boot-{uuid4().hex}",
        workspace_path=payload["workspace_path"],
        memory_root=payload["memory_root"],
        model_identity=_value_from_report(payload["model_report"], "model_identity", "model_id"),
        tokenizer_identity=_value_from_report(
            payload["model_report"], "tokenizer_identity", "tokenizer_id"
        ),
        validation_status=_value_from_report(payload["model_report"], "validation_status"),
        selected_adapter_config=_selected_adapter_config(payload["model_report"]),
        workspace_id=payload["workspace_path"],
        model_adapter=_shared_model_adapter(payload["model_report"]),
        decoder_prior_scope=_shared_decoder_prior_scope(payload["model_report"]),
        materialization_compatibility=_shared_materialization_compatibility(
            payload["model_report"], payload
        ),
        index_readiness=_session_types.IndexReadinessMetadata(
            workspace_id=payload["workspace_path"],
            state=index_state,
            model_identity=_value_from_report(payload["model_report"], "model_identity", "model_id"),
            tokenizer_identity=_value_from_report(
                payload["model_report"], "tokenizer_identity", "tokenizer_id"
            ),
            adapter_config_id=_adapter_config_id(payload["model_report"]),
            memory_families=(
                _session_types.MemoryFamily.USER,
                _session_types.MemoryFamily.TASK,
            ),
            missing_reasons=tuple(payload["jit_actions"]),
        ),
        user_memory=_session_types.UserMemoryStatusMetadata(
            state=user_state,
            store_id=payload["user_memory_root"],
            missing_reasons=() if payload["user_memory_ready"] else ("user_memory_root_missing",),
        ),
        task_memory=_session_types.TaskMemoryStatusMetadata(
            state=workspace_state,
            workspace_id=payload["workspace_path"],
            store_id=payload["workspace_memory_root"],
            missing_reasons=()
            if payload["workspace_memory_ready"]
            else ("workspace_task_memory_root_missing",),
        ),
        provenance=_session_types.Provenance(
            source="chuk_lazarus.harness.boot",
            collector="boot_harness",
            evidence=tuple(payload["jit_actions"]),
        ),
        jit_required=payload["jit_required"],
        jit_actions=tuple(payload["jit_actions"]),
        warnings=warnings,
        metadata=payload,
    )


def _shared_model_adapter(model_report: Any | None) -> Any | None:
    if model_report is None or _session_types is None:
        return None

    adapter = getattr(model_report, "adapter", None)
    model_identity = _value_from_report(model_report, "model_identity", "model_id")
    tokenizer_identity = _value_from_report(model_report, "tokenizer_identity", "tokenizer_id")
    if adapter is not None:
        model_identity = model_identity or getattr(adapter, "model", None) or "unknown"
        tokenizer_identity = tokenizer_identity or getattr(adapter, "tokenizer_identity", None) or "unknown"
    if model_identity is None and tokenizer_identity is None:
        return None

    review_status = _session_types.ReviewStatus.MEASURED if _is_validation_success(model_report) else _session_types.ReviewStatus.NEEDS_REVIEW
    return _session_types.ModelAdapterMetadata(
        model_identity=str(model_identity or "unknown"),
        tokenizer_identity=str(tokenizer_identity or "unknown"),
        adapter_family=str(getattr(adapter, "adapter_family", None) or "unknown"),
        model_revision=getattr(adapter, "model_revision_or_hash", None),
        adapter_config_id=_adapter_config_id(model_report),
        hidden_size=getattr(adapter, "hidden_size", None),
        attention_head_count=getattr(adapter, "num_attention_heads", None),
        kv_head_count=getattr(adapter, "num_key_value_heads", None),
        projection_aliases=dict(getattr(adapter, "projection_aliases", {}) or {}),
        route_layer_candidate=getattr(adapter, "route_layer", None),
        route_dimension=getattr(adapter, "route_dimension", None),
        route_query_head_candidate=getattr(adapter, "route_query_head", None),
        boundary_layer_candidate=getattr(adapter, "boundary_layer", None),
        residual_capture_layer_candidate=getattr(adapter, "residual_capture_layer", None),
        kv_source_layer_candidate=getattr(adapter, "kv_source_layer", None),
        kv_target_layer_candidate=getattr(adapter, "kv_target_layer", None),
        projection_producer_layer_candidate=getattr(adapter, "projection_producer_layer", None),
        behavior_cache_layer_candidate=getattr(adapter, "behavior_cache_layer", None),
        insertion_family=getattr(adapter, "insertion_family", None),
        kv_layout=getattr(adapter, "kv_layout", None),
        confidence=_confidence_as_float(getattr(model_report, "confidence", None)),
        review_status=review_status,
    )


def _shared_decoder_prior_scope(model_report: Any | None) -> Any | None:
    if model_report is None or _session_types is None:
        return None

    adapter = getattr(model_report, "adapter", None)
    model_identity = _value_from_report(model_report, "model_identity", "model_id")
    tokenizer_identity = _value_from_report(model_report, "tokenizer_identity", "tokenizer_id")
    adapter_config_id = _adapter_config_id(model_report)
    target_layer = getattr(adapter, "kv_target_layer", None)
    if target_layer is None:
        target_layer = _value_from_report(model_report, "kv_target_layer", "injection_layer")

    if model_identity is None and tokenizer_identity is None and adapter_config_id is None:
        return None

    return _session_types.DecoderPriorScope(
        model_identity=str(model_identity) if model_identity is not None else None,
        tokenizer_identity=str(tokenizer_identity) if tokenizer_identity is not None else None,
        adapter_config_id=str(adapter_config_id) if adapter_config_id is not None else None,
        layer_scope=(int(target_layer),) if target_layer is not None else (),
    )


def _shared_materialization_compatibility(model_report: Any | None, payload: Mapping[str, Any]) -> Any | None:
    if model_report is None or _session_types is None:
        return None

    adapter = getattr(model_report, "adapter", None)
    model_identity = _value_from_report(model_report, "model_identity", "model_id")
    tokenizer_identity = _value_from_report(model_report, "tokenizer_identity", "tokenizer_id")
    source_layer = getattr(adapter, "kv_source_layer", None)
    target_layer = getattr(adapter, "kv_target_layer", None) or getattr(adapter, "injection_layer", None)
    insertion_family = getattr(adapter, "insertion_family", None)
    strategy = _session_types.MaterializationStrategy.NONE
    if insertion_family and "kv" in str(insertion_family).lower() and target_layer is not None:
        strategy = _session_types.MaterializationStrategy.KV_DIRECT
    elif getattr(adapter, "boundary_layer", None) is not None:
        strategy = _session_types.MaterializationStrategy.BOUNDARY_ONLY

    return _session_types.MaterializationCompatibilityMetadata(
        compatible=bool(_is_validation_success(model_report) and strategy is not _session_types.MaterializationStrategy.NONE),
        strategy=strategy,
        memory_family=_session_types.MemoryFamily.TASK,
        workspace_id=payload["workspace_path"],
        source_model_identity=str(model_identity) if model_identity is not None else None,
        target_model_identity=str(model_identity) if model_identity is not None else None,
        source_tokenizer_identity=str(tokenizer_identity) if tokenizer_identity is not None else None,
        target_tokenizer_identity=str(tokenizer_identity) if tokenizer_identity is not None else None,
        adapter_config_id=_adapter_config_id(model_report),
        source_layer=source_layer,
        target_layer=target_layer,
        insertion_family=str(insertion_family) if insertion_family is not None else None,
        residual_available=bool(payload["workspace_index_ready"]),
        kv_ready=bool(payload["workspace_index_ready"] and target_layer is not None),
        refusal_reasons=() if payload["workspace_index_ready"] else ("workspace_index_not_ready",),
    )


def _selected_adapter_config(model_report: Any | None) -> dict[str, Any] | None:
    selected = getattr(model_report, "selected_config", None)
    if isinstance(selected, Mapping):
        return dict(selected)
    if isinstance(model_report, Mapping):
        selected = model_report.get("selected_config")
        if isinstance(selected, Mapping):
            return dict(selected)
    return None


def _adapter_config_id(model_report: Any | None) -> str | None:
    explicit = _value_from_report(model_report, "adapter_config_id")
    if explicit is not None:
        return str(explicit)

    selected = _selected_adapter_config(model_report) or {}
    parts = [
        _value_from_report(model_report, "model_identity", "model_id"),
        selected.get("adapter_family") or getattr(getattr(model_report, "adapter", None), "adapter_family", None),
        selected.get("route_layer"),
        selected.get("boundary_layer"),
        selected.get("kv_source_layer"),
        selected.get("kv_target_layer"),
        selected.get("insertion_family"),
    ]
    stable_parts = [str(part) for part in parts if part not in (None, "")]
    return ":".join(stable_parts) if stable_parts else None


def _confidence_as_float(value: Any | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
