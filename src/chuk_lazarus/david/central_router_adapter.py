"""Product adapter for David's full central-router contract.

The adapter keeps product orchestration behind the stable mini-router protocol:
``route(method=..., prompt=..., evidence=...) -> RoutePacket``.  It does not
import benchmark proof rigs, and it only loads the David router wrapper on
demand.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from .product_router import METHOD_TO_METHODOLOGY
from .routing import RoutePacket


ModuleProvider = Callable[[], Any]


class CentralRouterAdapter:
    """Bridge the full David router plan into the product ``RoutePacket``."""

    def __init__(
        self,
        *,
        module: Any | None = None,
        module_provider: ModuleProvider | None = None,
        hot_count: int = 2,
        warm_count: int = 3,
    ) -> None:
        self._module = module
        self._module_provider = module_provider or load_david_router_wrapper
        self._hot_count = hot_count
        self._warm_count = warm_count

    @classmethod
    def from_stable_wrapper_if_available(cls) -> "CentralRouterAdapter | None":
        """Return an adapter when the stable David wrapper can be loaded."""

        if david_router_wrapper_path().exists():
            return cls()
        return None

    def route(
        self,
        *,
        method: str,
        prompt: str,
        session_id: str,
        evidence: list[dict[str, Any]],
        max_tokens: int,
        adapter_metadata: Mapping[str, Any] | None = None,
        index_readiness_metadata: Mapping[str, Any] | None = None,
        apollo_residual_readiness: Mapping[str, Any] | None = None,
        path_hints: Sequence[str] = (),
        identifiers: Sequence[str] = (),
        ordinal: int | None = None,
        decode_constraints: Mapping[str, Any] | None = None,
        verification_expectations: Sequence[str] = (),
        writeback_targets: Sequence[str] = (),
        allow_jit_indexing: bool = True,
    ) -> RoutePacket:
        module = self._router_module()
        methodology = METHOD_TO_METHODOLOGY.get(method, "dependency_source")
        windows = _evidence_to_windows(module, evidence, session_id=session_id)
        request_metadata = {
            "session_id": session_id,
            "max_tokens": max_tokens,
            "product_adapter": "chuk_lazarus.david.central_router_adapter",
            "adapter_metadata": _plain_mapping(adapter_metadata),
            "index_metadata": _plain_mapping(index_readiness_metadata),
            "index_readiness_metadata": _plain_mapping(index_readiness_metadata),
            "apollo_residual_metadata": _plain_mapping(apollo_residual_readiness),
            "apollo_residual_readiness": _plain_mapping(apollo_residual_readiness),
            "path_hints": tuple(str(item) for item in path_hints),
            "identifiers": tuple(str(item) for item in identifiers),
            "ordinal": ordinal,
            "decode_constraints": _plain_mapping(decode_constraints),
            "verification_expectations": tuple(str(item) for item in verification_expectations),
            "write_back_targets": tuple(str(item) for item in writeback_targets),
            "allow_jit_indexing": bool(allow_jit_indexing),
            "require_apollo_residual_ready": bool(apollo_residual_readiness),
        }
        request = _construct_supported(
            module.RouteRequest,
            query=str(prompt or ""),
            capability_mode=methodology,
            scope_key=session_id,
            ordinal=ordinal,
            path_hints=tuple(str(item) for item in path_hints),
            identifiers=tuple(str(item) for item in identifiers),
            metadata=request_metadata,
        )
        router = module.CentralRouter(hot_count=self._hot_count, warm_count=self._warm_count)
        try:
            plan = router.route(request, windows)
        except Exception as exc:
            if _is_jit_required_signal(exc):
                return route_packet_from_jit_signal(
                    exc,
                    method=method,
                    session_id=session_id,
                    evidence=evidence,
                    max_tokens=max_tokens,
                    request_metadata=request_metadata,
                )
            raise
        return route_packet_from_plan(
            plan,
            method=method,
            session_id=session_id,
            evidence=evidence,
            max_tokens=max_tokens,
        )

    def _router_module(self) -> Any:
        if self._module is None:
            self._module = self._module_provider()
        return self._module


def david_router_wrapper_path() -> Path:
    return Path(__file__).resolve().parents[3] / "David" / "router.py"


def load_david_router_wrapper() -> Any:
    """Load the stable David router wrapper by path, without touching scripts."""

    module_path = david_router_wrapper_path()
    if not module_path.exists():
        raise FileNotFoundError(f"David router wrapper not found: {module_path}")
    module_name = "chuk_lazarus_david_router_wrapper"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load David router wrapper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def route_packet_from_plan(
    plan: Any,
    *,
    method: str,
    session_id: str,
    evidence: list[dict[str, Any]],
    max_tokens: int,
) -> RoutePacket:
    """Collapse a full central ``RoutePlan`` into the product route packet."""

    selected_windows = _selected_window_texts(plan)
    token_cost = min(max_tokens, sum(len(text.split()) for text in selected_windows))
    route_reason = _route_reason(plan, method)
    materialization = getattr(plan, "materialization_plan", None)
    route_packet = getattr(plan, "route_packet", None)
    provenance = {
        "router": "david.central_router.full",
        "adapter": "chuk_lazarus.david.central_router_adapter",
        "route_plan_metadata": _plain_mapping(getattr(plan, "router_metadata", {})),
        "selected_window_ids": _selected_window_ids(plan),
        "tiers_present": list(getattr(plan, "tiers_present", lambda: ())()),
        "tier_details": _tier_details(plan),
        "materialization_plan": _materialization_provenance(materialization),
        "evidence_supports": _evidence_supports(plan, route_packet),
        "protected_imports": "not_imported",
    }
    compatibility = getattr(route_packet, "compatibility_proof", None)
    if compatibility is not None:
        provenance["compatibility_proof"] = _plain_dataclass_or_mapping(compatibility)
    for field_name in ("decode_policy", "verification_plan", "write_back_policy"):
        field_value = getattr(route_packet, field_name, None)
        if field_value is not None:
            provenance[field_name] = _plain_dataclass_or_mapping(field_value)
    jit_actions = _jit_actions_from_plan(materialization, compatibility, provenance["route_plan_metadata"])
    if jit_actions:
        provenance["jit_required"] = True
        provenance["jit_actions"] = jit_actions
    return RoutePacket(
        method=method,
        selected_windows=selected_windows,
        memory_family=_memory_family(method),
        session_id=session_id,
        tier=_product_tier(plan),
        route_reason=route_reason,
        evidence=evidence,
        token_cost=token_cost,
        activation_score=_max_candidate_score(plan, "activation_score"),
        lexical_score=_max_candidate_score(plan, "lexical_score", "literal_score"),
        ordinal_score=_max_candidate_score(plan, "ordinal_score"),
        recency_score=_max_candidate_score(plan, "freshness_score"),
        residual_available=bool(
            getattr(materialization, "apollo_residual_ready", False)
            or getattr(materialization, "boundary_residual_paths", ())
            or getattr(materialization, "residual_stream_paths", ())
        ),
        kv_ready=bool(getattr(materialization, "kv_direct_ready", False)),
        provenance=provenance,
    )


def route_packet_from_jit_signal(
    signal: BaseException,
    *,
    method: str,
    session_id: str,
    evidence: list[dict[str, Any]],
    max_tokens: int,
    request_metadata: Mapping[str, Any],
) -> RoutePacket:
    """Convert a full-router JIT stop signal into explicit product metadata."""

    selected_windows = [
        str(item.get("text") or item.get("reason") or item.get("path") or "")
        for item in evidence[:3]
    ]
    token_cost = min(max_tokens, sum(len(text.split()) for text in selected_windows))
    compatibility = getattr(signal, "compatibility_proof", None)
    metadata = _plain_mapping(getattr(signal, "metadata", {}))
    jit_actions = [
        str(item)
        for item in getattr(signal, "jit_indexing_actions", ())
        or metadata.get("jit_indexing_actions", ())
    ]
    provenance = {
        "router": "david.central_router.full",
        "adapter": "chuk_lazarus.david.central_router_adapter",
        "protected_imports": "not_imported",
        "jit_required": True,
        "jit_actions": jit_actions,
        "jit_metadata": metadata,
        "route_plan_metadata": {
            "jit_indexing_required": True,
            "jit_indexing_stop": metadata.get("jit_indexing_stop", "pre_route"),
        },
        "product_request_metadata": _plain_mapping(request_metadata),
    }
    if compatibility is not None:
        provenance["compatibility_proof"] = _plain_dataclass_or_mapping(compatibility)
    return RoutePacket(
        method=method,
        selected_windows=selected_windows,
        memory_family=_memory_family(method),
        session_id=session_id,
        tier="warm" if evidence else "cold",
        route_reason="full David central router requires JIT before routing",
        evidence=evidence,
        token_cost=token_cost,
        residual_available=bool(getattr(compatibility, "apollo_residual_ready", False)),
        kv_ready=False,
        provenance=provenance,
    )


def _evidence_to_windows(module: Any, evidence: Sequence[Mapping[str, Any]], *, session_id: str) -> list[Any]:
    windows: list[Any] = []
    for index, item in enumerate(evidence):
        data = dict(item)
        text = str(data.get("text") or data.get("reason") or data.get("path") or "")
        window_id = str(data.get("window_id") or data.get("id") or data.get("path") or f"evidence:{index}")
        metadata = dict(data.get("metadata") or {})
        for key in ("path", "kind", "score", "symbols", "import_tokens", "language", "provenance"):
            if key in data:
                metadata.setdefault(key, data[key])
        windows.append(
            _construct_supported(
                module.RouteWindow,
                window_id=window_id,
                text=text,
                scope_key=session_id,
                metadata=metadata,
                lexical_score=_float_or_none(data.get("score")),
            )
        )
    return windows


def _selected_window_texts(plan: Any) -> list[str]:
    materialization = getattr(plan, "materialization_plan", None)
    windows = tuple(getattr(materialization, "windows", ()) or ())
    if not windows:
        windows = tuple(getattr(candidate, "window", None) for candidate in getattr(plan, "candidates", ()) or ())
    return [str(getattr(window, "text", "")) for window in windows if window is not None]


def _selected_window_ids(plan: Any) -> list[str]:
    materialization = getattr(plan, "materialization_plan", None)
    windows = tuple(getattr(materialization, "windows", ()) or ())
    if not windows:
        windows = tuple(getattr(candidate, "window", None) for candidate in getattr(plan, "candidates", ()) or ())
    return [str(getattr(window, "window_id", "")) for window in windows if window is not None]


def _route_reason(plan: Any, method: str) -> str:
    metadata = getattr(plan, "router_metadata", {}) or {}
    mode = metadata.get("capability_mode") if isinstance(metadata, Mapping) else None
    selected = getattr(plan, "selected_candidate", None)
    reasons = tuple(getattr(selected, "reasons", ()) or ()) if selected is not None else ()
    if reasons:
        return str(reasons[0])
    if mode:
        return f"{mode} capability routed by full David central router"
    return f"{method} routed by full David central router"


def _product_tier(plan: Any) -> str:
    tiers = [
        str(getattr(assignment, "tier", "")).lower()
        for assignment in getattr(plan, "tier_assignments", ()) or ()
        if getattr(assignment, "windows", ())
    ]
    for tier in ("hot", "warm", "cold"):
        if tier in tiers:
            return tier
    return "cold"


def _memory_family(method: str) -> str:
    return "user" if method in {"user_continuity", "temporal_recall"} else "task"


def _max_candidate_score(plan: Any, *names: str) -> float:
    values: list[float] = []
    for candidate in getattr(plan, "candidates", ()) or ():
        for name in names:
            value = _float_or_none(getattr(candidate, name, None))
            if value is not None:
                values.append(value)
    return max(values) if values else 0.0


def _tier_details(plan: Any) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for assignment in getattr(plan, "tier_assignments", ()) or ():
        windows = tuple(getattr(assignment, "windows", ()) or ())
        details.append(
            {
                "tier": str(getattr(assignment, "tier", "")),
                "window_ids": [str(getattr(window, "window_id", "")) for window in windows],
                "candidate_ids": [str(item) for item in getattr(assignment, "candidate_ids", ()) or ()],
                "score_range": list(getattr(assignment, "score_range", ()) or ()),
                "reason": str(getattr(assignment, "reason", "")),
            }
        )
    return details


def _materialization_provenance(materialization: Any) -> dict[str, Any]:
    if materialization is None:
        return {}
    keys = (
        "tier_order",
        "tier_window_ids",
        "per_tier_counts",
        "notes",
        "token_budget",
        "payload_plan",
        "boundary_residual_paths",
        "residual_stream_paths",
        "kv_direct_ready",
        "recapture_requirements",
        "insertion_layer",
        "decode_constraints",
        "verification_expectations",
        "write_back_targets",
        "fast_route_ready",
        "apollo_residual_ready",
        "apollo_residual_identity",
        "apollo_manifest_path",
        "apollo_torch_store_path",
    )
    return {
        key: _plain_value(getattr(materialization, key))
        for key in keys
        if hasattr(materialization, key)
    }


def _evidence_supports(plan: Any, route_packet: Any) -> list[dict[str, Any]]:
    supports = tuple(getattr(plan, "evidence_supports", ()) or ())
    if not supports and route_packet is not None:
        supports = tuple(getattr(route_packet, "evidence", ()) or ())
    output: list[dict[str, Any]] = []
    for support in supports:
        output.append(
            {
                "window_id": str(getattr(support, "window_id", "")),
                "supports_claim": str(getattr(support, "supports_claim", "")),
                "confidence": _float_or_none(getattr(support, "confidence", None)) or 0.0,
                "evidence": _plain_value(getattr(support, "evidence", ())),
                "mode": str(getattr(support, "mode", "")),
                "route_trace": _plain_value(getattr(support, "route_trace", {})),
            }
        )
    return output


def _jit_actions_from_plan(materialization: Any, compatibility: Any, metadata: Mapping[str, Any]) -> list[str]:
    actions: list[str] = []
    for source in (
        getattr(compatibility, "jit_indexing_actions", ()) if compatibility is not None else (),
        getattr(materialization, "recapture_requirements", ()) if materialization is not None else (),
        metadata.get("jit_indexing_actions", ()),
    ):
        if isinstance(source, str):
            actions.append(source)
        else:
            actions.extend(str(item) for item in source or ())
    return list(dict.fromkeys(item for item in actions if item))


def _is_jit_required_signal(exc: BaseException) -> bool:
    return hasattr(exc, "jit_indexing_actions") and hasattr(exc, "compatibility_proof")


def _construct_supported(factory: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(factory)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return factory(**kwargs)
    return factory(**{key: value for key, value in kwargs.items() if key in parameters})


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plain_dataclass_or_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return _plain_mapping(asdict(value))
    return _plain_mapping(value)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, Any] = {}
    for key, item in value.items():
        output[str(key)] = _plain_value(item)
    return output


def _plain_value(value: Any) -> Any:
    if is_dataclass(value):
        return _plain_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_plain_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
