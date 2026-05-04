"""Product adapter for David's full central-router contract.

The adapter keeps product orchestration behind the stable mini-router protocol:
``route(method=..., prompt=..., evidence=...) -> RoutePacket``.  It does not
import benchmark proof rigs, and it only loads the David router wrapper on
demand.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib.util
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
    ) -> RoutePacket:
        module = self._router_module()
        methodology = METHOD_TO_METHODOLOGY.get(method, "dependency_source")
        windows = _evidence_to_windows(module, evidence, session_id=session_id)
        request = module.RouteRequest(
            query=str(prompt or ""),
            capability_mode=methodology,
            scope_key=session_id,
            metadata={
                "session_id": session_id,
                "max_tokens": max_tokens,
                "product_adapter": "chuk_lazarus.david.central_router_adapter",
            },
        )
        router = module.CentralRouter(hot_count=self._hot_count, warm_count=self._warm_count)
        plan = router.route(request, windows)
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
        "protected_imports": "not_imported",
    }
    compatibility = getattr(route_packet, "compatibility_proof", None)
    if compatibility is not None:
        provenance["compatibility_proof"] = _plain_dataclass_or_mapping(compatibility)
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
            module.RouteWindow(
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
        if isinstance(item, Mapping):
            output[str(key)] = _plain_mapping(item)
        elif isinstance(item, tuple):
            output[str(key)] = list(item)
        else:
            output[str(key)] = item
    return output
