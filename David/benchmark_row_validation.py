#!/usr/bin/env python3
"""One-row benchmark/chat validation for David's central router.

This script intentionally uses only local fixtures and the Python standard
library. It is a contract check for ``David/central router.py`` and should be
run from the repository root with:

    python3 David/benchmark_row_validation.py
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from typing import Any


REQUIRED_TIERS = {"HOT", "WARM", "COLD"}
NON_COLD_TIERS = {"HOT", "WARM"}
TIER_RANK = {"HOT": 0, "WARM": 1, "COLD": 2}

ASSUMPTIONS = (
    "David/central router.py is importable via importlib despite the space in the filename.",
    "The module exports RouteRequest, RouteWindow, CentralRouter, and a RoutePlan returned by routing.",
    "RouteWindow accepts keyword-compatible window id, text/content, source path, kind/role, turn index, and metadata fields.",
    "RouteRequest accepts keyword-compatible row id, benchmark, capability/mode, query/question, windows/candidates, and metadata fields.",
    "CentralRouter() can be constructed without arguments and routes with route(request) or an equivalent route-like method.",
    "RoutePlan.assert_tier_coverage() enforces HOT/WARM/COLD coverage when at least three route windows exist.",
    "RoutePlan exposes inspectable tiers through tier_assignments/tiered windows and route metadata/trace for validation summaries.",
)


class RouterUnavailable(RuntimeError):
    """Raised when the central router module has not landed yet."""


class RouterContractError(RuntimeError):
    """Raised when the router does not satisfy the agreed validation API."""


class ValidationFailure(AssertionError):
    """Raised when a representative row fails its expected route behavior."""


@dataclass(frozen=True)
class FixtureWindow:
    window_id: str
    text: str
    path: str
    kind: str
    role: str
    turn_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RowFixture:
    name: str
    benchmark: str
    capability: str
    row_id: str
    query: str
    windows: tuple[FixtureWindow, ...]
    metadata: Mapping[str, Any]
    expected_ids: tuple[str, ...] = ()
    expected_cold_ids: tuple[str, ...] = ()
    expected_ahead_ids: tuple[str, ...] = ()
    expected_behind_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedPlan:
    plan: Any
    tier_assignments: dict[str, str]
    tier_counts: dict[str, int]
    selected_ids: list[str]
    ordered_ids: list[str]
    order_source: str
    route_trace: Any
    metadata: dict[str, Any]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return getattr(value, "value")
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return repr(value)


def _normalize_tier(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = getattr(value, "value")
    text = str(value).strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.upper()
    if text in REQUIRED_TIERS:
        return text
    lowered = text.lower()
    if lowered in {"hot", "warm", "cold"}:
        return lowered.upper()
    return None


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _field_value(value: Any, names: Iterable[str], default: Any = None) -> Any:
    if value is None:
        return default
    if _is_mapping(value):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _metadata_from(value: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("metadata", "route_metadata", "routing_metadata", "router_metadata"):
        candidate = _field_value(value, (name,))
        if isinstance(candidate, Mapping):
            metadata.update(candidate)
    for key in (
        "row_id",
        "request_id",
        "benchmark",
        "capability",
        "capability_mode",
        "mode",
    ):
        candidate = _field_value(value, (key,))
        if candidate is not None:
            metadata[key] = candidate
    return metadata


def _window_id(value: Any) -> str | None:
    raw = _field_value(value, ("window_id", "id", "uid", "key", "name"))
    if raw is None:
        return None
    return str(raw)


def _window_text(value: Any) -> str:
    raw = _field_value(value, ("text", "content", "body", "payload"), "")
    return str(raw)


def _window_path(value: Any) -> str | None:
    raw = _field_value(value, ("source_path", "path", "file_path", "filepath"))
    if raw is not None:
        return str(raw)
    metadata = _metadata_from(value)
    for key in ("source_path", "path", "file_path", "filepath"):
        if key in metadata:
            return str(metadata[key])
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if _is_mapping(value):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _ids_from_value(value: Any) -> list[str]:
    ids: list[str] = []
    for item in _as_list(value):
        if isinstance(item, (str, int)):
            ids.append(str(item))
            continue
        window_id = _window_id(item)
        if window_id is not None:
            ids.append(window_id)
    return ids


def _construct(cls: type[Any], payload: Mapping[str, Any], label: str) -> Any:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**payload)

    params = signature.parameters
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    kwargs: dict[str, Any] = {}
    missing: list[str] = []

    for name, param in params.items():
        if name == "self" or param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        if name in payload:
            kwargs[name] = payload[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        available = ", ".join(sorted(payload))
        raise RouterContractError(
            f"{label} constructor requires unsupported fields {missing}; "
            f"available fixture keys: {available}"
        )

    if has_kwargs:
        for key, value in payload.items():
            kwargs.setdefault(key, value)

    return cls(**kwargs)


def _load_router_module() -> Any:
    module_path = Path(__file__).resolve().with_name("central router.py")
    if not module_path.exists():
        raise RouterUnavailable(f"Router module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("david_central_router", module_path)
    if spec is None or spec.loader is None:
        raise RouterUnavailable(f"Could not build import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as exc:
        raise RouterUnavailable(f"Router module not found: {module_path}") from exc

    for name in ("RouteRequest", "RouteWindow", "CentralRouter"):
        if not hasattr(module, name):
            raise RouterContractError(f"Router module is missing required export {name}")
    return module


def _make_route_window(route_window_cls: type[Any], row: RowFixture, window: FixtureWindow) -> Any:
    source_authority = (
        window.metadata.get("source_authority")
        or {
            "user_preference": "user",
            "tool": "tool",
            "task": "task",
        }.get(window.role)
    )
    memory_scope = window.metadata.get("memory_scope") or row.metadata.get("memory_scope")
    scope_key = window.metadata.get("scope_key") or row.metadata.get("scope_key")
    canonical_request = (
        window.metadata.get("canonical_request")
        or window.metadata.get("request")
        or row.metadata.get("canonical_request")
    )
    updated_at = (
        window.metadata.get("updated_at")
        or window.metadata.get("effective_at")
        or window.metadata.get("created_at")
    )
    metadata = {
        **dict(window.metadata),
        "row_id": row.row_id,
        "benchmark": row.benchmark,
        "capability": row.capability,
        "path": window.path,
        "source_path": window.path,
        "kind": window.kind,
        "role": window.role,
        "turn_index": window.turn_index,
        "window_id": window.window_id,
    }
    if source_authority:
        metadata["source_authority"] = source_authority
    if memory_scope:
        metadata["memory_scope"] = memory_scope
    if scope_key:
        metadata["scope_key"] = scope_key
    if canonical_request:
        metadata["canonical_request"] = canonical_request
    if updated_at:
        metadata["updated_at"] = updated_at

    payload = {
        "window_id": window.window_id,
        "id": window.window_id,
        "text": window.text,
        "content": window.text,
        "body": window.text,
        "source_path": window.path,
        "path": window.path,
        "kind": window.kind,
        "role": window.role,
        "turn_index": window.turn_index,
        "session_turn_index": window.turn_index,
        "scope_key": scope_key,
        "canonical_request": canonical_request,
        "memory_scope": memory_scope,
        "source_authority": source_authority,
        "source_author": source_authority,
        "stale": bool(window.metadata.get("stale", False)),
        "superseded_by": window.metadata.get("superseded_by"),
        "updated_at": updated_at,
        "metadata": metadata,
    }
    return _construct(route_window_cls, payload, "RouteWindow")


def _make_route_request(
    route_request_cls: type[Any],
    route_windows: list[Any],
    row: RowFixture,
) -> Any:
    metadata = {
        **dict(row.metadata),
        "row_id": row.row_id,
        "benchmark": row.benchmark,
        "capability": row.capability,
        "expected_ids": list(row.expected_ids),
        "expected_cold_ids": list(row.expected_cold_ids),
    }
    path_hints = tuple(
        metadata.get("path_hints")
        or metadata.get("expected_paths")
        or metadata.get("source_hints")
        or ()
    )
    identifiers = tuple(
        item
        for item in (
            metadata.get("symbol"),
            metadata.get("dependency"),
            *metadata.get("expected_variables", ()),
        )
        if item
    )
    payload = {
        "row_id": row.row_id,
        "request_id": row.row_id,
        "benchmark": row.benchmark,
        "benchmark_name": row.name,
        "capability": row.capability,
        "capability_mode": row.capability,
        "mode": row.capability,
        "query": row.query,
        "question": row.query,
        "prompt": row.query,
        "ordinal": metadata.get("ordinal") or metadata.get("target_occurrence"),
        "scope_key": metadata.get("scope_key") or metadata.get("memory_scope"),
        "canonical_request": metadata.get("canonical_request"),
        "path_hints": path_hints,
        "identifiers": identifiers,
        "entities": tuple(metadata.get("entities") or ()),
        "windows": route_windows,
        "candidate_windows": route_windows,
        "candidates": route_windows,
        "metadata": metadata,
    }
    return _construct(route_request_cls, payload, "RouteRequest")


def _make_router(module: Any) -> Any:
    router_cls = getattr(module, "CentralRouter")
    try:
        return router_cls()
    except TypeError as exc:
        raise RouterContractError("CentralRouter must be constructible without arguments") from exc


def _call_route_method(method: Any, request: Any, windows: list[Any]) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(request)

    params = [
        param
        for param in signature.parameters.values()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    positional = [
        param
        for param in params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        param
        for param in positional
        if param.default is inspect.Parameter.empty
    ]

    if len(required) >= 2 or len(positional) >= 2:
        return method(request, windows)
    if len(positional) == 1:
        return method(request)

    kwargs: dict[str, Any] = {}
    for param in params:
        if param.name in {"request", "route_request", "req"}:
            kwargs[param.name] = request
        elif param.name in {"windows", "route_windows", "candidate_windows", "candidates"}:
            kwargs[param.name] = windows
    if kwargs:
        return method(**kwargs)

    return method(request)


def _route(router: Any, request: Any, windows: list[Any]) -> Any:
    for method_name in ("route", "route_request", "build_plan", "plan", "select"):
        method = getattr(router, method_name, None)
        if callable(method):
            return _call_route_method(method, request, windows)
    if callable(router):
        return _call_route_method(router, request, windows)
    raise RouterContractError("CentralRouter exposes no route-like method")


def _extract_route_trace(plan: Any) -> Any:
    trace = _field_value(
        plan,
        ("route_trace", "trace", "routing_trace", "diagnostics", "events"),
    )
    if trace is not None:
        return trace
    for metadata_name in ("metadata", "route_metadata", "routing_metadata", "router_metadata"):
        metadata = _field_value(plan, (metadata_name,))
        if isinstance(metadata, Mapping):
            for key in ("route_trace", "trace", "routing_trace"):
                if key in metadata:
                    return metadata[key]
    traces: list[Any] = []
    for collection_name in ("candidates", "evidence_supports"):
        for item in _as_list(_field_value(plan, (collection_name,))):
            item_trace = _field_value(item, ("route_trace", "trace", "routing_trace"))
            if item_trace:
                traces.append(item_trace)
    if traces:
        return traces
    return None


def _trace_length(trace: Any) -> int:
    if trace is None:
        return 0
    if isinstance(trace, (str, bytes)):
        return 1 if trace else 0
    if isinstance(trace, Mapping):
        return len(trace)
    if isinstance(trace, Iterable):
        return len(list(trace))
    return 1


def _window_collections(plan: Any) -> list[tuple[str, list[Any]]]:
    collections: list[tuple[str, list[Any]]] = []
    for name in (
        "windows",
        "route_windows",
        "routed_windows",
        "tiered_windows",
        "window_plans",
        "selected_windows",
        "evidence_windows",
        "context_windows",
    ):
        value = _field_value(plan, (name,))
        if value is not None:
            collections.append((name, _as_list(value)))

    for tier, names in (
        ("HOT", ("hot_windows", "hot")),
        ("WARM", ("warm_windows", "warm")),
        ("COLD", ("cold_windows", "cold")),
    ):
        for name in names:
            value = _field_value(plan, (name,))
            if value is not None:
                tagged = []
                for item in _as_list(value):
                    if _is_mapping(item):
                        with_tier = dict(item)
                        with_tier.setdefault("tier", tier)
                        tagged.append(with_tier)
                    else:
                        tagged.append(item)
                collections.append((name, tagged))
    return collections


def _extract_tier_assignments(plan: Any) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for name in (
        "tier_assignments",
        "tier_by_window_id",
        "tier_by_id",
        "tiers",
        "tier_map",
    ):
        value = _field_value(plan, (name,))
        if isinstance(value, Mapping):
            for raw_id, raw_tier in value.items():
                tier = _normalize_tier(raw_tier)
                if tier is not None:
                    assignments[str(raw_id)] = tier

    for assignment in _as_list(_field_value(plan, ("tier_assignments",))):
        tier = _normalize_tier(_field_value(assignment, ("tier", "tier_label", "name")))
        if tier is None:
            continue
        for window_id in _ids_from_value(_field_value(assignment, ("candidate_ids", "window_ids", "ids"))):
            assignments[window_id] = tier
        for item in _as_list(_field_value(assignment, ("windows", "route_windows"))):
            window_id = _window_id(item)
            if window_id is not None:
                assignments[window_id] = tier

    materialization = _field_value(plan, ("materialization_plan",))
    tier_window_ids = _field_value(materialization, ("tier_window_ids",))
    if isinstance(tier_window_ids, Mapping):
        for raw_tier, raw_ids in tier_window_ids.items():
            tier = _normalize_tier(raw_tier)
            if tier is None:
                continue
            for window_id in _ids_from_value(raw_ids):
                assignments.setdefault(window_id, tier)

    for source_name, windows in _window_collections(plan):
        source_tier = _normalize_tier(source_name.replace("_windows", ""))
        for item in windows:
            window_id = _window_id(item)
            if window_id is None:
                continue
            tier = _normalize_tier(
                _field_value(item, ("tier", "tier_label", "route_tier", "label"))
            )
            if tier is None:
                tier = source_tier
            if tier is not None:
                assignments.setdefault(window_id, tier)

    return assignments


def _extract_selected_ids(plan: Any, row: RowFixture, tiers: Mapping[str, str]) -> list[str]:
    for name in (
        "selected_window_ids",
        "selected_ids",
        "chosen_window_ids",
        "evidence_window_ids",
        "context_window_ids",
        "hot_window_ids",
        "warm_window_ids",
    ):
        ids = _ids_from_value(_field_value(plan, (name,)))
        if ids:
            return ids

    for name in (
        "selected_windows",
        "selected",
        "chosen_windows",
        "evidence_windows",
        "context_windows",
    ):
        ids = _ids_from_value(_field_value(plan, (name,)))
        if ids:
            return ids

    hot_warm_ids: list[str] = []
    for name in ("hot_windows", "warm_windows"):
        hot_warm_ids.extend(_ids_from_value(_field_value(plan, (name,))))
    if hot_warm_ids:
        return list(dict.fromkeys(hot_warm_ids))

    return [window.window_id for window in row.windows if tiers.get(window.window_id) in NON_COLD_TIERS]


def _extract_ordered_ids(plan: Any, selected_ids: list[str]) -> tuple[list[str], str]:
    for name in (
        "ordered_window_ids",
        "ranked_window_ids",
        "route_window_ids",
        "routed_window_ids",
        "selected_window_ids",
        "selected_ids",
    ):
        ids = _ids_from_value(_field_value(plan, (name,)))
        if ids:
            return ids, name

    for name in (
        "candidates",
        "ranked_candidates",
        "ranked_windows",
        "ordered_windows",
        "routed_windows",
        "route_windows",
        "selected_windows",
        "evidence_windows",
        "context_windows",
        "windows",
    ):
        ids = _ids_from_value(_field_value(plan, (name,)))
        if ids:
            return ids, name

    return selected_ids, "selected_ids_fallback"


def _resolve_plan(plan: Any, row: RowFixture) -> ResolvedPlan:
    if plan is None:
        raise RouterContractError("Router returned None instead of a RoutePlan")

    assert_tier_coverage = getattr(plan, "assert_tier_coverage", None)
    if not callable(assert_tier_coverage):
        raise RouterContractError("RoutePlan is missing assert_tier_coverage()")
    assert_tier_coverage()

    tier_assignments = _extract_tier_assignments(plan)
    tier_counts = {tier: 0 for tier in sorted(REQUIRED_TIERS)}
    for tier in tier_assignments.values():
        if tier in tier_counts:
            tier_counts[tier] += 1

    selected_ids = _extract_selected_ids(plan, row, tier_assignments)
    ordered_ids, order_source = _extract_ordered_ids(plan, selected_ids)
    metadata = _metadata_from(plan)
    route_trace = _extract_route_trace(plan)

    if len(row.windows) >= len(REQUIRED_TIERS):
        missing_tiers = sorted(REQUIRED_TIERS.difference(tier_assignments.values()))
        _assert(
            not missing_tiers,
            f"{row.name} missing tier coverage for {missing_tiers}; "
            f"tier assignments: {tier_assignments}",
        )

    return ResolvedPlan(
        plan=plan,
        tier_assignments=tier_assignments,
        tier_counts=tier_counts,
        selected_ids=selected_ids,
        ordered_ids=ordered_ids,
        order_source=order_source,
        route_trace=route_trace,
        metadata=metadata,
    )


def _assert_non_cold(resolved: ResolvedPlan, window_ids: Iterable[str], row_name: str) -> list[str]:
    checks: list[str] = []
    for window_id in window_ids:
        tier = resolved.tier_assignments.get(window_id)
        _assert(
            tier in NON_COLD_TIERS,
            f"{row_name} expected {window_id} to be selected as HOT/WARM, got {tier}",
        )
        _assert(
            window_id in resolved.selected_ids,
            f"{row_name} expected {window_id} in selected ids {resolved.selected_ids}",
        )
        checks.append(f"{window_id} selected as {tier}")
    return checks


def _assert_cold(resolved: ResolvedPlan, window_ids: Iterable[str], row_name: str) -> list[str]:
    checks: list[str] = []
    for window_id in window_ids:
        tier = resolved.tier_assignments.get(window_id)
        _assert(tier == "COLD", f"{row_name} expected {window_id} to be COLD, got {tier}")
        checks.append(f"{window_id} filtered as COLD")
    return checks


def _assert_filtered(resolved: ResolvedPlan, window_ids: Iterable[str], row_name: str) -> list[str]:
    stale_ids = set(_ids_from_value(resolved.metadata.get("filtered_stale_window_ids")))
    superseded_ids = set(_ids_from_value(resolved.metadata.get("filtered_superseded_window_ids")))
    filtered_ids = stale_ids | superseded_ids
    checks: list[str] = []
    for window_id in window_ids:
        _assert(
            window_id in filtered_ids,
            f"{row_name} expected {window_id} in stale/superseded filters; got {sorted(filtered_ids)}",
        )
        _assert(
            window_id not in resolved.selected_ids,
            f"{row_name} expected filtered {window_id} outside selected ids {resolved.selected_ids}",
        )
        tier = resolved.tier_assignments.get(window_id)
        _assert(
            tier not in NON_COLD_TIERS,
            f"{row_name} expected filtered {window_id} outside HOT/WARM tiers, got {tier}",
        )
        checks.append(f"{window_id} filtered before active memory routing")
    return checks


def _assert_ahead(
    resolved: ResolvedPlan,
    ahead_ids: Iterable[str],
    behind_ids: Iterable[str],
    row_name: str,
) -> list[str]:
    ahead = list(ahead_ids)
    behind = list(behind_ids)
    checks: list[str] = []

    order = resolved.ordered_ids
    if order:
        present_ahead = [window_id for window_id in ahead if window_id in order]
        present_behind = [window_id for window_id in behind if window_id in order]
        if present_ahead and present_behind:
            first_ahead = min(order.index(window_id) for window_id in present_ahead)
            first_behind = min(order.index(window_id) for window_id in present_behind)
            _assert(
                first_ahead < first_behind,
                f"{row_name} expected implementation ids {ahead} ahead of {behind}; "
                f"order source {resolved.order_source}: {order}",
            )
            checks.append(f"implementation precedes distractors via {resolved.order_source}")
            return checks

    for ahead_id in ahead:
        ahead_tier = resolved.tier_assignments.get(ahead_id)
        _assert(ahead_tier in TIER_RANK, f"{row_name} missing tier for {ahead_id}")
        for behind_id in behind:
            behind_tier = resolved.tier_assignments.get(behind_id)
            _assert(behind_tier in TIER_RANK, f"{row_name} missing tier for {behind_id}")
            _assert(
                TIER_RANK[ahead_tier] <= TIER_RANK[behind_tier],
                f"{row_name} expected {ahead_id} ({ahead_tier}) to outrank "
                f"{behind_id} ({behind_tier})",
            )
    checks.append("implementation tiers outrank distractors")
    return checks


def _assert_trace(resolved: ResolvedPlan, row_name: str) -> list[str]:
    length = _trace_length(resolved.route_trace)
    _assert(length > 0, f"{row_name} expected a non-empty route_trace")
    return [f"route_trace present with {length} step(s)"]


def _assert_metadata(resolved: ResolvedPlan, row: RowFixture) -> list[str]:
    metadata = resolved.metadata
    row_id = metadata.get("row_id") or metadata.get("request_id")
    capability = metadata.get("capability") or metadata.get("capability_mode") or metadata.get("mode")
    if row_id is not None:
        _assert(row_id == row.row_id, f"{row.name} row_id metadata mismatch; got {metadata}")
    if metadata.get("benchmark") is not None:
        _assert(metadata.get("benchmark") == row.benchmark, f"{row.name} benchmark metadata mismatch; got {metadata}")
    _assert(capability == row.capability, f"{row.name} missing capability metadata; got {metadata}")
    _assert(metadata.get("input_window_count") == len(row.windows), f"{row.name} missing input count metadata; got {metadata}")
    _assert(int(metadata.get("eligible_window_count") or 0) >= 3, f"{row.name} missing eligible count metadata; got {metadata}")
    _assert(metadata.get("selected_window_id"), f"{row.name} missing selected window metadata; got {metadata}")
    _assert(set(metadata.get("tiers_present") or ()) >= REQUIRED_TIERS, f"{row.name} missing tiers_present metadata; got {metadata}")
    _assert(metadata.get("tier_invariant"), f"{row.name} missing tier invariant metadata; got {metadata}")
    _assert(resolved.tier_assignments, f"{row.name} expected populated tier metadata")
    _assert(_trace_length(resolved.route_trace) > 0, f"{row.name} expected populated route_trace metadata")
    if row.name == "SWE":
        for key in ("source_hints", "triad_roles", "selected_tests", "patch_target_policy"):
            _assert(metadata.get(key), f"SWE missing {key} route metadata; got {metadata}")
    return ["route metadata populated"]


def _window_payload_summary(row: RowFixture) -> dict[str, str]:
    return {window.window_id: window.path for window in row.windows}


def _validate_row(row: RowFixture, resolved: ResolvedPlan) -> list[str]:
    checks: list[str] = ["assert_tier_coverage passed", "HOT/WARM/COLD coverage present"]

    if row.name == "MRCR":
        expected = row.expected_ids[0]
        tier = resolved.tier_assignments.get(expected)
        _assert(tier == "HOT", f"MRCR expected third duplicate {expected} as HOT, got {tier}")
        _assert(expected in resolved.selected_ids, f"MRCR expected selected id {expected}, got {resolved.selected_ids}")
        checks.append("third duplicate response selected as HOT")
    elif row.name == "RULER":
        checks.extend(_assert_non_cold(resolved, row.expected_ids, row.name))
        checks.append("assignment chain evidence retained")
    elif row.name == "LoCo":
        checks.extend(_assert_non_cold(resolved, row.expected_ids, row.name))
        checks.extend(_assert_cold(resolved, row.expected_cold_ids, row.name))
        checks.extend(_assert_trace(resolved, row.name))
        checks.append("source/dependency paths selected")
    elif row.name == "SWE":
        checks.extend(_assert_non_cold(resolved, row.expected_ids, row.name))
        checks.extend(_assert_ahead(resolved, row.expected_ahead_ids, row.expected_behind_ids, row.name))
        checks.extend(_assert_metadata(resolved, row))
    elif row.name == "Chat":
        checks.extend(_assert_non_cold(resolved, row.expected_ids, row.name))
        checks.extend(_assert_filtered(resolved, row.expected_cold_ids, row.name))
        checks.append("current user memory selected and stale memories filtered")
    else:
        raise ValidationFailure(f"Unknown row fixture {row.name}")

    return checks


def _fixtures() -> tuple[RowFixture, ...]:
    return (
        RowFixture(
            name="MRCR",
            benchmark="mrcr",
            capability="temporal_ordinal",
            row_id="mrcr-third-duplicate-response",
            query=(
                "The request 'Return the relay code' appeared four times. "
                "What was the response to the 3rd occurrence?"
            ),
            windows=(
                FixtureWindow(
                    "mrcr-1",
                    "REQUEST: Return the relay code\nRESPONSE: relay-code-alpha",
                    "mrcr/session/turn-001.json",
                    "conversation_turn",
                    "duplicate_request_response",
                    1,
                    {"occurrence": 1, "request": "Return the relay code", "response": "relay-code-alpha"},
                ),
                FixtureWindow(
                    "mrcr-2",
                    "REQUEST: Return the relay code\nRESPONSE: relay-code-bravo",
                    "mrcr/session/turn-019.json",
                    "conversation_turn",
                    "duplicate_request_response",
                    19,
                    {"occurrence": 2, "request": "Return the relay code", "response": "relay-code-bravo"},
                ),
                FixtureWindow(
                    "mrcr-3",
                    "REQUEST: Return the relay code\nRESPONSE: relay-code-sapphire",
                    "mrcr/session/turn-047.json",
                    "conversation_turn",
                    "duplicate_request_response",
                    47,
                    {"occurrence": 3, "request": "Return the relay code", "response": "relay-code-sapphire"},
                ),
                FixtureWindow(
                    "mrcr-4",
                    "REQUEST: Return the relay code\nRESPONSE: relay-code-delta",
                    "mrcr/session/turn-088.json",
                    "conversation_turn",
                    "duplicate_request_response",
                    88,
                    {"occurrence": 4, "request": "Return the relay code", "response": "relay-code-delta"},
                ),
            ),
            metadata={
                "target_occurrence": 3,
                "expected_response": "relay-code-sapphire",
                "canonical_request": "Return the relay code",
                "scope_key": "mrcr-relay-code",
            },
            expected_ids=("mrcr-3",),
        ),
        RowFixture(
            name="RULER",
            benchmark="ruler",
            capability="symbolic_chain",
            row_id="ruler-assignment-chain-sapphire",
            query="Given VAR A = B, VAR B = C, and VAR C = sapphire, which variables are assigned sapphire?",
            windows=(
                FixtureWindow(
                    "ruler-distractor",
                    "VAR Z = topaz",
                    "ruler/assignments/noise-007.txt",
                    "assignment",
                    "distractor",
                    7,
                    {"variable": "Z", "value": "topaz"},
                ),
                FixtureWindow(
                    "ruler-a",
                    "VAR A = B",
                    "ruler/assignments/a.txt",
                    "assignment",
                    "symbolic_chain",
                    10,
                    {"variable": "A", "value": "B"},
                ),
                FixtureWindow(
                    "ruler-b",
                    "VAR B = C",
                    "ruler/assignments/b.txt",
                    "assignment",
                    "symbolic_chain",
                    11,
                    {"variable": "B", "value": "C"},
                ),
                FixtureWindow(
                    "ruler-c",
                    "VAR C = sapphire",
                    "ruler/assignments/c.txt",
                    "assignment",
                    "symbolic_chain",
                    12,
                    {"variable": "C", "value": "sapphire"},
                ),
            ),
            metadata={"target_value": "sapphire", "expected_variables": ["A", "B", "C"]},
            expected_ids=("ruler-a", "ruler-b", "ruler-c"),
        ),
        RowFixture(
            name="LoCo",
            benchmark="loco",
            capability="dependency_source",
            row_id="loco-build-index-dependency",
            query=(
                "For build_index(), select the implementation and the helper "
                "source that provides normalize_key; ignore tests, docs, and assets."
            ),
            windows=(
                FixtureWindow(
                    "loco-test",
                    "def test_build_index_normalizes_keys(): assert build_index(['A']) == {'a': 1}",
                    "tests/search/test_indexer.py",
                    "test",
                    "distractor",
                    4,
                    {"language": "python"},
                ),
                FixtureWindow(
                    "loco-doc",
                    "# Search indexing\nThe indexer normalizes keys before storing entries.",
                    "docs/search-index.md",
                    "documentation",
                    "distractor",
                    5,
                    {},
                ),
                FixtureWindow(
                    "loco-asset",
                    "<svg><title>Search Icon</title></svg>",
                    "assets/search.svg",
                    "asset",
                    "distractor",
                    6,
                    {},
                ),
                FixtureWindow(
                    "loco-impl",
                    (
                        "from pkg.search.helpers import normalize_key\n\n"
                        "def build_index(records):\n"
                        "    return {normalize_key(record.name): record for record in records}\n"
                    ),
                    "pkg/search/indexer.py",
                    "source",
                    "implementation",
                    1,
                    {"defines": ["build_index"], "imports": ["pkg.search.helpers.normalize_key"]},
                ),
                FixtureWindow(
                    "loco-helper",
                    (
                        "def normalize_key(value):\n"
                        "    return value.strip().lower().replace(' ', '-')\n"
                    ),
                    "pkg/search/helpers.py",
                    "source",
                    "dependency",
                    2,
                    {"defines": ["normalize_key"]},
                ),
            ),
            metadata={
                "symbol": "build_index",
                "dependency": "normalize_key",
                "expected_paths": ["pkg/search/indexer.py", "pkg/search/helpers.py"],
            },
            expected_ids=("loco-impl", "loco-helper"),
            expected_cold_ids=("loco-asset",),
        ),
        RowFixture(
            name="SWE",
            benchmark="swe",
            capability="patch_target",
            row_id="swe-router-lifecycle-metadata",
            query=(
                "Fix the router so lifecycle metadata is initialized before the adapter "
                "handles requests. Which implementation files should be patched first?"
            ),
            windows=(
                FixtureWindow(
                    "swe-problem",
                    (
                        "Problem: route metadata is empty when CentralRouter invokes the "
                        "adapter lifecycle entrypoint during request handling."
                    ),
                    "swe/problem_statement.md",
                    "problem_statement",
                    "task",
                    1,
                    {},
                ),
                FixtureWindow(
                    "swe-test",
                    "def test_router_populates_lifecycle_metadata_before_adapter(): ...",
                    "tests/router/test_lifecycle_metadata.py",
                    "test",
                    "selected_test",
                    2,
                    {"selected": True},
                ),
                FixtureWindow(
                    "swe-doc",
                    "# Router lifecycle\nDocuments the adapter lifecycle phases.",
                    "docs/router-lifecycle.md",
                    "documentation",
                    "distractor",
                    3,
                    {},
                ),
                FixtureWindow(
                    "swe-asset",
                    "binary diagram placeholder",
                    "assets/router-lifecycle.png",
                    "asset",
                    "distractor",
                    4,
                    {},
                ),
                FixtureWindow(
                    "swe-adapter",
                    (
                        "class RouteAdapter:\n"
                        "    def handle(self, request, route_metadata):\n"
                        "        return self.entrypoint.dispatch(request, route_metadata)\n"
                    ),
                    "src/lazarus/router/adapter.py",
                    "source",
                    "implementation",
                    5,
                    {"component": "adapter", "patch_hint": True},
                ),
                FixtureWindow(
                    "swe-entrypoint",
                    (
                        "def dispatch_route(request):\n"
                        "    metadata = {}\n"
                        "    return lifecycle.invoke(request, metadata)\n"
                    ),
                    "src/lazarus/router/entrypoint.py",
                    "source",
                    "implementation",
                    6,
                    {"component": "entrypoint", "patch_hint": True},
                ),
                FixtureWindow(
                    "swe-lifecycle",
                    (
                        "def invoke(request, route_metadata):\n"
                        "    route_metadata.setdefault('route_trace', [])\n"
                        "    return adapter.handle(request, route_metadata)\n"
                    ),
                    "src/lazarus/router/lifecycle.py",
                    "source",
                    "implementation",
                    7,
                    {"component": "lifecycle", "patch_hint": True},
                ),
            ),
            metadata={
                "issue_type": "repo_task",
                "selected_tests": ["tests/router/test_lifecycle_metadata.py"],
                "source_hints": [
                    "src/lazarus/router/adapter.py",
                    "src/lazarus/router/entrypoint.py",
                    "src/lazarus/router/lifecycle.py",
                ],
            },
            expected_ids=("swe-adapter", "swe-entrypoint", "swe-lifecycle"),
            expected_ahead_ids=("swe-adapter", "swe-entrypoint", "swe-lifecycle"),
            expected_behind_ids=("swe-test", "swe-doc", "swe-asset"),
        ),
        RowFixture(
            name="Chat",
            benchmark="interactive_chat",
            capability="durable_chat_memory",
            row_id="chat-current-preference-vs-stale",
            query="How should validation output be formatted for this user right now?",
            windows=(
                FixtureWindow(
                    "chat-stale-pref",
                    "Stale user preference from 2024: answer with long prose paragraphs and no JSON.",
                    "chat/memory/user_pref_2024_legacy.json",
                    "memory",
                    "user_preference",
                    1,
                    {"status": "superseded", "stale": True, "superseded_by": "chat-current-pref"},
                ),
                FixtureWindow(
                    "chat-tool-memory",
                    "Tool memory: run tinydex scan before touching files in this repository.",
                    "chat/memory/tooling_tinydex.json",
                    "memory",
                    "tool",
                    2,
                    {"status": "current"},
                ),
                FixtureWindow(
                    "chat-task-memory",
                    "Task memory: previous batch focused on a smoke test for benchmark imports.",
                    "chat/memory/task_previous_batch.json",
                    "memory",
                    "task",
                    3,
                    {"status": "completed"},
                ),
                FixtureWindow(
                    "chat-current-pref",
                    (
                        "Current user preference: print concise JSON pass summaries for "
                        "validation scripts, with stale memories filtered out."
                    ),
                    "chat/memory/user_pref_2026_current.json",
                    "memory",
                    "user_preference",
                    4,
                    {"status": "current", "effective_at": "2026-05-02"},
                ),
                FixtureWindow(
                    "chat-stale-style",
                    "Superseded preference: prefer Markdown-only reports with no machine-readable fields.",
                    "chat/memory/user_pref_2025_superseded.json",
                    "memory",
                    "user_preference",
                    5,
                    {"status": "superseded", "stale": True, "superseded_by": "chat-current-pref"},
                ),
            ),
            metadata={
                "memory_scope": "durable_chat",
                "expected_current_memory": "chat-current-pref",
                "filtered_statuses": ["superseded", "stale"],
            },
            expected_ids=("chat-current-pref",),
            expected_cold_ids=("chat-stale-pref", "chat-stale-style"),
        ),
    )


def _run_one(module: Any, router: Any, row: RowFixture) -> dict[str, Any]:
    route_window_cls = getattr(module, "RouteWindow")
    route_request_cls = getattr(module, "RouteRequest")
    route_windows = [_make_route_window(route_window_cls, row, window) for window in row.windows]
    request = _make_route_request(route_request_cls, route_windows, row)
    plan = _route(router, request, route_windows)
    resolved = _resolve_plan(plan, row)
    checks = _validate_row(row, resolved)
    return {
        "row": row.name,
        "benchmark": row.benchmark,
        "capability": row.capability,
        "status": "passed",
        "checks": checks,
        "selected_window_ids": resolved.selected_ids,
        "ordered_window_ids": resolved.ordered_ids,
        "order_source": resolved.order_source,
        "tier_counts": resolved.tier_counts,
        "route_trace_steps": _trace_length(resolved.route_trace),
        "metadata_keys": sorted(resolved.metadata),
        "fixture_paths": _window_payload_summary(row),
    }


def run_validation() -> tuple[int, dict[str, Any]]:
    try:
        module = _load_router_module()
        router = _make_router(module)
    except RouterUnavailable as exc:
        return 2, {
            "status": "router_unavailable",
            "message": str(exc),
            "assumptions": list(ASSUMPTIONS),
            "rows": [],
        }
    except RouterContractError as exc:
        return 2, {
            "status": "router_contract_error",
            "message": str(exc),
            "assumptions": list(ASSUMPTIONS),
            "rows": [],
        }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in _fixtures():
        try:
            rows.append(_run_one(module, router, row))
        except (RouterContractError, ValidationFailure, AssertionError) as exc:
            failure = {
                "row": row.name,
                "benchmark": row.benchmark,
                "capability": row.capability,
                "status": "failed",
                "message": str(exc),
            }
            rows.append(failure)
            failures.append(failure)

    status = "passed" if not failures else "failed"
    return 0 if not failures else 1, {
        "status": status,
        "assumptions": list(ASSUMPTIONS),
        "rows": rows,
    }


def main() -> int:
    exit_code, report = run_validation()
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
