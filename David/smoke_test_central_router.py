#!/usr/bin/env python3
"""Standalone smoke tests for David's central router.

Run from the repository root:
    python3 David/smoke_test_central_router.py
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


TIERS = ("HOT", "WARM", "COLD")
MODES = (
    "temporal_ordinal",
    "symbolic_chain",
    "dependency_source",
    "patch_target",
    "durable_chat_memory",
    "general_recall",
)


def load_router_module() -> Any:
    router_path = Path(__file__).with_name("central router.py")
    if not router_path.exists():
        raise FileNotFoundError(
            f"Expected router module at {router_path}. "
            "Worker A owns that file; this smoke test imports it by path."
        )

    spec = importlib.util.spec_from_file_location("david_central_router", router_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {router_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for api_name in ("RouteRequest", "RouteWindow", "CentralRouter"):
        assert hasattr(module, api_name), f"central router.py must expose {api_name}"
    return module


def make_instance(cls: type, **payload: Any) -> Any:
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**payload)

    parameters = signature.parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    kwargs = payload if accepts_kwargs else {k: v for k, v in payload.items() if k in parameters}

    missing = [
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in kwargs
    ]
    assert not missing, f"{cls.__name__} constructor needs unsupported args: {missing}"
    return cls(**kwargs)


def make_window(
    module: Any,
    window_id: str,
    tier: str,
    content: str,
    modes: Iterable[str],
    **metadata: Any,
) -> Any:
    modes = tuple(modes)
    source_path = metadata.get("source_path") or metadata.get("path") or ""
    payload = {
        "window_id": window_id,
        "id": window_id,
        "key": window_id,
        "name": window_id,
        "tier": tier,
        "tier_name": tier,
        "content": content,
        "text": content,
        "body": content,
        "summary": content,
        "source_path": source_path,
        "path": source_path,
        "source": source_path,
        "capability_mode": modes[0] if modes else None,
        "mode": modes[0] if modes else None,
        "capability_modes": list(modes),
        "capabilities": list(modes),
        "modes": list(modes),
        "eligible_modes": list(modes),
        "metadata": dict(metadata),
        "meta": dict(metadata),
        "tags": list(metadata.get("tags", ())),
        "scope": metadata.get("scope"),
        "scope_key": metadata.get("scope_key") or metadata.get("scope"),
        "canonical_request": metadata.get("canonical_request"),
        "session_turn_index": metadata.get("session_turn_index"),
        "memory_scope": metadata.get("memory_scope"),
        "source_authority": metadata.get("source_authority"),
        "source_author": metadata.get("source_author"),
        "ordinal": metadata.get("ordinal"),
        "created_at": metadata.get("created_at", "2026-05-02T00:00:00Z"),
        "updated_at": metadata.get("updated_at", "2026-05-02T00:00:00Z"),
        "tokens": len(content.split()),
        "token_count": len(content.split()),
        "route_trace": metadata.get("route_trace", []),
        "stale": metadata.get("stale", False),
        "superseded_by": metadata.get("superseded_by"),
    }
    return make_instance(module.RouteWindow, **payload)


def make_request(module: Any, mode: str, query: str, **metadata: Any) -> Any:
    payload = {
        "request_id": f"smoke-{mode}",
        "id": f"smoke-{mode}",
        "mode": mode,
        "capability_mode": mode,
        "capability": mode,
        "query": query,
        "text": query,
        "prompt": query,
        "scope": metadata.get("scope"),
        "scope_key": metadata.get("scope_key") or metadata.get("scope"),
        "canonical_request": metadata.get("canonical_request"),
        "ordinal": metadata.get("ordinal"),
        "path_hints": tuple(metadata.get("path_hints", ())),
        "identifiers": tuple(metadata.get("identifiers", ())),
        "entities": tuple(metadata.get("entities", ())),
        "metadata": dict(metadata),
        "meta": dict(metadata),
        "max_windows": metadata.get("max_windows", 12),
        "top_k": metadata.get("top_k", 12),
        "now": metadata.get("now", "2026-05-02T00:00:00Z"),
    }
    return make_instance(module.RouteRequest, **payload)


def route(module: Any, mode: str, query: str, windows: list[Any], **metadata: Any) -> Any:
    request = make_request(module, mode, query, **metadata)

    try:
        router = module.CentralRouter()
    except TypeError:
        router = module.CentralRouter(windows)

    assert hasattr(router, "route"), "CentralRouter must expose route(request, windows)"
    return router.route(request, windows)


def field(value: Any, names: Iterable[str], default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def unwrap_window(value: Any) -> Any:
    return field(value, ("window", "route_window", "source_window"), value)


def norm_tier(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    elif hasattr(value, "name") and str(value).upper() not in TIERS:
        value = value.name
    tier = str(value).upper()
    return tier if tier in TIERS else None


def window_id(value: Any) -> str:
    value = unwrap_window(value)
    if isinstance(value, str):
        assert norm_tier(value) is None, f"Tier label is not a window id: {value!r}"
        return value
    found = field(value, ("window_id", "id", "key", "name"))
    assert found is not None, f"Window-like object lacks an id: {value!r}"
    return str(found)


def window_tier(value: Any) -> str | None:
    value = unwrap_window(value)
    return norm_tier(field(value, ("tier", "tier_name")))


def selected_windows(plan: Any) -> list[Any]:
    candidates = field(plan, ("selected_windows", "ranked_windows", "windows", "candidates"))
    assert candidates is not None, (
        "RoutePlan must expose selected_windows, ranked_windows, windows, or candidates"
    )
    return list(candidates)


def materialization_windows(plan: Any) -> Any:
    candidates = field(plan, ("materialization_windows", "materialized_windows"))
    if candidates is None:
        candidates = field(plan, ("materialization_plan", "materialization"))
    assert candidates is not None, (
        "RoutePlan must expose materialization_windows or materialization_plan "
        "across HOT/WARM/COLD tiers"
    )
    return candidates


def tiers_from(value: Any) -> set[str]:
    tiers: set[str] = set()
    if value is None:
        return tiers
    tier_window_ids = field(value, ("tier_window_ids",))
    if isinstance(tier_window_ids, Mapping):
        for tier, window_ids in tier_window_ids.items():
            normalized = norm_tier(tier)
            if normalized and window_ids:
                tiers.add(normalized)
        if tiers:
            return tiers
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_tier = norm_tier(key)
            if key_tier and item:
                tiers.add(key_tier)
            tiers.update(tiers_from(item))
        return tiers
    if isinstance(value, (str, bytes)):
        tier = norm_tier(value)
        return {tier} if tier else set()
    try:
        iterator = iter(value)
    except TypeError:
        tier = window_tier(value) or norm_tier(value)
        return {tier} if tier else set()

    for item in iterator:
        tier = window_tier(item) or norm_tier(item)
        if tier:
            tiers.add(tier)
        else:
            tiers.update(tiers_from(item))
    return tiers


def selected_ids(plan: Any) -> list[str]:
    return [window_id(item) for item in selected_windows(plan)]


def materialized_ids(plan: Any) -> list[str]:
    materialized = materialization_windows(plan)
    tier_window_ids = field(materialized, ("tier_window_ids",))
    if isinstance(tier_window_ids, Mapping):
        output: list[str] = []
        for value in tier_window_ids.values():
            if isinstance(value, (str, bytes)):
                output.append(str(value))
            else:
                output.extend(str(item) for item in value)
        return output
    materialized_plan_windows = field(materialized, ("windows",))
    if materialized_plan_windows is not None:
        return [window_id(item) for item in materialized_plan_windows]
    if isinstance(materialized, Mapping):
        items: list[Any] = []
        for key, value in materialized.items():
            if norm_tier(key):
                if isinstance(value, (str, bytes)):
                    items.append(value)
                else:
                    try:
                        items.extend(value)
                    except TypeError:
                        items.append(value)
            elif norm_tier(value):
                items.append(key)
            else:
                items.append(key)
                if isinstance(value, (str, bytes)):
                    items.append(value)
                else:
                    try:
                        items.extend(value)
                    except TypeError:
                        items.append(value)
        return [window_id(item) for item in items if norm_tier(item) is None]
    return [window_id(item) for item in materialized]


def jsonish(value: Any) -> str:
    def default(item: Any) -> Any:
        if hasattr(item, "__dict__"):
            return vars(item)
        return repr(item)

    return json.dumps(value, default=default, sort_keys=True)


def evidence_blob(plan: Any) -> dict[str, Any]:
    return {
        "evidence": field(plan, ("chain_evidence", "evidence", "route_evidence")),
        "evidence_supports": field(plan, ("evidence_supports", "supports"), ()),
        "router_metadata": field(plan, ("router_metadata", "metadata"), {}),
        "candidates": selected_windows(plan),
    }


def route_trace_blob(plan: Any) -> dict[str, Any]:
    candidate_traces = [
        field(candidate, ("route_trace",), {})
        for candidate in selected_windows(plan)
    ]
    support_traces = [
        field(support, ("route_trace",), {})
        for support in field(plan, ("evidence_supports", "supports"), ())
    ]
    return {
        "route_trace": field(plan, ("route_trace", "trace", "routing_trace")),
        "candidate_traces": candidate_traces,
        "support_traces": support_traces,
        "router_metadata": field(plan, ("router_metadata", "metadata"), {}),
    }


def assert_full_tier_invariant(plan: Any, mode: str) -> dict[str, list[str]]:
    assert hasattr(plan, "assert_tier_coverage"), (
        "RoutePlan must expose assert_tier_coverage()"
    )
    plan.assert_tier_coverage()

    assignments = field(plan, ("tier_assignments", "assignments"))
    assert assignments is not None, "RoutePlan must expose tier_assignments"
    assignment_tiers = tiers_from(assignments)
    materialization_tiers = tiers_from(materialization_windows(plan))

    expected = set(TIERS)
    assert expected <= assignment_tiers, (
        f"{mode} must assign all tiers; got {sorted(assignment_tiers)}"
    )
    assert expected <= materialization_tiers, (
        f"{mode} must materialize all tiers; got {sorted(materialization_tiers)}"
    )
    return {
        "assignment_tiers": sorted(assignment_tiers),
        "materialization_tiers": sorted(materialization_tiers),
    }


def assert_before(ids: list[str], earlier: str, later: str) -> None:
    assert earlier in ids, f"Expected {earlier} in selected ids: {ids}"
    assert later in ids, f"Expected {later} in selected ids: {ids}"
    assert ids.index(earlier) < ids.index(later), (
        f"Expected {earlier} to rank before {later}; got {ids}"
    )


def summarize(mode: str, plan: Any, **extra: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "mode": mode,
        "selected_ids": selected_ids(plan),
    }
    summary.update(assert_full_tier_invariant(plan, mode))
    summary.update(extra)
    return summary


def smoke_temporal_ordinal(module: Any) -> dict[str, Any]:
    mode = "temporal_ordinal"
    windows = [
        make_window(
            module,
            "temporal-alpha-1",
            "HOT",
            "MRCR duplicate request in scope alpha: request item 1 asks for value orchid.",
            [mode],
            scope="alpha",
            scope_key="alpha",
            ordinal=1,
            session_turn_index=1,
            duplicate_key="mrcr-alpha-value",
        ),
        make_window(
            module,
            "temporal-alpha-2",
            "WARM",
            "MRCR duplicate request in scope alpha: request item 2 asks for value quartz.",
            [mode],
            scope="alpha",
            scope_key="alpha",
            ordinal=2,
            session_turn_index=2,
            duplicate_key="mrcr-alpha-value",
        ),
        make_window(
            module,
            "temporal-alpha-3",
            "COLD",
            "MRCR duplicate request in scope alpha: request item 3 asks for value umber.",
            [mode],
            scope="alpha",
            scope_key="alpha",
            ordinal=3,
            session_turn_index=3,
            duplicate_key="mrcr-alpha-value",
        ),
        make_window(
            module,
            "temporal-beta-2",
            "HOT",
            "MRCR duplicate request in scope beta: request item 2 asks for value wrong-scope.",
            [mode],
            scope="beta",
            scope_key="beta",
            ordinal=2,
            session_turn_index=2,
            duplicate_key="mrcr-beta-value",
        ),
    ]
    plan = route(
        module,
        mode,
        "Within scope alpha, select the second duplicate MRCR-style request.",
        windows,
        scope="alpha",
        scope_key="alpha",
        ordinal=2,
    )
    ids = selected_ids(plan)
    assert ids[0] == "temporal-alpha-2", (
        "temporal_ordinal must rank the second scoped duplicate first; "
        f"got {ids}"
    )
    assert "temporal-beta-2" not in ids[:3], f"Wrong-scope duplicate leaked early: {ids}"
    return summarize(mode, plan, top_id=ids[0])


def smoke_symbolic_chain(module: Any) -> dict[str, Any]:
    mode = "symbolic_chain"
    final_value = "aurora-citadel"
    windows = [
        make_window(module, "chain-a-to-b", "HOT", "A -> B", [mode]),
        make_window(
            module,
            "chain-b-to-final-key",
            "WARM",
            "B -> KAPPA_42",
            [mode],
        ),
        make_window(
            module,
            "chain-final-value",
            "COLD",
            f"KAPPA_42 -> {final_value.replace('-', '_')}",
            [mode],
        ),
        make_window(
            module,
            "chain-distractor",
            "HOT",
            "Symbol A resolves to unused branch Z in an unrelated scope.",
            [mode],
            scope="other",
        ),
    ]
    plan = route(
        module,
        mode,
        "Resolve symbolic chain A to the final value.",
        windows,
        identifiers=("A",),
    )
    ids = selected_ids(plan)
    for expected_id in ("chain-a-to-b", "chain-b-to-final-key", "chain-final-value"):
        assert expected_id in ids, f"symbolic_chain missed {expected_id}: {ids}"

    evidence = evidence_blob(plan)
    evidence_text = jsonish(evidence)
    for expected_text in ("A", "B", "KAPPA_42", final_value.replace("-", "_")):
        assert expected_text in evidence_text, (
            f"chain evidence must include {expected_text!r}; got {evidence_text}"
        )
    return summarize(
        mode,
        plan,
        chain_evidence_recorded=True,
        resolved_value=final_value.replace("-", "_"),
    )


def smoke_dependency_source(module: Any) -> dict[str, Any]:
    mode = "dependency_source"
    windows = [
        make_window(
            module,
            "dep-source-router",
            "HOT",
            "Implementation source for payment route graph and dependency injection.",
            [mode],
            source_path="src/payments/router.py",
            kind="source",
            recursive_map={"root": "payment-root", "dependency": "src/payments/dependencies.py"},
            hop="PAYMENT_GATEWAY",
        ),
        make_window(
            module,
            "dep-source-dependencies",
            "WARM",
            "Dependency provider source for payment gateways and ledger client.",
            [mode],
            source_path="src/payments/dependencies.py",
            kind="dependency",
            recursive_map={"root": "payment-root", "provider": "ledger-client"},
            hop="PAYMENT_GATEWAY",
        ),
        make_window(
            module,
            "dep-source-settings",
            "COLD",
            "Durable configuration source for payment dependency environment.",
            [mode],
            source_path="src/payments/settings.py",
            kind="dependency_source",
            recursive_map={"root": "payment-root", "reads": "PAYMENT_GATEWAY"},
            hop="PAYMENT_GATEWAY",
        ),
        make_window(
            module,
            "dep-doc-readme",
            "HOT",
            "README overview for payment routes, not the source of dependencies.",
            [mode],
            source_path="docs/payments.md",
            kind="docs",
        ),
    ]
    plan = route(
        module,
        mode,
        "Find the source windows for payment dependencies in src/payments/router.py.",
        windows,
        path_hints=("src/payments/router.py", "src/payments/dependencies.py"),
        identifiers=("payment", "dependencies", "PAYMENT_GATEWAY"),
    )
    ids = selected_ids(plan)
    assert_before(ids, "dep-source-router", "dep-doc-readme")
    assert_before(ids, "dep-source-dependencies", "dep-doc-readme")

    trace = route_trace_blob(plan)
    trace_text = jsonish(trace)
    for expected_text in ("payment-root", "src/payments/dependencies.py", "PAYMENT_GATEWAY"):
        assert expected_text in trace_text, (
            f"route_trace must preserve {expected_text!r}; got {trace_text}"
        )
    return summarize(mode, plan, route_trace_preserved=True)


def smoke_patch_target(module: Any) -> dict[str, Any]:
    mode = "patch_target"
    windows = [
        make_window(
            module,
            "patch-src-login",
            "HOT",
            "Implementation source LoginWidget validates redirect target and session token.",
            [mode],
            source_path="src/ui/LoginWidget.tsx",
            kind="source",
        ),
        make_window(
            module,
            "patch-test-login",
            "WARM",
            "Tests for LoginWidget redirect validation behavior.",
            [mode],
            source_path="tests/ui/LoginWidget.test.tsx",
            kind="test",
        ),
        make_window(
            module,
            "patch-doc-login",
            "COLD",
            "Documentation for login redirects and user-facing wording.",
            [mode],
            source_path="docs/login.md",
            kind="docs",
        ),
        make_window(
            module,
            "patch-asset-login",
            "COLD",
            "Static login illustration asset, not an implementation target.",
            [mode],
            source_path="assets/login.svg",
            kind="asset",
        ),
    ]
    plan = route(module, mode, "Patch the LoginWidget redirect validation bug.", windows)
    ids = selected_ids(plan)
    assert ids[0] == "patch-src-login", f"patch_target must rank implementation first: {ids}"
    for lower_priority_id in ("patch-test-login", "patch-doc-login", "patch-asset-login"):
        assert_before(ids, "patch-src-login", lower_priority_id)
    return summarize(mode, plan, top_id=ids[0])


def smoke_durable_chat_memory(module: Any) -> dict[str, Any]:
    mode = "durable_chat_memory"
    windows = [
        make_window(
            module,
            "memory-current-user",
            "HOT",
            "Current user memory: timezone Australia/Perth; preferred editor Helix.",
            [mode],
            subject="user",
            current=True,
            memory_scope="current",
            source_authority="user",
            updated_at="2026-05-02T03:00:00Z",
        ),
        make_window(
            module,
            "memory-current-project",
            "WARM",
            "Current project memory: Lazarus David router smoke tests are active.",
            [mode],
            subject="project",
            current=True,
            memory_scope="current",
            source_authority="tool",
            updated_at="2026-05-02T02:00:00Z",
        ),
        make_window(
            module,
            "memory-current-stable",
            "COLD",
            "Current durable preference: use concise JSON summaries for smoke results.",
            [mode],
            subject="user",
            current=True,
            memory_scope="current",
            source_authority="user",
            updated_at="2026-04-30T00:00:00Z",
        ),
        make_window(
            module,
            "memory-stale-user",
            "HOT",
            "Stale user memory: timezone America/New_York; preferred editor Vim.",
            [mode],
            subject="user",
            memory_scope="current",
            source_authority="user",
            stale=True,
            superseded_by="memory-current-user",
            updated_at="2025-01-01T00:00:00Z",
        ),
        make_window(
            module,
            "memory-superseded-current",
            "WARM",
            "Superseded memory: user now works in a retired project called OldRouter.",
            [mode],
            subject="user",
            memory_scope="current",
            source_authority="user",
            superseded_by="memory-current-project",
            updated_at="2025-06-01T00:00:00Z",
        ),
    ]
    plan = route(
        module,
        mode,
        "What is my current timezone, editor, and current project state?",
        windows,
        current_state=True,
        memory_scope="current",
    )
    ids = selected_ids(plan)
    assert ids[0] == "memory-current-user", (
        f"durable_chat_memory must prefer current user memory first: {ids}"
    )
    banned = {"memory-stale-user", "memory-superseded-current"}
    assert banned.isdisjoint(ids), f"Stale/superseded memories selected: {ids}"
    materialized = materialized_ids(plan)
    assert banned.isdisjoint(materialized), (
        f"Stale/superseded memories materialized: {materialized}"
    )
    return summarize(mode, plan, top_id=ids[0])


def smoke_general_recall(module: Any) -> dict[str, Any]:
    mode = "general_recall"
    windows = [
        make_window(
            module,
            "recall-lexical-hot",
            "HOT",
            "Alpha spindle cache invalidation runbook with exact lexical query terms.",
            [mode],
            candidate_family="lexical",
        ),
        make_window(
            module,
            "recall-hybrid-warm",
            "WARM",
            "Spindle lifecycle note: eviction flushes stale cache entries after invalidation.",
            [mode],
            candidate_family="hybrid",
        ),
        make_window(
            module,
            "recall-lexical-cold",
            "COLD",
            "Archive note for alpha spindle token map and cache namespace.",
            [mode],
            candidate_family="lexical",
        ),
        make_window(
            module,
            "recall-distractor",
            "HOT",
            "Unrelated recipe text about deploy calendars.",
            [mode],
            candidate_family="noise",
        ),
    ]
    plan = route(module, mode, "alpha spindle cache invalidation notes", windows)
    ids = selected_ids(plan)
    lexical = {"recall-lexical-hot", "recall-lexical-cold"}
    hybrid = {"recall-hybrid-warm"}
    assert lexical.intersection(ids), f"general_recall must return lexical candidates: {ids}"
    assert hybrid.intersection(ids), f"general_recall must return hybrid candidates: {ids}"
    assert "recall-distractor" not in ids[:3], f"Noise ranked too early: {ids}"
    return summarize(mode, plan)


def main() -> None:
    module = load_router_module()
    smoke_tests = (
        smoke_temporal_ordinal,
        smoke_symbolic_chain,
        smoke_dependency_source,
        smoke_patch_target,
        smoke_durable_chat_memory,
        smoke_general_recall,
    )
    results = [test(module) for test in smoke_tests]
    print(
        json.dumps(
            {
                "ok": True,
                "modes": list(MODES),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
