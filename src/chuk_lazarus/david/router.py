from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


PRIMARY_CAPABILITY_MODES: tuple[str, ...] = (
    "patch_target",
    "dependency_source",
    "symbolic_chain",
    "temporal_ordinal",
    "durable_chat_memory",
    "general_recall",
)

_TASK_TYPE_TO_MODE: dict[str, str] = {
    "repo_patch": "patch_target",
    "source_dependency_reasoning": "dependency_source",
    "symbolic_multi_hop": "symbolic_chain",
    "temporal_recall": "temporal_ordinal",
    "user_continuity": "durable_chat_memory",
    "terminal_triage": "general_recall",
}

_MODE_ALIASES: dict[str, str] = {
    **{mode: mode for mode in PRIMARY_CAPABILITY_MODES},
    **_TASK_TYPE_TO_MODE,
    "repo_patch_target": "patch_target",
    "source_dependency": "dependency_source",
    "source_dependency_routing": "dependency_source",
    "temporal_memory": "temporal_ordinal",
    "temporal_ordinal_recall": "temporal_ordinal",
    "chat_memory": "durable_chat_memory",
    "user_memory": "durable_chat_memory",
    "durable_user_memory": "durable_chat_memory",
    "task_memory": "general_recall",
}

_MODE_TO_TASK_TYPE: dict[str, str] = {
    "patch_target": "repo_patch",
    "dependency_source": "source_dependency_reasoning",
    "symbolic_chain": "symbolic_multi_hop",
    "temporal_ordinal": "temporal_recall",
    "durable_chat_memory": "user_continuity",
    "general_recall": "terminal_triage",
}

_MEMORY_FAMILY: dict[str, str] = {
    "patch_target": "code_task",
    "dependency_source": "code_task",
    "symbolic_chain": "task_memory",
    "temporal_ordinal": "chat_user_memory",
    "durable_chat_memory": "chat_user_memory",
    "general_recall": "task_memory",
}

_MODE_REASONS: dict[str, str] = {
    "patch_target": "Use repo patch-target routing for code edits, tests, and patchable files.",
    "dependency_source": "Use source/dependency routing for imports, symbols, and file spans.",
    "symbolic_chain": "Use symbolic chain routing for exact IDs and multi-hop task state.",
    "temporal_ordinal": "Use temporal ordinal recall for earlier, later, dated, or ordered events.",
    "durable_chat_memory": "Use durable chat/user memory for preferences, decisions, and user context.",
    "general_recall": "Use general product routing when no specialized capability is stronger.",
}

_MODE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "patch_target": (
        "fix",
        "bug",
        "patch",
        "implement",
        "edit",
        "failing",
        "pytest",
        "diff",
        ".py",
        ".ts",
        ".js",
    ),
    "dependency_source": (
        "import",
        "dependency",
        "depends",
        "call graph",
        "symbol",
        "source",
        "span",
        "module",
    ),
    "symbolic_chain": (
        "multi-hop",
        "chain",
        "derive",
        "exact id",
        "constraint",
        "prove",
    ),
    "temporal_ordinal": (
        "first",
        "second",
        "third",
        "previous",
        "earlier",
        "later",
        "last time",
        "yesterday",
        "today",
        "tomorrow",
        "deadline",
        "when did",
    ),
    "durable_chat_memory": (
        "remember",
        "preference",
        "prefer",
        "decision",
        "user context",
        "my goal",
        "profile",
    ),
    "general_recall": (),
}


@dataclass(frozen=True)
class RouteEvidence:
    """Traceable evidence for why a route packet selected a window."""

    evidence_id: str
    source: str
    summary: str
    reason: str
    score: float
    window_id: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["provenance"] = dict(self.provenance)
        return data


@dataclass(frozen=True)
class DavidRouteRequest:
    """Product-facing request for David's router facade."""

    task: str
    task_type: str | None = None
    capability_mode: str | None = None
    query: str | None = None
    paths: tuple[str, ...] = ()
    windows: tuple[Mapping[str, Any], ...] = ()
    workspace_path: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    index_ready: bool = False
    jit_allowed: bool = True
    residual_available: bool = False
    kv_available: bool = False
    token_budget: int = 2048
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DavidRoutePacket:
    """Deterministic routing result consumed by David product surfaces."""

    task_type: str
    capability_mode: str
    memory_family: str
    selected_windows: tuple[Mapping[str, Any], ...]
    evidence: tuple[RouteEvidence, ...]
    tier: str
    scores: Mapping[str, float]
    token_cost: int
    route_reasons: tuple[str, ...]
    provenance: Mapping[str, Any]
    residual_ready: bool
    kv_ready: bool
    index_ready: bool
    jit_required: bool
    jit_ready: bool
    readiness_status: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selected_windows"] = [dict(window) for window in self.selected_windows]
        data["evidence"] = [item.to_dict() for item in self.evidence]
        data["scores"] = dict(self.scores)
        data["provenance"] = dict(self.provenance)
        return data


class DavidProductRouter:
    """Capability-based router facade with deterministic stdlib fallback."""

    def __init__(self, backend: Any | None = None, *, prefer_backend: bool = False) -> None:
        self.backend = backend if backend is not None else _load_optional_backend(prefer_backend)

    def route(self, request: DavidRouteRequest | Mapping[str, Any] | str) -> DavidRoutePacket:
        route_request = coerce_route_request(request)
        if self.backend is not None:
            try:
                return _coerce_backend_packet(self.backend.route(route_request))
            except Exception as exc:  # pragma: no cover - defensive compatibility shim
                return _stdlib_route(
                    route_request,
                    extra_provenance={
                        "backend_error": f"{type(exc).__name__}: {exc}",
                        "backend_fallback": "stdlib",
                    },
                )
        return _stdlib_route(route_request)


def coerce_route_request(request: DavidRouteRequest | Mapping[str, Any] | str) -> DavidRouteRequest:
    if isinstance(request, DavidRouteRequest):
        return request
    if isinstance(request, str):
        return DavidRouteRequest(task=request)
    data = dict(request)
    data["paths"] = tuple(data.get("paths") or ())
    data["windows"] = tuple(_normalise_window(window, idx) for idx, window in enumerate(data.get("windows") or ()))
    metadata = data.get("metadata") or {}
    data["metadata"] = dict(metadata)
    return DavidRouteRequest(**data)


def detect_capability_mode(
    task: str,
    *,
    task_type: str | None = None,
    capability_mode: str | None = None,
    paths: Sequence[str] = (),
) -> tuple[str, str, float]:
    """Return canonical product mode, mode source, and confidence."""

    if capability_mode:
        mode = _canonical_mode(capability_mode)
        if mode:
            return mode, "request.capability_mode", 0.96
    if task_type:
        mode = _canonical_mode(task_type)
        if mode:
            return mode, "request.task_type", 0.94

    runtime_mode = _detect_with_runtime(task, paths=paths)
    if runtime_mode is not None:
        return runtime_mode

    return _detect_with_heuristics(task, paths=paths)


def route(request: DavidRouteRequest | Mapping[str, Any] | str) -> DavidRoutePacket:
    return DavidProductRouter().route(request)


def route_task(
    task: str,
    *,
    task_type: str | None = None,
    capability_mode: str | None = None,
    query: str | None = None,
    paths: Sequence[str] = (),
    windows: Sequence[Mapping[str, Any]] = (),
    workspace_path: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    index_ready: bool = False,
    jit_allowed: bool = True,
    residual_available: bool = False,
    kv_available: bool = False,
    token_budget: int = 2048,
    metadata: Mapping[str, Any] | None = None,
) -> DavidRoutePacket:
    request = DavidRouteRequest(
        task=task,
        task_type=task_type,
        capability_mode=capability_mode,
        query=query,
        paths=tuple(paths),
        windows=tuple(_normalise_window(window, idx) for idx, window in enumerate(windows)),
        workspace_path=workspace_path,
        session_id=session_id,
        user_id=user_id,
        index_ready=index_ready,
        jit_allowed=jit_allowed,
        residual_available=residual_available,
        kv_available=kv_available,
        token_budget=token_budget,
        metadata=dict(metadata or {}),
    )
    return route(request)


def _stdlib_route(
    request: DavidRouteRequest, *, extra_provenance: Mapping[str, Any] | None = None
) -> DavidRoutePacket:
    mode, mode_source, confidence = detect_capability_mode(
        request.task,
        task_type=request.task_type,
        capability_mode=request.capability_mode,
        paths=request.paths,
    )
    query = request.query or request.task
    selected, evidence, component_scores = _select_windows(mode, query, request)
    token_cost = _estimate_token_cost(query) + sum(_window_token_cost(window) for window in selected)
    residual_ready = request.residual_available or any(_bool_field(window, "residual_ready") for window in selected)
    kv_ready = request.kv_available or any(_bool_field(window, "kv_ready") for window in selected)
    jit_required = not request.index_ready
    jit_ready = jit_required and request.jit_allowed
    readiness_status = _readiness_status(request.index_ready, request.jit_allowed)
    tier = _tier_for(request.index_ready, bool(selected), residual_ready, kv_ready)
    route_reasons = tuple(
        reason
        for reason in (
            _MODE_REASONS[mode],
            _readiness_reason(readiness_status),
            _selection_reason(selected, request.windows),
        )
        if reason
    )
    provenance = {
        "router": "david.product_router.stdlib",
        "router_version": "1",
        "mode_source": mode_source,
        "workspace_path": request.workspace_path,
        "session_id": request.session_id,
        "user_id": request.user_id,
        "index_state": "ready" if request.index_ready else "missing",
        "primary_modes": PRIMARY_CAPABILITY_MODES,
        **dict(request.metadata),
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    scores = {
        "route_confidence": round(confidence, 4),
        "activation_score": round(component_scores.get("activation_score", 0.0), 4),
        "lexical_score": round(component_scores.get("lexical_score", 0.0), 4),
        "ordinal_score": round(component_scores.get("ordinal_score", 0.0), 4),
        "recency_score": round(component_scores.get("recency_score", 0.0), 4),
        "window_score": round(component_scores.get("window_score", 0.0), 4),
    }
    return DavidRoutePacket(
        task_type=_MODE_TO_TASK_TYPE[mode],
        capability_mode=mode,
        memory_family=_MEMORY_FAMILY[mode],
        selected_windows=tuple(selected),
        evidence=tuple(evidence),
        tier=tier,
        scores=scores,
        token_cost=token_cost,
        route_reasons=route_reasons,
        provenance=provenance,
        residual_ready=residual_ready,
        kv_ready=kv_ready,
        index_ready=request.index_ready,
        jit_required=jit_required,
        jit_ready=jit_ready,
        readiness_status=readiness_status,
    )


def _select_windows(
    mode: str, query: str, request: DavidRouteRequest
) -> tuple[list[Mapping[str, Any]], list[RouteEvidence], dict[str, float]]:
    scored: list[tuple[float, str, Mapping[str, Any], dict[str, float]]] = []
    for idx, raw_window in enumerate(request.windows):
        window = _normalise_window(raw_window, idx)
        components = _score_window(mode, query, window)
        score = components["window_score"]
        stable_key = str(window.get("id") or window.get("source") or idx)
        if score > 0:
            scored.append((score, stable_key, window, components))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: list[Mapping[str, Any]] = []
    evidence: list[RouteEvidence] = []
    budget_left = max(request.token_budget, 0)
    totals = {
        "activation_score": 0.0,
        "lexical_score": 0.0,
        "ordinal_score": 0.0,
        "recency_score": 0.0,
        "window_score": 0.0,
    }

    for score, stable_key, window, components in scored:
        cost = _window_token_cost(window)
        if selected and cost > budget_left:
            continue
        budget_left -= cost
        selected.append(window)
        for key in totals:
            totals[key] = max(totals[key], components.get(key, 0.0))
        evidence.append(
            RouteEvidence(
                evidence_id=f"route-evidence-{len(evidence) + 1}",
                source=str(window.get("source") or stable_key),
                summary=_window_summary(window),
                reason=_evidence_reason(mode, components),
                score=round(score, 4),
                window_id=str(window.get("id") or stable_key),
                provenance={
                    "kind": window.get("kind"),
                    "memory_family": window.get("memory_family") or window.get("family"),
                },
            )
        )
        if len(selected) >= 4:
            break

    return selected, evidence, totals


def _score_window(mode: str, query: str, window: Mapping[str, Any]) -> dict[str, float]:
    query_tokens = _tokens(query)
    text = " ".join(str(window.get(key, "")) for key in ("id", "source", "kind", "text", "summary"))
    text_tokens = _tokens(text)
    if query_tokens:
        lexical = len(query_tokens & text_tokens) / len(query_tokens)
    else:
        lexical = 0.0
    mode_bonus = _mode_bonus(mode, text)
    ordinal = _ordinal_score(mode, query, text)
    recency = _recency_score(window)
    activation = _float_field(window, "activation_score", _float_field(window, "score", 0.0))
    score = min(1.0, lexical * 0.5 + mode_bonus * 0.3 + ordinal * 0.1 + recency * 0.05 + activation * 0.05)
    return {
        "activation_score": activation,
        "lexical_score": lexical,
        "ordinal_score": ordinal,
        "recency_score": recency,
        "window_score": score,
    }


def _mode_bonus(mode: str, text: str) -> float:
    lowered = text.lower()
    bonus = 0.0
    if any(keyword in lowered for keyword in _MODE_KEYWORDS[mode]):
        bonus += 0.7
    if mode in {"patch_target", "dependency_source"} and (
        "/" in lowered or "\\" in lowered or ".py" in lowered
    ):
        bonus += 0.3
    if mode in {"temporal_ordinal", "durable_chat_memory"} and (
        "user" in lowered or "chat" in lowered or "session" in lowered
    ):
        bonus += 0.2
    return min(1.0, bonus)


def _ordinal_score(mode: str, query: str, text: str) -> float:
    if mode != "temporal_ordinal":
        return 0.0
    combined = f"{query} {text}".lower()
    ordinal_terms = ("first", "second", "third", "previous", "earlier", "later", "last time")
    date_terms = ("yesterday", "today", "tomorrow", "deadline", "date", "timestamp")
    return min(
        1.0,
        (0.6 if any(term in combined for term in ordinal_terms) else 0.0)
        + (0.4 if any(term in combined for term in date_terms) else 0.0),
    )


def _recency_score(window: Mapping[str, Any]) -> float:
    if any(key in window for key in ("timestamp", "created_at", "updated_at", "turn_index")):
        return 0.6
    text = " ".join(str(window.get(key, "")) for key in ("text", "summary")).lower()
    if any(term in text for term in ("recent", "latest", "last time", "today", "yesterday")):
        return 0.4
    return 0.0


def _canonical_mode(value: str) -> str | None:
    normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _MODE_ALIASES.get(normalised)


def _detect_with_runtime(task: str, *, paths: Sequence[str]) -> tuple[str, str, float] | None:
    try:
        from .runtime import detect_task_methodology

        methodology = detect_task_methodology(task, paths=paths)
    except Exception:
        return None
    mode = _canonical_mode(str(getattr(methodology, "capability_mode", "")))
    if not mode:
        return None
    confidence = _float_field({"confidence": getattr(methodology, "confidence", 0.0)}, "confidence", 0.7)
    return mode, "runtime.detect_task_methodology", confidence


def _detect_with_heuristics(task: str, *, paths: Sequence[str]) -> tuple[str, str, float]:
    text = f"{task or ''} {' '.join(paths)}".lower()
    scores: dict[str, int] = {mode: 0 for mode in PRIMARY_CAPABILITY_MODES}
    for mode, keywords in _MODE_KEYWORDS.items():
        scores[mode] += sum(1 for keyword in keywords if keyword in text)
    if any(path.endswith((".py", ".ts", ".js")) for path in paths):
        scores["patch_target"] += 2
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] <= 0:
        best_mode = "general_recall"
    return best_mode, "stdlib.heuristic", min(0.9, 0.45 + scores[best_mode] * 0.08)


def _normalise_window(window: Mapping[str, Any], idx: int) -> Mapping[str, Any]:
    data = dict(window)
    data.setdefault("id", data.get("window_id") or f"window-{idx + 1}")
    data.setdefault("source", data.get("path") or data.get("file") or data["id"])
    if "text" not in data:
        data["text"] = data.get("content") or data.get("summary") or ""
    return data


def _coerce_backend_packet(packet: Any) -> DavidRoutePacket:
    if isinstance(packet, DavidRoutePacket):
        return packet
    if isinstance(packet, Mapping):
        data = dict(packet)
        data["selected_windows"] = tuple(data.get("selected_windows") or ())
        data["evidence"] = tuple(
            item if isinstance(item, RouteEvidence) else RouteEvidence(**dict(item))
            for item in data.get("evidence", ())
        )
        data["route_reasons"] = tuple(data.get("route_reasons") or ())
        return DavidRoutePacket(**data)
    raise TypeError("backend route() must return DavidRoutePacket or mapping")


def _load_optional_backend(prefer_backend: bool) -> Any | None:
    if not prefer_backend:
        return None
    for module_name in (
        "chuk_lazarus.david.central_router",
        "chuk_lazarus.david.central_router_wrapper",
    ):
        try:
            module = __import__(module_name, fromlist=["router"])
        except Exception:
            continue
        for attr_name in ("router", "ROUTER", "CentralRouter", "DavidCentralRouter"):
            candidate = getattr(module, attr_name, None)
            if candidate is None:
                continue
            return candidate() if isinstance(candidate, type) else candidate
    return None


def _readiness_status(index_ready: bool, jit_allowed: bool) -> str:
    if index_ready:
        return "ready"
    if jit_allowed:
        return "jit_required"
    return "no_index"


def _readiness_reason(readiness_status: str) -> str:
    return {
        "ready": "Memory index is ready for warm or hot routing.",
        "jit_required": "No ready index was supplied; request is stable and marked for JIT indexing.",
        "no_index": "No ready index was supplied and JIT indexing is disabled.",
    }[readiness_status]


def _selection_reason(
    selected: Sequence[Mapping[str, Any]], windows: Sequence[Mapping[str, Any]]
) -> str:
    if selected:
        return f"Selected {len(selected)} evidence window(s) within the route token budget."
    if windows:
        return "No supplied window scored as route-relevant for the selected capability."
    return "No memory windows were supplied; packet carries readiness and route intent only."


def _tier_for(index_ready: bool, has_windows: bool, residual_ready: bool, kv_ready: bool) -> str:
    if not index_ready:
        return "cold"
    if has_windows and (residual_ready or kv_ready):
        return "hot"
    if has_windows:
        return "warm"
    return "cold"


def _evidence_reason(mode: str, components: Mapping[str, float]) -> str:
    strongest = max(components, key=components.get)
    return f"{mode} selected this window by {strongest.replace('_', ' ')}."


def _window_summary(window: Mapping[str, Any]) -> str:
    summary = str(window.get("summary") or window.get("text") or window.get("source") or "")
    return summary[:160]


def _window_token_cost(window: Mapping[str, Any]) -> int:
    value = window.get("tokens") or window.get("token_cost")
    if isinstance(value, int):
        return max(value, 0)
    return _estimate_token_cost(str(window.get("text") or window.get("summary") or ""))


def _estimate_token_cost(text: str) -> int:
    return max(1, (len(text) + 3) // 4) if text else 0


def _tokens(text: str) -> set[str]:
    normalised = "".join(ch.lower() if ch.isalnum() or ch in "_./\\" else " " for ch in text)
    return {token for token in normalised.split() if len(token) > 1}


def _bool_field(mapping: Mapping[str, Any], name: str) -> bool:
    return bool(mapping.get(name) or mapping.get(name.replace("_ready", "_available")))


def _float_field(mapping: Mapping[str, Any], name: str, default: float) -> float:
    value = mapping.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "PRIMARY_CAPABILITY_MODES",
    "RouteEvidence",
    "DavidRouteRequest",
    "DavidRoutePacket",
    "DavidProductRouter",
    "coerce_route_request",
    "detect_capability_mode",
    "route",
    "route_task",
]
