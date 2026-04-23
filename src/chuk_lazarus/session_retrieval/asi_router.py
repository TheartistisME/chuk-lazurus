"""ASI-Evolve-adapted UCB1 router for offline window selection.

Source anchors (upstream references only; we do NOT vendor code):

- ``GAIR-NLP/ASI-Evolve/database/algorithms/ucb1.py`` — canonical UCB1 arm
  selection used by ASI-Evolve to pick the next program to evolve.
- ``GAIR-NLP/ASI-Evolve/database/algorithms/island.py`` — island / migration
  bookkeeping for the MAP-Elites-style population.
- ``GAIR-NLP/ASI-Evolve/database/database.py`` — persistence layer for
  per-program visit counts and mean rewards.
- ``GAIR-NLP/ASI-Evolve/config.yaml`` — default ``ucb1_c=1.414``,
  ``num_islands=5``, ``migration_interval=10``, ``migration_rate=0.1``,
  ``exploration_ratio=0.2``, ``exploitation_ratio=0.3``.

Formula
-------
::

    ucb1(w) = q_w + c * sqrt( ln(N) / n_w )    when n_w > 0
    ucb1(w) = +inf                              when n_w == 0

with ``c = ucb1_c = 1.414`` (from ASI-Evolve config.yaml default).

Adaptations from ASI-Evolve
---------------------------

(i) Reward signal adaptation: ASI-Evolve scores program candidates against an
    executable evaluator returning reward in [0,1]. For offline query-time
    window selection no evaluator exists. This runtime defines q_w as the
    per-candidate-pool min-max-normalized TFIDFRouter.score_window output
    from src/chuk_lazarus/inference/context/knowledge/route.py. If the pool
    has a single unique score the normalization degenerates to 1.0 and the
    adaptation surfaces reward_signal_degenerate=true in the per-query
    metadata.

(ii) Visit-count adaptation: ASI-Evolve maintains n_w across multi-iteration
    evolutionary search. For one-shot query selection all n_w start at 0 and
    the UCB1 exploration term is degenerate (+inf). This runtime operates in
    exploration-disabled mode at query time by default: when all candidates
    have n_w == 0 the tie-break falls through to q_w descending with a
    deterministic secondary key (session_id, window_id). When prior telemetry
    is present (visit_counts / mean_rewards loaded from asi_router_state.json)
    the full UCB1 formula is evaluated.

(iii) Island assignment adaptation: ASI-Evolve assigns programs to islands via
    a feature-map. For windows this runtime defines the feature-map over a
    stable tuple (session_id, recency_bucket, keyword_count_bucket) with
    recency_bucket = floor(log2(1 + session_age_seconds)) clamped to
    [0, num_islands-1] and keyword_count_bucket = min(keyword_count // 5,
    num_islands-1). The final island_id assigned to a window is the pair-sum
    modulo num_islands.

(iv) Multi-candidate emission adaptation: ASI-Evolve returns one selected
    program per island cycle. Window selection needs a ranked SET of size
    candidate_pool so axis-3 can tier. This runtime emits the full ranked
    list of AsiRouterCandidate, sorted by ucb1_score descending, with
    deterministic secondary keys.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle, load_store

ASI_ROUTER_STATE_FILENAME: str = "asi_router_state.json"
_STATE_SCHEMA_VERSION: int = 1
USER_ROLE_BOOST_ENV_VAR: str = "LAZARUS_KV_USER_ROLE_BOOST"


@dataclass(frozen=True)
class AsiRouterCandidate:
    """Ranked (handle, window_id) candidate; fields re-emit the four adaptations."""

    handle: CheckpointHandle
    window_id: int
    ucb1_score: float
    raw_router_score: float
    island_id: int
    visit_count: int
    mean_reward: float
    role: str = "unknown"
    turn_index: int = -1


@dataclass
class AsiRouterState:
    """Persistent UCB1 / island state. Composite key = f"{session_id}:{window_id}"."""

    current_island: int
    total_selections: int
    num_islands: int
    migration_interval: int
    migration_rate: float
    visit_counts: dict[str, int] = field(default_factory=dict)
    mean_rewards: dict[str, float] = field(default_factory=dict)
    islands: list[dict[str, Any]] = field(default_factory=list)
    feature_map: dict[str, int] = field(default_factory=dict)
    schema_version: int = _STATE_SCHEMA_VERSION


def _fresh_state(ni: int, mi: int, mr: float) -> AsiRouterState:
    return AsiRouterState(
        current_island=0, total_selections=0,
        num_islands=int(ni), migration_interval=int(mi), migration_rate=float(mr),
    )


def load_asi_router_state(
    archive_root: Path,
    *,
    num_islands: int = 5,
    migration_interval: int = 10,
    migration_rate: float = 0.1,
) -> AsiRouterState:
    """Load state from ``<archive_root>/asi_router_state.json``.

    Missing file OR missing ``schema_version`` returns fresh defaults.
    Mismatched schema raises :class:`RuntimeError`; no silent fallback.
    """
    archive_root = Path(archive_root)
    state_path = archive_root / ASI_ROUTER_STATE_FILENAME
    if not state_path.is_file():
        return _fresh_state(num_islands, migration_interval, migration_rate)
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"{state_path}: not a JSON object")
    sv = raw.get("schema_version")
    if sv is None:
        return _fresh_state(num_islands, migration_interval, migration_rate)
    if int(sv) != _STATE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{state_path}: schema_version={sv!r}; expected "
            f"{_STATE_SCHEMA_VERSION}. Refusing silent fallback."
        )
    return AsiRouterState(
        current_island=int(raw.get("current_island", 0)),
        total_selections=int(raw.get("total_selections", 0)),
        num_islands=int(raw.get("num_islands", num_islands)),
        migration_interval=int(raw.get("migration_interval", migration_interval)),
        migration_rate=float(raw.get("migration_rate", migration_rate)),
        visit_counts={str(k): int(v) for k, v in (raw.get("visit_counts") or {}).items()},
        mean_rewards={str(k): float(v) for k, v in (raw.get("mean_rewards") or {}).items()},
        islands=[dict(e) for e in (raw.get("islands") or [])],
        feature_map={str(k): int(v) for k, v in (raw.get("feature_map") or {}).items()},
        schema_version=_STATE_SCHEMA_VERSION,
    )


def save_asi_router_state(archive_root: Path, state: AsiRouterState) -> Path:
    """Atomically persist ``state`` via temp-file + :func:`os.replace`."""
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    state_path = archive_root / ASI_ROUTER_STATE_FILENAME
    payload: dict[str, Any] = {
        "schema_version": _STATE_SCHEMA_VERSION,
        "current_island": int(state.current_island),
        "total_selections": int(state.total_selections),
        "num_islands": int(state.num_islands),
        "migration_interval": int(state.migration_interval),
        "migration_rate": float(state.migration_rate),
        "visit_counts": {str(k): int(v) for k, v in state.visit_counts.items()},
        "mean_rewards": {str(k): float(v) for k, v in state.mean_rewards.items()},
        "islands": [dict(e) for e in state.islands],
        "feature_map": {str(k): int(v) for k, v in state.feature_map.items()},
    }
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, state_path)
    return state_path


def compute_ucb1(q_w: float, n_w: int, total_visits: int, *, ucb1_c: float = 1.414) -> float:
    """``q_w + ucb1_c * sqrt(ln(max(1, N)) / n_w)``; ``+inf`` when ``n_w == 0``."""
    if not isinstance(n_w, int):
        raise TypeError(f"n_w must be int, got {type(n_w).__name__}")
    if not isinstance(total_visits, int):
        raise TypeError(f"total_visits must be int, got {type(total_visits).__name__}")
    q_w = float(q_w); ucb1_c = float(ucb1_c)
    if n_w == 0:
        return math.inf
    if n_w < 0:
        raise ValueError(f"n_w must be >= 0, got {n_w}")
    return q_w + ucb1_c * math.sqrt(math.log(max(1, total_visits)) / n_w)


def assign_island(
    session_id: str, window_id: int, *,
    keyword_count: int, session_age_seconds: float, num_islands: int = 5,
) -> int:
    """Deterministic island id per adaptation (iii)."""
    if num_islands <= 0:
        raise ValueError(f"num_islands must be >= 1, got {num_islands}")
    age = max(0.0, float(session_age_seconds))
    recency_bucket = max(0, min(int(math.floor(math.log2(1.0 + age))), num_islands - 1))
    keyword_count_bucket = max(0, min(int(keyword_count) // 5, num_islands - 1))
    # session_id / window_id are part of the feature-map key, not bucket arithmetic.
    _ = (session_id, int(window_id))
    return (recency_bucket + keyword_count_bucket) % int(num_islands)


def advance_island(state: AsiRouterState) -> int:
    """Advance ``current_island`` and bump ``total_selections`` by 1.

    Migrates ``floor(migration_rate * len(visit_counts))`` highest-mean-reward
    keys from the outgoing island whenever ``total_selections`` becomes a
    non-zero multiple of ``migration_interval``; zero rounded count = no-op.
    """
    num_islands = max(1, int(state.num_islands))
    outgoing_island = int(state.current_island) % num_islands
    new_island = (outgoing_island + 1) % num_islands
    state.current_island = new_island
    state.total_selections = int(state.total_selections) + 1
    if state.migration_interval <= 0:
        return new_island
    if state.total_selections % int(state.migration_interval) != 0:
        return new_island
    if len(state.visit_counts) == 0:
        return new_island
    migrate_count = math.floor(float(state.migration_rate) * float(len(state.visit_counts)))
    if migrate_count <= 0:
        return new_island
    outgoing_keys = [
        k for k in state.visit_counts
        if int(state.feature_map.get(k, 0)) % num_islands == outgoing_island
    ]
    outgoing_keys.sort(key=lambda k: (-float(state.mean_rewards.get(k, 0.0)), k))
    for key in outgoing_keys[:migrate_count]:
        state.feature_map[key] = new_island
    return new_island


def _encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(t) for t in token_ids]


def _session_age_seconds(handle: CheckpointHandle) -> float:
    """Best-effort session age (seconds); never raises."""
    created_at = (handle.manifest or {}).get("created_at")
    now = time.time()
    if isinstance(created_at, (int, float)):
        return max(0.0, float(now - float(created_at)))
    if isinstance(created_at, str):
        try:
            return max(0.0, now - float(created_at))
        except ValueError:
            pass
    try:
        return max(0.0, now - handle.checkpoint_dir.stat().st_mtime)
    except OSError:
        return 0.0


def _extract_window_role_and_turn_index(window_meta: Any) -> tuple[str, int]:
    """Return ``(role, turn_index)`` with grammar-of-absence defaults."""
    role = "unknown"
    turn_index = -1
    if not isinstance(window_meta, dict):
        return role, turn_index

    raw_role = window_meta.get("role", window_meta.get("speaker_role"))
    if isinstance(raw_role, str):
        normalized_role = raw_role.strip().lower()
        if normalized_role in {"user", "assistant"}:
            role = normalized_role

    raw_turn_index = window_meta.get("turn_index")
    if raw_turn_index is not None and not isinstance(raw_turn_index, bool):
        try:
            turn_index = int(raw_turn_index)
        except (TypeError, ValueError):
            turn_index = -1

    return role, turn_index


def _user_role_boost() -> float:
    raw_value = os.environ.get(USER_ROLE_BOOST_ENV_VAR, "1.0")
    try:
        return float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"asi_route_candidates: {USER_ROLE_BOOST_ENV_VAR} must be a float, "
            f"got {raw_value!r}"
        ) from exc


# (raw_score, session_id, window_id, handle, keyword_count, role, turn_index)
_RawRow = tuple[float, str, int, CheckpointHandle, int, str, int]


def _score_all_windows(
    handles: Sequence[CheckpointHandle], query_ids: list[int],
) -> list[_RawRow]:
    """Score every window via TFIDFRouter.score_window; no silent fallback."""
    raw: list[_RawRow] = []
    for handle in handles:
        try:
            store = load_store(handle)
        except Exception as exc:
            raise RuntimeError(
                f"asi_route_candidates: failed to load store for session "
                f"{handle.session_id!r} at {handle.torch_store_dir}: {exc!r}"
            ) from exc
        window_tokens = getattr(store, "window_tokens", None)
        idf = getattr(store, "idf", None)
        if not window_tokens or idf is None:
            raise RuntimeError(
                f"asi_route_candidates: store for session "
                f"{handle.session_id!r} exposes no window_tokens/idf; "
                f"cannot score windows without silent fallback."
            )
        router = TFIDFRouter(window_tokens, idf)
        store_keywords: dict[int, list[str]] = getattr(store, "keywords", {}) or {}
        store_window_metadata: dict[int, dict[str, Any]] = (
            getattr(store, "window_metadata", {}) or {}
        )
        num_windows = int(getattr(store, "num_windows", 0) or 0)
        if num_windows <= 0:
            num_windows = max(window_tokens.keys(), default=-1) + 1
        for window_id in range(num_windows):
            if window_id not in window_tokens:
                continue
            raw_score = float(router.score_window(query_ids, int(window_id)))
            kc = len(store_keywords.get(int(window_id), []))
            meta = store_window_metadata.get(int(window_id), {})
            role, turn_index = _extract_window_role_and_turn_index(meta)
            raw.append(
                (
                    raw_score,
                    handle.session_id,
                    int(window_id),
                    handle,
                    int(kc),
                    role,
                    turn_index,
                )
            )
    return raw


def asi_route_candidates(
    handles: Sequence[CheckpointHandle],
    query_text: str,
    tokenizer: Any,
    *,
    ucb1_c: float = 1.414,
    num_islands: int = 5,
    migration_interval: int = 10,
    migration_rate: float = 0.1,
    exploration_ratio: float = 0.2,
    exploitation_ratio: float = 0.3,
    candidate_pool: int = 64,
    archive_root: Path | None = None,
) -> list[AsiRouterCandidate]:
    """Return the full ranked list of :class:`AsiRouterCandidate`.

    Pipeline: (a) load state from ``archive_root`` or fresh defaults;
    (b) :func:`_score_all_windows`; (c) truncate to top ``candidate_pool``
    by raw score; (d) min-max-normalise into ``q_w`` (degenerate → 1.0);
    (e-f) for composite key ``f"{session_id}:{window_id}"`` use
    ``state.mean_rewards[key]`` as ``q_for_ucb`` if present, else the
    normalised ``q_w``; apply :func:`compute_ucb1`; (g) :func:`advance_island`
    once; ``island_id`` via :func:`assign_island`; (h) sort ucb1_score desc,
    tie-break ``(session_id, window_id)`` asc; (i) return list. State is NOT
    persisted here. ``exploration_ratio`` / ``exploitation_ratio`` are
    retained for signature parity; downstream axes consume them for tiering.
    """
    if ucb1_c < 0.0:
        raise ValueError(f"ucb1_c must be >= 0, got {ucb1_c}")
    if num_islands <= 0:
        raise ValueError(f"num_islands must be >= 1, got {num_islands}")
    if candidate_pool <= 0:
        raise ValueError(f"candidate_pool must be >= 1, got {candidate_pool}")
    _ = (float(exploration_ratio), float(exploitation_ratio))

    if archive_root is not None:
        state = load_asi_router_state(
            Path(archive_root),
            num_islands=num_islands,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
        )
    else:
        state = _fresh_state(num_islands, migration_interval, migration_rate)

    query_ids = _encode_token_ids(tokenizer, query_text)
    raw_scored = _score_all_windows(handles, query_ids)
    if not raw_scored:
        return []

    raw_scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    pool = raw_scored[: int(candidate_pool)]
    raw_values = [item[0] for item in pool]
    hi, lo = max(raw_values), min(raw_values)
    if hi == lo:
        normalised: list[float] = [1.0 for _ in pool]
    else:
        span = hi - lo
        normalised = [(v - lo) / span for v in raw_values]
    user_role_boost = _user_role_boost()

    advance_island(state)

    candidates: list[AsiRouterCandidate] = []
    for (
        raw_score,
        session_id,
        window_id,
        handle,
        kc,
        role,
        turn_index,
    ), q_w in zip(pool, normalised):
        boosted_q_w = float(q_w)
        if user_role_boost > 1.0 and role == "user":
            boosted_q_w *= float(user_role_boost)
        composite_key = f"{session_id}:{window_id}"
        n_w = int(state.visit_counts.get(composite_key, 0))
        if composite_key in state.mean_rewards:
            mean_reward = float(state.mean_rewards[composite_key])
            q_for_ucb = mean_reward
        else:
            mean_reward = 0.0
            q_for_ucb = boosted_q_w
        ucb1_score = compute_ucb1(q_for_ucb, n_w, int(state.total_selections), ucb1_c=ucb1_c)
        island_id = assign_island(
            session_id, int(window_id),
            keyword_count=int(kc),
            session_age_seconds=_session_age_seconds(handle),
            num_islands=int(num_islands),
        )
        candidates.append(AsiRouterCandidate(
            handle=handle, window_id=int(window_id),
            ucb1_score=float(ucb1_score), raw_router_score=boosted_q_w,
            island_id=int(island_id), visit_count=int(n_w),
            mean_reward=float(mean_reward),
            role=str(role),
            turn_index=int(turn_index),
        ))

    # Primary: ucb1_score desc. At cold start every ucb1_score == +inf so
    # all candidates tie on the primary key; per adaptation (ii) the
    # secondary key is q_w (normalised raw router score) descending, and
    # the tertiary key is (session_id, window_id) ascending for
    # determinism. Without the raw_router_score tie-break the final
    # ordering collapses to pure lex on session_id at cold start —
    # defeating the point of the TF-IDF pool.
    candidates.sort(
        key=lambda c: (
            -c.ucb1_score,
            -c.raw_router_score,
            c.handle.session_id,
            c.window_id,
        )
    )
    return candidates


__all__ = [
    "ASI_ROUTER_STATE_FILENAME", "AsiRouterCandidate", "AsiRouterState",
    "advance_island", "asi_route_candidates", "assign_island",
    "compute_ucb1", "load_asi_router_state", "save_asi_router_state",
]
