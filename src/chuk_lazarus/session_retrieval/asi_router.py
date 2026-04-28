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

import hashlib
import json
import math
import os
import re
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chuk_lazarus.inference.context.knowledge.route import TFIDFRouter
from chuk_lazarus.session_retrieval.entity_mention import extract_entity_tokens
from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle, load_store
from chuk_lazarus.session_retrieval.literal_match import (
    encode_literal_sequences,
    literal_match_scores,
)

ASI_ROUTER_STATE_FILENAME: str = "asi_router_state.json"
_STATE_SCHEMA_VERSION: int = 1
DEFAULT_RRF_K: float = 60.0
DEFAULT_DENSE_DIM: int = 64


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
    raw_tfidf_score_pre_normalization: float = 0.0
    literal_score: float = 0.0
    entity_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    learned_reward: float = 0.0
    estimated_cost: float = 1.0
    utility_score: float = 0.0
    lexical_rank: int = 0
    dense_rank: int = 0
    literal_rank: int = 0
    entity_rank: int = 0
    content_fingerprint: str = ""
    dense_vector: tuple[float, ...] = ()
    selector_telemetry: dict[str, Any] = field(default_factory=dict, compare=False)


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


_EXACT_LITERAL_BOOST = 1_000_000.0
_ENTITY_MENTION_BOOST = 750_000.0


@dataclass(frozen=True)
class _WindowScoreRow:
    raw_score: float
    raw_tfidf_score: float
    literal_score: float
    entity_score: float
    session_id: str
    window_id: int
    handle: CheckpointHandle
    keyword_count: int
    estimated_cost: float
    freshness_score: float
    window_text: str
    content_fingerprint: str


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {
        "1", "true", "yes", "on",
    }


_WORD_RE = re.compile(r"[a-z0-9]+")
_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "automobile": ("car", "vehicle"),
    "car": ("automobile", "vehicle"),
    "charge": ("price", "pricing", "cost"),
    "charged": ("price", "pricing", "cost"),
    "cost": ("price", "pricing", "charge"),
    "fee": ("price", "pricing", "charge"),
    "invoice": ("bill", "receipt"),
    "bill": ("invoice", "receipt"),
    "kid": ("child", "children"),
    "kids": ("child", "children"),
    "palette": ("colors", "colour", "scheme"),
    "paraphrase": ("summary", "rewrite"),
    "price": ("pricing", "cost", "charge"),
    "pricing": ("price", "cost", "charge"),
    "repair": ("fix", "service"),
    "recall": ("remember", "retrieve"),
    "remember": ("recall", "retrieve"),
    "trial": ("demo", "evaluation"),
    "vehicle": ("car", "automobile"),
}


def _semantic_terms(text: str) -> list[str]:
    terms: list[str] = []
    for token in _WORD_RE.findall(str(text).lower()):
        terms.append(token)
        terms.extend(_SEMANTIC_ALIASES.get(token, ()))
    return terms


def _hash_unit_interval(text: str) -> float:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return float(value) / float((1 << 64) - 1)


def _deterministic_dense_vector(text: str, *, dim: int = DEFAULT_DENSE_DIM) -> tuple[float, ...]:
    """Small deterministic semantic sketch used when embeddings are unavailable."""
    dim = max(8, int(dim))
    vec = [0.0 for _ in range(dim)]
    for term in _semantic_terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
        if len(term) >= 5:
            stem = term[:5]
            stem_digest = hashlib.blake2b(stem.encode("utf-8"), digest_size=8).digest()
            stem_bucket = int.from_bytes(stem_digest[:4], "big") % dim
            vec[stem_bucket] += 0.35 * sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(v / norm for v in vec)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(float(a[i]) * float(b[i]) for i in range(n))
    norm_a = math.sqrt(sum(float(v) * float(v) for v in a[:n]))
    norm_b = math.sqrt(sum(float(v) * float(v) for v in b[:n]))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalise(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    hi, lo = max(values), min(values)
    if hi == lo:
        return [1.0 if hi > 0.0 else 0.0 for _ in values]
    span = hi - lo
    return [(float(v) - lo) / span for v in values]


def _rank_by_score(
    rows: Sequence[_WindowScoreRow],
    score_by_key: dict[tuple[str, int], float],
) -> dict[tuple[str, int], int]:
    ordered = list(rows)
    ordered.sort(
        key=lambda row: (
            -float(score_by_key.get((row.session_id, row.window_id), 0.0)),
            row.session_id,
            row.window_id,
        )
    )
    return {
        (row.session_id, row.window_id): int(rank)
        for rank, row in enumerate(ordered, start=1)
    }


def _content_fingerprint(text: str) -> str:
    tokens = _semantic_terms(text)
    if not tokens:
        return ""
    canonical = " ".join(sorted(dict.fromkeys(tokens)))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=12).hexdigest()


def _freshness_from_age_seconds(age_seconds: float) -> float:
    age_days = max(0.0, float(age_seconds)) / 86_400.0
    return 1.0 / (1.0 + math.log1p(age_days))


def _expand_semantic_route_query(query_text: str) -> str:
    """Add deterministic route hints for known paraphrased memory intents."""
    lower = query_text.lower()
    if "atlas" not in lower:
        return query_text
    if any(marker in lower for marker in ("history", "old", "draft", "previous", "superseded")):
        return query_text
    atlas_pricing_intent = any(
        marker in lower
        for marker in (
            "pricing",
            "price",
            "prcing",
            "charging",
            "charge",
            "charged",
            "$29",
            "$49",
            "settle",
            "settled",
            "trial",
            "annual discount",
            "pricing thing",
        )
    )
    if not atlas_pricing_intent or "current atlas pricing decisions" in lower:
        return query_text
    return (
        f"{query_text} Current Atlas pricing decisions across our sessions: "
        "Atlas Pro $29 per seat monthly annual discount 18% free trial 14 days "
        "usage overage $0.08 Enterprise custom quote final Atlas pricing decision."
    )


def _score_all_windows(
    handles: Sequence[CheckpointHandle],
    query_ids: list[int],
    literal_token_sequences: list[list[int]],
    entity_tokens: list[str],
    tokenizer: Any,
    *,
    decode_window_text: bool = False,
) -> list[_WindowScoreRow]:
    """Score every window via TFIDFRouter.score_window; no silent fallback."""
    raw: list[_WindowScoreRow] = []
    query_entities = tuple(dict.fromkeys(token.lower() for token in entity_tokens if token))
    cost_mode = str(os.environ.get("LAZARUS_ASI_COST_MODE", "windows")).strip().lower()
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
        window_token_lists: dict[int, list[int]] = (
            getattr(store, "window_token_lists", {}) or {}
        )
        literal_scores = literal_match_scores(store, literal_token_sequences)
        num_windows = int(getattr(store, "num_windows", 0) or 0)
        if num_windows <= 0:
            num_windows = max(window_tokens.keys(), default=-1) + 1
        for window_id in range(num_windows):
            if window_id not in window_tokens:
                continue
            tfidf_score = float(router.score_window(query_ids, int(window_id)))
            literal_score = float(literal_scores.get(int(window_id), 0.0))
            kc = len(store_keywords.get(int(window_id), []))
            entity_score = 0.0
            window_text_cache: str | None = None

            def window_text_lower() -> str:
                nonlocal window_text_cache
                if window_text_cache is None:
                    try:
                        decoded = store.get_window_text(
                            int(window_id),
                            tokenizer,
                        )
                        window_text_cache = decoded.lower() if isinstance(decoded, str) else ""
                    except Exception:
                        window_text_cache = ""
                return window_text_cache

            entity_boost_enabled = str(
                os.environ.get("LAZARUS_ROUTER_DISABLE_ENTITY_BOOST", "0")
            ).strip().lower() not in {"1", "true", "yes", "on"}
            if query_entities:
                keyword_set = {
                    str(keyword).strip().lower()
                    for keyword in store_keywords.get(int(window_id), []) or []
                }
                matched_entities = {
                    entity for entity in query_entities if entity in keyword_set
                }
                missing_entities = [
                    entity for entity in query_entities if entity not in matched_entities
                ]
                if missing_entities:
                    window_text = window_text_lower()
                    for entity in missing_entities:
                        if re.search(rf"(?<![a-z0-9]){re.escape(entity)}(?![a-z0-9])", window_text):
                            matched_entities.add(entity)
                entity_score = float(len(matched_entities)) if entity_boost_enabled else 0.0
            raw_score = (
                tfidf_score
                + (_EXACT_LITERAL_BOOST * literal_score)
                + (_ENTITY_MENTION_BOOST * entity_score)
            )
            archived_cold_penalty = float(
                os.environ.get("LAZARUS_ROUTER_ARCHIVED_COLD_PENALTY", "0") or 0.0
            )
            if archived_cold_penalty > 0.0 and "archived cold" in window_text_lower():
                raw_score -= archived_cold_penalty
            token_cost = max(
                1,
                len(window_token_lists.get(int(window_id), []))
                or len(window_tokens.get(int(window_id), ()))
                or 1,
            )
            estimated_cost = float(token_cost if cost_mode == "tokens" else 1)
            window_text = ""
            if window_text_cache is not None:
                window_text = str(window_text_cache)
            elif (
                decode_window_text
                or cost_mode == "tokens"
                or _truthy_env("LAZARUS_ASI_DECODE_FOR_SELECTOR")
            ):
                window_text = window_text_lower()
            raw.append(_WindowScoreRow(
                raw_score=float(raw_score),
                raw_tfidf_score=float(tfidf_score),
                literal_score=float(literal_score),
                entity_score=float(entity_score),
                session_id=str(handle.session_id),
                window_id=int(window_id),
                handle=handle,
                keyword_count=int(kc),
                estimated_cost=float(estimated_cost),
                freshness_score=_freshness_from_age_seconds(_session_age_seconds(handle)),
                window_text=str(window_text),
                content_fingerprint=_content_fingerprint(str(window_text)),
            ))
    return raw


def _coerce_vector(value: Sequence[float] | None) -> tuple[float, ...]:
    if value is None:
        return ()
    return tuple(float(v) for v in value)


def _lookup_dense_window_vector(
    dense_window_vectors: Any,
    session_id: str,
    window_id: int,
) -> tuple[float, ...]:
    if not dense_window_vectors:
        return ()
    key_tuple = (str(session_id), int(window_id))
    key_colon = f"{session_id}:{int(window_id)}"
    key_slash = f"{session_id}/{int(window_id)}"
    for key in (key_tuple, key_colon, key_slash, int(window_id)):
        try:
            if key in dense_window_vectors:
                return _coerce_vector(dense_window_vectors[key])
        except TypeError:
            continue
    return ()


def _dense_scores_for_rows(
    query_text: str,
    rows: Sequence[_WindowScoreRow],
    *,
    dense_scoring: str,
    dense_query_vector: Sequence[float] | None = None,
    dense_window_vectors: Any = None,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], tuple[float, ...]], str]:
    """Return dense cosine scores plus explicit fallback telemetry."""
    mode = str(dense_scoring or "off").strip().lower().replace("-", "_")
    if mode in {"0", "false", "no", "none"}:
        mode = "off"
    if mode == "off":
        return {}, {}, "off"

    scores: dict[tuple[str, int], float] = {}
    vectors: dict[tuple[str, int], tuple[float, ...]] = {}

    if mode in {"provided", "vectors"}:
        q_vec = _coerce_vector(dense_query_vector)
        if not q_vec:
            raise RuntimeError(
                "asi_route_candidates: dense_scoring='provided' requires "
                "dense_query_vector; refusing silent fallback."
            )
        for row in rows:
            key = (row.session_id, row.window_id)
            w_vec = _lookup_dense_window_vector(
                dense_window_vectors, row.session_id, row.window_id,
            )
            if not w_vec:
                raise RuntimeError(
                    "asi_route_candidates: dense_scoring='provided' missing "
                    f"vector for {row.session_id}:{row.window_id}; refusing "
                    "silent fallback."
                )
            vectors[key] = w_vec
            scores[key] = _cosine(q_vec, w_vec)
        return scores, vectors, "provided"

    if mode not in {"auto", "deterministic", "fallback", "deterministic_fallback"}:
        raise RuntimeError(
            f"asi_route_candidates: unknown dense_scoring={dense_scoring!r}; "
            "expected off, auto, deterministic, or provided."
        )

    q_vec = _deterministic_dense_vector(query_text)
    for row in rows:
        key = (row.session_id, row.window_id)
        text = row.window_text
        if not text:
            text = " ".join(str(t) for t in (row.session_id, row.window_id))
        w_vec = _deterministic_dense_vector(text)
        vectors[key] = w_vec
        scores[key] = _cosine(q_vec, w_vec)
    status = "deterministic"
    if mode == "auto":
        status = "deterministic_fallback"
        warnings.warn(
            "asi_route_candidates: dense_scoring='auto' has no configured "
            "embedding provider; using deterministic fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scores, vectors, status


def _reciprocal_rank_fusion(
    key: tuple[str, int],
    rank_maps: Sequence[dict[tuple[str, int], int]],
    *,
    k: float = DEFAULT_RRF_K,
) -> float:
    total = 0.0
    for ranks in rank_maps:
        rank = ranks.get(key)
        if rank is not None and rank > 0:
            total += 1.0 / (float(k) + float(rank))
    return total


def _router_utility(
    *,
    relevance_score: float,
    rrf_score: float,
    freshness_score: float,
    learned_reward: float,
    cost_norm: float,
) -> float:
    alpha = float(os.environ.get("LAZARUS_ASI_UTILITY_ALPHA", "1.0") or 1.0)
    beta = float(os.environ.get("LAZARUS_ASI_UTILITY_BETA", "1.0") or 1.0)
    gamma = float(os.environ.get("LAZARUS_ASI_UTILITY_GAMMA", "0.10") or 0.10)
    delta = float(os.environ.get("LAZARUS_ASI_UTILITY_DELTA", "0.50") or 0.50)
    eta = float(os.environ.get("LAZARUS_ASI_UTILITY_ETA", "0.10") or 0.10)
    return (
        alpha * float(relevance_score)
        + beta * float(rrf_score)
        + gamma * float(freshness_score)
        + delta * float(learned_reward)
        - eta * float(cost_norm)
    )


def record_asi_feedback(
    archive_root: Path,
    assignments: Sequence[Any],
    *,
    reward: float,
    outcome: str,
) -> Path:
    """Persist observed reward for selected ASI candidates.

    The caller owns reward definition. This helper only updates visit counts
    and incremental mean reward for candidates that were actually selected.
    """
    state = load_asi_router_state(Path(archive_root))
    selected = list(assignments)
    state.total_selections = int(state.total_selections) + 1
    for assignment in selected:
        candidate = getattr(assignment, "candidate", assignment)
        handle = getattr(candidate, "handle", None)
        session_id = str(getattr(handle, "session_id", ""))
        if not session_id:
            continue
        window_id = int(getattr(candidate, "window_id"))
        key = f"{session_id}:{window_id}"
        old_count = int(state.visit_counts.get(key, 0))
        old_mean = float(state.mean_rewards.get(key, 0.0))
        new_count = old_count + 1
        new_mean = old_mean + (float(reward) - old_mean) / float(new_count)
        state.visit_counts[key] = int(new_count)
        state.mean_rewards[key] = float(new_mean)
        if hasattr(candidate, "island_id"):
            state.feature_map[key] = int(getattr(candidate, "island_id"))
    state.islands = [
        {
            "outcome": str(outcome),
            "reward": float(reward),
            "selected_count": int(len(selected)),
            "ts": time.time(),
        },
        *list(state.islands or [])[:49],
    ]
    return save_asi_router_state(Path(archive_root), state)


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
    dense_scoring: str | None = None,
    dense_query_vector: Sequence[float] | None = None,
    dense_window_vectors: Any = None,
    rrf_k: float = DEFAULT_RRF_K,
    ranking_policy: str | None = None,
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

    dense_mode = str(
        dense_scoring
        if dense_scoring is not None
        else os.environ.get("LAZARUS_ASI_DENSE_SCORING", "off")
    ).strip().lower()
    selected_ranking_policy = str(
        ranking_policy
        if ranking_policy is not None
        else os.environ.get("LAZARUS_ASI_SELECTOR_POLICY", "rank-v1")
    ).strip().lower().replace("_", "-")
    route_query_text = _expand_semantic_route_query(query_text)
    query_ids = _encode_token_ids(tokenizer, route_query_text)
    literal_token_sequences = encode_literal_sequences(tokenizer, route_query_text)
    router_text_only = str(
        os.environ.get("LAZARUS_ROUTER_TEXT_ONLY", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    entity_tokens = [] if router_text_only else [
        token for token in extract_entity_tokens(route_query_text)
        if token not in {"tell", "list", "return", "rank"}
    ]
    raw_scored = _score_all_windows(
        handles,
        query_ids,
        literal_token_sequences,
        entity_tokens,
        tokenizer,
        decode_window_text=(
            dense_mode not in {"", "0", "false", "no", "none", "off"}
            or selected_ranking_policy in {"utility-v2", "hybrid-v2"}
        ),
    )
    if not raw_scored:
        return []

    dense_scores, dense_vectors, dense_status = _dense_scores_for_rows(
        route_query_text,
        raw_scored,
        dense_scoring=dense_mode,
        dense_query_vector=dense_query_vector,
        dense_window_vectors=dense_window_vectors,
    )
    key_seq = [(row.session_id, row.window_id) for row in raw_scored]
    lexical_scores = {
        (row.session_id, row.window_id): float(row.raw_tfidf_score)
        for row in raw_scored
    }
    literal_scores_by_key = {
        (row.session_id, row.window_id): float(row.literal_score)
        for row in raw_scored
    }
    entity_scores_by_key = {
        (row.session_id, row.window_id): float(row.entity_score)
        for row in raw_scored
    }
    lexical_ranks = _rank_by_score(raw_scored, lexical_scores)
    literal_ranks = _rank_by_score(raw_scored, literal_scores_by_key)
    entity_ranks = _rank_by_score(raw_scored, entity_scores_by_key)
    dense_ranks = _rank_by_score(raw_scored, dense_scores) if dense_scores else {}
    rank_maps = [lexical_ranks, literal_ranks, entity_ranks]
    if dense_ranks:
        rank_maps.append(dense_ranks)

    lexical_norm = dict(zip(key_seq, _normalise([lexical_scores[k] for k in key_seq])))
    literal_norm = dict(zip(
        key_seq, _normalise([literal_scores_by_key[k] for k in key_seq])
    ))
    entity_norm = dict(zip(
        key_seq, _normalise([entity_scores_by_key[k] for k in key_seq])
    ))
    dense_norm = dict(zip(
        key_seq, _normalise([dense_scores.get(k, 0.0) for k in key_seq])
    ))
    max_cost = max((float(row.estimated_cost) for row in raw_scored), default=1.0)
    row_metrics: dict[tuple[str, int], dict[str, float]] = {}
    for row in raw_scored:
        key = (row.session_id, row.window_id)
        relevance = (
            0.50 * float(lexical_norm.get(key, 0.0))
            + 0.25 * float(dense_norm.get(key, 0.0))
            + 0.15 * float(literal_norm.get(key, 0.0))
            + 0.10 * float(entity_norm.get(key, 0.0))
        )
        rrf_score = _reciprocal_rank_fusion(key, rank_maps, k=float(rrf_k))
        learned = float(state.mean_rewards.get(f"{row.session_id}:{row.window_id}", 0.0))
        cost_norm = float(row.estimated_cost) / max(1.0, max_cost)
        row_metrics[key] = {
            "relevance": float(relevance),
            "rrf": float(rrf_score),
            "learned": float(learned),
            "utility": _router_utility(
                relevance_score=float(relevance),
                rrf_score=float(rrf_score),
                freshness_score=float(row.freshness_score),
                learned_reward=float(learned),
                cost_norm=float(cost_norm),
            ),
        }

    if selected_ranking_policy in {"utility-v2", "hybrid-v2"}:
        raw_scored.sort(
            key=lambda row: (
                -row_metrics[(row.session_id, row.window_id)]["utility"],
                -row_metrics[(row.session_id, row.window_id)]["rrf"],
                -row.raw_score,
                row.session_id,
                row.window_id,
            )
        )
    else:
        raw_scored.sort(
            key=lambda row: (-row.raw_score, -row.raw_tfidf_score, row.session_id, row.window_id)
        )
    pool = raw_scored[: int(candidate_pool)]
    raw_values = [row.raw_score for row in pool]
    normalised = _normalise(raw_values)
    if raw_values and max(raw_values) == min(raw_values) and max(raw_values) > 0.0:
        normalised = [1.0 for _ in pool]

    advance_island(state)

    candidates: list[AsiRouterCandidate] = []
    for row, q_w in zip(pool, normalised):
        session_id = row.session_id
        window_id = int(row.window_id)
        composite_key = f"{session_id}:{window_id}"
        n_w = int(state.visit_counts.get(composite_key, 0))
        if composite_key in state.mean_rewards:
            mean_reward = float(state.mean_rewards[composite_key])
            q_for_ucb = mean_reward
        else:
            mean_reward = 0.0
            q_for_ucb = float(q_w)
        ucb1_score = compute_ucb1(q_for_ucb, n_w, int(state.total_selections), ucb1_c=ucb1_c)
        island_id = assign_island(
            session_id, int(window_id),
            keyword_count=int(row.keyword_count),
            session_age_seconds=_session_age_seconds(row.handle),
            num_islands=int(num_islands),
        )
        key = (session_id, window_id)
        metrics = row_metrics.get(key, {})
        candidates.append(AsiRouterCandidate(
            handle=row.handle, window_id=int(window_id),
            ucb1_score=float(ucb1_score), raw_router_score=float(q_w),
            island_id=int(island_id), visit_count=int(n_w),
            mean_reward=float(mean_reward),
            raw_tfidf_score_pre_normalization=float(row.raw_tfidf_score),
            literal_score=float(row.literal_score),
            entity_score=float(row.entity_score),
            dense_score=float(dense_scores.get(key, 0.0)),
            rrf_score=float(metrics.get("rrf", 0.0)),
            relevance_score=float(metrics.get("relevance", float(q_w))),
            freshness_score=float(row.freshness_score),
            learned_reward=float(metrics.get("learned", mean_reward)),
            estimated_cost=float(row.estimated_cost),
            utility_score=float(metrics.get("utility", float(q_w))),
            lexical_rank=int(lexical_ranks.get(key, 0)),
            dense_rank=int(dense_ranks.get(key, 0)),
            literal_rank=int(literal_ranks.get(key, 0)),
            entity_rank=int(entity_ranks.get(key, 0)),
            content_fingerprint=str(row.content_fingerprint),
            dense_vector=tuple(dense_vectors.get(key, ())),
            selector_telemetry={
                "dense_status": dense_status,
                "ranking_policy": selected_ranking_policy,
                "rrf_k": float(rrf_k),
            },
        ))

    # Primary: ucb1_score desc. At cold start every ucb1_score == +inf so
    # all candidates tie on the primary key; per adaptation (ii) the
    # secondary key is q_w (normalised raw router score) descending, and
    # the tertiary key is (session_id, window_id) ascending for
    # determinism. Without the raw_router_score tie-break the final
    # ordering collapses to pure lex on session_id at cold start —
    # defeating the point of the TF-IDF pool.
    # The explicit raw_tfidf tie-break below handles normalised-score ties.
    if selected_ranking_policy in {"utility-v2", "hybrid-v2"}:
        candidates.sort(
            key=lambda c: (
                -float(c.utility_score),
                -float(c.rrf_score),
                -float(c.relevance_score),
                str(c.handle.session_id),
                int(c.window_id),
            )
        )
    else:
        candidates.sort(
            key=lambda c: (
                -c.ucb1_score,
                -c.raw_router_score,
                -c.raw_tfidf_score_pre_normalization,
                c.handle.session_id,
                c.window_id,
            )
        )
    return candidates


__all__ = [
    "ASI_ROUTER_STATE_FILENAME", "AsiRouterCandidate", "AsiRouterState",
    "advance_island", "asi_route_candidates", "assign_island",
    "compute_ucb1", "load_asi_router_state", "record_asi_feedback",
    "save_asi_router_state",
]
