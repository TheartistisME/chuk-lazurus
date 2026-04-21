"""Fresh-retriever cross-session query harness.

Given a :class:`SessionRetriever` and the per-session result dicts
returned by :func:`pipeline.run_session`, :func:`run_cross_session_queries`
exercises all three routing paths and captures the six strict-mode
assertions plus a verbatim-hit check per query.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

SIX_STRICT_KEYS: tuple[str, ...] = (
    "cuda_available",
    "model_on_cuda",
    "residual_compatible",
    "hook_fired",
    "gpu_memory_grew",
    "store_window_nonempty",
)


class QueryExecution(BaseModel):
    """One routed-and-generated query in the cross-session demo."""

    mode: str
    query_text: str
    source_session: str
    window_id: int
    routing_score: float | None
    generated_answer: str
    strict_assertions: dict[str, bool]
    verbatim_hit: bool
    planted_phrase: str
    matched_window_text: str


# Topical queries are tuned to be token-unique to the target session. The
# primitive's cross-store TF-IDF ranking compares raw scores across stores
# (not normalised globally), so a query phrased in generic English can lose
# to a store whose vocabulary makes its local TF-IDF spike. Using rare
# planted-phrase keywords keeps discrimination strong.
_TOPICAL_QUESTIONS: dict[str, str] = {
    "dubai-trip": "coral reef shimmered cobalt at dawn over Fujairah observations",
    "alice-project": "Alice Mauve Refactor Commit on the 14th",
    "sydney-conference": "keynote mentioned quokka benchmarks in Gosford speaker",
    "rust-migration": "traitful coroutine wrapper panicked at dawn Rust migration",
    "pottery-class": "kiln glowed at cone six during the thunderstorm pottery",
}

# Entity-mention queries capitalise every candidate entity token so the
# extractor in session_retrieval.entity_mention.extract_entity_tokens (which
# only harvests Title-Case / ALL-CAPS tokens) picks them up. More entity
# tokens per query drive assistant-turn windows above user-turn window 0,
# since user-turn records usually only tag a single entity.
_ENTITY_QUESTIONS: dict[str, str] = {
    "dubai-trip": "What was said about Coral Reef Observations at Fujairah Dawn?",
    "alice-project": "What did Alice decide about Mauve Refactor Commit Branch?",
    "sydney-conference": "What about Quokka Benchmarks Keynote Speaker at Gosford?",
    "rust-migration": "What happened to the Traitful Coroutine Wrapper Rust Migration?",
    "pottery-class": "What did we observe at Cone Six Kiln Firing Thunderstorm?",
}


def _build_execution(
    mode: str,
    query_text: str,
    planted_phrase: str,
    result: Any,
) -> QueryExecution:
    """Normalise a retriever ``QueryResult`` into a ``QueryExecution``."""
    answer = getattr(result, "generated_answer", "") or ""
    matched_window_text = getattr(result, "matched_window_text", "") or ""
    assertions = dict(getattr(result, "strict_assertions", {}) or {})
    # Enforce shape: must have the 6 canonical keys. Missing keys default
    # to False so downstream aggregation is safe.
    normalised_assertions: dict[str, bool] = {
        k: bool(assertions.get(k, False)) for k in SIX_STRICT_KEYS
    }
    return QueryExecution(
        mode=mode,
        query_text=query_text,
        source_session=getattr(result, "source_session", ""),
        window_id=int(getattr(result, "window_id", -1)),
        routing_score=getattr(result, "routing_score", None),
        generated_answer=answer,
        strict_assertions=normalised_assertions,
        verbatim_hit=(planted_phrase.lower() in answer.lower()),
        planted_phrase=planted_phrase,
        matched_window_text=matched_window_text,
    )


def run_cross_session_queries(
    retriever: Any,
    sessions: list[dict[str, Any]],
) -> list[QueryExecution]:
    """Execute the three query paths across the session catalogue.

    ``sessions[0]`` is queried via exact-ID, ``sessions[1]`` via topical,
    ``sessions[2]`` via entity-mention. Extra sessions are present so the
    routers have to pick across >3 checkpoints (cross-session discrimination).
    """
    if len(sessions) < 3:
        raise RuntimeError(
            f"cross_session_demo requires >= 3 sessions, got {len(sessions)}"
        )

    executions: list[QueryExecution] = []

    # Exact-ID against sessions[0].
    s0 = sessions[0]
    handle_0 = s0["planted_handle"]
    r0 = retriever.query_exact_id(handle_0)
    executions.append(
        _build_execution("exact", handle_0, s0["planted_phrase"], r0)
    )

    # Topical against sessions[1].
    s1 = sessions[1]
    topic_1 = _TOPICAL_QUESTIONS[s1["plan"]["topic"]]
    r1 = retriever.query_topical(topic_1)
    executions.append(
        _build_execution("topical", topic_1, s1["planted_phrase"], r1)
    )

    # Entity-mention against sessions[2].
    s2 = sessions[2]
    entity_q_2 = _ENTITY_QUESTIONS[s2["plan"]["topic"]]
    r2 = retriever.query_entity_mention(entity_q_2)
    executions.append(
        _build_execution("entity_mention", entity_q_2, s2["planted_phrase"], r2)
    )

    return executions


def all_assertions_pass(execs: list[QueryExecution]) -> bool:
    """Return ``True`` iff every query passes all six strict assertions
    and also carries a verbatim hit."""
    if not execs:
        return False
    for ex in execs:
        for key in SIX_STRICT_KEYS:
            if not ex.strict_assertions.get(key, False):
                return False
        if not ex.verbatim_hit:
            return False
    return True


__all__ = [
    "QueryExecution",
    "SIX_STRICT_KEYS",
    "all_assertions_pass",
    "run_cross_session_queries",
]
