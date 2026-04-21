"""Exact-dotted-handle routing via direct lookup in the primitive's
``ClauseMetadataRouter.clause_id_to_primary_windows`` dict. Bypasses the regex
extractor at ``route._CLAUSE_ID_RE`` which only matches all-digit dotted IDs,
so non-numeric (e.g. hex-prefixed) handles resolve correctly.

Given a ``"<session_id>.<turn_index>.<chunk_index>"`` handle, find the matching
:class:`CheckpointHandle` and the store-internal window id. Because axis-2
emits clause_ids whose session-id prefix is a 32-char hex string, the
primitive's ``_collect_exact_matches`` path (which goes through
``_CLAUSE_ID_RE = r"\\b\\d+(?:\\.\\d+)+\\b"``) never matches them. We
therefore consult the router's already-built ``clause_id_to_primary_windows``
mapping directly — a ``dict[str, list[int]]`` keyed by the verbatim
clause_id string — and take the first primary window.

Primitive delegated to
----------------------
``ClauseMetadataRouter.clause_id_to_primary_windows`` obtained from
``TorchKnowledgeStore._get_clause_router()``. No primitive code is modified;
the router is purely read-accessed.
"""

from __future__ import annotations

from collections.abc import Sequence

from chuk_lazarus.session_retrieval.enumeration import CheckpointHandle, load_store
from chuk_lazarus.session_retrieval.handle import parse_handle


def route_exact_id(
    handles: Sequence[CheckpointHandle],
    dotted_handle: str,
) -> tuple[CheckpointHandle, int] | None:
    """Route a dotted handle to ``(CheckpointHandle, window_id)`` or ``None``.

    Algorithm:
    1. ``parse_handle(dotted_handle)`` extracts ``(session_id, ...)`` and
       validates the dotted shape (raises ``ValueError`` on malformed input).
    2. Find the single :class:`CheckpointHandle` whose ``session_id`` matches.
       If none match, return ``None`` (caller converts to a strict
       ``ValueError``).
    3. Load that checkpoint's store and obtain its
       :class:`ClauseMetadataRouter` via ``store._get_clause_router()``. If
       the store has no window_metadata (router is ``None``), return ``None``.
    4. Look up ``dotted_handle`` verbatim in
       ``router.clause_id_to_primary_windows`` — a dict keyed by the exact
       clause_id string that the primitive built at load time. The first
       entry is the primary window for that clause.

    Returns
    -------
    ``(handle, window_id)`` when a primary window exists, otherwise ``None``.
    """
    target_session, _turn, _chunk = parse_handle(dotted_handle)
    target_handle: CheckpointHandle | None = None
    for handle in handles:
        if handle.session_id == target_session:
            target_handle = handle
            break
    if target_handle is None:
        return None

    store = load_store(target_handle)
    router = store._get_clause_router()
    if router is None:
        return None
    window_ids = router.clause_id_to_primary_windows.get(dotted_handle)
    if not window_ids:
        return None
    return target_handle, int(window_ids[0])


__all__ = ["route_exact_id"]
