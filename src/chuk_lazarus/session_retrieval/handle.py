"""Dotted-handle helpers for axis-4 retrieval.

Re-exports :func:`parse_handle` and :func:`compose_handle` from axis-2's
``chuk_lazarus.session_close.dotted_handle`` so axis-5 consumers can import them
directly from the retrieval package. Adds a small convenience predicate.

Primitive delegated to
----------------------
``chuk_lazarus.session_close.dotted_handle`` - the single source of truth for
the ``"<session_id>.<turn_index>.<chunk_index>"`` handle format.
"""

from __future__ import annotations

from chuk_lazarus.session_close.dotted_handle import compose_handle, parse_handle


def handle_belongs_to_session(handle: str, session_id: str) -> bool:
    """Return ``True`` iff ``handle`` parses to a tuple whose session matches.

    Parameters
    ----------
    handle:
        A dotted handle string ``"<session_id>.<turn_index>.<chunk_index>"``.
    session_id:
        The candidate session id.

    Raises
    ------
    ValueError
        Propagated from :func:`parse_handle` if ``handle`` is malformed.
    """
    parsed_session, _turn, _chunk = parse_handle(handle)
    return parsed_session == session_id


__all__ = [
    "compose_handle",
    "handle_belongs_to_session",
    "parse_handle",
]
