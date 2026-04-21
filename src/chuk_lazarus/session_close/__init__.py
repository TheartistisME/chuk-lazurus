"""axis-2 session-close emitter: turn records -> AUS3000-shaped JSON files.

Public surface for converting a finished ChatLoopSession (or an equivalent
list of TurnRecord) into a directory of per-turn-chunk JSON files that
round-trip cleanly through ``tools.build_clause_aligned_store.load_clause_records``.

Exports:
    - ``emit_session``        end-to-end emission from (turns, session_id, out_dir)
    - ``emit_from_handoff``   convenience wrapper around a TranscriptHandoff
    - ``WindDownEmitter``     class form of the above with configurable policy
    - ``compose_handle``      build ``"<sid>.<ti>.<ci>"`` clause_id
    - ``parse_handle``        inverse of ``compose_handle``
    - ``build_records``       pure (turns -> list[dict]) transformer
    - ``write_records``       serialize records to disk with the §8 naming scheme
"""

from __future__ import annotations

from chuk_lazarus.session_close.aus3000_emit import (
    build_records,
    write_records,
)
from chuk_lazarus.session_close.dotted_handle import (
    compose_handle,
    parse_handle,
)
from chuk_lazarus.session_close.wind_down import (
    WindDownEmitter,
    emit_from_handoff,
    emit_session,
)

__all__ = [
    "emit_session",
    "emit_from_handoff",
    "WindDownEmitter",
    "compose_handle",
    "parse_handle",
    "build_records",
    "write_records",
]
