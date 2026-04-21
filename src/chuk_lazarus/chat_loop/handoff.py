"""Transcript handoff hook for axis-2 consumption.

TranscriptHandoff is the surface axis-2 (AUS3000 emitter) uses to read
the per-turn records produced by the chat loop. The primary path is
`snapshot()`, which returns a deep copy so downstream mutations cannot
affect the live session; `raw_session()` is the explicit opt-in for
callers that want the live object.
"""

from __future__ import annotations

from typing import Any

from chuk_lazarus.chat_loop.session import ChatLoopSession, TurnRecord


class TranscriptHandoff:
    """Hand off chat-loop transcript records to downstream consumers."""

    def __init__(self, session: ChatLoopSession) -> None:
        self._session = session

    def snapshot(self) -> list[TurnRecord]:
        """Return a deep copy of the current turn records.

        Safe to pass to axis-2 pipelines; downstream mutations cannot
        affect the live session.
        """
        return [turn.model_copy(deep=True) for turn in self._session.turns]

    def raw_session(self) -> ChatLoopSession:
        """Return the underlying session by reference.

        Axis-2 pipeline-leads opt in explicitly when they need the live
        object (e.g. to observe turns as they are appended).
        """
        return self._session

    def as_dict_list(self) -> list[dict[str, Any]]:
        """Return turn records as JSON-serializable dicts."""
        return [turn.model_dump(mode="json") for turn in self._session.turns]
