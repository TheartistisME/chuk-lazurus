"""Per-turn transcript records for the chat loop.

Defines pydantic models describing a single assistant/user turn, the
token-window chunk boundaries emitted during streaming, and a session-level
container that owns the list of turns.

Design principles:
- Pydantic BaseModel everywhere; no dict goop
- Monotonic turn_index assigned by the session
- UTC ISO-8601 timestamps with trailing 'Z' (microsecond precision)
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from chuk_lazarus.inference.chat import Role


def _utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 with trailing 'Z'."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _new_session_id() -> str:
    """Generate a fresh session id (UUID4 hex)."""
    return uuid4().hex


class ChunkBoundary(BaseModel):
    """A single 512-token window boundary emitted during streaming.

    start/end offsets are absolute token positions within the assistant
    reply for this turn (0-indexed, end-exclusive).
    """

    chunk_index: int
    start_token_offset: int
    end_token_offset: int
    emitted_at: str


class TurnRecord(BaseModel):
    """A single turn in the chat loop (user or assistant).

    chunk_boundaries is populated for assistant turns as the stream is
    windowed in real time; user turns typically have an empty list.
    """

    turn_index: int
    role: Role
    text: str
    started_at: str = Field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    chunk_boundaries: list[ChunkBoundary] = Field(default_factory=list)

    def mark_finished(self) -> None:
        """Stamp finished_at with the current UTC time."""
        self.finished_at = _utc_now_iso()


class ChatLoopSession(BaseModel):
    """Container for a single chat-loop session.

    Owns monotonic turn indexing and exposes the minimal surface the
    streaming code needs to record progress.
    """

    session_id: str = Field(default_factory=_new_session_id)
    started_at: str = Field(default_factory=_utc_now_iso)
    turns: list[TurnRecord] = Field(default_factory=list)

    def begin_turn(self, role: Role, text: str) -> TurnRecord:
        """Create and append a new TurnRecord with the next turn_index."""
        turn = TurnRecord(
            turn_index=len(self.turns),
            role=role,
            text=text,
        )
        self.turns.append(turn)
        return turn

    def append_chunk(self, turn: TurnRecord, boundary: ChunkBoundary) -> None:
        """Attach a chunk boundary to the given turn."""
        turn.chunk_boundaries.append(boundary)

    def finish_turn(self, turn: TurnRecord) -> None:
        """Mark the given turn as finished (sets finished_at)."""
        turn.mark_finished()
