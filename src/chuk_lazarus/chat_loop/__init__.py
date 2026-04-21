"""Chat loop package for turn-aligned local REPL against Gemma.

Provides:
- ChatLoopSession: container for per-turn transcript records
- TurnRecord / ChunkBoundary: pydantic models describing a turn and its streamed
  512-token windows
- StreamingWindower: hot-path windowing of assistant token stream into 512-token
  chunks with 64-token overlap (matches axis-3 defaults)
- TranscriptHandoff: handoff hook surfacing raw turn records for axis-2

Heavy dependencies (torch, transformers) are deferred to function-scope imports
in cli.py so `import chuk_lazarus.chat_loop` stays cheap.
"""

from __future__ import annotations

from chuk_lazarus.chat_loop.handoff import TranscriptHandoff
from chuk_lazarus.chat_loop.session import ChatLoopSession, ChunkBoundary, TurnRecord
from chuk_lazarus.chat_loop.streaming import StreamingWindower

__all__ = [
    "ChatLoopSession",
    "ChunkBoundary",
    "StreamingWindower",
    "TranscriptHandoff",
    "TurnRecord",
]
