"""
Session-keyed WARM residual cache for the torch Qwen serve path.

Persists per-conversation ``past_key_values`` + boundary residual so that
subsequent turns in a multi-turn chat can skip the split-prefill forward.
Each cache hit is a pure "extend by delta tokens" operation: feed only the
new tokens appended since the last turn through a single cached-KV forward,
then continue into the normal decode loop.

Torch-agnostic at import time
-----------------------------
This module must not ``import torch`` at module scope. It manipulates
``past_key_values`` only via the eviction callback injected by the runtime,
and only knows that entries have a ``.past_key_values`` field that owns GPU
memory. The runtime is responsible for providing an eviction hook that can
safely ``del`` / ``torch.cuda.empty_cache()`` when an entry is dropped.

Keying strategy
---------------
Entries are keyed by a **conversation id**, resolved in order:

1. ``user`` field on the OpenAI chat request (``InternalRequest.conversation_id``).
2. If no explicit id, a **prefix-hash fallback**: the runtime looks up the
   longest cached entry whose ``cold_token_ids`` is a prefix of the current
   tokenised prompt and reuses that entry under a deterministic anonymous
   key derived from its prefix.

The cache enforces an LRU bound and calls the eviction hook on every evicted
entry so the bounded GPU memory envelope stays honest.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

DEFAULT_MAX_SESSIONS = 8


@dataclass
class ResidualSessionEntry:
    """One conversation's cached state.

    Fields
    ------
    cold_token_ids
        Every token seen in this conversation so far (including the tokens
        the model just generated). Used for prefix-match validation on the
        next turn.
    past_key_values
        The live torch ``Cache`` object on GPU, bounded to the same hot
        budget. Opaque to this module.
    boundary_residual
        The most recently captured pre-final-norm last-position hidden state
        (Markov-sufficient statistic for everything before the hot window).
    boundary_residual_position
        Absolute token index at which ``boundary_residual`` was captured.
    total_seen_tokens
        Absolute count of tokens processed so far. Drives RoPE position ids
        on the delta forward.
    hot_window_start
        Absolute index of the first token currently represented in the hot KV.
    last_used_at
        Monotonic time of last access — drives LRU eviction.
    conversation_id
        Resolved cache key. Either an explicit ``user`` id from the OpenAI
        request or the prefix-hash fallback derived from the first tokens.
    """

    cold_token_ids: list[int] = field(default_factory=list)
    past_key_values: Any | None = None
    boundary_residual: Any | None = None
    boundary_residual_position: int = -1
    total_seen_tokens: int = 0
    hot_window_start: int = 0
    last_used_at: float = field(default_factory=time.monotonic)
    conversation_id: str = ""

    def touch(self) -> None:
        """Mark this entry as recently used."""
        self.last_used_at = time.monotonic()


def _free_entry_default(entry: ResidualSessionEntry) -> None:
    """Default eviction hook — drops the Python references.

    The torch runtime installs a richer callback that also calls
    ``torch.cuda.empty_cache()``. We only cover the no-torch / testing path
    here so the module remains importable without torch.
    """

    entry.past_key_values = None
    entry.boundary_residual = None


class ResidualSessionCache:
    """Session-keyed LRU cache of hot KV + boundary residual snapshots.

    Thread-safe: all mutating ops hold a single lock. The cache assumes
    single-writer semantics per conversation id — concurrent generations for
    the same id will race on the same entry and the second write wins. In
    practice the ``ModelEngine._lock`` already serialises generation per
    engine, so this is not a concern for the current server.
    """

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        eviction_hook: Callable[[ResidualSessionEntry], None] | None = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError(f"max_sessions must be >= 1, got {max_sessions}")
        self._max_sessions = int(max_sessions)
        self._entries: OrderedDict[str, ResidualSessionEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._eviction_hook = eviction_hook or _free_entry_default

    # ── Introspection ────────────────────────────────────────────────────

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries.keys())

    # ── Lookup / store ───────────────────────────────────────────────────

    def get(self, conversation_id: str) -> ResidualSessionEntry | None:
        """Return the cached entry for ``conversation_id`` or ``None``."""
        with self._lock:
            entry = self._entries.get(conversation_id)
            if entry is None:
                return None
            self._entries.move_to_end(conversation_id)
            entry.touch()
            return entry

    def find_prefix_match(
        self, prompt_token_ids: Iterable[int]
    ) -> ResidualSessionEntry | None:
        """Find the longest cached entry whose cold tokens prefix the prompt.

        Used when no explicit conversation id is supplied. We scan every
        entry and return the one whose ``cold_token_ids`` is a proper prefix
        of ``prompt_token_ids`` and is longest. Returns ``None`` on no match.

        Tokenisation is deterministic across turns for the chat template
        used by Qwen models, so an exact prefix match is a reliable signal
        that the new prompt is "the same conversation plus a delta".
        """
        token_list = list(prompt_token_ids)
        with self._lock:
            best: ResidualSessionEntry | None = None
            for entry in self._entries.values():
                cold = entry.cold_token_ids
                cold_len = len(cold)
                if cold_len == 0 or cold_len >= len(token_list):
                    continue
                if token_list[:cold_len] != cold:
                    continue
                if best is None or cold_len > len(best.cold_token_ids):
                    best = entry
            if best is not None:
                self._entries.move_to_end(best.conversation_id)
                best.touch()
            return best

    def put(self, entry: ResidualSessionEntry) -> None:
        """Insert or replace ``entry`` in the cache, evicting LRU if full."""
        if not entry.conversation_id:
            raise ValueError("ResidualSessionEntry.conversation_id must be non-empty.")
        entry.touch()
        with self._lock:
            existing = self._entries.get(entry.conversation_id)
            if existing is not None and existing is not entry:
                self._eviction_hook(existing)
            self._entries[entry.conversation_id] = entry
            self._entries.move_to_end(entry.conversation_id)
            self._evict_if_needed_locked()

    def discard(self, conversation_id: str) -> bool:
        """Remove a single entry; return True if it existed."""
        with self._lock:
            entry = self._entries.pop(conversation_id, None)
            if entry is None:
                return False
            self._eviction_hook(entry)
            return True

    def clear(self) -> None:
        """Free every cached entry."""
        with self._lock:
            for entry in self._entries.values():
                self._eviction_hook(entry)
            self._entries.clear()

    # ── Internal ────────────────────────────────────────────────────────

    def _evict_if_needed_locked(self) -> None:
        while len(self._entries) > self._max_sessions:
            _key, evicted = self._entries.popitem(last=False)
            self._eviction_hook(evicted)


# ── Conversation-id resolution helpers ─────────────────────────────────────


def _prefix_hash_key(prompt_token_ids: Iterable[int], window: int = 32) -> str:
    """Deterministic anonymous id derived from the first ``window`` tokens.

    Used when an OpenAI client does not set the ``user`` field. Stable across
    turns because the chat template's system + first user message are
    constant for the lifetime of a conversation.
    """
    import hashlib

    token_list = list(prompt_token_ids)[:window]
    blob = ",".join(str(int(tok)) for tok in token_list).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return f"prefix::{digest[:24]}"


def resolve_conversation_id(
    *,
    explicit_id: str | None,
    prompt_token_ids: Iterable[int] | None,
) -> str | None:
    """Resolve the cache key for a request.

    Prefers the explicit id (OpenAI ``user`` field). Falls back to the
    prefix-hash derived from the first tokens. Returns ``None`` when the
    caller supplied neither.
    """
    if explicit_id:
        stripped = str(explicit_id).strip()
        if stripped:
            return stripped
    if prompt_token_ids is None:
        return None
    return _prefix_hash_key(prompt_token_ids)


__all__ = [
    "DEFAULT_MAX_SESSIONS",
    "ResidualSessionCache",
    "ResidualSessionEntry",
    "resolve_conversation_id",
]
