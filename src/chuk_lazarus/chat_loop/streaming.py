"""Hot-path 512-token windowing of a streaming assistant reply.

StreamingWindower buffers token ids as the model streams decoded text
deltas; each time the buffer grows past the next window-start boundary
by `window_tokens`, a ChunkBoundary is emitted and the start cursor
advances by `window_tokens - overlap_tokens` (sliding window with overlap).

The caller drives the loop (one `feed_text` call per streamer delta);
this class does not own I/O or threads. A final partial tail may be
emitted via `flush()` when the stream ends.

Defaults (window=512, overlap=64) match the axis-3 store builder for
compatibility with downstream clause-aligned pipelines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from chuk_lazarus.chat_loop.session import ChunkBoundary

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


OnWindow = Callable[[ChunkBoundary, str], None]


def _utc_now_iso() -> str:
    """Return the current UTC time as ISO-8601 with trailing 'Z'."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class StreamingWindower:
    """Slice a streaming assistant reply into overlapping token windows.

    Window size and overlap default to 512 / 64, matching the axis-3
    clause-aligned store builder.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        window_tokens: int = 512,
        overlap_tokens: int = 64,
        on_window: OnWindow | None = None,
    ) -> None:
        if window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= window_tokens:
            raise ValueError("overlap_tokens must be strictly less than window_tokens")

        self._tokenizer = tokenizer
        self._window_tokens = window_tokens
        self._overlap_tokens = overlap_tokens
        self._on_window = on_window

        self._token_ids: list[int] = []
        self._next_window_start_offset: int = 0
        self._chunk_index: int = 0
        self._last_full_window_end: int = 0

    # ── Public surface ──────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        """Number of assistant tokens fed so far."""
        return len(self._token_ids)

    @property
    def emitted_chunks(self) -> int:
        """Number of full-window chunks emitted so far (excludes tail)."""
        return self._chunk_index

    def feed_text(self, text_delta: str) -> list[ChunkBoundary]:
        """Feed a streamed text delta; return any boundaries emitted this call.

        Tokenizes the delta without special tokens, appends to the internal
        buffer, and emits as many full windows as the new buffer length
        supports. Called once per streamer delta on the hot path.
        """
        if not text_delta:
            return []

        new_ids = self._tokenizer(text_delta, add_special_tokens=False).input_ids
        if not new_ids:
            return []

        self._token_ids.extend(new_ids)

        emitted: list[ChunkBoundary] = []
        while (
            len(self._token_ids) - self._next_window_start_offset
            >= self._window_tokens
        ):
            start = self._next_window_start_offset
            end = start + self._window_tokens
            boundary = ChunkBoundary(
                chunk_index=self._chunk_index,
                start_token_offset=start,
                end_token_offset=end,
                emitted_at=_utc_now_iso(),
            )
            if self._on_window is not None:
                decoded = self._tokenizer.decode(
                    self._token_ids[start:end],
                    skip_special_tokens=True,
                )
                self._on_window(boundary, decoded)

            emitted.append(boundary)
            self._last_full_window_end = end
            self._next_window_start_offset += (
                self._window_tokens - self._overlap_tokens
            )
            self._chunk_index += 1

        return emitted

    def flush(self) -> ChunkBoundary | None:
        """Emit a final partial boundary for the tail, if any."""
        end = len(self._token_ids)
        start = self._next_window_start_offset
        if end <= start:
            return None
        # Skip phantom tail: buffer is exactly aligned with the last full
        # window, so the would-be tail [start, end) is entirely the overlap
        # region already covered by the previous emission.
        if end == self._last_full_window_end:
            return None

        boundary = ChunkBoundary(
            chunk_index=self._chunk_index,
            start_token_offset=start,
            end_token_offset=end,
            emitted_at=_utc_now_iso(),
        )
        if self._on_window is not None:
            decoded = self._tokenizer.decode(
                self._token_ids[start:end],
                skip_special_tokens=True,
            )
            self._on_window(boundary, decoded)

        self._chunk_index += 1
        self._next_window_start_offset = end
        return boundary

    def reset(self) -> None:
        """Clear all state so the instance can be reused for the next turn."""
        self._token_ids = []
        self._next_window_start_offset = 0
        self._chunk_index = 0
        self._last_full_window_end = 0
