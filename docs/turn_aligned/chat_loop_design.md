# chat_loop — axis-1 design note

## 1. Purpose

`chuk_lazarus.chat_loop` is a local, turn-aligned chat REPL against
Gemma-4-E2B-it that streams assistant output to stdout while simultaneously
chunking that stream into overlapping 512-token windows on the hot path
(during generation, not post-hoc). Each user and assistant turn is captured
in-memory as a `TurnRecord`, owned by a `ChatLoopSession` that assigns
monotonic turn indices and a stable `session_id`. Downstream consumers
(axis-2 pipeline-lead, axis-5 verification-lead) observe the transcript
through a single explicit handoff surface (`TranscriptHandoff`) rather than
reaching into session internals. Axis-1 is deliberately narrow: it produces
raw transcript records plus chunk boundaries, and nothing else.

## 2. Public surface (import contract)

Everything in this section is re-exported from `chuk_lazarus.chat_loop`.
Axis-2 and axis-5 should rely only on these names; `cli.py` helpers
(`load_gemma`, `run_repl`, `stream_assistant_reply`) are not part of the
stable surface.

### 2.1 Re-exports (`chuk_lazarus/chat_loop/__init__.py`)

```python
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
```

### 2.2 Session models (`session.py`)

```python
class ChunkBoundary(BaseModel):
    chunk_index: int
    start_token_offset: int
    end_token_offset: int
    emitted_at: str
```

```python
class TurnRecord(BaseModel):
    turn_index: int
    role: Role
    text: str
    started_at: str = Field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    chunk_boundaries: list[ChunkBoundary] = Field(default_factory=list)

    def mark_finished(self) -> None: ...
```

```python
class ChatLoopSession(BaseModel):
    session_id: str = Field(default_factory=_new_session_id)
    started_at: str = Field(default_factory=_utc_now_iso)
    turns: list[TurnRecord] = Field(default_factory=list)

    def begin_turn(self, role: Role, text: str) -> TurnRecord: ...
    def append_chunk(self, turn: TurnRecord, boundary: ChunkBoundary) -> None: ...
    def finish_turn(self, turn: TurnRecord) -> None: ...
```

`Role` is imported from `chuk_lazarus.inference.chat`:

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"  # used by some models (Gemma)
```

Turn records in this REPL use `Role.USER` and `Role.ASSISTANT`; `Role.SYSTEM`
and `Role.MODEL` are not produced by `run_repl` but the enum accepts them.

### 2.3 Streaming windower (`streaming.py`)

```python
OnWindow = Callable[[ChunkBoundary, str], None]

class StreamingWindower:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        window_tokens: int = 512,
        overlap_tokens: int = 64,
        on_window: OnWindow | None = None,
    ) -> None: ...

    @property
    def total_tokens(self) -> int: ...
    @property
    def emitted_chunks(self) -> int: ...

    def feed_text(self, text_delta: str) -> list[ChunkBoundary]: ...
    def flush(self) -> ChunkBoundary | None: ...
    def reset(self) -> None: ...
```

`StreamingWindower` is used internally by `cli.py` on the hot path. Axis-2
should not need to instantiate it directly; it consumes the already-emitted
`ChunkBoundary` objects attached to each `TurnRecord`.

### 2.4 Handoff (`handoff.py`)

```python
class TranscriptHandoff:
    def __init__(self, session: ChatLoopSession) -> None: ...
    def snapshot(self) -> list[TurnRecord]: ...          # deep copy, safe
    def raw_session(self) -> ChatLoopSession: ...         # by reference, opt-in
    def as_dict_list(self) -> list[dict[str, Any]]: ...   # JSON-ready dicts
```

## 3. Data shapes

All timestamps are UTC ISO-8601 with microsecond precision and a trailing
`Z`, produced by the internal helper `_utc_now_iso()`.

### 3.1 TurnRecord with two chunk boundaries

```json
{
  "turn_index": 3,
  "role": "assistant",
  "text": "Sure. The three canonical invariants are ... (full assistant reply text)",
  "started_at": "2026-04-21T14:32:07.512034Z",
  "finished_at": "2026-04-21T14:32:11.908221Z",
  "chunk_boundaries": [
    {
      "chunk_index": 0,
      "start_token_offset": 0,
      "end_token_offset": 512,
      "emitted_at": "2026-04-21T14:32:10.214770Z"
    },
    {
      "chunk_index": 1,
      "start_token_offset": 448,
      "end_token_offset": 640,
      "emitted_at": "2026-04-21T14:32:11.908120Z"
    }
  ]
}
```

The second boundary is a partial tail emitted by `flush()`; note that
`start_token_offset` (`448`) equals
`chunk_0.end_token_offset (512) - overlap_tokens (64)`.

### 3.2 ChatLoopSession with two user + two assistant turns

```json
{
  "session_id": "4f9b6d0e2a7c4f1e9b3a6d0c1e7f2a84",
  "started_at": "2026-04-21T14:31:55.000012Z",
  "turns": [
    {
      "turn_index": 0,
      "role": "user",
      "text": "Summarise the clause-aligned store invariants.",
      "started_at": "2026-04-21T14:32:01.110044Z",
      "finished_at": "2026-04-21T14:32:01.110210Z",
      "chunk_boundaries": []
    },
    {
      "turn_index": 1,
      "role": "assistant",
      "text": "The store preserves clause boundaries ...",
      "started_at": "2026-04-21T14:32:01.110899Z",
      "finished_at": "2026-04-21T14:32:04.772005Z",
      "chunk_boundaries": [
        {
          "chunk_index": 0,
          "start_token_offset": 0,
          "end_token_offset": 512,
          "emitted_at": "2026-04-21T14:32:04.108340Z"
        }
      ]
    },
    {
      "turn_index": 2,
      "role": "user",
      "text": "And what's the default overlap?",
      "started_at": "2026-04-21T14:32:20.300001Z",
      "finished_at": "2026-04-21T14:32:20.300120Z",
      "chunk_boundaries": []
    },
    {
      "turn_index": 3,
      "role": "assistant",
      "text": "64 tokens, matching the axis-3 builder ...",
      "started_at": "2026-04-21T14:32:20.301004Z",
      "finished_at": "2026-04-21T14:32:22.994711Z",
      "chunk_boundaries": [
        {
          "chunk_index": 0,
          "start_token_offset": 0,
          "end_token_offset": 512,
          "emitted_at": "2026-04-21T14:32:22.400117Z"
        },
        {
          "chunk_index": 1,
          "start_token_offset": 448,
          "end_token_offset": 601,
          "emitted_at": "2026-04-21T14:32:22.994655Z"
        }
      ]
    }
  ]
}
```

## 4. Hot-path windowing invariants

- **Defaults.** `window_tokens=512`, `overlap_tokens=64`. This matches the
  axis-3 `build_clause_aligned_store.py --overlap-tokens 64` default so the
  chunk boundaries emitted here are directly compatible with the
  clause-aligned store pipeline downstream.
- **Validation.** The constructor raises `ValueError` if
  `window_tokens <= 0`, if `overlap_tokens < 0`, or if
  `overlap_tokens >= window_tokens`.
- **During-stream emission.** `feed_text(delta)` is called once per
  streamer delta. It tokenizes the delta with `add_special_tokens=False`,
  extends the internal token buffer, and emits zero or more
  `ChunkBoundary` objects (one per complete window crossed by this delta).
  A single `feed_text` call can therefore return an empty list, one
  boundary, or several.
- **Sliding stride.** After each full-window emission, the window start
  advances by `window_tokens - overlap_tokens` (default stride 448). So
  window `n` spans `[n*448, n*448 + 512)` in absolute turn-local token
  offsets, and consecutive windows share `overlap_tokens` tokens.
- **Tail via `flush()`.** On turn end, `flush()` emits at most one final
  partial boundary for any tail tokens past `next_window_start_offset`.
  If there is no tail, it returns `None`.
- **Idempotent `flush()`.** After a non-`None` flush, the internal
  `next_window_start_offset` is advanced to the end of the buffer, so a
  second call returns `None`. Callers can flush defensively without
  double-emitting.
- **Timestamps.** `emitted_at` on every `ChunkBoundary` is the UTC wall
  clock at the moment the boundary was constructed, formatted ISO-8601
  with microsecond precision and a trailing `Z`.
- **Offsets are turn-local.** `start_token_offset` and `end_token_offset`
  are absolute token positions within the current assistant turn, counted
  from offset 0 at the start of the reply. They are NOT session-global.
- **Per-turn instances.** The REPL constructs a fresh `StreamingWindower`
  for every assistant turn. `reset()` exists for callers who prefer to
  reuse an instance across turns, but `run_repl` does not use it.
- **`on_window` callback.** Optional; if supplied, invoked for every
  emitted boundary (full and tail) with the decoded window text
  (`skip_special_tokens=True`). It is decoded lazily only when a callback
  is present.

## 5. Handoff contract for axis-2 (pipeline-lead)

Axis-2's responsibility is to emit AUS3000-schema JSON from the raw
transcript produced by axis-1. The contract is:

1. At session wind-down (after `run_repl` returns, or after a known
   quiescent point), construct the handoff:
   ```python
   handoff = TranscriptHandoff(session)
   turns = handoff.snapshot()
   ```
   `snapshot()` returns a deep copy — downstream mutations cannot race or
   corrupt the live session. Prefer this path.
2. For each `TurnRecord` with non-empty `chunk_boundaries`, axis-2 emits
   one AUS3000 record per `ChunkBoundary`, using the dotted handle
   `<session_id>.<turn_index>.<chunk_index>` as the clause_id.
3. The decoded chunk text can be reconstructed by the caller if needed
   from the full `turn.text` and the token offsets; alternatively,
   axis-2's pipeline can re-tokenize and slice. Axis-1 does not persist
   per-chunk text.
4. For JSON-serializable payloads, use `handoff.as_dict_list()` which
   calls `model_dump(mode="json")` on each turn.
5. `handoff.raw_session()` is available for callers that explicitly need
   the live object (e.g. to observe turns as they are appended during a
   long session). This is opt-in; the default remains `snapshot()`.

Axis-2 specifically does **not**:

- Instantiate `StreamingWindower` — boundaries are already on the turn.
- Call `build_clause_aligned_store.py` — that is axis-3's responsibility.

Axis-1 specifically does **not**:

- Emit AUS3000 JSON — that is axis-2's responsibility.
- Persist anything to disk — the session is purely in-memory.

## 6. Handoff contract for axis-5 (verification-lead)

Axis-5 runs the cross-session demo and verifies retrieval integrity. The
invariants it can rely on:

- **`session_id`** is a UUID4 hex string (32 lowercase hex chars, no
  dashes), produced once per `ChatLoopSession` via `uuid4().hex`. It is
  stable for the life of the session and unique across sessions with
  overwhelming probability.
- **Canonical retrieval key.** The dotted handle
  `<session_id>.<turn_index>.<chunk_index>` uniquely identifies a single
  chunk across all sessions. Axis-5 should use this as the primary key in
  cross-session verification.
- **`turn_index`** is 0-based and monotonically increasing within a
  session; it is assigned by `ChatLoopSession.begin_turn` as
  `len(self.turns)` at insertion time. User and assistant turns share the
  same index space, so turn `2n` is typically user and `2n+1` assistant
  when a REPL runs without a system prompt turn.
- **`chunk_index`** is 0-based and monotonically increasing within a
  single turn; it resets to 0 at the start of every assistant turn (every
  turn uses a fresh `StreamingWindower`). User turns produce zero chunks.
- **Timestamps.** `started_at` / `finished_at` on a `TurnRecord` and
  `emitted_at` on a `ChunkBoundary` are all UTC ISO-8601 with microsecond
  precision and a trailing `Z`. They are monotonic within a session
  subject to wall-clock resolution.

## 7. Out-of-scope / absent (grammar of absence)

Explicitly not produced by axis-1:

- **No AUS3000 JSON emission.** Axis-1 stops at `ChunkBoundary` /
  `TurnRecord`. Schema serialization lives in axis-2.
- **No torch checkpoint handling.** `load_gemma` in `cli.py` loads weights
  for inference only; there is no save / checkpoint / resume path.
- **No retrieval wiring.** The chat loop does not perform retrieval, RAG,
  or any query-time lookup. Retrieval lives in axis-3/axis-4.
- **No cross-session orchestration.** A session is self-contained; there
  is no session registry, no persistence, no cross-session state.
- **No `pyproject` / CLI registration.** No `console_scripts` entry is
  declared for the CLI by axis-1. The REPL is reachable only via
  `python -m chuk_lazarus.chat_loop.cli`. If axis-2 requires a registered
  entrypoint, that is a cross-lead scope change and must be requested via
  `vee record pattern ... --tag cross-lead-touch-requested`.

## 8. Running the REPL

```bash
python -m chuk_lazarus.chat_loop.cli --max-new-tokens 256
```

Optional flags:

- `--system TEXT` — seed a system prompt on the `ChatHistory`.
- `--model-path PATH` — override the Gemma snapshot location (default:
  the local snapshot at
  `/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`,
  falling back to the hub id `google/gemma-4-E2B-it` if the directory is
  absent).
- `--max-new-tokens INT` — per-turn generation cap (default 512).

Exit the loop with `/quit`, an empty line, or EOF (`Ctrl-D`). On exit,
the CLI prints a handoff summary line of the form
`Handoff ready: session_id=<hex> turns=<N> chunks=<M>`.

This works when `chuk_lazarus` is importable (installed or on
`PYTHONPATH`). A `console_script` entrypoint is NOT registered by
axis-1.

## 9. Known limitations / future work hooks

All deliberately deferred for the POC scope of axis-1:

- **No multi-turn context truncation.** `ChatHistory` grows unbounded;
  once it exceeds the model's context window, inference will fail. A
  future sliding-history policy is out of scope here.
- **No mid-stream interrupt handling.** `Ctrl-C` during generation is not
  handled gracefully; the generation thread is a daemon and the main
  thread re-raises whatever the generate call raised.
- **No persistence.** Sessions live in memory only. Axis-2 / axis-5 are
  expected to serialize via `TranscriptHandoff.as_dict_list()` if they
  need durable records.
- **No streaming backpressure.** `feed_text` is synchronous with the
  generator; a slow tokenizer or callback will stall the stream.
- **No fallback for missing tokenizer `chat_template`.** `format_history`
  (imported from `chuk_lazarus.inference.chat`) has a `_format_simple`
  fallback, but Gemma-4-E2B-it ships a template so this path is not
  exercised in the REPL.
- **No explicit `Role.MODEL` handling in the REPL.** The `Role` enum
  includes `MODEL` for Gemma-family compatibility, but `run_repl` writes
  assistant turns as `Role.ASSISTANT` regardless.
