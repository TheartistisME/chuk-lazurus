# session_close — axis-2 design note

## §1 Purpose

`chuk_lazarus.session_close` (axis-2, pipeline-lead) converts a finished
`ChatLoopSession` — or an equivalent `list[TurnRecord]` produced by axis-1 —
into a directory of per-turn-chunk JSON files that load cleanly through
`tools.build_clause_aligned_store.load_clause_records` so axis-3 can build a
torch checkpoint UNMODIFIED. Conformance is *implicit AUS3000*: the target
schema is not a standalone JSON Schema file but the
`ClauseRecord` dataclass shape defined at
`tools/build_clause_aligned_store.py` lines 88-97 and enforced by the loader
at lines 244-279. Axis-2 emits JSON objects whose five required fields
(`source_file` is synthesized from the filename; the other four —
`standard_id`, `standard_title`, `clause_id`, `clause_title`, plus
`clause_content`) round-trip through that loader without modification.

## §2 Handoff contract (consume from axis-1)

Axis-2 reads the shapes defined in
`src/chuk_lazarus/chat_loop/session.py` and consumed via
`src/chuk_lazarus/chat_loop/handoff.py`. Reproduced verbatim:

```python
# session.py:37-47
class ChunkBoundary(BaseModel):
    chunk_index: int
    start_token_offset: int
    end_token_offset: int
    emitted_at: str
```

```python
# session.py:50-66
class TurnRecord(BaseModel):
    turn_index: int
    role: Role
    text: str
    started_at: str = Field(default_factory=_utc_now_iso)
    finished_at: str | None = None
    chunk_boundaries: list[ChunkBoundary] = Field(default_factory=list)

    def mark_finished(self) -> None:
        self.finished_at = _utc_now_iso()
```

```python
# session.py:69-88
class ChatLoopSession(BaseModel):
    session_id: str = Field(default_factory=_new_session_id)
    started_at: str = Field(default_factory=_utc_now_iso)
    turns: list[TurnRecord] = Field(default_factory=list)
    ...
```

The `Role` enum is verified at
`src/chuk_lazarus/inference/chat.py` lines 24-30 with exact values:

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MODEL = "model"  # Used by some models (Gemma)
```

Axis-2 calls `TranscriptHandoff.snapshot()` (preferred; deep copy, safe
against races — see `handoff.py` line 29) to obtain `list[TurnRecord]`. For
tests or non-live callers, axis-2 also accepts an equivalent
`list[TurnRecord]` directly. As stated in
`docs/turn_aligned/chat_loop_design.md` §5.3 and §7, **axis-1 does NOT
persist per-chunk decoded text** — only token offsets on `ChunkBoundary`.
Axis-2 is therefore responsible for text reconstruction (see §4).

## §3 ClauseRecord mapping (per-turn-chunk)

One JSON file is emitted per `(turn, chunk)` pair with the following
mapping:

| ClauseRecord field | Source | Rationale |
|---|---|---|
| `standard_id` | `session_id` (UUID4 hex, 32 lowercase hex chars) | Each session is one "standard" in the axis-3 knowledge store. |
| `standard_title` | `session_title_from_turns(turns)` — first non-empty user-turn text, truncated to 120 chars; fallback `"chat-session <session_id[:8]>"` if no user turn is present | Human-readable label for the session; fallback guarantees non-empty (loader requires, line 256). |
| `clause_id` | `"<session_id>.<turn_index>.<chunk_index>"` (dotted 3-part; session_id is hex, turn_index and chunk_index are decimal integers) | Matches axis-1 §6 canonical retrieval key; unique per chunk across all sessions. |
| `clause_title` | `condense(turn.text, chunk_text, role, turn_index)` per §5 | Deterministic, non-empty, human-legible summary line. |
| `clause_content` | `chunk_text` per §4 | The substantive payload; non-empty after normalization. |

The loader at `tools/build_clause_aligned_store.py` lines 256-260 rejects
records with any empty value among `{standard_id, standard_title,
clause_id, clause_title}`; line 254 uses `clause_content or clause_title`
as a fallback, so `clause_content` is never required to be non-empty by
the loader — but axis-2 emits non-empty `clause_content` as a matter of
discipline.

## §4 Chunk reconstruction policy (POC trade-off)

Axis-1 retains `chunk_boundaries` (token offsets) but **not** per-chunk
decoded text (confirmed at `streaming.py` lines 100-126 — the decoded
string is only materialized inside an optional `on_window` callback and
never stored on `ChunkBoundary`). Axis-2 must reconstruct the content
string.

**Primary strategy (POC default): full-turn text per chunk.**

For every `ChunkBoundary` on a given turn,

```
clause_content := turn.text
```

Rationale:

1. **Deterministic** — zero tokenizer dependency; axis-2 does not need to
   load Gemma or import `transformers`.
2. **No loss of retrieval granularity** — axis-3 re-chunks at 512/64
   internally (defaults per `build_clause_aligned_store.py` lines 138-146),
   so axis-2-level sub-chunk boundaries do not govern final window
   content. The distinct `clause_id` per chunk preserves the addressing
   surface demanded by axis-1 §6.
3. **Matches axis-1's guidance** — `chat_loop_design.md` §5.3 explicitly
   notes that "decoded chunk text can be reconstructed by the caller if
   needed from the full `turn.text` and the token offsets; alternatively,
   axis-2's pipeline can re-tokenize and slice".
4. **Negligible duplication cost** — POC session sizes are measured in
   tens of turns; full-turn duplication across a handful of chunks per
   turn is acceptable.

**Alternative strategy (documented, NOT implemented by default):
token-accurate slicing via a pluggable tokenizer callback.**

The emitter accepts an optional `text_slicer` parameter of type:

```python
TextSlicer = Callable[[str, int, int], str]
# (turn_text, start_token_offset, end_token_offset) -> sliced_text
```

When `text_slicer is None` the primary strategy applies. When supplied,
the emitter invokes `text_slicer(turn.text, boundary.start_token_offset,
boundary.end_token_offset)` per chunk. This interface keeps axis-2
tokenizer-free by default while leaving a future seam for faithful
window content.

**Edge case — empty `chunk_boundaries`.**

User turns always have empty `chunk_boundaries` (verified at
`chat_loop_design.md` §6 and the sample in §3.2). Assistant turns with
empty or too-short text also produce no boundaries. In either case,
axis-2 emits **exactly one** record with `chunk_index = 0` and
`clause_content = turn.text` (using the synthesized fallback from §5 if
`turn.text` is empty/whitespace). Rationale: the mission invariant is
"every user and assistant turn is catalogued — no turn dropped, no turn
duplicated".

## §5 Condensation policy (deterministic, no-LLM)

`clause_title` is produced by a pure function of `(content, role,
turn_index)`:

1. Strip leading/trailing whitespace from `content`.
2. If the result is empty, return the synthesized fallback
   `"<role> turn <turn_index>"` (e.g. `"assistant turn 3"`). This
   guarantees non-empty output — the loader rejects empty `clause_title`
   at line 256.
3. Otherwise, find the **first sentence terminator** — the earliest
   occurrence of any of `"."`, `"!"`, `"?"`, or `"\n\n"`. Take the
   substring up to (but not including) that terminator. If no terminator
   is found, take the entire stripped content.
4. Collapse internal whitespace runs to single spaces.
5. If the result is longer than `max_chars` (default 160), truncate to
   `max_chars - 3` characters and append `"..."`.
6. Return the result.

This mirrors the AUS3000 convention used elsewhere in the project and
produces a short, legible, deterministic title with no LLM call.

## §6 Topic tags (deterministic, no-LLM)

`topic_tags` is produced by top-K frequency keyword extraction:

1. Lowercase `clause_content`.
2. Split on non-alphanumeric characters (regex `[^a-z0-9]+`).
3. Drop tokens shorter than 3 characters.
4. Drop tokens that are purely numeric.
5. Drop tokens appearing in the inline `STOPWORDS` set (approximately 30
   common English function words):

```python
STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "for", "with", "by", "it", "this",
    "that", "these", "those", "i", "you", "we", "they", "he", "she",
    "as", "be", "been",
})
```

6. Count remaining tokens.
7. Return the top-K (default K=5), ordered by `(count desc,
   first_occurrence_index asc)`. Ties break on first occurrence.

## §7 Secondary metadata (required on every emitted JSON file)

In addition to the five `ClauseRecord` fields, every emitted JSON object
MUST include the following metadata fields:

| Field | Source | Shape |
|---|---|---|
| `iso_timestamp` | `turn.finished_at` if non-null else `turn.started_at` | string; UTC ISO-8601 with trailing `Z` |
| `speaker_role` | `turn.role.value` | string: one of `"user"`, `"assistant"`, `"system"`, `"model"` (verified at `src/chuk_lazarus/inference/chat.py` lines 27-30) |
| `session_uuid` | `session_id` | string (mirror of `standard_id`; kept as explicit field for downstream clarity) |
| `topic_tags` | output of §6 | `list[str]`, length 0-5 |

**These secondary fields are IGNORED by the axis-3 loader** — `load_clause_record`
at `tools/build_clause_aligned_store.py` lines 244-270 reads only the five
ClauseRecord fields — but they are preserved on disk for axis-4 (retrieval)
and axis-5 (verification) consumers.

## §8 Output directory layout

For a session with `session_id = S` producing `K` records (one per chunk
per §4), the emitter writes:

```
<out_dir>/<session_id>/<NNN>_<turn_index>_<chunk_index>.json
```

- `<NNN>` is a zero-padded three-digit emission counter (`000`, `001`,
  ...) so that `natural_sort_key` at
  `tools/build_clause_aligned_store.py` lines 179-187 preserves the
  emission order.
- Filenames MUST NOT end with `_metadata.json` — the loader skips those
  at lines 276-277.
- The axis-3 handoff contract is: axis-3 is invoked with
  `--input-dir <out_dir>/<session_id>`, producing one checkpoint per
  session.
- File format: a single JSON object (NOT NDJSON), encoded as UTF-8
  without BOM. The loader reads with `encoding="utf-8-sig"` (line 245),
  which tolerates a BOM if present, but axis-2 emits plain UTF-8.
- Each emitted file is pretty-printed with `indent=2` for human
  inspection; whitespace is immaterial to the loader.

## §9 Conformance test strategy

Conformance is defined as **round-trip through the loader**, not schema
validation:

```python
from tools.build_clause_aligned_store import load_clause_records

# 1. emit a fake transcript into a tmp_path session dir
# 2. call the loader
records = load_clause_records(tmp_path / session_id)

# 3. assertions:
#    (a) records is non-empty
#    (b) each record.clause_id matches r"^[0-9a-f]{32}\.\d+\.\d+$"
#    (c) each record.clause_title is non-empty
#    (d) each record.clause_content is non-empty
```

The test does **not** execute `build_clause_aligned_store.py` as a
subprocess — that is axis-3's responsibility. Axis-2's conformance test
verifies only that the loader accepts the emitted JSON and returns the
expected dataclass-shaped records.

## §10 Module layout (implementation spec)

```
src/chuk_lazarus/session_close/
├── __init__.py          re-exports WindDownEmitter, emit_session,
│                        emit_from_handoff, dotted_handle helpers
├── dotted_handle.py     compose/parse for "<sid>.<ti>.<ci>" clause_id
├── condense.py          condense_title deterministic summarizer
├── topic_tags.py        extract_topic_tags + STOPWORDS
├── aus3000_emit.py      TurnRecord list -> per-chunk dicts; write_records
├── wind_down.py         end-to-end orchestration; public surface
└── cli.py               python -m chuk_lazarus.session_close.cli ...
```

| File | Responsibility |
|---|---|
| `__init__.py` | Public import surface. Re-exports `emit_session`, `emit_from_handoff`, `WindDownEmitter`, `compose_handle`, `parse_handle`. |
| `dotted_handle.py` | Pure functions for constructing and parsing the `session_id.turn_index.chunk_index` handle. |
| `condense.py` | Deterministic title condensation per §5. |
| `topic_tags.py` | Deterministic top-K keyword extraction per §6. |
| `aus3000_emit.py` | Given a `list[TurnRecord]`, produce a `list[dict]` in AUS3000-implicit shape; `write_records` serializes to disk with the §8 naming scheme. |
| `wind_down.py` | End-to-end orchestrator. `emit_session(turns, session_id, out_dir, ...)` and `emit_from_handoff(handoff, out_dir)` are the two public entry points. |
| `cli.py` | `python -m chuk_lazarus.session_close.cli --session-json <path> --out-dir <path>` — reads a JSON-dumped session (produced by `TranscriptHandoff.as_dict_list()` plus a `session_id`) and writes the output directory. Useful for decoupled testing without running the REPL. |

## §11 API contracts (authoritative function signatures)

The implementation sub-agent MUST match these signatures verbatim.

```python
# dotted_handle.py
def compose_handle(session_id: str, turn_index: int, chunk_index: int) -> str: ...

def parse_handle(handle: str) -> tuple[str, int, int]:
    """Parse a dotted handle. Raises ValueError if not 3 parts or ints invalid."""
```

```python
# condense.py
def condense_title(
    content: str,
    role: str,
    turn_index: int,
    *,
    max_chars: int = 160,
) -> str: ...
```

```python
# topic_tags.py
STOPWORDS: frozenset[str]

def extract_topic_tags(text: str, *, top_k: int = 5) -> list[str]: ...
```

```python
# aus3000_emit.py
from typing import Callable, Optional
from chuk_lazarus.chat_loop.session import TurnRecord
from pathlib import Path

TextSlicer = Callable[[str, int, int], str]

def build_records(
    turns: list[TurnRecord],
    session_id: str,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[dict]:
    """Produce one AUS3000-shaped dict per (turn, chunk). See §3–§7."""

def write_records(
    out_dir: Path,
    session_id: str,
    records: list[dict],
) -> list[Path]:
    """Write records to <out_dir>/<session_id>/<NNN>_<ti>_<ci>.json.
    Creates the session dir if needed. Returns the list of written paths
    in emission order."""
```

```python
# wind_down.py
from chuk_lazarus.chat_loop.handoff import TranscriptHandoff
from chuk_lazarus.chat_loop.session import TurnRecord
from pathlib import Path

class WindDownEmitter:
    def __init__(
        self,
        *,
        session_title: str | None = None,
        text_slicer: TextSlicer | None = None,
    ) -> None: ...

    def emit(
        self,
        turns: list[TurnRecord],
        session_id: str,
        out_dir: Path,
    ) -> list[Path]: ...

def emit_session(
    turns: list[TurnRecord],
    session_id: str,
    out_dir: Path,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[Path]: ...

def emit_from_handoff(
    handoff: TranscriptHandoff,
    out_dir: Path,
    *,
    session_title: str | None = None,
    text_slicer: TextSlicer | None = None,
) -> list[Path]:
    """Convenience wrapper: pulls session_id from handoff.raw_session()
    and turns from handoff.snapshot()."""
```

```python
# cli.py  (no function signatures — argparse only)
# python -m chuk_lazarus.session_close.cli \
#     --session-json <path to JSON dump: {"session_id": str, "turns": [...]}> \
#     --out-dir <path>
```

Helper — session title synthesis used internally by `build_records`:

```python
def session_title_from_turns(
    turns: list[TurnRecord],
    session_id: str,
    *,
    max_chars: int = 120,
) -> str:
    """First non-empty user-turn text trimmed to max_chars, else
    'chat-session <session_id[:8]>'. Guaranteed non-empty."""
```

## §12 Grammar-of-absence (out-of-scope)

Axis-2 explicitly does **NOT**:

- Invoke `tools/build_clause_aligned_store.py` (that is axis-3).
- Load Gemma, run inference, or import `transformers`. The POC slicer
  strategy is character-level (full-turn text per chunk); tokenizer-based
  slicing is available only via the opt-in `text_slicer` callback.
- Implement retrieval, ranking, or query handling (axis-4).
- Run the cross-session demo (axis-5).
- Modify `tools/build_clause_aligned_store.py` or any file under
  `src/chuk_lazarus/chat_loop/`.
- Emit NDJSON, CSV, Parquet, or torch tensors — only per-record JSON
  files.
- Register a `console_scripts` entrypoint in `pyproject.toml`. Axis-1
  §7 notes that any such entrypoint is a cross-lead scope change; axis-2
  respects the same constraint. Axis-2 is reachable only via
  `python -m chuk_lazarus.session_close.cli`.
- Perform schema validation against any standalone JSON Schema file.
  Conformance is defined exclusively as round-trip through
  `load_clause_records` (see §9).
- Persist any state across sessions. Each `emit_session` call is
  self-contained.
