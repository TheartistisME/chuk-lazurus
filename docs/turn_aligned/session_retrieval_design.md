# Axis-4 session_retrieval Design (Turn-Aligned Infinite-Memory POC)

Status: DESIGN (API CONTRACT — tests are written against this document)
Scope: `src/chuk_lazarus/session_retrieval/**`
Zero-modification invariant: the primitive at
`src/chuk_lazarus/inference/context/knowledge/{inject.py,torch_query.py,torch_store.py}`
is imported ONLY through its already-public surface. Axis-4 NEVER edits these files.

---

## 1. Purpose

Axis-4 is the **retrieval lead**: given a root directory of per-session
checkpoints produced by axis-3 (`<checkpoint_root>/<session_id>/torch_store/…`),
it answers a question by:

1. enumerating every valid clause-aligned checkpoint under the root,
2. routing the question across all checkpoints via one of three deterministic
   paths (exact, topical, entity-mention),
3. loading the winning store + boundary tensor,
4. running the Apollo-11-pattern residual-injection pipeline (as reproduced
   verbatim from `examples/inference/demo_clause_aligned_strict.py`), and
5. returning a `QueryResult` enriched with six strict-mode assertions so that
   an axis-5 caller can reject silent fallbacks.

The primitive is not re-implemented — we delegate to
`TorchKnowledgeStore`, `torch_query._residual_is_compatible`, and
`TorchInferenceRuntime.generate_with_residual`.

---

## 2. Public API (the contract)

All symbols are exported from `chuk_lazarus.session_retrieval`.

### 2.1 `CheckpointHandle` (dataclass)

```python
@dataclass
class CheckpointHandle:
    session_id: str
    checkpoint_dir: Path
    torch_store_dir: Path
    manifest: dict
    original_input_dir: Path | None
```

`manifest` is the parsed `torch_store/manifest.json`. Validity gate at
enumeration time (mirrors axis-3 `verify_checkpoint`):
`clause_aligned is True`, `num_windows >= 1`, `num_entries >= 1`.

### 2.2 `QueryResult` (dataclass)

```python
@dataclass
class QueryResult:
    routing_mode: str              # "exact" | "topical" | "entity_mention"
    source_session: str            # session_id of winning checkpoint
    window_id: int                 # store window id that was routed
    matched_window_text: str       # store.get_window_text(window_id, tokenizer)
    window_keywords: list[str]     # store.keywords.get(window_id, [])
    generated_answer: str          # runtime.generate_with_residual(...).text
    strict_assertions: dict[str, bool]
    routing_score: float | None    # score for topical / entity_mention; None for exact
```

`strict_assertions` has exactly these keys (mirror the 6 demo assertions):

| Key | Meaning | Gating |
|-----|---------|--------|
| `cuda_available` | `torch.cuda.is_available()` | every call |
| `model_on_cuda` | first model param device is `cuda` | every call |
| `residual_compatible` | `_residual_is_compatible(...)` returned True | every call (RAISE on False) |
| `hook_fired` | instrumented forward-pre-hook on `layers[crystal_layer]` fired | every call |
| `gpu_memory_grew` | `torch.cuda.max_memory_allocated() > mem_after_load_at_init` | CUDA-only (True on non-CUDA) |
| `store_window_nonempty` | `len(get_window_text(...)) > 0` | every call (RAISE on False) |

Note on `gpu_memory_grew`: when CUDA is not available the memory-growth check
is semantically irrelevant. We set it to `True` so that consumers that assert
"all strict assertions are True" still pass when the test runner is CPU-only
(e.g. unit tests that exercise enumeration/handle/topical-score/entity-extract
without instantiating a model). The three hard failures (`residual_compatible`,
`store_window_nonempty`, `hook_fired`) still raise — semantics of the demo are
preserved on CUDA.

### 2.3 `SessionRetriever` (class)

```python
class SessionRetriever:
    handles: list[CheckpointHandle]
    runtime: TorchInferenceRuntime
    tokenizer: Any
    crystal_layer: int
    system_prompt: str
    device: str
    _mem_after_load_at_init: int

    @classmethod
    def from_checkpoint_root(
        cls,
        checkpoint_root: Path,
        *,
        model_id: str = "google/gemma-4-E2B-it",
        device: str = "cuda",
        original_input_root: Path | None = None,
        system_prompt: str | None = None,
    ) -> "SessionRetriever": ...

    def query_exact_id(self, dotted_handle: str) -> QueryResult: ...
    def query_topical(self, query_text: str) -> QueryResult: ...
    def query_entity_mention(self, query_text: str) -> QueryResult: ...
```

Internals:
- `_generate_from_window(handle, window_id, question_text, routing_mode, routing_score=None) -> QueryResult`
  encapsulates the 13-step residual-injection pipeline transcribed from the
  demo (lines 97–298). Every query method funnels through it.

### 2.4 Axis-5 consumer contract

Axis-5 is expected to construct the retriever exactly once per chat session:

```python
from chuk_lazarus.session_retrieval import SessionRetriever

retriever = SessionRetriever.from_checkpoint_root(
    Path("./_sessions/checkpoints"),
    original_input_root=Path("./_sessions/records"),
    device="cuda",
)
result = retriever.query_topical("What did we decide about the boundary tensor?")
print(result.generated_answer)
assert all(result.strict_assertions.values())
```

`from_checkpoint_root` does CUDA init (model load, bfloat16); it MUST NOT be
called at import time. Unit tests that do not need generation use the
sub-modules directly (`enumeration`, `handle`, `topical.route_topical`,
`entity_mention.extract_entity_tokens`, `entity_mention.score_window_entity_overlap`).

---

## 3. Routing paths

All three paths produce a `(CheckpointHandle, window_id [, score])` triple that
is handed to `_generate_from_window`.

### 3.1 Exact (`query_exact_id`)

Input: a dotted handle `"<session_id>.<turn_index>.<chunk_index>"`.
Algorithm (`exact_id.route_exact_id`):

1. `parse_handle(handle)` → `(session_id, turn_index, chunk_index)`.
2. Find the `CheckpointHandle` whose `session_id` matches. If none, return `None`.
3. Load that one store and obtain its `ClauseMetadataRouter` via
   `store._get_clause_router()`. If the store has no window_metadata the
   router is `None`, return `None`.
4. Look up the verbatim `dotted_handle` string in
   `router.clause_id_to_primary_windows` (a `dict[str, list[int]]` keyed by
   the exact clause_id the primitive built at load time). Take the first
   primary window id. If absent / empty, return `None`.
5. On miss `SessionRetriever.query_exact_id` raises
   `ValueError(f"STRICT: no session/window matches handle {dotted_handle!r}")`.

We do **not** re-implement clause-ID parsing — the router dict owns the
clause_id → primary_window mapping.

#### Exact-ID routing strategy (why we bypass `_collect_exact_matches`)

The primitive exposes `TorchKnowledgeStore._collect_exact_matches`, which
internally delegates to `ClauseMetadataRouter.collect_clause_id_matches`.
That path ultimately runs a regex over the query string:

```python
_CLAUSE_ID_RE = re.compile(r"\b\d+(?:\.\d+)+\b")  # route.py:22
```

The pattern requires **all-digit** dotted IDs (e.g. `"1.0.0"`, `"12.3.4"`).
Axis-2, however, emits clause_ids whose first dotted component is the
32-char lowercase **hex** `session_id` — e.g.
`"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.0.0"`. The leading `\b\d` character
class rejects the hex prefix, so `_collect_exact_matches` returns an empty
match list for every production handle.

Rather than modify the primitive (a hard zero-modification invariant for
axis-4), `route_exact_id` sidesteps the regex entirely: it looks the
handle up directly in `ClauseMetadataRouter.clause_id_to_primary_windows`,
the dict the router populates at construction from
`window_metadata.json`. The dict is keyed by the verbatim clause_id
string, so any handle shape resolves — numeric, hex-prefixed, or
otherwise — provided axis-3 wrote it into window_metadata.

Invariant preserved: `git diff HEAD --
src/chuk_lazarus/inference/context/knowledge/` remains empty. The router
object is read-accessed through its already-public `_get_clause_router()`
accessor and its already-public `clause_id_to_primary_windows` attribute.

### 3.2 Topical (`query_topical`)

Algorithm (`topical.route_topical`):

1. For every `CheckpointHandle`, load the store and call
   `store.route_top_k(query_text, tokenizer, k=top_k_per_checkpoint)`.
2. For each returned `window_id`, re-score via
   `store._get_tfidf_router().score_window(query_ids, window_id)`
   so we have a comparable float across checkpoints (the same function the
   primitive uses internally — not re-implemented).
3. Globally rank by `(score, session_id, window_id)` with `-score` first for
   descending. Return top-1 `(handle, window_id, score)`. `None` if empty.

### 3.3 Entity-mention (`query_entity_mention`)

Algorithm (`entity_mention.route_entity_mention`):

1. `extract_entity_tokens(query_text)` deterministically produces a list:
   capitalised/Title-Case words (length ≥2), ALL-CAPS words (length ≥2),
   quoted substrings. Stopwords removed. Normalised to lowercase, deduped
   ordered.
2. For each `(handle, window_id)` in the cross-product of valid checkpoints,
   compute `score_window_entity_overlap(handle, window_id, entity_set)`:
   - window entity set = `store.keywords.get(window_id, [])` lowercased ∪
     `topic_tags` from the matching per-turn record (resolved via
     `load_original_turn_records(handle)` if `original_input_dir` is set and
     any per-turn JSON has a `clause_id` whose `(turn_index, chunk_index)`
     matches this window's metadata).
   - score = `|entities ∩ window_set| / max(1, len(entity_tokens))`.
3. Return the pair with maximum score `> 0`. Tie-break `(session_id, window_id)`
   lexicographic for determinism. `None` if nothing overlaps.

---

## 4. Strict-mode assertion surface (the 6 assertions)

These are asserted in `_generate_from_window` in the order the demo asserts
them (see `demo_clause_aligned_strict.py` lines 73–298).

| # | Assertion | Behaviour on fail |
|---|-----------|-------------------|
| 1 | `torch.cuda.is_available()` | recorded in `strict_assertions["cuda_available"]`; does NOT raise by itself (the device flag controls whether CUDA is strictly required). If `self.device.startswith("cuda")` and not available, we never get past `from_checkpoint_root`. |
| 2 | `next(runtime._model.parameters()).device.type == "cuda"` | recorded in `strict_assertions["model_on_cuda"]`; on CUDA device this is a precondition for assertion 5. Set at `from_checkpoint_root` time and again on each query. |
| 3 | `torch_query._residual_is_compatible(...)` returns True | HARD RAISE `RuntimeError` naming the reason (mirrors demo line 173–176). |
| 4 | Spy forward-pre-hook on `layers[crystal_layer]` fired ≥ 1 time | HARD RAISE `RuntimeError` (mirrors demo line 274–279). |
| 5 | `torch.cuda.max_memory_allocated() > mem_after_load_at_init` | HARD RAISE `RuntimeError` when on CUDA (mirrors demo line 267–272). On non-CUDA: skipped, flag set True. |
| 6 | Routed window text is non-empty | HARD RAISE `RuntimeError(f"STRICT: store.get_window_text({window_id}) returned empty")` (mirrors demo line 150–151). |

**Auto-gated by CUDA availability**: #1, #2, #5.
**Run every time**: #3, #4, #6.

---

## 5. Zero-modification invariant

The three primitive paths are imported read-only:

```python
from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
from chuk_lazarus.inference.context.knowledge import torch_query  # only _residual_is_compatible
from chuk_lazarus.inference.backends import LazarusBackend, ResidualState, TorchInferenceRuntime
from chuk_lazarus.inference.generation import GenerationConfig
```

Axis-4 does NOT re-implement: TF-IDF routing, clause-ID parsing, keyword
routing, residual compatibility, or Apollo-11 injection. `git diff HEAD --
src/chuk_lazarus/inference/context/knowledge/` MUST be empty after axis-4 work.

---

## 6. Module layout

```
src/chuk_lazarus/session_retrieval/
├── __init__.py          # exports SessionRetriever, QueryResult, CheckpointHandle
├── enumeration.py       # iter_checkpoint_handles, load_store, load_original_turn_records
├── handle.py            # parse_handle, compose_handle (re-export), handle_belongs_to_session
├── exact_id.py          # route_exact_id
├── topical.py           # route_topical
├── entity_mention.py    # extract_entity_tokens, score_window_entity_overlap, route_entity_mention
├── retriever.py         # SessionRetriever, QueryResult, _generate_from_window
└── cli.py               # python -m chuk_lazarus.session_retrieval.cli
```

---

## 7. Out-of-scope

- Tests under `tests/session_retrieval/**` — produced by a separate agent.
- Any edit to primitive files, chat_loop, session_close, session_store.
- Multi-window fusion (we route to exactly one window per query; scoring
  ranks across all checkpoints but selects top-1).
- Query expansion via `_expand_query` (would require a CUDA generate call
  during routing — routing must stay CPU-cheap).
