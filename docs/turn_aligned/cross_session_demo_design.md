# Axis-5: `cross_session_demo` design

Terminal axis of the `turn-aligned` run. Orchestrates axes 1-4 end to end
for >=5 synthetic sessions and verifies, from a fresh retriever process,
that all three routing paths recall verbatim planted phrases while the
six Apollo-11 strict-mode assertions pass on every query.

## 1. Acceptance criteria mapping

| ID | Criterion | Module / function |
|----|-----------|-------------------|
| a | Produce >=5 distinct sessions via axis-1 | `session_generator.generate_session` + `DEFAULT_PLANS` |
| b | Emit per-turn AUS3000 records via axis-2 | `pipeline.run_session` (calls `emit_from_handoff`) |
| c | Build per-session stores via axis-3 | `pipeline.run_session` (calls `invoke_build`) |
| d | Total corpus exceeds 128k Gemma tokens | `token_budget.measure_total_tokens` (subprocess `tools/count_tokens.py`) |
| e | Construct a fresh SessionRetriever after all builds | `cli.main` / `verification.run_cross_session_queries` |
| f | Exact-ID query routes to its planted window | `retriever.query_exact_id(planted_handle)` |
| g | Topical query routes to its owning session | `retriever.query_topical(topic_question)` |
| h | Entity-mention query routes to its owning session | `retriever.query_entity_mention(entity_question)` |
| i | Six strict-mode assertions True for every query (18 total) | `verification.all_assertions_pass` |
| j | Planted verbatim phrase appears in each generated answer | `verification.QueryExecution.verbatim_hit` |
| k | Zero-modification on upstream axes | `report.build_report` records a `zero_mod_upstream` diff map |

## 2. Content policy: SYNTHETIC, DETERMINISTIC

All conversational content is synthesised in-process by
`session_generator.generate_session`. This is deliberate:

- Reproducibility: runs are deterministic per `SessionPlan.seed`.
- CI-friendliness: no external fetches, no human-typed transcripts.
- Recall verifiability: each session owns a unique multi-word phrase
  with no cross-session overlap, so a verbatim substring check is a
  precise routing-correctness assertion.

Each session topic carries a distinctive entity (proper noun or
near-proper-noun) that will repeat many times across the session so it
ends up in `topic_tags` and store keywords, making entity-mention
routing deterministic.

## 3. Session catalogue (five plans)

| # | topic | entity (repeats >=10x) | planted phrase |
|---|-------|------------------------|----------------|
| 1 | dubai-trip | Fujairah | the coral reef shimmered cobalt at dawn over Fujairah |
| 2 | alice-project | Alice | Alice committed the mauve refactor on the 14th |
| 3 | sydney-conference | Gosford | the keynote mentioned quokka benchmarks in Gosford |
| 4 | rust-migration | coroutine | the traitful coroutine wrapper panicked at dawn |
| 5 | pottery-class | cone six | the kiln glowed at cone six during the thunderstorm |

### Sizing for >128k token budget

The Gemma tokenizer is used via subprocess (`tools/count_tokens.py`).
Target corpus size:

- ~40 turn pairs per session = ~80 turns
- assistant turn >= 1500 characters ≈ 350-400 Gemma tokens
- per-session tokens ≈ 40 * 400 = ~16000 tokens of assistant content
- plus user turns (~60 tokens each) = ~2400 user tokens
- total per-session ≈ ~18-30k tokens

Five sessions × ~30k ≈ 150k tokens — comfortable headroom over the
128k threshold. The generator deliberately uses a large deterministic
filler vocabulary (topic-specific sentences about scenes, people,
sub-topics, dates) so each assistant turn is substantial prose, not
a boilerplate repeat.

### Phrase planting

`generate_session` chooses one fixed assistant-turn index (default
`turn_index=11`, which is the 6th assistant turn) and appends the
planted phrase verbatim to that turn's text. The last assistant turn
also appends the phrase as a redundancy layer, so that the Apollo-11
injection path has multiple in-context occurrences to reproduce when
prompted to "quote relevant verbatim phrases" (see
`DEFAULT_SYSTEM_PROMPT` in `session_retrieval.retriever`).

`plant_location(session, plan)` resolves the dotted handle
`"<session_id>.<turn_index>.0"` pointing at the first turn containing
the phrase. chunk_index is always 0 because the streaming windower is
not engaged in synthetic sessions — `chunk_boundaries` is empty, so
axis-2 emits exactly one record per turn with `chunk_index=0`
(axis-2 §4 edge case). The axis-3 build script re-windows
`clause_content` at 512-token windows internally.

## 4. Three query paths

1. **Exact-ID**: `retriever.query_exact_id(sessions[0].planted_handle)` —
   handle is the literal dotted handle for session-1's planted turn.
2. **Topical**: `retriever.query_topical(topical_question_for_plan_2)` —
   short human-phrased question about session-2's topic that does NOT
   contain the planted phrase; TF-IDF routing must pick a window in
   session-2 on topical overlap alone.
3. **Entity-mention**: `retriever.query_entity_mention(entity_question_for_plan_3)`
   — short human-phrased question that explicitly names session-3's
   distinctive entity ("Gosford", "quokka"); overlap routing must pick
   a window in session-3.

For each query the verification pipeline captures:

- the six-key `strict_assertions` dict from `QueryResult`
- a `verbatim_hit` boolean: `planted_phrase in result.generated_answer`
- the `source_session` that actually served the routed window
- the `routing_score` when applicable

## 5. Verification report

`VerificationReport` (pydantic) persisted as JSON includes:

- `num_sessions`, `session_ids`, `timestamp`
- `total_tokens`, `token_budget_met` (>128_000)
- `query_executions: list[QueryExecution]` (one per query)
- `all_six_strict_per_query_pass: bool`
- `verbatim_recall_per_query: list[bool]`
- `zero_mod_upstream: dict[str, str]` — relative path -> "clean" or
  diff excerpt
- `acceptance_passed: bool` — single master pass/fail flag

## 6. Zero-modification enforcement

Scope binding prohibits any edit outside
`src/chuk_lazarus/cross_session_demo/**`,
`tests/cross_session_demo/**`,
`examples/cross_session_demo.py`,
`scripts/cross_session_demo.sh`,
`docs/turn_aligned/cross_session_demo*.md`.

The upstream axes (chat_loop / session_close / session_store /
session_retrieval) and the build tool `tools/build_clause_aligned_store.py`
are Read-only. `report.build_report` runs `git diff --name-only` over
these paths; any non-clean entry reduces `acceptance_passed` to
`False` regardless of query outcomes.

## 7. Failure semantics (fail-fast)

- axis-3 subprocess non-zero returncode → `RuntimeError` inside
  `pipeline.run_session` with the stderr tail.
- missing Gemma snapshot → `RuntimeError` from
  `token_budget.measure_total_tokens` pointing operators at the HF
  cache path.
- `torch.cuda.is_available() == False` → `SessionRetriever.from_checkpoint_root`
  raises `RuntimeError("STRICT: CUDA device requested ...")` (upstream).
- Any strict-mode assertion fails → the retriever itself raises
  `RuntimeError` inside `_generate_from_window`. The cross-session
  demo catches nothing; a raised error terminates the run with a
  non-zero exit code.

## 8. CLI and wrappers

- `python -m chuk_lazarus.cross_session_demo.cli --inputs-root P --checkpoints-root Q --report-out R.json`
  - orchestrates everything end-to-end
  - writes `R.json` (VerificationReport)
  - exits 0 iff `acceptance_passed == True`
- `examples/cross_session_demo.py` — thin wrapper importing `run_demo`
- `scripts/cross_session_demo.sh` — bash wrapper that sets up
  `<out>/inputs` and `<out>/checkpoints` then invokes the CLI.

## 9. Tests

| file | type | gating |
|------|------|--------|
| `test_session_generator.py` | unit | always |
| `test_token_budget.py` | unit | skip if gemma snapshot missing |
| `test_pipeline.py` | integration (1 session build) | `@pytest.mark.cuda` |
| `test_verification.py` | integration (full 5-session demo) | `@pytest.mark.cuda` |
| `test_six_strict_assertions.py` | integration | `@pytest.mark.cuda` |
| `test_verbatim_recall.py` | integration | `@pytest.mark.cuda` |
