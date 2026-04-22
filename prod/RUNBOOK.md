# RUNBOOK — turn-aligned-canonical-port

Everything you need to run, test, and extend the canonical prefill port. All paths are relative to the repo root (one level above `prod/`).

## Prerequisites

- **GPU:** CUDA device with ≥32 GiB (Gemma-4-E2B-it is an 8B-parameter model)
- **Python:** 3.12 with the project's `pyproject.toml` installed (`uv sync` or equivalent)
- **Transformers:** must include Gemma-4 modeling (the OOM patch works around a deterministic trap in `get_per_layer_inputs` — see `prod/ARCHITECTURE.md` §Gemma-4 adaptation)
- **Model weights:** Gemma-4-E2B-it (`google/gemma-4-e2b-it` or local checkpoint; `HF_HOME` pin honoured by wrapper scripts)
- **Environment:** `LAZARUS_MAX_NEW_TOKENS` controls decode length (default 120)

## 1. Run the regression test suite

```bash
pytest tests/session_retrieval/ -v
```

**Expected:** `45 passed, 1 skipped, 0 failed`. This suite covers the retriever, the strict assertions, the pinned-file invariants for zero-mod primitives, and deterministic-tie-break behaviour.

Evidence of last-good run: `prod/validation/06-pytest-round3.log`.

## 2. Direct UUID probe (fastest smoke test)

Quickest way to verify the new method is wired and Gemma-4 produces coherent output without OOM.

```bash
# Requires /tmp/csd-multi/checkpoints (run the cross_session_demo first if missing)
python prod/validation/direct_probe.py
```

**Expected output:** JSON with 3 probes, `"ok": true`, `"strict_assertions"` all `true`, `"answer_head"` containing the planted phrase verbatim. See `prod/validation/direct_probe_results.json` for the golden run.

## 3. Apollo demo (unified memory store)

Verifies the new method against the Apollo-style unified flat-corpus store.

```bash
# One-time build (~200s, writes to /tmp/unified-memory/)
python scripts/build_unified_memory_store.py

# Run the demo — 3 queries, each goes through generate_with_residual_prefill_seeded
bash scripts/run_apollo_memory_demo.sh
```

**Expected:** 3/3 coherent English answers, zero OOM. Evidence of last-good run: `prod/validation/08-apollo-q{1,2,3}-round3.log`.

## 4. Criterion-4 end-to-end harness (chat_loop → session_close → fresh retrieval)

The canonical proof that the port works for conversational memory, not just knowledge-store retrieval.

```bash
# Writes to /tmp/csd-criterion4 — pass a different --output-root for a side-by-side run
python scripts/criterion4_e2e_verify.py --output-root /tmp/csd-criterion4

# Exit 0 → PASS (prints CRITERION_4_PASS banner)
# Exit 1 → FAIL (prints CRITERION_4_FAIL banner)
# Exit 2 → ERROR
```

What it does:
1. **axis-1 chat_loop** — scripted conversation with a planted verbatim phrase
2. **axis-2 session_close** — AUS3000 clause emission + condense + wind-down
3. **axis-3 session_store** — clause-aligned torch store build
4. **axis-4 fresh retriever** — empty-context `SessionRetriever`
5. **axis-5 exact-ID probe** — fetch verbatim content from a prior session's handle

Evidence of last-good run: `prod/validation/CRITERION4_VERDICT.md` + `prod/validation/criterion4_report.json` + `prod/validation/09-criterion4-e2e.log`.

## 5. Shell wrapper (full 5-axis flow, fresh sessions)

```bash
bash scripts/cross_session_demo.sh /tmp/csd-custom-output
```

Internally runs axes 1-3 to build 5 fresh sessions, then exercises axis-4 and axis-5 retrieval with multiple query modes (exact-id, topical, entity-mention). Writes `report.json` to the output root.

## 6. Interactive pseudo-infinite-memory chat (the "live" test)

Exercises the full pipeline in a REPL so you can talk to the model, compress the conversation to vectors on demand, and verify the recall in a fresh session. All routing metadata is printed inline.

```bash
scripts/run_interactive_memory_chat.sh
# default store: artifacts/manual/repl_loop_<utcstamp>
# override with --store-root or LAZARUS_STORE_DIR
```

The wrapper also defaults the model to `google/gemma-4-E2B-it` unless
`LAZARUS_MODEL` or `--model-path` overrides it. The raw Python entry point is
still available if you want to manage all flags and env vars yourself.

### Workflow

1. **Chat normally.** The model streams replies; the session accumulates in memory.
2. **Type `/save`** to compress the current session:
   - transcript JSON → `<store>/transcripts/<session_id>.json`
   - AUS3000 clauses → `<store>/inputs/<session_id>/`
   - clause-aligned torch store → `<store>/checkpoints/<session_id>/`
   - retriever is rebuilt against the new store
3. **Type `/new`** to start a fresh session. From this point on, every turn where the memory-mode isn't `off` will:
   - run `retriever.query_topical(...)` against your input + recent-context window
   - inject the matched window's residual at `layers[0]` position 0 via `generate_with_residual_prefill_seeded` (the new canonical port)
   - print a **ROUTING + STRICT ASSERTIONS** debug block above every reply:
     - `routing_mode`, `source_session`, `window_id`, `routing_score`
     - first ~220 chars of `matched_window_text`
     - top keywords from the routed window
     - all six strict assertions (`cuda_available`, `model_on_cuda`, `residual_compatible`, `hook_fired`, `gpu_memory_grew`, `store_window_nonempty`)
     - timing split (retrieve / generate / total)
     - token counts (prompt / generated)

### Slash commands

| Command | Effect |
|---|---|
| `/save` | compress current session to the store, keep chatting |
| `/new` | `/save`, then start a fresh session |
| `/query <text>` | topical recall probe (no chat-history mutation) |
| `/exact <handle>` | exact-id recall probe (e.g. `11a1c9ad.1.0`) |
| `/entity <text>` | entity-mention recall probe |
| `/stats` | store summary (sessions, windows, tokens, crystal_layer) |
| `/last` | reprint the last turn's routing metadata |
| `/history` | dump the current session transcript |
| `/memory [topical\|entity_mention\|off]` | toggle / set recall mode |
| `/help` | show command list |
| `/quit` or `/exit` or Ctrl-D | prompt to save, then exit |

### Environment

| Var | Default | Purpose |
|---|---|---|
| `LAZARUS_STORE_DIR` | `/tmp/interactive-memory` | persistent store root |
| `LAZARUS_MODEL` | local Gemma snapshot → `google/gemma-4-E2B-it` | model id/path |
| `LAZARUS_MAX_NEW_TOKENS` | `180` | decode length |
| `LAZARUS_MEMORY_MODE` | `topical` | recall routing mode for post-save turns |

### Example session

```
you> My dog's name is Banjo and he loves chasing dragonflies at dusk.
gemma> That sounds lovely ...

you> /save
[save] AUS3000 clauses -> 6 record(s) under /tmp/interactive-memory/inputs/b4f8…
[save] torch store built -> /tmp/interactive-memory/checkpoints/b4f8… in 38.2s
[save] retriever refreshed: 1 session(s) indexed · crystal_layer=29

you> /new
[new] fresh session started.

you> What was my dog's name again?
===== ROUTING + STRICT ASSERTIONS ===================================
  mode            : topical
  source_session  : b4f8e9a4c7e14d2a8f1b…
  window_id       : 3
  routing_score   : 0.7421
  matched_window  : Turn 1 on dog-chat: My dog's name is Banjo and he loves
                    chasing dragonflies at dusk…
  keywords        : ['banjo', 'dog', 'dragonflies', 'dusk', 'chasing']
  strict_asserts  : cuda_available=True model_on_cuda=True residual_compatible=True
                    hook_fired=True gpu_memory_grew=True store_window_nonempty=True
  timing (s)      : retrieve=0.43  generate=1.82  total=2.25
  tokens          : prompt=287  generated=52
======================================================================
gemma> Your dog's name is Banjo. You mentioned he loves chasing dragonflies at dusk.
```

### What to eyeball

- **`hook_fired=True`** — confirms the `forward_pre_hook` on `layers[0]` ran (i.e. residual was injected). If `False`, the new canonical method didn't run.
- **`routing_score`** — topical TF-IDF score. Low scores (<0.1) mean the retriever grabbed a marginal window; the answer may not actually relate to your query. Try `/exact` with a specific handle, or plant more distinctive content.
- **`matched_window`** — the actual prior-session excerpt being anchored in the KV cache. If it contains the answer verbatim, the output echoing it is retrieval-pipeline working; if the question forces the model to reconstruct *without* the window text echoing the answer, that's stronger residual-only recall.
- **timing split** — `retrieve` is the topical TF-IDF + store-load cost; `generate` is the forward_pre_hook + `generate()` call. Retrieval dominating means your store is big / keywords expensive; generation dominating is normal for Gemma-4.
- **`gpu_memory_grew=True`** — sanity that generation actually ran (didn't silently early-exit).

## 7. Multi-probe (stress test)

```bash
python scripts/multi_probe_test.py       # 15-query harness (has a pre-existing UUID bug — see NEXT-RUN WORK)
python scripts/multi_probe_query_only.py # query-only variant (same caveat)
```

**Known issue:** these scripts regenerate `session_id` via `uuid.uuid4().hex` per call, which never matches on-disk checkpoint UUIDs. They are safe to run but will report misses against pre-built stores. Fix is a next-run item — see `prod/CHANGELOG.md`.

## Extend the port

### A) Add a new query mode (e.g. semantic-search)

1. Open `src/chuk_lazarus/session_retrieval/retriever.py`
2. Add a new method alongside `query_exact_id`, `query_topical`, `query_entity_mention`
3. Route it through `_generate_from_window` (line 394 calls `generate_with_residual_prefill_seeded`)
4. Add a test under `tests/session_retrieval/` — follow the pattern in `test_topical.py` or `test_entity_mention.py`
5. Update `criterion4_e2e_verify.py` if the new mode should be part of acceptance

### B) Harden the build pipeline (sporadic 1-of-5 session-build failure)

The sporadic failure lives in `tools/build_clause_aligned_store.py`, which is a **zero-mod primitive**. Changes require explicit supervisor scope-grant before starting. File a scope-expansion request first, then:

1. Reproduce with `scripts/build_session_store.sh`
2. Root-cause via `tools/build_clause_aligned_store.py` + capture logs
3. Patch with the scope-grant reference in the commit message

### C) Residual-only recall (tighter criterion-4)

The current criterion-4 exit condition accepts verbatim hits where the matched window text is echoed into the prompt template. A stricter test would ask a question that **doesn't** cause the stored text to be echoed into the prompt, proving recall from the residual alone. This is a next-run item — see `prod/CHANGELOG.md` §Follow-ups.

## Troubleshooting

### OOM on generate call

You are on the `inputs_embeds` path, not the `input_ids` path. The new method `generate_with_residual_prefill_seeded` must be the one invoked — verify via:

```bash
grep -n generate_with_residual_prefill_seeded src/chuk_lazarus/session_retrieval/retriever.py
# expected: 394:  result = self.runtime.generate_with_residual_prefill_seeded(prompt, residual_state, gen_config)
```

If you see `generate_with_residual` (without `_prefill_seeded`), you're on the old path — rewire.

### `hook_fired=False` in strict_assertions

The `forward_pre_hook` on `layers[0]` isn't installing. Most common cause: the retriever's spy hook was installed first and blocked the pre-hook. Check `retriever.py` — the spy hook is a `forward_hook` (post), not a `forward_pre_hook`, so the two should coexist. If not, file a bug-report.

### Model outputs token salad

If the output is coherent but wrong-topic → the store may be stale. Rebuild with `build_session_store.sh`. If the output is actual token salad → you're likely on the old method; see §OOM above.

### Session generator plant strategy drift

`src/chuk_lazarus/cross_session_demo/session_generator.py` has uncommitted changes flipping the plant strategy from **5x repetition** to **1x natural**. Pre-built checkpoints at `/tmp/csd-multi` were built with 5x; `/tmp/csd-criterion4` was built fresh with the current 1x strategy. If you rebuild stores you'll get 1x; if you query the old 5x stores the answer may repeat the planted phrase multiple times.

## vee workspace

All run-1 records are in `.vee/` (gitignored — local only). To replicate on a fresh clone, the handoff record is re-fetchable via its title; see `prod/VEE_RECORDS.md` for curated IDs.
