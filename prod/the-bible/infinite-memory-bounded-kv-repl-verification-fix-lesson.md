# Infinite Memory Bounded KV REPL Verification Fix Lesson

## TL;DR

The Gemma-4 seeded residual fix proved one leaf: boundary injection can work.
It did not prove that the full chat REPL composes correctly across saving,
routing, KV-direct materialization, token budgeting, crash recovery, and
concurrent writes.

The right lesson from this run is simple: every subsystem needs a runnable
proof, and every proof must avoid depending on accidental data shape.

We added an operator harness at `scripts/auto_verify_memory_repl.py` and a CI
wrapper at `scripts/run_infinite_memory_ci_gate.sh`. The harness drives the
real `MemoryChat` path, logs every check to JSONL plus transcript files, and
exits non-zero on the first failed invariant.

## What The Harness Proves

The harness is designed to prove these 12 checks:

- `PREFLIGHT`
- `LIVE_SAVE`
- `ROUTING_SCALE`
- `TOPICAL_RECALL`
- `PROBE_NO_MUTATION`
- `KV_DIRECT_RECALL`
- `TOKEN_BUDGET`
- `VRAM_BOUNDED`
- `FALLBACK_TRUTH`
- `MEMORY_OFF`
- `CRASH_GATE`
- `PARALLEL_WRITES`

Passing smoke does not replace the full 100-session x 100-turn proof. It proves
the automation path is meaningful enough to run at full scale.

The current handoff state is:

- Smoke proof: rerun after the primary-marker fix; the previous pass was
  masking a cold-start tie-break flake.
- Full proof entry point: `make verify-infinite-memory`.
- Default full workload: 100 sessions x 100 turns.
- CI contract: exit 0 only when every scripted invariant passes; first failure
  exits non-zero with a named invariant and evidence fields.

This is intentionally stricter than "the REPL seemed to work". The script is
allowed to be annoying. If it fails, it should leave the next agent with the
smallest named bug, not a mystery pile.

## Failure 1: Routing Scale Picked The Wrong Session

Symptom:

`ROUTING_SCALE` returned the earlier `LIVE_SAVE` session for every scale probe.
The top score was `Infinity`, which looked important but was only UCB1 cold
start behavior.

Root cause:

The harness planted markers like:

```text
SCALE_MARKER_<runid>_0000_0000
SCALE_MARKER_<runid>_0001_0000
```

Every scale marker shared the same run id. The `LIVE_SAVE` marker also shared
that run id. Under cold-start UCB1, every candidate has `ucb1_score = inf`, so
the effective tie-break became raw TF-IDF. The shared run id and words like
`memory`, `marker`, and `exact` polluted the discriminator.

Fix:

`scripts/auto_verify_memory_repl.py` now uses `secrets.token_hex(8)` per scale
marker and queries only that unique key. It also logs the top 5 candidates and
adds a note when `Infinity` means a UCB1 cold-start tie.

Lesson:

Routing tests must not plant identifiers with shared high-IDF substrings unless
the test is explicitly about shared-substring ambiguity.

## Failure 2: Topical Recall Picked A Scale Window

Symptom:

`TOPICAL_RECALL` asked for the primary dog-name marker, but routing returned a
scale-session window containing a different random key.

Root cause:

This was the same marker-design bug from the opposite direction. The scale
markers had been fixed to random hex, but the primary marker still used:

```text
MEMORY_MARKER_PRIMARY_<runid>
```

When all sessions were cold, UCB1 made the primary score `Infinity` for every
candidate. If normalized raw router scores tied, the final tie-break was
alphabetical `(session_id, window_id)`. In one run the live-save session won by
luck; in the next run a scale session UUID sorted earlier and won.

Fix:

`scripts/auto_verify_memory_repl.py` now seeds the primary and color markers
with `secrets.token_hex(8)` too. The discriminating token no longer contains
the run id, `MEMORY`, `MARKER`, or any other shared prefix. The harness logs the
chosen markers in `events.jsonl` under `markers.seeded`. The live-save fixture
also repeats the unique key in the planted text, matching the scale fixture's
"key appears twice" shape so the raw router has a stronger clean signal.

Lesson:

Every planted fact needs an independent discriminator. Fixing only the scale
markers left the primary recall probe able to fail on router cold-start
tie-break luck.

## Failure 3: Token Budget Did Not Bind

Symptom:

`TOKEN_BUDGET` expected the governor to drop assignments, but sometimes only
two assignments reached `_apply_token_budget`. With budget `2048` and
per-fact cost `1024`, two facts fit exactly, so no drop occurred.

Root cause:

The test borrowed whichever handle ASI routed to from the earlier scale pool.
`kv_query_turn` intentionally filters assignments to one handle:

```python
top_handle = tier_assignments[0].candidate.handle
assignments_for_handle = [
    a for a in tier_assignments
    if a.candidate.handle.session_id == top_handle.session_id
]
```

That is correct because KV materialization is per checkpoint. The test was
wrong because it assumed the selected handle would have enough windows to
overflow the budget.

Fixes:

- `scripts/interactive_memory_chat.py` now supports
  `LAZARUS_MAX_TOTAL_INJECT_TOKENS`.
- `scripts/auto_verify_memory_repl.py` creates a dedicated 8-window
  budget-pressure session.
- The harness temporarily sets `LAZARUS_MAX_TOTAL_INJECT_TOKENS=1024` only for
  the `TOKEN_BUDGET` check, then restores the previous environment.

Lesson:

An integration proof can still own its load shape. Do not depend on unrelated
earlier checks to accidentally create the right pressure.

## Failure 4: In-Flight Manifests Were Enumerated

Symptom:

`TOKEN_BUDGET` fell back before reaching the budget governor:

```text
mode: none
kv_direct_active: false
no_silent_fallback: false
```

The warning said ASI could not score a store because `window_tokens.npz` and
`idf.json` were missing.

Root cause:

`LiveIndexer` writes an interim manifest with:

```json
{
  "in_flight": true
}
```

That manifest means "not ready". But
`src/chuk_lazarus/session_retrieval/enumeration.py` treated any manifest as a
retrieval-ready checkpoint. Unsaved sessions from earlier checks created
orphan in-flight stores, and `iter_checkpoint_handles()` let them into the
retriever pool.

Fix:

`iter_checkpoint_handles()` now skips manifests where `in_flight is True`
before validity checks. A regression test was added in
`tests/session_retrieval/test_enumeration.py`.

The harness `CRASH_GATE` was strengthened to prove both incomplete cases:

- no manifest
- manifest exists with `in_flight: true`

Lesson:

Readiness gates must be honored by every reader. A sentinel flag that no
consumer respects is not a safety mechanism.

## Verification Artifacts

Each harness run writes:

```text
prod/validation/repl-autoverify/<timestamp>-<run-id>/
  events.jsonl
  transcript.log
  summary.json
  environment.json
  git-metadata.json
```

Future agents should start with `summary.json`, then inspect `events.jsonl`,
then use `transcript.log` for full REPL context.

The CI wrapper also writes a top-level gate log:

```text
prod/validation/infinite-memory-ci-gate-<timestamp>.log
```

That log captures wrapper preflight, the exact harness command, and the full
Python harness stream. It is useful when CI preserves only a single uploaded
log file.

## How To Run

Smoke:

```bash
python scripts/auto_verify_memory_repl.py --smoke --fresh
```

Full proof gate:

```bash
make verify-infinite-memory
```

CI wrapper directly:

```bash
bash scripts/run_infinite_memory_ci_gate.sh --full
```

Smoke through the same CI wrapper:

```bash
bash scripts/run_infinite_memory_ci_gate.sh --smoke
```

Useful full-run overrides:

```bash
bash scripts/run_infinite_memory_ci_gate.sh --full --model-path /models/gemma-4-e2b-it
bash scripts/run_infinite_memory_ci_gate.sh --full --store-root /fastssd/lazarus-memory-proof
bash scripts/run_infinite_memory_ci_gate.sh --full -- --vram-ceiling-mib 30000
```

The wrapper exits with the same status as the Python harness. Exit 0 means the
selected proof passed. Any non-zero exit is a real gate failure and should stop
the pipeline.

The wrapper also checks that the seeded-residual lesson exists before running.
That makes the task #9 documentation dependency visible in CI instead of
letting the proof machinery drift away from the Apollo/Gemma-4 fix that made
it possible.

## CI Integration Pattern

Use this as the CI gate step on a Linux/CUDA runner:

```bash
make verify-infinite-memory
```

The gate should archive:

- `prod/validation/infinite-memory-ci-gate-*.log`
- `prod/validation/repl-autoverify/*/summary.json`
- `prod/validation/repl-autoverify/*/events.jsonl`
- `prod/validation/repl-autoverify/*/transcript.log`

Do not convert failures into warnings. A non-zero exit means the system did not
prove infinite memory plus bounded KV for that workload.

## Code Reading Trail

- `scripts/auto_verify_memory_repl.py`
- `scripts/run_infinite_memory_ci_gate.sh`
- `scripts/interactive_memory_chat.py`
- `src/chuk_lazarus/session_retrieval/enumeration.py`
- `tests/session_retrieval/test_enumeration.py`
- `prod/the-bible/apollo-gemma4-zero-injection-token-salad-fix-lesson.md`

## Relationship To The Apollo Gemma-4 Fix

The Apollo/Gemma-4 fix solved the residual injection invariant: do not overwrite
the final live prompt token; inject into a seeded carrier slot instead.

This lesson is the next layer up. It proves that the chat REPL must validate
composition:

- save lifecycle
- live indexer readiness
- routing quality
- residual/KV injection
- token-budget pressure
- fallback truthfulness
- crash visibility
- concurrent session isolation

The seeded residual fix made the REPL worth testing. The REPL verifier makes it
hard to ship a false memory system by accident.
