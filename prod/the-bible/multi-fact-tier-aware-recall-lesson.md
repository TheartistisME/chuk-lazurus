# Multi-Fact Tier-Aware Recall Lesson

## TL;DR

Single-fact infinite memory proves that the REPL can find one exact planted
fact from one old session. Multi-fact recall proves the harder thing: one query
can route across many sessions, select many facts, apply HOT/WARM/COLD attention
policy, and answer with only the facts that the tier policy allows.

The shipping path is:

1. Save every session as clause/window stores, including boundary residuals and
   full residual streams.
2. Route a query across all checkpoint handles with `asi_route_candidates`.
3. Assign tiers with `assign_tiers`.
4. Keep assignments across multiple sessions in `/kv_query`.
5. Materialize K/V from each selected session/window and concatenate them into
   one synthetic archived prefix.
6. Apply `apply_tier_attention_mask` at generation time so HOT/WARM/COLD have
   different attention behavior.
7. Decode the answer from the tier-selected HOT/WARM semantic surface.

The proof is `MULTI_FACT_RECALL`, now the 14th invariant in
`make verify-infinite-memory`.

## The Puzzle This Solves

The old path passed single-fact recall because every probe asked for one marker
from one session. That left three gaps:

- `/kv_query` filtered assignments to only the top session handle.
- topical multi-window mode could stuff windows into the prompt, but it did not
  inject multiple residual/KV sources.
- no harness check proved HOT facts win, WARM facts contribute, and COLD facts
  are muted in one semantic answer.

The new invariant plants 12 different color facts across 12 sessions and asks
for the website color-scheme decisions in one query. It requires:

- all 12 facts appear in the routed candidate pool;
- tiers split 4 HOT, 4 WARM, 4 COLD;
- the generated answer mentions all 4 HOT facts;
- the answer mentions at least 3 of 4 WARM facts;
- the answer mentions 0 COLD facts;
- telemetry reports `mask_penalty_applied=True`;
- telemetry reports `selected_tier=hot`.

## Implementation Map

Main operator path:

- `scripts/interactive_memory_chat.py`
  - `kv_query_turn` routes candidates, assigns tiers, applies token budget, and
    calls the multi-session KV-direct path when available.
  - The old single-handle filter remains only as a legacy fallback.
  - `/save` records residual stream metadata in save-state.

Retrieval and materialization:

- `src/chuk_lazarus/session_retrieval/retriever.py`
  - `prepare_kv_direct_materialization` gathers residuals for one checkpoint.
  - `answer_with_kv_direct_multi` groups assignments by session, materializes
    each group, concatenates K/V, creates collision-free synthetic ranges, and
    calls the runtime with tier metadata.
  - The answer path uses a bounded HOT/WARM semantic prefix. COLD windows are
    never placed in that prefix.

Residual stream storage:

- `src/chuk_lazarus/session_store/live_indexer.py`
  - writes `torch_store/residual_streams/window_NNN.npy`;
  - includes residual stream readiness in selection descriptors and manifest.
- `src/chuk_lazarus/inference/context/knowledge/torch_capture.py`
  - can return both boundary residual and full per-token residual stream.
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
  - loads `residual_streams/window_NNN.npy`.
- `src/chuk_lazarus/inference/context/knowledge/torch_build.py`
  - offline store builds now preserve residual streams too.

Bounded K/V construction:

- `src/chuk_lazarus/inference/backends/_torch_residual_bounded.py`
  - `gather_selected_residuals(..., residual_mode="stream")` gathers rank-2
    per-window streams and records token ranges.
  - `materialize_kv_direct` projects selected residual slots through K/V
    projections.
  - If an older checkpoint predates residual streams, stream gathering warns and
    falls back to the boundary residual instead of crashing.

Tier math:

- `src/chuk_lazarus/session_retrieval/tier_policy.py`
  - `assign_tiers(K_HOT=4, K_WARM=4, candidate_pool=12)` produces HOT/WARM/COLD
    partitions from the already-ranked candidates.
- `src/chuk_lazarus/inference/backends/torch_runtime.py`
  - `apply_tier_attention_mask` is still the bit-exact axis-4 contract:
    - COLD slots get `-inf`, so post-softmax weight is exactly 0.
    - WARM slots subtract `warm_config.penalty_value`, default 4.0.
    - HOT slots add `hot_bonus_value`, default 0.0, so default HOT is identity.

Harnesses:

- `scripts/auto_verify_memory_repl.py`
  - adds the `MULTI_FACT_RECALL` check to the 14-check suite.
- `scripts/verify_memory_recall_scale.py`
  - adds `--mode multi_fact`;
  - uses a disposable `scale-multi-fact-store` so probes do not get polluted by
    older validation checkpoints.

## Query-Time Flow

The query:

```text
List all favorite color answers for prior website color scheme palette
decisions tagged <topic_key>. Rank by relevance to design palette decisions.
Return only the color words.
```

The flow:

1. `asi_route_candidates` searches every checkpoint handle and returns the top
   12 windows.
2. The harness confirms those 12 windows cover the 12 planted facts.
3. `assign_tiers` marks ranks 0-3 HOT, ranks 4-7 WARM, and ranks 8-11 COLD.
4. `kv_query_turn` keeps all budgeted tier assignments across sessions instead
   of filtering to `tier_assignments[0].candidate.handle`.
5. `answer_with_kv_direct_multi` groups assignments by session because each
   checkpoint has its own store.
6. For each group, residual streams are loaded and projected into K/V.
7. K/V tensors are concatenated along the archived-prefix axis.
8. Synthetic window ids and `per_window_token_ranges` are built so `window_id=0`
   from different sessions cannot collide.
9. The runtime receives:
   - combined K/V;
   - per-window token ranges;
   - HOT/WARM/COLD tier labels;
   - warm penalty config.
10. `apply_tier_attention_mask` fires during attention over the archived prefix.
11. The semantic decoder receives only HOT/WARM selected facts. COLD facts are
    excluded from the semantic prefix and hard-muted by attention policy.
12. Telemetry is returned through `TurnMetadata`.

## What The 14th Invariant Proves

`MULTI_FACT_RECALL` proves this exact behavior:

```text
HOT  = 4/4 recalled
WARM >= 3/4 recalled
COLD = 0/4 recalled
selected_tier = hot
mask_penalty_applied = True
kv_direct_active = True
no_silent_fallback = True
```

It also indirectly proves that the single-handle filter is gone from the active
multi path, because the 12 planted facts live in 12 different sessions.

The full suite still proves the original foundations:

- `PREFLIGHT`
- `LIVE_SAVE`
- `ROUTING_SCALE`
- `TOPICAL_RECALL`
- `PROBE_NO_MUTATION`
- `KV_DIRECT_RECALL`
- `MULTI_FACT_RECALL`
- `TOKEN_BUDGET`
- `VRAM_BOUNDED`
- `FALLBACK_TRUTH`
- `INFINITE_TURN_LATENCY`
- `MEMORY_OFF`
- `CRASH_GATE`
- `PARALLEL_WRITES`

## Important Caveat

Do not oversell this as "pure residual-only natural-language decoding for every
fact shape."

The proof exercises real multi-session routing, real K/V materialization, real
tier attention mask telemetry, and real HOT/WARM/COLD answer behavior. For
color-style facts, the production multi-fact answer path also uses a bounded
HOT/WARM semantic prefix plus extracted-value synthesis. That exists because
Gemma-4-E2B-it did not reliably decode arbitrary color words from residual
streams alone in early probes.

COLD facts are still excluded twice:

- by the attention mask (`-inf` over COLD slots);
- by not entering the HOT/WARM semantic answer prefix.

There is a follow-up bead to harden or separately prove residual-only decoding
for broader fact forms: `chuk-lazurus-82f`.

## Why The Scale Verifier Uses A Disposable Store

`scripts/verify_memory_recall_scale.py --mode multi_fact` plants 4 facts per
probe, asks for all 4, and repeats this N times. It intentionally uses:

```text
prod/validation/repl-autoverify/<run>/scale-multi-fact-store
```

and clears it before each probe.

That keeps old validation checkpoints from polluting the top candidates. In an
earlier run, late probes started routing into accumulated prior probe sessions
and then an old checkpoint without residual streams crashed the stream loader.
The disposable store makes the scale verifier about repeated correctness, not
about accidental historical clutter.

In `--mode multi_fact`, all four facts are HOT (`K_HOT=4`, `K_WARM=0`), so
`mask_penalty_applied` may be false there. The tier-mask proof with WARM/COLD
is the harness invariant `MULTI_FACT_RECALL`.

## Commands To Reproduce

Smoke:

```bash
bash scripts/run_infinite_memory_ci_gate.sh --smoke
```

Full 100 x 100 proof:

```bash
make verify-infinite-memory
```

Multi-fact scale verifier:

```bash
uv run --extra dev python scripts/verify_memory_recall_scale.py \
  --sample-size 50 \
  --mode multi_fact \
  --required-hit-rate 0.95 \
  --quiet-model-output
```

Expected success shape:

```text
PASS SCALE_ACTUAL_RECALL: mode=multi_fact hit_rate=1.000 passed=50/50
```

## Known Green Artifacts

Full harness:

```text
prod/validation/repl-autoverify/20260426T140535Z-20260426t140535/summary.json
```

Summary:

```text
status: PASS
checks: 14
MULTI_FACT_RECALL: multi-fact recall HOT=4/4 WARM=4/4 COLD=0/4 selected_tier=hot
VRAM_BOUNDED: vram peaks=[20013.0, 20013.0, 20013.0] delta=0.0 MiB
INFINITE_TURN_LATENCY: 50 measured turns flat
```

Scale verifier:

```text
prod/validation/repl-autoverify/20260426T083505Z-20260426t083505/scale-actual-recall-multi_fact.json
```

Summary:

```text
mode: multi_fact
sample_size: 50
passed: 50
hit_rate: 1.000
required_hit_rate: 0.950
```

## Example Harness Output

Full `MULTI_FACT_RECALL` planted these facts under:

```text
topic_key = website_palette_decision_20260426t140535_654f163f
```

The query was:

```text
List all favorite color answers for prior website color scheme palette
decisions tagged website_palette_decision_20260426t140535_654f163f.
Rank by relevance to design palette decisions. Return only the color words.
```

The model answered:

```text
periwinkle, chartreuse, crimson, ultramarine, cerulean, indigo, vermillion, turquoise
```

The tier split was:

```text
HOT:  periwinkle, chartreuse, crimson, ultramarine
WARM: cerulean, indigo, vermillion, turquoise
COLD: malachite, saffron, ochre, magenta
```

Assertions:

```text
HOT hits:  4/4
WARM hits: 4/4
COLD hits: 0/4
mask_penalty_applied: True
selected_tier: hot
kv_direct_active: True
no_silent_fallback: True
```

Example scale verifier probes:

```text
Query:
List all favorite color answers from prior website color scheme palette
decisions in group scale_multi_fact_0001_9eadc90c11cf. Return only the
four color words.

Expected:
cerulean, saffron, chartreuse, vermillion

Answer:
cerulean, saffron, vermillion, chartreuse
```

```text
Query:
List all favorite color answers from prior website color scheme palette
decisions in group scale_multi_fact_0002_b8bb60bea712. Return only the
four color words.

Expected:
saffron, chartreuse, vermillion, indigo

Answer:
vermillion, chartreuse, indigo, saffron
```

```text
Query:
List all favorite color answers from prior website color scheme palette
decisions in group scale_multi_fact_0005_2cbbd7c7c366. Return only the
four color words.

Expected:
indigo, magenta, ochre, turquoise

Answer:
ochre, magenta, indigo, turquoise
```

## Failure Signs

If `MULTI_FACT_RECALL` fails, inspect in this order:

1. `events.jsonl` for `multi_fact.session_saved` and `multi_fact.meta`.
2. Whether `candidate_pool=12` covered all planted markers.
3. Whether `assign_tiers` produced 4/4/4.
4. Whether `/kv_query` called `answer_with_kv_direct_multi` instead of legacy
   single-handle fallback.
5. Whether `matched_window_text` includes multiple session ids.
6. Whether `mask_penalty_applied` is true for the 12-fact harness.
7. Whether old checkpoints lack `residual_streams`; they should warn and fall
   back, not crash.

The quickest sanity check is:

```bash
uv run --extra dev python scripts/auto_verify_memory_repl.py --list-checks
```

It should list 17 checks and include `MULTI_FACT_RECALL`,
`REAL_WORLD_MULTI_FACT_RECALL`, `DIRTY_STORE_REAL_WORLD_RECALL`, and
`MEMORY_LAWS_REAL_WORLD_RECALL`.

## Real-World Color-Scheme Invariant

The follow-on proof is `REAL_WORLD_MULTI_FACT_RECALL`. It plants twelve
natural website color-scheme memories across twelve sessions and asks the
realistic user query:

```text
Tell me everything we discussed about the website's color scheme across all our sessions.
```

The check keeps the same bounded KV path and telemetry contract, but scores
semantic synthesis instead of exact planted color tokens:

```text
HOT:  4/4 natural memories must be mentioned.
WARM: at least 3/4 natural memories must be mentioned.
COLD: 0 COLD-only details may appear.
```

It also requires a preserved revision and a current/final decision. A passing
answer should capture the shape of the actual design history:

```text
Teal was considered and later replaced by sage. The purple-blue gradient was
rejected. The final direction is warm white backgrounds, graphite headings,
sage accents, and amber primary CTAs.
```

The green smoke run on 2026-04-26 reported:

```text
PASS REAL_WORLD_MULTI_FACT_RECALL:
natural multi-fact recall HOT=4/4 WARM=4/4 COLD=0/4
conflict_preserved=True
final_decision_present=True
selected_tier=hot
mask_penalty_applied=True
```

The scale verifier now has an optional natural mode:

```bash
uv run --extra dev python scripts/verify_memory_recall_scale.py \
  --sample-size 25 \
  --mode real_world_multi_fact \
  --required-hit-rate 0.90
```

The green 25-probe run on 2026-04-26 reported:

```text
PASS SCALE_ACTUAL_RECALL:
mode=real_world_multi_fact hit_rate=1.000 passed=25/25
```

## Dirty-Store Real-World Invariant

The next proof is `DIRTY_STORE_REAL_WORLD_RECALL`. It keeps the same natural
website color-scheme query, but first plants a cluttered long-term store:

- configurable dirty sessions, with smoke mode using 50 and full mode defaulting
  to 500;
- near-miss color projects, stale decisions, duplicate memories, contradictory
  old facts, unrelated domains, same words in wrong projects, and long irrelevant
  sessions;
- twelve target website color-scheme memories across twelve target sessions;
- two non-color mini-probes covering pricing decisions and bug history.

The query still has no marker and no answer list:

```text
Tell me everything we discussed about the website's color scheme across all our sessions.
```

The invariant requires dirty routing and synthesis to prove all of these at
once:

```text
target session coverage >= 10/12
HOT semantic recall = 4/4
WARM semantic recall >= 3/4
COLD-only details = 0/4
near_miss_leak_count = 0
wrong_project_detail_count = 0
stale_fact_marked_or_excluded = True
conflict_preserved = True
final_decision_present = True
```

Telemetry also has to stay strict:

```text
kv_direct_active = True
mask_penalty_applied = True
selected_tier = hot
no_silent_fallback = True
multi_session_count = True
semantic_prefix_active = True
```

The dirty-store proof adds session-deduped routing for the harness and `/kv_query`
path through `LAZARUS_KV_ROUTE_CANDIDATE_POOL` and
`LAZARUS_KV_DEDUP_SESSION`. That lets the router search a wider candidate pool
without stuffing the answer prompt, then keep one best candidate per session
before assigning HOT/WARM/COLD tiers.

The router also gives natural named project terms a bounded exact-mention
boost. That matters for dirty stores because a query about `Atlas` pricing or
`Meridian` checkout should not be drowned out by generic memories that happen
to share words like "current", "sessions", or "website".

The scale verifier has a matching dirty mode:

```bash
uv run --extra dev python scripts/verify_memory_recall_scale.py \
  --sample-size 25 \
  --mode dirty_real_world_multi_fact \
  --required-hit-rate 0.90
```

Each dirty scale probe creates an isolated store, plants noise plus target
natural memories, asks one unmarked query, and fails if wrong-project pollution
appears in the answer.

The WSL smoke run on 2026-04-26 reported:

```text
prod/validation/repl-autoverify/20260426T224203Z-20260426t224203/summary.json
status: PASS
checks: 16
DIRTY_STORE_REAL_WORLD_RECALL:
target_coverage=12/12 HOT=4/4 WARM=4/4 COLD=0/4
near_miss_leak_count=0 wrong_project_detail_count=0
conflict_preserved=True final_decision_present=True
selected_tier=hot mask_penalty_applied=True domain_probes=2/2
VRAM_BOUNDED: vram peaks=[19957.0, 19957.0, 19957.0] delta=0.0 MiB
INFINITE_TURN_LATENCY: 8 measured turns flat
```

The WSL dirty scale run on 2026-04-26 reported:

```text
prod/validation/repl-autoverify/20260426T205135Z-20260426t205135/scale-actual-recall-dirty_real_world_multi_fact.json
mode: dirty_real_world_multi_fact
sample_size: 25
passed: 25
hit_rate: 1.000
required_hit_rate: 0.900
pollution: 0 for every probe
```

## Memory-Laws Real-World Invariant

`MEMORY_LAWS_REAL_WORLD_RECALL` is the next invariant. Positive dirty-store
recall is no longer enough; the system must also know when not to recall and
how to treat stale or confusable memories.

The check proves five laws:

```text
NO_MEMORY_NO_HALLUCINATION
IRRELEVANT_NOISE_INVARIANCE
DUPLICATE_INVARIANCE
TEMPORAL_OVERRIDE
ENTITY_SCOPE
```

The no-memory probe plants dirty unrelated memories and asks:

```text
What did we decide about the Solace website color palette?
```

The only passing shape is explicit absence, for example:

```text
I do not have a stored decision about the Solace website color palette.
```

The Atlas pricing probes then require current facts to remain stable under
noise and duplicate pressure:

```text
Atlas Pro $29 per seat
18% annual discount
14-day trial
$0.08 overage
Enterprise custom quote
final/current decision present
```

Wrong-entity facts such as Nimbus `$39`, Acme `$19`, Acme/Nimbus long trials,
and the old Atlas `$49` draft must not appear as current facts. Duplicate stale
notes are allowed to exist in the store, but repetition must not make them true.

Temporal override is covered by a CTA correction chain:

```text
Crimson was originally planned.
Crimson caused contrast issues, so Amber superseded it.
Amber remains final.
```

Current queries must return Amber. History queries may mention Crimson only as
the superseded earlier decision.

Every sublaw records the bounded-path telemetry:

```text
kv_direct_active
selected_tier
mask_penalty_applied
candidate_count
tier_assignment_count
budgeted_assignment_count
multi_session_count
semantic_prefix_active
no_silent_fallback
candidate_recall_at_4/8/12/64
latency_ms
vram_peak_mib
```

Smoke mode uses noise levels `10,100`; full mode defaults to `10,100,1000`.
Long/nightly sweeps can pass `--memory-laws-noise-levels 10,100,1000,10000`.

The scale verifier has a matching mode:

```bash
uv run --extra dev python scripts/verify_memory_recall_scale.py \
  --sample-size 25 \
  --mode memory_laws \
  --required-hit-rate 0.90
```
