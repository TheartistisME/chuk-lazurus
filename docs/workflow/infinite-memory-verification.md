# Infinite Memory Verification

## Production Gate Result

On the 100 session x 100 turn full harness run, the verifier proved the original
12 production invariants:

| Check | Result | Evidence |
| --- | --- | --- |
| `PREFLIGHT` | PASS | Linux/CUDA, Gemma-4 layer table, REPL import |
| `LIVE_SAVE` | PASS | live-indexed session saved as a loadable store |
| `ROUTING_SCALE` | PASS | `hit_rate=1.000` over 100 planted sessions |
| `TOPICAL_RECALL` | PASS | residual injection recalled the planted Banjo fact |
| `PROBE_NO_MUTATION` | PASS | `/query`, `/exact`, `/entity` left state unchanged |
| `KV_DIRECT_RECALL` | PASS | KV-direct recall ran with no silent fallback |
| `TOKEN_BUDGET` | PASS | token-budget governor dropped assignments under pressure |
| `VRAM_BOUNDED` | PASS | peaks `[19916, 19916, 19916]` MiB, delta `0` |
| `FALLBACK_TRUTH` | PASS | forced failure printed WARN and set `no_silent_fallback=False` |
| `MEMORY_OFF` | PASS | memory-off path returned non-retrieval mode |
| `CRASH_GATE` | PASS | no-manifest and `in_flight=true` stores stayed invisible |
| `PARALLEL_WRITES` | PASS | two concurrent saves loaded without cross-corruption |

The decisive production-scale routing result moved from `0.43` to `1.00`.
The current harness adds a 13th invariant, `INFINITE_TURN_LATENCY`, for
video-style bounded active-KV chat. The full gate exits `0` only when every
current check passes:

```bash
make verify-infinite-memory
```

## Fixes Proven By The Gate

1. Gemma-4 residual injection no longer destroys the last live token because
   the residual path uses a seeded carrier slot.
2. Planted fixtures use independent random hex markers instead of shared run-id
   substrings.
3. `ROUTING_SCALE` cold-start ranking is deterministic:
   high-entropy literals receive an exact token-sequence boost, and ASI routing
   stores `raw_tfidf_score_pre_normalization` for a final tie-break after
   normalized pool score.
4. `LAZARUS_MAX_TOTAL_INJECT_TOKENS` lets the harness force the budget governor
   to bind with a dedicated pressure session.
5. `iter_checkpoint_handles()` skips stores with `in_flight: true`.

## Bounded Active-KV Chat Mode

There are two separate "KV direct" surfaces:

- `/kv_query` is the axis-5 memory-recall path. It materializes archived
  residuals into K/V for selected memory windows.
- `residual_bounded_kv_direct` is the active inference engine. It bounds the
  live per-turn KV cache, keeps a WARM residual session cache, and reuses that
  cache by `conversation_id` across turns.

To combine topical memory with the bounded active-KV engine:

```bash
python scripts/interactive_memory_chat.py \
  --memory-mode topical \
  --generation-engine residual_bounded_kv_direct \
  --hot-budget-mib 150 \
  --session-cache-size 8
```

In this mode, `MemoryChat` builds a `TorchInferenceRuntime` around the live
model, attaches a `ResidualSessionCache`, and passes
`conversation_id=session.session_id` through both plain turns and topical
recall turns. Topical recall still uses the seeded residual carrier slot; when
the runtime engine is `residual_bounded_kv_direct`, that seeded recall path now
runs on top of the bounded active-KV prefill/decode engine.

The new harness check:

- runs one same-session chat for `--turn-latency-turns 50` measured turns,
- uses `--turn-latency-warmup-turns 2`,
- sets deterministic runtime sampling (`temperature=0`, `top_p=1`),
- asserts every measured turn uses the `session_reused` generation path,
- measures turn duration with `time.perf_counter()` to avoid wall-clock skew,
- prompts for longer replies (`--turn-latency-max-new-tokens 48` by default)
  so the timing signal is less dominated by sub-100 ms jitter,
- compares the mean of the first `--turn-latency-window-size 10` measured
  turns with the mean of the last 10, and fails if the trend grows beyond
  `--turn-latency-tolerance 0.20`.

Smoke mode reduces the latency probe to 8 measured turns and a 4-turn trend
window.

## Actual-Use Recall Test

The harness proves routing, injection, budgeting, bounded active-KV latency,
crash safety, and fallback honesty. It has one intentionally small
generated-answer check for topical recall and one for KV-direct recall.

For a stronger "does the model answer from the 100x100 memory archive?" proof,
run the post-harness recall verifier:

```bash
make verify-memory-recall-scale
```

The Make target prefers `.venv/bin/python3` when present, matching the
production harness environment recorded in `environment.json`. If no repo venv
exists, it falls back to `uv run --extra dev`.

That target reuses the latest production-scale `HARNESS_PASS` run under
`prod/validation/repl-autoverify`, samples the 100 planted scale markers from
the run's `events.jsonl`, sends real `MemoryChat.kv_query_turn()` messages, and
fails unless each generated answer contains:

- the queried marker,
- the expected planted session number,
- the expected planted turn number,
- `no_silent_fallback=True`,
- and a matched memory window containing the marker.

Direct invocation:

```bash
uv run --extra dev python scripts/verify_memory_recall_scale.py \
  --run-dir prod/validation/repl-autoverify/<PASS_RUN_DIR> \
  --sample-size 100 \
  --mode kv_direct \
  --required-hit-rate 0.99
```

For a faster confidence check, reduce `--sample-size` to an evenly distributed
subset such as `25`. Use `--sample-size 0` to run every parsed routing probe.
The verifier writes a JSON report named
`scale-actual-recall-<mode>.json` inside the source run directory.

The prompt deliberately does not include the expected session or turn numbers.
It only supplies the retrieval key and output schema, so a pass proves the
answer used retrieved memory rather than prompt leakage.

## Reproducibility Standard

The current bar is:

1. `make verify-infinite-memory` passes at 100 x 100, including
   `INFINITE_TURN_LATENCY`.
2. `make verify-memory-recall-scale` passes against the resulting run.
3. Repeat the pair across 5 consecutive runs before calling the result
   reproducible rather than a single successful production-scale sample.
