# CHANGELOG

## 2026-04-21 — run-1 canonical prefill port (ACHIEVED, 4/4 criteria)

### Delivered

- **New method** `TorchInferenceRuntime.generate_with_residual_prefill_seeded` in `src/chuk_lazarus/inference/backends/torch_runtime.py` (lines 307-472, +193 / -0, additive — old `generate_with_residual` preserved unchanged at line 254).
- **Retriever rewire** in `src/chuk_lazarus/session_retrieval/retriever.py:394` so all three query entry points (exact-id, topical, entity-mention) route through the new method.
- **Demo rewire** in `examples/inference/demo_c_apollo11_torch.py` (verification-only).
- **Reproducible criterion-4 harness** at `scripts/criterion4_e2e_verify.py` (new file, 128 lines; exit 0 = PASS, 1 = FAIL, 2 = ERROR).
- **Production hub** at `prod/` (this directory) — README, RUNBOOK, ARCHITECTURE, VEE_RECORDS, CHANGELOG, canonical excerpts, validation artefacts.

### Verified

| Criterion | Status | Evidence |
|-----------|--------|----------|
| 1. Apollo demo ≥1/3 coherent | PASS (3/3) | `prod/validation/08-apollo-q{1,2,3}-round3.log` |
| 2. AUS3000 strict-asserts no regression | PASS | `prod/validation/06-pytest-round3.log` — 45 passed, 1 skipped, 0 failed; strict 6/6 true on every probe |
| 3. Multi-probe verbatim + coherent | PASS (3/3 direct) | `prod/validation/direct_probe_results.json` |
| 4. Chat_loop → session_close → fresh retrieval | PASS | `prod/validation/CRITERION4_VERDICT.md` + `criterion4_report.json` + `09-criterion4-e2e.log` |

### Architecture

Canonical `chrishayuk/chuk-lazurus` MLX `prefill_to_layer(initial_residual=boundary)` mechanism ported to Gemma-4 PyTorch-HF. See `prod/ARCHITECTURE.md` for the parity spec and the Gemma-4 `inputs_embeds` OOM adaptation (seed-token + `forward_pre_hook` on `layers[0]`, shape-gated for prefill only).

### Also committed in this snapshot (prior-run work)

The commit bundle includes run-A-era / run-B-era untracked work from the prior turn-aligned series that was present in the working tree but not yet on main:

- `src/chuk_lazarus/chat_loop/` (axis-1 conversational entry)
- `src/chuk_lazarus/session_close/` (axis-2 AUS3000 clause emission)
- `src/chuk_lazarus/session_store/` (axis-3 clause-aligned torch store builder)
- `src/chuk_lazarus/session_retrieval/_gemma_patches.py` (ClippableLinear + eos-id patches used at model-load)
- `scripts/cross_session_demo.sh`, `scripts/build_session_store.sh`, `scripts/build_apollo_style_memory_store.py`, `scripts/build_unified_memory_store.py`, `scripts/run_apollo_memory_demo.sh`, `scripts/smoke_gemma_patches.py`
- `tests/chat_loop/`, `tests/session_close/`, `tests/session_store/`, `tests/session_retrieval/`, `tests/cross_session_demo/`
- `docs/turn_aligned/`

Without these, the criterion-4 end-to-end flow can't be reproduced from a fresh clone.

### Known state (flagged, not blocking)

- **`session_generator.py` plant-strategy drift** — working-tree modification flipped from 5x repetition to 1x natural plant. Pre-built `/tmp/csd-multi` checkpoints use 5x; `/tmp/csd-criterion4` was built fresh with 1x. This is intentional but the store / harness combination needs explicit alignment on each use. Next-run item: decide canonical strategy and align.
- **Sporadic 1-of-5 session-build failure** — observed in the criterion-4 fresh build. Root cause lives in `tools/build_clause_aligned_store.py`, a zero-mod primitive. Requires explicit supervisor scope-grant to fix. Next-run item.
- **`multi_probe_query_only.py` UUID non-determinism** — script regenerates `session_id` via `uuid.uuid4().hex` per call, so it never matches on-disk checkpoint UUIDs. Script is present and runnable but misses against pre-built stores. Next-run item: derive `session_id` deterministically from plan topic.
- **vee v0.1.0 CLI drifts** — `vee record show` command missing; `vee session open` returns stale session on reuse. Workarounds in place (event-log grep, tag-based polling). File upstream against `chrishayuk/vee`.
- **vee index backlog** — ~3600 pending embed jobs, oversize records need dead-letter handling at batch boundaries. Local workspace only; does not affect the port.

### Follow-ups (next-run candidates, not in this delivery)

1. **Residual-only recall test** — tighter criterion-4 where the question doesn't echo the stored text into the prompt template, proving recall from residual alone rather than retrieval-pipeline window echo.
2. **`tools/build_clause_aligned_store.py` fix** — root-cause + patch the 1-of-5 sporadic build failure (needs zero-mod primitive scope grant).
3. **`session_generator.py` alignment** — rebuild `/tmp/csd-multi` with 1x strategy or revert the working-tree change.
4. **`multi_probe_query_only.py` UUID determinism** — derive from plan topic deterministically.
5. **vee v0.1.0 upstream patches** — `vee record show` + `vee session open` fixes.
6. **vee index backlog drainage** — batch-size tuning or per-record retry-limit enforcement so oversize records dead-letter instead of cycling.

### Supervisor-side decisions recorded (the "why")

- `ve-ins-0mo8bkkal0000f9c8f3` — authorised in-run scope expansion for criterion-4 verification.
- `ve-ins-0mo89rrpv0000fb4a80` — acknowledged tooling-drift bug-report (non-blocking; LEAD proceed).
- `ve-ins-0mo8b0g6u000059795f` — specific-acknowledgement for LEAD's pre-flight validator probe (Karpathy 'run it first' discipline).
- `ve-ins-0mo8beu8j000096b679` — own-fault acknowledgement + GOAL_BODY shorthand correction (canonical MLX seeds at embedding layer output, not at crystal layer).

Full curated list: `prod/VEE_RECORDS.md`.
