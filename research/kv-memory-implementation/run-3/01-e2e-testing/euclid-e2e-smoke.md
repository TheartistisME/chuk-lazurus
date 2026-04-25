# /euclid proof chain — axis-E e2e-smoke (synthetic KV-direct materialization at L=29)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

A synthetic K/V tensor pair, fed through `vec_inject_to_kv_direct` to produce a
`KVDirectMaterialization`, then through `generate_with_kv_direct_materialization` at target layer
L=29 (a global-attention layer per the axis-A fixture), produces 8 output tokens on real
Gemma-4-E2B-it under bf16 on CUDA, with `kv_direct_active = true` reported on the generation
metadata. The pipeline does not crash, does not silently fall back to the standard generation
path, and emits a valid token sequence of the expected length.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29`
- run command: `uv run pytest tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29 -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_E_20260425T103103Z-b1b736a8.jsonl`
- result: PASS
- evidence file:line citations:
  - `src/chuk_lazarus/inference/backends/torch_runtime.py` — `generate_with_kv_direct_materialization`
    entry point
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:151-251` —
    `vec_inject_to_kv_direct` (synthetic-input adapter call)
  - axis-A fixture: `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py:73-75`
    (L=29 ∈ global set)
  - axis-runtime-fix anchor: `torch_runtime.py:1873-1920`, `:2004`, `:2367` — the consumer-layer
    routing the smoke test exercises (L=29 is a consumer of producer L=24 under the Gemma-4
    KV-share map)

## FAIL behavior

The smoke test asserts, in order:

1. The `vec_inject_to_kv_direct` call returns without raising; the K.0 guard does not fire (L=29
   is global-attention).
2. `generate_with_kv_direct_materialization` runs to completion and returns 8 output tokens.
3. The generation metadata reports `kv_direct_active = true`. If the runtime silently fell back
   to the standard path, this flag is `false` and the assertion fires.
4. Pre-fix (i.e. without axis-runtime-fix in place), the patched_forward path on L=29 would
   raise `AttributeError: 'Gemma4DecoderLayer' object has no attribute 'k_proj'` (or analogous
   for `v_proj`/`k_norm`/`v_norm`). The FAIL→PASS transition for this canary was the empirical
   evidence anchoring axis-runtime-fix.

## UNKNOWN edges (out of scope of this proof chain)

- **Live retrieval path:** this is a SYNTHETIC-input smoke test. It does not exercise the live
  vec_inject retrieve→materialize→generate path. The live path is covered by the axis-7 chat
  REPL test (filed as AMBER; see `testing-report.md` and bug record
  `ve-ins-0moe7d32a00007113fb`).
- **Coefficient-weighted retrieval:** the synthetic input bypasses `match.coefficient` entirely;
  the coefficient-drop bug in axis-BC (`ve-ins-0moe6w4su0000096c6a`) is therefore NOT exercised
  by this smoke test.
- **Other global-attention layers:** smoke is at L=29 only. Coverage of L ∈ {4, 9, 14, 19, 24, 34}
  is via the axis-F regression battery
  (`tests/inference/backends/test_kv_direct_materialized_real_gemma4.py`), not here.
- **Output-token semantic correctness:** the smoke asserts shape and `kv_direct_active`, not
  semantic correctness of the 8 generated tokens. Logits parity vs the omitted-norm pipeline at
  recipe-empirical L is covered by axis-D.

## adaptation-status

- run-3 verdict: PASS (synthetic smoke)
- known bugs:
  - none in the synthetic path itself
  - the live path (chat REPL, axis-7) is AMBER; see `ve-ins-0moe7d32a00007113fb` and
    `ve-ins-0moe7elql0000afaa2b`
- regression risk: low for the synthetic smoke; medium for the live path, gated by axis-BC
  coefficient propagation and the `/kv_query` `WarmPenaltyConfig` kwarg crash
- next-mission recommendations: a live-retrieval smoke variant that drives the chat REPL
  end-to-end and asserts auto-recall under non-trivial coefficients, contingent on
  `axis-BC-coefficient-fix` and the chat-REPL `/kv_query` wiring fix landing first.

## Cross-refs

- Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- axis-A: `euclid-axis-A.md`
- axis-BC: `euclid-axis-BC.md`
- axis-runtime-fix: `euclid-axis-runtime-fix.md` (smoke is the FAIL→PASS canary)
- axis-rope-phase-fix: `euclid-axis-rope-phase-fix.md`
- axis-D: `euclid-axis-D.md`
- chat-REPL transcript: `research/kv-memory-implementation/run-3/01-e2e-testing/chat-repl-transcript.txt`
- chat-REPL jsonl: `prod/validation/diagnostic_e2e_test_chat_repl_20260425T102328Z-ac274bc6.jsonl`
