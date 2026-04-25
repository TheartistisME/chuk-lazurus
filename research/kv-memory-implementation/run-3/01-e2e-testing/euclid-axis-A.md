# /euclid proof chain — axis-A (Gemma-4-E2B-it global-attention layer enumeration)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

The fixture `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = frozenset({4, 9, 14, 19, 24, 29, 34})` defined at
`src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py:73-75` is the canonical, byte-equal
enumeration of layer indices whose `model.config.layer_types` is `full_attention` (aliased to
`global_attention` per AMD 8) on pinned HuggingFace snapshot
`b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`, with `LAYER_COUNT = 35`. The diagnostic JSONL produced by the
axis-A test matches the module's serialized form byte-for-byte.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_A_gemma4_layers_enumeration.py::test_global_attention_indices_match`
- run command: `uv run pytest tests/inference/backends/test_axis_A_gemma4_layers_enumeration.py::test_global_attention_indices_match -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_A_20260425T103103Z-b7972334.jsonl`
- result: PASS (1/1)
- evidence file:line citations:
  - `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py:73-75` — fixture definition
    (`GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS`, `LAYER_COUNT`, snapshot pin)
  - AMD 8 — `full_attention` ↔ `global_attention` alias governance
  - Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]` (recipe is
    materialization-self-consistent at the omitted-norm layers; the canonical global set is THIS axis-A
    fixture, not the recipe's empirical L-set).

## FAIL behavior

If the fixture were silently mutated, or if the loaded snapshot drifted from
`b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`, the equality assertion in
`test_global_attention_indices_match` fires comparing the live-enumerated set from
`model.config.layer_types` against the module fixture. Any divergence (added/removed indices, alias
miscount, snapshot drift) raises an `AssertionError` with both sets printed, and the run-3 jsonl record
is annotated with `verdict=FAIL`.

## UNKNOWN edges (out of scope of this proof chain)

- Cross-snapshot drift (other Gemma-4 variants, e.g. 9B/27B) is NOT covered; the fixture is pinned to
  E2B-it `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`.
- The K.0 sliding-window-refusal guard that *consumes* this fixture is proved by axis-BC, not here.
- Whether the empirical recipe L-set (L ∈ {27, 28, 30, 31, 32}) coincides with the fixture is
  intentionally NOT tested — those are materialization-self-consistency layers under the omitted-norm
  pipeline, not the canonical global-attention enumeration.

## adaptation-status

- run-3 verdict: PASS
- known bugs: none
- regression risk: none — fixture is a frozen literal, snapshot is pinned, the test is direct equality
- next-mission recommendations: none
