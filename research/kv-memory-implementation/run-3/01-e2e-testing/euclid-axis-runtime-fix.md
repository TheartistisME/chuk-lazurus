# /euclid proof chain — axis-runtime-fix (Gemma-4 KV-share patched_forward routing)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

For Gemma-4-E2B-it consumer layers 29..34 — which strip `k_proj`, `v_proj`, `k_norm`, and
`v_norm` modules and instead reference a producer layer via `kv_shared_layer_index` — the
patched_forward path in `torch_runtime.py` walks the `kv_shared_layer_index` chain to the
producer module for projection calls, and `_prepare_archived_prefix` accesses the producer's
`k_norm`/`v_norm` for archived-prefix construction. Producer-layer behavior at idx 14 (a
non-shared layer) is unaffected.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py`
  (whole-file invocation; 3 tests)
- run command: `uv run pytest tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_runtime_fix_20260425T103103Z-80721d1f.jsonl`
- result: PASS (3/3)
- evidence file:line citations:
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:1873-1920` — `_prepare_archived_prefix`,
    walks `kv_shared_layer_index` to access producer's `k_norm`/`v_norm`
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:2004` — `kv_shared_layer_index` routing
    in patched_forward (consumer→producer projection-module redirect)
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:2367` — producer-module resolution helper
  - The smoke E2E canary at `test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29`
    transitions FAIL→PASS via this fix (covered separately by `euclid-e2e-smoke.md`)

## FAIL behavior

If the consumer-layer routing regresses, three failure modes surface:

1. `AttributeError: 'Gemma4DecoderLayer' object has no attribute 'k_proj'` (or `v_proj`/`k_norm`/
   `v_norm`) when patched_forward is invoked on a consumer layer in {29..34}. The corresponding
   axis-runtime-fix test asserts the patched_forward call returns successfully on layer 29; an
   `AttributeError` propagating up fails the test.
2. Producer-resolution drift: if `kv_shared_layer_index` is misread (off-by-one, or routed to a
   non-producer module), the test asserts that the projected K/V tensors match the producer's
   own forward output bit-for-bit; any mismatch raises `AssertionError`.
3. Producer-layer regression at idx 14: the test exercises a non-shared producer layer and
   asserts unchanged behavior; any divergence (e.g. accidentally routing the producer through
   itself or through a different layer) raises `AssertionError`.

## UNKNOWN edges (out of scope of this proof chain)

- Cross-architecture KV-sharing patterns (other Gemma variants, other model families) are NOT
  covered; this proof chain is Gemma-4-E2B-it-specific.
- Performance regression (e.g. extra dict lookups on hot path) is not asserted; only correctness
  is.
- The interaction with the axis-rope-phase-fix per-position cos/sin construction is asserted
  separately by that proof chain (sliding_attention and full_attention parametric coverage).

## adaptation-status

- run-3 verdict: PASS
- known bugs: none
- regression risk: low — three direct assertions cover the three documented failure modes; the
  fix is localized to two functions in `torch_runtime.py`; the producer-layer regression test at
  idx 14 catches accidental over-rewriting.
- next-mission recommendations: none

## Cross-refs

- Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- axis-A fixture: `euclid-axis-A.md` (consumer layers 29, 34 are members of the global set)
- axis-E e2e-smoke: `euclid-e2e-smoke.md` (FAIL→PASS canary at layer 29)
