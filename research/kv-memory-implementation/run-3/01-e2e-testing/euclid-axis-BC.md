# /euclid proof chain — axis-BC (PROP K.5 KV-direct adapter + PROP K.0 sliding-window guard)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

`vec_inject_to_kv_direct` (at `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:151-251`)
converts per-window pre-RoPE K/V pages from `LocalVecInjectProvider.retrieve_sync()` into a
`KVDirectMaterialization` with shape-correct `(pages, sizes, offset)`, and its co-located PROP K.0
guard `assert_global_attention_layer` (`kv_direct_adapter.py:131-141`) hard-rejects any `target_layer`
not in `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}` by raising
`SlidingWindowLayerRefusedError` — preserving the AMD 11 sliding-window-hazard invariant. The
provider's `kv_for_match()` consumes `_flat_v` correctly and returns raw pre-RoPE,
PRE-L2-norm K/V tensors of the expected `(n_facts, head_dim)` shape.

## SUB-CLAIM (FALSIFIED — see "UNKNOWN edges" and bug record)

> "vec_inject_to_kv_direct applies `match.coefficient` to K and V before materializing them into the
> KVDirectMaterialization shape."

This sub-claim is FALSE in run-3. See the `UNKNOWN edges` section below; bug record
`ve-ins-0moe6w4su0000096c6a` carries the full diff and supervisor-alert classification.

## PASS test

- pytest node-id: `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py`
  (whole-file invocation; 9 tests)
- run command: `uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_BC_20260425T103103Z-88ab5373.jsonl`
- result: PASS (9/9)
- evidence file:line citations:
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:131-141` —
    `assert_global_attention_layer`, raises `SlidingWindowLayerRefusedError` for non-global layers
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:151-251` —
    `vec_inject_to_kv_direct`, builds `KVDirectMaterialization(pages, sizes, offset)` from
    `LocalVecInjectProvider` matches
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-231` — K/V are
    stacked from `provider.kv_for_match(m)` returns; `m.coefficient` is **not** referenced
  - `src/chuk_lazarus/inference/context/research/vec_inject/providers/_local_file_torch.py:184-225` —
    `kv_for_match()` is keyed on `(window_id, position)` only; returns raw pre-RoPE K/V; coefficient
    is not threaded
  - `src/chuk_lazarus/inference/context/research/vec_inject/providers/_local_file_torch.py:313-316` —
    coefficient IS loaded into `VecInjectMatch` at `retrieve_sync()` but never propagated downstream
    to materialization
  - `src/chuk_lazarus/inference/context/research/vec_inject/providers/_local_file_torch.py:564` —
    `_from_npz` preserves PRE-L2-norm K/V at load time
  - `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py:247-259` —
    K.0 guard refusal test (asserts `SlidingWindowLayerRefusedError` for layer 30 ∉ global set)

## FAIL behavior

The 9 axis-BC tests assert, in concert:

1. Shape-correctness: any deviation in `pages.shape`, `sizes.shape`, or `offset.shape` from the
   contract documented in `kv_direct_adapter.py:151-251` fires the corresponding `assert` in the
   test body.
2. K.0 guard: passing `target_layer=30` (or any sliding-window layer in {0..34} ∖ {4,9,14,19,24,29,34})
   to `vec_inject_to_kv_direct` is expected to raise `SlidingWindowLayerRefusedError`; the test at
   `test_axis_BC_kv_direct_adapter.py:247-259` checks via `pytest.raises(SlidingWindowLayerRefusedError)`.
   If the guard is bypassed or weakened, the `pytest.raises` context manager exits without the
   expected exception and the test fails with `DID NOT RAISE`.
3. Provider plumbing: `kv_for_match` returning a tensor whose first dim does not equal the number
   of facts, or whose head_dim disagrees with the model's declared head dim, fires shape asserts
   in the materialization step; an `AssertionError` is raised with both shapes printed.

## UNKNOWN edges (out of scope of this proof chain)

- **Coefficient propagation (LOAD-BEARING for run-3 chat-loop integration):** the existing axis-BC
  test suite does NOT assert that `match.coefficient` is applied to K and V before materialization.
  Code inspection confirms it is **not** applied (`kv_direct_adapter.py:225-231` stacks the provider's
  K/V tensors unscaled). This is asymmetric with the dense `vec_inject` injection path at
  `src/chuk_lazarus/inference/context/research/vec_inject/injection_torch.py:52`, which DOES apply
  coefficient as `h + coefficient * direction`. The asymmetry is filed as supervisor-alert bug
  `ve-ins-0moe6w4su0000096c6a` with the proposed minimal diff (multiply both K and V tensors by
  `float(m.coefficient)` after `_to_cuda_bf16`). Patch is **NOT** applied in run-3 — this run-3
  axis-BC proof chain attests existing behavior only.
- Non-global-layer behavior is covered by the sibling K.0 guard test, not by the main adapter
  shape/plumbing tests; full coverage of all non-global indices is parametric-spot-check, not
  exhaustive.
- Cross-window page concatenation order is asserted only for the synthetic fixtures used by the
  test file; live retrieval-time ordering correctness is exercised separately by the e2e-smoke
  proof chain (axis-E).

## adaptation-status

- run-3 verdict: PASS-with-known-bug
- known bugs:
  - `ve-ins-0moe6w4su0000096c6a` (axis-BC ignores `match.coefficient`; supervisor-alert)
- regression risk: medium — coefficient is silently dropped, so any downstream consumer that
  relies on coefficient-weighted retrieval (e.g. confidence-weighted recall in the chat-loop)
  receives unscaled K/V. Existing tests do not catch this because all current synthetic fixtures
  use coefficient = 1.0.
- next-mission recommendations: spawn `axis-BC-coefficient-fix` mission on a separate branch to
  (a) apply the proposed diff in `kv_direct_adapter.py:225-231`, (b) extend
  `test_axis_BC_kv_direct_adapter.py` with a parametric coefficient ∈ {0.5, 1.0, 2.0} test that
  asserts post-materialization K/V scale, and (c) re-run the live tmux REPL chat e2e (axis-7) to
  verify auto-recall behavior under coefficient-weighted retrieval.

## Cross-refs

- Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- axis-A fixture: `euclid-axis-A.md` (canonical global set consumed by K.0 guard)
- Bug: `ve-ins-0moe6w4su0000096c6a` (coefficient drop, supervisor-alert)
- AMD 11: sliding-window hazard invariant
