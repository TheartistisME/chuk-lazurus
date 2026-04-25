# Smoke log — axis-BC coefficient propagation fix (run-4)

> Compact verdict ledger. Detailed proof chain in
> `euclid-axis-BC-coefficient-fix.md`; operational context in `notes.md`.

## Test invocation commands (verbatim)

```bash
# New parametric coefficient propagation test (axis-1 deliverable)
uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py -v --tb=short

# Existing axis-BC regression battery (no-regression check)
uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py -v --tb=short
```

## Pytest verdict summary

| Suite | Passed | Failed | Errors | Skipped | Total | Verdict |
|-------|--------|--------|--------|---------|-------|---------|
| `test_axis_BC_coefficient_propagation` (NEW) | 12 | 0 | 0 | 0 | 12 | **PASS** |
| `test_axis_BC_kv_direct_adapter` (regression) | 9 | 0 | 0 | 0 | 9 | **PASS** |
| **Aggregate** | **21** | **0** | **0** | **0** | **21** | **GREEN** |

## JSONL diagnostic path

`prod/validation/diagnostic_axis_BC_coefficient_20260425T132358Z-e035a992.jsonl`

Top-level fields: `axis="axis-BC-coefficient-fix"`, `run=4`,
`lead_session=ve-ses-0moed1ikk00008d3de0`,
`branch=impl/kv-memory-finalize-run-4`,
`bug_authority=ve-ins-0moe6w4su0000096c6a`,
`axis_end_state=ve-ins-0moebmc0b0000f914ac`,
`scope_manifest=ve-ins-0moecnb79000015175a`,
`baseline_record=ve-ins-0moed6kqg00003a3c13`,
`patch_location=src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236`,
`amd_14_honored=true`, `cuda_available=true`,
`regression_status=GREEN`.

## Wallclock timing summary

| Suite | Duration |
|-------|----------|
| `test_axis_BC_coefficient_propagation` (12 tests) | 15.06 s |
| `test_axis_BC_kv_direct_adapter` (9 tests) | 15.35 s |
| **Combined** | **30.41 s** |

Per-test wall budget averages ~1.25 s for the new suite (cuda-warm fixture
load dominates; per-fact algebraic work is sub-millisecond).

## Hardware

- **GPU:** NVIDIA RTX 5090, CUDA available (`torch.cuda.is_available() == True`).
- **Model:** `google/gemma-3n-E2B-it` snapshot
  `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`, dtype bf16.
- **AMD 14 honoured:** CUDA-only execution; module-level
  `pytest.mark.skipif(not torch.cuda.is_available(), ...)` is the only
  skip marker; no CPU fallback paths exist in either suite.
- **AMD 11 honoured:** all tests target `layer=29 ∈
  GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`; the
  PROP K.0 sliding-window guard accepts every call. The regression battery
  separately exercises the refusal path at `target_layer=30` (sliding
  layer) via `test_adapter_refuses_sliding_layer_via_full_call`.

## Test-by-test PASS list

### `test_axis_BC_coefficient_propagation.py` (5 functions, 12 parametric instances)

1. **`test_coefficient_zero_produces_all_zero_kv`** — 3 parametric instances
   (one per `position ∈ {0, 1, 2}`):
   - `test_coefficient_zero_produces_all_zero_kv[0]` — PASS
   - `test_coefficient_zero_produces_all_zero_kv[1]` — PASS
   - `test_coefficient_zero_produces_all_zero_kv[2]` — PASS
   - Asserts strict `torch.equal(k_page, torch.zeros_like(k_page))` and the
     V-page analogue at coefficient=0.0; cold-state propagation canary.

2. **`test_coefficient_one_matches_unscaled_kv`** — 1 instance, PASS.
   - Asserts `torch.equal` of materialised page vs `_to_cuda_bf16(raw) * 1.0`
     across every kv-head slot; identity-multiply correctness.

3. **`test_coefficient_two_matches_ratio_scaling`** — 3 parametric instances
   (one per `coefficient_pair ∈ {(0.5, 1.0), (1.0, 2.0), (0.5, 2.0)}`):
   - `test_coefficient_two_matches_ratio_scaling[coefficient_pair0]` — PASS
     (c0=0.5, c1=1.0)
   - `test_coefficient_two_matches_ratio_scaling[coefficient_pair1]` — PASS
     (c0=1.0, c1=2.0)
   - `test_coefficient_two_matches_ratio_scaling[coefficient_pair2]` — PASS
     (c0=0.5, c1=2.0)
   - Asserts per-match heterogeneous coefficient application and the
     per-kv-head broadcast invariant.

4. **`test_coefficient_full_matrix`** — 4 parametric instances (axis-1
   acceptance grid `{0.0, 0.5, 1.0, 2.0}`):
   - `test_coefficient_full_matrix[0.0]` — PASS (with strict-zero
     `torch.equal` boundary)
   - `test_coefficient_full_matrix[0.5]` — PASS
   - `test_coefficient_full_matrix[1.0]` — PASS
   - `test_coefficient_full_matrix[2.0]` — PASS

5. **`test_coefficient_negative_two`** — 1 instance, PASS.
   - Sign-flip + magnitude assertions at `coefficient=-2.0`; covers the
     "deprecated / contradicted" tier-aware retrieval case.

### `test_axis_BC_kv_direct_adapter.py` (9 regression tests, all PASS)

1. `test_adapter_returns_correct_pages_shape_on_global_layer` — PASS
2. `test_adapter_broadcast_replicates_single_head_across_n_kv_heads` — PASS
3. `test_adapter_raises_on_empty_matches` — PASS
4. `test_adapter_raises_on_multi_window_matches` — PASS
5. `test_adapter_default_offset_zero_when_target_offsets_none` — PASS
6. `test_adapter_raises_when_target_offsets_missing_window` — PASS
7. `test_kv_for_match_helper` — PASS
8. `test_kv_for_match_unknown_position_raises` — PASS
9. `test_adapter_refuses_sliding_layer_via_full_call` — PASS (PROP K.0 guard
   refusal at `target_layer=30`)

## Final ledger

- **Patch location:**
  `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236`
- **Aggregate verdict:** **GREEN** — 21/21 PASS, 0 regressions, AMD 11 + 14
  honoured, sub-claim CONFIRMED (run-3 falsification reversed).
