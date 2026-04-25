# /euclid proof chain — axis-rope-phase-fix (per-position RoPE in _prepare_archived_prefix)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

The fix in `_prepare_archived_prefix` (at `src/chuk_lazarus/inference/backends/torch_runtime.py:1907-1919`)
constructs RoPE cos/sin tensors per-position via `torch.arange(N)` rather than a single broadcast
phase, producing a 3D rotary tensor of shape `(B, N, 2 * inv_freq.shape[0])` with strict
per-position dynamism. At position zero the rotary embedding is the identity transform. The
parametrization covers both `sliding_attention` and `full_attention` layer types.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_rope_phase_fix_unit.py`
  (whole-file invocation; 16 parametrized tests)
- run command: `uv run pytest tests/inference/backends/test_axis_rope_phase_fix_unit.py -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_rope_phase_fix_20260425T103103Z-7307dc18.jsonl`
- result: PASS (16/16)
- evidence file:line citations:
  - `src/chuk_lazarus/inference/backends/torch_runtime.py:1907-1919` — per-position cos/sin via
    `torch.arange(N)`; produces 3D rotary tensor `(B, N, 2 * inv_freq.shape[0])`
  - parametric coverage in the test file across `layer_type ∈ {sliding_attention, full_attention}`
    and a sweep of N and B values, totaling 16 parametric cases

## FAIL behavior

The 16 parametric cases assert, in concert:

1. Shape: the rotary tensor has rank 3 with last-dim `2 * inv_freq.shape[0]`. Any rank or last-dim
   mismatch raises `AssertionError`.
2. Per-position dynamism: at least two distinct positions in the same window produce distinct
   cos/sin slices. If the implementation regresses to a single broadcast phase, two positions
   will be byte-equal and the dynamism assertion fires.
3. Position-zero identity: at position 0, the rotary embedding must compose to the identity
   transform on a probe vector (cos=1, sin=0 along the relevant axes). Any non-identity output
   at position 0 raises `AssertionError`.
4. Layer-type parametrization: the same shape and dynamism guarantees hold for both
   `sliding_attention` and `full_attention` layers. A regression that special-cases one layer
   type (e.g. drops per-position phase for sliding) is caught by the parametric sweep.

## UNKNOWN edges (out of scope of this proof chain)

- Numerical equivalence vs HuggingFace's reference RoPE implementation byte-for-byte is NOT
  asserted in this unit test; it is implicitly covered by the axis-D logits-equivalence proof
  chain at the global-attention layers used by KV-direct.
- Long-horizon position drift (positions beyond the test sweep range) is not asserted;
  `inv_freq` precision at extreme positions is a separate concern not in scope here.
- Interaction with KV-sharing producer/consumer routing is asserted by axis-runtime-fix, not
  here; this proof chain is RoPE-construction-correctness only.

## adaptation-status

- run-3 verdict: PASS
- known bugs: none
- regression risk: low — 16 parametric cases lock in shape, dynamism, identity-at-zero, and
  per-layer-type behavior; the fix is localized to a few lines in `_prepare_archived_prefix`.
- next-mission recommendations: none

## Cross-refs

- Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- axis-A fixture: `euclid-axis-A.md`
- axis-runtime-fix: `euclid-axis-runtime-fix.md` (KV-share routing, distinct concern)
- axis-D logits-equivalence: `euclid-axis-D.md` (downstream parity check)
