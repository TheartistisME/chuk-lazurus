# /euclid proof chain — axis-D (K.4.NORM logits-equivalence)

> Authored as part of kv-memory-implementation run-3 axis e2e-testing.
> Lead session: ve-ses-0moe6fapv000010c07f.
> Branch: impl/e2e-testing-run-3.

## CLAIM

The recipe-correct pre-norm K/V projection pipeline (raw `k_proj`/`v_proj` followed by
`k_norm`/`v_norm`, as encoded by `[OWNER_KV_RECIPE_V1]`) produces byte-equal greedy text and
byte-equal token-id sequences vs the omitted-norm pipeline currently exercised by KV-direct
materialization, with `kv_direct_active = True` on both paths. In other words: under the
omitted-norm pipeline used by the KV-direct path, the resulting logits at the global-attention
layers of interest are equivalent to those that would be produced if the recipe-correct pre-norm
pipeline were threaded through, which is the empirical claim recorded in the recipe authority for
parity at L ∈ {27, 28, 30, 31, 32}.

## PASS test

- pytest node-id: `tests/inference/backends/test_axis_D_logits_equivalence.py::test_input_layernorm_omission_logits_equivalence`
- run command: `uv run pytest tests/inference/backends/test_axis_D_logits_equivalence.py::test_input_layernorm_omission_logits_equivalence -v --tb=short`
- run-3 jsonl: `prod/validation/diagnostic_e2e_test_axis_D_20260425T103103Z-01c74d91.jsonl`
- result: PASS (1/1)
- evidence file:line citations:
  - `src/chuk_lazarus/inference/backends/_torch_residual_bounded.py:759-761` — raw K/V projection
    (the omitted-norm-pipeline anchor)
  - `src/chuk_lazarus/inference/context/research/vec_inject/gemma_adapter.py:48-49` — recipe-correct
    pre-norm K/V projection (the recipe-anchored anchor)
  - `tests/inference/backends/test_axis_D_logits_equivalence.py` — equivalence assertion: greedy
    decode is byte-equal across both paths AND the token-id sequence is byte-equal
  - Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]` — empirical
    parity claim at L ∈ {27, 28, 30, 31, 32} (materialization-self-consistency under omitted-norm)

## FAIL behavior

If the omitted-norm pipeline diverges from the recipe-correct pre-norm pipeline at any global
attention layer of interest, the equivalence assertion fires:

- text-equality assertion: any non-empty diff between the two greedy-decoded strings raises an
  `AssertionError` printing both decodings.
- token-id-equality assertion: any non-empty diff in the token-id tensors raises an
  `AssertionError` printing the tensors and the first divergent index.
- `kv_direct_active` flag must be `True` on both paths; if either path is silently falling back
  to the standard (non-KV-direct) path, the flag assertion fires.

## UNKNOWN edges (out of scope of this proof chain)

- Cross-snapshot equivalence: the test pins to the axis-A snapshot
  `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`; equivalence under other snapshots is not asserted.
- Non-greedy decoding (sampling, beam search): the parity assertion is greedy-only.
- Long-horizon equivalence beyond the smoke prompt length used in the test is not asserted; the
  recipe authority's empirical claim is layer-local and does not imply unbounded-length equivalence.

## adaptation-status

- run-3 verdict: PASS
- known bugs: none
- regression risk: low — the test pins to a fixed snapshot and a fixed prompt; behavior is
  deterministic and the assertion is byte-equality. Risk surface is limited to silent changes in
  either projection path.
- next-mission recommendations: none

## Cross-refs

- Recipe authority: vee record `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- axis-A fixture: `euclid-axis-A.md`
