# /euclid proof chain — axis-BC coefficient propagation fix (run-4)

> Authored as part of kv-memory-implementation run-4 axis-1 (axis-BC coefficient fix).
> Lead session: `ve-ses-0moed1ikk00008d3de0`.
> Parent (orchestrator) session: `ve-ses-0moeccrpq00009c1d99`.
> Pane: `kv-memory-implementation-lead-axis-1`.
> Branch: `impl/kv-memory-finalize-run-4` (off `main` @ `f6129e2`).

## CLAIM

`vec_inject_to_kv_direct` (at
`src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236`)
multiplies the per-match `(k_vec, v_vec)` returned by
`LocalVecInjectProvider.kv_for_match()` by `float(m.coefficient)` BEFORE the
`(1, n_kv_heads, n_facts, head_dim)` broadcast + `.contiguous()` materialisation,
thereby propagating the upstream tier-aware ranking weight (assigned by ASI Evolve
/ retrieve_sync's cold/warm/hot scoring) all the way through to the page tensor
consumed by `KVDirectGenerator.inject_pre_rope_kv`.

Concretely: for each `m` in `matches_list`, the adapter computes
`coef = float(m.coefficient)`, then appends `_to_cuda_bf16(k_vec) * coef` and
`_to_cuda_bf16(v_vec) * coef` into `k_list` / `v_list` (per
`kv_direct_adapter.py:232-236`). The subsequent `torch.stack` → `unsqueeze` →
`expand(1, n_kv_heads, n_facts, head_dim)` → `.contiguous()` chain preserves
the per-fact scaling, so each `n_facts` slot in the resulting page carries the
coefficient assigned to its originating match. This restores symmetry with the
sibling dense-injection path
(`src/chuk_lazarus/inference/context/research/vec_inject/injection_torch.py:52`,
which applies the coefficient as `h + (coefficient * direction)` after
post-cast residual addition): both call sites now consume `match.coefficient: float`
as the singular ranking-weight conduit.

## SUB-CLAIM (CONFIRMED — no longer FALSIFIED)

In run-3, the axis-BC `/euclid` proof chain explicitly recorded the following
sub-claim as **FALSIFIED**
(see `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-BC.md`,
"SUB-CLAIM (FALSIFIED — see UNKNOWN edges and bug record)" section, lines 18-24):

> "vec_inject_to_kv_direct applies `match.coefficient` to K and V before
> materializing them into the KVDirectMaterialization shape."

Run-3 falsified the sub-claim by code inspection
(`kv_direct_adapter.py:225-231` in run-3 stacked the provider's K/V tensors
unscaled; `m.coefficient` was loaded into `VecInjectMatch` at
`retrieve_sync()` but never propagated downstream to materialisation). The
falsification was filed as supervisor-alert bug `ve-ins-0moe6w4su0000096c6a`,
along with the proposed minimal 4-line diff.

**Run-4 reversal:** this run-4 chain CONFIRMS the run-3 sub-claim. The 4-line
diff carried by `ve-ins-0moe6w4su0000096c6a` is now applied at
`kv_direct_adapter.py:225-236` (per the working tree at run-4 entry; see baseline
record `ve-ins-0moed6kqg00003a3c13`). The new parametric test suite
exercises the propagation path across the run-4 acceptance grid
{0.0, 0.5, 1.0, 2.0, -2.0} and the sub-claim transitions
**FALSIFIED → CONFIRMED**. The asymmetry between dense `vec_inject` injection
and KV-direct injection — flagged in the run-3 "UNKNOWN edges" section — is
hereby resolved.

## PASS test

- **New parametric file:**
  `tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py`
- **Test functions and parametric IDs (12/12 PASS):**
  1. `test_coefficient_zero_produces_all_zero_kv[0]`
  2. `test_coefficient_zero_produces_all_zero_kv[1]`
  3. `test_coefficient_zero_produces_all_zero_kv[2]`
  4. `test_coefficient_one_matches_unscaled_kv`
  5. `test_coefficient_two_matches_ratio_scaling[coefficient_pair0]` — (0.5, 1.0)
  6. `test_coefficient_two_matches_ratio_scaling[coefficient_pair1]` — (1.0, 2.0)
  7. `test_coefficient_two_matches_ratio_scaling[coefficient_pair2]` — (0.5, 2.0)
  8. `test_coefficient_full_matrix[0.0]`
  9. `test_coefficient_full_matrix[0.5]`
  10. `test_coefficient_full_matrix[1.0]`
  11. `test_coefficient_full_matrix[2.0]`
  12. `test_coefficient_negative_two`
- **Run command (verbatim):**
  `uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py -v --tb=short`
- **Regression battery (existing axis-BC suite):**
  `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py` —
  9/9 PASS, no regression. Run command:
  `uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py -v --tb=short`
- **Validator JSONL diagnostic:**
  `prod/validation/diagnostic_axis_BC_coefficient_20260425T132358Z-e035a992.jsonl`
- **Per-test verdict counts:**
  - new parametric: 12 passed, 0 failed, 0 errors, 0 skipped, total 12 (15.06 s wall)
  - existing battery: 9 passed, 0 failed, 0 errors, 0 skipped, total 9 (15.35 s wall)
  - aggregate `regression_status`: **GREEN**
- **Evidence file:line citations:**
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236` —
    propagation operator. The loop body
    `coef = float(m.coefficient); k_list.append(_to_cuda_bf16(k_vec) * coef);
    v_list.append(_to_cuda_bf16(v_vec) * coef)` is the run-4 patch authority's
    implementation.
  - `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:238-249` —
    downstream `torch.stack` + `expand(1, n_kv_heads, n_facts, head_dim)` +
    `.contiguous()`. Preserves per-fact scaling across kv-head broadcast.
  - `src/chuk_lazarus/inference/context/research/vec_inject/_types.py` —
    `VecInjectMatch.coefficient: float` field consumed at the call site.
  - `src/chuk_lazarus/inference/context/research/vec_inject/injection_torch.py:52` —
    sibling dense path; symmetric coefficient application
    (`h + (coefficient * direction)`).
  - `src/chuk_lazarus/inference/context/research/vec_inject/providers/_local_file_torch.py` —
    `kv_for_match()` returns raw pre-RoPE, PRE-L2-norm K/V on cuda bf16
    (no in-provider coefficient scaling — scaling is the adapter's
    responsibility, by design).
  - PROP K.0 guard layer: `target_layer = 29 ∈
    GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`
    (axis-A fixture). AMD 11 honoured — no sliding-window injection.

## FAIL behavior

If the run-4 patch at `kv_direct_adapter.py:225-236` is reverted or weakened, the
new parametric tests fire as follows:

1. **Patch fully reverted** (no scaling applied): every `coefficient != 1.0` case
   fails. `test_coefficient_zero_produces_all_zero_kv[0..2]` fails first via
   `torch.equal(k_page, torch.zeros_like(k_page))` — the page tensor would carry
   the unscaled raw K/V rather than zero. The error message reports the nonzero
   element count.

2. **Single-coefficient lookup (e.g. only the first match's coefficient applied
   to the whole batch):** `test_coefficient_two_matches_ratio_scaling[*]` fails
   on the second fact slot — `actual_k1` would equal `expected_k0 * c1 / c0`
   instead of `expected_k1`. The `torch.allclose(..., atol=1e-2, rtol=1e-2)` margin
   is tight enough to detect ratio errors in distinct underlying raw K/V draws.

3. **`abs()` coercion (drops sign):** `test_coefficient_negative_two` fails on
   the sign-flip assertion
   `torch.sign(actual_k[nonzero_k]) == -torch.sign(raw_k_bf16[nonzero_k])`. The
   magnitude check would still pass, isolating the failure mode to sign loss.

4. **Zero-coefficient escape (e.g. `coef = max(coef, eps)`):**
   `test_coefficient_zero_produces_all_zero_kv[*]` fails the strict
   `torch.equal` assertion (which intentionally rejects "close-to-zero"
   substitutes); `test_coefficient_full_matrix[0.0]` fails the same way.

5. **Sign-flip drop (e.g. clamp `coef >= 0`):** identical to the `abs()` case;
   `test_coefficient_negative_two` fires.

6. **Batch-uniform scaling (e.g. all matches scaled by the mean coefficient):**
   `test_coefficient_two_matches_ratio_scaling[*]` fails on at least one fact
   slot per pair. The mismatch is reported per-kv-head with the offending
   coefficient embedded in the assertion message (e.g.
   `f"fact-0 K mismatch at kv-head={h} for c0={c0}"`).

The `torch.equal` strict-zero assertion at `coefficient=0.0` is the signature
"is the multiply wired at all?" canary — a no-op identity adapter (one that
stacks raw K/V without scaling) cannot pass it under any tolerance relaxation.

## UNKNOWN edges (out of scope of this proof chain)

- **Cross-architecture portability beyond Gemma-4-E2B-it.** The K.0 guard pins
  the global-attention layer set to the axis-A fixture; behaviour on other
  Gemma-family or non-Gemma architectures is not exercised here.

- **fp16 dtype.** All tests run under bf16 (AMD 14, snapshot
  `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`); behaviour under fp16 with its
  narrower exponent range and stricter overflow profile is untested.

- **Coefficient bounds beyond [-2, +2].** The acceptance grid covers
  {0.0, 0.5, 1.0, 2.0, -2.0}. Behaviour at `|coefficient| ≫ 2.0` (potential
  bf16 overflow risk on large raw K/V magnitudes) and at sub-normal magnitudes
  (e.g. 1e-30) is unexercised.

- **Multi-window batches.** `vec_inject_to_kv_direct` raises `ValueError` for
  matches spanning multiple `window_id`s (per `kv_direct_adapter.py:195-202`);
  multi-window coefficient propagation is deferred to axis-E E2E.

- **Coefficient = NaN / Inf.** Currently UNGUARDED — the adapter performs no
  validity check on `float(m.coefficient)` before the multiply. NaN propagates
  silently into the page tensor and can poison downstream attention. **NEXT-MISSION
  recommendation:** add a `torch.isfinite` precondition (or an upstream
  `VecInjectMatch.__post_init__` validator) and parametric tests asserting
  rejection of NaN/Inf coefficients. File as a separate axis when run-4 lead
  surfaces capacity.

- **Coefficient typing edge:** `_make_match` hands `float` literally; behaviour
  if upstream callers ever pass a `np.float32 / torch.Tensor`-shaped coefficient
  is coerced by the explicit `float(m.coefficient)` cast in the adapter, but
  no test exercises that exact coercion path.

## adaptation-status

- **Verdict:** PASS. Sub-claim CONFIRMED (run-3 falsification reversed).
- **Newly-discovered bugs:** none.
- **Regression risk:** LOW. The 4-line patch is additive (adds a per-match
  scalar multiply); the 9/9 existing axis-BC battery passes unchanged. No
  call-site signature change. The patch is consistent with the sibling dense
  injection path's coefficient semantics.
- **Run-4 follow-ups surfaced:** the NaN/Inf guard described under
  "UNKNOWN edges" is a candidate next-mission item, not a blocker.

## Cross-refs

- **Records (vee):**
  - Bug authority: `ve-ins-0moe6w4su0000096c6a`
  - Axis-1 end-state: `ve-ins-0moebmc0b0000f914ac`
  - Mission proposal: `ve-ins-0moebpc3v0000f6afeb`
  - Scope manifest: `ve-ins-0moecnb79000015175a`
  - Baseline-of-absence (lead's run-4 entry record):
    `ve-ins-0moed6kqg00003a3c13`
  - Recipe authority: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
  - Run-3 README/Axis-6 non-overlap resolution:
    `ve-ins-0moecpmiv000006f45e`
- **Files:**
  - Patch site:
    `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236`
  - New test file:
    `tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py`
  - Regression battery:
    `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py`
  - Validator JSONL:
    `prod/validation/diagnostic_axis_BC_coefficient_20260425T132358Z-e035a992.jsonl`
  - Run-3 parent euclid (precedent):
    `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-BC.md`
  - Sibling dense-injection path:
    `src/chuk_lazarus/inference/context/research/vec_inject/injection_torch.py:52`
  - Axis-A global-attention fixture:
    `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
- **AMD invariants honoured:** AMD 11 (sliding-window guard via target_layer=29 ∈
  global set), AMD 14 (CUDA-only RTX 5090 + Gemma-4-E2B-it bf16),
  AMD 8 (`full_attention` ≡ `global_attention` per fixture aliasing),
  AMD 10 (research-dir mirror, no README.md).
