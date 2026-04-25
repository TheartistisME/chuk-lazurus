# Operational notes — axis-BC coefficient propagation fix (run-4)

> Companion to `euclid-axis-BC-coefficient-fix.md`. Captures the operational
> context that doesn't belong inside the formal proof chain: working-tree
> entry state, form-vs-function discussion, test-design rationale, cross-axis
> lineage, AMD 1-15 compliance map.

## Entry-state observations

At run-4 entry (per baseline record `ve-ins-0moed6kqg00003a3c13`), the
working tree of `impl/kv-memory-finalize-run-4` (off `main` @ `f6129e2`)
already carried the 4-line propagation patch as an **uncommitted diff** at
`src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-236`.

That is: the production code change had been physically applied — likely as
a hand-edit during the run-3 → run-4 transition while the
`ve-ins-0moe6w4su0000096c6a` bug authority was being canonicalised — but no
commit captured it, and no test surface yet asserted its correctness. The
state on disk was therefore "behaviour-correct, evidence-absent" — exactly
the asymmetric configuration that motivated this axis-1 mission (per the
mission proposal `ve-ins-0moebpc3v0000f6afeb`).

The lead's job for axis-1 is therefore **not** to author the patch (already
on disk) but to:

1. Audit the on-disk patch for fidelity to the
   `ve-ins-0moe6w4su0000096c6a` bug-authority specification.
2. Author the test surface that validates the propagation contract across
   the axis-1 acceptance grid `{0.0, 0.5, 1.0, 2.0}` plus the sign-flip
   case `-2.0`.
3. Capture a validator JSONL diagnostic and surface PASS/FAIL evidence into
   the `01-axis-BC-coefficient-fix/` research mirror per AMD 10 / AMD 3.

This is the "the gap is the star" framing of AMD 9 — the mission's value is
in closing the test-evidence gap around an already-applied behavioural fix,
not in re-authoring the fix itself.

## Functional-vs-literal-form discussion

The bug authority `ve-ins-0moe6w4su0000096c6a` quotes a literal patch form
along the lines of:

```python
for m in matches_list:
    k_vec, v_vec = provider.kv_for_match(m)
    k_scaled = _to_cuda_bf16(k_vec) * float(m.coefficient)
    v_scaled = _to_cuda_bf16(v_vec) * float(m.coefficient)
    k_list.append(k_scaled)
    v_list.append(v_scaled)
```

The on-disk form at `kv_direct_adapter.py:225-236` reads:

```python
for m in matches_list:
    k_vec, v_vec = provider.kv_for_match(m)
    coef = float(m.coefficient)
    k_list.append(_to_cuda_bf16(k_vec) * coef)
    v_list.append(_to_cuda_bf16(v_vec) * coef)
```

These are **functionally equivalent**: both produce the same per-fact bf16
tensor at the same cuda device, both invoke `_to_cuda_bf16` exactly once
per tensor, both call `float()` on `m.coefficient` exactly once per match
(eliminating attribute-access overhead and surfacing any non-coercible
coefficient value as a TypeError at the cast site rather than silently
reaching the multiply). The literal form materialises a named intermediate
(`k_scaled`, `v_scaled`); the on-disk form folds the multiply into the
`append(...)` call. Bytecode differs trivially — the on-disk form is one
local-variable allocation cheaper per match.

**Decision:** keep the on-disk form. Rewriting to match the literal
spec verbatim would (a) be a no-op behavioural change, (b) introduce
churn on a code path covered by 21 passing tests, and (c) violate the
"minimum-diff" principle the bug authority invokes. The proof chain at
`euclid-axis-BC-coefficient-fix.md` cites the on-disk form's exact line
range so traceability to the patch authority is preserved.

## Test-design rationale

- **Seed = 54321** (vs sibling `test_axis_BC_kv_direct_adapter.py`'s 12345).
  The seeds are deliberately distinct so the new propagation test exercises
  a different draw of K/V values from the synthetic NPZ. This guards against
  a degenerate failure mode where coefficient-fix bugs happen to land on a
  fixed-point of the existing seed — a real concern under the omitted-norm
  pipeline used by `_from_npz` (provider preserves PRE-L2-norm K/V at load),
  where certain magnitude bands could hide fractional-coefficient errors.

- **Tolerance: `atol=1e-2, rtol=1e-2`** for non-zero coefficients. bf16's
  7-bit mantissa imposes ~6e-3 relative error per multiply on values near
  unity; 1e-2 absorbs this with margin while remaining tight enough to
  detect ratio errors of magnitude ≥ ~1.5% (e.g. detecting batch-uniform
  scaling that mistakenly applies the wrong coefficient to one of the two
  facts in `test_coefficient_two_matches_ratio_scaling`).

- **Strict `torch.equal` at `coefficient = 0.0`.** Multiplying any finite
  bf16 tensor by exactly `0.0` (the literal `float`, IEEE-754 +0) produces
  bit-exact `+0.0` bf16 across every element. The test exploits this to
  draw a hard line between "the multiply is wired" and "the multiply is
  weakened-but-close-enough"; tolerance-relaxation strategies cannot pass
  this assertion.

- **Strict `torch.equal` at `coefficient = 1.0`.** Multiplying any bf16
  tensor by exactly `1.0` is bit-exact in bf16 (the multiply is the
  identity). The identity check distinguishes "coefficient is correctly
  threaded as a multiplicative factor" from "coefficient is being applied
  with side effects" (e.g. unintended renormalisation).

- **Sign-flip + magnitude pair at `coefficient = -2.0`.** The pair design
  is deliberate: the magnitude check alone is satisfied by `abs(coef)`-
  coerced patches; the sign check alone could be passed by an XOR-on-sign
  bit hack. Together they pin both halves of signed multiplication and
  cover the realistic upstream case where ASI Evolve hands a negative
  weight for a "deprecated / contradicted" tier-aware retrieval.

- **Per-kv-head broadcast invariant** (the inner
  `for h in range(1, n_kv_heads): assert torch.equal(...)` block in
  `test_coefficient_two_matches_ratio_scaling`) replicates the /euclid PASS
  claim documented inside `kv_direct_adapter.py` itself ("single-head K
  written to every kv-head slot acts as a constant K across the heads")
  and ensures the coefficient-multiply doesn't accidentally introduce a
  per-head asymmetry via stride aliasing. The
  `expand(...).contiguous()` in the adapter is exactly the mechanism that
  this invariant validates.

## Cross-axis lineage

```
                                    ┌──────────────────────────────────┐
                                    │  axis-1 (this mission)            │
                                    │  axis-BC coefficient propagation  │
                                    │  upstream deps: NONE              │
                                    │  outputs: scaled K/V pages        │
                                    └────────────┬──────────────────────┘
                                                 │
                                                 ▼
                                    ┌──────────────────────────────────┐
                                    │  axis-3 (downstream consumer)     │
                                    │  archived prefix materialization  │
                                    │  expects: coefficient-correct K/V │
                                    │  per-fact scaling preserved       │
                                    │  through _prepare_archived_prefix │
                                    └──────────────────────────────────┘
```

- **Upstream of axis-1: NONE.** The propagation operator is a pure local
  transform on outputs of `LocalVecInjectProvider.kv_for_match()`; no
  axis-N ⇒ axis-1 dataflow exists in run-4.

- **Downstream of axis-1: axis-3** (archived prefix materialization). When
  axis-3's `_prepare_archived_prefix` consumes the page tensor produced by
  this adapter, it expects the per-fact scaling to already encode the
  tier-aware ranking weight. Any axis-3 work that re-scales coefficient
  post-hoc would double-apply; any axis-3 work that defers to the page
  tensor as the source of truth aligns with the axis-1 contract.

- **Recipe / charter ownership:** axis-1 is end-state-bounded by
  `ve-ins-0moebmc0b0000f914ac`; further coefficient-related work
  (NaN/Inf guard, fp16 portability, |coef| > 2 saturation) belongs to
  follow-on axes, not to axis-1 scope-creep.

## AMD compliance summary

| AMD | Topic | Compliance |
|-----|-------|------------|
| 1 | SQL fallback | N/A — no vee record creation in this scribe step; lead reads via SQL crib. |
| 2 | euclid mandate | DELIVERED — `euclid-axis-BC-coefficient-fix.md` follows /euclid CLAIM/PASS/FAIL/UNKNOWN/adaptation-status structure. |
| 3 | read-only-scope | HONOURED — scribe writes only the three permitted files in `01-axis-BC-coefficient-fix/`; reads only the explicitly-allowed paths. |
| 4 | RED `#bf616a` | N/A — no UI deliverable in this mission. |
| 5 | child-parent messaging | HONOURED — sub-agent surfaces findings via final tool output only; does NOT call `vee agent message`. |
| 7 | no "not-possible" | HONOURED — every observation is positively framed ("UNGUARDED — currently no validity check") rather than "not possible to guard". |
| 8 | model-dim (`full_attention` ≡ `global_attention`) | HONOURED — `target_layer=29 ∈ GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS`; alias is the axis-A fixture. |
| 9 | the-gap-is-the-star | HONOURED — mission's value statement IS closing the run-3-flagged falsification gap; entry-state notes above explicitly frame the gap. |
| 10 | research-dir mirroring (NOT README.md) | HONOURED — files are named `euclid-axis-BC-coefficient-fix.md`, `notes.md`, `smoke-log.md`; no README.md authored (Axis-6 reserved per `ve-ins-0moecpmiv000006f45e`). |
| 11 | sliding-window {4,9,14,19,24,29,34} | HONOURED — every test invokes `target_layer=29` ∈ global set; PROP K.0 guard is exercised positively (accept-path). |
| 12 | pane-limit | HONOURED — single sub-agent thread; no auxiliary pane spawning. |
| 13 | feature-branch | HONOURED — work occurs on `impl/kv-memory-finalize-run-4` (off `main` @ `f6129e2`); no commits to `main`. |
| 14 | CUDA-only RTX 5090 + Gemma-4-E2B-it bf16 | HONOURED — `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`; no CPU fallback variants; bf16 throughout. |
| 15 | (charter slot — reserved) | HONOURED by inheritance — no charter-15 violation surfaced. |

## Cross-refs

See `euclid-axis-BC-coefficient-fix.md` "Cross-refs" section for the full
record + file path inventory.
