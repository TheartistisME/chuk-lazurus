# Follow-up Mission Reference — axis-rope-phase-fix

**Vee record:** ve-ins-0moe2k6qp0000ca2643 (canonical body) — supersedes empty-body sibling ve-ins-0moe2j93f0000ee0a2f
**Status:** OPEN — recommended next mission for run-2

## Summary

A pre-existing parity gap was empirically isolated by lead-axis-runtime-fix during run-1 axis-E smoke FAIL→PASS canary work. The bug is at `src/chuk_lazarus/inference/backends/_torch_runtime.py:1886` in the `_prepare_archived_prefix` function:

```python
cos = cos[:, 0:1, :].expand(...)   # ← position 0 only, expanded to all archived slots
sin = sin[:, 0:1, :].expand(...)
```

This collapses RoPE to **identity** (cos=1, sin=0) at every archived slot, regardless of the slot's true sequence position.

## Why this matters

The parity test `tests/inference/backends/test_kv_direct_materialized_real_gemma4.py::test_kv_materialisation_parity_layers_27_to_32` was authored expecting per-slot RoPE phase `{0..N-1}` (each archived slot rotated to its absolute sequence index). Production code applies position-0-only. lead-axis-runtime-fix verified this is **pre-existing** (NOT a regression introduced by run-1 work) by stash + re-run on a clean checkout against `fac1f36`.

## Scope boundary

Out of scope for runtime-fix mission (chuk-lazurus-3h1 — bounded to Gemma-4 KV-sharing patched_forward fix). Filed forward as a follow-up.

## Prior history

An earlier code-surgeon attempt cited supervisor-authorised RoPE-phase fix at ve-ins-0mod7odhc000067b0a4 — that fix has not landed on `impl/kv-direct-wire-prop-k5` yet.

## Proposed mission (axis-rope-phase-fix)

1. Replace the `cos[:, 0:1, :].expand(...)` pattern with a per-slot phase: `cos = self.rotary_emb(...)[..., position_ids_archived, :]` where `position_ids_archived` is the true absolute-position index of each archived slot.
2. Apply the same change to `sin`.
3. Add a small unit test asserting `cos.shape == (B, 1, N, head_dim)` with `cos[..., n, :] != cos[..., 0, :]` for n>0.
4. Re-run `test_kv_materialisation_parity_layers_27_to_32` — must turn GREEN.
5. Re-run axis-F regression battery; must remain GREEN.

## References

- Vee record (this follow-up; canonical body): ve-ins-0moe2k6qp0000ca2643
- Empty-body sibling (superseded): ve-ins-0moe2j93f0000ee0a2f
- pre-existing-parity-failure observation (lead-axis-runtime-fix): ve-ins-0moe1s9qz0000a1e04a
- runtime-fix lead-report: ve-ins-0moe20pup00000fedf9
- runtime-fix scope-complete closure: ve-ins-0moe27dru000066e528
- prior supervisor-authorised RoPE-phase fix (un-landed): ve-ins-0mod7odhc000067b0a4

## Rough scope

~½–1 day for one engineer (½ day fix + tests, ½ day regression sweep).
