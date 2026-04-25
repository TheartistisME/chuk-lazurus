# Follow-up Mission Reference — apollo-demo data product extension

**Vee record:** ve-ins-0moe2ijrd000095213a
**Status:** OPEN — non-bead, separate workstream from kv-memory implementation

## Summary

The axis-E end-to-end test against the apollo-demo dataset materially passed the **adapter integration** (axis-BC's `kv_direct_adapter.py` mechanically receives `retrieve_sync` output from `LocalVecInjectProvider`), but DEFERRED the full **runtime fact-recall** assertion because the apollo-demo data product on disk does not contain a `vec_inject.npz` built for Gemma-4-E2B-it. Supervisor decision Framing B (ve-ins-0modyzsr80000077a1d) accepted the ADAPTER-PASS + RUNTIME-DEFER framing.

## Where the data lives

- **Source-of-truth:** `/tmp/research_apollo_demo/` (entries.npz + window_tokens.npz + window_token_lists.npz + boundaries/)
- **SHA-verified mirror per Q5/AMD 15:** `research/kv-memory-implementation/run-1/04-axis-E-apollo-data-copy/`
- **Manifest:** `research/kv-memory-implementation/run-1/04-axis-E-apollo-data-copy-manifest.json`
- **MISSING file:** `vec_inject.npz` with per-window `'w{N}/k_vecs'` + `'w{N}/v_vecs'` (the producer `prefill_torch.py:625` generates this per model)

## Steps to flip the canary D4 (currently FAIL → PASS)

D4 is the runtime-fact-recall canary that today raises *"No vec-inject index found"* during test_axis_E_kv_direct_e2e_apollo.py.

1. Run: `lazarus context prefill --phases vec_inject --backend torch` against snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` at `injection_layer=29` (Gemma-4-E2B-it canonical global-attention layer per axis-A §3.7 Step 0).
2. Save the resulting `vec_inject.npz` alongside `entries.npz` in `research_apollo_demo/`.
3. Re-run `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py` with fact-recall assertion enabled.
4. Confirm: `model.generate()` with KV-direct injection produces tokens consistent with the apollo facts.

## References

- Vee record (this follow-up): ve-ins-0moe2ijrd000095213a
- lead-axis-E lead-report (initial FAIL): ve-ins-0modywiua0000e18e37
- supervisor decision Framing B: ve-ins-0modyzsr80000077a1d
- lead-axis-E lead-report (AMENDED ADAPTER-PASS + RUNTIME-FAIL): ve-ins-0moe09xiq00003ea9c8
- apollo-data-copy manifest: research/kv-memory-implementation/run-1/04-axis-E-apollo-data-copy-manifest.json
- axis-E e2e diagnostics: research/kv-memory-implementation/run-1/05-axis-E-e2e/

## Rough scope

~1–2 days (1 hr to run prefill; 1 day to extend test_axis_E_kv_direct_e2e_apollo.py with fact-recall assertion + golden answers; ½ day to package as a data-product release). NON-bead in chuk-lazurus repo — this is data-product-extension territory.
