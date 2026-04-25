# kv-memory-implementation run-4 — axis-e2e-verify (axis-5) smoke-log

> Lead session: `ve-ses-0moefvzu00000a2cb66`.
> Branch: `impl/kv-memory-finalize-run-4`.
> Validator agent: **axis-5 validator** (regression sanity + no-regressions verdict).
> Hardware: CUDA RTX 5090, Gemma-4-E2B-it bf16, snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`.
> Date: 2026-04-25.
> AMD 14 compliance: CUDA tests NOT skipped (all tests run on target hardware).

## Mission

Run the consolidated regression battery (axis-1 + axis-2 + axis-3 + axis-4 union) plus optional run-3 broader 35-test battery to confirm run-4 deliverables carry NO REGRESSIONS from their predecessor leads.

## Summary verdict

**AGGREGATE: GREEN** — All predecessor union tests pass (43/43); optional extended battery at 69/70 (1 expected data-schema failure unrelated to code changes).

### Predecessor axis-1+2+3+4 union battery

- **Total: 43/43 PASSED** in 78.94s.
- **Execution:** `uv run pytest tests/inference/context/research/vec_inject/test_axis_BC_coefficient_propagation.py tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py tests/inference/backends/test_axis_WarmPenaltyConfig_contract.py tests/integration/test_chat_save_emit_emit_store.py tests/integration/test_chat_session_route_inject.py -v --tb=short --no-header`

#### Per-file verdicts (axis-1+2+3+4)

| File | Tests | Pass | Fail | Notes |
| --- | --- | --- | --- | --- |
| `test_axis_BC_coefficient_propagation.py` | 12 | 12 | 0 | Parametric zero/one/two/full-matrix coefficient scaling |
| `test_axis_BC_kv_direct_adapter.py` | 9 | 9 | 0 | Adapter shape, broadcast, sliding-layer refusal |
| `test_axis_WarmPenaltyConfig_contract.py` | 5 | 5 | 0 | hot_bonus_value contract + bonus arithmetic |
| `test_chat_save_emit_emit_store.py` | 9 | 9 | 0 | emit_store + dirty-flag + full-prefill smoke |
| `test_chat_session_route_inject.py` | 8 | 8 | 0 | token-budget + AMD 11 invariant + kv_query smoke |
| **TOTAL** | **43** | **43** | **0** | **No failures, no skips** |

### Optional extended run-3 35-test battery (union + axis-A/D/rope-phase-fix)

- **Total: 69/70 PASSED** in 101.60s.
- **Execution:** extended command adding `test_axis_A_gemma4_layers_enumeration.py`, `test_axis_D_logits_equivalence.py`, `test_axis_runtime_fix_kv_consumer_layers.py`, `test_axis_rope_phase_fix_unit.py`, `test_axis_E_kv_direct_e2e_apollo.py`.

#### Per-file verdicts (extended battery)

| File | Tests | Pass | Fail | Notes |
| --- | --- | --- | --- | --- |
| `test_axis_BC_coefficient_propagation.py` | 12 | 12 | 0 | (re-run from above) |
| `test_axis_BC_kv_direct_adapter.py` | 9 | 9 | 0 | (re-run from above) |
| `test_axis_A_gemma4_layers_enumeration.py` | 5 | 5 | 0 | Snapshot/layer-count/layer-types/global-indices byte-for-byte |
| `test_axis_D_logits_equivalence.py` | 1 | 1 | 0 | Omitted input-layernorm logits parity |
| `test_axis_runtime_fix_kv_consumer_layers.py` | 3 | 3 | 0 | Consumer layer routing (L=30, L=34) via kv_shared_layer_index |
| `test_axis_rope_phase_fix_unit.py` | 16 | 16 | 0 | RoPE per-position dynamism (sliding+full, pos-zero identity) |
| `test_axis_E_kv_direct_e2e_apollo.py` | 2 | 1 | 1 | test_kv_direct_synthetic_smoke_e2e_layer_29 PASS; test_kv_direct_path_a_e2e_apollo_fact_recall **XFAIL** (data schema gap) |
| `test_axis_WarmPenaltyConfig_contract.py` | 5 | 5 | 0 | (re-run from above) |
| `test_chat_save_emit_emit_store.py` | 9 | 9 | 0 | (re-run from above) |
| `test_chat_session_route_inject.py` | 8 | 8 | 0 | (re-run from above) |
| **TOTAL** | **70** | **69** | **1** | **1 expected D4 schema-gap (not a code regression)** |

#### Failure detail (axis-E D4 schema-gap)

**Test:** `test_kv_direct_path_a_e2e_apollo_fact_recall`  
**Verdict:** Expected failure (D4 schema-gap, per AMD 7).  
**Reason:** Apollo store lacks vec_inject.npz with v_vecs artifacts. This is a data fixture issue (upstream axis-B/axis-G rebuild required), NOT a run-4 code regression.

```
charter directive: AMD 7 says UNKNOWN != 'not possible' — name the missing piece precisely. 
The missing piece is v_vecs at the target global-attention layer in a vec_inject.npz next to the apollo store.
```

The test correctly names the missing piece and does NOT silently fall back, per AMD 7 directive.

---

## Chat-loop e2e verdict (axis-5 baseline)

**Baseline jsonl:** `prod/validation/diagnostic_e2e_chat_loop_20260425T145119Z-eea1879d.jsonl`  
**Verdict:** 8/8 PASSED (no xfails, no crashes).

| Test | Verdict | Notes |
| --- | --- | --- |
| `test_session_pair_fact_recall` | PASS | Pipeline integrity primary (recall_observed is informational) |
| `test_multi_prompt_battery_three_facts[teal]` | PASS | Soft recall assertion |
| `test_multi_prompt_battery_three_facts[aurora]` | PASS | Soft recall assertion |
| `test_multi_prompt_battery_three_facts[sushi]` | PASS | Soft recall assertion |
| `test_hot_facts_strong_recall` | PASS | HOT bonus boosted via env knob (79 MiB vram delta) |
| `test_warm_facts_partial_recall` | PASS | WARM-tier observability via moderate HOT bonus (79 MiB vram delta) |
| `test_kv_query_no_crash` | PASS | Run-3 kv_query_turn TypeError regression catcher (ve-ins-0moe7elql0000afaa2b) |
| `test_amd_11_layer_29_in_global_attention_set` | PASS | AMD 11 fixture invariant (module-load assertion mirror) |
| **TOTAL** | **8/8 PASS** | **No crashes, no unexpected failures** |

---

## Predecessor lead-report cross-references

| Axis | Lead Report | Record ID |
| --- | --- | --- |
| Axis-1 | Coefficient propagation + adapter shape coverage | `ve-ins-0moedtikd0000fb0bfb` |
| Axis-2 | WarmPenaltyConfig hot_bonus_value contract extension | `ve-ins-0moedugx70000803ac3` |
| Axis-3+4 | Chat-loop integration + token-budget governor + AMD 11 invariant | `ve-ins-0moefr9vp0000f6f74f` |
| Axis-5 baseline | Chat-loop e2e kv_direct mode pipeline integrity | `ve-ins-0moefzb7l0000ebed6a` |

---

## Hardware + environment

- **CUDA RTX 5090** ✓
- **Gemma-4-E2B-it, bf16** ✓
- **HuggingFace snapshot pin:** `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` ✓
- **Test framework:** `uv run pytest` ✓
- **AMD 14 compliance:** CUDA tests NOT skipped on this hardware; all CUDA tests ran. ✓

---

## No-regressions criterion

| Criterion | Status |
| --- | --- |
| Axis-1+2+3+4 union (43 tests): all PASS | ✓ GREEN |
| Axis-5 chat-loop e2e (8 tests): all PASS | ✓ GREEN |
| Run-3 optional extended battery: 69/70 (1 expected schema-gap) | ✓ AMBER (expected) |
| AMD 14 CUDA-only enforcement | ✓ GREEN |
| No new test failures vs predecessors | ✓ GREEN |

**OVERALL: NO REGRESSIONS DETECTED.** Run-4 axis-1+2+3+4 deliverables maintain test parity with their predecessor leads.

---

## Artifact paths

- Smoke-log: `/research/kv-memory-implementation/run-4/05-axis-e2e-verify/smoke-log.md`
- Axis-1+2+3+4 test log: `/tmp/axis5_regression.log` (43/43 PASS)
- Extended battery log: `/tmp/axis5_regression_extended.log` (69/70 PASS)
- Chat-loop e2e baseline: `prod/validation/diagnostic_e2e_chat_loop_20260425T145119Z-eea1879d.jsonl` (8/8 PASS)
