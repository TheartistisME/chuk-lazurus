# kv-memory-implementation run-3 — e2e-testing comprehensive report

> Lead session: `ve-ses-0moe6fapv000010c07f`.
> Branch: `impl/e2e-testing-run-3` (forked from `main` @ `49a9db2`).
> Recipe authority: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`.
> Hardware: CUDA RTX 5090, Gemma-4-E2B-it bf16, snapshot
> `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`.
> Manifest: `ve-ins-0moe6eumd00003dc3c6`.

## Summary

Run-3 e2e-testing exercised six independent component proof chains plus a live tmux REPL chat
end-to-end probe. The unit-and-component layer is **GREEN**: all six per-component proof chains
(axis-A, axis-BC, axis-D, axis-runtime-fix, axis-rope-phase-fix, axis-E e2e-smoke) report PASS,
and a consolidated regression battery of 35 tests passes 35/35 in 76.95s on the run-3 branch.

The live tmux REPL chat e2e probe (axis-7) is **AMBER**. Cold-start, fact-assertion, `/save`,
and `/new` stages all pass; however the `/save` flow does NOT emit `vec_inject.npz` (or the
adjacent `k_vecs` / `v_vecs` / `entries.npz` artifacts), the auto-recall query path silently
falls back to no-candidates with a STRICT topical-routing WARN, and the manual `/kv_query`
diagnostic command crashes with a `TypeError` on a kwarg mismatch between
`scripts/interactive_memory_chat.py:998` and the `WarmPenaltyConfig` dataclass at
`src/chuk_lazarus/inference/backends/torch_runtime.py:2910`.

Three canonical records were filed (two supervisor-alert classification): the axis-BC adapter
silently drops `match.coefficient` from materialization (`ve-ins-0moe6w4su0000096c6a`), the
chat REPL `/kv_query` command crashes on an unknown kwarg
(`ve-ins-0moe7elql0000afaa2b`), and a follow-up combining the chat-loop auto-recall gap with
the `/kv_query` crash (`ve-ins-0moe7d32a00007113fb`). The run-3 branch carries **NO** code
changes for these — all fixes are deferred to follow-up missions on separate branches under
supervisor authorisation.

## Mission and topology

- run-3 spec: kv-memory-implementation, run-3, axis e2e-testing.
- Topology: lead-only (single Euclid-style proof-chain authoring lead; explore agent feeding
  evidence dossier).
- Supervisor-gated: merge to `main` requires explicit supervisor authorisation.
- Mode: autonomous (per MODE OVERRIDE PROTOCOL); HALT-records written and proceed; final
  status-supersede deferred to supervisor.
- Session: `ve-ses-0moe6fapv000010c07f`.
- Branch: `impl/e2e-testing-run-3`.

## End-state acceptance vs delivery

End-state record: `ve-ins-0moe6o9qe0000b98de9`. The six end-state acceptance criteria and
their delivery status:

- **AC-1: axis-A fixture proven against pinned snapshot** — PASS
  (`euclid-axis-A.md`; jsonl `diagnostic_e2e_test_axis_A_20260425T103103Z-b7972334.jsonl`).
- **AC-2: axis-BC adapter shape + K.0 guard proven** — PASS-with-known-bug
  (`euclid-axis-BC.md`; coefficient drop filed as `ve-ins-0moe6w4su0000096c6a`; jsonl
  `diagnostic_e2e_test_axis_BC_20260425T103103Z-88ab5373.jsonl`).
- **AC-3: axis-D logits-equivalence (recipe-correct vs omitted-norm) proven** — PASS
  (`euclid-axis-D.md`; jsonl `diagnostic_e2e_test_axis_D_20260425T103103Z-01c74d91.jsonl`).
- **AC-4: axis-runtime-fix consumer-layer routing proven** — PASS
  (`euclid-axis-runtime-fix.md`; jsonl
  `diagnostic_e2e_test_axis_runtime_fix_20260425T103103Z-80721d1f.jsonl`).
- **AC-5: axis-rope-phase-fix per-position rotary proven** — PASS
  (`euclid-axis-rope-phase-fix.md`; jsonl
  `diagnostic_e2e_test_axis_rope_phase_fix_20260425T103103Z-7307dc18.jsonl`).
- **AC-6: axis-E synthetic smoke at L=29 on real Gemma-4-E2B-it CUDA proven** — PASS
  (`euclid-e2e-smoke.md`; jsonl `diagnostic_e2e_test_axis_E_20260425T103103Z-b1b736a8.jsonl`).

Auxiliary live-path probe (not an AC): axis-7 chat REPL — AMBER (see dedicated section).

## Per-component /euclid proof chains

| Component | File | Tests | Verdict | Notes |
| --- | --- | --- | --- | --- |
| axis-A (global-attention enumeration) | `euclid-axis-A.md` | 1/1 | PASS | Frozen literal vs live model.config.layer_types; pinned snapshot |
| axis-BC (KV-direct adapter + K.0 guard) | `euclid-axis-BC.md` | 9/9 | PASS-with-known-bug | Coefficient drop filed; existing suite does not cover coefficient propagation |
| axis-D (K.4.NORM logits-equivalence) | `euclid-axis-D.md` | 1/1 | PASS | Greedy text and token-id byte-equality |
| axis-runtime-fix (KV-share routing) | `euclid-axis-runtime-fix.md` | 3/3 | PASS | Consumer 29..34 routes through `kv_shared_layer_index`; producer L=14 unaffected |
| axis-rope-phase-fix (per-position RoPE) | `euclid-axis-rope-phase-fix.md` | 16/16 | PASS | sliding+full attention; pos-zero identity; per-position dynamism |
| axis-E e2e-smoke (synthetic L=29) | `euclid-e2e-smoke.md` | 1/1 | PASS | `kv_direct_active=true`; 8 tokens emitted; FAIL→PASS canary for axis-runtime-fix |

## Live tmux REPL chat e2e test (axis-7)

**Verdict: AMBER (with NEW supervisor-alert bug).**

### Stage-by-stage

| Stage | Verdict | Notes |
| --- | --- | --- |
| Cold-start chat | PASS | model loaded on `cuda` in 44.7s |
| Fact-assert ("My favorite color is teal") | PASS | assistant acknowledged "teal" as a plain turn |
| `/save` | PASS | `torch_store/`, `save-state.json`, AUS3000 input clauses written |
| `/save` → `vec_inject.npz` emission | FAIL | NO `vec_inject.npz` / `k_vecs` / `v_vecs` / `entries.npz` produced |
| `/new` (fresh session) | PASS | retriever rebuilt against saved checkpoint |
| Recall query "What is my favorite color?" | FAIL | STRICT topical routing produced no candidates; SILENT FALLBACK WARN; assistant disclaims knowledge |
| Manual `/kv_query` attempt | FAIL (CRASH) | `TypeError: WarmPenaltyConfig.__init__() got an unexpected keyword argument 'hot_bonus_value'` — REPL crashed |

### Wire-up gap (precise file:line)

- caller: `scripts/interactive_memory_chat.py:998` — passes
  `WarmPenaltyConfig(hot_bonus_value=hot_bonus_value)`
- callee: `src/chuk_lazarus/inference/backends/torch_runtime.py:2910` — `WarmPenaltyConfig`
  defines only `{penalty_value, per_warm_uniform, clamp_min}` — there is **no** `hot_bonus_value`
  field

The caller and callee diverged at some prior commit. The crash is deterministic and reproducible.

### Artifacts

- transcript: `research/kv-memory-implementation/run-3/01-e2e-testing/chat-repl-transcript.txt`
- jsonl: `prod/validation/diagnostic_e2e_test_chat_repl_20260425T102328Z-ac274bc6.jsonl`
- preserved workspace: `/tmp/run3_chat_workspace_20260425T102328Z`

### Recommendations (for follow-up missions)

1. Restore `vec_inject.npz` emission on `/save`. Until this is fixed, the chat-REPL recall path
   has no source-of-truth K/V to materialize from.
2. Resolve the `WarmPenaltyConfig` kwarg drift either by removing `hot_bonus_value=` from the
   caller or by adding the field to the dataclass — supervisor decides intent.
3. Investigate the STRICT topical-routing fallback semantics: silent WARN + zero-candidates is
   a poor UX; either the routing should attempt a relaxed match or the WARN should be promoted
   to an actionable error visible to the user.
4. Re-run the axis-7 chat REPL probe AFTER the axis-BC coefficient-fix lands, to validate that
   coefficient-weighted retrieval surfaces the "teal" memory under non-trivial scoring.

## Regression sanity battery

A single consolidated invocation across the six component test files reports **35/35 PASS** in
**76.95s** on the run-3 branch. The composite jsonl is
`prod/validation/diagnostic_e2e_test_regression_battery_20260425T103103Z-945787f6.jsonl`. No
flakes, no skips, no warnings beyond the standard `kv_direct_active=true` narrative log lines.

## Key findings filed as canonical records

- **`ve-ins-0moe6w4su0000096c6a`** — axis-BC adapter ignores `match.coefficient` when stacking
  K/V into `KVDirectMaterialization`. Asymmetric vs the dense `vec_inject` injection path
  (`injection_torch.py:52`) which DOES apply coefficient as `h + coefficient * direction`.
  Supervisor-alert classification. Proposed minimal diff included in the record; **NOT** applied
  in run-3.
- **`ve-ins-0moe7elql0000afaa2b`** — `/kv_query` REPL command crashes with `TypeError` on
  `WarmPenaltyConfig` `hot_bonus_value` kwarg drift between
  `scripts/interactive_memory_chat.py:998` and
  `src/chuk_lazarus/inference/backends/torch_runtime.py:2910`. Supervisor-alert classification.
- **`ve-ins-0moe7d32a00007113fb`** — follow-up record combining the chat-loop auto-recall gap
  (no `vec_inject.npz` emitted on `/save`; STRICT topical-routing produces zero candidates with
  silent fallback WARN) with the `/kv_query` crash. Follow-up-mission-requested.

## Recommendations and follow-up missions (supervisor-decides)

1. **`axis-BC-coefficient-fix` mission** on a separate branch:
   (a) apply the proposed diff in
   `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py:225-231` so K/V
   tensors are scaled by `float(m.coefficient)` after `_to_cuda_bf16`;
   (b) extend `tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py`
   with a parametric coefficient ∈ {0.5, 1.0, 2.0} test asserting post-materialization K/V
   scale;
   (c) re-run the axis-7 chat REPL probe to verify auto-recall under coefficient-weighted
   retrieval.

2. **`chat-repl-kv-query-fix` mission** on a separate branch: resolve the `WarmPenaltyConfig`
   kwarg drift (supervisor decides whether to add `hot_bonus_value` to the dataclass or remove
   it from the caller); add a unit test exercising `/kv_query` end-to-end so future drift is
   caught.

3. **`chat-repl-save-emits-vec-inject` mission** on a separate branch: restore `vec_inject.npz`
   (and `k_vecs` / `v_vecs` / `entries.npz`) emission on the `/save` flow; the chat-loop recall
   path is currently un-exercisable without these artifacts.

4. **`chat-repl-strict-topical-routing-ux` mission**: promote the silent fallback WARN to either
   a relaxed-match retry or an actionable user-visible error.

5. **(Out-of-scope, FYI)** `axis-G` remains deferred (`chuk-lazurus-cr8`); re-enabling
   sliding-window injection is gated by that work.

## Hardware + environment

- CUDA RTX 5090.
- Gemma-4-E2B-it, bf16.
- HuggingFace snapshot pin: `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`.
- `pytest` invocation via `uv run pytest`.
- All run-3 tests are CUDA-only by design; per AMD 14, this report avoids "hardware-absence"
  framing — on this hardware CUDA is the target. Narrow gaps not covered by run-3 (e.g. a
  hypothetical mlx-fixture-capture parity) are explicitly out of scope.

## Cross-refs

### Canonical records
- LEAD comprehension: `ve-ins-0moe6nutt000006ea15`
- End-state: `ve-ins-0moe6o9qe0000b98de9`
- BUG axis-BC ignores `match.coefficient` (supervisor-alert): `ve-ins-0moe6w4su0000096c6a`
- BUG `/kv_query` `WarmPenaltyConfig` kwarg crash (supervisor-alert): `ve-ins-0moe7elql0000afaa2b`
- Follow-up — chat REPL auto-recall gap + `/kv_query` crash combined: `ve-ins-0moe7d32a00007113fb`
- Manifest: `ve-ins-0moe6eumd00003dc3c6`
- Recipe authority: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`

### /euclid proof chains
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-A.md`
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-BC.md`
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-D.md`
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-runtime-fix.md`
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-axis-rope-phase-fix.md`
- `research/kv-memory-implementation/run-3/01-e2e-testing/euclid-e2e-smoke.md`

### JSONL diagnostic artifacts
- `prod/validation/diagnostic_e2e_test_axis_A_20260425T103103Z-b7972334.jsonl`
- `prod/validation/diagnostic_e2e_test_axis_BC_20260425T103103Z-88ab5373.jsonl`
- `prod/validation/diagnostic_e2e_test_axis_D_20260425T103103Z-01c74d91.jsonl`
- `prod/validation/diagnostic_e2e_test_axis_runtime_fix_20260425T103103Z-80721d1f.jsonl`
- `prod/validation/diagnostic_e2e_test_axis_rope_phase_fix_20260425T103103Z-7307dc18.jsonl`
- `prod/validation/diagnostic_e2e_test_axis_E_20260425T103103Z-b1b736a8.jsonl`
- `prod/validation/diagnostic_e2e_test_regression_battery_20260425T103103Z-945787f6.jsonl`
- `prod/validation/diagnostic_e2e_test_chat_repl_20260425T102328Z-ac274bc6.jsonl`

### Other artifacts
- chat-REPL transcript: `research/kv-memory-implementation/run-3/01-e2e-testing/chat-repl-transcript.txt`
- preserved chat workspace: `/tmp/run3_chat_workspace_20260425T102328Z`

### AMD references
- AMD 8 — `full_attention` ↔ `global_attention` alias governance
- AMD 11 — sliding-window-hazard invariant (enforced by K.0 guard)
- AMD 14 — no "hardware-absence" framing on CUDA targets
