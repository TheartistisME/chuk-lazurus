# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## kv-memory implementation (PROP K.5 KV-direct wire-up)

The `impl/kv-direct-wire-prop-k5` feature branch carries the multi-axis
charter known as `kv-memory-implementation` run 1. The merge of this branch
to `main` is **supervisor-gated** — do not merge it without explicit
supervisor authorisation.

### Entry points to know

- **PROP K.5 adapter (axis-BC):**
  `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py`
  Converts per-window pre-RoPE K/V pages from
  `LocalVecInjectProvider.retrieve_sync()` into the `KVDirectMaterialization`
  shape consumed by `runtime.generate_with_kv_direct_materialization`.

- **PROP K.0 guard (axis-BC):**
  Lives inside the same `kv_direct_adapter.py`. Hard-rejects any
  `target_layer` not in the global-attention set; raises
  `SlidingWindowLayerRefusedError` to enforce the AMD 11
  sliding-window-hazard invariant.

- **Global-attention layer fixture (axis-A):**
  `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
  Exposes `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`
  derived from §3.7 Step 0 enumeration of `model.config.layer_types` on
  pinned snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`. The literal
  label `full_attention` is aliased to `global_attention` per AMD 8.

- **Gemma-4 KV-sharing handling in patched_forward (axis-runtime-fix):**
  `src/chuk_lazarus/inference/backends/_torch_runtime.py`
  Gemma-4-E2B-it strips `k_proj` / `v_proj` / `k_norm` / `v_norm` from
  KV-consumer layers (29..34); the runtime walks
  `kv_shared_layer_index` to the producer module for projection calls and
  for `_prepare_archived_prefix`'s `k_norm` / `v_norm` access. The smoke E2E
  canary (`test_kv_direct_synthetic_smoke_e2e_layer_29`) transitions
  FAIL → PASS via this fix.

### Test surface

- `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py` — smoke E2E
  on real Gemma-4-E2B-it; CUDA-gated (no CPU fallback).
- `tests/inference/backends/test_kv_direct_materialized_real_gemma4.py` —
  axis-F regression battery (parity at layers 27..32).
- `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py` —
  consumer-layer routing coverage.

### Run-1 deliverables and follow-ups

- Comprehensive run-1 debrief:
  `research/kv-memory-implementation/run-1/run-debrief.md`
- Wire-up narrative: `docs/kv-memory-prop-k5-wire-up.md`
- Synthesis dir: `research/kv-memory-implementation/run-1/09-axis-H-synthesis/`
- Follow-up references (NON-bead, tracked outside the bead system):
  - `research/kv-memory-implementation/run-1/follow-up-mission-axis-G.md`
  - `research/kv-memory-implementation/run-1/follow-up-mission-apollo-data-extension.md`
  - `research/kv-memory-implementation/run-1/follow-up-mission-vee-cli-patch.md`
  - `research/kv-memory-implementation/run-1/follow-up-mission-axis-rope-phase-fix.md`

### Operational rules

- **Hardware:** kv-memory tests are CUDA-only. Do not introduce CPU-run
  variants of these tests.
- **Branch hygiene:** all run-1 commits land on `impl/kv-direct-wire-prop-k5`.
  The merge to `main` is supervisor-gated.
- **Recipe authority:** `[OWNER_KV_RECIPE_V1]` is vee record
  `ve-ins-0modtwi7v0000ff6d88`. The recipe's empirical claim about parity at
  L ∈ {27, 28, 30, 31, 32} is materialization-self-consistency under the
  omitted-norm pipeline, NOT an actual global-attention enumeration. The
  canonical global set is the axis-A fixture above.
- **Sliding-window injection** is intentionally rejected by the K.0 guard
  for run-1. Re-enabling it requires the deferred axis-G work
  (chuk-lazurus-cr8).
