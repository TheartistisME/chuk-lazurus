# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) — entries are
grouped under [Unreleased] until the next tagged release, then promoted under a
dated version heading.

## [Unreleased]

### Added — kv-memory-implementation run 1 (PROP K.5 wire-up)

**Title:** `feat(kv-memory): wire LocalVecInjectProvider.retrieve_sync → axis-BC adapter → KVDirectMaterialization (PROP K.5) on Gemma-4-E2B-it global layers`

**Recipe:** [OWNER_KV_RECIPE_V1] (vee record `ve-ins-0modtwi7v0000ff6d88`)
**Feature branch:** `impl/kv-direct-wire-prop-k5` (merge to `main` is supervisor-gated)

This run closes the PROP K.5 / PROP K.0 / PROP K.4.NORM ship for
Gemma-4-E2B-it global-attention layers. The work is the multi-axis charter
known internally as `kv-memory-implementation` run 1.

#### Headline deliverables

- **PROP K.5 — KV-direct adapter** (`axis-BC`, commit `b22c561`)
  Adds `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py`,
  which converts per-window pre-RoPE K/V pages produced by
  `LocalVecInjectProvider.retrieve_sync()` into the
  `KVDirectMaterialization` shape consumed by
  `runtime.generate_with_kv_direct_materialization`. 22/22 unit tests PASS on
  CUDA bf16. Mathematical correctness on synthetic data is established.

- **PROP K.0 — sliding-window guard** (`axis-BC`, commit `b22c561`)
  Adds a hard guard in the adapter that rejects any `target_layer` not in the
  Gemma-4-E2B-it global-attention layer set
  `{4, 9, 14, 19, 24, 29, 34}` (consumed from the axis-A fixture below). The
  guard preserves correctness; sliding-window injection is intentionally
  out-of-scope for run-1 (see `axis-G` deferred follow-up).

- **PROP K.4.NORM — PATH-2 logits-equivalence** (`axis-D`)
  Resolves the K-norm / V-norm omission concern via PATH-2 (logits-equivalence
  proof): under greedy argmax decoding at temperature 0.0, the omitted-norm
  pipeline is byte-identical at the token level to the full-norm pipeline,
  despite tensor-level cosine ≈ 0.53 (K) / 0.57 (V) and L∞ ≈ 77 / 166. See
  decision pointer `ve-ins-0modx9azu000033688c`. Caveat: invariance does not
  generalise to sampled / beam decoding.

- **axis-A — global-attention layer enumeration**
  Adds `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
  exposing `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`,
  derived from §3.7 Step 0 enumeration of `model.config.layer_types` on the
  pinned snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`. The literal label
  `full_attention` is aliased to `global_attention` per AMD 8 of the run-1
  amendment table.

- **axis-runtime-fix — Gemma-4 KV-sharing patched_forward fix** (commit `fac1f36`)
  Fixes an `AttributeError` on KV-consumer layers (29..34) inside
  `runtime.generate_with_kv_direct_materialization`'s `patched_forward`
  closure. Gemma-4-E2B-it strips `k_proj` / `v_proj` / `k_norm` / `v_norm`
  from KV-consumer layers; the fix routes consumer-target requests to the
  producer module's projections via `kv_shared_layer_index` and stamps the
  prefix-augmented K/V into `shared_kv_states[producer_idx]` so downstream
  consumers inherit the prefix. The smoke E2E canary
  `test_kv_direct_synthetic_smoke_e2e_layer_29` transitions FAIL → PASS as a
  result.

- **axis-E — end-to-end smoke E2E proof-of-integration**
  Adds `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py`
  including `test_kv_direct_synthetic_smoke_e2e_layer_29`. With the
  axis-runtime-fix landed, this test exercises the complete wire on a real
  Gemma-4-E2B-it model: synthetic K/V → adapter → KVDirectMaterialization →
  `model.generate`. Apollo-fact-recall remains a follow-up workstream pending
  a `vec_inject.npz` rebuild for the demo dataset (see follow-up below).

- **axis-F — regression gate** (PASS)
  Re-runs `test_kv_materialisation_parity_layers_27_to_32` on the post-change
  worktree; 5/5 PASS, 1 SKIP (layer 29 SKIP_LINEAGE matches PROP K.4 guard
  exactly). No timing or warning regressions vs. the GREEN baseline at
  `ccf5eda`.

#### Architecture entry points (new / updated)

| File | Status | Purpose |
|---|---|---|
| `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py` | NEW | global-attention layer fixture |
| `src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py` | NEW | PROP K.5 adapter + PROP K.0 guard |
| `src/chuk_lazarus/inference/backends/_torch_runtime.py` | MODIFIED | Gemma-4 KV-sharing fix in `patched_forward` (consumer-target stamp + `_prepare_archived_prefix` producer-module routing) |
| `tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py` | NEW | smoke E2E on real Gemma-4-E2B-it |
| `tests/inference/backends/test_axis_runtime_fix_kv_consumer_layers.py` | NEW | regression coverage for KV-consumer-layer routing |

#### Follow-ups filed (NON-bead in this run; tracked in `research/kv-memory-implementation/run-1/`)

1. **axis-G — sliding-window window-offset bookkeeping** (DEFERRED)
   Mission anchor `chuk-lazurus-cr8`. Vee reference `ve-ins-0moe2i06j0000a88437`.
   See `research/kv-memory-implementation/run-1/follow-up-mission-axis-G.md`.

2. **apollo-demo data product extension**
   Rebuild `vec_inject.npz` for Gemma-4-E2B-it to flip the schema-gap canary
   (`No vec-inject index found`) to PASS on full apollo fact-recall.
   Vee reference `ve-ins-0moe2ijrd000095213a`.
   See `research/kv-memory-implementation/run-1/follow-up-mission-apollo-data-extension.md`.

3. **vee CLI patch** (`P1` flag plumbing + `P2/P3` concurrent session_id)
   Owner team `vee-maintainer`. Triggered by run-1 session_id collisions.
   Vee reference `ve-ins-0moe2j2ym00006ece06`.
   See `research/kv-memory-implementation/run-1/follow-up-mission-vee-cli-patch.md`.

4. **axis-rope-phase-fix** — `_prepare_archived_prefix` RoPE-identity bug
   at `_torch_runtime.py:1886`. Pre-existing parity gap surfaced (but not
   caused) by run-1. Vee reference `ve-ins-0moe2k6qp0000ca2643`.
   See `research/kv-memory-implementation/run-1/follow-up-mission-axis-rope-phase-fix.md`.

#### Run notes

- Recipe-correction note: the canonical global-attention set is the axis-A
  enumeration `{4, 9, 14, 19, 24, 29, 34}`. The recipe's empirical claim about
  parity at L ∈ {27, 28, 30, 31, 32} is materialization-self-consistency
  under the omitted-norm pipeline, NOT an actual global-attention enumeration.
  Owner config `config.py:383-387` Gemma-4-E2B-it preset `injection_layer=29`
  matches the canonical set; axis-E E2E selected layer 29 accordingly.
- All work is on CUDA bf16 against pinned snapshot
  `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` (NVIDIA RTX 5090).
- See `research/kv-memory-implementation/run-1/run-debrief.md` for the
  comprehensive multi-axis debrief.
