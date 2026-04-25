# PROP K.5 KV-direct wire-up — narrative

This document explains the run-1 deliverable of the
`kv-memory-implementation` charter: wiring `LocalVecInjectProvider.retrieve_sync`
through to `runtime.generate_with_kv_direct_materialization` on
Gemma-4-E2B-it global-attention layers.

## The problem

The `[OWNER_KV_RECIPE_V1]` recipe (vee record `ve-ins-0modtwi7v0000ff6d88`)
specified a PROP K.5 step: convert per-window pre-RoPE K/V pages produced by
the `vec_inject` retriever into the `KVDirectMaterialization` data structure
that the torch backend's `generate_with_kv_direct_materialization` expects.
Two related properties needed to ship together:

- **PROP K.0** — sliding-window guard: only global-attention layers may host
  KV-direct injection in run-1. Sliding-window correctness is deferred.
- **PROP K.4.NORM** — handling of the K-norm / V-norm transforms that
  Gemma-4-E2B-it applies after RoPE: do they need to be inverted on the
  retrieved (un-normed) cached K/V before injection?

Three secondary correctness gaps surfaced during the run:

1. The recipe enumerated parity at L ∈ {27, 28, 30, 31, 32} but did not list
   the canonical global-attention layers; this needed an authoritative
   enumeration before PROP K.0 could ship.
2. Gemma-4-E2B-it implements KV-sharing by *removing*
   `k_proj` / `v_proj` / `k_norm` / `v_norm` from KV-consumer layers
   (29..34). The runtime's `patched_forward` closure assumed every patched
   layer carries those attributes — it raises `AttributeError` on a
   consumer layer.
3. `_prepare_archived_prefix` slices RoPE `cos` / `sin` at position 0 only
   and expands to all archived slots, collapsing to RoPE identity — a
   pre-existing parity gap, surfaced (not caused) by this run.

## The solution

### axis-A — global-attention layer fixture
`src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py`
exports `GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS = {4, 9, 14, 19, 24, 29, 34}`
derived from §3.7 Step 0 enumeration of `model.config.layer_types` on
snapshot `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`. The literal label
`full_attention` is aliased to `global_attention` per AMD 8.

### axis-BC — PROP K.5 adapter + PROP K.0 guard
`src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py`
takes the dict-of-pages output of `LocalVecInjectProvider.retrieve_sync()`
and constructs a `KVDirectMaterialization` with shapes matching what the
runtime expects. Before doing so it consults
`GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS` and raises
`SlidingWindowLayerRefusedError` for any non-global `target_layer`. 22/22
unit tests PASS on CUDA bf16 (commit `b22c561`).

### axis-D — PROP K.4.NORM resolved via PATH-2 (logits-equivalence)
Rather than implementing inverse-norm bookkeeping (PATH-1, which would
require additional state tracking), axis-D demonstrates that under greedy
argmax decoding at temperature 0.0, the omitted-norm pipeline produces
byte-identical generated tokens to the full-norm pipeline. Tensor-level K/V
cosine similarity is ≈ 0.53 (K) / 0.57 (V) and L∞ ≈ 77 / 166 — both
informational. The decision to ship PATH-2 is at vee record
`ve-ins-0modx9azu000033688c`. **Caveat:** the invariance does not generalise
to sampled or beam decoding; documented in the run-debrief.

### axis-runtime-fix — Gemma-4 KV-sharing in patched_forward
`src/chuk_lazarus/inference/backends/_torch_runtime.py` is patched at three
sites (commit `fac1f36`):

1. The `is_shared_follower` predicate broadens to detect
   `not hasattr(module_self, "k_proj")` or
   `getattr(module_self, "is_kv_shared_layer", False)`.
2. The target-stamp block stamps prefix-augmented K/V into
   `shared_kv_states[producer_idx]` for the consumer-target case so
   downstream consumers (HF semantic at `gemma4/modeling_gemma4.py:1208`)
   inherit the prefix.
3. `generate_with_kv_direct_materialization` walks
   `target.kv_shared_layer_index` to the producer module and routes
   `_prepare_archived_prefix`'s `k_norm` / `v_norm` calls to that producer.

## The test

### axis-E — smoke E2E proof-of-integration
`tests/inference/backends/test_axis_E_kv_direct_e2e_apollo.py::test_kv_direct_synthetic_smoke_e2e_layer_29`
exercises the entire wire on a real Gemma-4-E2B-it model with synthetic
minimal-shape K/V (n_facts=2, 2 tokens):

```
synthetic K/V → axis-BC adapter (vec_inject_to_kv_direct)
              → KVDirectMaterialization at injection_layer=29
              → runtime.generate_with_kv_direct_materialization
              → model.generate (CUDA bf16, NVIDIA RTX 5090)
```

Before axis-runtime-fix landed, this test FAILED with
`AttributeError: 'Gemma4TextAttention' object has no attribute 'k_proj'`.
After axis-runtime-fix landed, the test PASSES — this transition is the
load-bearing acceptance signal for run-1 (per supervisor's Framing α
decision at `ve-ins-0moe06p5g00002de88a`).

### axis-F — regression gate
`tests/inference/backends/test_kv_direct_materialized_real_gemma4.py::test_kv_materialisation_parity_layers_27_to_32`
remains GREEN on the post-change worktree (commit `b22c561`): 5 PASS, 1
SKIP (layer 29 SKIP_LINEAGE matches PROP K.4 guard exactly). No timing or
warning regressions vs. the baseline at `ccf5eda`.

## The deferred extension

### axis-G — sliding-window window-offset bookkeeping (DEFERRED)
The PROP K.0 guard sidesteps sliding-layer correctness by hard-rejecting
non-global `target_layer` values. Re-enabling sliding-layer KV-direct
injection requires either:

- a producer-side rebuild of `vec_inject.npz` recording per-window
  `window_offset` metadata, or
- an adapter-side bookkeeping shim deriving `window_offset` from
  `model.config.sliding_window` + the per-window slot position,

plus a regression test instrumented at one of the sliding indices.
Mission anchor `chuk-lazurus-cr8`. See
`research/kv-memory-implementation/run-1/follow-up-mission-axis-G.md`.

## Pointers

- Recipe: `ve-ins-0modtwi7v0000ff6d88` `[OWNER_KV_RECIPE_V1]`
- Run-debrief: `research/kv-memory-implementation/run-1/run-debrief.md`
- Synthesis dir: `research/kv-memory-implementation/run-1/09-axis-H-synthesis/`
- Feature branch: `impl/kv-direct-wire-prop-k5` (merge to main is supervisor-gated)
