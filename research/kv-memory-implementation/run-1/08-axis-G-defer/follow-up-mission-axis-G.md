# axis-G OPTIONAL EXTENSION — DEFERRED (mission chuk-lazurus-cr8)

**Defer authority:** supervisor Q3 ACK ve-ins-0modv5m4z00001881b9
**Vee follow-up reference:** ve-ins-0moe2i06j0000a88437
**End-state pointer (axis-G):** ve-ins-0moduwdsv0000e84dea (DEFERRED)
**Cited authorities:** AMD 11 (sliding-window-hazard) + user research-directive ve-ins-0modt1oxl0000b1e823

## One-line summary

axis-G — sliding-window window-offset bookkeeping — is the engineering work explicitly left on the table for run-1 per the user-directive. PROP K.5 + K.0 + K.4.NORM ship on **global-attention** layers per AMD 9 (gap-is-the-star). Sliding-layer KV-direct injection is hard-rejected by the K.0 guard and must remain so until axis-G lands.

## What's missing (what axis-G would deliver)

`window_token_lists.npz` consumes per-window slot indices. On a true sliding-window layer, those indices would need to be remapped through the layer's local-context `window_offset` (causal mask hop) before injection. Without this remap, sliding-layer KV-direct injection silently shifts retrieved K/V by `(effective_position − window_position)`. The K.0 guard sidesteps this by rejecting non-global `target_layer` values.

## Re-pickup conditions for run-2+

(a) Producer-side rebuild of `vec_inject.npz` recording per-window `window_offset` metadata, **OR**
(b) Adapter-side bookkeeping shim deriving `window_offset` from `model.config.sliding_window` + per-window slot position, **AND**
(c) Regression test instrumented at one of the sliding indices `{1, 2, 3, 5, 6, 7, 8, 10..28, 30, 31, 32, 33}` (complement of `{4, 9, 14, 19, 24, 29, 34}` per axis-A enumeration).

## See also

Top-level pointer: `research/kv-memory-implementation/run-1/follow-up-mission-axis-G.md`
