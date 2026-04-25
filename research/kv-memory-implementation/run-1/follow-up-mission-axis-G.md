# Follow-up Mission Reference — axis-G (DEFERRED)

**Mission ID anchor:** chuk-lazurus-cr8
**Vee record:** ve-ins-0moe2i06j0000a88437
**Status:** DEFERRED for run-1 per supervisor Q3 ACK (ve-ins-0modv5m4z00001881b9)
**Authority:** AMD 11 (sliding-window-hazard) + user research-directive ve-ins-0modt1oxl0000b1e823

## Summary

axis-G is the optional sliding-window window-offset bookkeeping extension to PROP K.5 KV-direct injection. The user-directive explicitly named this as engineering work left on the table for run-1.

## Why deferred

The headline run-1 deliverable per AMD 9 (gap-is-the-star) is the PROP K.5 + K.0 + K.4.NORM ship on **global-attention** layers of Gemma-4-E2B-it. Axis-A's §3.7 Step 0 enumeration of `model.config.layer_types` names global-attention indices `{4, 9, 14, 19, 24, 29, 34}` (literal label `full_attention` aliased to `global_attention` per AMD 8). The recipe's empirical claim about parity at L ∈ {27, 28, 30, 31, 32} is materialization-self-consistency under the omitted-norm pipeline — *not* an actual global-attention enumeration. Owner config.py:383-387 Gemma-4-E2B-it preset `injection_layer=29` already matches the canonical global set. Axis-E E.10 PASS-attest (within smoke scope) at L=29.

The PROP K.0 guard in axis-BC's adapter currently rejects any `target_layer` value not in the global-attention set, sidestepping sliding-layer correctness entirely. This is *safe* (the guard never produces a wrong answer) but *incomplete* (sliding layers cannot host KV-direct injection at all).

## What is left on the table (the deferred axis-G work)

- `window_token_lists.npz` consumes per-window slot indices that, on a true sliding-window layer, would need to be remapped through the layer's local-context `window_offset` (causal mask hop) before injection.
- Without this remapping, sliding-layer KV-direct injection silently shifts retrieved K/V by an offset of (effective_position − window_position).
- The K.0 guard currently sidesteps this by hard-rejecting non-global `target_layer` values.

## Re-pickup conditions for run-2+ axis-G

(a) A producer-side rebuild of `vec_inject.npz` that records per-window `window_offset` metadata, **OR**
(b) An adapter-side bookkeeping shim that derives `window_offset` from `model.config.sliding_window` + the per-window position of the slot, **AND**
(c) A regression test similar to axis-F's parity battery but instrumented at one of the sliding indices `{1, 2, 3, 5, 6, 7, 8, 10..28, 30, 31, 32, 33}` — the complement of `{4, 9, 14, 19, 24, 29, 34}` (per axis-A enumeration).

## Rough scope

~2–4 days for one engineer (1 day producer-side metadata; 1 day adapter shim; 1 day regression; 1 day buffer). Risk: **medium** — local-window mask interactions are subtle, and `sliding_window` in Gemma-4-E2B-it is `config.sliding_window=512` per axis-A reading.

## References

- Vee record (this follow-up): ve-ins-0moe2i06j0000a88437
- supervisor Q3 ACK: ve-ins-0modv5m4z00001881b9
- user research-directive: ve-ins-0modt1oxl0000b1e823
- axis-A §3.7 Step 0 enumeration: research/kv-memory-implementation/run-1/01-axis-A-fixture/
- axis-BC adapter (where K.0 guard lives): src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py
