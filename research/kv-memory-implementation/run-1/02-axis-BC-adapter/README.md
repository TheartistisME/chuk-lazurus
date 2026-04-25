# Axis-BC: PROP K.5 KV-Direct Adapter + PROP K.0 Sliding-Window Guard (run 1)

## Summary

Axis-BC closes the PROP K.5 vector-to-KV-direct adapter and the PROP K.0
sliding-window guard for the Gemma-4-E2B-it backend. The adapter materialises
captured `(K, V)` pairs onto exact global-attention layers in the live
KV-direct cache while explicitly refusing sliding-window layers (the recipe
defers any sliding-window remap to axis-G as an OPTIONAL EXTENSION per Q3).
A sibling guard helper exposes the canonical global-attention layer set
`{4, 9, 14, 19, 24, 29, 34}` so callers can validate target layer indices
before invoking the adapter. Provider plumbing was extended with a
`kv_for_match` accessor plus the `_flat_v` and `_flat_index_map` views needed
by downstream retrieval consumers. CUDA-only, bf16, head_dim=256, n_kv_heads=1.

## Adapter API surface

```python
def vec_inject_to_kv_direct(
    cache: KVDirectCache,
    captured: CapturedKV,                      # (K, V) per global layer, bf16, cuda
    target_layers: Sequence[int],              # must be subset of global-attention set
    target_offsets: Optional[Sequence[int]] = None,  # default 0; logging.warning if None
    *,
    n_kv_heads: int = 1,                       # broadcast K/V over heads -> see /euclid claim
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> KVDirectCache: ...

def assert_global_attention_layer(layer_idx: int) -> None:
    """Raises SlidingWindowLayerRefusedError if layer_idx not in
    {4, 9, 14, 19, 24, 29, 34}."""

class SlidingWindowLayerRefusedError(ValueError):
    """Raised by PROP K.0 guard when a sliding-window layer is targeted.
    Auto-remap is deferred to axis-G OPTIONAL EXTENSION (recipe Q3)."""
```

## PROP K.0 guard semantics

The guard REFUSES on any layer outside the global-attention set. There is no
auto-remap to the nearest global layer in run-1 - that behaviour is deferred
to axis-G as an OPTIONAL EXTENSION per recipe question Q3. Callers must
either select a global-attention layer up front or catch
`SlidingWindowLayerRefusedError` and route the injection elsewhere.

## n_kv_heads broadcast choice

Gemma-4-E2B-it materialises a single KV head (`n_kv_heads=1`); captured
tensors are broadcast across query heads at attention time rather than
materialised per-head in cache. /euclid claim: `materialised_kv.shape[H]==1`
so storage is O(L*S*D) not O(L*S*Hq*D), and downstream attention's existing
GQA broadcast reproduces full-head behaviour bit-exactly.

## Default offset and logging

When `target_offsets is None`, the adapter writes at offset 0 for every
target layer and emits a single `logging.warning(...)` documenting the
default. Explicit `target_offsets` suppresses the warning.

## Provider extension

The KV-direct provider gains `kv_for_match(layer, offset, length)` which
returns the materialised slice; `_flat_v` exposes the value tensor as a
flat `(layers*slots, head_dim)` view, and `_flat_index_map` returns the
companion `(layer, offset)` index tuples aligned to that flat view.
Together these power retrieval-side cosine match without per-call gather
in hot paths and preserve the bf16/cuda contract end-to-end.

## Test verdict

- 22/22 axis-BC tests GREEN (global-attention guard + KV-direct adapter)
- 5/5 axis-F regression GREEN with 1 by-design skip on layer 29 in
  `test_kv_materialisation_parity_layers_27_to_32`
- 6/6 KV-direct provider regression GREEN (`kv_for_match`, `_flat_v`,
  `_flat_index_map`)

## Files created/modified

- src/chuk_lazarus/inference/context/research/vec_inject/kv_direct_adapter.py
- tests/inference/backends/test_axis_BC_global_attention_guard.py
- tests/inference/context/research/vec_inject/test_axis_BC_kv_direct_adapter.py

## Missions closed

- chuk-lazurus-vnw
- chuk-lazurus-3y8

## References

- Recipe: ve-ins-0modtwi7v0000ff6d88
- axis-B end-state: ve-ins-0moduw7c30000c9b244
- axis-C end-state: ve-ins-0moduw8p40000564cfa
- Lead session: ve-ses-0modwqdrc0000bfaf68
- Branch: impl/kv-direct-wire-prop-k5
