# axis-A §3.7 Step 0 — Gemma-4-E2B-it layer-type enumeration

| Field | Value |
|---|---|
| Mission | `chuk-lazurus-164` |
| Axis | A (Step 0 enumeration) |
| Run | 1 |
| Timestamp (UTC) | `2026-04-25T05:22:33Z` |

## Capture method

Loaded `Gemma-4-E2B-it` on CUDA (RTX 5090, CUDA driver 591.86 / CUDA 13.1)
in `torch.bfloat16` via `transformers.AutoModelForCausalLM.from_pretrained`,
revision pinned to `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`. The
enumeration was read directly from
`model.config.get_text_config().layer_types` (transformers stores the layer
types under the text sub-config for `Gemma4ForConditionalGeneration` VLM
checkpoints; `model.config.layer_types` is the conceptual capture surface).

## Snapshot

- Snapshot id: `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`
- Local cache: `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`

## Enumeration

- `LAYER_COUNT` = 35
- Global-attention literal in this transformers schema: `"full_attention"`
  (semantically equivalent to "global_attention" in the axis-BC taxonomy;
   AMD 11 precondition for PROP K.0)
- Global-attention layer count = 7
- Global-attention layer indices: `[4, 9, 14, 19, 24, 29, 34]`
- Sliding-attention layer count = 28
- Sliding-attention literal: `"sliding_attention"`

The enumeration follows a strict 4-sliding + 1-global repeating block of
length 5; the global layer is always the final element of each block,
giving global indices `4 + 5k` for `k = 0..6`.

## File pointers

| Deliverable | Path |
|---|---|
| D1 (module — frozen constants) | `src/chuk_lazarus/inference/context/knowledge/gemma4_e2b_it_layers.py` |
| D2 (prod/validation JSONL) | `prod/validation/gemma4_e2b_it_layer_types_b4a60110.jsonl` |
| D3 (research-mirror JSONL) | `research/kv-memory-implementation/run-1/01-axis-A-fixture/gemma4_e2b_it_layer_types_b4a60110.jsonl` |
| D4 (CUDA test) | `tests/inference/backends/test_axis_A_gemma4_layers_enumeration.py` |
| D3 supplement (this README) | `research/kv-memory-implementation/run-1/01-axis-A-fixture/README.md` |

## Cross-references

- Recipe: `ve-ins-0modtwi7v0000ff6d88`
- End-state: `ve-ins-0moduw60z00000872b2`
- AMD 11: sliding-window-hazard precondition for PROP K.0 (axis-BC consumes
  the global-attention index set from D1 / D2 / D3 to gate sliding-window
  KV-direct propagation safely).

## Q4 statement

> Module + JSONL agree byte-for-byte on global-attention set per Q4.

`GEMMA4_E2B_IT_GLOBAL_ATTENTION_LAYERS` (D1) ==
`frozenset(global_attention_layer_indices)` (D2) ==
`frozenset(global_attention_layer_indices)` (D3). Verified by D4 test
`test_axis_A_gemma4_layers_enumeration.py`.
