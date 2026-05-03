# Gemma-4 E2B DFlash Notes

This note records the current Qwen DFlash read-through and the first Gemma-4
E2B porting target. It is a research scaffold, not a trained draft checkpoint.

## What DFlash Does For Qwen

DFlash keeps the target LLM lossless by using the target for verification. The
draft model only proposes a block.

The official Qwen Transformers implementation:

1. Runs the target prefill with `output_hidden_states=True`.
2. Selects target layers with `build_target_layer_ids()`. For Qwen3-8B:
   `num_target_layers=36`, `num_hidden_layers=5`, so the layers are
   `[1, 9, 17, 25, 33]`.
3. Concatenates those hidden states and fuses them with a linear
   `5 * hidden_size -> hidden_size`.
4. Feeds a draft block shaped like `[verified_token, mask, ..., mask]`.
5. In every draft attention layer, projects the fused target context into K/V,
   projects the draft block embeddings into K/V, concatenates them, and runs
   non-causal block attention.
6. Applies the target LM head to the draft hidden states.
7. Verifies the whole proposed block with one causal target forward, accepts the
   longest matching prefix, then uses the target posterior at the first mismatch
   as the bonus token.

Important source anchors:

- Paper: https://arxiv.org/abs/2602.06036
- Official repo: https://github.com/z-lab/dflash
- Qwen layer selection and context fusion: `dflash/model.py`
- Qwen draft/verify loop: `dflash/model.py::dflash_generate`
- Qwen draft-side KV injection: `Qwen3DFlashAttention.forward`
- Qwen config example: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16/blob/main/config.json

## Gemma-4 E2B Calibration

`David/get_model_config.py` was run against the local Gemma-4 E2B-it snapshot:

```text
/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf
```

Topology:

- 35 text decoder layers
- hidden size 1536
- 8 query heads
- 1 KV head
- sliding-window layers except full-attention layers `4,9,14,19,24,29,34`
- layer 13 is sliding attention with `head_dim=256`, K/V out 256, and
  `store_full_length_kv=true`
- layer 14 is full attention with `head_dim=512`, K/V out 512, and
  `store_full_length_kv=true`
- later sliding layers share K/V from producer layer 13
- later full layers share K/V from producer layer 14

The narrow calibration run over layers `13,14,18,19` recommends:

```text
route_layer=13
boundary_layer=13
kv_source_layer=13
kv_target_layer=14
insertion_family=full_attention
confidence=high
```

The generated artifacts were:

- `/mnt/c/Users/jehma/AppData/Local/Temp/gemma4_e2b_dflash_inspect.json`
- `/mnt/c/Users/jehma/AppData/Local/Temp/gemma4_e2b_dflash_calibration_smoke.json`

## First Gemma Draft Plan

The Qwen-uniform DFlash rule for a 35-layer target and five draft layers gives:

```text
[1, 9, 16, 24, 32]
```

That should be kept as an ablation baseline. The first Gemma-specific plan
should include the measured handoff:

```text
target_layer_ids=[4, 9, 13, 14, 29]
block_size=16
draft_layers=5
hidden_size=1536
intermediate_size=6144
num_attention_heads=8
num_key_value_heads=1
preferred full-attention draft head_dim=512
```

`src/chuk_lazarus/inference/speculative/dflash.py` records this as
`build_gemma4_e2b_dflash_plan()`.

## Missing Pieces

- A Gemma-specific trainable DFlash draft module.
- A `mask_token_id` strategy for the Gemma tokenizer and frozen target
  embedding table.
- A target-regenerated Gemma training corpus.
- Offline or online hidden-state caching for the selected target layers.
- Sparse block-diffusion training attention: bidirectional inside each sampled
  block, no cross-block leakage.
- Verifier cache rollback tests on real Gemma shared-K/V caches.

The official repo is MIT licensed, but its README currently says the training
recipe will be open-sourced later. That means the current implementation target
is a compatible scaffold plus our own training pipeline, not a direct checkpoint
conversion.
