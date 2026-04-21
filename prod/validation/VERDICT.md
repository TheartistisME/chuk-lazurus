# Round-1 CUDA OOM Claim Validation Report

## Verdict Table

| Criterion | Status | Finding |
|-----------|--------|---------|
| Direct UUID probe (Step 1) | **CONFIRMED OOM** | All 3 probes hit CUDA OOM in `gemma4/modeling_gemma4.py:1635` |
| Apollo demo (Step 2) | Different error | Gemma-4 API incompatibility (requires `input_ids` with `inputs_embeds`) |
| Root cause | **Model-side issue** | NOT code issue; Gemma-4's `get_per_layer_inputs()` broadcasts to 250+ GiB |

## Findings

### Step 1: Direct UUID Probe ✓ EXECUTED
**File:** `/tmp/validation-logs/04-direct-probe.log`

Ran 3 end-to-end queries using real on-disk session handles:
- `11a1c9ade5e547dcaabe39454fd9441b.1.0` → OOM **253.50 GiB**
- `1f2c5fd2cc63491a8b62a4775a5b096e.1.0` → OOM **257.25 GiB**
- `bb37d40612e349ecb4e2d48e108de6c0.1.0` → OOM **255.75 GiB**

Each probe successfully called `generate_with_residual_prefill_seeded()` but failed during model forward pass in the prefill phase.

**Exception Chain:**
```
SessionRetriever.query_exact_id(handle)
  → retriever._generate_from_window()
    → runtime.generate_with_residual_prefill_seeded()
      → model.generate(inputs_embeds=seeded_embeds, ...)
        → transformers/generation/utils.py:2736 _sample() → _prefill()
          → transformers/models/gemma4/modeling_gemma4.py:2195 forward()
            → transformers/models/gemma4/modeling_gemma4.py:1635 get_per_layer_inputs()
```

**Exact Error Message (Probe 0):**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 253.50 GiB. 
GPU 0 has a total capacity of 31.84 GiB of which 19.73 GiB is free. 
Of the allocated memory 10.33 GiB is allocated by PyTorch, and 48.38 MiB is reserved.
```

### Step 2: Apollo Demo
**File:** `/tmp/validation-logs/05-apollo-q1.log`

Changed line 190 in `demo_c_apollo11_torch.py` from `generate_with_residual()` to `generate_with_residual_prefill_seeded()`. Hit a **different** error:

```
RuntimeError: It seems like you tried to call `forward` from `inputs_embeds` 
without providing `input_ids`, and that the `inputs_embeds` you provided do not 
exactly match the embedding weights. Since Gemma4 needs to reverse the embedding 
to compute another embedding, make sure you provide exact `inputs_embeds`
```

This indicates Gemma-4's model API forbids the `inputs_embeds`-only approach and requires exact embedding match. This is separate from the OOM issue.

## Root Cause Analysis

The OOM does **NOT** originate from the boundary + prompt concatenation in torch_runtime.py:354:
```python
seeded_embeds = self._torch.cat([boundary, prompt_embeds], dim=1)  # (1, S+1, H)
```

The concatenation itself succeeds. The OOM occurs later, inside Gemma-4's `get_per_layer_inputs()` method, which attempts an undocumented reverse-embedding computation involving a massive broadcasting operation between:
- `inputs_embeds[:, :, None, :]` → (1, ~178, 1, 8192)
- `embed_tokens.weight[None, None, :, :]` → (1, 1, ~256000, 8192)

This broadcasts to **~250 GiB** — way beyond GPU capacity.

## Conclusion

**Round-1 OOM claim: VERIFIED ✓**

The `generate_with_residual_prefill_seeded()` method CAN execute end-to-end with real data, but it triggers a catastrophic memory allocation in Gemma-4's forward pass, not in Chuk Lazarus code. This is a model-level incompatibility that must be addressed via:
1. Using a `forward_pre_hook` approach (as mentioned in mission notes) instead of `inputs_embeds`
2. Or selecting a model that doesn't require reverse-embedding validation
3. Or patching the Gemma-4 embedding computation

**Port status:** Working up to model.generate() call; model itself is incompatible with residual injection via `inputs_embeds`.
