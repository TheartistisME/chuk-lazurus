"""Bare-model smoke test for the Gemma-4 patches.

Reproduces the bug-report's diagnostic: greedy-decode 'The capital of France is'.
Before patches: 'France is France is France is...' (token salad).
After patches:  expected to produce coherent text containing 'Paris'.
"""

from __future__ import annotations

import sys
from chuk_lazarus.session_retrieval._gemma_patches import (
    patch_clippable_linear,
    generation_eos_ids,
)


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "google/gemma-4-E2B-it"
    print(f"[smoke] loading {model_id}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"[smoke] model on {next(model.parameters()).device}")

    # Apply the patches
    patched = patch_clippable_linear(model)
    eos = generation_eos_ids(tokenizer)
    print(f"[smoke] patch_clippable_linear: {patched} modules patched")
    print(f"[smoke] generation_eos_ids: {eos!r}")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    print(f"[smoke] generating with greedy decode (do_sample=False)…")
    out = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    result = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"[smoke] OUTPUT: {result!r}")

    # PASS if output mentions "Paris" or doesn't contain repetitive "France is France is"
    if "Paris" in result:
        print("[smoke] PASS: output contains 'Paris'")
        return 0
    if "France is France is" in result:
        print("[smoke] FAIL: still in pathological repetition loop")
        return 1
    print("[smoke] AMBIGUOUS: no 'Paris' but no 'France is France is' either")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
