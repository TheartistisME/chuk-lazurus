"""Gemma-4 generation-time patches ported verbatim from TradeGuru.

Source: /mnt/c/users/jehma/desktop/tradeguru/training-data/e2b-finetune/
        chat_with_trained_model.py (lines 437-465)

Both functions are required for greedy decoding on google/gemma-4-E2B-it
to produce coherent output instead of pathological repetition (e.g.
"France is France is France is..."). See bug-report
ve-ins-0mo846s9m0000fb829b for the diagnosis trail.
"""

from __future__ import annotations

from typing import Any


def patch_clippable_linear(model: Any) -> int:
    """Replace Gemma-4's ``ClippableLinear`` wrappers with their raw linear.

    Gemma-4 contains custom ``ClippableLinear`` modules whose forward path
    can produce degenerate logits under residual injection or extended
    greedy decoding. The fix is to swap each wrapper with its underlying
    ``.linear`` submodule before any ``model.generate(...)`` call.

    Returns the number of modules patched (0 means the model didn't have
    any wrappers, which is fine for non-Gemma-4 models).
    """
    patched = 0
    for name, module in list(model.named_modules()):
        if "ClippableLinear" not in type(module).__name__:
            continue
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, module.linear)
        patched += 1
    return patched


def generation_eos_ids(tokenizer: Any) -> list[int] | int | None:
    """Return the multi-ID EOS list Gemma-4 needs to halt cleanly.

    ``tokenizer.eos_token_id`` alone is insufficient — Gemma-4-it has
    additional stop tokens (``<turn|>``, ``<end_of_turn>``) that must
    also be passed to ``model.generate(..., eos_token_id=...)`` or the
    model runs past its natural stops into garbage territory.

    Returns:
      - None if no EOS tokens are resolvable (caller should fall back).
      - int if exactly one valid EOS id was found.
      - list[int] otherwise.
    """
    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for token in ("<turn|>",):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
            and token_id not in eos_ids
        ):
            eos_ids.append(token_id)
    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


__all__ = ["patch_clippable_linear", "generation_eos_ids"]
