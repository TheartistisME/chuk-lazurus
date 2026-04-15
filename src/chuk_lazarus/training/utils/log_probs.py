"""
Log probability extraction for RL training.

Dual-backend: operations dispatch on tensor type so MLX and torch tensors
both work; outputs match within ``atol=1e-5, rtol=1e-4``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chuk_lazarus.training._backend_math import detect_backend, xp_for

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401


def compute_log_probs_from_logits(logits: Any, actions: Any) -> Any:
    """
    Compute log probabilities of ``actions`` given ``logits``.

    Args:
        logits: Shape (batch, seq_len, vocab_size)
        actions: Shape (batch, seq_len)

    Returns:
        log_probs: Shape (batch, seq_len)
    """
    bk = detect_backend(logits)
    xp = xp_for(bk)
    log_probs_all = xp.log(xp.softmax(logits, axis=-1) + 1e-10)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = log_probs_all.reshape(-1, vocab_size)
    actions_flat = actions.reshape(-1)
    indices = xp.arange(batch_size * seq_len)
    if bk == "torch":
        import torch

        indices = indices.to(dtype=torch.long, device=actions_flat.device)
        actions_flat = actions_flat.to(dtype=torch.long)
    log_probs = logits_flat[indices, actions_flat]
    return log_probs.reshape(batch_size, seq_len)


def extract_log_probs(
    model: Any, input_ids: Any, attention_mask: Any = None
) -> tuple[Any, Any]:
    """Run a forward pass and return (shifted log probs, shifted logits)."""
    output = model(input_ids, cache=None)
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output
    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = compute_log_probs_from_logits(shifted_logits, targets)
    if attention_mask is not None:
        shifted_mask = attention_mask[:, 1:]
        log_probs = log_probs * shifted_mask
    return log_probs, shifted_logits


def compute_sequence_log_prob(log_probs: Any, attention_mask: Any = None) -> Any:
    """Sum log probs across sequence to get total sequence log probability."""
    bk = detect_backend(log_probs)
    xp = xp_for(bk)
    if attention_mask is not None:
        return xp.sum(log_probs * attention_mask, axis=-1)
    return xp.sum(log_probs, axis=-1)
