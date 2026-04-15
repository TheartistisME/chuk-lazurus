"""
Supervised Fine-Tuning (SFT) Loss.

Dual-backend: dispatches on tensor type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chuk_lazarus.training._backend_math import detect_backend, xp_for

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401


class SFTLossConfig(BaseModel):
    """Configuration for SFT loss computation."""

    model_config = ConfigDict(frozen=True)
    mask_prompt: bool = Field(default=True, description="Mask prompt tokens in loss")
    max_seq_length: int = Field(default=512, ge=1, description="Maximum sequence length")


def sft_loss(
    logits: Any, labels: Any, loss_mask: Any
) -> tuple[Any, dict[str, Any]]:
    """Compute SFT cross-entropy loss."""
    bk = detect_backend(logits)
    xp = xp_for(bk)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)
    log_probs = xp.log(xp.softmax(logits_flat, axis=-1) + 1e-10)
    indices = xp.arange(logits_flat.shape[0])
    if bk == "torch":
        import torch

        indices = indices.to(dtype=torch.long, device=labels_flat.device)
        labels_flat = labels_flat.to(dtype=torch.long)
    token_log_probs = log_probs[indices, labels_flat]
    masked_log_probs = token_log_probs * mask_flat
    num_tokens = xp.sum(mask_flat) + 1e-10
    loss = -xp.sum(masked_log_probs) / num_tokens
    perplexity = xp.exp(loss)
    metrics = {
        "loss": loss,
        "perplexity": perplexity,
        "num_tokens": num_tokens,
    }
    return loss, metrics
