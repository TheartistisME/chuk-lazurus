"""
Dual-Reward Loss for Classifier Emergence

Combines:
1. Classification loss at intermediate layer (vocab-aligned classifier)
2. Answer loss at final layer (correct outputs)

This loss function trains V/O projections to create vocabulary-mappable
classifiers while maintaining answer quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chuk_lazarus.training._backend_math import detect_backend, xp_for

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401


class DualRewardLossConfig(BaseModel):
    """Configuration for dual-reward loss."""

    model_config = ConfigDict(frozen=True)
    classifier_layer: int = Field(
        default=-1, description="Intermediate layer for classification (-1 means 55% depth)"
    )
    classifier_weight: float = Field(
        default=0.4, ge=0, le=1, description="Weight for classification loss"
    )
    classifier_targets: dict[str, int] = Field(
        default_factory=dict, description="Target tokens for classification (operation -> token_id)"
    )
    use_softmax: bool = Field(
        default=True, description="Whether to use softmax or direct logit for classification"
    )


def dual_reward_loss(
    final_logits: mx.array,
    classifier_logits: mx.array,
    labels: mx.array,
    classifier_labels: mx.array,
    loss_mask: mx.array,
    config: DualRewardLossConfig,
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Compute dual-reward loss combining classification and answer losses.

    Args:
        final_logits: Logits from final layer, shape (batch, seq_len, vocab_size)
        classifier_logits: Logits from classifier layer, shape (batch, seq_len, vocab_size)
        labels: Target answer token ids, shape (batch, seq_len)
        classifier_labels: Target classification token ids, shape (batch,)
        loss_mask: Mask for answer tokens, shape (batch, seq_len)
        config: Loss configuration

    Returns:
        total_loss: Combined loss
        metrics: Dict with individual losses and metrics
    """
    bk = detect_backend(final_logits)
    xp = xp_for(bk)
    batch_size = final_logits.shape[0]
    vocab_size = final_logits.shape[-1]

    logits_flat = final_logits.reshape(-1, vocab_size)
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
    answer_loss = -xp.sum(masked_log_probs) / num_tokens

    cls_logits = classifier_logits[:, -1, :]
    if config.use_softmax:
        cls_probs = xp.softmax(cls_logits, axis=-1)
        cls_log_probs = xp.log(cls_probs + 1e-10)
    else:
        cls_log_probs = xp.log_softmax(cls_logits, axis=-1)

    batch_indices = xp.arange(batch_size)
    cls_labels = classifier_labels
    if bk == "torch":
        import torch

        batch_indices = batch_indices.to(dtype=torch.long, device=cls_labels.device)
        cls_labels = cls_labels.to(dtype=torch.long)
    cls_token_log_probs = cls_log_probs[batch_indices, cls_labels]
    classifier_loss = -xp.mean(cls_token_log_probs)

    cls_weight = config.classifier_weight
    ans_weight = 1.0 - cls_weight
    total_loss = cls_weight * classifier_loss + ans_weight * answer_loss

    answer_perplexity = xp.exp(answer_loss)
    cls_predictions = xp.argmax(cls_logits, axis=-1)
    cls_correct = xp.sum(cls_predictions == cls_labels)
    cls_accuracy = cls_correct / batch_size

    metrics = {
        "loss": total_loss,
        "answer_loss": answer_loss,
        "classifier_loss": classifier_loss,
        "answer_perplexity": answer_perplexity,
        "classifier_accuracy": cls_accuracy,
        "num_tokens": num_tokens,
    }

    return total_loss, metrics


def classification_only_loss(
    classifier_logits: mx.array,
    classifier_labels: mx.array,
) -> tuple[mx.array, dict[str, mx.array]]:
    """
    Compute classification-only loss (for probing/evaluation).

    Args:
        classifier_logits: Logits from classifier layer, shape (batch, vocab_size)
        classifier_labels: Target classification token ids, shape (batch,)

    Returns:
        loss: Classification loss
        metrics: Dict with accuracy, etc.
    """
    bk = detect_backend(classifier_logits)
    xp = xp_for(bk)
    batch_size = classifier_logits.shape[0]

    cls_probs = xp.softmax(classifier_logits, axis=-1)
    cls_log_probs = xp.log(cls_probs + 1e-10)

    batch_indices = xp.arange(batch_size)
    cls_labels = classifier_labels
    if bk == "torch":
        import torch

        batch_indices = batch_indices.to(dtype=torch.long, device=cls_labels.device)
        cls_labels = cls_labels.to(dtype=torch.long)
    cls_token_log_probs = cls_log_probs[batch_indices, cls_labels]
    loss = -xp.mean(cls_token_log_probs)

    predictions = xp.argmax(classifier_logits, axis=-1)
    correct = xp.sum(predictions == cls_labels)
    accuracy = correct / batch_size

    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }

    return loss, metrics
