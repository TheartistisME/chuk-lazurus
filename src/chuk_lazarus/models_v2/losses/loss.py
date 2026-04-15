"""
Loss functions for language model training.

Top-level ``import mlx.*`` is intentionally absent; imports are deferred
inside each function.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

logger = logging.getLogger(__name__)


def compute_lm_loss(
    model: Any,
    input_ids,
    labels,
    attention_mask=None,
):
    """
    Compute language modeling loss with optional attention mask.

    Returns:
        Tuple of (loss, num_tokens)
    """
    import mlx.core as mx
    import mlx.nn as nn

    try:
        output = model(input_ids, labels=labels)

        if output.loss is not None:
            loss = output.loss
            if attention_mask is not None:
                ntoks = mx.maximum(attention_mask.sum(), 1)
            else:
                ntoks = labels.size
            return loss, int(ntoks)

        logits = output.logits

        ce = nn.losses.cross_entropy(logits, labels)

        if attention_mask is not None:
            ce = ce * attention_mask
            ntoks = mx.maximum(attention_mask.sum(), 1)
        else:
            ntoks = mx.array(labels.size)

        loss = ce.sum() / ntoks

        return loss, int(ntoks)

    except Exception as e:
        logger.error(f"Error during loss computation: {e}")
        return mx.array(0.0), 1


def compute_cross_entropy_loss(
    logits,
    labels,
    attention_mask=None,
    label_smoothing: float = 0.0,
):
    """
    Compute cross-entropy loss from logits.

    Returns:
        Tuple of (loss, num_tokens)
    """
    import mlx.core as mx
    import mlx.nn as nn

    ce = nn.losses.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    if attention_mask is not None:
        ce = ce * attention_mask
        ntoks = mx.maximum(attention_mask.sum(), 1)
    else:
        ntoks = mx.array(labels.size)

    loss = ce.sum() / ntoks

    return loss, int(ntoks)
