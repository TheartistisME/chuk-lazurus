"""
KL divergence computation for RL training.

Used for:
- DPO implicit reward computation
- PPO KL penalty (optional)
- Monitoring policy drift from reference

Dual-backend: dispatches on tensor type so both MLX and torch tensors work;
produces results within ``atol=1e-5, rtol=1e-4`` across backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chuk_lazarus.training._backend_math import detect_backend, xp_for

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401


def compute_kl_divergence(
    log_probs_p: Any, log_probs_q: Any, mask: Any = None
) -> Any:
    """
    Compute KL divergence: KL(P || Q) = E_P[log P - log Q]

    Args:
        log_probs_p: Log probs under distribution P (current policy)
        log_probs_q: Log probs under distribution Q (reference)
        mask: Optional mask for valid tokens

    Returns:
        kl: Scalar KL divergence (averaged over valid tokens)
    """
    bk = detect_backend(log_probs_p)
    xp = xp_for(bk)
    kl_per_token = log_probs_p - log_probs_q
    if mask is not None:
        kl_per_token = kl_per_token * mask
        total_tokens = xp.sum(mask)
        return xp.sum(kl_per_token) / (total_tokens + 1e-8)
    return xp.mean(kl_per_token)


def compute_approx_kl(
    old_log_probs: Any, new_log_probs: Any, mask: Any = None
) -> Any:
    """
    Compute approximate KL divergence for PPO early stopping.

    Uses: KL ≈ 0.5 * E[(log_ratio)^2]

    Args:
        old_log_probs: Log probs from old policy
        new_log_probs: Log probs from current policy
        mask: Optional mask for valid tokens

    Returns:
        approx_kl: Scalar approximate KL divergence
    """
    bk = detect_backend(old_log_probs)
    xp = xp_for(bk)
    log_ratio = new_log_probs - old_log_probs
    if mask is not None:
        masked_ratio = log_ratio * mask
        total_tokens = xp.sum(mask)
        return 0.5 * xp.sum(masked_ratio ** 2) / (total_tokens + 1e-8)
    return 0.5 * xp.mean(log_ratio ** 2)
