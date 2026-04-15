"""
Group Relative Policy Optimization (GRPO) Loss

GRPO is a simpler alternative to PPO that:
- Generates multiple responses per prompt (a "group")
- Computes rewards for each response
- Uses group-relative advantages (no value function needed!)
- Updates policy to prefer higher-reward responses

This is inspired by DeepSeek's approach and is particularly
well-suited for LLM training where:
- Generating multiple samples is easy
- Value estimation for text is hard
- Relative comparisons are more meaningful than absolute values

Key benefits:
- No value function required (simpler architecture)
- Natural exploration through group sampling
- Works well with sparse rewards
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chuk_lazarus.training._backend_math import detect_backend, xp_for

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401


class GRPOConfig(BaseModel):
    """Configuration for GRPO training."""

    model_config = ConfigDict(frozen=True)
    group_size: int = Field(default=4, ge=1, description="Responses per prompt")
    clip_epsilon: float = Field(default=0.2, gt=0, le=1, description="PPO-style clipping")
    kl_coef: float = Field(default=0.1, ge=0, description="KL penalty coefficient")
    entropy_coef: float = Field(default=0.01, ge=0, description="Entropy bonus")
    normalize_advantages: bool = Field(default=True, description="Normalize within group")
    temperature: float = Field(default=1.0, gt=0, description="Sampling temperature")


def grpo_loss(
    log_probs: mx.array,
    ref_log_probs: mx.array,
    rewards: mx.array,
    group_size: int,
    config: GRPOConfig = None,
) -> tuple[mx.array, dict]:
    """
    Compute GRPO loss.

    Args:
        log_probs: Current policy log probs, shape (batch,)
            where batch = num_prompts * group_size
        ref_log_probs: Reference policy log probs, shape (batch,)
        rewards: Rewards for each response, shape (batch,)
        group_size: Number of responses per prompt
        config: GRPO configuration

    Returns:
        loss: GRPO loss
        metrics: Training metrics
    """
    if config is None:
        config = GRPOConfig()

    bk = detect_backend(log_probs)
    xp = xp_for(bk)

    batch_size = log_probs.shape[0]
    num_prompts = batch_size // group_size

    rewards_grouped = rewards.reshape(num_prompts, group_size)
    group_mean = xp.mean(rewards_grouped, axis=1, keepdims=True)
    advantages = rewards_grouped - group_mean

    if config.normalize_advantages:
        group_std = xp.sqrt(xp.var(rewards_grouped, axis=1, keepdims=True) + 1e-8)
        advantages = advantages / group_std

    advantages = advantages.reshape(-1)

    ratio = xp.exp(log_probs - ref_log_probs)
    clipped_ratio = xp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)

    policy_loss_unclipped = -advantages * ratio
    policy_loss_clipped = -advantages * clipped_ratio
    policy_loss = xp.mean(xp.maximum(policy_loss_unclipped, policy_loss_clipped))

    kl_penalty = xp.mean(log_probs - ref_log_probs)

    total_loss = policy_loss + config.kl_coef * kl_penalty

    clip_mask = xp.abs(ratio - 1) > config.clip_epsilon
    if bk == "torch":
        clip_mask_f = clip_mask.to(dtype=xp.float32)
    else:
        clip_mask_f = clip_mask.astype(xp.float32)

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "kl_penalty": kl_penalty,
        "mean_reward": xp.mean(rewards),
        "reward_std": xp.sqrt(xp.var(rewards)),
        "mean_advantage": xp.mean(advantages),
        "clip_fraction": xp.mean(clip_mask_f),
    }

    return total_loss, metrics


def compute_grpo_advantages(rewards: mx.array, group_size: int, normalize: bool = True) -> mx.array:
    """
    Compute group-relative advantages.

    Args:
        rewards: Shape (batch,) where batch = num_prompts * group_size
        group_size: Number of responses per prompt
        normalize: Whether to normalize by group std

    Returns:
        advantages: Shape (batch,)
    """
    bk = detect_backend(rewards)
    xp = xp_for(bk)
    batch_size = rewards.shape[0]
    num_prompts = batch_size // group_size
    rewards_grouped = rewards.reshape(num_prompts, group_size)
    group_mean = xp.mean(rewards_grouped, axis=1, keepdims=True)
    advantages = rewards_grouped - group_mean
    if normalize:
        group_std = xp.sqrt(xp.var(rewards_grouped, axis=1, keepdims=True) + 1e-8)
        advantages = advantages / group_std
    return advantages.reshape(-1)


class GRPOBatch:
    """
    Helper class for organizing GRPO batches.

    Ensures proper grouping of responses for the same prompt.
    """

    def __init__(self, group_size: int):
        self.group_size = group_size
        self.prompts: list[str] = []
        self.responses: list[list[str]] = []  # List of response groups
        self.rewards: list[list[float]] = []

    def add_prompt_group(self, prompt: str, responses: list[str], rewards: list[float]):
        """Add a prompt with its group of responses and rewards."""
        assert len(responses) == self.group_size
        assert len(rewards) == self.group_size

        self.prompts.append(prompt)
        self.responses.append(responses)
        self.rewards.append(rewards)

    def get_flat_rewards(self) -> Any:
        """Get flattened rewards array (MLX by default)."""
        import mlx.core as mx

        flat = []
        for group in self.rewards:
            flat.extend(group)
        return mx.array(flat, dtype=mx.float32)

    def get_all_sequences(self) -> list[str]:
        """Get all prompt+response sequences for tokenization."""
        sequences = []
        for prompt, response_group in zip(self.prompts, self.responses):
            for response in response_group:
                sequences.append(prompt + response)
        return sequences

    def __len__(self) -> int:
        return len(self.prompts)
