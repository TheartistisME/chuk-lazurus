"""
Training infrastructure for chuk-lazarus.

This module provides:
- Loss functions (DPO, GRPO, PPO, SFT)
- Trainer classes for different training paradigms
- Utility functions for RL training
- Learning rate schedulers
"""

# Base Trainer
from .base_trainer import BaseTrainer, BaseTrainerConfig

# Classification Trainer
from .classification_trainer import (
    ClassificationTrainer,
    ClassificationTrainerConfig,
    EncoderProtocol,
    evaluate,
    predict,
    train_classifier,
)

# Loss functions
from .losses import (
    DPOConfig,
    GRPOBatch,
    GRPOConfig,
    PPOConfig,
    SFTLossConfig,
    compute_grpo_advantages,
    compute_ppo_loss_for_batch,
    create_dpo_loss_fn,
    dpo_loss,
    grpo_loss,
    ppo_loss,
    sft_loss,
)

# Learning rate schedulers
from .schedulers import schedule_learning_rate

# Trainers (MLX-only in EWS-9 era). Skipped entirely when CHUK_BACKEND=torch
# so training submodules remain lazy-import-clean on torch hosts.
# EWS-10 replaces this with dual-backend dispatch.
import os as _os

if _os.environ.get("CHUK_BACKEND", "").lower() != "torch":
    try:
        from .trainers import (
            DPOTrainer,
            DPOTrainerConfig,
            GRPOTrainer,
            GRPOTrainerConfig,
            PPOTrainer,
            PPOTrainerConfig,
            SFTConfig,
            SFTTrainer,
        )
    except ImportError:
        DPOTrainer = DPOTrainerConfig = None  # type: ignore[assignment]
        GRPOTrainer = GRPOTrainerConfig = None  # type: ignore[assignment]
        PPOTrainer = PPOTrainerConfig = None  # type: ignore[assignment]
        SFTConfig = SFTTrainer = None  # type: ignore[assignment]
else:
    DPOTrainer = DPOTrainerConfig = None  # type: ignore[assignment]
    GRPOTrainer = GRPOTrainerConfig = None  # type: ignore[assignment]
    PPOTrainer = PPOTrainerConfig = None  # type: ignore[assignment]
    SFTConfig = SFTTrainer = None  # type: ignore[assignment]

# Utilities
from .utils import (
    compute_approx_kl,
    compute_gae,
    compute_kl_divergence,
    compute_log_probs_from_logits,
    compute_returns,
    compute_sequence_log_prob,
    extract_log_probs,
    normalize_advantages,
)

__all__ = [
    # Base
    "BaseTrainer",
    "BaseTrainerConfig",
    # Classification
    "ClassificationTrainer",
    "ClassificationTrainerConfig",
    "EncoderProtocol",
    "evaluate",
    "predict",
    "train_classifier",
    # Losses
    "DPOConfig",
    "GRPOBatch",
    "GRPOConfig",
    "PPOConfig",
    "SFTConfig",
    "SFTLossConfig",
    "compute_grpo_advantages",
    "compute_ppo_loss_for_batch",
    "create_dpo_loss_fn",
    "dpo_loss",
    "grpo_loss",
    "ppo_loss",
    "sft_loss",
    # Schedulers
    "schedule_learning_rate",
    # Trainers
    "DPOTrainer",
    "DPOTrainerConfig",
    "GRPOTrainer",
    "GRPOTrainerConfig",
    "PPOTrainer",
    "PPOTrainerConfig",
    "SFTConfig",
    "SFTTrainer",
    # Utils
    "compute_approx_kl",
    "compute_gae",
    "compute_kl_divergence",
    "compute_log_probs_from_logits",
    "compute_returns",
    "compute_sequence_log_prob",
    "extract_log_probs",
    "normalize_advantages",
]
