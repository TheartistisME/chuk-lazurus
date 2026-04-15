"""
PPO Trainer - Proximal Policy Optimization training loop.

This trainer handles:
- Environment interaction (rollout collection)
- PPO updates with clipped objective
- Support for both Mistral (LLM) and RNN experts
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .._lazy_mlx import mx, nn, optim
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.ppo_loss import PPOConfig, ppo_loss

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core  # noqa: F401
    import mlx.nn  # noqa: F401
    import mlx.optimizers  # noqa: F401

    from ...data import RolloutBuffer

logger = logging.getLogger(__name__)


def _is_torch_model(model: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False

    return isinstance(model, torch.nn.Module)


def _backend_name(model: Any | None = None) -> str:
    backend = os.environ.get("CHUK_BACKEND", "").lower()
    if backend in {"mlx", "torch"}:
        return backend
    return "torch" if model is not None and _is_torch_model(model) else "mlx"


def _resolve_torch_device(model: Any) -> Any:
    import torch

    raw = os.environ.get("CHUK_DEVICE")
    if raw:
        if raw.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CHUK_DEVICE={raw!r} requested CUDA but torch.cuda.is_available() is False."
            )
        return torch.device(raw)

    if _is_torch_model(model):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_autocast(model: Any):
    import torch

    device = _resolve_torch_device(model)
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _save_torch_state(model: Any, path: Path) -> None:
    from safetensors.torch import save_file

    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(path))


def _load_torch_state(model: Any, path: str) -> None:
    from safetensors.torch import load_file

    state = load_file(path)
    model.load_state_dict(state)


class _TorchRolloutBuffer:
    def __init__(
        self,
        buffer_size: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_envs: int = 1,
    ) -> None:
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.observations: list[Any] = []
        self.actions: list[Any] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.advantages = None
        self.returns = None
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_reward = 0.0
        self._current_length = 0

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        done: bool,
        log_prob: float,
        value: float = 0.0,
        hidden_state: Any = None,
        env_idx: int = 0,
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self._current_reward += reward
        self._current_length += 1
        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            self._current_reward = 0.0
            self._current_length = 0

    def compute_advantages(self, last_values: Any = None):
        import torch

        last_value = 0.0
        if last_values is not None:
            if isinstance(last_values, torch.Tensor) and last_values.numel() > 0:
                last_value = float(last_values.reshape(-1)[0].detach().cpu().item())
            else:
                last_value = float(last_values)

        advantages = [0.0] * len(self.rewards)
        last_gae = 0.0
        for t in range(len(self.rewards) - 1, -1, -1):
            next_value = last_value if t == len(self.rewards) - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.returns = self.advantages + torch.tensor(self.values, dtype=torch.float32)

    def get_batches(self, batch_size: int, shuffle: bool = True) -> Iterator[dict[str, Any]]:
        import random
        import torch

        if self.advantages is None:
            self.compute_advantages()

        indices = list(range(len(self.observations)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield {
                "observations": [self.observations[i] for i in batch_indices],
                "actions": torch.tensor([self.actions[i] for i in batch_indices]),
                "old_log_probs": torch.tensor(
                    [self.log_probs[i] for i in batch_indices], dtype=torch.float32
                ),
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
                "values": torch.tensor([self.values[i] for i in batch_indices], dtype=torch.float32),
            }

    def get_episode_stats(self) -> dict[str, float]:
        if not self.episode_rewards:
            return {"num_episodes": 0, "mean_reward": 0.0, "mean_length": 0.0}
        rewards = self.episode_rewards
        lengths = self.episode_lengths
        mean_reward = sum(rewards) / len(rewards)
        return {
            "num_episodes": len(rewards),
            "mean_reward": mean_reward,
            "std_reward": (
                sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
            )
            ** 0.5,
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_length": sum(lengths) / len(lengths),
        }

    def __len__(self) -> int:
        return len(self.observations)


class PPOTrainerConfig(BaseTrainerConfig):
    """Configuration for PPO training."""

    # PPO hyperparameters
    ppo: PPOConfig = Field(default_factory=PPOConfig, description="PPO loss configuration")
    # Rollout settings
    rollout_steps: int = Field(default=2048, ge=1, description="Steps per rollout")
    num_envs: int = Field(default=1, ge=1, description="Parallel environments")
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")
    gae_lambda: float = Field(default=0.95, ge=0, le=1, description="GAE lambda")
    # Update settings
    num_epochs_per_rollout: int = Field(default=4, ge=1, description="PPO epochs per rollout")
    batch_size: int = Field(default=64, ge=1, description="Minibatch size")
    learning_rate: float = Field(default=3e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    # Training settings
    total_timesteps: int = Field(default=1_000_000, ge=1, description="Total timesteps")
    warmup_steps: int = Field(default=0, ge=0, description="Warmup steps")
    # Logging and checkpoints
    log_interval: int = Field(default=1, ge=1, description="Log every N rollouts")
    checkpoint_interval: int = Field(default=10, ge=1, description="Checkpoint every N rollouts")
    checkpoint_dir: str = Field(default="./checkpoints/ppo", description="Checkpoint directory")
    # Early stopping
    max_steps: int | None = Field(default=None, description="Maximum training steps")
    target_reward: float | None = Field(
        default=None, description="Target reward for early stopping"
    )


class PPOTrainer(BaseTrainer):
    """
    Trainer for Proximal Policy Optimization.

    Works with:
    - RNN experts (continuous/discrete control)
    - Future: LLM policies (Mistral)

    Usage:
        trainer = PPOTrainer(
            policy=expert,
            env=env,
            config=config
        )
        trainer.train()
    """

    def __init__(
        self,
        policy: nn.Module,
        env: Any,  # Environment interface
        config: PPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        if optimizer is None and _is_torch_model(policy):
            os.environ.setdefault("CHUK_BACKEND", "torch")
        config = config or PPOTrainerConfig()
        super().__init__(policy, None, config, optimizer)  # No tokenizer for PPO

        self.policy = policy
        self.env = env

        # Rollout buffer
        if _backend_name(policy) == "torch":
            self.buffer = _TorchRolloutBuffer(
                buffer_size=config.rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                num_envs=config.num_envs,
            )
        else:
            from ...data.rollout_buffer import RolloutBuffer

            self.buffer = RolloutBuffer(
                buffer_size=config.rollout_steps,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                num_envs=config.num_envs,
            )

        # PPO-specific state
        self.num_rollouts = 0
        self.best_reward = float("-inf")

    @property
    def ppo_config(self) -> PPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Not used directly - PPO has custom training loop."""
        raise NotImplementedError("PPO uses custom training loop")

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Not used - PPO generates rollouts from environment."""
        raise NotImplementedError("PPO generates rollouts from environment")

    def train(self):
        """Run PPO training loop."""
        logger.info(f"Starting PPO training for {self.ppo_config.total_timesteps} timesteps")
        logger.info(f"Config: {self.config}")

        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        obs = self.env.reset()

        while self.global_step < self.ppo_config.total_timesteps:
            # Collect rollout
            obs = self._collect_rollout(obs)
            self.num_rollouts += 1

            # Get episode stats before update
            episode_stats = self.buffer.get_episode_stats()

            # Compute advantages
            # Get value estimate for last observation
            last_obs = mx.array([self._obs_to_tensor(obs)])
            result = self.policy(last_obs, deterministic=True)
            last_value = result.get("value", mx.zeros((1,)))
            mx.eval(last_value)

            self.buffer.compute_advantages(last_value)

            # PPO updates
            update_metrics = self._ppo_update()

            # Logging
            if self.num_rollouts % self.config.log_interval == 0:
                elapsed = time.time() - self._start_time
                fps = self.global_step / elapsed

                logger.info(
                    f"Rollout {self.num_rollouts} | "
                    f"Steps: {self.global_step} | "
                    f"Mean Reward: {episode_stats['mean_reward']:.2f} | "
                    f"Policy Loss: {update_metrics['policy_loss']:.4f} | "
                    f"Value Loss: {update_metrics['value_loss']:.4f} | "
                    f"FPS: {fps:.0f}"
                )

                self.metrics_history.append(
                    {
                        "rollout": self.num_rollouts,
                        "step": self.global_step,
                        **episode_stats,
                        **update_metrics,
                    }
                )

            # Checkpoint
            if self.num_rollouts % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"rollout_{self.num_rollouts}")

            # Check for target reward
            if self.ppo_config.target_reward is not None:
                if episode_stats["mean_reward"] >= self.ppo_config.target_reward:
                    logger.info(f"Target reward reached: {episode_stats['mean_reward']:.2f}")
                    break

            # Update best reward
            if episode_stats["mean_reward"] > self.best_reward:
                self.best_reward = episode_stats["mean_reward"]
                self.save_checkpoint("best")

            # Reset buffer for next rollout
            self.buffer.reset()

        # Final checkpoint
        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def _collect_rollout(self, initial_obs):
        """Collect rollout_steps of experience."""
        if _backend_name(self.policy) == "torch":
            import torch

            obs = initial_obs
            device = _resolve_torch_device(self.policy)
            self.policy.to(device)

            if hasattr(self.policy, "reset_hidden"):
                self.policy.reset_hidden(batch_size=1)

            for _ in range(self.ppo_config.rollout_steps):
                obs_tensor = torch.tensor(
                    [self._obs_to_tensor(obs)],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad(), _torch_autocast(self.policy):
                    result = self.policy(obs_tensor, deterministic=False)

                action = result["action"][0]
                log_prob = float(result["log_prob"][0].detach().cpu().item())
                value_tensor = result.get("value")
                value = (
                    float(value_tensor[0].detach().cpu().item())
                    if value_tensor is not None
                    else 0.0
                )

                env_action = action.detach().cpu().tolist() if torch.is_tensor(action) else action
                next_obs, reward, done, info = self.env.step(env_action)
                self.buffer.add(
                    observation=obs,
                    action=env_action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )
                self.global_step += 1
                obs = self.env.reset() if done else next_obs

            return obs

        obs = initial_obs

        # Reset hidden state for new rollout
        if hasattr(self.policy, "reset_hidden"):
            self.policy.reset_hidden(batch_size=1)

        for _ in range(self.ppo_config.rollout_steps):
            # Get action from policy
            obs_tensor = mx.array([self._obs_to_tensor(obs)])

            # Forward pass (no gradients needed for rollout collection)
            result = self.policy(obs_tensor, deterministic=False)
            mx.eval(result["action"], result["log_prob"], result.get("value"))

            action = result["action"][0]
            log_prob = float(result["log_prob"][0])
            value = float(result["value"][0]) if result["value"] is not None else 0.0

            # Step environment
            next_obs, reward, done, info = self.env.step(action)

            # Store in buffer
            self.buffer.add(
                observation=obs,
                action=action.tolist() if isinstance(action, mx.array) else action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            self.global_step += 1

            # Reset if done
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs

        return obs

    def _ppo_update(self) -> dict[str, float]:
        """Perform PPO update on collected rollout."""
        if _backend_name(self.policy) == "torch":
            import torch

            device = _resolve_torch_device(self.policy)
            self.policy.to(device)
            all_metrics = {
                "policy_loss": [],
                "value_loss": [],
                "entropy_loss": [],
                "approx_kl": [],
                "clip_fraction": [],
            }

            for epoch in range(self.ppo_config.num_epochs_per_rollout):
                for batch in self.buffer.get_batches(
                    batch_size=self.ppo_config.batch_size, shuffle=True
                ):
                    batch_size = len(batch["observations"])
                    if hasattr(self.policy, "reset_hidden"):
                        self.policy.reset_hidden(batch_size=batch_size)

                    obs_tensors = torch.tensor(
                        [self._obs_to_tensor(o) for o in batch["observations"]],
                        dtype=torch.float32,
                        device=device,
                    )
                    actions = batch["actions"].to(device)
                    old_log_probs = batch["old_log_probs"].to(device)
                    advantages = batch["advantages"].to(device)
                    returns = batch["returns"].to(device)

                    self.optimizer.zero_grad(set_to_none=True)
                    with _torch_autocast(self.policy):
                        result = self.policy.get_action_and_value(obs_tensors, action=actions)
                        loss, metrics = ppo_loss(
                            log_probs=result["log_prob"],
                            old_log_probs=old_log_probs,
                            advantages=advantages,
                            values=result["value"],
                            returns=returns,
                            entropy=result["entropy"],
                            config=self.ppo_config.ppo,
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.ppo_config.ppo.max_grad_norm
                    )
                    self.optimizer.step()

                    metric_values = {
                        key: float(value.detach().cpu().item())
                        if torch.is_tensor(value)
                        else float(value)
                        for key, value in metrics.items()
                    }
                    for key in all_metrics:
                        if key in metric_values:
                            all_metrics[key].append(metric_values[key])

                    if metric_values["approx_kl"] > 1.5 * self.ppo_config.ppo.target_kl:
                        logger.debug(f"Early stopping at epoch {epoch} due to high KL")
                        break

            return {key: sum(values) / len(values) if values else 0.0 for key, values in all_metrics.items()}

        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(self.ppo_config.num_epochs_per_rollout):
            for batch in self.buffer.get_batches(
                batch_size=self.ppo_config.batch_size, shuffle=True
            ):
                # Reset hidden state for batch processing (stateless for PPO updates)
                batch_size = len(batch["observations"])
                if hasattr(self.policy, "reset_hidden"):
                    self.policy.reset_hidden(batch_size=batch_size)

                # Convert batch observations to tensors
                obs_tensors = mx.array(
                    [self._obs_to_tensor(o) for o in batch["observations"]],
                    dtype=mx.float32,
                )
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Define loss function for gradients
                def loss_fn(model):
                    result = model.get_action_and_value(obs_tensors, action=actions)
                    loss, _ = ppo_loss(
                        log_probs=result["log_prob"],
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        values=result["value"],
                        returns=returns,
                        entropy=result["entropy"],
                        config=self.ppo_config.ppo,
                    )
                    return loss

                # Compute loss and gradients
                loss, grads = nn.value_and_grad(self.policy, loss_fn)(self.policy)

                # Clip gradients
                grads = self.clip_gradients(grads, self.ppo_config.ppo.max_grad_norm)

                # Update
                self.optimizer.update(self.policy, grads)
                mx.eval(self.policy.parameters())

                # Compute metrics separately (for logging)
                result = self.policy.get_action_and_value(obs_tensors, action=actions)
                _, metrics = ppo_loss(
                    log_probs=result["log_prob"],
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    values=result["value"],
                    returns=returns,
                    entropy=result["entropy"],
                    config=self.ppo_config.ppo,
                )

                # Track metrics
                for key in all_metrics:
                    if key in metrics:
                        all_metrics[key].append(float(metrics[key]))

                # Early stopping on KL
                if float(metrics["approx_kl"]) > 1.5 * self.ppo_config.ppo.target_kl:
                    logger.debug(f"Early stopping at epoch {epoch} due to high KL")
                    break

        # Average metrics
        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _obs_to_tensor(self, obs) -> list[float]:
        """Convert observation to tensor format."""
        try:
            import torch

            if isinstance(obs, torch.Tensor):
                return obs.detach().cpu().tolist()
        except ImportError:
            pass

        if isinstance(obs, (list, tuple)):
            return list(obs)
        elif isinstance(obs, mx.array):
            return obs.tolist()
        elif isinstance(obs, dict):
            # Flatten dict observation
            flat = []
            for v in obs.values():
                if isinstance(v, (int, float)):
                    flat.append(float(v))
                elif isinstance(v, (list, tuple)):
                    flat.extend([float(x) for x in v])
            return flat
        else:
            return [0.0] * 10  # Default

    def save_checkpoint(self, name: str):
        """Save model checkpoint in safetensors format.

        Uses base class _flatten_params for consistent weight handling.
        """
        if _backend_name(self.policy) == "torch":
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_dir / f"{name}.safetensors"
            _save_torch_state(self.policy, weights_path)
            logger.info(f"Saved checkpoint: {weights_path}")
            return

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        weights_path = checkpoint_dir / f"{name}.safetensors"
        weights = dict(self.policy.parameters())
        flat_weights = self._flatten_params(weights)
        mx.save_safetensors(str(weights_path), flat_weights)
        logger.info(f"Saved checkpoint: {weights_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if _backend_name(self.policy) == "torch":
            _load_torch_state(self.policy, path)
            self.policy.to(_resolve_torch_device(self.policy))
            logger.info(f"Loaded checkpoint: {path}")
            return

        weights = mx.load(path)
        self.policy.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
