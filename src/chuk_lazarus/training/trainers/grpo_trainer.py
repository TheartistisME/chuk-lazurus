"""
GRPO Trainer - Group Relative Policy Optimization for LLMs.

GRPO is particularly well-suited for training Mistral as a tool-using agent
because:
- Generates multiple responses per prompt (natural for LLMs)
- No value function needed (simpler than PPO for text)
- Group-relative advantages work well with sparse rewards
- Can integrate MCP tools during sample generation
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .._lazy_mlx import mx, nn, optim
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.grpo_loss import GRPOBatch, GRPOConfig, grpo_loss
from ..utils.log_probs import compute_sequence_log_prob, extract_log_probs

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core  # noqa: F401
    import mlx.nn  # noqa: F401
    import mlx.optimizers  # noqa: F401

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


def _metric_to_float(value: Any) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().float().mean().cpu().item())
    except ImportError:
        pass

    return float(value)


def _detach_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: _metric_to_float(value) for key, value in metrics.items()}


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


def _load_reward_script(path: Path) -> tuple[Callable[[str, str], float], Callable[[], list[str]]]:
    spec = importlib.util.spec_from_file_location(
        "chuk_lazarus.training.reward_script", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load reward script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    reward_fn = getattr(module, "reward_fn", None)
    get_prompts = getattr(module, "get_prompts", None)
    if not callable(reward_fn):
        raise ValueError(f"{path} must define reward_fn(prompt, response) -> float")
    if not callable(get_prompts):
        raise ValueError(f"{path} must define get_prompts() -> list[str]")
    return reward_fn, get_prompts


class GRPOTrainingConfig(BaseModel):
    """High-level configuration for GRPO training."""

    model: str = Field(..., description="Policy model path or HuggingFace name")
    ref_model: str | None = Field(default=None, description="Reference model path")
    reward_script: Path = Field(..., description="Python script with reward_fn/get_prompts")
    output_dir: Path = Field(default=Path("./checkpoints/grpo"), description="Output directory")
    num_iterations: int = Field(default=1000, ge=1, description="Training iterations")
    prompts_per_iteration: int = Field(default=16, ge=1, description="Prompts per iteration")
    group_size: int = Field(default=4, ge=1, description="Responses per prompt")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    kl_coef: float = Field(default=0.1, ge=0, description="KL penalty coefficient")
    max_response_length: int = Field(default=256, ge=1, description="Maximum response length")
    temperature: float = Field(default=1.0, gt=0, description="Sampling temperature")
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(default=16.0, gt=0, description="LoRA alpha scaling")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="LoRA target modules",
    )
    log_interval: int = Field(default=1, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=50, ge=1, description="Checkpoint interval")

    @property
    def reference_model(self) -> str:
        return self.ref_model or self.model


class GRPOTrainingResult(BaseModel):
    """Result of GRPO training."""

    output_dir: Path = Field(..., description="Output directory")
    iterations_completed: int = Field(..., description="Iterations completed")
    final_mean_reward: float | None = Field(default=None, description="Final mean reward")
    final_kl: float | None = Field(default=None, description="Final KL penalty")
    adapter_path: Path | None = Field(default=None, description="Saved adapter path")


class GRPOTrainerConfig(BaseTrainerConfig):
    """Configuration for GRPO training."""

    # GRPO hyperparameters
    grpo: GRPOConfig = Field(default_factory=GRPOConfig, description="GRPO loss configuration")
    # Training settings
    num_iterations: int = Field(default=1000, ge=1, description="Number of training iterations")
    prompts_per_iteration: int = Field(default=16, ge=1, description="Prompts per iteration")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    # Generation settings
    max_response_length: int = Field(default=256, ge=1, description="Maximum response length")
    temperature: float = Field(default=1.0, gt=0, description="Sampling temperature")
    # Logging and checkpoints
    log_interval: int = Field(default=1, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=50, ge=1, description="Checkpoint interval")
    checkpoint_dir: str = Field(default="./checkpoints/grpo", description="Checkpoint directory")
    # Early stopping
    max_steps: int | None = Field(default=None, description="Maximum training steps")
    target_reward: float | None = Field(
        default=None, description="Target reward for early stopping"
    )


class GRPOTrainer(BaseTrainer):
    """
    Trainer for Group Relative Policy Optimization.

    Designed for training Mistral-7B (or similar) with tool use.

    The training loop:
    1. Sample prompts
    2. Generate group_size responses per prompt
    3. Compute rewards for each response (via MCP tools)
    4. Update policy using group-relative advantages

    Usage:
        trainer = GRPOTrainer(
            policy_model=mistral,
            reference_model=ref_mistral,
            tokenizer=tokenizer,
            reward_fn=compute_reward,
            config=config
        )
        trainer.train(prompt_dataset)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        reward_fn: Callable[[str, str], float],
        config: GRPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        if optimizer is None and _is_torch_model(policy_model):
            os.environ.setdefault("CHUK_BACKEND", "torch")
        config = config or GRPOTrainerConfig()
        super().__init__(policy_model, tokenizer, config, optimizer)

        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_fn = reward_fn

        # Freeze reference model
        if hasattr(self.reference_model, "freeze"):
            self.reference_model.freeze()
        else:
            try:
                for param in self.reference_model.parameters():
                    param.requires_grad_(False)
            except TypeError:
                pass
            if hasattr(self.reference_model, "eval"):
                self.reference_model.eval()

        # GRPO-specific state
        self.iteration = 0
        self.best_reward = float("-inf")

    @classmethod
    def run(cls, config: GRPOTrainingConfig) -> GRPOTrainingResult:
        backend = _backend_name()
        if backend == "torch" and config.use_lora:
            raise RuntimeError(
                "GRPOTrainer.run with CHUK_BACKEND=torch and use_lora=True is still blocked "
                "by the MLX-only models_v2 LoRA stack outside EWS-10."
            )
        try:
            from ...models_v2 import (
                LoRAConfig,
                load_model,
                load_model_with_lora,
                save_adapter,
            )
        except Exception as exc:  # pragma: no cover - exercised via backend smoke only
            raise RuntimeError(
                "GRPOTrainer.run low-level torch training is ready, but high-level model "
                "loading is still blocked by out-of-scope models_v2 eager MLX imports "
                "(notably models_v2/__init__.py, models/base.py, and loader.py)."
            ) from exc

        reward_fn, get_prompts = _load_reward_script(config.reward_script)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading policy model: {config.model}")
        lora_layers = None
        lora_config = None
        if config.use_lora:
            lora_config = LoRAConfig(
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=0.0,
                target_modules=config.lora_targets,
            )
            result = load_model_with_lora(config.model, lora_config)
            policy_model = result.model
            tokenizer = result.tokenizer
            lora_layers = result.lora_layers
        else:
            result = load_model(config.model)
            policy_model = result.model
            tokenizer = result.tokenizer

        logger.info(f"Loading reference model: {config.reference_model}")
        ref_result = load_model(config.reference_model)
        reference_model = ref_result.model

        trainer_config = GRPOTrainerConfig(
            grpo=GRPOConfig(
                group_size=config.group_size,
                kl_coef=config.kl_coef,
                temperature=config.temperature,
            ),
            num_iterations=config.num_iterations,
            prompts_per_iteration=config.prompts_per_iteration,
            learning_rate=config.learning_rate,
            max_response_length=config.max_response_length,
            checkpoint_dir=str(output_dir / "checkpoints"),
            log_interval=config.log_interval,
            checkpoint_interval=config.checkpoint_interval,
        )
        trainer = cls(policy_model, reference_model, tokenizer, reward_fn, trainer_config)

        if lora_layers:
            trainer.lora_layers = lora_layers
            trainer.lora_config = lora_config

        trainer.train(get_prompts)

        adapter_path = None
        if config.use_lora and lora_layers:
            adapter_path = output_dir / "adapters"
            save_adapter(lora_layers, adapter_path, lora_config=lora_config)

        final_metrics = trainer.metrics_history[-1] if trainer.metrics_history else {}
        return GRPOTrainingResult(
            output_dir=output_dir,
            iterations_completed=trainer.iteration,
            final_mean_reward=final_metrics.get("mean_reward"),
            final_kl=final_metrics.get("kl_penalty"),
            adapter_path=adapter_path,
        )

    @property
    def grpo_config(self) -> GRPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """
        Compute GRPO loss. Not used directly - GRPO has custom training loop.
        """
        raise NotImplementedError("GRPO uses custom training loop via train()")

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Not used - GRPO generates samples on the fly."""
        raise NotImplementedError("GRPO generates samples on the fly")

    def train(self, prompt_source: Callable[[], list[str]]):
        """
        Run GRPO training.

        Args:
            prompt_source: Function that returns a batch of prompts
        """
        logger.info(f"Starting GRPO training for {self.grpo_config.num_iterations} iterations")
        logger.info(f"Group size: {self.grpo_config.grpo.group_size}")
        logger.info(f"Prompts per iteration: {self.grpo_config.prompts_per_iteration}")

        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()

        for self.iteration in range(1, self.grpo_config.num_iterations + 1):
            # Get prompts
            prompts = prompt_source()[: self.grpo_config.prompts_per_iteration]

            # Generate samples and compute rewards
            batch = self._generate_grpo_batch(prompts)

            # Compute GRPO update
            metrics = self._grpo_update(batch)

            # Logging
            if self.iteration % self.config.log_interval == 0:
                elapsed = time.time() - self._start_time
                logger.info(
                    f"Iter {self.iteration} | "
                    f"Mean Reward: {metrics['mean_reward']:.4f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"KL: {metrics['kl_penalty']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

                self.metrics_history.append({"iteration": self.iteration, **metrics})

            # Checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f"iter_{self.iteration}")

            # Early stopping
            if self.grpo_config.target_reward is not None:
                if metrics["mean_reward"] >= self.grpo_config.target_reward:
                    logger.info(f"Target reward reached: {metrics['mean_reward']:.4f}")
                    break

            # Track best
            if metrics["mean_reward"] > self.best_reward:
                self.best_reward = metrics["mean_reward"]
                self.save_checkpoint("best")

        self.save_checkpoint("final")
        logger.info(f"Training complete. Iterations: {self.iteration}")

    def _generate_grpo_batch(self, prompts: list[str]) -> GRPOBatch:
        """Generate responses and compute rewards for a batch of prompts."""
        batch = GRPOBatch(self.grpo_config.grpo.group_size)

        for prompt in prompts:
            responses = []
            rewards = []

            # Generate group_size responses
            for _ in range(self.grpo_config.grpo.group_size):
                response = self._generate_response(prompt)
                responses.append(response)

                # Compute reward (this is where MCP tools get called)
                reward = self.reward_fn(prompt, response)
                rewards.append(reward)

            batch.add_prompt_group(prompt, responses, rewards)

        return batch

    def _generate_response(self, prompt: str) -> str:
        """Generate a single response from the policy model."""
        if _backend_name(self.policy_model) == "torch":
            import torch

            input_ids = self.tokenizer.encode(prompt)
            generated = list(input_ids)
            if hasattr(self.policy_model, "set_mode"):
                self.policy_model.set_mode("INFERENCE")

            device = _resolve_torch_device(self.policy_model)
            self.policy_model.to(device)
            self.policy_model.eval()

            for _ in range(self.grpo_config.max_response_length):
                input_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                with torch.no_grad():
                    output = self.policy_model(input_tensor)
                    if hasattr(output, "logits"):
                        logits = output.logits
                    elif isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                next_logits = logits[0, -1, :] / self.grpo_config.temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = self._sample_token(probs)
                generated.append(int(next_token.item()))
                if hasattr(self.tokenizer, "eos_token_id"):
                    if int(next_token.item()) == self.tokenizer.eos_token_id:
                        break

            response_ids = generated[len(input_ids) :]
            return self.tokenizer.decode(response_ids)

        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        generated = list(input_ids)

        # Set model to inference mode if applicable
        if hasattr(self.policy_model, "set_mode"):
            self.policy_model.set_mode("INFERENCE")

        max_new_tokens = self.grpo_config.max_response_length

        for _ in range(max_new_tokens):
            input_tensor = mx.array([generated])

            # Forward pass without gradient tracking
            output = self.policy_model(input_tensor)
            # Handle both tuple and ModelOutput returns
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            logits = mx.stop_gradient(logits)

            # Get logits for last position
            next_logits = logits[0, -1, :] / self.grpo_config.temperature

            # Sample
            probs = mx.softmax(next_logits)
            next_token = self._sample_token(probs)

            generated.append(int(next_token))

            # Check for EOS
            if hasattr(self.tokenizer, "eos_token_id"):
                if int(next_token) == self.tokenizer.eos_token_id:
                    break

        # Decode response (exclude prompt)
        response_ids = generated[len(input_ids) :]
        response = self.tokenizer.decode(response_ids)

        return response

    def _grpo_update(self, batch: GRPOBatch) -> dict[str, float]:
        """Perform GRPO update on the batch."""
        if _backend_name(self.policy_model) == "torch":
            import torch

            device = _resolve_torch_device(self.policy_model)
            self.policy_model.to(device)
            self.reference_model.to(device)
            self.policy_model.train()
            self.reference_model.eval()

            sequences = batch.get_all_sequences()
            rewards = torch.tensor(
                [reward for group in batch.rewards for reward in group],
                dtype=torch.float32,
                device=device,
            )

            all_input_ids: list[list[int]] = []
            all_masks: list[list[float]] = []
            max_len = 0
            for seq in sequences:
                tokens = self.tokenizer.encode(seq)
                all_input_ids.append(tokens)
                max_len = max(max_len, len(tokens))

            pad_id = self.pad_token_id
            for idx, tokens in enumerate(all_input_ids):
                padding = [pad_id] * (max_len - len(tokens))
                mask = [1.0] * len(tokens) + [0.0] * len(padding)
                all_input_ids[idx] = tokens + padding
                all_masks.append(mask)

            input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device)
            attention_mask = torch.tensor(all_masks, dtype=torch.float32, device=device)
            mask_shifted = attention_mask[:, 1:]

            with torch.no_grad():
                ref_log_probs, _ = extract_log_probs(
                    self.reference_model, input_ids, attention_mask
                )
                ref_seq_log_probs = compute_sequence_log_prob(ref_log_probs, mask_shifted)

            self.optimizer.zero_grad(set_to_none=True)
            with _torch_autocast(self.policy_model):
                policy_log_probs, _ = extract_log_probs(self.policy_model, input_ids, attention_mask)
                policy_seq_log_probs = compute_sequence_log_prob(policy_log_probs, mask_shifted)
                loss, _ = grpo_loss(
                    log_probs=policy_seq_log_probs,
                    ref_log_probs=ref_seq_log_probs,
                    rewards=rewards,
                    group_size=self.grpo_config.grpo.group_size,
                    config=self.grpo_config.grpo,
                )
            loss.backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()

            with torch.no_grad():
                policy_log_probs, _ = extract_log_probs(
                    self.policy_model, input_ids, attention_mask
                )
                policy_seq_log_probs = compute_sequence_log_prob(policy_log_probs, mask_shifted)
                _, metrics = grpo_loss(
                    log_probs=policy_seq_log_probs,
                    ref_log_probs=ref_seq_log_probs,
                    rewards=rewards,
                    group_size=self.grpo_config.grpo.group_size,
                    config=self.grpo_config.grpo,
                )

            return _detach_metrics(metrics)

        # Get all sequences
        sequences = batch.get_all_sequences()
        rewards = batch.get_flat_rewards()

        # Tokenize all sequences
        all_input_ids = []
        all_masks = []
        max_len = 0

        for seq in sequences:
            tokens = self.tokenizer.encode(seq)
            all_input_ids.append(tokens)
            max_len = max(max_len, len(tokens))

        # Pad sequences
        pad_id = self.pad_token_id
        for i, tokens in enumerate(all_input_ids):
            padding = [pad_id] * (max_len - len(tokens))
            mask = [1.0] * len(tokens) + [0.0] * len(padding)
            all_input_ids[i] = tokens + padding
            all_masks.append(mask)

        input_ids = mx.array(all_input_ids)
        attention_mask = mx.array(all_masks)

        # Get log probs from policy
        policy_log_probs, _ = extract_log_probs(self.policy_model, input_ids, attention_mask)

        # Get log probs from reference (no gradients needed)
        ref_log_probs, _ = extract_log_probs(self.reference_model, input_ids, attention_mask)
        ref_log_probs = mx.stop_gradient(ref_log_probs)

        # Sum to sequence level
        mask_shifted = attention_mask[:, 1:]
        policy_seq_log_probs = compute_sequence_log_prob(policy_log_probs, mask_shifted)
        ref_seq_log_probs = compute_sequence_log_prob(ref_log_probs, mask_shifted)

        # Compute loss
        def loss_fn(model):
            p_log_probs, _ = extract_log_probs(model, input_ids, attention_mask)
            p_seq = compute_sequence_log_prob(p_log_probs, mask_shifted)

            loss, _ = grpo_loss(
                log_probs=p_seq,
                ref_log_probs=ref_seq_log_probs,
                rewards=rewards,
                group_size=self.grpo_config.grpo.group_size,
                config=self.grpo_config.grpo,
            )
            return loss

        # Compute gradients
        loss, grads = nn.value_and_grad(self.policy_model, loss_fn)(self.policy_model)

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            grads = self.clip_gradients(grads, self.config.max_grad_norm)

        # Update
        self.optimizer.update(self.policy_model, grads)
        mx.eval(self.policy_model.parameters())

        # Compute metrics
        _, metrics = grpo_loss(
            log_probs=policy_seq_log_probs,
            ref_log_probs=ref_seq_log_probs,
            rewards=rewards,
            group_size=self.grpo_config.grpo.group_size,
            config=self.grpo_config.grpo,
        )

        return {k: float(v) for k, v in metrics.items()}

    def _sample_token(self, probs: mx.array) -> mx.array:
        """Sample from probability distribution using Gumbel-softmax trick."""
        if _backend_name(self.policy_model) == "torch":
            import torch

            return torch.multinomial(probs, 1).squeeze(-1)

        u = mx.random.uniform(shape=probs.shape)
        gumbel = -mx.log(-mx.log(u + 1e-10) + 1e-10)
        return mx.argmax(mx.log(probs + 1e-10) + gumbel)

    def save_checkpoint(self, name: str):
        """Save model checkpoint in safetensors format.

        Uses base class implementation for LoRA-aware saving.
        """
        if _backend_name(self.policy_model) == "torch":
            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_dir / f"{name}.safetensors"
            _save_torch_state(self.policy_model, weights_path)
            logger.info(f"Saved checkpoint: {weights_path}")
            return

        # GRPO uses policy_model instead of model
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check for LoRA layers
        if hasattr(self, "lora_layers") and self.lora_layers:
            from ...models_v2.loader import save_adapter

            adapter_path = checkpoint_dir / name
            lora_config = getattr(self, "lora_config", None)
            save_adapter(self.lora_layers, adapter_path, lora_config=lora_config)
            logger.info(f"Saved LoRA adapter: {adapter_path}")
        else:
            # Save full model weights
            weights_path = checkpoint_dir / f"{name}.safetensors"
            weights = dict(self.policy_model.parameters())
            flat_weights = self._flatten_params(weights)
            mx.save_safetensors(str(weights_path), flat_weights)
            logger.info(f"Saved checkpoint: {weights_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if _backend_name(self.policy_model) == "torch":
            _load_torch_state(self.policy_model, path)
            self.policy_model.to(_resolve_torch_device(self.policy_model))
            logger.info(f"Loaded checkpoint: {path}")
            return

        weights = mx.load(path)
        self.policy_model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
