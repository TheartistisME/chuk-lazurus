"""
DPO Trainer - Direct Preference Optimization training loop.

This trainer integrates with your existing chuk-mlx training infrastructure
while adding DPO-specific functionality.

Usage:
    # High-level API (recommended for CLI):
    result = DPOTrainer.run(DPOTrainingConfig(
        model="meta-llama/Llama-3.2-1B",
        data_path="preferences.jsonl",
        output_dir="./output",
    ))

    # Low-level API (for custom pipelines):
    trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
    trainer.train(dataset)
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .._lazy_mlx import mx, nn, optim
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.dpo_loss import DPOConfig, dpo_loss

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core  # noqa: F401
    import mlx.nn  # noqa: F401
    import mlx.optimizers  # noqa: F401

    from ...data import PreferenceDataset

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


def _move_batch_to_torch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif hasattr(value, "tolist"):
            moved[key] = torch.as_tensor(value.tolist(), device=device)
        else:
            moved[key] = value
    return moved


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


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


class _TorchPreferenceDataset:
    def __init__(
        self,
        data_path: Path,
        tokenizer: Any,
        max_length: int,
        max_prompt_length: int = 256,
    ) -> None:
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.pairs = self._load_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pairs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                chosen = item["chosen"]
                rejected = item["rejected"]
                if isinstance(chosen, list):
                    chosen = chosen[-1]["content"] if chosen else ""
                if isinstance(rejected, list):
                    rejected = rejected[-1]["content"] if rejected else ""
                rows.append(
                    {
                        "prompt": str(item["prompt"]),
                        "chosen": str(chosen),
                        "rejected": str(rejected),
                    }
                )
        return rows

    def _tokenize(self, pair: dict[str, Any]) -> dict[str, Any]:
        prompt_tokens = _encode_text(self.tokenizer, pair["prompt"])
        if len(prompt_tokens) > self.max_prompt_length:
            prompt_tokens = prompt_tokens[: self.max_prompt_length]

        chosen_tokens = _encode_text(self.tokenizer, pair["prompt"] + pair["chosen"])
        rejected_tokens = _encode_text(self.tokenizer, pair["prompt"] + pair["rejected"])

        return {
            "prompt_length": len(prompt_tokens),
            "chosen_input_ids": chosen_tokens[: self.max_length],
            "rejected_input_ids": rejected_tokens[: self.max_length],
        }

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, pad_token_id: int = 0
    ) -> Iterator[dict[str, Any]]:
        import random
        import torch

        indices = list(range(len(self.pairs)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            items = [self._tokenize(self.pairs[idx]) for idx in batch_indices]
            max_len = max(
                max(len(item["chosen_input_ids"]), len(item["rejected_input_ids"]))
                for item in items
            )

            chosen_ids = []
            rejected_ids = []
            chosen_mask = []
            rejected_mask = []
            prompt_lengths = []
            for item in items:
                chosen_len = len(item["chosen_input_ids"])
                rejected_len = len(item["rejected_input_ids"])
                chosen_pad = max_len - chosen_len
                rejected_pad = max_len - rejected_len
                chosen_ids.append(item["chosen_input_ids"] + [pad_token_id] * chosen_pad)
                rejected_ids.append(item["rejected_input_ids"] + [pad_token_id] * rejected_pad)
                chosen_mask.append([1.0] * chosen_len + [0.0] * chosen_pad)
                rejected_mask.append([1.0] * rejected_len + [0.0] * rejected_pad)
                prompt_lengths.append(item["prompt_length"])

            yield {
                "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
                "chosen_attention_mask": torch.tensor(chosen_mask, dtype=torch.float32),
                "rejected_attention_mask": torch.tensor(rejected_mask, dtype=torch.float32),
                "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            }


class DPOTrainingConfig(BaseModel):
    """Complete configuration for running DPO training.

    This is the high-level config used by CLI and run() method.
    Includes model paths, data paths, and all training parameters.
    """

    # Model
    model: str = Field(..., description="Policy model path or HuggingFace name")
    ref_model: str | None = Field(default=None, description="Reference model (defaults to policy)")
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(default=16.0, description="LoRA alpha scaling")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="LoRA target modules",
    )

    # Data
    data_path: Path = Field(..., description="Path to preference data (JSONL)")
    eval_data_path: Path | None = Field(default=None, description="Path to eval data")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")

    # Training
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    beta: float = Field(default=0.1, gt=0, description="DPO beta parameter")
    max_steps: int | None = Field(default=None, description="Max steps (overrides epochs)")

    # Output
    output_dir: Path = Field(default=Path("./checkpoints/dpo"), description="Output directory")
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")

    @property
    def reference_model(self) -> str:
        """Get reference model name (defaults to policy model)."""
        return self.ref_model or self.model


class DPOTrainingResult(BaseModel):
    """Result of DPO training."""

    output_dir: Path = Field(..., description="Output directory")
    epochs_completed: int = Field(..., description="Epochs completed")
    final_loss: float | None = Field(default=None, description="Final training loss")
    adapter_path: Path | None = Field(default=None, description="Path to saved LoRA adapter")


class DPOTrainerConfig(BaseTrainerConfig):
    """Configuration for DPO training."""

    # DPO hyperparameters
    dpo: DPOConfig = Field(default_factory=DPOConfig, description="DPO loss configuration")
    # Training settings
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-6, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    # Logging and checkpoints
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    eval_interval: int = Field(default=100, ge=1, description="Evaluation interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")
    checkpoint_dir: str = Field(default="./checkpoints/dpo", description="Checkpoint directory")
    # Early stopping
    max_steps: int | None = Field(default=None, description="Maximum training steps")
    target_reward_margin: float = Field(
        default=2.0, gt=0, description="Stop if margin exceeds this"
    )


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization.

    Usage:
        # High-level API (recommended):
        result = DPOTrainer.run(DPOTrainingConfig(
            model="meta-llama/Llama-3.2-1B",
            data_path="preferences.jsonl",
        ))

        # Low-level API:
        trainer = DPOTrainer(policy_model, ref_model, tokenizer, config)
        trainer.train(train_dataset)
    """

    @classmethod
    def run(cls, config: DPOTrainingConfig) -> DPOTrainingResult:
        """Run complete DPO training from config.

        This is the high-level entry point that handles:
        - Model loading (policy and reference, with optional LoRA)
        - Dataset loading
        - Training
        - Checkpoint saving

        Args:
            config: Complete training configuration

        Returns:
            DPOTrainingResult with training outcomes
        """
        backend = _backend_name()
        if backend == "torch" and config.use_lora:
            raise RuntimeError(
                "DPOTrainer.run with CHUK_BACKEND=torch and use_lora=True is still blocked "
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
                "DPOTrainer.run low-level torch training is ready, but high-level model "
                "loading is still blocked by out-of-scope models_v2 eager MLX imports "
                "(notably models_v2/__init__.py, models/base.py, and loader.py)."
            ) from exc

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load policy model
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
            logger.info(
                f"  Loaded with LoRA: {len(lora_layers)} layers, "
                f"{result.lora_parameter_count:,} trainable params"
            )
        else:
            result = load_model(config.model)
            policy_model = result.model
            tokenizer = result.tokenizer

        # Load reference model (never with LoRA - frozen)
        logger.info(f"Loading reference model: {config.reference_model}")
        ref_result = load_model(config.reference_model)
        ref_model = ref_result.model

        # Load datasets
        logger.info(f"Loading dataset: {config.data_path}")
        if backend == "torch":
            train_dataset = _TorchPreferenceDataset(
                config.data_path,
                tokenizer,
                max_length=config.max_length,
            )
        else:
            from ...data import PreferenceDataset

            train_dataset = PreferenceDataset(
                str(config.data_path),
                tokenizer,
                max_length=config.max_length,
            )
        logger.info(f"  Loaded {len(train_dataset)} preference pairs")

        eval_dataset = None
        if config.eval_data_path:
            if backend == "torch":
                eval_dataset = _TorchPreferenceDataset(
                    config.eval_data_path,
                    tokenizer,
                    max_length=config.max_length,
                )
            else:
                eval_dataset = PreferenceDataset(
                    str(config.eval_data_path),
                    tokenizer,
                    max_length=config.max_length,
                )
            logger.info(f"  Loaded {len(eval_dataset)} eval pairs")

        # Create trainer config
        trainer_config = DPOTrainerConfig(
            dpo=DPOConfig(beta=config.beta),
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            checkpoint_dir=str(output_dir / "checkpoints"),
            log_interval=config.log_interval,
            max_steps=config.max_steps,
            checkpoint_interval=config.checkpoint_interval,
        )

        # Create and run trainer
        trainer = cls(policy_model, ref_model, tokenizer, trainer_config)

        # Attach LoRA layers for checkpoint saving
        if lora_layers:
            trainer.lora_layers = lora_layers
            trainer.lora_config = lora_config

        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)

        # Save LoRA adapters
        adapter_path = None
        if config.use_lora and lora_layers:
            adapter_path = output_dir / "adapters"
            save_adapter(lora_layers, adapter_path, lora_config=lora_config)
            logger.info(f"Saved LoRA adapters to {adapter_path}")

        # Get final loss from metrics
        final_loss = None
        if trainer.metrics_history:
            final_loss = trainer.metrics_history[-1].get("loss")

        logger.info(f"Training complete. Output saved to {output_dir}")

        return DPOTrainingResult(
            output_dir=output_dir,
            epochs_completed=config.num_epochs,
            final_loss=final_loss,
            adapter_path=adapter_path,
        )

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        config: DPOTrainerConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        if optimizer is None and _is_torch_model(policy_model):
            os.environ.setdefault("CHUK_BACKEND", "torch")
        config = config or DPOTrainerConfig()
        super().__init__(policy_model, tokenizer, config, optimizer)

        self.policy_model = policy_model
        self.reference_model = reference_model

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

        # DPO-specific state
        self.best_reward_margin = float("-inf")

    @property
    def dpo_config(self) -> DPOTrainerConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute DPO loss for a batch."""
        loss, metrics = dpo_loss(
            policy_model=self.policy_model,
            reference_model=self.reference_model,
            chosen_input_ids=batch["chosen_input_ids"],
            rejected_input_ids=batch["rejected_input_ids"],
            chosen_attention_mask=batch["chosen_attention_mask"],
            rejected_attention_mask=batch["rejected_attention_mask"],
            config=self.dpo_config.dpo,
        )
        return loss, metrics

    def get_train_batches(self, dataset: PreferenceDataset) -> Iterator[dict[str, mx.array]]:
        """Get iterator over training batches."""
        return dataset.iter_batches(
            batch_size=self.dpo_config.batch_size,
            shuffle=True,
            pad_token_id=self.pad_token_id,
        )

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: PreferenceDataset = None,
        callback: Callable[[dict], None] = None,
    ):
        """
        Run DPO training.

        Args:
            train_dataset: Training preference pairs
            eval_dataset: Optional evaluation dataset
            callback: Optional callback called after each log interval
        """
        logger.info(f"Starting DPO training with {len(train_dataset)} preference pairs")
        if _backend_name(self.policy_model) != "torch":
            super().train(
                dataset=train_dataset,
                num_epochs=self.dpo_config.num_epochs,
                eval_dataset=eval_dataset,
                callback=callback,
            )
            return

        import torch

        device = _resolve_torch_device(self.policy_model)
        self.policy_model.to(device)
        self.reference_model.to(device)
        self.reference_model.eval()
        self._start_time = time.time()

        for epoch in range(self.dpo_config.num_epochs):
            self.current_epoch = epoch
            self.policy_model.train()
            epoch_metrics = self._create_epoch_metrics()
            avg_metrics: dict[str, float] = {}

            for batch in self.get_train_batches(train_dataset):
                self.global_step += 1
                batch = _move_batch_to_torch(batch, device)

                self.optimizer.zero_grad(set_to_none=True)
                with _torch_autocast(self.policy_model):
                    loss, metrics = self.compute_loss(batch)
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(), self.config.max_grad_norm
                    )
                self.optimizer.step()

                metric_values = _detach_metrics(metrics)
                self._accumulate_metrics(epoch_metrics, metric_values)

                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = self._compute_avg_metrics(epoch_metrics)
                    self._log_metrics(avg_metrics)
                    if callback:
                        callback(avg_metrics)

                if eval_dataset and hasattr(self.config, "eval_interval"):
                    if self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate(eval_dataset)
                        self._log_eval_metrics(eval_metrics)

                if self.global_step % self.config.checkpoint_interval == 0:
                    if avg_metrics:
                        self._save_checkpoint_if_best(avg_metrics)
                    self.save_checkpoint(f"step_{self.global_step}")

                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    logger.info(f"Reached max steps ({self.config.max_steps})")
                    break

                if avg_metrics and self._should_stop_early(avg_metrics):
                    break

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        self.save_checkpoint("final")
        logger.info(f"Training complete. Total steps: {self.global_step}")

    def evaluate(self, dataset: PreferenceDataset) -> dict[str, float]:
        """Evaluate on a dataset."""
        all_metrics = {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

        if _backend_name(self.policy_model) == "torch":
            import torch

            device = _resolve_torch_device(self.policy_model)
            self.policy_model.to(device)
            self.reference_model.to(device)
            self.policy_model.eval()
            self.reference_model.eval()
            with torch.no_grad():
                for batch in dataset.iter_batches(
                    batch_size=self.dpo_config.batch_size,
                    shuffle=False,
                    pad_token_id=self.pad_token_id,
                ):
                    batch = _move_batch_to_torch(batch, device)
                    loss, metrics = dpo_loss(
                        policy_model=self.policy_model,
                        reference_model=self.reference_model,
                        chosen_input_ids=batch["chosen_input_ids"],
                        rejected_input_ids=batch["rejected_input_ids"],
                        chosen_attention_mask=batch["chosen_attention_mask"],
                        rejected_attention_mask=batch["rejected_attention_mask"],
                        config=self.dpo_config.dpo,
                    )
                    metric_values = _detach_metrics(metrics)
                    metric_values.setdefault("loss", _metric_to_float(loss))
                    for key in all_metrics:
                        if key in metric_values:
                            all_metrics[key].append(metric_values[key])
            return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

        for batch in dataset.iter_batches(
            batch_size=self.dpo_config.batch_size,
            shuffle=False,
            pad_token_id=self.pad_token_id,
        ):
            loss, metrics = dpo_loss(
                policy_model=self.policy_model,
                reference_model=self.reference_model,
                chosen_input_ids=batch["chosen_input_ids"],
                rejected_input_ids=batch["rejected_input_ids"],
                chosen_attention_mask=batch["chosen_attention_mask"],
                rejected_attention_mask=batch["rejected_attention_mask"],
                config=self.dpo_config.dpo,
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Create DPO-specific metrics accumulator."""
        return {
            "loss": [],
            "chosen_reward": [],
            "rejected_reward": [],
            "reward_margin": [],
            "accuracy": [],
        }

    def _log_metrics(self, metrics: dict[str, float]):
        """Log DPO-specific metrics."""
        elapsed = time.time() - self._start_time
        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {metrics.get('loss', 0):.4f} | "
            f"Margin: {metrics.get('reward_margin', 0):.4f} | "
            f"Acc: {metrics.get('accuracy', 0):.2%} | "
            f"Time: {elapsed:.1f}s"
        )

        self.metrics_history.append({"step": self.global_step, **metrics})

    def _log_eval_metrics(self, metrics: dict[str, float]):
        """Log evaluation metrics."""
        logger.info(
            f"Eval | Margin: {metrics['reward_margin']:.4f} | Acc: {metrics['accuracy']:.2%}"
        )

    def _should_stop_early(self, metrics: dict[str, float]) -> bool:
        """Check if we should stop due to reaching target reward margin."""
        current_margin = metrics.get("reward_margin", 0)
        if current_margin >= self.dpo_config.target_reward_margin:
            logger.info(f"Target reward margin reached: {current_margin:.4f}")
            return True
        return False

    def save_checkpoint(self, name: str):
        """Save model checkpoint in safetensors format."""
        if _backend_name(self.policy_model) == "torch":
            checkpoint_path = Path(self.config.checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_path / f"{name}.safetensors"
            _save_torch_state(self.policy_model, weights_path)
            logger.info(f"Saved checkpoint: {weights_path}")
            return

        checkpoint_path = Path(self.config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        weights_path = checkpoint_path / f"{name}.safetensors"
        weights = dict(self.policy_model.parameters())
        mx.save_safetensors(str(weights_path), weights)
        logger.info(f"Saved checkpoint: {weights_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint from safetensors format."""
        if _backend_name(self.policy_model) == "torch":
            _load_torch_state(self.policy_model, path)
            self.policy_model.to(_resolve_torch_device(self.policy_model))
            logger.info(f"Loaded checkpoint: {path}")
            return

        weights = mx.load(path)
        self.policy_model.load_weights(list(weights.items()))
        logger.info(f"Loaded checkpoint: {path}")
