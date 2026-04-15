"""
SFT Trainer - Supervised Fine-Tuning for Tool Use.

This trainer handles the initial supervised training phase:
- Teach the model tool-calling syntax
- Learn structured output formats
- Build foundation for RL fine-tuning

Stage 1 of the Lazarus training pipeline:
    SFT (learn syntax) -> DPO (learn preferences) -> Deploy

Usage:
    # High-level API (recommended for CLI):
    result = SFTTrainer.run(SFTTrainingConfig(
        model="meta-llama/Llama-3.2-1B",
        data_path="train.jsonl",
        output_dir="./output",
    ))

    # Low-level API (for custom pipelines):
    trainer = SFTTrainer(model, tokenizer, config)
    trainer.train(dataset)
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .._lazy_mlx import mx, nn, optim
from ..base_trainer import BaseTrainer, BaseTrainerConfig
from ..losses.sft_loss import sft_loss

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core  # noqa: F401
    import mlx.nn  # noqa: F401
    import mlx.optimizers  # noqa: F401

    from ...data import SFTDataset

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


class _TorchSFTDataset:
    def __init__(self, data_path: Path, tokenizer: Any, max_length: int, mask_prompt: bool):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.samples = self._load_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def _load_samples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self.data_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if "messages" in item:
                    prompt_parts = []
                    response = ""
                    for msg in item["messages"]:
                        role = msg.get("role")
                        content = msg.get("content", "")
                        if role in {"user", "system"}:
                            prompt_parts.append(content)
                        elif role == "assistant":
                            response = content
                    prompt = "\n".join(prompt_parts).strip()
                else:
                    prompt = str(item.get("prompt", item.get("input", ""))).strip()
                    response = str(
                        item.get("response", item.get("output", item.get("completion", "")))
                    ).strip()
                rows.append({"prompt": prompt, "response": response})
        return rows

    def _tokenize(self, sample: dict[str, Any]) -> dict[str, Any]:
        prompt_tokens = _encode_text(self.tokenizer, sample["prompt"])
        full_tokens = _encode_text(self.tokenizer, sample["prompt"] + sample["response"])
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[: self.max_length]
        eos_token_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
        labels = full_tokens[1:] + [eos_token_id]
        if self.mask_prompt:
            prompt_len = min(len(prompt_tokens), len(labels))
            loss_mask = [0.0] * prompt_len + [1.0] * (len(labels) - prompt_len)
        else:
            loss_mask = [1.0] * len(labels)
        return {
            "input_ids": full_tokens,
            "labels": labels,
            "loss_mask": loss_mask,
        }

    def iter_batches(
        self, batch_size: int, shuffle: bool = True, pad_token_id: int = 0
    ) -> Iterator[dict[str, Any]]:
        import random
        import torch

        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            items = [self._tokenize(self.samples[idx]) for idx in batch_indices]
            max_len = max(len(item["input_ids"]) for item in items)

            input_ids = []
            labels = []
            loss_mask = []
            attention_mask = []
            for item in items:
                seq_len = len(item["input_ids"])
                pad_len = max_len - seq_len
                input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
                labels.append(item["labels"] + [pad_token_id] * pad_len)
                loss_mask.append(item["loss_mask"] + [0.0] * pad_len)
                attention_mask.append([1.0] * seq_len + [0.0] * pad_len)

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            }


class SFTTrainingConfig(BaseModel):
    """Complete configuration for running SFT training.

    This is the high-level config used by CLI and run() method.
    Includes model path, data paths, and all training parameters.
    """

    # Model
    model: str = Field(..., description="Model path or HuggingFace name")
    use_lora: bool = Field(default=False, description="Use LoRA adapters")
    lora_rank: int = Field(default=8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(default=16.0, description="LoRA alpha scaling")
    lora_targets: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="LoRA target modules",
    )

    # Data
    data_path: Path = Field(..., description="Path to training data (JSONL)")
    eval_data_path: Path | None = Field(default=None, description="Path to eval data")
    max_length: int = Field(default=512, ge=1, description="Max sequence length")
    mask_prompt: bool = Field(default=False, description="Mask prompt in loss")

    # Training
    num_epochs: int = Field(default=3, ge=1, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    max_steps: int | None = Field(default=None, description="Max steps (overrides epochs)")

    # Output
    output_dir: Path = Field(default=Path("./checkpoints/sft"), description="Output directory")
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")


class SFTTrainingResult(BaseModel):
    """Result of SFT training."""

    output_dir: Path = Field(..., description="Output directory")
    epochs_completed: int = Field(..., description="Epochs completed")
    final_loss: float | None = Field(default=None, description="Final training loss")
    adapter_path: Path | None = Field(default=None, description="Path to saved LoRA adapter")


class SFTConfig(BaseTrainerConfig):
    """Low-level configuration for SFT trainer.

    Used internally by the trainer. For high-level usage, see SFTTrainingConfig.
    """

    # Training settings
    num_epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-5, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.0, ge=0, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, gt=0, description="Maximum gradient norm")
    warmup_steps: int = Field(default=100, ge=0, description="Warmup steps")
    # Logging and checkpoints
    log_interval: int = Field(default=10, ge=1, description="Log interval")
    eval_interval: int = Field(default=100, ge=1, description="Evaluation interval")
    checkpoint_interval: int = Field(default=500, ge=1, description="Checkpoint interval")
    checkpoint_dir: str = Field(default="./checkpoints/sft", description="Checkpoint directory")
    # Early stopping
    max_steps: int | None = Field(default=None, description="Maximum training steps")
    min_loss: float | None = Field(
        default=None, description="Target minimum loss for early stopping"
    )


class SFTTrainer(BaseTrainer):
    """
    Trainer for Supervised Fine-Tuning.

    This is Stage 1 of the Lazarus training pipeline:
    - Teaches the model tool-calling syntax
    - Learns structured output formats
    - Prepares for DPO preference learning

    Usage:
        # High-level API (recommended):
        result = SFTTrainer.run(SFTTrainingConfig(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            data_path="train.jsonl",
        ))

        # Low-level API:
        trainer = SFTTrainer(model, tokenizer, config)
        trainer.train(dataset)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: SFTConfig = None,
        optimizer: optim.Optimizer = None,
    ):
        if optimizer is None and _is_torch_model(model):
            os.environ.setdefault("CHUK_BACKEND", "torch")
        config = config or SFTConfig()
        super().__init__(model, tokenizer, config, optimizer)

    @classmethod
    def run(cls, config: SFTTrainingConfig) -> SFTTrainingResult:
        """Run complete SFT training from config.

        This is the high-level entry point that handles:
        - Model loading (with optional LoRA)
        - Dataset loading
        - Training
        - Checkpoint saving

        Args:
            config: Complete training configuration

        Returns:
            SFTTrainingResult with training outcomes
        """
        backend = _backend_name()
        if backend == "torch" and config.use_lora:
            raise RuntimeError(
                "SFTTrainer.run with CHUK_BACKEND=torch and use_lora=True is still blocked "
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
                "SFTTrainer.run low-level torch training is ready, but high-level model "
                "loading is still blocked by out-of-scope models_v2 eager MLX imports "
                "(notably models_v2/__init__.py, models/base.py, and loader.py)."
            ) from exc

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading model: {config.model}")
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
            model = result.model
            tokenizer = result.tokenizer
            lora_layers = result.lora_layers
            logger.info(
                f"  Loaded with LoRA: {len(lora_layers)} layers, "
                f"{result.lora_parameter_count:,} trainable params"
            )
        else:
            result = load_model(config.model)
            model = result.model
            tokenizer = result.tokenizer

        # Load datasets
        logger.info(f"Loading dataset: {config.data_path}")
        if backend == "torch":
            train_dataset = _TorchSFTDataset(
                config.data_path,
                tokenizer,
                max_length=config.max_length,
                mask_prompt=config.mask_prompt,
            )
        else:
            from ...data import SFTDataset

            train_dataset = SFTDataset(
                str(config.data_path),
                tokenizer,
                max_length=config.max_length,
                mask_prompt=config.mask_prompt,
            )
        logger.info(f"  Loaded {len(train_dataset)} training samples")

        eval_dataset = None
        if config.eval_data_path:
            if backend == "torch":
                eval_dataset = _TorchSFTDataset(
                    config.eval_data_path,
                    tokenizer,
                    max_length=config.max_length,
                    mask_prompt=config.mask_prompt,
                )
            else:
                eval_dataset = SFTDataset(
                    str(config.eval_data_path),
                    tokenizer,
                    max_length=config.max_length,
                    mask_prompt=config.mask_prompt,
                )
            logger.info(f"  Loaded {len(eval_dataset)} eval samples")

        # Create trainer config
        trainer_config = SFTConfig(
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            checkpoint_dir=str(output_dir / "checkpoints"),
            log_interval=config.log_interval,
            max_steps=config.max_steps,
            checkpoint_interval=config.checkpoint_interval,
        )

        # Create and run trainer
        trainer = cls(model, tokenizer, trainer_config)

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

        return SFTTrainingResult(
            output_dir=output_dir,
            epochs_completed=config.num_epochs,
            final_loss=final_loss,
            adapter_path=adapter_path,
        )

    @property
    def sft_config(self) -> SFTConfig:
        """Type-safe access to config."""
        return self.config

    def compute_loss(self, batch: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
        """Compute SFT loss for a batch."""
        output = self.model(batch["input_ids"])
        # Handle different model output formats:
        # - Some models return just logits (mlx-lm models)
        # - Some return (logits, cache) tuple
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        loss, metrics = sft_loss(
            logits=logits, labels=batch["labels"], loss_mask=batch["loss_mask"]
        )
        return loss, metrics

    def get_train_batches(self, dataset: SFTDataset) -> Iterator[dict[str, mx.array]]:
        """Get iterator over training batches."""
        return dataset.iter_batches(
            batch_size=self.sft_config.batch_size,
            shuffle=True,
            pad_token_id=self.pad_token_id,
        )

    def train(
        self,
        train_dataset: SFTDataset,
        eval_dataset: SFTDataset = None,
        callback: Callable[[dict], None] = None,
    ):
        """
        Run SFT training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            callback: Optional callback after each log interval
        """
        logger.info(f"Starting SFT training with {len(train_dataset)} samples")
        if _backend_name(self.model) != "torch":
            super().train(
                dataset=train_dataset,
                num_epochs=self.sft_config.num_epochs,
                eval_dataset=eval_dataset,
                callback=callback,
            )
            return

        import torch

        device = _resolve_torch_device(self.model)
        self.model.to(device)
        self._start_time = __import__("time").time()

        for epoch in range(self.sft_config.num_epochs):
            self.current_epoch = epoch
            self.model.train()
            epoch_metrics = self._create_epoch_metrics()
            avg_metrics: dict[str, float] = {}

            for batch in self.get_train_batches(train_dataset):
                self.global_step += 1
                batch = _move_batch_to_torch(batch, device)

                self.optimizer.zero_grad(set_to_none=True)
                with _torch_autocast(self.model):
                    loss, metrics = self.compute_loss(batch)
                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
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

    def evaluate(self, dataset: SFTDataset) -> dict[str, float]:
        """Evaluate on a dataset."""
        all_metrics = {"loss": [], "perplexity": [], "num_tokens": []}

        if _backend_name(self.model) == "torch":
            import torch

            device = _resolve_torch_device(self.model)
            self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                for batch in dataset.iter_batches(
                    batch_size=self.sft_config.batch_size,
                    shuffle=False,
                    pad_token_id=self.pad_token_id,
                ):
                    batch = _move_batch_to_torch(batch, device)
                    output = self.model(batch["input_ids"])
                    logits = output[0] if isinstance(output, tuple) else output
                    loss, metrics = sft_loss(
                        logits=logits,
                        labels=batch["labels"],
                        loss_mask=batch["loss_mask"],
                    )
                    metric_values = _detach_metrics(metrics)
                    for key in all_metrics:
                        if key in metric_values:
                            all_metrics[key].append(metric_values[key])
            return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

        for batch in dataset.iter_batches(
            batch_size=self.sft_config.batch_size,
            shuffle=False,
            pad_token_id=self.pad_token_id,
        ):
            output = self.model(batch["input_ids"])
            # Handle different model output formats
            logits = output[0] if isinstance(output, tuple) else output
            loss, metrics = sft_loss(
                logits=logits, labels=batch["labels"], loss_mask=batch["loss_mask"]
            )

            for key in all_metrics:
                if key in metrics:
                    all_metrics[key].append(float(metrics[key]))

        return {k: sum(v) / len(v) if v else 0.0 for k, v in all_metrics.items()}

    def _create_epoch_metrics(self) -> dict[str, list[float]]:
        """Create SFT-specific metrics accumulator."""
        return {
            "loss": [],
            "perplexity": [],
            "num_tokens": [],
        }

    def _log_metrics(self, metrics: dict[str, float]):
        """Log SFT-specific metrics."""
        import time

        elapsed = time.time() - self._start_time

        # Calculate tokens per second if we have token counts
        epoch_metrics = getattr(self, "_current_epoch_metrics", {})
        total_tokens = sum(epoch_metrics.get("num_tokens", [0]))
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {metrics.get('loss', 0):.4f} | "
            f"PPL: {metrics.get('perplexity', 0):.2f} | "
            f"Tok/s: {tokens_per_sec:.0f} | "
            f"Time: {elapsed:.1f}s"
        )

        self.metrics_history.append(
            {"step": self.global_step, "epoch": self.current_epoch, **metrics}
        )

    def _log_eval_metrics(self, metrics: dict[str, float]):
        """Log evaluation metrics."""
        logger.info(f"Eval | Loss: {metrics['loss']:.4f} | PPL: {metrics['perplexity']:.2f}")

    def _should_stop_early(self, metrics: dict[str, float]) -> bool:
        """Check if we should stop due to reaching target loss."""
        if self.sft_config.min_loss is not None:
            if metrics.get("loss", float("inf")) < self.sft_config.min_loss:
                logger.info(f"Reached target loss: {metrics['loss']:.4f}")
                return True
        return False

    def save_checkpoint(self, name: str):
        if _backend_name(self.model) != "torch":
            return super().save_checkpoint(name)

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weights_path = checkpoint_dir / f"{name}.safetensors"
        _save_torch_state(self.model, weights_path)
        logger.info(f"Saved checkpoint: {weights_path}")

    def load_checkpoint(self, path: str):
        if _backend_name(self.model) != "torch":
            return super().load_checkpoint(path)

        _load_torch_state(self.model, path)
        self.model.to(_resolve_torch_device(self.model))
        logger.info(f"Loaded checkpoint: {path}")
