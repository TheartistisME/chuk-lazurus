"""Torch-native base trainer.

Mirrors the MLX ``BaseTrainer`` / ``ClassificationTrainer`` contract without
sharing code or importing any MLX module. This file must remain free of MLX
imports so the torch subpackage is usable on hosts without MLX.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class TorchTrainerConfig:
    """Configuration for torch classification trainers."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_size: int = 32
    num_epochs: int = 1
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    device: str = "cpu"
    cosine_schedule: bool = False


class TorchBaseTrainer(ABC):
    """Abstract torch trainer with AdamW, optional cosine LR, grad clipping,
    and ``.pt`` checkpointing.

    Subclasses implement :meth:`compute_loss` and :meth:`get_train_batches`.
    """

    def __init__(self, model: torch.nn.Module, config: TorchTrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler is built lazily inside ``train()`` once the effective
        # number of epochs is known; constructing it here with
        # ``config.num_epochs`` would desync when the caller passes a different
        # ``num_epochs`` to ``train()`` (and collapse to LR=0 after step 1 when
        # config.num_epochs=1).
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        self.global_step: int = 0
        self.current_epoch: int = 0
        self.metrics_history: list[dict[str, float]] = []
        self._start_time: float | None = None

    @abstractmethod
    def compute_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(loss_tensor, metrics_dict)`` for a batch."""

    @abstractmethod
    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, Any]]:
        """Yield training batches from ``dataset``."""

    def clip_gradients(self, max_norm: float) -> None:
        """In-place global gradient clipping via ``torch.nn.utils.clip_grad_norm_``."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def save_checkpoint(self, name: str) -> Path:
        """Write ``{checkpoint_dir}/{name}.pt`` containing state_dict + config."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{name}.pt"
        payload = {
            "state_dict": self.model.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, path)
        logger.info("Saved checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint written by :meth:`save_checkpoint` into ``self.model``.

        Uses ``weights_only=True`` -- the checkpoint payload is locked to
        ``{"state_dict": tensors, "config": primitives}`` by
        :meth:`save_checkpoint`, so no arbitrary-object deserialisation is
        needed. A malformed payload raises ``ValueError``.
        """
        payload = torch.load(Path(path), map_location=self.device, weights_only=True)
        if (
            not isinstance(payload, dict)
            or "state_dict" not in payload
            or "config" not in payload
        ):
            raise ValueError(
                "checkpoint payload missing required keys {'state_dict','config'}"
            )
        self.model.load_state_dict(payload["state_dict"])
        logger.info("Loaded checkpoint: %s", path)

    def train(
        self,
        dataset: Any,
        num_epochs: int | None = None,
        callback: Callable[[dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run the training loop and return per-step metric entries."""
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Build (or rebuild) the cosine scheduler with the effective epoch
        # count so T_max always matches the loop about to run.
        if self.config.cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, epochs)
            )

        self.model.train()
        self._start_time = time.time()

        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            self.current_epoch = epoch
            logger.info("Epoch %d/%d", epoch + 1, epochs)

            for batch in self.get_train_batches(dataset):
                self.global_step += 1

                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.compute_loss(batch)
                loss.backward()

                if self.config.max_grad_norm > 0:
                    self.clip_gradients(self.config.max_grad_norm)

                self.optimizer.step()

                entry: dict[str, float] = {
                    "step": float(self.global_step),
                    "epoch": float(epoch),
                    **{k: float(v) for k, v in metrics.items()},
                }
                history.append(entry)
                self.metrics_history.append(entry)

                if self.global_step % self.config.log_interval == 0:
                    elapsed = time.time() - (self._start_time or time.time())
                    logger.info(
                        "Step %d | Loss: %.4f | Time: %.1fs",
                        self.global_step,
                        entry.get("loss", 0.0),
                        elapsed,
                    )
                    if callback is not None:
                        callback(entry)

                if self.global_step % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            if self.scheduler is not None:
                self.scheduler.step()

        return history
