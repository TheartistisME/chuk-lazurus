"""Torch-native classification trainer.

Cross-entropy trainer that mirrors the MLX ``ClassificationTrainer`` contract
using real torch tensors only. Samples may expose ``features`` (a list of
floats) or ``text`` combined with an encoder callable; encoders are either
callables returning ``list[float]`` or objects exposing an ``encode`` method.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.nn import functional as F

from chuk_lazarus.training.torch.torch_base_trainer import (
    TorchBaseTrainer,
    TorchTrainerConfig,
)

EncoderType = Callable[[str], list[float]] | Any


def _get_field(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class TorchClassificationTrainer(TorchBaseTrainer):
    """Cross-entropy classification trainer with accuracy tracking."""

    def __init__(
        self,
        model: torch.nn.Module,
        encoder: EncoderType | None,
        config: TorchTrainerConfig,
    ) -> None:
        super().__init__(model, config)
        self.encoder = encoder

    def compute_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch["input"].to(self.device)
        y = batch["label"].to(self.device)
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == y).float().mean()
        return loss, {
            "loss": float(loss.detach().item()),
            "accuracy": float(accuracy.detach().item()),
        }

    def _encode(self, text: str) -> list[float]:
        if self.encoder is None:
            raise ValueError("Sample has 'text' but no encoder was provided")
        encoder = self.encoder
        if hasattr(encoder, "encode"):
            return list(encoder.encode(text))
        return list(encoder(text))  # type: ignore[misc]

    def _get_features(self, sample: Any) -> Any:
        features = _get_field(sample, "features")
        if features is not None:
            return features
        text = _get_field(sample, "text")
        if text is not None:
            return [float(v) for v in self._encode(text)]
        raise ValueError("Sample must have 'features' or 'text' (with encoder)")

    def _get_label(self, sample: Any) -> int:
        label = _get_field(sample, "label")
        if label is None:
            raise ValueError("Sample is missing required 'label' field")
        return int(label)

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, torch.Tensor]]:
        batch_size = self.config.batch_size
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        device = self.device

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            samples = [dataset[j] for j in batch_indices]
            features = [self._get_features(s) for s in samples]
            labels = [self._get_label(s) for s in samples]
            first_feature = features[0]
            if isinstance(first_feature, torch.Tensor):
                inputs = torch.stack(
                    [feature.to(dtype=torch.float32, device=device) for feature in features]
                )
            else:
                inputs = torch.tensor(features, dtype=torch.float32, device=device)
            targets = torch.tensor(labels, dtype=torch.long, device=device)
            yield {"input": inputs, "label": targets}
