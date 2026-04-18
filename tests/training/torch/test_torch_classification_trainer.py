"""CPU-only tests for ``TorchClassificationTrainer``.

Covers the spec smoke target (>=0.99 final-batch accuracy on a
linearly-separable 2-D synthetic dataset), the encoder dispatch paths
(features vs. text+encoder), and the metrics contract of ``compute_loss``.
"""

from __future__ import annotations

import importlib.util
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from chuk_lazarus.training.torch import (
    TorchClassificationTrainer,
    TorchTrainerConfig,
)


def _load_torch_mlp_class():
    """Bypass classifiers package ``__init__`` (which imports MLX)."""
    src = (
        Path(__file__).resolve().parents[2]
        / ".."
        / "src"
        / "chuk_lazarus"
        / "models_v2"
        / "models"
        / "classifiers"
        / "torch_mlp.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "chuk_lazarus_torch_mlp_trainer_cls", src
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TorchMLPClassifier


TorchMLPClassifier = _load_torch_mlp_class()


@dataclass
class _Sample:
    """Minimal sample object with attribute access used by the trainer."""

    label: int
    features: list[float] | None = None
    text: str | None = None


def _make_blobs(
    n_per_class: int = 160, seed: int = 0
) -> list[_Sample]:
    """Two linearly-separable 2-D gaussian blobs (numpy only, no sklearn)."""
    rng = np.random.default_rng(seed)
    class0 = rng.normal(loc=(-3.0, -3.0), scale=0.5, size=(n_per_class, 2))
    class1 = rng.normal(loc=(3.0, 3.0), scale=0.5, size=(n_per_class, 2))
    samples: list[_Sample] = []
    for row in class0:
        samples.append(_Sample(label=0, features=[float(row[0]), float(row[1])]))
    for row in class1:
        samples.append(_Sample(label=1, features=[float(row[0]), float(row[1])]))
    return samples


class TestSmokeTrainsToHighAccuracy:
    """Train a tiny MLP on separable blobs and demand >=0.99 final batch accuracy."""

    def test_smoke_accuracy_threshold(self, tmp_path: Path) -> None:
        # ``get_train_batches`` shuffles via the stdlib ``random`` module, so
        # determinism requires seeding torch, numpy, and stdlib RNGs.
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        n_per_class = 160
        batch_size = 32
        num_epochs = 6
        steps_per_epoch = n_per_class * 2 // batch_size  # 10

        dataset = _make_blobs(n_per_class=n_per_class, seed=0)
        assert len(dataset) == n_per_class * 2

        model = TorchMLPClassifier(input_size=2, hidden_size=16, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=5e-2,
            batch_size=batch_size,
            num_epochs=num_epochs,  # 10 steps/epoch * 6 = 60 steps (spec target ~50)
            log_interval=100,
            checkpoint_interval=10_000,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=None, config=cfg)

        history = trainer.train(dataset)
        assert history, "train() returned empty history"
        assert trainer.global_step >= 50

        # Mean accuracy across the full final epoch, not a single shuffled batch.
        final_acc = statistics.mean(h["accuracy"] for h in history[-steps_per_epoch:])
        assert final_acc >= 0.99, f"final-epoch mean accuracy {final_acc:.4f} < 0.99"


class TestEncoderDispatch:
    """Trainer must accept features-only samples and text+encoder samples."""

    def test_features_path_without_encoder(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        dataset = [
            _Sample(label=0, features=[0.0, 0.0]),
            _Sample(label=1, features=[1.0, 1.0]),
            _Sample(label=0, features=[0.1, -0.1]),
            _Sample(label=1, features=[0.9, 1.1]),
        ]
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=2,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=None, config=cfg)

        batches = list(trainer.get_train_batches(dataset))
        assert batches, "no batches produced"
        for batch in batches:
            assert batch["input"].dtype == torch.float32
            assert batch["input"].shape[1] == 2
            assert batch["label"].dtype == torch.long

        history = trainer.train(dataset)
        assert history

    def test_text_plus_callable_encoder(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        dataset = [
            _Sample(label=0, text="neg"),
            _Sample(label=1, text="pos"),
            _Sample(label=0, text="neg"),
            _Sample(label=1, text="pos"),
        ]

        def encode(text: str) -> list[float]:
            return [1.0, 0.0] if text == "neg" else [0.0, 1.0]

        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=2,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=encode, config=cfg)
        batches = list(trainer.get_train_batches(dataset))
        assert batches
        first = batches[0]
        assert first["input"].shape[1] == 2

    def test_text_plus_encoder_object_with_encode_method(self, tmp_path: Path) -> None:
        torch.manual_seed(0)

        class StubEncoder:
            def encode(self, text: str) -> list[float]:
                return [float(len(text)), 0.0]

        dataset = [
            _Sample(label=0, text="a"),
            _Sample(label=1, text="abc"),
        ]
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=2,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=StubEncoder(), config=cfg)
        batches = list(trainer.get_train_batches(dataset))
        assert batches
        inputs = batches[0]["input"]
        assert inputs.shape == (2, 2)
        # Rows reflect encode(text) = [len(text), 0.0]
        flat_values = sorted(float(v) for v in inputs[:, 0].tolist())
        assert flat_values == [1.0, 3.0]

    def test_text_without_encoder_raises(self, tmp_path: Path) -> None:
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=1,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=None, config=cfg)
        dataset = [_Sample(label=0, text="no-encoder")]
        with pytest.raises(ValueError):
            list(trainer.get_train_batches(dataset))


class TestComputeLossContract:
    """``compute_loss`` must return a tensor loss and a dict of float metrics."""

    def test_returns_tensor_and_float_metrics(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=4,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=None, config=cfg)

        batch = {
            "input": torch.tensor(
                [[-1.0, -1.0], [-1.0, -1.1], [1.0, 1.0], [1.1, 1.0]],
                dtype=torch.float32,
            ),
            "label": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        }
        loss, metrics = trainer.compute_loss(batch)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.requires_grad  # gradient flows through cross-entropy
        assert set(metrics.keys()) == {"loss", "accuracy"}
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0
