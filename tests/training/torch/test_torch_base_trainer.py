"""CPU-only tests for ``TorchBaseTrainer`` and ``TorchTrainerConfig``.

Covers config defaults, checkpoint round-trip, gradient clipping on a
degenerate batch, and the optional cosine LR scheduler. Uses real torch
tensors throughout -- no mocks for tensor ops.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from chuk_lazarus.training.torch import (
    TorchBaseTrainer,
    TorchClassificationTrainer,
    TorchTrainerConfig,
)


def _load_torch_mlp_class():
    """Load ``TorchMLPClassifier`` directly from its source file.

    The ``classifiers`` package ``__init__.py`` imports MLX-dependent siblings,
    so we bypass it for this torch-only test.
    """
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
        "chuk_lazarus_torch_mlp_trainer_base", src
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TorchMLPClassifier


TorchMLPClassifier = _load_torch_mlp_class()


class _MinimalTrainer(TorchBaseTrainer):
    """Concrete trainer used to exercise base-class hooks in isolation."""

    def compute_loss(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        x = batch["input"]
        y = batch["label"]
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        return loss, {"loss": float(loss.detach().item())}

    def get_train_batches(self, dataset: Any) -> Iterator[dict[str, torch.Tensor]]:
        for batch in dataset:
            yield batch


def _make_separable_batch(batch_size: int = 32) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    half = batch_size // 2
    cls0 = torch.randn(half, 2, generator=g) + torch.tensor([-5.0, -5.0])
    cls1 = torch.randn(batch_size - half, 2, generator=g) + torch.tensor([5.0, 5.0])
    features = torch.cat([cls0, cls1], dim=0)
    labels = torch.cat(
        [torch.zeros(half, dtype=torch.long), torch.ones(batch_size - half, dtype=torch.long)]
    )
    return {"input": features, "label": labels}


class TestTorchTrainerConfig:
    def test_default_values_match_spec(self) -> None:
        cfg = TorchTrainerConfig()
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.weight_decay == 0.0
        assert cfg.max_grad_norm == 1.0
        assert cfg.batch_size == 32
        assert cfg.num_epochs == 1
        assert cfg.log_interval == 10
        assert cfg.checkpoint_interval == 500
        assert cfg.checkpoint_dir == "./checkpoints"
        assert cfg.device == "cpu"
        assert cfg.cosine_schedule is False


class TestCheckpointRoundTrip:
    def test_save_and_load_restores_parameters(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model_a = TorchMLPClassifier(input_size=2, hidden_size=8, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=1e-2,
            batch_size=16,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer_a = TorchClassificationTrainer(model_a, encoder=None, config=cfg)

        batch = _make_separable_batch(batch_size=16)
        for _ in range(5):
            trainer_a.optimizer.zero_grad(set_to_none=True)
            loss, _ = trainer_a.compute_loss(batch)
            loss.backward()
            trainer_a.optimizer.step()
            trainer_a.global_step += 1

        ckpt_path = trainer_a.save_checkpoint("mid_train")
        assert ckpt_path.exists()
        assert ckpt_path.suffix == ".pt"
        assert ckpt_path.parent == tmp_path

        # Pin the spec contract: payload schema is exactly {state_dict, config}
        # and deserialises under the safer weights_only=True path.
        raw_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert isinstance(raw_payload, dict)
        assert set(raw_payload.keys()) == {"state_dict", "config"}
        assert isinstance(raw_payload["config"], dict)
        # ``config`` must be the serialised dataclass, not the dataclass itself.
        assert raw_payload["config"]["learning_rate"] == pytest.approx(cfg.learning_rate)
        assert raw_payload["config"]["batch_size"] == cfg.batch_size
        assert raw_payload["config"]["device"] == cfg.device

        torch.manual_seed(42)  # different seed -> different fresh weights
        model_b = TorchMLPClassifier(input_size=2, hidden_size=8, num_labels=2)
        trainer_b = TorchClassificationTrainer(model_b, encoder=None, config=cfg)

        # Sanity: pre-load weights differ so the load is actually doing work.
        params_before = {k: v.detach().clone() for k, v in trainer_b.model.named_parameters()}
        any_differ = any(
            not torch.allclose(params_before[k], p)
            for k, p in trainer_a.model.named_parameters()
        )
        assert any_differ, "fresh model unexpectedly matched trained model before load"

        trainer_b.load_checkpoint(ckpt_path)
        for (name_a, p_a), (name_b, p_b) in zip(
            trainer_a.model.named_parameters(), trainer_b.model.named_parameters()
        ):
            assert name_a == name_b
            assert torch.allclose(p_a, p_b, atol=0.0, rtol=0.0), f"param {name_a} mismatch"

    def test_load_rejects_malformed_payload(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=8, num_labels=2)
        cfg = TorchTrainerConfig(
            batch_size=4,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        trainer = TorchClassificationTrainer(model, encoder=None, config=cfg)
        # Bare state_dict payload (missing the {"state_dict": ...} wrapper) must
        # now raise cleanly instead of silently succeeding via a fallback.
        bad_path = tmp_path / "bad.pt"
        torch.save(model.state_dict(), bad_path)
        with pytest.raises(ValueError, match="missing required keys"):
            trainer.load_checkpoint(bad_path)


class TestGradClip:
    def test_clipped_global_norm_bounded_by_max_norm(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=8, num_labels=2)
        max_norm = 0.5
        cfg = TorchTrainerConfig(
            learning_rate=1e-3,
            max_grad_norm=max_norm,
            batch_size=4,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )

        # Degenerate batch: huge inputs + a wrong label yields exploding gradients.
        features = torch.full((4, 2), fill_value=1e6, dtype=torch.float32)
        labels = torch.zeros(4, dtype=torch.long)
        batch = {"input": features, "label": labels}

        trainer = _MinimalTrainer(model, cfg)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, _ = trainer.compute_loss(batch)
        loss.backward()

        # Pre-clip global norm should be massive.
        pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e18)
        assert float(pre_clip) > 1e3, f"expected exploding grads, got {float(pre_clip)}"

        # Re-run to produce the same huge gradients then clip legitimately.
        trainer.optimizer.zero_grad(set_to_none=True)
        loss, _ = trainer.compute_loss(batch)
        loss.backward()
        trainer.clip_gradients(max_norm)

        squared = 0.0
        for p in model.parameters():
            if p.grad is not None:
                squared += float(p.grad.detach().pow(2).sum().item())
        post_norm = squared**0.5
        assert post_norm <= max_norm + 1e-6, f"post-clip norm {post_norm} > {max_norm}"

    def test_grad_clip_prevents_nan_through_train_loop(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: a deterministic exploding-gradient batch fed through
        ``trainer.train()`` must not emit NaN/Inf parameters -- proves
        clip-before-step ordering in the real loop with max_grad_norm=1.0.
        """
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=8, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=1e-1,
            max_grad_norm=1.0,
            batch_size=4,
            num_epochs=1,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        # Single deterministic exploding-gradient batch wrapped in a single-item
        # list dataset (as required by the review spec).
        exploding_batch = {
            "input": torch.full((4, 2), fill_value=1e6, dtype=torch.float32),
            "label": torch.zeros(4, dtype=torch.long),
        }
        dataset = [exploding_batch]

        trainer = _MinimalTrainer(model, cfg)
        trainer.train(dataset, num_epochs=1)

        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), f"param {name} went non-finite (clip missed)"


class TestTrainLoop:
    def test_train_steps_and_history(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=1e-2,
            batch_size=8,
            num_epochs=2,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
        )
        batches = [_make_separable_batch(batch_size=8) for _ in range(3)]
        trainer = _MinimalTrainer(model, cfg)

        history = trainer.train(batches)
        assert len(history) == 2 * len(batches)
        assert trainer.global_step == 2 * len(batches)
        assert all("loss" in entry and "step" in entry for entry in history)

    def test_cosine_schedule_reduces_lr(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=1.0,
            batch_size=8,
            num_epochs=4,
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
            cosine_schedule=True,
        )
        batches = [_make_separable_batch(batch_size=8)]
        trainer = _MinimalTrainer(model, cfg)
        start_lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.train(batches)
        end_lr = trainer.optimizer.param_groups[0]["lr"]
        assert end_lr < start_lr, f"cosine schedule should lower LR: {start_lr}->{end_lr}"

    def test_cosine_t_max_tracks_effective_num_epochs(self, tmp_path: Path) -> None:
        """train(num_epochs=X) must take precedence over config.num_epochs=1
        when sizing cosine T_max -- otherwise the scheduler would collapse LR
        to eta_min after a single step.
        """
        torch.manual_seed(0)
        model = TorchMLPClassifier(input_size=2, hidden_size=4, num_labels=2)
        cfg = TorchTrainerConfig(
            learning_rate=1.0,
            batch_size=8,
            num_epochs=1,  # misleading default
            log_interval=100,
            checkpoint_interval=100,
            checkpoint_dir=str(tmp_path),
            cosine_schedule=True,
        )
        batches = [_make_separable_batch(batch_size=8)]
        trainer = _MinimalTrainer(model, cfg)
        start_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train with an effective num_epochs of 4 -- scheduler must be built
        # with T_max=4, not T_max=1.
        trainer.train(batches, num_epochs=4)

        assert trainer.scheduler is not None
        assert trainer.scheduler.T_max == 4, (
            f"expected cosine T_max to track effective num_epochs=4, "
            f"got {trainer.scheduler.T_max}"
        )
        end_lr = trainer.optimizer.param_groups[0]["lr"]
        assert 0.0 <= end_lr < start_lr
