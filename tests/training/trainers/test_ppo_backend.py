"""Backend-focused tests for the PPO trainer file."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.training.trainers._backend_loader import REPO, load_repo_module


def test_ppo_no_top_level_mlx_ast() -> None:
    path = "src/chuk_lazarus/training/trainers/ppo_trainer.py"
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"mlx", "mlx_lm"}, path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"mlx", "mlx_lm"}, path


def test_ppo_trainer_uses_lazy_proxy() -> None:
    module = load_repo_module(
        "chuk_lazarus.training.trainers.ppo_trainer",
        "chuk_lazarus/training/trainers/ppo_trainer.py",
    )

    assert module.mx.__class__.__name__ == "_LazyMod"


def test_ppo_torch_rollout_and_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("CHUK_BACKEND", "torch")
    monkeypatch.delenv("CHUK_DEVICE", raising=False)
    module = load_repo_module(
        "chuk_lazarus.training.trainers.ppo_trainer",
        "chuk_lazarus/training/trainers/ppo_trainer.py",
    )

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Linear(2, 2)
            self.value_head = torch.nn.Linear(2, 1)

        def forward(self, obs, deterministic: bool = False):
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(dim=-1) if deterministic else dist.sample()
            value = self.value_head(obs).squeeze(-1)
            return {
                "action": action,
                "log_prob": dist.log_prob(action),
                "value": value,
            }

        def get_action_and_value(self, obs, action=None):
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            action = action.long()
            value = self.value_head(obs).squeeze(-1)
            return {
                "action": action,
                "log_prob": dist.log_prob(action),
                "entropy": dist.entropy(),
                "value": value,
            }

    class TinyEnv:
        def __init__(self):
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return [0.0, 0.0]

        def step(self, action):
            self.step_count += 1
            reward = 1.0 if int(action) == 0 else 0.5
            done = self.step_count >= 2
            return [float(self.step_count), 0.0], reward, done, {}

    policy = TinyPolicy()
    env = TinyEnv()
    config = module.PPOTrainerConfig(
        rollout_steps=4,
        num_epochs_per_rollout=1,
        batch_size=2,
        learning_rate=1e-2,
        checkpoint_dir=str(tmp_path),
    )
    trainer = module.PPOTrainer(policy, env, config)

    assert trainer.buffer.__class__.__name__ == "_TorchRolloutBuffer"
    final_obs = trainer._collect_rollout(env.reset())
    assert len(trainer.buffer) == 4
    metrics = trainer._ppo_update()

    assert final_obs is not None
    assert "clip_fraction" in metrics

    trainer.save_checkpoint("ppo_smoke")
    assert (tmp_path / "ppo_smoke.safetensors").exists()
