"""Backend-focused tests for the GRPO trainer and CLI-owned GRPO files."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.training.trainers._backend_loader import (
    REPO,
    TinyTokenizer,
    TinyTorchLM,
    clone_tiny_lm,
    load_repo_module,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/chuk_lazarus/training/trainers/grpo_trainer.py",
        "src/chuk_lazarus/cli/commands/train/grpo.py",
        "src/chuk_lazarus/cli/_parsers/_train_rlhf.py",
    ],
)
def test_grpo_no_top_level_mlx_ast(path: str) -> None:
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"mlx", "mlx_lm"}, path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"mlx", "mlx_lm"}, path


def test_grpo_trainer_uses_lazy_proxy() -> None:
    module = load_repo_module(
        "chuk_lazarus.training.trainers.grpo_trainer",
        "chuk_lazarus/training/trainers/grpo_trainer.py",
    )

    assert module.mx.__class__.__name__ == "_LazyMod"


def test_grpo_torch_train_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    torch.manual_seed(42)
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    monkeypatch.delenv("CHUK_DEVICE", raising=False)
    module = load_repo_module(
        "chuk_lazarus.training.trainers.grpo_trainer",
        "chuk_lazarus/training/trainers/grpo_trainer.py",
    )

    policy_model = TinyTorchLM(vocab_size=32, hidden_size=16)
    reference_model = clone_tiny_lm(policy_model)
    tokenizer = TinyTokenizer()
    config = module.GRPOTrainerConfig(
        grpo=module.GRPOConfig(group_size=2, kl_coef=0.05),
        num_iterations=2,
        prompts_per_iteration=2,
        learning_rate=5e-2,
        max_response_length=2,
        log_interval=1,
        checkpoint_interval=10,
        checkpoint_dir=str(tmp_path),
    )

    def reward_fn(prompt: str, response: str) -> float:
        return float(len(response) >= 1)

    trainer = module.GRPOTrainer(
        policy_model,
        reference_model,
        tokenizer,
        reward_fn,
        config,
    )
    before = {key: value.detach().clone() for key, value in policy_model.state_dict().items()}
    trainer.train(lambda: ["ab", "cd"])

    assert trainer.metrics_history
    assert (tmp_path / "final.safetensors").exists()
    assert any(
        not torch.equal(before[key], value)
        for key, value in policy_model.state_dict().items()
    )
