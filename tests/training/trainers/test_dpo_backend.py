"""Backend-focused tests for the DPO trainer and CLI-owned DPO files."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.training.trainers._backend_loader import (
    REPO,
    StaticBatchDataset,
    TinyTokenizer,
    TinyTorchLM,
    clone_tiny_lm,
    load_repo_module,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/chuk_lazarus/training/trainers/dpo_trainer.py",
        "src/chuk_lazarus/cli/commands/train/dpo.py",
        "src/chuk_lazarus/cli/_parsers/_train_rlhf.py",
    ],
)
def test_dpo_no_top_level_mlx_ast(path: str) -> None:
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"mlx", "mlx_lm"}, path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"mlx", "mlx_lm"}, path


def test_dpo_trainer_uses_lazy_proxy() -> None:
    module = load_repo_module(
        "chuk_lazarus.training.trainers.dpo_trainer",
        "chuk_lazarus/training/trainers/dpo_trainer.py",
    )

    assert module.mx.__class__.__name__ == "_LazyMod"


def test_dpo_torch_train_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("CHUK_BACKEND", "torch")
    monkeypatch.delenv("CHUK_DEVICE", raising=False)
    module = load_repo_module(
        "chuk_lazarus.training.trainers.dpo_trainer",
        "chuk_lazarus/training/trainers/dpo_trainer.py",
    )

    policy_model = TinyTorchLM(vocab_size=32, hidden_size=16)
    reference_model = clone_tiny_lm(policy_model)
    tokenizer = TinyTokenizer()
    config = module.DPOTrainerConfig(
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-2,
        log_interval=1,
        checkpoint_interval=10,
        checkpoint_dir=str(tmp_path),
        max_steps=2,
    )
    trainer = module.DPOTrainer(policy_model, reference_model, tokenizer, config)

    dataset = StaticBatchDataset(
        [
            {
                "chosen_input_ids": torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]], dtype=torch.long),
                "rejected_input_ids": torch.tensor([[2, 3, 6, 7], [5, 6, 3, 2]], dtype=torch.long),
                "chosen_attention_mask": torch.ones((2, 4), dtype=torch.float32),
                "rejected_attention_mask": torch.ones((2, 4), dtype=torch.float32),
            },
            {
                "chosen_input_ids": torch.tensor([[8, 7, 6, 5], [4, 3, 2, 1]], dtype=torch.long),
                "rejected_input_ids": torch.tensor([[8, 7, 5, 4], [4, 3, 5, 6]], dtype=torch.long),
                "chosen_attention_mask": torch.ones((2, 4), dtype=torch.float32),
                "rejected_attention_mask": torch.ones((2, 4), dtype=torch.float32),
            },
        ]
    )

    before = {key: value.detach().clone() for key, value in policy_model.state_dict().items()}
    trainer.train(dataset)

    assert trainer.metrics_history
    assert (tmp_path / "final.safetensors").exists()
    assert any(
        not torch.equal(before[key], value)
        for key, value in policy_model.state_dict().items()
    )
