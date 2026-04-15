"""Backend-focused tests for the SFT trainer and CLI-owned SFT files."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.training.trainers._backend_loader import (
    REPO,
    StaticBatchDataset,
    TinyTokenizer,
    TinyTorchLM,
    load_repo_module,
)


@pytest.mark.parametrize(
    "path",
    [
        "src/chuk_lazarus/training/trainers/sft_trainer.py",
        "src/chuk_lazarus/cli/commands/train/sft.py",
        "src/chuk_lazarus/cli/_parsers/_train_sft.py",
    ],
)
def test_sft_no_top_level_mlx_ast(path: str) -> None:
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"mlx", "mlx_lm"}, path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"mlx", "mlx_lm"}, path


def test_sft_trainer_uses_lazy_proxy() -> None:
    module = load_repo_module(
        "chuk_lazarus.training.trainers.sft_trainer",
        "chuk_lazarus/training/trainers/sft_trainer.py",
    )

    assert module.mx.__class__.__name__ == "_LazyMod"
    assert module.nn.__class__.__name__ == "_LazyMod"
    assert module.optim.__class__.__name__ == "_LazyMod"


def test_sft_torch_train_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setenv("CHUK_BACKEND", "torch")
    monkeypatch.delenv("CHUK_DEVICE", raising=False)
    module = load_repo_module(
        "chuk_lazarus.training.trainers.sft_trainer",
        "chuk_lazarus/training/trainers/sft_trainer.py",
    )

    model = TinyTorchLM(vocab_size=32, hidden_size=16)
    tokenizer = TinyTokenizer()
    config = module.SFTConfig(
        num_epochs=1,
        batch_size=2,
        learning_rate=5e-2,
        log_interval=1,
        checkpoint_interval=10,
        checkpoint_dir=str(tmp_path),
        max_steps=2,
    )
    trainer = module.SFTTrainer(model, tokenizer, config)

    dataset = StaticBatchDataset(
        [
            {
                "input_ids": torch.tensor([[2, 3, 4, 5], [5, 6, 7, 8]], dtype=torch.long),
                "labels": torch.tensor([[3, 4, 5, 1], [6, 7, 8, 1]], dtype=torch.long),
                "loss_mask": torch.ones((2, 4), dtype=torch.float32),
            },
            {
                "input_ids": torch.tensor([[8, 7, 6, 5], [4, 3, 2, 1]], dtype=torch.long),
                "labels": torch.tensor([[7, 6, 5, 1], [3, 2, 1, 1]], dtype=torch.long),
                "loss_mask": torch.ones((2, 4), dtype=torch.float32),
            },
        ]
    )

    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer.train(dataset)

    assert trainer.metrics_history
    assert (tmp_path / "final.safetensors").exists()
    assert any(
        not torch.equal(before[key], value)
        for key, value in model.state_dict().items()
    )
