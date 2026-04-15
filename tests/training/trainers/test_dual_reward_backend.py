"""Backend-focused tests for the dual-reward trainer file."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.training.trainers._backend_loader import REPO, TinyTokenizer, TinyTorchLM, load_repo_module


def test_dual_reward_no_top_level_mlx_ast() -> None:
    path = "src/chuk_lazarus/training/trainers/dual_reward_trainer.py"
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"mlx", "mlx_lm"}, path
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"mlx", "mlx_lm"}, path


def test_dual_reward_trainer_uses_lazy_proxy() -> None:
    module = load_repo_module(
        "chuk_lazarus.training.trainers.dual_reward_trainer",
        "chuk_lazarus/training/trainers/dual_reward_trainer.py",
    )

    assert module.mx.__class__.__name__ == "_LazyMod"


def test_dual_reward_torch_reports_precise_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    module = load_repo_module(
        "chuk_lazarus.training.trainers.dual_reward_trainer",
        "chuk_lazarus/training/trainers/dual_reward_trainer.py",
    )

    with pytest.raises(RuntimeError, match="models_v2.adapters.lora"):
        module.DualRewardTrainer(
            TinyTorchLM(vocab_size=32, hidden_size=16),
            TinyTokenizer(),
            module.DualRewardTrainerConfig(),
        )
