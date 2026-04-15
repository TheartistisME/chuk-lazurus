"""Backend-focused tests for the GRPO train parser and command wrapper."""

from __future__ import annotations

import argparse
import asyncio
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from tests.training.trainers._backend_loader import load_repo_module, stub_module


def _load_parser_module():
    async def _noop(_args):
        return None

    return load_repo_module(
        "chuk_lazarus.cli._parsers._train_rlhf",
        "chuk_lazarus/cli/_parsers/_train_rlhf.py",
        stubs={
            "chuk_lazarus.cli.commands.train": stub_module(
                "chuk_lazarus.cli.commands.train",
                train_dpo_cmd=_noop,
                train_grpo_cmd=_noop,
            )
        },
    )


def _build() -> argparse.ArgumentParser:
    module = _load_parser_module()
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="mode")
    module.register_train_grpo_parser(sub)
    return root


def test_grpo_parser_accepts_backend() -> None:
    root = _build()
    args = root.parse_args(
        ["grpo", "--model", "x", "--reward-script", "r.py", "--backend", "mlx"]
    )
    assert args.backend == "mlx"


def test_grpo_parser_accepts_device() -> None:
    root = _build()
    args = root.parse_args(
        [
            "grpo",
            "--model",
            "x",
            "--reward-script",
            "r.py",
            "--backend",
            "torch",
            "--device",
            "cuda:0",
        ]
    )
    assert args.backend == "torch"
    assert args.device == "cuda:0"


def test_grpo_command_threads_backend_and_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class DummyGRPOConfig:
        @classmethod
        def from_args(cls, args: Namespace):
            return SimpleNamespace(
                model=args.model,
                reference_model=args.ref_model or args.model,
                output=Path(args.output),
                iterations=args.iterations,
                prompts_per_iteration=args.prompts_per_iteration,
                group_size=args.group_size,
                learning_rate=args.learning_rate,
                kl_coef=args.kl_coef,
                max_response_length=args.max_response_length,
                temperature=args.temperature,
                use_lora=args.use_lora,
                lora_rank=args.lora_rank,
                reward_script=Path(args.reward_script) if args.reward_script else None,
            )

    class DummyMode:
        GRPO = SimpleNamespace(value="grpo")

    class DummyTrainResult:
        def __init__(self, *, mode, checkpoint_dir, epochs_completed):
            self.mode = mode
            self.checkpoint_dir = checkpoint_dir
            self.epochs_completed = epochs_completed

        def to_display(self) -> str:
            return f"{self.mode.value}:{self.epochs_completed}:{self.checkpoint_dir}"

    class DummyTrainingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyTrainer:
        @staticmethod
        def run(config):
            captured["config"] = config
            return SimpleNamespace(output_dir=Path("/tmp/grpo-out"), iterations_completed=4)

    module = load_repo_module(
        "chuk_lazarus.cli.commands.train.grpo",
        "chuk_lazarus/cli/commands/train/grpo.py",
        stubs={
            "chuk_lazarus.cli.commands.train._types": stub_module(
                "chuk_lazarus.cli.commands.train._types",
                GRPOConfig=DummyGRPOConfig,
                TrainMode=DummyMode,
                TrainResult=DummyTrainResult,
            ),
            "chuk_lazarus.training.trainers.grpo_trainer": stub_module(
                "chuk_lazarus.training.trainers.grpo_trainer",
                GRPOTrainer=DummyTrainer,
                GRPOTrainingConfig=DummyTrainingConfig,
            ),
        },
    )

    args = Namespace(
        model="tiny-model",
        ref_model="ref-model",
        output="out",
        iterations=4,
        prompts_per_iteration=2,
        group_size=3,
        learning_rate=1e-5,
        kl_coef=0.3,
        max_response_length=16,
        temperature=0.9,
        use_lora=False,
        lora_rank=8,
        reward_script="reward.py",
        backend="torch",
        device="cuda:0",
    )
    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    asyncio.run(module.train_grpo_cmd(args))

    cfg = captured["config"]
    assert cfg.model == "tiny-model"
    assert cfg.ref_model == "ref-model"
    assert cfg.reward_script == Path("reward.py")
    assert cfg.group_size == 3
    assert cfg.kl_coef == 0.3
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda:0"
    assert "grpo:4:/tmp/grpo-out" in capsys.readouterr().out
