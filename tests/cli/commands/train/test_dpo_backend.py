"""Backend-focused tests for the DPO train parser and command wrapper."""

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
    module.register_train_dpo_parser(sub)
    return root


def test_dpo_parser_accepts_backend() -> None:
    root = _build()
    args = root.parse_args(["dpo", "--model", "x", "--data", "y", "--backend", "torch"])
    assert args.backend == "torch"


def test_dpo_parser_device_flag() -> None:
    root = _build()
    args = root.parse_args(["dpo", "--model", "x", "--data", "y", "--device", "cpu"])
    assert args.device == "cpu"


def test_dpo_command_threads_backend_and_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class DummyDPOConfig:
        @classmethod
        def from_args(cls, args: Namespace):
            return SimpleNamespace(
                model=args.model,
                ref_model=args.ref_model,
                data=Path(args.data),
                eval_data=Path(args.eval_data) if args.eval_data else None,
                output=Path(args.output),
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta=args.beta,
                max_length=args.max_length,
                use_lora=args.use_lora,
                lora_rank=args.lora_rank,
            )

    class DummyMode:
        DPO = SimpleNamespace(value="dpo")

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
            return SimpleNamespace(output_dir=Path("/tmp/dpo-out"), epochs_completed=3)

    module = load_repo_module(
        "chuk_lazarus.cli.commands.train.dpo",
        "chuk_lazarus/cli/commands/train/dpo.py",
        stubs={
            "chuk_lazarus.cli.commands.train._types": stub_module(
                "chuk_lazarus.cli.commands.train._types",
                DPOConfig=DummyDPOConfig,
                TrainMode=DummyMode,
                TrainResult=DummyTrainResult,
            ),
            "chuk_lazarus.training.trainers.dpo_trainer": stub_module(
                "chuk_lazarus.training.trainers.dpo_trainer",
                DPOTrainer=DummyTrainer,
                DPOTrainingConfig=DummyTrainingConfig,
            ),
        },
    )

    args = Namespace(
        model="tiny-model",
        ref_model="ref-model",
        data="pairs.jsonl",
        eval_data=None,
        output="out",
        epochs=3,
        batch_size=4,
        learning_rate=1e-5,
        beta=0.2,
        max_length=64,
        use_lora=False,
        lora_rank=8,
        backend="torch",
        device="cuda:0",
    )
    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    asyncio.run(module.train_dpo_cmd(args))

    cfg = captured["config"]
    assert cfg.model == "tiny-model"
    assert cfg.ref_model == "ref-model"
    assert cfg.data_path == Path("pairs.jsonl")
    assert cfg.beta == 0.2
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda:0"
    assert "dpo:3:/tmp/dpo-out" in capsys.readouterr().out
