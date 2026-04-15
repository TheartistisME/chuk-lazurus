"""Backend-focused tests for the SFT train parser and command wrapper."""

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
        "chuk_lazarus.cli._parsers._train_sft",
        "chuk_lazarus/cli/_parsers/_train_sft.py",
        stubs={
            "chuk_lazarus.cli.commands.train": stub_module(
                "chuk_lazarus.cli.commands.train",
                train_sft_cmd=_noop,
                generate_data_cmd=_noop,
            )
        },
    )


def _build() -> argparse.ArgumentParser:
    module = _load_parser_module()
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="mode")
    module.register_train_sft_parser(sub)
    return root


def test_sft_parser_accepts_backend_mlx() -> None:
    root = _build()
    args = root.parse_args(["sft", "--model", "x", "--data", "y", "--backend", "mlx"])
    assert args.backend == "mlx"


def test_sft_parser_accepts_backend_torch_with_device() -> None:
    root = _build()
    args = root.parse_args(
        ["sft", "--model", "x", "--data", "y", "--backend", "torch", "--device", "cuda"]
    )
    assert args.backend == "torch"
    assert args.device == "cuda"


def test_sft_parser_default_backend_is_none() -> None:
    root = _build()
    args = root.parse_args(["sft", "--model", "x", "--data", "y"])
    assert args.backend is None


def test_sft_command_threads_backend_and_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class DummySFTConfig:
        @classmethod
        def from_args(cls, args: Namespace):
            return SimpleNamespace(
                model=args.model,
                data=Path(args.data),
                eval_data=Path(args.eval_data) if args.eval_data else None,
                output=Path(args.output),
                epochs=args.epochs,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                use_lora=args.use_lora,
                lora_rank=args.lora_rank,
                mask_prompt=args.mask_prompt,
                log_interval=args.log_interval,
            )

    class DummyMode:
        SFT = SimpleNamespace(value="sft")

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
            return SimpleNamespace(output_dir=Path("/tmp/sft-out"), epochs_completed=2)

    module = load_repo_module(
        "chuk_lazarus.cli.commands.train.sft",
        "chuk_lazarus/cli/commands/train/sft.py",
        stubs={
            "chuk_lazarus.cli.commands.train._types": stub_module(
                "chuk_lazarus.cli.commands.train._types",
                SFTConfig=DummySFTConfig,
                TrainMode=DummyMode,
                TrainResult=DummyTrainResult,
            ),
            "chuk_lazarus.training.trainers.sft_trainer": stub_module(
                "chuk_lazarus.training.trainers.sft_trainer",
                SFTTrainer=DummyTrainer,
                SFTTrainingConfig=DummyTrainingConfig,
            ),
        },
    )

    args = Namespace(
        model="tiny-model",
        data="train.jsonl",
        eval_data=None,
        output="out",
        epochs=2,
        max_steps=5,
        batch_size=4,
        learning_rate=1e-4,
        max_length=64,
        use_lora=False,
        lora_rank=8,
        mask_prompt=True,
        log_interval=3,
        backend="torch",
        device="cuda:0",
    )
    monkeypatch.delenv("CHUK_BACKEND", raising=False)
    monkeypatch.delenv("CHUK_DEVICE", raising=False)

    asyncio.run(module.train_sft_cmd(args))

    cfg = captured["config"]
    assert cfg.model == "tiny-model"
    assert cfg.data_path == Path("train.jsonl")
    assert cfg.output_dir == Path("out")
    assert cfg.max_steps == 5
    assert cfg.mask_prompt is True
    assert os.environ["CHUK_BACKEND"] == "torch"
    assert os.environ["CHUK_DEVICE"] == "cuda:0"
    assert "sft:2:/tmp/sft-out" in capsys.readouterr().out
