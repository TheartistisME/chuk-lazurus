"""EWS-4 backend plumbing for ``knowledge build``."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.knowledge import _build


def test_build_forwards_backend_and_device(tmp_path: Path) -> None:
    input_file = tmp_path / "doc.txt"
    input_file.write_text("hello world", encoding="utf-8")
    output_dir = tmp_path / "store"

    fake_tokenizer = MagicMock()
    fake_pipeline = MagicMock()
    fake_pipeline.model = MagicMock()
    fake_pipeline.tokenizer = fake_tokenizer
    fake_pipeline.config = MagicMock()
    fake_store = MagicMock()
    fake_builder = MagicMock(return_value=fake_store)

    args = argparse.Namespace(
        model="m",
        input=str(input_file),
        output=str(output_dir),
        window_size=512,
        entries_per_window=8,
        max_tokens=None,
        backend="torch",
        device="cuda:0",
    )

    with (
        patch(
            "chuk_lazarus.cli.commands.knowledge._backend.load_pipeline",
            return_value=fake_pipeline,
        ) as mocked_load_pipeline,
        patch(
            "chuk_lazarus.cli.commands.knowledge._backend.load_build_backend",
            return_value=fake_builder,
        ) as mocked_load_builder,
        patch(
            "chuk_lazarus.inference.context.knowledge.config.ArchitectureConfig.from_model_config",
            return_value=MagicMock(crystal_layer=0, window_size=512, entries_per_window=8),
        ),
    ):
        import asyncio

        asyncio.run(_build.knowledge_build_cmd(args))

    mocked_load_pipeline.assert_called_once_with("m", backend="torch", device="cuda:0")
    mocked_load_builder.assert_called_once_with("torch")
    kwargs = fake_builder.call_args.kwargs
    assert kwargs["model"] is fake_pipeline.model
    assert kwargs["tokenizer"] is fake_tokenizer
    assert kwargs["raw_text"] == "hello world"
    assert kwargs["output_path"] == output_dir
    assert kwargs["max_tokens"] is None
    assert "document_tokens" not in kwargs
    assert callable(kwargs["progress_fn"])


def test_build_defaults_when_flags_absent(tmp_path: Path) -> None:
    input_file = tmp_path / "doc.txt"
    input_file.write_text("x", encoding="utf-8")

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=[1])
    fake_kv_gen = MagicMock()
    fake_pipeline = MagicMock()
    fake_pipeline.config = MagicMock()
    fake_store = MagicMock()
    fake_builder = MagicMock(return_value=fake_store)

    args = argparse.Namespace(
        model="m",
        input=str(input_file),
        output=str(tmp_path / "out"),
        window_size=512,
        entries_per_window=8,
        max_tokens=None,
    )

    with (
        patch(
            "chuk_lazarus.cli.commands.knowledge._common.load_model",
            return_value=(fake_pipeline, fake_kv_gen, fake_tokenizer),
        ) as mocked_load,
        patch(
            "chuk_lazarus.cli.commands.knowledge._backend.load_build_backend",
            return_value=fake_builder,
        ) as mocked_load_builder,
        patch(
            "chuk_lazarus.inference.context.knowledge.config.ArchitectureConfig.from_model_config",
            return_value=MagicMock(crystal_layer=0, window_size=512, entries_per_window=8),
        ),
    ):
        import asyncio

        asyncio.run(_build.knowledge_build_cmd(args))

    mocked_load.assert_called_once_with("m", backend=None, device=None)
    mocked_load_builder.assert_called_once_with("mlx")
    kwargs = fake_builder.call_args.kwargs
    assert kwargs["kv_gen"] is fake_kv_gen
    assert kwargs["document_tokens"] == [1]
    assert kwargs["tokenizer"] is fake_tokenizer
    assert callable(kwargs["progress_fn"])
