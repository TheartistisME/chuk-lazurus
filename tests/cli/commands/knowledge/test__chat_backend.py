"""EWS-4 backend plumbing for ``knowledge chat``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.knowledge import _chat


def test_chat_torch_backend_stops_at_dispatch_seam(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    args = argparse.Namespace(
        model="m",
        store=str(store_dir),
        max_tokens=1,
        temperature=0.0,
        backend="torch",
        device="cuda:0",
    )

    with patch(
        "chuk_lazarus.inference.context.knowledge.torch_query.run_torch_chat_command"
    ) as mocked_torch_chat:
        asyncio.run(_chat.knowledge_chat_cmd(args))

    mocked_torch_chat.assert_called_once_with(
        model_id="m",
        store_path=str(store_dir),
        max_new_tokens=1,
        temperature=0.0,
        device="cuda:0",
    )


def test_chat_defaults_when_flags_absent(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    args = argparse.Namespace(
        model="m",
        store=str(store_dir),
        max_tokens=1,
        temperature=0.0,
    )

    fake_tokenizer = MagicMock()
    fake_tokenizer.eos_token_id = 99

    fake_store = MagicMock()
    fake_store.log_stats = MagicMock()
    fake_store_cls = MagicMock()
    fake_store_cls.load.return_value = fake_store

    with (
        patch(
            "chuk_lazarus.cli.commands.knowledge._common.load_model",
            return_value=(MagicMock(), MagicMock(), fake_tokenizer),
        ) as mocked_load,
        patch(
            "chuk_lazarus.cli.commands.knowledge._backend.load_store_backend",
            return_value=fake_store_cls,
        ) as mocked_load_store_backend,
        patch("builtins.input", side_effect=EOFError()),
    ):
        asyncio.run(_chat.knowledge_chat_cmd(args))

    mocked_load.assert_called_once_with("m", backend=None, device=None)
    mocked_load_store_backend.assert_called_once_with("mlx")
