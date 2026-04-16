"""EWS-4 backend plumbing for ``knowledge query``."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.cli.commands.knowledge import _query


def test_query_torch_backend_dispatches_to_torch_runtime(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    args = argparse.Namespace(
        model="m",
        store=str(store_dir),
        prompt="hi",
        max_tokens=1,
        temperature=0.0,
        top_k=3,
        backend="torch",
        device="cuda:0",
    )

    with patch(
        "chuk_lazarus.inference.context.knowledge.torch_query.run_torch_query_command"
    ) as mocked_torch_query:
        asyncio.run(_query.knowledge_query_cmd(args))

    mocked_torch_query.assert_called_once_with(
        model_id="m",
        prompt="hi",
        max_new_tokens=1,
        temperature=0.0,
        top_k=3,
        device="cuda:0",
        store_path=str(store_dir),
    )


def test_query_defaults_when_flags_absent() -> None:
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template = MagicMock(side_effect=Exception())
    fake_tokenizer.encode = MagicMock(return_value=[1])
    fake_tokenizer.eos_token_id = 99
    fake_tokenizer.decode = MagicMock(return_value="")

    args = argparse.Namespace(
        model="m",
        store=None,
        prompt="hi",
        max_tokens=1,
        temperature=0.0,
        top_k=3,
    )

    with (
        patch(
            "chuk_lazarus.cli.commands.knowledge._common.load_model",
            return_value=(MagicMock(), MagicMock(), fake_tokenizer),
        ) as mocked_load,
        patch(
            "chuk_lazarus.cli.commands.knowledge._common.generate_plain",
            return_value=[],
        ),
    ):
        asyncio.run(_query.knowledge_query_cmd(args))

    mocked_load.assert_called_once_with("m", backend=None, device=None)
