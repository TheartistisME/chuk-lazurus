from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _make_args(**overrides) -> Namespace:
    base = dict(
        model="test-model",
        checkpoint=None,
        prompt="hello world",
        prompt_file=None,
        max_tokens=32,
        temperature=0.25,
        replay=None,
        find=None,
        top_k=None,
        strategy=None,
        speculative_tokens=50,
        max_rounds=3,
        system_prompt=None,
        no_chat_template=False,
        max_keywords=3,
        no_framing=False,
        span_radius=None,
        mode="auto",
        token_budget=None,
        broad_windows=None,
        broad_strategy=None,
        routing_layer=29,
        routing_head=4,
        confidence_threshold=0.15,
        backend=None,
        device=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _write_torch_sidecar(
    checkpoint: Path,
    *,
    store_dir: str = "torch_store",
    ready: bool = True,
    status: str = "complete",
) -> Path:
    checkpoint.mkdir(parents=True, exist_ok=True)
    store_path = checkpoint / store_dir
    store_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": status,
        "backend": "torch",
        "model": {"id": "sidecar-model"},
        "windowing": {"num_windows": 2, "num_tokens": 12},
        "artifacts": {"store_dir": store_dir},
        "generate_batch": {"ready": ready},
    }
    (checkpoint / "torch_prefill.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return store_path


def test_generate_plain_torch_uses_plain_path(capsys) -> None:
    from chuk_lazarus.cli.commands.context.generate._cmd import context_generate_cmd

    args = _make_args(backend="torch", device="cuda:1")
    seen: dict[str, object] = {}

    def _fake_plain_generate(config, runtime_args, *, backend_name=None, device=None):
        seen["prompt"] = config.prompt_text
        seen["backend_name"] = backend_name
        seen["device"] = device
        assert runtime_args is args
        return SimpleNamespace(to_display=lambda: "plain-torch-ok")

    with (
        patch(
            "chuk_lazarus.models_v2.core.backend.get_backend",
            return_value=SimpleNamespace(name="torch", device="cuda:1"),
        ),
        patch(
            "chuk_lazarus.cli.commands.context.generate._plain._plain_generate",
            side_effect=_fake_plain_generate,
        ),
    ):
        asyncio.run(context_generate_cmd(args))

    assert seen == {
        "prompt": "hello world",
        "backend_name": "torch",
        "device": "cuda:1",
    }
    captured = capsys.readouterr()
    assert "plain-torch-ok" in captured.out


def test_generate_checkpoint_torch_dispatches_to_sidecar_query(tmp_path: Path) -> None:
    from chuk_lazarus.cli.commands.context.generate._cmd import context_generate_cmd

    checkpoint = tmp_path / "checkpoint"
    store_path = _write_torch_sidecar(checkpoint)
    (store_path / "manifest.json").write_text("{}\n")
    args = _make_args(
        checkpoint=str(checkpoint),
        top_k=7,
        backend="torch",
        device="cuda:0",
        system_prompt="keep it grounded",
        no_chat_template=True,
    )

    with (
        patch(
            "chuk_lazarus.models_v2.core.backend.get_backend",
            return_value=SimpleNamespace(name="torch", device="cuda:0"),
        ),
        patch(
            "chuk_lazarus.inference.context.knowledge.torch_query.run_torch_query_command"
        ) as mocked_query,
    ):
        asyncio.run(context_generate_cmd(args))

    mocked_query.assert_called_once_with(
        model_id="test-model",
        prompt="hello world",
        max_new_tokens=32,
        temperature=0.25,
        top_k=7,
        device="cuda:0",
        store_path=store_path,
        system_prompt="keep it grounded",
        no_chat_template=True,
    )


def test_generate_checkpoint_torch_errors_when_sidecar_missing(
    tmp_path: Path,
    capsys,
) -> None:
    from chuk_lazarus.cli.commands.context.generate._cmd import context_generate_cmd

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    args = _make_args(
        checkpoint=str(checkpoint),
        backend="torch",
        device="cuda:0",
    )

    with (
        patch(
            "chuk_lazarus.models_v2.core.backend.get_backend",
            return_value=SimpleNamespace(name="torch", device="cuda:0"),
        ),
        patch(
            "chuk_lazarus.inference.context.knowledge.torch_query.run_torch_query_command"
        ) as mocked_query,
    ):
        asyncio.run(context_generate_cmd(args))

    mocked_query.assert_not_called()
    captured = capsys.readouterr()
    assert "torch_prefill.json" in captured.err


def test_generate_checkpoint_torch_errors_when_store_manifest_missing(
    tmp_path: Path,
    capsys,
) -> None:
    from chuk_lazarus.cli.commands.context.generate._cmd import context_generate_cmd

    checkpoint = tmp_path / "checkpoint"
    _write_torch_sidecar(checkpoint)
    args = _make_args(
        checkpoint=str(checkpoint),
        backend="torch",
        device="cuda:0",
    )

    with (
        patch(
            "chuk_lazarus.models_v2.core.backend.get_backend",
            return_value=SimpleNamespace(name="torch", device="cuda:0"),
        ),
        patch(
            "chuk_lazarus.inference.context.knowledge.torch_query.run_torch_query_command"
        ) as mocked_query,
    ):
        asyncio.run(context_generate_cmd(args))

    mocked_query.assert_not_called()
    captured = capsys.readouterr()
    assert "manifest.json" in captured.err
