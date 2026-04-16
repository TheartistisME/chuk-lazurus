"""Assert the post-EWS-0 parser split keeps every pre-split subcommand wired.

The split extracts train/context/serve parsers into their own modules and
reduces the legacy files to one-line shims.  Downstream importers still
target the legacy names — this test proves both paths still produce
identical registration.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

import pytest

from chuk_lazarus.cli import _parsers
from chuk_lazarus.cli._parsers import (
    _context as legacy_context,
    _context_parsers,
    _infer as legacy_infer,
    _infer_run,
    _train as legacy_train,
    _train_parsers,
    _train_rlhf,
    _train_sft,
)


def _subparser_names(register, add_top_level: bool = False) -> set[str]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register(subparsers)
    # Recursively walk subparser actions to collect every registered name.
    names: set[str] = set()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                names.add(name)
                for inner in sub._actions:
                    if isinstance(inner, argparse._SubParsersAction):
                        for sub_name in inner.choices:
                            names.add(f"{name}.{sub_name}")
    return names


def test_train_shim_matches_orchestrator():
    assert hasattr(legacy_train, "register_train_parsers")
    assert legacy_train.register_train_parsers is _train_parsers.register_train_parsers


def test_context_shim_matches_orchestrator():
    assert hasattr(legacy_context, "register_context_parsers")
    assert legacy_context.register_context_parsers is _context_parsers.register_context_parsers


def test_infer_shim_matches_split():
    assert hasattr(legacy_infer, "register_infer_parser")
    assert legacy_infer.register_infer_parser is _infer_run.register_infer_parser


def test_train_split_covers_all_subcommands():
    names = _subparser_names(_parsers.register_train_parsers)
    # Top-level generate, train.sft/dpo/grpo
    assert "train" in names
    assert "generate" in names
    assert "train.sft" in names
    assert "train.dpo" in names
    assert "train.grpo" in names


def test_context_split_covers_all_subcommands():
    names = _subparser_names(_parsers.register_context_parsers)
    assert "context" in names
    assert "context.prefill" in names
    assert "context.generate" in names
    assert "context.calibrate-frames" in names


def test_infer_split_still_registers_infer():
    names = _subparser_names(_parsers.register_infer_parser)
    assert "infer" in names


def test_add_backend_flags_reexported_from_parsers_package():
    # Shared helper is re-exported so bucket parsers don't need to reach into
    # cli.commands._base directly.
    assert callable(_parsers.add_backend_flags)
    assert _parsers.BACKEND_CHOICES == ("mlx", "torch")


@pytest.mark.parametrize(
    "module,attr",
    [
        (_train_sft, "register_train_sft_parser"),
        (_train_sft, "register_generate_parser"),
        (_train_rlhf, "register_train_dpo_parser"),
        (_train_rlhf, "register_train_grpo_parser"),
    ],
)
def test_split_modules_export_registrars(module, attr):
    assert hasattr(module, attr)
    assert callable(getattr(module, attr))


def test_parser_package_and_introspect_import_cleanly_under_torch():
    script = textwrap.dedent(
        """
        import argparse
        import chuk_lazarus.cli._parsers as parsers
        import chuk_lazarus.cli._parsers._context_generate  # noqa: F401
        import chuk_lazarus.cli._parsers._context_prefill  # noqa: F401
        import chuk_lazarus.cli._parsers._data  # noqa: F401
        import chuk_lazarus.cli._parsers._infer  # noqa: F401
        import chuk_lazarus.cli._parsers._infer_run  # noqa: F401
        import chuk_lazarus.cli._parsers._introspect as introspect
        import chuk_lazarus.cli._parsers._knowledge  # noqa: F401
        import chuk_lazarus.cli._parsers._tokenizer  # noqa: F401

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        introspect.register_introspect_parsers(subparsers)

        assert callable(parsers.register_infer_parser)
        assert "introspect" in next(
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ).choices
        """
    )
    repo_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
    )
    env = {**os.environ, "CHUK_BACKEND": "torch", "PYTHONPATH": repo_src}
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
