"""Torch-runtime import gates for train-facing CLI package surfaces."""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={"CHUK_BACKEND": "torch", "PYTHONPATH": "src"},
    )


def test_cli_package_import_is_torch_clean() -> None:
    result = _run("import chuk_lazarus.cli as cli; print(cli.__name__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "chuk_lazarus.cli"


def test_train_package_import_is_torch_clean() -> None:
    result = _run("import chuk_lazarus.cli.commands.train as train; print(train.__name__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "chuk_lazarus.cli.commands.train"


def test_train_grpo_module_import_is_torch_clean() -> None:
    result = _run("import chuk_lazarus.cli.commands.train.grpo as mod; print(mod.__name__)")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "chuk_lazarus.cli.commands.train.grpo"
