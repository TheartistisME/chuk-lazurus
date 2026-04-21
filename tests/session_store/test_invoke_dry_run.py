"""Unit tests for ``chuk_lazarus.session_store.invoke``.

Dry-run tests assemble argv without spawning a subprocess. A monkeypatched
``subprocess.run`` is used for the failure-path test so we never hit disk
with real CUDA work.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import pytest

from chuk_lazarus.session_store.invoke import (
    BUILD_SCRIPT_PATH,
    SessionStoreResult,
    invoke_build,
)


def test_dry_run_returns_argv_without_subprocess(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        dry_run=True,
    )
    assert isinstance(result, SessionStoreResult)
    assert result.returncode == 0
    assert result.stderr == ""
    tokens = shlex.split(result.stdout)
    for flag in ("--input-dir", "--checkpoint", "--window-size", "--overlap-tokens", "--device"):
        assert flag in tokens, f"missing {flag} in argv {tokens!r}"
    assert str(BUILD_SCRIPT_PATH) in tokens
    assert result.checkpoint == tmp_path / "out" / "sess-1"


def test_dry_run_includes_model_when_provided(tmp_path: Path) -> None:
    model_path = tmp_path / "my-model"
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        model=model_path,
        dry_run=True,
    )
    tokens = shlex.split(result.stdout)
    assert "--model" in tokens
    assert str(model_path) in tokens


def test_dry_run_omits_model_when_none(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        dry_run=True,
    )
    tokens = shlex.split(result.stdout)
    assert "--model" not in tokens


def test_dry_run_includes_force_flag_when_true(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        force=True,
        dry_run=True,
    )
    tokens = shlex.split(result.stdout)
    assert "--force" in tokens


def test_dry_run_omits_force_flag_when_false(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        dry_run=True,
    )
    tokens = shlex.split(result.stdout)
    assert "--force" not in tokens


def test_dry_run_uses_custom_python_when_given(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        python="/custom/bin/python",
        dry_run=True,
    )
    tokens = shlex.split(result.stdout)
    assert tokens[0] == "/custom/bin/python"


def test_dry_run_does_not_create_checkpoint_dir(tmp_path: Path) -> None:
    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
        dry_run=True,
    )
    assert result.checkpoint.exists() is False


def test_invoke_build_returns_nonzero_when_script_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv, check=False, capture_output=False, text=False):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=argv, returncode=2, stdout="", stderr="boom"
        )

    monkeypatch.setattr(
        "chuk_lazarus.session_store.invoke.subprocess.run", fake_run
    )

    result = invoke_build(
        session_id="sess-1",
        input_dir=tmp_path / "in",
        checkpoint_root=tmp_path / "out",
    )
    assert result.returncode == 2
    assert result.stderr == "boom"
    log_file = result.checkpoint / "invoke.log"
    assert log_file.exists()
