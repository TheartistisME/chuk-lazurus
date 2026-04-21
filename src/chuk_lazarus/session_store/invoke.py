"""Thin subprocess wrapper around ``tools/build_clause_aligned_store.py``.

HARD CONSTRAINT
---------------
The build script at ``tools/build_clause_aligned_store.py`` is INVOKE-ONLY.
This module assembles argv and runs it via :func:`subprocess.run`; it never
imports or mutates the script.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from chuk_lazarus.session_store.layout import log_path, per_session_checkpoint


# Resolve the authoritative build-script path once. The script lives at
# ``<repo_root>/tools/build_clause_aligned_store.py``. This module sits at
# ``<repo_root>/src/chuk_lazarus/session_store/invoke.py``, i.e. four parents
# up from this file.
_REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT_PATH = _REPO_ROOT / "tools" / "build_clause_aligned_store.py"


@dataclass
class SessionStoreResult:
    """Outcome of a single ``invoke_build`` call.

    Attributes
    ----------
    session_id:
        Logical session identifier; also the basename of ``checkpoint``.
    checkpoint:
        Per-session checkpoint directory under which the build script writes
        its artifacts.
    returncode:
        Subprocess exit code (``0`` on success). For ``dry_run=True`` this is
        always ``0`` and no subprocess is spawned.
    stdout:
        Captured stdout text. For ``dry_run=True`` this is the shell-quoted
        argv that *would* have been executed, for debugging.
    stderr:
        Captured stderr text. Empty string for ``dry_run=True``.
    """

    session_id: str
    checkpoint: Path
    returncode: int
    stdout: str
    stderr: str


def _assemble_argv(
    *,
    python: str,
    build_script: Path,
    input_dir: Path,
    checkpoint: Path,
    model: Path | None,
    window_size: int,
    overlap_tokens: int,
    device: str,
    force: bool,
) -> list[str]:
    argv: list[str] = [
        python,
        str(build_script),
        "--input-dir",
        str(input_dir),
        "--checkpoint",
        str(checkpoint),
        "--window-size",
        str(window_size),
        "--overlap-tokens",
        str(overlap_tokens),
        "--device",
        device,
    ]
    if model is not None:
        argv.extend(["--model", str(model)])
    if force:
        argv.append("--force")
    return argv


def invoke_build(
    *,
    session_id: str,
    input_dir: Path,
    checkpoint_root: Path,
    model: Path | None = None,
    window_size: int = 512,
    overlap_tokens: int = 64,
    device: str = "cuda",
    force: bool = False,
    python: str | None = None,
    dry_run: bool = False,
) -> SessionStoreResult:
    """Invoke the clause-aligned build script for a single session.

    Parameters
    ----------
    session_id:
        Logical session ID; used as the per-session checkpoint subdirectory.
    input_dir:
        Directory containing per-clause JSON files emitted by axis-2.
    checkpoint_root:
        Parent directory for per-session checkpoints. The actual checkpoint
        for this invocation lands at ``<checkpoint_root>/<session_id>``.
    model:
        Optional override for the base model path. When ``None`` the build
        script's own default is used (no ``--model`` arg is passed).
    window_size, overlap_tokens, device, force:
        Pass-through CLI arguments for the build script.
    python:
        Python interpreter to invoke. Defaults to ``sys.executable``.
    dry_run:
        When ``True``, do NOT spawn a subprocess. Return a
        :class:`SessionStoreResult` whose ``stdout`` is the shell-quoted argv
        that would have been executed. Used for unit testing the argv builder
        without loading CUDA / models.

    Returns
    -------
    SessionStoreResult
        The subprocess outcome. Non-zero returncodes are NOT raised; callers
        are responsible for deciding how to react.
    """
    checkpoint = per_session_checkpoint(Path(checkpoint_root), session_id)
    interpreter = python or sys.executable

    argv = _assemble_argv(
        python=interpreter,
        build_script=BUILD_SCRIPT_PATH,
        input_dir=Path(input_dir),
        checkpoint=checkpoint,
        model=Path(model) if model is not None else None,
        window_size=window_size,
        overlap_tokens=overlap_tokens,
        device=device,
        force=force,
    )

    if dry_run:
        return SessionStoreResult(
            session_id=session_id,
            checkpoint=checkpoint,
            returncode=0,
            stdout=shlex.join(argv),
            stderr="",
        )

    proc = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )

    # Best-effort log capture. The checkpoint directory is created by the
    # build script on success; if it failed before mkdir we still try to
    # create the parent so the log survives.
    try:
        checkpoint.mkdir(parents=True, exist_ok=True)
        log_file = log_path(checkpoint)
        log_file.write_text(
            "# argv\n"
            + shlex.join(argv)
            + "\n\n# stdout\n"
            + (proc.stdout or "")
            + "\n\n# stderr\n"
            + (proc.stderr or "")
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        # Never let logging bury the real subprocess error.
        pass

    return SessionStoreResult(
        session_id=session_id,
        checkpoint=checkpoint,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
