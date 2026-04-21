"""Subprocess wrapper around ``tools/count_tokens.py`` for the demo's 128k gate.

The underlying tool is invoke-only (scope-lead guarantee). This module
assembles argv, runs it via ``subprocess.run``, and parses the single
integer emitted by ``--total-only``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

GEMMA_SNAPSHOT_DEFAULT = (
    "/home/jehmal/.cache/huggingface/hub/"
    "models--google--gemma-4-E2B-it/snapshots/"
    "b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)

# Resolve the repo root so we can invoke tools/count_tokens.py regardless
# of the caller's CWD. Layout is src/chuk_lazarus/cross_session_demo/*.py,
# so three parents up is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_COUNT_TOKENS = _REPO_ROOT / "tools" / "count_tokens.py"


def _resolve_subprocess_python() -> str:
    """Pick the Python interpreter most likely to have repo deps installed.

    Preference order:
      1. ``$VIRTUAL_ENV/bin/python`` if set (active venv).
      2. ``<repo>/.venv/bin/python`` if present (uv-managed venv).
      3. ``sys.executable`` (caller's interpreter — may be a test runner
         from a different environment).
    """
    active = os.environ.get("VIRTUAL_ENV")
    if active:
        cand = Path(active) / "bin" / "python"
        if cand.is_file():
            return str(cand)
    repo_venv = _REPO_ROOT / ".venv" / "bin" / "python"
    if repo_venv.is_file():
        return str(repo_venv)
    return sys.executable


def _require_tokenizer(tokenizer_path: str) -> None:
    if not Path(tokenizer_path).is_dir():
        raise RuntimeError(
            "Gemma tokenizer snapshot directory missing at "
            f"{tokenizer_path!r}. Expected a HuggingFace cache snapshot. "
            "Ensure the model was fetched into the local HF cache "
            "(HF_HOME or ~/.cache/huggingface/hub)."
        )


def measure_total_tokens(
    texts: list[str],
    tokenizer_path: str = GEMMA_SNAPSHOT_DEFAULT,
    *,
    python: str | None = None,
) -> int:
    """Return the sum of Gemma-tokenized token counts across ``texts``.

    Each text is written to its own tempfile; ``tools/count_tokens.py``
    is invoked once with all file paths and ``--total-only`` to keep
    output parseable. Tempfiles are always cleaned up.
    """
    _require_tokenizer(tokenizer_path)
    if not _COUNT_TOKENS.is_file():
        raise RuntimeError(
            f"count_tokens.py missing at expected path {_COUNT_TOKENS!s}"
        )

    interpreter = python or _resolve_subprocess_python()

    with tempfile.TemporaryDirectory(prefix="xsess_tok_") as tmpdir:
        tmp_root = Path(tmpdir)
        file_paths: list[Path] = []
        for i, text in enumerate(texts):
            # Deterministic filenames make subprocess argv reproducible
            # and keep failure messages readable.
            p = tmp_root / f"session_{i:03d}.txt"
            p.write_text(text, encoding="utf-8")
            file_paths.append(p)

        argv = [
            interpreter,
            str(_COUNT_TOKENS),
            "--tokenizer",
            tokenizer_path,
            "--total-only",
            *[str(p) for p in file_paths],
        ]
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "count_tokens.py failed with returncode "
                f"{proc.returncode}: stderr={proc.stderr!r}"
            )
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            raise RuntimeError(
                "count_tokens.py returned no stdout; stderr="
                f"{proc.stderr!r}"
            )
        try:
            return int(raw[-1].strip())
        except ValueError as exc:
            raise RuntimeError(
                "count_tokens.py --total-only output not an int: "
                f"{raw!r}"
            ) from exc


__all__ = [
    "GEMMA_SNAPSHOT_DEFAULT",
    "measure_total_tokens",
]
