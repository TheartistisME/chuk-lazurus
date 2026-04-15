"""EWS-4 backend plumbing for ``knowledge._common``.

Validates:
- importing the knowledge command package does not pull mlx at module scope
  when ``CHUK_BACKEND=torch`` (EWS-4 MLX-preservation / lazy-import rule);
- ``_resolve_backend_device`` threads Epic 2 flags off ``args``;
- ``load_model`` forwards the backend/device pair into
  ``UnifiedPipelineConfig`` without requiring mlx/torch actually import.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch

from chuk_lazarus.cli.commands.knowledge import _common


def test_resolve_backend_device_pulls_from_namespace() -> None:
    ns = argparse.Namespace(backend="torch", device="cuda:0")
    assert _common._resolve_backend_device(ns) == ("torch", "cuda:0")


def test_resolve_backend_device_defaults_when_absent() -> None:
    assert _common._resolve_backend_device(argparse.Namespace()) == (None, None)


def test_load_model_forwards_backend_and_device_into_pipeline_config() -> None:
    fake_pipeline = MagicMock()
    fake_pipeline.model = MagicMock()
    fake_pipeline.config = MagicMock()
    fake_pipeline.tokenizer = MagicMock()
    fake_kv_gen = MagicMock()
    fake_kv_gen.prefill = MagicMock(return_value=(MagicMock(), MagicMock()))

    with (
        patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained", return_value=fake_pipeline)
        as mocked_from_pretrained,
        patch(
            "chuk_lazarus.inference.context.kv_generator.make_kv_generator",
            return_value=fake_kv_gen,
        ),
        patch("mlx.core.array", return_value=MagicMock()),
    ):
        _common.load_model("model-id", backend="torch", device="cuda:0")

    args, kwargs = mocked_from_pretrained.call_args
    assert args[0] == "model-id"
    assert kwargs.get("verbose") is False
    cfg = kwargs["pipeline_config"]
    assert cfg is not None
    assert cfg.backend.value == "cuda"  # LazarusBackend("torch") normalises to CUDA
    assert cfg.device == "cuda:0"


def test_load_model_uses_default_config_when_flags_absent() -> None:
    fake_pipeline = MagicMock()
    fake_pipeline.model = MagicMock()
    fake_pipeline.config = MagicMock()
    fake_pipeline.tokenizer = MagicMock()
    fake_kv_gen = MagicMock()
    fake_kv_gen.prefill = MagicMock(return_value=(MagicMock(), MagicMock()))

    with (
        patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained", return_value=fake_pipeline)
        as mocked_from_pretrained,
        patch(
            "chuk_lazarus.inference.context.kv_generator.make_kv_generator",
            return_value=fake_kv_gen,
        ),
        patch("mlx.core.array", return_value=MagicMock()),
    ):
        _common.load_model("model-id")

    _args, kwargs = mocked_from_pretrained.call_args
    assert kwargs["pipeline_config"] is None


def test_knowledge_modules_lazy_import_mlx_under_torch_backend() -> None:
    """Importing in-scope knowledge modules must not force mlx under torch."""
    script = textwrap.dedent(
        """
        import sys
        import chuk_lazarus.cli.commands.knowledge  # noqa: F401
        import chuk_lazarus.cli.commands.knowledge._common  # noqa: F401
        import chuk_lazarus.cli.commands.knowledge._build  # noqa: F401
        import chuk_lazarus.cli.commands.knowledge._query  # noqa: F401
        import chuk_lazarus.cli.commands.knowledge._chat  # noqa: F401
        assert "mlx" not in sys.modules, sorted(m for m in sys.modules if m.startswith("mlx"))
        """
    )
    repo_src = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
    existing_pp = os.environ.get("PYTHONPATH", "")
    pythonpath = (
        os.pathsep.join([os.path.abspath(repo_src), existing_pp])
        if existing_pp
        else os.path.abspath(repo_src)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "CHUK_BACKEND": "torch",
            "PYTHONPATH": pythonpath,
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
