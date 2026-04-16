"""Backend and lazy-import coverage for ``context calibrate-frames``."""

from __future__ import annotations

import ast
import asyncio
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "src/chuk_lazarus/cli/commands/context/calibrate_frames.py"
)


def _make_args(tmp_path: Path, **overrides) -> Namespace:
    base = dict(
        model="org/model",
        output=str(tmp_path / "frame_bank.npz"),
        dims_per_frame=2,
        layer_frac=0.5,
        method="whitening",
        backend=None,
        device=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_calibrate_frames_torch_backend_uses_runtime_and_writes_frame_bank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from chuk_lazarus.cli.commands.context.calibrate_frames import context_calibrate_frames_cmd
    from chuk_lazarus.inference.unified import UnifiedPipeline

    prompts = [f"probe-{idx}" for idx in range(6)]
    monkeypatch.setattr(
        "chuk_lazarus.cli.commands.context.calibrate_frames.PROBE_PROMPTS",
        prompts,
    )

    class _StubBackend:
        name = "torch"
        device = "cuda:7"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )

    calls: list[tuple[str, int]] = []

    class _Runtime:
        def extract_residual_state(self, prompt: str, layer_index: int):
            idx = len(calls)
            calls.append((prompt, layer_index))
            vector = np.array(
                [[idx, idx + 1, idx + 2, 2 * idx, (-1) ** idx, idx % 3, idx / 2, idx + 0.25]],
                dtype=np.float32,
            )
            return SimpleNamespace(tensor=vector)

    pipeline = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(num_hidden_layers=10),
        runtime=_Runtime(),
    )
    seen: dict[str, object] = {}

    def _fake_from_pretrained(cls, model_id: str, **kwargs):
        seen["model_id"] = model_id
        seen["kwargs"] = kwargs
        return pipeline

    monkeypatch.setattr(
        UnifiedPipeline,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )

    args = _make_args(tmp_path, backend="torch", device="cuda:7")
    asyncio.run(context_calibrate_frames_cmd(args))

    assert calls == [(prompt, 5) for prompt in prompts]
    pipeline_config = seen["kwargs"]["pipeline_config"]
    assert pipeline_config.backend_name == "torch"
    assert pipeline_config.device == "cuda:7"

    with np.load(tmp_path / "frame_bank.npz", allow_pickle=False) as archive:
        frame_bank = archive["frame_bank"]
        assert frame_bank.shape == (2, 8)
        assert archive["compass_layer"].item() == 5
        assert archive["dims_per_frame"].item() == 2
        assert archive["n_categories"].item() == 0


def test_calibrate_frames_mlx_backend_keeps_lazy_mlx_save_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from chuk_lazarus.cli.commands.context.calibrate_frames import context_calibrate_frames_cmd
    from chuk_lazarus.inference.unified import UnifiedPipeline

    prompts = [f"probe-{idx}" for idx in range(6)]
    monkeypatch.setattr(
        "chuk_lazarus.cli.commands.context.calibrate_frames.PROBE_PROMPTS",
        prompts,
    )

    class _StubBackend:
        name = "mlx"
        device = "mps"

    monkeypatch.setattr(
        "chuk_lazarus.models_v2.core.backend.get_backend",
        lambda *a, **k: _StubBackend(),
    )

    save_calls: list[tuple[str, tuple[str, ...]]] = []
    warmed: list[object] = []

    def _mx_array(value):
        return np.asarray(value)

    def _mx_savez(path, **kwargs):
        save_calls.append((path, tuple(sorted(kwargs.keys()))))
        np.savez(str(path), **{key: np.asarray(value) for key, value in kwargs.items()})

    mlx_core = ModuleType("mlx.core")
    mlx_core.array = _mx_array
    mlx_core.eval = lambda *args: None
    mlx_core.savez = _mx_savez

    mlx_pkg = ModuleType("mlx")
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    calls: list[tuple[str, int]] = []

    class _Runtime:
        def extract_residual_state(self, prompt: str, layer_index: int):
            idx = len(calls)
            calls.append((prompt, layer_index))
            vector = np.array(
                [[idx + 0.5, idx + 1.5, idx + 2.5, idx + 3.5, idx + 4.5, idx + 5.5]],
                dtype=np.float32,
            )
            return SimpleNamespace(tensor=vector)

    pipeline = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(num_hidden_layers=12),
        runtime=_Runtime(),
    )
    seen: dict[str, object] = {}

    def _fake_from_pretrained(cls, model_id: str, **kwargs):
        seen["model_id"] = model_id
        seen["kwargs"] = kwargs
        return pipeline

    monkeypatch.setattr(
        UnifiedPipeline,
        "from_pretrained",
        classmethod(_fake_from_pretrained),
    )
    monkeypatch.setattr(
        "chuk_lazarus.cli.commands.context.calibrate_frames._warm_mlx_pipeline",
        lambda pipeline_arg, mx_arg: warmed.append((pipeline_arg, mx_arg)),
    )

    args = _make_args(tmp_path, backend="mlx")
    asyncio.run(context_calibrate_frames_cmd(args))

    assert calls == [(prompt, 6) for prompt in prompts]
    pipeline_config = seen["kwargs"]["pipeline_config"]
    assert pipeline_config.backend_name == "mlx"
    assert pipeline_config.device is None
    assert len(warmed) == 1
    assert save_calls
    assert save_calls[0][1] == (
        "compass_layer",
        "dims_per_frame",
        "frame_bank",
        "n_categories",
    )

    with np.load(tmp_path / "frame_bank.npz", allow_pickle=False) as archive:
        frame_bank = archive["frame_bank"]
        assert frame_bank.shape == (2, 6)
        assert archive["compass_layer"].item() == 6
        assert archive["dims_per_frame"].item() == 2


def test_calibrate_frames_no_top_level_backend_imports() -> None:
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith("mlx"), (
                    f"top-level mlx import at line {node.lineno}: {name}"
                )
                assert not name.startswith("torch"), (
                    f"top-level torch import at line {node.lineno}: {name}"
                )


def test_calibrate_frames_handler_imports_mlx_lazily() -> None:
    tree = ast.parse(SOURCE.read_text())
    handler = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "context_calibrate_frames_cmd"
    )
    inner_mlx_imports = [
        stmt
        for stmt in ast.walk(handler)
        if isinstance(stmt, ast.Import)
        and any(alias.name.startswith("mlx") for alias in stmt.names)
    ]
    assert inner_mlx_imports, "expected an in-handler mlx import for the MLX-only branch"
