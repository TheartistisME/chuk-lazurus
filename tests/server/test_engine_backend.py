"""EWS-8: ModelEngine.load threads backend/device into UnifiedPipelineConfig."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.inference.backends.types import LazarusBackend
from chuk_lazarus.server.engine import ModelEngine
from chuk_lazarus.server.schemas.internal import (
    FinishReason,
    InternalMessage,
    InternalRequest,
    MessageRole,
)


@pytest.fixture
def fake_pipeline() -> MagicMock:
    p = MagicMock()
    p.tokenizer = MagicMock()
    p.model = MagicMock()
    return p


@patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
def test_load_sync_without_backend_passes_none_config(
    mock_from_pretrained, fake_pipeline
):
    mock_from_pretrained.return_value = fake_pipeline

    engine = ModelEngine._load_sync("fake/model", verbose=False)

    assert isinstance(engine, ModelEngine)
    mock_from_pretrained.assert_called_once()
    _, kwargs = mock_from_pretrained.call_args
    assert kwargs["pipeline_config"] is None


@patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
def test_load_sync_torch_backend_maps_to_cuda(mock_from_pretrained, fake_pipeline):
    from chuk_lazarus.inference.backends.types import LazarusBackend

    mock_from_pretrained.return_value = fake_pipeline
    ModelEngine._load_sync("fake/model", False, "torch", "cuda:0")

    cfg = mock_from_pretrained.call_args.kwargs["pipeline_config"]
    assert cfg is not None
    assert cfg.backend == LazarusBackend.CUDA
    assert cfg.backend_name == "torch"
    assert cfg.device == "cuda:0"


@patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
def test_load_sync_mlx_backend_preserves_enum(mock_from_pretrained, fake_pipeline):
    from chuk_lazarus.inference.backends.types import LazarusBackend

    mock_from_pretrained.return_value = fake_pipeline
    ModelEngine._load_sync("fake/model", False, "mlx", "mps")

    cfg = mock_from_pretrained.call_args.kwargs["pipeline_config"]
    assert cfg.backend == LazarusBackend.MLX
    assert cfg.backend_name == "mlx"
    assert cfg.device == "mps"


@patch("chuk_lazarus.inference.UnifiedPipeline.from_pretrained")
def test_load_async_threads_backend_kwarg(mock_from_pretrained, fake_pipeline):
    mock_from_pretrained.return_value = fake_pipeline

    engine = asyncio.run(
        ModelEngine.load("fake/model", verbose=False, backend="torch", device="cuda")
    )
    assert engine.model_id == "fake/model"
    cfg = mock_from_pretrained.call_args.kwargs["pipeline_config"]
    assert cfg.backend_name == "torch"
    assert cfg.device == "cuda"


@pytest.mark.asyncio
async def test_astream_uses_torch_runtime_stream(monkeypatch, fake_pipeline):
    from chuk_lazarus.server import engine as engine_module

    runtime = MagicMock()
    runtime.backend = LazarusBackend.CUDA
    fake_pipeline.runtime = runtime

    monkeypatch.setattr(engine_module, "_apply_template", lambda tokenizer, request: "prompt")

    def fake_stream(runtime_obj, prompt, config):
        assert runtime_obj is runtime
        assert prompt == "prompt"
        assert config.max_new_tokens == 8
        yield "hello"
        yield " world"

    monkeypatch.setattr(engine_module, "_stream_tokens_torch", fake_stream)

    engine = ModelEngine(fake_pipeline, "fake/model")
    request = InternalRequest(
        model="fake/model",
        max_tokens=8,
        messages=[InternalMessage(role=MessageRole.USER, content="hi")],
    )

    chunks = [chunk async for chunk in engine.astream(request)]

    assert [chunk.content for chunk in chunks[:-1]] == ["hello", " world"]
    assert chunks[-1].content is None
    assert chunks[-1].finish_reason == FinishReason.STOP
