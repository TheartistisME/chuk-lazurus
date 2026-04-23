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
    ModelEngine._load_sync(
        "fake/model",
        False,
        "torch",
        "cuda:0",
        "bounded_kv_direct",
        1024,
        "served-qwen",
    )

    cfg = mock_from_pretrained.call_args.kwargs["pipeline_config"]
    assert cfg is not None
    assert cfg.backend == LazarusBackend.CUDA
    assert cfg.backend_name == "torch"
    assert cfg.device == "cuda:0"
    assert cfg.engine.value == "bounded_kv_direct"
    assert cfg.hot_budget_mib == 1024


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
        ModelEngine.load(
            "fake/model",
            verbose=False,
            backend="torch",
            device="cuda",
            engine="kv_direct",
            hot_budget_mib=2048,
            served_model_name="served-qwen",
        )
    )
    assert engine.model_id == "served-qwen"
    assert engine.source_model_id == "fake/model"
    cfg = mock_from_pretrained.call_args.kwargs["pipeline_config"]
    assert cfg.backend_name == "torch"
    assert cfg.device == "cuda"
    assert cfg.engine.value == "kv_direct"
    assert cfg.hot_budget_mib == 2048


@pytest.mark.asyncio
async def test_astream_uses_torch_runtime_stream(monkeypatch, fake_pipeline):
    from chuk_lazarus.server import engine as engine_module

    runtime = MagicMock()
    runtime.backend = LazarusBackend.CUDA
    fake_pipeline.runtime = runtime

    monkeypatch.setattr(engine_module, "_apply_template", lambda tokenizer, request: "prompt")

    def fake_stream(runtime_obj, prompt, config, *, conversation_id=None):
        assert runtime_obj is runtime
        assert prompt == "prompt"
        assert config.max_new_tokens == 8
        # The WARM residual session cache keying — the default InternalRequest
        # has no conversation_id, so this must be None on the pass-through.
        assert conversation_id is None
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


def test_apply_template_threads_chat_template_kwargs():
    from chuk_lazarus.server.engine import _apply_template

    class DummyTokenizer:
        chat_template = "dummy"

        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "prompt"

    tokenizer = DummyTokenizer()
    request = InternalRequest(
        model="fake/model",
        messages=[InternalMessage(role=MessageRole.USER, content="hi")],
        chat_template_kwargs={"enable_thinking": True},
    )

    prompt = _apply_template(tokenizer, request)

    assert prompt == "prompt"
    _, kwargs = tokenizer.calls[0]
    assert kwargs["enable_thinking"] is True
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True
