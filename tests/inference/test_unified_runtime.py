"""Focused runtime-dispatch tests for UnifiedPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from chuk_lazarus.inference.chat import ChatHistory
from chuk_lazarus.inference.generation import GenerationResult, GenerationStats, StopReason
from chuk_lazarus.inference.unified import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families import FamilyInfo, ModelFamilyType


class _DummyConfig(BaseModel):
    hidden_size: int = 8
    num_hidden_layers: int = 2


class _DummyTokenizer:
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"formatted:{len(messages)}:{add_generation_prompt}"


def _family_info() -> FamilyInfo:
    return FamilyInfo(
        family_type=ModelFamilyType.LLAMA,
        config_class=_DummyConfig,
        model_class=MagicMock,
        model_types=["llama"],
        architectures=["LlamaForCausalLM"],
    )


def _generation_result(text: str = "ok") -> GenerationResult:
    return GenerationResult(
        text=text,
        stats=GenerationStats(
            input_tokens=3,
            output_tokens=2,
            total_time_seconds=0.1,
            tokens_per_second=20.0,
        ),
        stop_reason=StopReason.EOS,
    )


def test_generate_paths_use_runtime_object() -> None:
    runtime = MagicMock()
    expected = _generation_result()
    runtime.generate.return_value = expected

    pipeline = UnifiedPipeline(
        model=MagicMock(),
        tokenizer=_DummyTokenizer(),
        model_config=_DummyConfig(),
        family_info=_family_info(),
        runtime=runtime,
    )

    history = ChatHistory()
    history.add_user("hello")
    history.add_assistant("hi")

    assert pipeline.chat("tell me more") == expected
    assert pipeline.chat_with_history(history) == expected
    assert pipeline.generate("raw prompt") == expected
    assert runtime.generate.call_count == 3


@patch("chuk_lazarus.inference.unified.TorchInferenceRuntime")
def test_constructor_selects_torch_runtime_from_pipeline_config(mock_runtime_cls) -> None:
    runtime = MagicMock()
    mock_runtime_cls.return_value = runtime

    pipeline = UnifiedPipeline(
        model=MagicMock(),
        tokenizer=_DummyTokenizer(),
        model_config=_DummyConfig(),
        family_info=_family_info(),
        pipeline_config=UnifiedPipelineConfig(backend_name="torch", device="cuda:2"),
    )

    assert pipeline.runtime is runtime
    _, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "cuda:2"


@patch.object(UnifiedPipeline, "_from_pretrained_torch")
@patch("chuk_lazarus.inference.unified.detect_model_family")
@patch("chuk_lazarus.inference.unified.HFLoader.download")
def test_from_pretrained_dispatches_torch_backend(
    mock_download,
    mock_detect,
    mock_torch_loader,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    hf_config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    config_path.write_text(json.dumps(hf_config))

    mock_download.return_value = SimpleNamespace(model_path=tmp_path)
    mock_detect.return_value = ModelFamilyType.LLAMA
    sentinel = MagicMock()
    mock_torch_loader.return_value = sentinel

    pipeline = UnifiedPipeline.from_pretrained(
        "org/model",
        pipeline_config=UnifiedPipelineConfig(backend_name="torch", device="cuda:1"),
        verbose=False,
    )

    assert pipeline is sentinel
    _, kwargs = mock_torch_loader.call_args
    assert kwargs["model_id"] == "org/model"
    assert kwargs["model_path"] == tmp_path
    assert kwargs["hf_config"] == hf_config
    assert kwargs["family_type"] == ModelFamilyType.LLAMA
    assert kwargs["pipeline_config"].backend_name == "torch"
    assert kwargs["pipeline_config"].device == "cuda:1"
