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


def test_generate_threads_conversation_id_when_supplied() -> None:
    """``conversation_id`` flows verbatim to the runtime when non-None."""
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

    assert pipeline.generate("raw prompt", conversation_id="alice") == expected
    _, kwargs = runtime.generate.call_args
    assert kwargs.get("conversation_id") == "alice"


def test_generate_omits_conversation_id_for_legacy_runtimes() -> None:
    """Runtimes whose ``generate`` doesn't accept the kwarg still work.

    Mirrors the MLX runtime surface, which is deliberately unaware of the
    torch-only session cache.
    """
    runtime = MagicMock()
    expected = _generation_result()

    call_log: list[dict] = []

    def _generate(prompt, config, **kwargs):
        if "conversation_id" in kwargs:
            raise TypeError("legacy runtime: no conversation_id kwarg")
        call_log.append({"prompt": prompt, "config": config})
        return expected

    runtime.generate.side_effect = _generate

    pipeline = UnifiedPipeline(
        model=MagicMock(),
        tokenizer=_DummyTokenizer(),
        model_config=_DummyConfig(),
        family_info=_family_info(),
        runtime=runtime,
    )

    assert pipeline.generate("raw prompt", conversation_id="alice") == expected
    # Two calls on the mock: one TypeError attempt, one fallback. The
    # side_effect records only the successful fallback.
    assert len(call_log) == 1
    assert call_log[0]["prompt"] == "raw prompt"


@patch("chuk_lazarus.inference.unified.TorchInferenceRuntime")
def test_constructor_selects_torch_runtime_from_pipeline_config(mock_runtime_cls) -> None:
    runtime = MagicMock()
    mock_runtime_cls.return_value = runtime

    pipeline = UnifiedPipeline(
        model=MagicMock(),
        tokenizer=_DummyTokenizer(),
        model_config=_DummyConfig(),
        family_info=_family_info(),
        pipeline_config=UnifiedPipelineConfig(
            backend_name="torch",
            device="cuda:2",
            engine="bounded_kv_direct",
            hot_budget_mib=1024,
        ),
    )

    assert pipeline.runtime is runtime
    _, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "cuda:2"
    assert kwargs["engine"] == "bounded_kv_direct"
    assert kwargs["hot_budget_mib"] == 1024


@patch("chuk_lazarus.inference.unified.TorchInferenceRuntime")
def test_constructor_threads_residual_bounded_engine_through(mock_runtime_cls) -> None:
    """The residual-backed bounded KV-direct mode must propagate through the
    pipeline config unchanged — this is the Qwen Pi serve path."""
    runtime = MagicMock()
    mock_runtime_cls.return_value = runtime

    pipeline = UnifiedPipeline(
        model=MagicMock(),
        tokenizer=_DummyTokenizer(),
        model_config=_DummyConfig(),
        family_info=_family_info(),
        pipeline_config=UnifiedPipelineConfig(
            backend_name="torch",
            device="cuda:0",
            engine="residual_bounded_kv_direct",
            hot_budget_mib=150,
        ),
    )

    assert pipeline.runtime is runtime
    _, kwargs = mock_runtime_cls.call_args
    assert kwargs["engine"] == "residual_bounded_kv_direct"
    assert kwargs["hot_budget_mib"] == 150


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


# ---------------------------------------------------------------------------
# Axis-5 leg-1 learning: ``_from_pretrained_torch`` must thread
# ``pipeline_config.hot_budget_mib`` into the real ``TorchInferenceRuntime``
# constructor. Prior to this fix the kwarg was silently dropped on the
# ``from_pretrained`` path, leaving a runtime with ``_hot_budget_bytes=None``
# after loading. This test exercises the integration seam with all heavy
# transformers machinery stubbed out — the model/tokenizer stubs are what
# ``_from_pretrained_torch`` reads when it constructs the runtime.
# ---------------------------------------------------------------------------


def test_from_pretrained_torch_threads_hot_budget_mib_to_runtime(
    tmp_path: Path,
) -> None:
    """Regression: ``hot_budget_mib`` flows from pipeline_config → runtime.

    Leg-1 learning (ve-ins-0moa09rhm00002c707e) called out that
    ``_from_pretrained_torch`` previously dropped ``hot_budget_mib``. The
    unified.py fix at line 514 passes it through. This test patches the
    heavy transformer+loader imports and verifies the real
    ``TorchInferenceRuntime`` constructor receives the configured MiB
    budget, rounded into bytes on the resulting runtime.
    """
    # Torch is needed only by the real TorchInferenceRuntime class. Skip
    # when it isn't importable (matches the rest of the file's CPU-runnable
    # posture — no CUDA device is required).
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - defensive
        import pytest

        pytest.skip("torch not available for _from_pretrained_torch threading test")

    fake_model = MagicMock(name="fake_model")
    fake_model.eval.return_value = fake_model
    fake_model.to.return_value = fake_model
    fake_model.parameters.return_value = iter([])

    fake_tokenizer = MagicMock(name="fake_tokenizer")
    fake_tokenizer.__len__ = MagicMock(return_value=32000)

    fake_auto_config = MagicMock(name="fake_auto_config")
    fake_auto_config.from_pretrained.return_value = SimpleNamespace(
        hidden_size=128,
        num_hidden_layers=4,
        text_config=None,
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )
    fake_auto_model_cls = MagicMock(name="fake_auto_model")
    fake_auto_model_cls.from_pretrained.return_value = fake_model

    captured: dict = {}

    class _RecordingRuntime:
        def __init__(self, model, tokenizer, *, device, engine, family_type, hot_budget_mib):
            captured["model"] = model
            captured["tokenizer"] = tokenizer
            captured["device"] = device
            captured["engine"] = engine
            captured["family_type"] = family_type
            captured["hot_budget_mib"] = hot_budget_mib
            # Mirror the real runtime's byte conversion so callers can read
            # the derived ``_hot_budget_bytes`` attribute.
            self._hot_budget_bytes = (
                int(hot_budget_mib) * 1024 * 1024 if hot_budget_mib is not None else None
            )

    pipeline_config = UnifiedPipelineConfig(
        backend_name="torch",
        device="cpu",  # avoid real CUDA gating inside _from_pretrained_torch
        engine="bounded_kv_direct",
        hot_budget_mib=512,
    )

    with patch(
        "chuk_lazarus.inference.unified.TorchInferenceRuntime",
        _RecordingRuntime,
    ), patch(
        "chuk_lazarus.inference.unified.HFLoader.load_tokenizer",
        return_value=fake_tokenizer,
    ), patch(
        "transformers.AutoConfig", fake_auto_config
    ), patch(
        "transformers.AutoModelForCausalLM", fake_auto_model_cls
    ):
        pipeline = UnifiedPipeline._from_pretrained_torch(
            model_id="org/fake",
            model_path=tmp_path,
            hf_config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
            family_type=ModelFamilyType.LLAMA,
            pipeline_config=pipeline_config,
            verbose=False,
        )

    # The runtime the pipeline uses IS the recording stub.
    assert isinstance(pipeline.runtime, _RecordingRuntime)
    # The kwarg was threaded verbatim.
    assert captured["hot_budget_mib"] == 512
    assert captured["engine"] == "bounded_kv_direct"
    # Downstream derived field matches the documented 1 MiB = 1024*1024 bytes.
    assert pipeline.runtime._hot_budget_bytes == 512 * 1024 * 1024
