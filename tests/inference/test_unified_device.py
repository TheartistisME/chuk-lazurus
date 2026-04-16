"""Device flag propagation tests for UnifiedPipeline._from_pretrained_torch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from chuk_lazarus.inference.loader import DType
from chuk_lazarus.inference.unified import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.registry import ModelFamilyType


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self) -> int:
        return 32


def _make_fake_model(reported_device: str):
    moved = mock.MagicMock()
    moved.eval.return_value = None
    moved.parameters.return_value = iter(
        [SimpleNamespace(device=reported_device, numel=lambda: 4)]
    )
    raw = mock.MagicMock()
    raw.to.return_value = moved
    return raw, moved


def _invoke(device_arg, dtype=DType.BFLOAT16):
    raw, moved = _make_fake_model(device_arg or "cuda")

    with mock.patch(
        "transformers.AutoModelForCausalLM.from_pretrained", return_value=raw
    ) as mock_from_pretrained, mock.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(hidden_size=8, num_hidden_layers=2),
    ), mock.patch(
        "chuk_lazarus.inference.unified.HFLoader.load_tokenizer",
        return_value=_DummyTokenizer(),
    ), mock.patch(
        "chuk_lazarus.inference.unified.TorchInferenceRuntime"
    ) as mock_runtime_cls, mock.patch.object(
        torch.cuda, "is_bf16_supported", return_value=True
    ):
        pipeline = UnifiedPipeline._from_pretrained_torch(
            model_id="dummy/model",
            model_path=Path("/tmp/fake-model"),
            hf_config={"model_type": "llama", "architectures": ["LlamaForCausalLM"]},
            family_type=ModelFamilyType.LLAMA,
            pipeline_config=UnifiedPipelineConfig(dtype=dtype, device=device_arg),
            verbose=False,
        )

    return pipeline, raw, moved, mock_runtime_cls, mock_from_pretrained


def test_device_cuda_1_threads_through():
    _pipe, raw, _moved, mock_runtime_cls, mock_from_pretrained = _invoke("cuda:1")

    raw.to.assert_called_once_with("cuda:1", non_blocking=True)
    _args, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "cuda:1"
    assert mock_from_pretrained.call_args.kwargs["dtype"] is torch.bfloat16


def test_device_cpu_forces_float32_and_no_non_blocking():
    _pipe, raw, _moved, mock_runtime_cls, mock_from_pretrained = _invoke(
        "cpu", dtype=DType.BFLOAT16
    )

    raw.to.assert_called_once_with("cpu")
    assert "non_blocking" not in raw.to.call_args.kwargs
    _args, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "cpu"
    assert mock_from_pretrained.call_args.kwargs["dtype"] is torch.float32


def test_device_default_none_uses_cuda():
    _pipe, raw, _moved, mock_runtime_cls, _ = _invoke(None)

    raw.to.assert_called_once_with("cuda", non_blocking=True)
    _args, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "cuda"


def test_device_mps_uses_float16():
    _pipe, raw, _moved, mock_runtime_cls, mock_from_pretrained = _invoke(
        "mps", dtype=DType.BFLOAT16
    )

    raw.to.assert_called_once_with("mps")
    assert mock_from_pretrained.call_args.kwargs["dtype"] is torch.float16
    _args, kwargs = mock_runtime_cls.call_args
    assert kwargs["device"] == "mps"
