"""Regression guards for the Qwen3.5/3.6-MoE compressed-tensors load path.

These tests pin down the two invariants that were violated by the previous
``device_map={"":...} + low_cpu_mem_usage=True`` configuration on the
compressed-tensors branch of ``UnifiedPipeline._from_pretrained_torch``:

1. On a compressed-tensors Qwen3.5-MoE checkpoint the loader MUST NOT pass
   ``device_map=`` or ``low_cpu_mem_usage=`` to ``from_pretrained``.  Those
   flags together route through a meta-init that fires before the
   compressed-tensors quantizer walks modules, which silently leaves every
   expert projection random-initialised.
2. The loader MUST capture the HF loading-info report and fail loud when
   expert weight keys land in ``missing_keys`` — rather than serving a
   model whose experts are untrained and whose forward hangs inside the
   Python per-expert dispatch loop.

Both checks run without loading a real model.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chuk_lazarus.inference import unified


# ---------------------------------------------------------------------------
# _raise_if_experts_random_init
# ---------------------------------------------------------------------------


class TestRaiseIfExpertsRandomInit:
    def test_raises_when_expert_projection_weight_is_missing(self) -> None:
        """The canonical symptom: missing expert weight keys.

        This is exactly what the Infatoshi Qwen3.6-35B NVFP4 checkpoint
        used to report under the broken ``device_map`` config.
        """
        info = {
            "missing_keys": [
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
                "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            ],
            "unexpected_keys": [],
        }
        with pytest.raises(RuntimeError, match="random-initialised"):
            unified._raise_if_experts_random_init(info)

    def test_raises_when_some_experts_missing(self) -> None:
        info = {
            "missing_keys": [
                "model.language_model.layers.17.mlp.experts.42.gate_proj.weight",
                "model.embed_tokens.weight",
            ],
            "unexpected_keys": ["model.layers.17.mlp.experts.42.gate_proj.weight_packed"],
        }
        with pytest.raises(RuntimeError) as excinfo:
            unified._raise_if_experts_random_init(info)

        message = str(excinfo.value)
        # The error message should point the operator at the real fix.
        assert "device_map" in message
        assert "low_cpu_mem_usage" in message
        assert "CompressedLinear" in message

    def test_silent_when_no_expert_weights_missing(self) -> None:
        """Benign missing keys (tied embeddings, buffers, etc.) must not raise."""
        info = {
            "missing_keys": [
                "model.lm_head.weight",
                "model.embed_tokens.weight",
            ],
            "unexpected_keys": [],
        }
        # Must not raise.
        unified._raise_if_experts_random_init(info)

    def test_silent_when_loading_info_is_none_or_empty(self) -> None:
        unified._raise_if_experts_random_init({})
        unified._raise_if_experts_random_init({"missing_keys": [], "unexpected_keys": []})

    def test_logs_ok_line_when_log_hook_supplied(self) -> None:
        captured: list[str] = []
        unified._raise_if_experts_random_init(
            {"missing_keys": ["other.key"], "unexpected_keys": ["foo.bar"]},
            log=captured.append,
        )
        assert captured, "log hook should have been called with the load report"
        assert "missing=1" in captured[0]
        assert "unexpected=1" in captured[0]
        assert "expert-weight MISSING keys: 0" in captured[0]

    def test_ignores_non_dict_loading_info(self) -> None:
        # ``output_loading_info=False`` means the tuple-return is a model
        # only; defend against that surface change.
        unified._raise_if_experts_random_init(None)  # type: ignore[arg-type]
        unified._raise_if_experts_random_init("not-a-dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# UnifiedPipeline._from_pretrained_torch — compressed-tensors branch contract
# ---------------------------------------------------------------------------


def _family_info():
    from chuk_lazarus.models_v2.families import FamilyInfo, ModelFamilyType

    return FamilyInfo(
        family_type=ModelFamilyType.QWEN3,
        config_class=MagicMock,
        model_class=MagicMock,
        model_types=["qwen3_5_moe"],
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )


def _run_compressed_tensors_load(tmp_path: Path, loading_info: dict):
    """Drive UnifiedPipeline._from_pretrained_torch down the compressed-tensors branch.

    Patches only the two HF entry points and the quant-method sniffer —
    ``torch`` itself is real (faking it via ``sys.modules`` collides with
    ``transformers``' own import-time detection).
    """
    from chuk_lazarus.inference.unified import (
        UnifiedPipeline,
        UnifiedPipelineConfig,
    )
    from chuk_lazarus.models_v2.families import ModelFamilyType

    hf_config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    }

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    fake_model.parameters.return_value = iter(
        [MagicMock(numel=MagicMock(return_value=0))]
    )

    fake_auto_config = SimpleNamespace(
        quantization_config={"quant_method": "compressed-tensors"},
        model_type="qwen3_5_moe",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        hidden_size=4096,
        num_hidden_layers=40,
        text_config=None,
    )

    captured_kwargs: dict = {}

    def fake_from_pretrained(path, **kwargs):
        captured_kwargs.update(kwargs)
        if kwargs.get("output_loading_info"):
            return fake_model, loading_info
        return fake_model

    with (
        patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=fake_auto_config,
        ),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained",
            side_effect=fake_from_pretrained,
        ),
        patch(
            "chuk_lazarus.inference.backends._torch_qwen_support.install_qwen3_5_moe_per_expert_patch",
            return_value="installed",
        ),
        patch(
            "chuk_lazarus.inference.backends._torch_qwen_support.install_blackwell_causal_conv1d_workaround",
            return_value="installed",
        ),
        patch(
            "chuk_lazarus.inference.backends._torch_qwen_support.compressed_tensors_quant_method",
            return_value="compressed-tensors",
        ),
        patch(
            "chuk_lazarus.inference.unified.HFLoader.load_tokenizer",
            return_value=MagicMock(__len__=lambda self: 1234),
        ),
        patch(
            "chuk_lazarus.inference.unified.TorchInferenceRuntime",
            return_value=MagicMock(),
        ),
    ):
        from chuk_lazarus.inference.backends.types import LazarusBackend

        pipeline = UnifiedPipeline._from_pretrained_torch(
            model_id="Infatoshi/Qwen3.6-35B-A3B-NVFP4-FP8",
            model_path=tmp_path,
            hf_config=hf_config,
            family_type=ModelFamilyType.QWEN3,
            pipeline_config=UnifiedPipelineConfig(
                backend=LazarusBackend.CUDA,
                device="cuda:0",
            ),
            verbose=False,
        )

    return pipeline, captured_kwargs, fake_model


class TestCompressedTensorsLoadContract:
    """Pin the exact kwargs the loader passes on the compressed-tensors branch."""

    def test_does_not_pass_device_map_or_low_cpu_mem_usage(self, tmp_path: Path) -> None:
        """Regression guard: these two kwargs are the ones that broke MoE load.

        Meta-init sequence fires BEFORE the compressed-tensors quantizer
        walks modules, leaving patched per-expert Linears on ``meta`` and
        defeating ``CompressedLinear.from_linear``'s ``delattr``.  Never
        again.
        """
        pipeline, kwargs, _ = _run_compressed_tensors_load(
            tmp_path,
            loading_info={"missing_keys": [], "unexpected_keys": []},
        )
        assert pipeline is not None
        assert "device_map" not in kwargs, (
            f"device_map must NOT be passed on the compressed-tensors branch "
            f"(meta-init defeats compressed-tensors quantizer); got kwargs={sorted(kwargs)}"
        )
        assert "low_cpu_mem_usage" not in kwargs, (
            "low_cpu_mem_usage must NOT be True on the compressed-tensors branch"
        )

    def test_passes_output_loading_info_and_fp16(self, tmp_path: Path) -> None:
        _, kwargs, _ = _run_compressed_tensors_load(
            tmp_path,
            loading_info={"missing_keys": [], "unexpected_keys": []},
        )
        assert kwargs.get("output_loading_info") is True, (
            "output_loading_info=True is required so the loader can assert "
            "no expert weights were left random-initialised."
        )
        import torch as _real_torch

        assert kwargs.get("dtype") is _real_torch.float16, (
            "Compressed-tensors branch must load in fp16 (HF's own recommendation)."
        )

    def test_raises_when_loader_returns_missing_expert_weights(self, tmp_path: Path) -> None:
        """End-to-end: if HF ever loads an NVFP4 MoE with experts missing, fail."""
        with pytest.raises(RuntimeError, match="random-initialised"):
            _run_compressed_tensors_load(
                tmp_path,
                loading_info={
                    "missing_keys": [
                        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                    ],
                    "unexpected_keys": [],
                },
            )

    def test_load_succeeds_and_moves_to_device_when_experts_present(
        self, tmp_path: Path
    ) -> None:
        pipeline, _, fake_model = _run_compressed_tensors_load(
            tmp_path,
            loading_info={"missing_keys": [], "unexpected_keys": []},
        )
        # ``.to(cuda)`` must be called explicitly — no device_map means the
        # model is born on CPU and needs the move to land on GPU.
        fake_model.to.assert_called_once()
        args, kwargs = fake_model.to.call_args
        assert args == ("cuda:0",) or kwargs.get("device") == "cuda:0"
        assert kwargs.get("non_blocking") is True
