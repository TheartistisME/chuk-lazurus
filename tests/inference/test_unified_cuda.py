"""Torch/CUDA import seam tests for unified family loading."""

from __future__ import annotations

import importlib
import sys


def _clear_modules(*prefixes: str) -> None:
    for prefix in prefixes:
        for name in list(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)


def test_torch_host_can_touch_gemma_family_package(monkeypatch):
    """Gemma package metadata should stay importable without MLX on torch hosts."""
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    _clear_modules("chuk_lazarus.models_v2.families.gemma", "mlx")

    gemma = importlib.import_module("chuk_lazarus.models_v2.families.gemma")

    assert "chuk_lazarus.models_v2.families.gemma.convert" not in sys.modules
    assert "mlx.core" not in sys.modules
    assert gemma.GemmaConfig.__name__ == "GemmaConfig"
    assert "chuk_lazarus.models_v2.families.gemma.convert" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_torch_host_can_resolve_gpt2_config_without_loading_mlx(monkeypatch):
    """Registry config lookup should not execute eager family package imports."""
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    _clear_modules(
        "chuk_lazarus.models_v2.families.registry",
        "chuk_lazarus.models_v2.families.gpt2",
        "chuk_lazarus.models_v2.families.gemma",
        "mlx",
    )

    registry = importlib.import_module("chuk_lazarus.models_v2.families.registry")

    assert registry.detect_model_family(
        {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}
    ) == registry.ModelFamilyType.GPT2

    family_info = registry.get_family_info(registry.ModelFamilyType.GPT2)
    assert family_info is not None

    config = family_info.config_class.from_hf_config(
        {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 32,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "n_positions": 16,
        }
    )

    assert config.model_type == "gpt2"
    assert "chuk_lazarus.models_v2.families.gpt2.convert" not in sys.modules
    assert "chuk_lazarus.models_v2.families.gemma.convert" not in sys.modules
    assert "mlx.core" not in sys.modules


def test_torch_host_detects_qwen35_moe_as_qwen_family(monkeypatch):
    """Qwen3.5/3.6 MoE checkpoints should route through the Qwen family."""
    monkeypatch.setenv("CHUK_BACKEND", "torch")
    _clear_modules("chuk_lazarus.models_v2.families.registry", "mlx")

    registry = importlib.import_module("chuk_lazarus.models_v2.families.registry")

    assert registry.detect_model_family(
        {"model_type": "qwen3_5_moe"}
    ) == registry.ModelFamilyType.QWEN3
    assert registry.detect_model_family(
        {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
    ) == registry.ModelFamilyType.QWEN3
