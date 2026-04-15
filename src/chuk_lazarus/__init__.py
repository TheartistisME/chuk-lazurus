"""
CHUK-MLX: A unified MLX-based LLM training framework.

This package provides:
- Model loading with LoRA support
- Multiple training paradigms (SFT, DPO, GRPO, PPO)
- Data generation and preprocessing
- Hybrid LLM + RNN expert architecture
- MCP tool integration

Quick Start:
    from chuk_lazarus.models_v2 import LlamaForCausalLM, LlamaConfig
    from chuk_lazarus.training import SFTTrainer
    from chuk_lazarus.data import SFTDataset

    model = LlamaForCausalLM(LlamaConfig.llama2_7b())
    trainer = SFTTrainer(model, tokenizer)
    trainer.train(dataset)

Top-level re-exports are resolved lazily so ``import chuk_lazarus`` stays
runtime-clean under ``CHUK_BACKEND=torch`` until a specific ``models_v2``
symbol is accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0"

if TYPE_CHECKING:  # pragma: no cover
    from .models_v2 import (  # noqa: F401
        CausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        LoRAConfig,
        LoRALinear,
        MambaConfig,
        MambaForCausalLM,
        Model,
        ModelConfig,
        ModelOutput,
        SequenceClassifier,
        TokenClassifier,
        apply_lora,
        compute_lm_loss,
        create_from_preset,
        create_model,
        load_model,
        load_model_async,
    )

__all__ = [
    # Models
    "Model",
    "ModelOutput",
    "ModelConfig",
    "CausalLM",
    "SequenceClassifier",
    "TokenClassifier",
    # Families
    "LlamaConfig",
    "LlamaForCausalLM",
    "MambaConfig",
    "MambaForCausalLM",
    # Loader
    "load_model",
    "load_model_async",
    "create_model",
    "create_from_preset",
    # Adapters
    "LoRAConfig",
    "LoRALinear",
    "apply_lora",
    # Training
    "compute_lm_loss",
]

_LAZY = {name: ("chuk_lazarus.models_v2", name) for name in __all__}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(globals().keys()) + __all__
