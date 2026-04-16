"""
Unified Model Architecture for chuk-mlx.

A composable, async-native, Pydantic-native model framework supporting:
- Transformers (Llama, Mistral, Gemma, and compatible)
- State Space Models (Mamba)
- Recurrent Networks (LSTM, GRU, MinGRU)
- Hybrid Architectures
- Classifiers

Design Principles:
- Async-native: All I/O operations are async
- Pydantic-native: All configs use BaseModel for validation
- No magic strings: Enums and constants for type safety
- No dictionary goop: Structured types throughout
- Backend-agnostic: Works on MLX, PyTorch, JAX

Architecture:
    Components -> Blocks -> Backbones -> Heads -> Models
                                   \\       /
                                    Families

Components: Embeddings, Attention, FFN, Normalization, SSM, Recurrent
Blocks: Transformer, Mamba, Recurrent, Hybrid
Backbones: Stack of blocks with embeddings
Heads: LM, Classifier, Regression
Models: Complete end-to-end models
Families: Architecture-specific implementations (Llama, Mamba, etc.)

Top-level mlx imports are intentionally absent.  Every re-exported
symbol is resolved lazily via PEP 562 module ``__getattr__`` so that
``import chuk_lazarus.models_v2`` succeeds under ``CHUK_BACKEND=torch``
without pulling ``libmlx.so`` until a symbol is actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    # Re-exports (import paths) — here only for IDE/type-checker support.
    from .adapters import (  # noqa: F401
        LoRAConfig,
        LoRALinear,
        apply_lora,
        count_lora_parameters,
        merge_lora_weights,
    )
    from .backbones import (  # noqa: F401
        Backbone,
        BackboneOutput,
        HybridBackbone,
        MambaBackbone,
        RecurrentBackbone,
        TransformerBackbone,
    )
    from .blocks import (  # noqa: F401
        Block,
        BlockOutput,
        HybridBlock,
        MambaBlockWrapper,
        RecurrentBlockWrapper,
        TransformerBlock,
    )
    from .components.attention import (  # noqa: F401
        GroupedQueryAttention,
        MultiHeadAttention,
        SlidingWindowAttention,
    )
    from .components.embeddings import (  # noqa: F401
        ALiBi,
        LearnedPositionEmbedding,
        RoPE,
        SinusoidalPositionEmbedding,
        TokenEmbedding,
    )
    from .components.ffn import GEGLU, MLP, MoE, SwiGLU  # noqa: F401
    from .components.normalization import GemmaNorm, LayerNorm, RMSNorm  # noqa: F401
    from .components.recurrent import GRU, LSTM, MinGRU  # noqa: F401
    from .components.ssm import Mamba, MambaBlock, SelectiveSSM  # noqa: F401
    from .core import (  # noqa: F401
        ActivationType,
        AttentionConfig,
        AttentionType,
        BackboneType,
        BackendType,
        BlockType,
        FFNConfig,
        FFNType,
        HeadType,
        ModelCapability,
        ModelConfig,
        ModelMode,
        NormConfig,
        NormType,
        PositionEmbeddingType,
        SSMConfig,
        find_models_by_capability,
        get_backend,
        get_factory,
        get_model_capabilities,
        list_models,
        register_model,
        set_backend,
    )
    from .families import granite, llama, llama4, mamba, qwen3  # noqa: F401
    from .families.granite import (  # noqa: F401
        GraniteConfig,
        GraniteForCausalLM,
        GraniteHybridConfig,
        GraniteHybridForCausalLM,
    )
    from .families.llama import LlamaConfig, LlamaForCausalLM  # noqa: F401
    from .families.llama4 import (  # noqa: F401
        Llama4Config,
        Llama4ForCausalLM,
        Llama4TextConfig,
    )
    from .families.mamba import MambaConfig, MambaForCausalLM  # noqa: F401
    from .families.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: F401
    from .heads import (  # noqa: F401
        ClassifierHead,
        Head,
        HeadOutput,
        LMHead,
        RegressionHead,
    )
    from .introspection import (  # noqa: F401
        FLOPsEstimate,
        MemoryEstimate,
        ModelCapabilities,
        ModelInfo,
        ParameterStats,
        count_parameters,
        detect_model_capabilities,
        estimate_flops,
        estimate_memory,
        get_model_info,
        introspect,
        print_introspection,
    )
    from .loader import (  # noqa: F401
        AdapterConfig,
        LoadedModel,
        LoadedModelWithLoRA,
        ModelDType,
        create_from_preset,
        create_model,
        load_model,
        load_model_async,
        load_model_tuple,
        load_model_with_lora,
        load_model_with_lora_async,
        save_adapter,
    )
    from .losses import compute_lm_loss  # noqa: F401
    from .models import (  # noqa: F401
        CausalLM,
        Model,
        ModelOutput,
        SequenceClassifier,
        TokenClassifier,
    )


# Map of public name -> (relative submodule path, attribute to pull).
# Kept as a plain dict so PEP 562 ``__getattr__`` can look symbols up in
# constant time without hitting multiple submodules unnecessarily.
_LAZY: dict[str, tuple[str, str]] = {
    # Core — enums, configs, registry, backend
    "ModelMode": (".core", "ModelMode"),
    "BlockType": (".core", "BlockType"),
    "BackboneType": (".core", "BackboneType"),
    "HeadType": (".core", "HeadType"),
    "AttentionType": (".core", "AttentionType"),
    "FFNType": (".core", "FFNType"),
    "NormType": (".core", "NormType"),
    "ActivationType": (".core", "ActivationType"),
    "PositionEmbeddingType": (".core", "PositionEmbeddingType"),
    "BackendType": (".core", "BackendType"),
    "ModelConfig": (".core", "ModelConfig"),
    "AttentionConfig": (".core", "AttentionConfig"),
    "FFNConfig": (".core", "FFNConfig"),
    "NormConfig": (".core", "NormConfig"),
    "SSMConfig": (".core", "SSMConfig"),
    "register_model": (".core", "register_model"),
    "get_factory": (".core", "get_factory"),
    "list_models": (".core", "list_models"),
    "ModelCapability": (".core", "ModelCapability"),
    "get_model_capabilities": (".core", "get_model_capabilities"),
    "find_models_by_capability": (".core", "find_models_by_capability"),
    "get_backend": (".core", "get_backend"),
    "set_backend": (".core", "set_backend"),
    # Components — embeddings
    "TokenEmbedding": (".components.embeddings", "TokenEmbedding"),
    "RoPE": (".components.embeddings", "RoPE"),
    "ALiBi": (".components.embeddings", "ALiBi"),
    "LearnedPositionEmbedding": (".components.embeddings", "LearnedPositionEmbedding"),
    "SinusoidalPositionEmbedding": (
        ".components.embeddings",
        "SinusoidalPositionEmbedding",
    ),
    # Components — attention
    "MultiHeadAttention": (".components.attention", "MultiHeadAttention"),
    "GroupedQueryAttention": (".components.attention", "GroupedQueryAttention"),
    "SlidingWindowAttention": (".components.attention", "SlidingWindowAttention"),
    # Components — FFN
    "MLP": (".components.ffn", "MLP"),
    "SwiGLU": (".components.ffn", "SwiGLU"),
    "GEGLU": (".components.ffn", "GEGLU"),
    "MoE": (".components.ffn", "MoE"),
    # Components — normalization
    "RMSNorm": (".components.normalization", "RMSNorm"),
    "LayerNorm": (".components.normalization", "LayerNorm"),
    "GemmaNorm": (".components.normalization", "GemmaNorm"),
    # Components — SSM / recurrent
    "SelectiveSSM": (".components.ssm", "SelectiveSSM"),
    "Mamba": (".components.ssm", "Mamba"),
    "MambaBlock": (".components.ssm", "MambaBlock"),
    "LSTM": (".components.recurrent", "LSTM"),
    "GRU": (".components.recurrent", "GRU"),
    "MinGRU": (".components.recurrent", "MinGRU"),
    # Blocks
    "Block": (".blocks", "Block"),
    "BlockOutput": (".blocks", "BlockOutput"),
    "TransformerBlock": (".blocks", "TransformerBlock"),
    "MambaBlockWrapper": (".blocks", "MambaBlockWrapper"),
    "RecurrentBlockWrapper": (".blocks", "RecurrentBlockWrapper"),
    "HybridBlock": (".blocks", "HybridBlock"),
    # Backbones
    "Backbone": (".backbones", "Backbone"),
    "BackboneOutput": (".backbones", "BackboneOutput"),
    "TransformerBackbone": (".backbones", "TransformerBackbone"),
    "MambaBackbone": (".backbones", "MambaBackbone"),
    "RecurrentBackbone": (".backbones", "RecurrentBackbone"),
    "HybridBackbone": (".backbones", "HybridBackbone"),
    # Heads
    "Head": (".heads", "Head"),
    "HeadOutput": (".heads", "HeadOutput"),
    "LMHead": (".heads", "LMHead"),
    "ClassifierHead": (".heads", "ClassifierHead"),
    "RegressionHead": (".heads", "RegressionHead"),
    # Models
    "Model": (".models", "Model"),
    "ModelOutput": (".models", "ModelOutput"),
    "CausalLM": (".models", "CausalLM"),
    "SequenceClassifier": (".models", "SequenceClassifier"),
    "TokenClassifier": (".models", "TokenClassifier"),
    # Families (submodules exposed as attributes)
    "granite": (".families", "granite"),
    "llama": (".families", "llama"),
    "llama4": (".families", "llama4"),
    "mamba": (".families", "mamba"),
    "qwen3": (".families", "qwen3"),
    "GraniteConfig": (".families.granite", "GraniteConfig"),
    "GraniteForCausalLM": (".families.granite", "GraniteForCausalLM"),
    "GraniteHybridConfig": (".families.granite", "GraniteHybridConfig"),
    "GraniteHybridForCausalLM": (".families.granite", "GraniteHybridForCausalLM"),
    "LlamaConfig": (".families.llama", "LlamaConfig"),
    "LlamaForCausalLM": (".families.llama", "LlamaForCausalLM"),
    "Llama4Config": (".families.llama4", "Llama4Config"),
    "Llama4TextConfig": (".families.llama4", "Llama4TextConfig"),
    "Llama4ForCausalLM": (".families.llama4", "Llama4ForCausalLM"),
    "MambaConfig": (".families.mamba", "MambaConfig"),
    "MambaForCausalLM": (".families.mamba", "MambaForCausalLM"),
    "Qwen3Config": (".families.qwen3", "Qwen3Config"),
    "Qwen3ForCausalLM": (".families.qwen3", "Qwen3ForCausalLM"),
    # Loader
    "ModelDType": (".loader", "ModelDType"),
    "LoadedModel": (".loader", "LoadedModel"),
    "LoadedModelWithLoRA": (".loader", "LoadedModelWithLoRA"),
    "AdapterConfig": (".loader", "AdapterConfig"),
    "load_model": (".loader", "load_model"),
    "load_model_async": (".loader", "load_model_async"),
    "load_model_tuple": (".loader", "load_model_tuple"),
    "load_model_with_lora": (".loader", "load_model_with_lora"),
    "load_model_with_lora_async": (".loader", "load_model_with_lora_async"),
    "save_adapter": (".loader", "save_adapter"),
    "create_model": (".loader", "create_model"),
    "create_from_preset": (".loader", "create_from_preset"),
    # Adapters
    "LoRAConfig": (".adapters", "LoRAConfig"),
    "LoRALinear": (".adapters", "LoRALinear"),
    "apply_lora": (".adapters", "apply_lora"),
    "merge_lora_weights": (".adapters", "merge_lora_weights"),
    "count_lora_parameters": (".adapters", "count_lora_parameters"),
    # Introspection
    "ParameterStats": (".introspection", "ParameterStats"),
    "FLOPsEstimate": (".introspection", "FLOPsEstimate"),
    "MemoryEstimate": (".introspection", "MemoryEstimate"),
    "ModelCapabilities": (".introspection", "ModelCapabilities"),
    "ModelInfo": (".introspection", "ModelInfo"),
    "count_parameters": (".introspection", "count_parameters"),
    "estimate_flops": (".introspection", "estimate_flops"),
    "estimate_memory": (".introspection", "estimate_memory"),
    "detect_model_capabilities": (".introspection", "detect_model_capabilities"),
    "get_model_info": (".introspection", "get_model_info"),
    "introspect": (".introspection", "introspect"),
    "print_introspection": (".introspection", "print_introspection"),
    # Losses
    "compute_lm_loss": (".losses", "compute_lm_loss"),
}

__all__ = sorted(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(globals().keys()) + __all__
