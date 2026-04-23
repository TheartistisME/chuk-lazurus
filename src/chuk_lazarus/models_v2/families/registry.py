"""
Model family registry for unified HuggingFace model loading.

Each model family provides:
- Config class (Pydantic BaseModel)
- Model class (CausalLM, etc.)
- Weight converter (HF -> our format)
- Optional: Introspection hooks, chat template, special tokens

The registry auto-detects the model family from HuggingFace config.json
and provides a unified interface for loading any supported model.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import logging
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .constants import HFArchitecture, HFModelType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _LazyModuleAttr:
    """Resolve a family class only when the caller first uses it."""

    _families_root = Path(__file__).resolve().parent
    _package_name = __package__ or "chuk_lazarus.models_v2.families"

    def __init__(self, module_name: str, attr_name: str) -> None:
        self._module_name = module_name
        self._attr_name = attr_name
        self._value: Any | None = None

    @classmethod
    def _ensure_namespace_package(cls, package_name: str, package_path: Path) -> bool:
        if package_name in sys.modules:
            return False

        module = types.ModuleType(package_name)
        module.__package__ = package_name
        module.__path__ = [str(package_path)]

        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
        module.__spec__ = spec

        sys.modules[package_name] = module
        return True

    def _import_without_package_init(self):
        full_name = importlib.util.resolve_name(self._module_name, self._package_name)
        if full_name in sys.modules:
            return sys.modules[full_name]

        if not full_name.startswith(f"{self._package_name}."):
            return importlib.import_module(full_name)

        relative_parts = full_name.removeprefix(f"{self._package_name}.").split(".")
        if len(relative_parts) < 2:
            return importlib.import_module(full_name)

        family_dir = self._families_root / relative_parts[0]
        module_path = family_dir.joinpath(*relative_parts[1:]).with_suffix(".py")
        if not module_path.exists():
            return importlib.import_module(full_name)

        family_package = f"{self._package_name}.{relative_parts[0]}"
        created_package = self._ensure_namespace_package(family_package, family_dir)

        spec = importlib.util.spec_from_file_location(full_name, module_path)
        if spec is None or spec.loader is None:
            if created_package:
                sys.modules.pop(family_package, None)
            return importlib.import_module(full_name)

        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(full_name, None)
            raise
        finally:
            if created_package:
                sys.modules.pop(family_package, None)

        return module

    def _resolve(self) -> Any:
        if self._value is None:
            module = self._import_without_package_init()
            self._value = getattr(module, self._attr_name)
        return self._value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __repr__(self) -> str:
        return f"<lazy {self._module_name}:{self._attr_name}>"


class ModelFamilyType(str, Enum):
    """Supported model family types."""

    LLAMA = "llama"
    LLAMA4 = "llama4"
    GEMMA = "gemma"
    GRANITE = "granite"
    GRANITE_HYBRID = "granitemoehybrid"
    JAMBA = "jamba"
    MAMBA = "mamba"
    STARCODER2 = "starcoder2"
    QWEN3 = "qwen3"
    GPT2 = "gpt2"
    GPT_NEO = "gpt_neo"
    GPT_NEOX = "gpt_neox"
    GPT_OSS = "gpt_oss"
    GPT_OSS_LITE = "gpt_oss_lite"
    OLMOE = "olmoe"


# Model type patterns for auto-detection from config.json
MODEL_TYPE_PATTERNS = {
    # Exact matches first
    "llama4": ModelFamilyType.LLAMA4,
    "granitemoehybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite_hybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite-hybrid": ModelFamilyType.GRANITE_HYBRID,
    "granite": ModelFamilyType.GRANITE,
    "jamba": ModelFamilyType.JAMBA,
    "mamba": ModelFamilyType.MAMBA,
    "starcoder2": ModelFamilyType.STARCODER2,
    "qwen3": ModelFamilyType.QWEN3,
    "qwen2": ModelFamilyType.QWEN3,  # Qwen2 uses same arch
    "qwen3_5_moe": ModelFamilyType.QWEN3,  # Qwen3.5/3.6 MoE uses Qwen path
    "gpt2": ModelFamilyType.GPT2,
    "gpt_neo": ModelFamilyType.GPT_NEO,
    "gpt_neox": ModelFamilyType.GPT_NEOX,
    "gpt-neo": ModelFamilyType.GPT_NEO,
    "gpt-neox": ModelFamilyType.GPT_NEOX,
    "gpt_oss": ModelFamilyType.GPT_OSS,
    "gpt_oss_lite": ModelFamilyType.GPT_OSS_LITE,
    "gpt_oss_lite_minimal": ModelFamilyType.GPT_OSS_LITE,
    "olmoe": ModelFamilyType.OLMOE,
    # Gemma variants
    "gemma": ModelFamilyType.GEMMA,
    "gemma2": ModelFamilyType.GEMMA,
    "gemma3": ModelFamilyType.GEMMA,  # May have nested text_config
    "gemma3_text": ModelFamilyType.GEMMA,
    "gemma-3": ModelFamilyType.GEMMA,
    # Gemma 4 — VLM-style wrapper (gemma4) with nested text backbone
    # (gemma4_text). Torch/CUDA backend uses HF AutoModelForCausalLM, so
    # this mapping is sufficient; the MLX family code still targets Gemma 3.
    "gemma4": ModelFamilyType.GEMMA,
    "gemma4_text": ModelFamilyType.GEMMA,
    "gemma-4": ModelFamilyType.GEMMA,
    # Llama-compatible (last, as catch-all for llama-like models)
    "llama": ModelFamilyType.LLAMA,
    "mistral": ModelFamilyType.LLAMA,
    "mixtral": ModelFamilyType.LLAMA,
    "codellama": ModelFamilyType.LLAMA,
}


# Architecture names for auto-detection
ARCHITECTURE_PATTERNS = {
    "Llama4ForCausalLM": ModelFamilyType.LLAMA4,
    "Llama4ForConditionalGeneration": ModelFamilyType.LLAMA4,
    "LlamaForCausalLM": ModelFamilyType.LLAMA,
    "MistralForCausalLM": ModelFamilyType.LLAMA,
    "MixtralForCausalLM": ModelFamilyType.LLAMA,
    "GemmaForCausalLM": ModelFamilyType.GEMMA,
    "Gemma2ForCausalLM": ModelFamilyType.GEMMA,
    "Gemma3ForCausalLM": ModelFamilyType.GEMMA,
    "Gemma3ForConditionalGeneration": ModelFamilyType.GEMMA,
    "PaliGemmaForConditionalGeneration": ModelFamilyType.GEMMA,
    "Gemma4ForCausalLM": ModelFamilyType.GEMMA,
    "Gemma4ForConditionalGeneration": ModelFamilyType.GEMMA,
    "GraniteForCausalLM": ModelFamilyType.GRANITE,
    "GraniteMoeHybridForCausalLM": ModelFamilyType.GRANITE_HYBRID,
    "JambaForCausalLM": ModelFamilyType.JAMBA,
    "MambaForCausalLM": ModelFamilyType.MAMBA,
    "Starcoder2ForCausalLM": ModelFamilyType.STARCODER2,
    "Qwen2ForCausalLM": ModelFamilyType.QWEN3,
    "Qwen3ForCausalLM": ModelFamilyType.QWEN3,
    "Qwen3_5MoeForConditionalGeneration": ModelFamilyType.QWEN3,
    "GPT2LMHeadModel": ModelFamilyType.GPT2,
    "GPTNeoForCausalLM": ModelFamilyType.GPT_NEO,
    "GPTNeoXForCausalLM": ModelFamilyType.GPT_NEOX,
    "GptOssForCausalLM": ModelFamilyType.GPT_OSS,
    "GPTOSSLiteForCausalLM": ModelFamilyType.GPT_OSS_LITE,
    "GptOssLiteForCausalLM": ModelFamilyType.GPT_OSS_LITE,
    "OlmoeForCausalLM": ModelFamilyType.OLMOE,
}


@runtime_checkable
class WeightConverter(Protocol):
    """Protocol for weight name converters."""

    def convert(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Convert weight names from HF format to our format."""
        ...

    def reverse_convert(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Convert weight names from our format back to HF format."""
        ...


@dataclass
class IntrospectionHooks:
    """Hooks for model introspection during inference.

    These hooks enable:
    - Logit lens analysis
    - Activation patching
    - Attention pattern extraction
    - Layer-wise feature analysis
    """

    # Called after each layer
    layer_output_hook: Any | None = None

    # Called after attention computation
    attention_hook: Any | None = None

    # Called after FFN computation
    ffn_hook: Any | None = None

    # Called after final norm (before LM head)
    pre_head_hook: Any | None = None

    # Called with the final logits
    logits_hook: Any | None = None

    # Track which layers to hook (None = all)
    layer_indices: list[int] | None = None


@dataclass
class FamilyInfo:
    """Information about a model family."""

    family_type: ModelFamilyType
    config_class: type[BaseModel]
    model_class: type
    weight_converter: WeightConverter | None = None

    # Model metadata
    model_types: list[str] = field(default_factory=list)
    architectures: list[str] = field(default_factory=list)

    # Capabilities
    supports_chat: bool = True
    supports_kv_cache: bool = True
    supports_introspection: bool = True

    # Default introspection hooks creator
    create_hooks: Any | None = None


class ModelFamilyRegistry:
    """Registry for model families.

    Provides auto-detection and unified loading for any supported model.
    """

    _instance: ModelFamilyRegistry | None = None
    _families: dict[ModelFamilyType, FamilyInfo]

    def __init__(self) -> None:
        self._families = {}
        self._initialized = False

    @classmethod
    def get_instance(cls) -> ModelFamilyRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> None:
        """Lazy initialization of family registry."""
        if self._initialized:
            return

        # Import and register all families
        self._register_builtin_families()
        self._initialized = True

    def _register_builtin_families(self) -> None:
        """Register all built-in model families."""
        def lazy(module_name: str, attr_name: str) -> _LazyModuleAttr:
            return _LazyModuleAttr(module_name, attr_name)

        # Llama family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.LLAMA,
                config_class=lazy(".llama.config", "LlamaConfig"),
                model_class=lazy(".llama.model", "LlamaForCausalLM"),
                model_types=[
                    HFModelType.LLAMA.value,
                    HFModelType.MISTRAL.value,
                    HFModelType.MIXTRAL.value,
                    HFModelType.CODELLAMA.value,
                ],
                architectures=["LlamaForCausalLM", "MistralForCausalLM"],
            )
        )

        # Llama4 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.LLAMA4,
                config_class=lazy(".llama4.config", "Llama4Config"),
                model_class=lazy(".llama4.model", "Llama4ForCausalLM"),
                model_types=[HFModelType.LLAMA4.value],
                architectures=["Llama4ForCausalLM"],
            )
        )

        # Gemma family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GEMMA,
                config_class=lazy(".gemma.config", "GemmaConfig"),
                model_class=lazy(".gemma.model", "GemmaForCausalLM"),
                model_types=[
                    HFModelType.GEMMA.value,
                    HFModelType.GEMMA2.value,
                    HFModelType.GEMMA3.value,
                    HFModelType.GEMMA3_TEXT.value,
                ],
                architectures=[
                    "GemmaForCausalLM",
                    "Gemma2ForCausalLM",
                    "Gemma3ForCausalLM",
                    "Gemma3ForConditionalGeneration",  # Text models with VLM wrapper
                ],
            )
        )

        # Granite family (dense)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GRANITE,
                config_class=lazy(".granite.config", "GraniteConfig"),
                model_class=lazy(".granite.model", "GraniteForCausalLM"),
                model_types=[HFModelType.GRANITE.value],
                architectures=["GraniteForCausalLM"],
            )
        )

        # Granite hybrid family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GRANITE_HYBRID,
                config_class=lazy(".granite.config", "GraniteHybridConfig"),
                model_class=lazy(".granite.hybrid", "GraniteHybridForCausalLM"),
                model_types=[HFModelType.GRANITE_MOE_HYBRID.value],
                architectures=["GraniteMoeHybridForCausalLM"],
            )
        )

        # Jamba family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.JAMBA,
                config_class=lazy(".jamba.config", "JambaConfig"),
                model_class=lazy(".jamba.model", "JambaForCausalLM"),
                model_types=[HFModelType.JAMBA.value],
                architectures=["JambaForCausalLM"],
            )
        )

        # Mamba family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.MAMBA,
                config_class=lazy(".mamba.config", "MambaConfig"),
                model_class=lazy(".mamba.model", "MambaForCausalLM"),
                model_types=[HFModelType.MAMBA.value],
                architectures=["MambaForCausalLM"],
            )
        )

        # StarCoder2 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.STARCODER2,
                config_class=lazy(".starcoder2.config", "StarCoder2Config"),
                model_class=lazy(".starcoder2.model", "StarCoder2ForCausalLM"),
                model_types=[HFModelType.STARCODER2.value],
                architectures=["Starcoder2ForCausalLM"],
            )
        )

        # Qwen3 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.QWEN3,
                config_class=lazy(".qwen3.config", "Qwen3Config"),
                model_class=lazy(".qwen3.model", "Qwen3ForCausalLM"),
                model_types=[
                    HFModelType.QWEN2.value,
                    HFModelType.QWEN3.value,
                    HFModelType.QWEN3_5_MOE.value,
                ],
                architectures=[
                    HFArchitecture.QWEN2_FOR_CAUSAL_LM,
                    HFArchitecture.QWEN3_FOR_CAUSAL_LM,
                    HFArchitecture.QWEN3_5_MOE_FOR_CONDITIONAL_GENERATION,
                ],
            )
        )

        # GPT-2 family
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GPT2,
                config_class=lazy(".gpt2.config", "GPT2Config"),
                model_class=lazy(".gpt2.model", "GPT2ForCausalLM"),
                model_types=[HFModelType.GPT2.value],
                architectures=["GPT2LMHeadModel"],
            )
        )

        # GPT-OSS family (OpenAI open-source MoE)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GPT_OSS,
                config_class=lazy(".gpt_oss.config", "GptOssConfig"),
                model_class=lazy(".gpt_oss.model", "GptOssForCausalLM"),
                model_types=[HFModelType.GPT_OSS.value],
                architectures=["GptOssForCausalLM"],
            )
        )

        # GPT-OSS-Lite family (reduced expert models)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.GPT_OSS_LITE,
                config_class=lazy(".gpt_oss_lite.config", "GptOssLiteConfig"),
                model_class=lazy(".gpt_oss_lite.model", "GptOssLiteForCausalLM"),
                model_types=["gpt_oss_lite", "gpt_oss_lite_minimal"],
                architectures=["GptOssLiteForCausalLM", "GPTOSSLiteForCausalLM"],
            )
        )

        # OLMoE family (Allen AI's open MoE)
        self.register(
            FamilyInfo(
                family_type=ModelFamilyType.OLMOE,
                config_class=lazy(".olmoe.config", "OLMoEConfig"),
                model_class=lazy(".olmoe.model", "OLMoEForCausalLM"),
                model_types=["olmoe"],
                architectures=["OlmoeForCausalLM"],
            )
        )

    def register(self, family_info: FamilyInfo) -> None:
        """Register a model family."""
        self._families[family_info.family_type] = family_info
        logger.debug(f"Registered model family: {family_info.family_type.value}")

    def get_family(self, family_type: ModelFamilyType) -> FamilyInfo | None:
        """Get family info by type."""
        self._ensure_initialized()
        return self._families.get(family_type)

    def detect_family(self, config: dict[str, Any]) -> ModelFamilyType | None:
        """Auto-detect model family from HuggingFace config.

        Args:
            config: Dict loaded from config.json

        Returns:
            Detected ModelFamilyType or None
        """
        self._ensure_initialized()

        # Try model_type first
        model_type = config.get("model_type", "").lower()
        if model_type in MODEL_TYPE_PATTERNS:
            return MODEL_TYPE_PATTERNS[model_type]

        # Try architectures
        architectures = config.get("architectures", [])
        for arch in architectures:
            if arch in ARCHITECTURE_PATTERNS:
                return ARCHITECTURE_PATTERNS[arch]

        # Try to infer from config structure
        # Granite hybrid has specific keys
        if "mamba_d_state" in config and "attn_layer_period" in config:
            return ModelFamilyType.GRANITE_HYBRID

        # Mamba has ssm-specific keys
        if "d_state" in config and "d_conv" in config:
            return ModelFamilyType.MAMBA

        # Gemma has sliding_window_pattern
        if "sliding_window_pattern" in config:
            return ModelFamilyType.GEMMA

        # Jamba has hybrid pattern keys
        if "attn_layer_period" in config and "expert_layer_period" in config:
            return ModelFamilyType.JAMBA

        return None

    def detect_family_from_path(self, model_path: Path) -> ModelFamilyType | None:
        """Detect family from model path."""
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        return self.detect_family(config)

    def list_families(self) -> list[ModelFamilyType]:
        """List all registered families."""
        self._ensure_initialized()
        return list(self._families.keys())


# Global registry instance
_registry = ModelFamilyRegistry.get_instance()


def get_family_registry() -> ModelFamilyRegistry:
    """Get the global family registry."""
    return _registry


def detect_model_family(config: dict[str, Any]) -> ModelFamilyType | None:
    """Detect model family from config dict."""
    return _registry.detect_family(config)


def get_family_info(family_type: ModelFamilyType) -> FamilyInfo | None:
    """Get family info by type."""
    return _registry.get_family(family_type)


def list_model_families() -> list[ModelFamilyType]:
    """List all registered model families."""
    return _registry.list_families()
