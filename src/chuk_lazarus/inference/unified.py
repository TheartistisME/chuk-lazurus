"""
Unified inference pipeline for any HuggingFace model.

Provides a single entry point that:
1. Auto-detects model family from config.json
2. Loads config and weights using family-specific converters
3. Provides unified chat/generate interface
4. Supports introspection hooks for all families

Usage:
    from chuk_lazarus.inference import UnifiedPipeline

    # Auto-detect and load any supported model
    pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Or async
    pipeline = await UnifiedPipeline.from_pretrained_async("model-id")

    # Chat
    result = pipeline.chat("What is the capital of France?")
    print(result.text)

    # Generate with introspection
    result = pipeline.generate(
        "The capital of France is",
        introspect=True,
    )
    print(result.hidden_states)  # Layer activations
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from chuk_lazarus.models_v2.families import (
    FamilyInfo,
    ModelFamilyType,
    detect_model_family,
    get_family_info,
)

from .backends import InferenceRuntime, MLXInferenceRuntime, TorchInferenceRuntime
from .backends.types import LazarusBackend
from .chat import ChatHistory, format_chat_prompt, format_history
from .generation import GenerationConfig, GenerationResult
from .loader import DType, HFLoader


class EngineMode(str, Enum):
    """Inference engine mode for UnifiedPipeline."""

    STANDARD = "standard"
    """Default KV-cached generation path."""

    KV_DIRECT = "kv_direct"
    """Stateful KVDirectGenerator — works with any supported model family."""

    BOUNDED_KV_DIRECT = "bounded_kv_direct"
    """Torch KV-direct with a bounded hot KV window."""

    RESIDUAL_BOUNDED_KV_DIRECT = "residual_bounded_kv_direct"
    """Torch KV-direct bounded by the hot budget, with a residual-backed
    rebuild path.  Evicted KV is not dropped outright: a
    Markov-sufficient residual snapshot is kept in device memory and used to
    re-anchor the hot window via a layer-0 residual-seeded prefill.  Ports
    the research methodology from
    ``chuk_lazarus.inference.context.research.bounded_engine`` onto the
    Qwen torch serve path."""


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class UnifiedPipelineConfig(BaseModel):
    """Configuration for the unified pipeline."""

    dtype: DType = Field(DType.BFLOAT16, description="Weight dtype")
    cache_dir: Path | None = Field(None, description="Model cache directory")
    backend: LazarusBackend | None = Field(
        None,
        description=(
            "Optional runtime backend override. Mirrors CHUK_BACKEND semantics "
            "for explicit selection."
        ),
    )
    backend_name: str | None = Field(
        None,
        description="Shared backend registry selector name (for example: torch or mlx)",
    )
    device: str | None = Field(None, description="Explicit runtime device override")
    cuda_check_sm: bool = Field(
        True,
        description="Validate CUDA SM compatibility when selecting the torch backend",
    )
    default_system_message: str | None = Field(
        "You are a helpful assistant.", description="Default system prompt"
    )
    default_max_tokens: int = Field(256, ge=1, description="Default max tokens")
    default_temperature: float = Field(0.7, ge=0.0, description="Default temperature")

    # Introspection
    enable_introspection: bool = Field(True, description="Enable layer hooks")
    introspection_layers: list[int] | None = Field(None, description="Layers to track (None = all)")

    # Engine
    engine: EngineMode = Field(EngineMode.STANDARD, description="Inference engine mode")
    hot_budget_mib: int | None = Field(
        None,
        ge=1,
        description="Optional hot-tier KV budget in MiB for bounded_kv_direct",
    )

    @model_validator(mode="after")
    def _bridge_backend_name(self) -> UnifiedPipelineConfig:
        if self.backend is None and self.backend_name is not None:
            normalized = self.backend_name.strip().lower()
            if normalized == "torch":
                self.backend = LazarusBackend.CUDA
                self.backend_name = "torch"
            elif normalized == "mlx":
                self.backend = LazarusBackend.MLX
                self.backend_name = "mlx"
        if self.backend_name is None and self.backend is not None:
            self.backend_name = "torch" if self.backend == LazarusBackend.CUDA else "mlx"
        return self


class UnifiedPipelineState(BaseModel):
    """Internal state of the unified pipeline."""

    model_id: str
    model_path: Path
    family_type: ModelFamilyType
    tensor_count: int
    is_loaded: bool = False


class IntrospectionResult(BaseModel):
    """Results from introspection during generation."""

    # Layer-wise hidden states (if captured)
    hidden_states: list[Any] | None = None

    # Attention patterns (if captured)
    attention_patterns: list[Any] | None = None

    # Pre-head activations
    pre_head_activations: Any | None = None


import re as _re


_EXPERT_WEIGHT_KEY_RE = _re.compile(
    r"(?:^|\.)mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
)


def _raise_if_experts_random_init(loading_info: dict[str, Any], *, log=None) -> None:
    """Guard against silent MoE expert random-init on compressed-tensors loads.

    ``AutoModelForCausalLM.from_pretrained(..., output_loading_info=True)``
    returns ``{"missing_keys": [...], "unexpected_keys": [...], ...}``.
    If the model's expert projections (``…mlp.experts.N.{gate,up,down}_proj.weight``)
    land in ``missing_keys``, the compressed-tensors quantizer did NOT
    replace the patched Linears with ``CompressedLinear`` — every expert
    is random-initialised and the server will either produce token salad
    or hang inside the Python per-expert dispatch loop.  Better to fail
    loud than serve garbage.
    """
    if not isinstance(loading_info, dict):
        return

    missing = list(loading_info.get("missing_keys") or [])
    missing_experts = [key for key in missing if _EXPERT_WEIGHT_KEY_RE.search(key)]
    if missing_experts:
        sample = ", ".join(missing_experts[:3])
        raise RuntimeError(
            "Compressed-tensors MoE load left expert projections random-initialised "
            f"({len(missing_experts)} of {len(missing)} missing keys are expert weights; "
            f"sample: {sample}). Compressed-tensors' CompressedLinear.from_linear did not "
            "fire on the patched per-expert Linears — check that "
            "AutoModelForCausalLM.from_pretrained is not using device_map= or "
            "low_cpu_mem_usage=True on the compressed-tensors branch, because meta-init "
            "there defeats the quantizer's weight replacement."
        )

    # Non-expert missing keys are usually benign (tied embeddings etc.).
    if log is not None:
        unexpected = list(loading_info.get("unexpected_keys") or [])
        log(
            f"   Load report: missing={len(missing)} unexpected={len(unexpected)} "
            f"(expert-weight MISSING keys: 0 — compressed-tensors replacement OK)"
        )


class UnifiedPipeline:
    """Unified inference pipeline for any supported model family.

    This is the primary interface for model inference. It:
    1. Auto-detects model family from HuggingFace config
    2. Uses family-specific loading and weight conversion
    3. Provides unified chat/generate interface
    4. Supports introspection hooks for interpretability research

    Example:
        # One-liner for any supported model
        pipeline = UnifiedPipeline.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Chat
        result = pipeline.chat("Hello!")
        print(result.text)

        # With introspection
        result, intro = pipeline.generate_with_introspection(
            "The capital of France is",
            capture_hidden_states=True,
        )
    """

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizer,
        model_config: BaseModel,
        family_info: FamilyInfo,
        pipeline_config: UnifiedPipelineConfig | None = None,
        state: UnifiedPipelineState | None = None,
        runtime: InferenceRuntime | None = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_config = model_config
        self._family_info = family_info
        self._pipeline_config = pipeline_config or UnifiedPipelineConfig()
        self._state = state
        self._runtime = runtime or self._select_runtime(
            model,
            tokenizer,
            self._pipeline_config,
            family_info.family_type,
        )

        # Introspection hooks (mutable state for capturing)
        self._introspection_data: IntrospectionResult | None = None

    @staticmethod
    def _select_runtime(
        model: Any,
        tokenizer: PreTrainedTokenizer,
        pipeline_config: UnifiedPipelineConfig,
        family_type: ModelFamilyType | None = None,
    ) -> InferenceRuntime:
        """Choose the runtime that matches the configured backend."""
        if pipeline_config.backend == LazarusBackend.CUDA:
            return TorchInferenceRuntime(
                model,
                tokenizer,
                device=pipeline_config.device or "cuda",
                engine=pipeline_config.engine.value,
                family_type=family_type,
                hot_budget_mib=pipeline_config.hot_budget_mib,
            )
        return MLXInferenceRuntime(model, tokenizer)

    @property
    def model(self) -> Any:
        """Access the underlying model."""
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Access the tokenizer."""
        return self._tokenizer

    @property
    def config(self) -> BaseModel:
        """Access the model config."""
        return self._model_config

    @property
    def family(self) -> FamilyInfo:
        """Access the model family info."""
        return self._family_info

    @property
    def family_type(self) -> ModelFamilyType:
        """Get the model family type."""
        return self._family_info.family_type

    @property
    def runtime(self) -> InferenceRuntime:
        """Access the backend-specific runtime."""
        return self._runtime

    @property
    def engine_mode(self) -> EngineMode:
        """Active engine mode for this pipeline."""
        runtime_engine = getattr(self._runtime, "engine_mode", None)
        if runtime_engine is not None:
            return EngineMode(runtime_engine)
        return self._pipeline_config.engine

    @property
    def last_generation_path(self) -> str | None:
        """Observable label for the most recent runtime generation path."""
        return getattr(self._runtime, "last_generation_path", None)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a model from HuggingFace, auto-detecting the family.

        Args:
            model_id: HuggingFace model ID or local path
            pipeline_config: Pipeline configuration
            verbose: Print loading progress

        Returns:
            Configured UnifiedPipeline instance
        """
        pipeline_config = pipeline_config or UnifiedPipelineConfig()

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        log(f"Loading {model_id}...")
        log("=" * 60)

        # Download
        log("\n1. Downloading model...")
        result = HFLoader.download(model_id, cache_dir=pipeline_config.cache_dir, verbose=verbose)
        log(f"   Path: {result.model_path}")

        # Load raw config for family detection
        log("\n2. Detecting model family...")
        config_path = result.model_path / "config.json"
        with open(config_path) as f:
            hf_config = json.load(f)

        family_type = detect_model_family(hf_config)
        if family_type is None:
            model_type = hf_config.get("model_type", "unknown")
            archs = hf_config.get("architectures", [])
            raise ValueError(
                f"Unable to detect model family. model_type={model_type}, "
                f"architectures={archs}. Model may not be supported yet."
            )

        if pipeline_config.backend == LazarusBackend.CUDA:
            return cls._from_pretrained_torch(
                model_id=model_id,
                model_path=result.model_path,
                hf_config=hf_config,
                family_type=family_type,
                pipeline_config=pipeline_config,
                verbose=verbose,
            )

        family_info = get_family_info(family_type)
        if family_info is None:
            raise ValueError(f"No family info registered for {family_type}")

        log(f"   Detected: {family_type.value}")

        # Load config using family-specific class
        log("\n3. Loading configuration...")
        config_class = family_info.config_class

        # Try from_hf_config first, then fall back to direct construction
        if hasattr(config_class, "from_hf_config"):
            model_config = config_class.from_hf_config(hf_config)
        else:
            model_config = config_class(**hf_config)

        log(f"   Hidden: {model_config.hidden_size}, Layers: {model_config.num_hidden_layers}")

        # Load tokenizer
        log("\n4. Loading tokenizer...")
        tokenizer = HFLoader.load_tokenizer(result.model_path)
        log(f"   Vocab size: {len(tokenizer)}")

        # Create model
        log("\n5. Creating model...")
        model_class = family_info.model_class
        model = model_class(model_config)

        # Load and apply weights using unified loader
        log("\n6. Loading and applying weights...")
        HFLoader.apply_weights_to_model(
            model,
            result.model_path,
            model_config,
            dtype=pipeline_config.dtype,
            verbose=verbose,
        )
        log("   Weights applied!")

        log("\n" + "=" * 60)
        log(f"Model loaded successfully! ({family_type.value})")

        # Count parameters (flatten nested structure)
        def count_params(params):
            total = 0
            for v in params.values():
                if isinstance(v, dict):
                    total += count_params(v)
                elif hasattr(v, "size"):
                    total += v.size
            return total

        param_count = count_params(model.parameters())

        state = UnifiedPipelineState(
            model_id=model_id,
            model_path=result.model_path,
            family_type=family_type,
            tensor_count=param_count,
            is_loaded=True,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            family_info=family_info,
            pipeline_config=pipeline_config,
            state=state,
            runtime=MLXInferenceRuntime(model, tokenizer),
        )

    @classmethod
    def _from_pretrained_torch(
        cls,
        model_id: str,
        model_path: Path,
        hf_config: dict[str, Any],
        family_type: ModelFamilyType,
        pipeline_config: UnifiedPipelineConfig,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a Hugging Face torch runtime for CUDA/CPU/MPS inference."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        from .backends._torch_qwen_support import (
            compressed_tensors_quant_method,
            install_blackwell_causal_conv1d_workaround,
            install_qwen3_5_moe_per_expert_patch,
        )

        family_info = get_family_info(family_type)
        if family_info is None:
            raise ValueError(f"No family info registered for {family_type}")

        resolved_device = pipeline_config.device or "cuda"
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        log(f"Loading {model_id}...")
        log("=" * 60)
        log("\n2. Detecting model family...")
        log(f"   Detected: {family_type.value}")
        log("\n3. Loading configuration...")
        model_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        # VLM wrappers (Gemma 4, Gemma 3 VLM, PaliGemma, ...) keep the
        # text-backbone dimensions under `text_config` rather than on the
        # top-level config. Unwrap for logging so we don't print None.
        hidden_size = getattr(model_config, "hidden_size", None)
        layer_count = getattr(model_config, "num_hidden_layers", None)
        text_config = getattr(model_config, "text_config", None)
        if (hidden_size is None or layer_count is None) and text_config is not None:
            hidden_size = hidden_size if hidden_size is not None else getattr(
                text_config, "hidden_size", "unknown"
            )
            layer_count = layer_count if layer_count is not None else getattr(
                text_config, "num_hidden_layers", "unknown"
            )
        if hidden_size is None:
            hidden_size = "unknown"
        if layer_count is None:
            layer_count = "unknown"
        log(f"   Hidden: {hidden_size}, Layers: {layer_count}")
        log("\n4. Loading tokenizer...")
        tokenizer = HFLoader.load_tokenizer(model_path)
        log(f"   Vocab size: {len(tokenizer)}")
        log("\n5. Loading torch model...")

        if resolved_device.startswith("cuda"):
            if pipeline_config.dtype == DType.FLOAT32:
                torch_dtype = torch.float32
            elif pipeline_config.dtype == DType.FLOAT16:
                torch_dtype = torch.float16
            else:
                torch_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
        elif resolved_device == "mps":
            torch_dtype = torch.float32 if pipeline_config.dtype == DType.FLOAT32 else torch.float16
        else:
            torch_dtype = torch.float32

        quant_method = compressed_tensors_quant_method(model_config)
        model_type = str(getattr(model_config, "model_type", "") or "").lower()
        architectures = [str(arch).lower() for arch in getattr(model_config, "architectures", [])]
        is_qwen35_moe = model_type == "qwen3_5_moe" or any(
            "qwen3_5moe" in arch for arch in architectures
        )

        if quant_method == "compressed-tensors" and is_qwen35_moe:
            log("   Applying Qwen3.5-MoE compressed-tensors load patches...")
            patch_status = install_qwen3_5_moe_per_expert_patch()
            workaround_status = install_blackwell_causal_conv1d_workaround()
            log(f"   Patch      : {patch_status}")
            log(f"   Workaround : {workaround_status}")
            # NO ``device_map`` and NO ``low_cpu_mem_usage`` here.  Those
            # two flags together trigger a meta-init path that runs
            # BEFORE the compressed-tensors quantizer walks modules with
            # ``apply_quantization_config`` — the patched per-expert
            # ``nn.Linear`` leaves are on ``meta`` when that walk happens
            # and ``CompressedLinear.from_linear``'s ``delattr(module,
            # "weight")`` step silently no-ops.  The result is every
            # expert projection ending up random-initialised and the
            # forward pass routing through all 256 experts in a Python
            # loop — which is what previously hung the server.  Eager
            # init on host + explicit ``.to(cuda)`` lets the quantizer
            # see real Linears and actually swap in ``CompressedLinear``.
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                dtype=torch.float16,
                output_loading_info=True,
            )
            _raise_if_experts_random_init(loading_info, log=log)
            model = model.to(resolved_device, non_blocking=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )

            to_kwargs = {"non_blocking": True} if resolved_device.startswith("cuda") else {}
            model = model.to(resolved_device, **to_kwargs)
        model.eval()

        runtime = TorchInferenceRuntime(
            model,
            tokenizer,
            device=resolved_device,
            engine=pipeline_config.engine.value,
            family_type=family_type,
            hot_budget_mib=pipeline_config.hot_budget_mib,
        )
        param_count = sum(int(param.numel()) for param in model.parameters())

        log("\n" + "=" * 60)
        log(f"Model loaded successfully! ({family_type.value})")

        state = UnifiedPipelineState(
            model_id=model_id,
            model_path=model_path,
            family_type=family_type,
            tensor_count=param_count,
            is_loaded=True,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            family_info=family_info,
            pipeline_config=pipeline_config,
            state=state,
            runtime=runtime,
        )

    @classmethod
    async def from_pretrained_async(
        cls,
        model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a model from HuggingFace asynchronously.

        Runs in a thread pool to avoid blocking.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls.from_pretrained(model_id, pipeline_config, verbose),
        )

    def chat(
        self,
        user_message: str,
        system_message: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a response to a chat message.

        Args:
            user_message: The user's message
            system_message: Optional system prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerationResult with text and stats
        """
        system = system_message or self._pipeline_config.default_system_message
        prompt = format_chat_prompt(self._tokenizer, user_message, system)

        config = GenerationConfig(
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else self._pipeline_config.default_max_tokens,
            temperature=temperature
            if temperature is not None
            else self._pipeline_config.default_temperature,
        )

        return self._runtime.generate(prompt, config)

    def chat_with_history(
        self,
        history: ChatHistory,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        """Generate a response using chat history.

        Args:
            history: ChatHistory with conversation
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            GenerationResult with text and stats
        """
        prompt = format_history(self._tokenizer, history)

        config = GenerationConfig(
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else self._pipeline_config.default_max_tokens,
            temperature=temperature
            if temperature is not None
            else self._pipeline_config.default_temperature,
        )

        return self._runtime.generate(prompt, config)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        config: GenerationConfig | None = None,
        *,
        conversation_id: str | None = None,
    ) -> GenerationResult:
        """Generate text from a raw prompt.

        Args:
            prompt: Input prompt (no formatting applied)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            config: Full generation config
            conversation_id: Optional per-session key used by the torch
                ``residual_bounded_kv_direct`` runtime to key its WARM
                residual cache. Ignored by other engine modes.

        Returns:
            GenerationResult with text and stats
        """
        if config is None:
            config = GenerationConfig(
                max_new_tokens=max_new_tokens
                if max_new_tokens is not None
                else self._pipeline_config.default_max_tokens,
                temperature=temperature
                if temperature is not None
                else self._pipeline_config.default_temperature,
            )

        # Runtimes that don't accept ``conversation_id`` (MLX, older torch
        # shims, mocks in tests) just get a plain positional call so the
        # signature contract stays minimal.
        generate_fn = self._runtime.generate
        if conversation_id is not None:
            try:
                return generate_fn(prompt, config, conversation_id=conversation_id)
            except TypeError:
                return generate_fn(prompt, config)
        return generate_fn(prompt, config)

    def make_engine(self):
        """
        Construct a KVDirectGenerator for this pipeline's model.

        Returns a ready-to-use KVDirectGenerator backed by the appropriate
        adapter for the loaded model family.

        Raises ValueError if the model family is not yet supported.

        Example::

            pipeline = UnifiedPipeline.from_pretrained("mlx-community/gemma-3-1b-it-bf16")
            engine = pipeline.make_engine()
            logits, kv_store = engine.prefill(input_ids)
        """
        if isinstance(self._runtime, TorchInferenceRuntime):
            raise RuntimeError(
                "UnifiedPipeline.make_engine() is only available for MLX-backed pipelines. "
                "Torch-backed pipelines should use generate()/chat(); set "
                "pipeline_config.engine='kv_direct' to select the torch KV-direct path."
            )

        from .context import make_kv_generator

        return make_kv_generator(self._model)

    def list_supported_families(self) -> list[str]:
        """List all supported model families."""
        from chuk_lazarus.models_v2.families import list_model_families

        return [f.value for f in list_model_families()]

    @classmethod
    def from_compressed(
        cls,
        compressed_path: str | Path,
        original_model_id: str,
        pipeline_config: UnifiedPipelineConfig | None = None,
        verbose: bool = True,
    ) -> UnifiedPipeline:
        """Load a compressed MoE model from overlay format.

        This loads a model that was compressed using the moe-overlay-compress command.
        The compressed model uses low-rank approximations of expert weights.

        Args:
            compressed_path: Path to compressed overlay directory
            original_model_id: Original model ID (for non-expert weights)
            pipeline_config: Pipeline configuration
            verbose: Print loading progress

        Returns:
            Configured UnifiedPipeline with compressed experts

        Example:
            pipeline = UnifiedPipeline.from_compressed(
                "gpt-oss-20b-overlay",
                "openai/gpt-oss-20b",
            )
            result = pipeline.chat("What is the capital of France?")
        """
        from chuk_lazarus.introspection.moe import OverlayExperts

        pipeline_config = pipeline_config or UnifiedPipelineConfig()
        compressed_path = Path(compressed_path)

        def log(msg: str) -> None:
            if verbose:
                print(msg)

        log(f"Loading compressed model from: {compressed_path}")
        log("=" * 60)

        # Load overlay
        log("\n1. Loading compressed overlay...")
        overlay = OverlayExperts.load(compressed_path)
        log(f"   Layers: {overlay.num_layers}, Experts: {overlay.num_experts}")
        log(f"   Compression: {overlay.config.compression_ratio:.1f}x")
        log(f"   Memory: {overlay.memory_usage_mb():.1f} MB")

        # Load original model using standard pipeline
        log("\n2. Loading base model...")
        pipeline = cls.from_pretrained(
            original_model_id,
            pipeline_config=pipeline_config,
            verbose=verbose,
        )

        # Patch experts to use overlay
        log("\n3. Patching experts with compressed weights...")
        cls._patch_experts_with_overlay(pipeline.model, overlay)
        log("   Experts patched!")

        # Update state
        if pipeline._state:
            pipeline._state.model_id = f"{original_model_id} (compressed)"

        log("\n" + "=" * 60)
        log(f"Compressed model loaded! ({overlay.config.compression_ratio:.1f}x compression)")

        return pipeline

    @staticmethod
    def _patch_experts_with_overlay(model, overlay) -> None:
        """Patch model to use compressed expert weights.

        Replaces expert forward passes with overlay-based reconstruction.
        """
        import mlx.core as mx
        import mlx.nn as nn

        # GPT-OSS custom SwiGLU activation
        def gpt_oss_swiglu(
            x_linear: mx.array, x_glu: mx.array, alpha: float = 1.702, limit: float = 7.0
        ) -> mx.array:
            """GPT-OSS custom SwiGLU: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)"""
            x_glu = mx.clip(x_glu, a_min=None, a_max=limit)
            x_linear = mx.clip(x_linear, a_min=-limit, a_max=limit)
            out_glu = x_glu * mx.sigmoid(alpha * x_glu)
            return out_glu * (x_linear + 1)

        # Create overlay-based batched experts class for GPT-OSS style models
        class OverlayBatchedExperts(nn.Module):
            """Batched experts using overlay reconstruction."""

            def __init__(self, layer_idx: int, overlay, hidden_size: int, num_experts: int):
                super().__init__()
                self.layer_idx = layer_idx
                self.overlay = overlay
                self.hidden_size = hidden_size
                self.num_experts = num_experts

            def __call__(
                self, x: mx.array, expert_indices: mx.array, expert_weights: mx.array
            ) -> mx.array:
                """Apply experts using overlay reconstruction."""
                num_tokens = x.shape[0]
                output = mx.zeros((num_tokens, self.hidden_size), dtype=x.dtype)

                for expert_idx in range(self.num_experts):
                    # Find which tokens use this expert
                    expert_mask = expert_indices == expert_idx
                    token_weights = mx.sum(
                        expert_weights * expert_mask.astype(expert_weights.dtype), axis=-1
                    )

                    if not mx.any(token_weights > 0):
                        continue

                    # Apply expert via overlay (efficient low-rank)
                    # Gate projection (even indices in original gate_up)
                    gate_out = self.overlay.apply_expert(self.layer_idx, "gate", expert_idx, x)
                    # Up projection (odd indices in original gate_up)
                    up_out = self.overlay.apply_expert(self.layer_idx, "up", expert_idx, x)
                    # GPT-OSS uses custom SwiGLU: (gate * sigmoid(alpha * gate)) * (up + 1)
                    hidden = gpt_oss_swiglu(up_out, gate_out)
                    expert_out = self.overlay.apply_expert(
                        self.layer_idx, "down", expert_idx, hidden
                    )

                    # Weight and accumulate
                    output = output + expert_out * token_weights[:, None]

                return output

        # Find layers and replace experts
        for layer_idx in overlay.moe_layer_indices:
            # Find the layer
            layer = None
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                layer = model.model.layers[layer_idx]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                layer = model.transformer.h[layer_idx]
            elif hasattr(model, "layers"):
                layer = model.layers[layer_idx]

            if layer is None:
                continue

            # Find MoE block - for GPT-OSS it's the mlp
            moe_block = None
            for attr in ["mlp", "block_sparse_moe", "moe", "feed_forward"]:
                if hasattr(layer, attr):
                    block = getattr(layer, attr)
                    if hasattr(block, "experts"):
                        moe_block = block
                        moe_attr = attr
                        break

            if moe_block is None:
                continue

            # Get hidden size from down projection output dimension
            # down_shape = (hidden_size, intermediate_size), so down_shape[0] = hidden_size
            hidden_size = overlay.config.down_shape[0]
            num_experts = overlay.num_experts

            # Replace the experts module with overlay version
            new_experts = OverlayBatchedExperts(layer_idx, overlay, hidden_size, num_experts)
            getattr(layer, moe_attr).experts = new_experts
