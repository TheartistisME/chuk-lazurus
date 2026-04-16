"""Async-native ExpertRouter for MoE expert manipulation.

Provides utilities for forcing, ablating, and analyzing expert routing
in Mixture of Experts models.

Example:
    >>> from chuk_lazarus.introspection.moe import ExpertRouter
    >>>
    >>> async with await ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
    ...     result = await router.generate_with_forced_expert(
    ...         prompt="127 * 89 = ",
    ...         expert_idx=6,
    ...         max_tokens=20,
    ...     )
    ...     print(result.response)
"""

from __future__ import annotations

import asyncio
import logging
import types
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any, Callable

from chuk_lazarus.introspection._backend_dispatch import lazy_mx as mx, register_hook

from .enums import MoEArchitecture, MoEImplementationType
from .models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    ExpertPair,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
    TopKVariationResult,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def torch_finfo_min(tensor: Any) -> float:
    """Best-effort negative sentinel for masking torch logits."""
    if not hasattr(tensor, "dtype"):
        return float("-inf")

    import torch

    return float(torch.finfo(tensor.dtype).min)


class ExpertRouter:
    """Async-native utility for manipulating expert routing.

    This class provides methods to:
    - Force all routing to a specific expert
    - Ablate (remove) experts from routing
    - Vary top-k expert selection
    - Capture and analyze router weights
    - Analyze expert co-activation patterns

    Example:
        >>> async with await ExpertRouter.from_pretrained("openai/gpt-oss-20b") as router:
        ...     # Chat with a specific expert
        ...     result = await router.chat_with_expert("127 * 89 = ", expert_idx=6)
        ...     print(result.response)
        ...
        ...     # Compare multiple experts
        ...     comparison = await router.compare_experts(
        ...         "def fibonacci(n):",
        ...         expert_indices=[6, 7, 20],
        ...     )
        ...     for r in comparison.expert_results:
        ...         print(f"Expert {r.expert_idx}: {r.response[:50]}...")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_info: MoEModelInfo,
        *,
        runtime: Any | None = None,
        pipeline: Any | None = None,
        backend_name: str | None = None,
    ):
        """Initialize ExpertRouter.

        Args:
            model: The loaded model.
            tokenizer: The tokenizer for the model.
            model_info: Information about the MoE architecture.
        """
        self._model = model
        self._tokenizer = tokenizer
        self._info = model_info
        self._runtime = runtime
        self._pipeline = pipeline
        self._backend_name = backend_name or self._resolve_backend_name(runtime) or "mlx"
        self._moe_type = self._detect_moe_type()

        if not self._info.moe_layers:
            raise ValueError("Model has no MoE layers")

    @classmethod
    async def from_pretrained(
        cls,
        model_id: str,
        *,
        backend: str | None = None,
        device: str | None = None,
    ) -> ExpertRouter:
        """Load model and create ExpertRouter.

        Args:
            model_id: HuggingFace model ID or local path.

        Returns:
            Configured ExpertRouter instance.

        Example:
            >>> router = await ExpertRouter.from_pretrained("openai/gpt-oss-20b")
        """
        # Run model loading in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        loaded = await loop.run_in_executor(
            None,
            cls._load_model_sync,
            model_id,
            backend,
            device,
        )
        if len(loaded) == 3:
            model, tokenizer, model_info = loaded
            runtime = None
            pipeline = None
            backend_name = None
        else:
            model, tokenizer, model_info, runtime, pipeline, backend_name = loaded
        return cls(
            model,
            tokenizer,
            model_info,
            runtime=runtime,
            pipeline=pipeline,
            backend_name=backend_name,
        )

    @staticmethod
    def _load_model_sync(
        model_id: str,
        backend: str | None = None,
        device: str | None = None,
    ) -> tuple[Any, Any, MoEModelInfo, Any, Any, str]:
        """Synchronously load model (called in thread pool)."""
        from ...inference import UnifiedPipeline, UnifiedPipelineConfig
        from ...models_v2.core.backend import get_backend

        logger.info(f"Loading model: {model_id}")

        resolved_backend = get_backend(name=backend, device=device)
        backend_name = str(resolved_backend.name).lower()
        resolved_device = getattr(resolved_backend, "device", None) if backend_name == "torch" else None

        pipeline_config = UnifiedPipelineConfig(
            backend_name=backend_name,
            device=resolved_device,
        )

        try:
            pipeline = UnifiedPipeline.from_pretrained(
                model_id,
                pipeline_config=pipeline_config,
                verbose=False,
            )
        except ValueError as exc:
            message = str(exc).lower()
            if "detect model family" in message or "supported yet" in message:
                raise ValueError(f"Unsupported model: {model_id}") from exc
            raise

        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Extract MoE info
        model_info = ExpertRouter._extract_moe_info(model)

        return model, tokenizer, model_info, pipeline.runtime, pipeline, backend_name

    @staticmethod
    def _resolve_backend_name(runtime: Any | None) -> str | None:
        backend = getattr(runtime, "backend", None)
        if backend is None:
            return None
        name = str(backend).lower()
        if name == "cuda":
            return "torch"
        if name == "mlx":
            return "mlx"
        return name

    @staticmethod
    def _get_layers_for_model(model: Any) -> list[Any]:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "layers"):
            return list(model.layers)
        raise AttributeError("Cannot find model layers for ExpertRouter.")

    @staticmethod
    def _extract_moe_info(model: Any) -> MoEModelInfo:
        """Extract MoE information from a model."""
        layers = ExpertRouter._get_layers_for_model(model)
        moe_layers: list[int] = []
        num_experts = 0
        num_experts_per_tok = 0
        has_shared_expert = False
        architecture = MoEArchitecture.GENERIC

        for i, layer in enumerate(layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_layers.append(i)
                router = layer.mlp.router
                num_experts = getattr(router, "num_experts", 0)
                num_experts_per_tok = getattr(router, "num_experts_per_tok", 0)

                # Detect architecture
                if hasattr(layer.mlp, "shared_expert"):
                    has_shared_expert = True
                    architecture = MoEArchitecture.LLAMA4

        # Detect GPT-OSS style
        if num_experts == 32 and num_experts_per_tok == 4:
            architecture = MoEArchitecture.GPT_OSS
        elif num_experts == 8 and num_experts_per_tok == 2:
            architecture = MoEArchitecture.MIXTRAL

        return MoEModelInfo(
            moe_layers=tuple(moe_layers),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            total_layers=len(layers),
            architecture=architecture,
            has_shared_expert=has_shared_expert,
        )

    def _detect_moe_type(self) -> MoEImplementationType:
        """Detect the MoE type based on model structure."""
        if not self._info.moe_layers:
            return MoEImplementationType.NONE

        layer_idx = self._info.moe_layers[0]
        layer = self._get_layers()[layer_idx]
        mlp = layer.mlp

        # Check for GPT-OSS batched experts style
        # GPT-OSS uses quantized weights: gate_up_proj_blocks, down_proj_blocks
        if hasattr(mlp, "experts") and hasattr(mlp.experts, "gate_up_proj_blocks"):
            return MoEImplementationType.GPT_OSS_BATCHED

        return MoEImplementationType.STANDARD

    async def __aenter__(self) -> ExpertRouter:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        pass

    @property
    def info(self) -> MoEModelInfo:
        """Get MoE model information."""
        return self._info

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._tokenizer

    def _get_layers(self) -> list[Any]:
        return self._get_layers_for_model(self._model)

    def _get_layer_mlp(self, layer_idx: int) -> Any:
        return self._get_layers()[layer_idx].mlp

    def _get_layer_router(self, layer_idx: int) -> Any:
        return self._get_layer_mlp(layer_idx).router

    def _resolve_target_layers(self, layers: list[int] | None) -> list[int]:
        return list(layers) if layers else list(self._info.moe_layers)

    @contextmanager
    def _patched_num_experts_per_tok(
        self,
        target_layers: list[int],
        new_k: int,
    ):
        originals: list[tuple[Any, int]] = []
        try:
            for layer_idx in target_layers:
                mlp = self._get_layer_mlp(layer_idx)
                router = self._get_layer_router(layer_idx)
                for obj in (mlp, router):
                    if obj is None or not hasattr(obj, "num_experts_per_tok"):
                        continue
                    originals.append((obj, getattr(obj, "num_experts_per_tok")))
                    max_k = getattr(obj, "num_experts", None)
                    effective_k = max(1, min(new_k, max_k)) if max_k is not None else max(1, new_k)
                    setattr(obj, "num_experts_per_tok", effective_k)
            yield
        finally:
            for obj, original_value in reversed(originals):
                setattr(obj, "num_experts_per_tok", original_value)

    @contextmanager
    def _patched_torch_routers(
        self,
        target_layers: list[int],
        patch_factory: Callable[[int, Any, Callable[..., Any]], Callable[..., Any]],
    ):
        originals: list[tuple[Any, Callable[..., Any]]] = []
        try:
            for layer_idx in target_layers:
                router = self._get_layer_router(layer_idx)
                original_forward = router.forward
                router.forward = types.MethodType(
                    patch_factory(layer_idx, router, original_forward),
                    router,
                )
                originals.append((router, original_forward))
            yield
        finally:
            for router, original_forward in reversed(originals):
                router.forward = original_forward

    @staticmethod
    def _torch_router_projection(router: Any) -> tuple[Any | None, Any | None]:
        for candidate in (router, getattr(router, "router", None), getattr(router, "gate", None)):
            if candidate is None:
                continue
            weight = getattr(candidate, "weight", None)
            if weight is not None:
                return weight, getattr(candidate, "bias", None)
        return None, None

    def _torch_router_logits(self, router: Any, x: Any, original_output: Any) -> Any:
        weight, bias = self._torch_router_projection(router)
        if weight is not None:
            logits = x @ weight.T
            if bias is not None:
                logits = logits + bias
            return logits
        if not isinstance(original_output, tuple):
            return original_output
        raise RuntimeError(
            f"Router {type(router).__name__} does not expose a linear projection for torch MoE patching."
        )

    @staticmethod
    def _torch_router_output_from_logits(
        router: Any,
        logits: Any,
        original_output: Any,
        *,
        k_override: int | None = None,
    ) -> Any:
        if not isinstance(original_output, tuple):
            return logits

        k = k_override
        if k is None:
            k = getattr(router, "num_experts_per_tok", None)
        if k is None:
            k = original_output[1].shape[-1]
        k = max(1, min(int(k), int(logits.shape[-1])))
        top_k_logits, top_k_indices = logits.topk(k=k, dim=-1)
        top_k_weights = top_k_logits.softmax(dim=-1)
        return top_k_weights, top_k_indices

    def _build_forced_expert_patch(
        self,
        forced_expert: int,
    ) -> Callable[[int, Any, Callable[..., Any]], Callable[..., Any]]:
        def factory(_layer_idx: int, router: Any, original_forward: Callable[..., Any]) -> Callable[..., Any]:
            def patched_forward(router_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
                import torch

                original_output = original_forward(x, *args, **kwargs)
                if isinstance(original_output, tuple):
                    expert_weights = torch.ones(
                        (*x.shape[:-1], 1),
                        dtype=x.dtype,
                        device=x.device,
                    )
                    expert_indices = torch.full(
                        (*x.shape[:-1], 1),
                        forced_expert,
                        dtype=torch.long,
                        device=x.device,
                    )
                    return expert_weights, expert_indices

                logits = self._torch_router_logits(router, x, original_output)
                forced_logits = logits.new_full(logits.shape, torch_finfo_min(logits))
                forced_logits[..., forced_expert] = 0.0
                return self._torch_router_output_from_logits(router_self, forced_logits, original_output)

            return patched_forward

        return factory

    def _build_ablation_patch(
        self,
        ablate_set: set[int],
    ) -> Callable[[int, Any, Callable[..., Any]], Callable[..., Any]]:
        def factory(_layer_idx: int, router: Any, original_forward: Callable[..., Any]) -> Callable[..., Any]:
            def patched_forward(router_self: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
                original_output = original_forward(x, *args, **kwargs)
                logits = self._torch_router_logits(router, x, original_output)
                masked_logits = logits.clone()
                if ablate_set:
                    masked_logits[..., sorted(ablate_set)] = torch_finfo_min(masked_logits)
                return self._torch_router_output_from_logits(router_self, masked_logits, original_output)

            return patched_forward

        return factory

    def _torch_generation_result(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        target_layers: list[int] | None = None,
        patch_factory: Callable[[int, Any, Callable[..., Any]], Callable[..., Any]] | None = None,
        topk_override: int | None = None,
    ):
        from ...inference.generation import GenerationConfig

        if self._runtime is None:
            raise RuntimeError("Torch ExpertRouter path requires an attached runtime.")

        effective_layers = target_layers or list(self._info.moe_layers)
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            use_plugins=False,
        )

        with ExitStack() as stack:
            if topk_override is not None:
                stack.enter_context(self._patched_num_experts_per_tok(effective_layers, topk_override))
            if patch_factory is not None:
                stack.enter_context(self._patched_torch_routers(effective_layers, patch_factory))
            return self._runtime.generate(prompt, generation_config)

    def _torch_forward_inputs(self, prompt: str) -> dict[str, Any]:
        runtime = self._runtime
        if runtime is not None:
            tokenize_prompt = getattr(runtime, "_tokenize_prompt", None)
            if callable(tokenize_prompt):
                return tokenize_prompt(prompt)

        import torch

        batch = self._tokenizer(prompt, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        return {
            name: tensor.to(model_device, non_blocking=True) for name, tensor in batch.items()
        }

    def _run_torch_forward(self, prompt: str) -> None:
        import torch

        model_inputs = self._torch_forward_inputs(prompt)
        try:
            with torch.inference_mode():
                self._model(**model_inputs, use_cache=False)
        except TypeError:
            with torch.inference_mode():
                self._model(input_ids=model_inputs["input_ids"], use_cache=False)

    # =========================================================================
    # Generation Methods
    # =========================================================================

    async def chat_with_expert(
        self,
        prompt: str,
        expert_idx: int,
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
        temperature: float = 0.0,
        apply_chat_template: bool = True,
    ) -> ExpertChatResult:
        """Generate response with routing forced to a specific expert.

        Args:
            prompt: The input prompt.
            expert_idx: Expert index to force routing to.
            max_tokens: Maximum tokens to generate.
            layers: Specific layers to modify (None = all MoE layers).
            temperature: Sampling temperature.
            apply_chat_template: Whether to apply chat template.

        Returns:
            ExpertChatResult with response and statistics.
        """
        loop = asyncio.get_event_loop()
        response, stats = await loop.run_in_executor(
            None,
            self._generate_with_forced_expert_sync,
            prompt,
            expert_idx,
            max_tokens,
            layers,
            temperature,
            apply_chat_template,
        )

        return ExpertChatResult(
            prompt=prompt,
            response=response,
            expert_idx=expert_idx,
            stats=stats,
        )

    def _generate_with_forced_expert_sync(
        self,
        prompt: str,
        expert_idx: int,
        max_tokens: int,
        layers: list[int] | None,
        temperature: float,
        apply_chat_template: bool,
    ) -> tuple[str, GenerationStats]:
        """Synchronous implementation of forced expert generation.

        Uses router patching to force all tokens to route to a specific expert,
        letting the model's own expert code handle the actual computation.
        """
        # Apply chat template if requested
        if apply_chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            if self._tokenizer.chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

        target_layers = self._resolve_target_layers(layers)
        if self._backend_name == "torch":
            result = self._torch_generation_result(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                target_layers=target_layers,
                patch_factory=self._build_forced_expert_patch(expert_idx),
                topk_override=1,
            )
            stats = GenerationStats(
                expert_idx=expert_idx,
                tokens_generated=result.stats.output_tokens,
                layers_modified=len(target_layers),
                moe_type=self._moe_type,
                prompt_tokens=result.stats.input_tokens,
            )
            return result.text, stats

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers_set = set(target_layers)
        forced_expert = expert_idx

        # Patch the router class to force routing to specific expert
        # This lets the MoE layer's existing expert code handle everything
        sample_layer = self._get_layers()[target_layers[0]]
        router_class = type(sample_layer.mlp.router)
        original_router_call = router_class.__call__

        def patched_router_call(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            # Find which layer this router belongs to
            layer_idx = -1
            for i, layer in enumerate(self._get_layers()):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            # Only force routing for target layers
            if layer_idx not in target_layers_set:
                return original_router_call(router_self, x)

            # Handle both 2D and 3D inputs
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])

            num_tokens = x.shape[0]

            # Force all tokens to route to the specified expert with weight 1.0
            # Use k=1 to route to single expert
            indices = mx.full((num_tokens, 1), forced_expert, dtype=mx.int32)
            weights = mx.ones((num_tokens, 1), dtype=x.dtype)

            return weights, indices

        try:
            # Apply class-level patch to router
            router_class.__call__ = patched_router_call

            # Generate
            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                output = self._model(input_ids, cache=cache)
                # Handle both tuple and ModelOutput returns
                if hasattr(output, "logits"):
                    logits = output.logits
                    cache = getattr(output, "cache", None)
                elif isinstance(output, tuple):
                    logits, cache = output
                else:
                    logits = output
                    cache = None
                next_token = self._sample_token(logits, temperature)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            # Restore original router class __call__
            router_class.__call__ = original_router_call

        stats = GenerationStats(
            expert_idx=expert_idx,
            tokens_generated=len(generated),
            layers_modified=len(target_layers),
            moe_type=self._moe_type,
            prompt_tokens=input_ids.shape[-1],
        )

        return text, stats

    def _sample_token(self, logits: mx.array, temperature: float) -> int:
        """Sample a token from logits."""
        logits = logits[:, -1, :]  # Get last position

        if temperature == 0.0:
            return int(mx.argmax(logits, axis=-1).item())

        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        return int(mx.random.categorical(mx.log(probs)).item())

    async def compare_experts(
        self,
        prompt: str,
        expert_indices: list[int],
        *,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> ExpertComparisonResult:
        """Compare multiple experts on the same prompt.

        Args:
            prompt: The input prompt.
            expert_indices: List of expert indices to compare.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            ExpertComparisonResult with results from each expert.
        """
        results: list[ExpertChatResult] = []

        for expert_idx in expert_indices:
            result = await self.chat_with_expert(
                prompt,
                expert_idx,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)

        return ExpertComparisonResult(
            prompt=prompt,
            expert_results=tuple(results),
        )

    async def generate_with_ablation(
        self,
        prompt: str,
        expert_indices: list[int],
        *,
        max_tokens: int = 100,
        layers: list[int] | None = None,
    ) -> tuple[str, GenerationStats]:
        """Generate with specific experts ablated (removed from routing).

        Args:
            prompt: The input prompt.
            expert_indices: Expert indices to ablate.
            max_tokens: Maximum tokens to generate.
            layers: Specific layers to modify (None = all MoE layers).

        Returns:
            Tuple of (response text, generation stats).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_with_ablation_sync,
            prompt,
            expert_indices,
            max_tokens,
            layers,
        )

    def _generate_with_ablation_sync(
        self,
        prompt: str,
        expert_indices: list[int],
        max_tokens: int,
        layers: list[int] | None,
    ) -> tuple[str, GenerationStats]:
        """Synchronous implementation of ablated generation.

        Uses router patching to exclude specific experts from routing,
        letting the model's own expert code handle everything else.
        """
        target_layers = self._resolve_target_layers(layers)
        if self._backend_name == "torch":
            result = self._torch_generation_result(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                target_layers=target_layers,
                patch_factory=self._build_ablation_patch(set(expert_indices)),
            )
            stats = GenerationStats(
                expert_idx=-1,
                tokens_generated=result.stats.output_tokens,
                layers_modified=len(target_layers),
                moe_type=self._moe_type,
                prompt_tokens=result.stats.input_tokens,
            )
            return result.text, stats

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers_set = set(target_layers)
        ablate_set = set(expert_indices)

        # Patch the router class to exclude ablated experts
        sample_layer = self._get_layers()[target_layers[0]]
        router_class = type(sample_layer.mlp.router)
        original_router_call = router_class.__call__

        def patched_router_call(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            # Find which layer this router belongs to
            layer_idx = -1
            for i, layer in enumerate(self._get_layers()):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            # Only modify routing for target layers
            if layer_idx not in target_layers_set:
                return original_router_call(router_self, x)

            # Handle both 2D and 3D inputs
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])

            # Compute router logits using router's own weights
            logits = x @ router_self.weight.T
            if hasattr(router_self, "bias") and router_self.bias is not None:
                logits = logits + router_self.bias

            # Mask out ablated experts with very negative value
            num_experts = logits.shape[-1]
            mask_values = [-1e9 if i in ablate_set else 0.0 for i in range(num_experts)]
            mask = mx.array(mask_values, dtype=logits.dtype)
            logits = logits + mask

            # Get top-k experts (using original k value)
            k = router_self.num_experts_per_tok
            partitioned_indices = mx.argpartition(logits, kth=-k, axis=-1)
            top_k_indices = partitioned_indices[..., -k:]

            # Get corresponding logits and apply softmax
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            top_k_weights = mx.softmax(top_k_logits, axis=-1)

            return top_k_weights, top_k_indices

        try:
            # Apply class-level patch to router
            router_class.__call__ = patched_router_call

            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                output = self._model(input_ids, cache=cache)
                # Handle both tuple and ModelOutput returns
                if hasattr(output, "logits"):
                    logits = output.logits
                    cache = getattr(output, "cache", None)
                elif isinstance(output, tuple):
                    logits, cache = output
                else:
                    logits = output
                    cache = None
                next_token = self._sample_token(logits, 0.0)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            # Restore original router class __call__
            router_class.__call__ = original_router_call

        stats = GenerationStats(
            expert_idx=-1,  # No specific expert forced
            tokens_generated=len(generated),
            layers_modified=len(target_layers),
            moe_type=self._moe_type,
            prompt_tokens=input_ids.shape[-1],
        )

        return text, stats

    async def generate_with_topk(
        self,
        prompt: str,
        k: int,
        *,
        max_tokens: int = 100,
    ) -> TopKVariationResult:
        """Generate with modified top-k expert selection.

        Args:
            prompt: The input prompt.
            k: Number of experts to use (instead of default).
            max_tokens: Maximum tokens to generate.

        Returns:
            TopKVariationResult with both normal and modified responses.
        """
        loop = asyncio.get_event_loop()

        # Get normal response
        normal_text = await loop.run_in_executor(
            None, self._generate_normal_sync, prompt, max_tokens
        )

        # Get modified top-k response
        topk_text = await loop.run_in_executor(
            None, self._generate_with_topk_sync, prompt, k, max_tokens
        )

        return TopKVariationResult(
            prompt=prompt,
            k_value=k,
            default_k=self._info.num_experts_per_tok,
            response=topk_text,
            normal_response=normal_text,
        )

    def _generate_normal_sync(self, prompt: str, max_tokens: int) -> str:
        """Synchronous normal generation."""
        if self._backend_name == "torch":
            result = self._torch_generation_result(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return result.text

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        generated: list[int] = []
        cache = None

        for _ in range(max_tokens):
            output = self._model(input_ids, cache=cache)
            # Handle both tuple and ModelOutput returns
            if hasattr(output, "logits"):
                logits = output.logits
                cache = getattr(output, "cache", None)
            elif isinstance(output, tuple):
                logits, cache = output
            else:
                logits = output
                cache = None
            next_token = self._sample_token(logits, 0.0)
            generated.append(next_token)

            if next_token == self._tokenizer.eos_token_id:
                break

            input_ids = mx.array([[next_token]])

        return self._tokenizer.decode(generated)

    def _generate_with_topk_sync(self, prompt: str, k: int, max_tokens: int) -> str:
        """Synchronous top-k modified generation.

        Modifies the router to use a different k value during generation.
        This allows testing how model behavior changes with more or fewer experts.

        Args:
            prompt: The input prompt.
            k: Number of experts to select per token.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text with modified top-k routing.
        """
        target_layers = self._resolve_target_layers(None)
        if self._backend_name == "torch":
            result = self._torch_generation_result(
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                target_layers=target_layers,
                topk_override=k,
            )
            return result.text

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        target_layers_set = set(target_layers)
        new_k = k

        # Patch the router class to return different k
        # This lets the MoE layer's existing expert code handle everything else
        sample_layer = self._get_layers()[target_layers[0]]
        router_class = type(sample_layer.mlp.router)
        original_router_call = router_class.__call__

        def patched_router_call(router_self: Any, x: mx.array) -> tuple[mx.array, mx.array]:
            # Find which layer this router belongs to
            layer_idx = -1
            for i, layer in enumerate(self._get_layers()):
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                    if layer.mlp.router is router_self:
                        layer_idx = i
                        break

            # Only modify routing for target layers
            if layer_idx not in target_layers_set:
                return original_router_call(router_self, x)

            # Handle both 2D and 3D inputs
            if x.ndim == 3:
                batch_size, seq_len, hidden_size = x.shape
                x = x.reshape(-1, hidden_size)

            # Compute router logits using router's own weights
            logits = x @ router_self.weight.T
            if hasattr(router_self, "bias") and router_self.bias is not None:
                logits = logits + router_self.bias

            # Clamp k to valid range
            effective_k = min(new_k, router_self.num_experts)
            effective_k = max(1, effective_k)

            # Get top-k experts with modified k (using argpartition like original)
            partitioned_indices = mx.argpartition(logits, kth=-effective_k, axis=-1)
            top_k_indices = partitioned_indices[..., -effective_k:]

            # Get corresponding logits and apply softmax
            top_k_logits = mx.take_along_axis(logits, top_k_indices, axis=-1)
            top_k_weights = mx.softmax(top_k_logits, axis=-1)

            return top_k_weights, top_k_indices

        try:
            # Apply class-level patch to router
            router_class.__call__ = patched_router_call

            # Generate with modified routing
            generated: list[int] = []
            cache = None

            for _ in range(max_tokens):
                output = self._model(input_ids, cache=cache)
                # Handle both tuple and ModelOutput returns
                if hasattr(output, "logits"):
                    logits = output.logits
                    cache = getattr(output, "cache", None)
                elif isinstance(output, tuple):
                    logits, cache = output
                else:
                    logits = output
                    cache = None
                next_token = self._sample_token(logits, 0.0)
                generated.append(next_token)

                if next_token == self._tokenizer.eos_token_id:
                    break

                input_ids = mx.array([[next_token]])

            text = self._tokenizer.decode(generated)

        finally:
            # Restore original router class __call__
            router_class.__call__ = original_router_call

        return text

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    async def capture_router_weights(
        self,
        prompt: str,
        *,
        layers: list[int] | None = None,
    ) -> list[LayerRouterWeights]:
        """Capture router weights for each token position.

        Args:
            prompt: The input prompt.
            layers: Specific layers to capture (None = all MoE layers).

        Returns:
            List of LayerRouterWeights for each layer.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture_router_weights_sync, prompt, layers)

    def _capture_router_weights_torch(
        self,
        prompt: str,
        layers: list[int] | None,
    ) -> list[LayerRouterWeights]:
        target_layers = self._resolve_target_layers(layers)
        token_ids = self._tokenizer.encode(prompt)
        tokens = [self._tokenizer.decode([t]) for t in token_ids]
        captured_weights: dict[int, list[tuple[list[int], list[float]]]] = {
            layer_idx: [] for layer_idx in target_layers
        }

        def make_hook(layer_idx: int):
            def hook(_module: Any, args: tuple[Any, ...], output: Any) -> None:
                if captured_weights[layer_idx]:
                    return

                import torch

                if isinstance(output, tuple):
                    weights, indices = output
                else:
                    router = self._get_layer_router(layer_idx)
                    logits = output
                    if logits.ndim == 3:
                        k = max(
                            1,
                            min(
                                getattr(router, "num_experts_per_tok", self._info.num_experts_per_tok),
                                int(logits.shape[-1]),
                            ),
                        )
                        top_logits, indices = torch.topk(logits, k=k, dim=-1)
                        weights = torch.softmax(top_logits, dim=-1)
                    else:
                        k = max(
                            1,
                            min(
                                getattr(router, "num_experts_per_tok", self._info.num_experts_per_tok),
                                int(logits.shape[-1]),
                            ),
                        )
                        top_logits, indices = torch.topk(logits, k=k, dim=-1)
                        weights = torch.softmax(top_logits, dim=-1)

                if weights.ndim == 3:
                    weights = weights[0]
                    indices = indices[0]
                elif weights.ndim == 1:
                    weights = weights.unsqueeze(0)
                    indices = indices.unsqueeze(0)

                for pos in range(weights.shape[0]):
                    pos_indices = [int(idx) for idx in indices[pos].detach().cpu().tolist()]
                    pos_weights = [float(value) for value in weights[pos].detach().cpu().tolist()]
                    captured_weights[layer_idx].append((pos_indices, pos_weights))

            return hook

        detachers: list[Callable[[], None]] = []
        try:
            for layer_idx in target_layers:
                router = self._get_layer_router(layer_idx)
                detachers.append(register_hook(self._backend_name, router, make_hook(layer_idx)))
            self._run_torch_forward(prompt)
        finally:
            for detach in reversed(detachers):
                detach()

        results: list[LayerRouterWeights] = []
        for layer_idx in target_layers:
            positions: list[RouterWeightCapture] = []
            for pos_idx, (exp_indices, weights) in enumerate(captured_weights[layer_idx]):
                token = tokens[pos_idx] if pos_idx < len(tokens) else ""
                positions.append(
                    RouterWeightCapture(
                        layer_idx=layer_idx,
                        position_idx=pos_idx,
                        token=token,
                        expert_indices=tuple(exp_indices),
                        weights=tuple(weights),
                    )
                )
            results.append(LayerRouterWeights(layer_idx=layer_idx, positions=tuple(positions)))

        return results

    def _capture_router_weights_sync(
        self,
        prompt: str,
        layers: list[int] | None,
    ) -> list[LayerRouterWeights]:
        """Synchronous router weight capture."""
        if self._backend_name == "torch":
            return self._capture_router_weights_torch(prompt, layers)

        input_ids = mx.array(self._tokenizer.encode(prompt))[None, :]
        tokens = [self._tokenizer.decode([t]) for t in input_ids[0].tolist()]
        target_layers = self._resolve_target_layers(layers)

        results: list[LayerRouterWeights] = []
        captured_weights: dict[int, list[tuple[list[int], list[float]]]] = {
            layer_idx: [] for layer_idx in target_layers
        }

        # Get the MLP class for class-level patching (instance patching doesn't work)
        mlp_class = type(self._get_layers()[0].mlp)
        original_call = mlp_class.__call__

        def patched_call(mlp_self: Any, x: mx.array) -> mx.array:
            # Find which layer this MLP belongs to
            layer_idx = -1
            for i, layer in enumerate(self._get_layers()):
                if layer.mlp is mlp_self:
                    layer_idx = i
                    break

            # Only capture for target layers
            if layer_idx in target_layers:
                router = mlp_self.router
                k = self._info.num_experts_per_tok

                # Flatten x for router (matches how MoE block calls router)
                if x.ndim == 3:
                    batch_size, seq_len, hidden_size = x.shape
                    x_flat = x.reshape(-1, hidden_size)
                else:
                    seq_len = 1
                    x_flat = x

                # Router may return (weights, indices) tuple or just logits
                router_result = router(x_flat)

                if isinstance(router_result, tuple):
                    # Router already computed weights and indices
                    weights, indices = router_result
                    # weights: (batch * seq_len, k), indices: (batch * seq_len, k)
                    for pos in range(seq_len):
                        pos_indices = indices[pos].tolist()
                        pos_weights = [float(weights[pos, i].item()) for i in range(k)]
                        captured_weights[layer_idx].append((pos_indices, pos_weights))
                else:
                    # Router returned logits, compute weights ourselves
                    router_logits = router_result
                    weights = mx.softmax(router_logits, axis=-1)
                    for pos in range(seq_len):
                        pos_weights = weights[pos]
                        top_indices = mx.argsort(pos_weights)[-k:][::-1].tolist()
                        top_weights = [float(pos_weights[i].item()) for i in top_indices]
                        captured_weights[layer_idx].append((top_indices, top_weights))

            return original_call(mlp_self, x)

        try:
            # Patch at class level
            mlp_class.__call__ = patched_call

            # Run forward pass
            self._model(input_ids)

        finally:
            # Restore original
            mlp_class.__call__ = original_call

        # Convert to structured results
        for layer_idx in target_layers:
            positions: list[RouterWeightCapture] = []
            for pos_idx, (exp_indices, weights) in enumerate(captured_weights[layer_idx]):
                token = tokens[pos_idx] if pos_idx < len(tokens) else ""
                positions.append(
                    RouterWeightCapture(
                        layer_idx=layer_idx,
                        position_idx=pos_idx,
                        token=token,
                        expert_indices=tuple(exp_indices),
                        weights=tuple(weights),
                    )
                )
            results.append(LayerRouterWeights(layer_idx=layer_idx, positions=tuple(positions)))

        return results

    async def analyze_coactivation(
        self,
        prompts: list[str],
        *,
        layer_idx: int | None = None,
    ) -> CoactivationAnalysis:
        """Analyze expert co-activation patterns across prompts.

        Args:
            prompts: List of prompts to analyze.
            layer_idx: Specific layer to analyze (None = first MoE layer).

        Returns:
            CoactivationAnalysis with co-activation statistics.
        """
        loop = asyncio.get_event_loop()
        target_layer = layer_idx if layer_idx is not None else self._info.moe_layers[0]

        return await loop.run_in_executor(
            None, self._analyze_coactivation_sync, prompts, target_layer
        )

    def _analyze_coactivation_sync(
        self,
        prompts: list[str],
        layer_idx: int,
    ) -> CoactivationAnalysis:
        """Synchronous co-activation analysis."""
        from collections import Counter

        expert_counts: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        total_activations = 0

        for prompt in prompts:
            weights_list = self._capture_router_weights_sync(prompt, [layer_idx])
            if not weights_list:
                continue

            layer_weights = weights_list[0]
            for pos in layer_weights.positions:
                experts = pos.expert_indices
                total_activations += 1

                for exp in experts:
                    expert_counts[exp] += 1

                # Count pairs
                for i, exp_a in enumerate(experts):
                    for exp_b in experts[i + 1 :]:
                        pair = (min(exp_a, exp_b), max(exp_a, exp_b))
                        pair_counts[pair] += 1

        # Build top pairs
        top_pairs: list[ExpertPair] = []
        for (exp_a, exp_b), count in pair_counts.most_common(20):
            rate = count / total_activations if total_activations > 0 else 0.0
            top_pairs.append(
                ExpertPair(
                    expert_a=exp_a,
                    expert_b=exp_b,
                    coactivation_count=count,
                    coactivation_rate=rate,
                )
            )

        # Find generalist experts (high activation rate)
        threshold = total_activations / self._info.num_experts * 1.5
        generalists = tuple(exp for exp, count in expert_counts.items() if count > threshold)

        return CoactivationAnalysis(
            layer_idx=layer_idx,
            total_activations=total_activations,
            top_pairs=tuple(top_pairs),
            specialist_pairs=(),  # Would need additional analysis
            generalist_experts=generalists,
        )
