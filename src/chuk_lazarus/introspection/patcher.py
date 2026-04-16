"""Activation patching for causal intervention experiments.

Provides tools for patching activations from one prompt into another
to test causal relationships in neural network computations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .accessor import AsyncModelAccessor
from .enums import PatchEffect
from .models.patching import PatchingLayerResult, PatchingResult

if TYPE_CHECKING:
    import mlx.core as mx

    from .models.patching import CommutativityResult


class LayerPatch(BaseModel):
    """A patch to apply at a specific layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    layer: int = Field(description="Layer index to patch")
    activation: Any = Field(description="Activation to patch (numpy or mlx array)")
    blend: float = Field(default=1.0, ge=0.0, le=1.0, description="Blend factor")
    position: int = Field(default=-1, description="Token position (-1 for last)")


class _RuntimeAwareIntrospector(BaseModel):
    """Shared backend-aware helpers for patching and commutativity analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any = Field(description="The neural network model")
    tokenizer: Any = Field(description="The tokenizer")
    config: Any = Field(default=None, description="Optional configuration")
    runtime: Any = Field(default=None, description="Optional inference runtime")
    _accessor: AsyncModelAccessor = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes after model creation."""
        self._accessor = AsyncModelAccessor(model=self.model, config=self.config)

    @property
    def num_layers(self) -> int:
        """Number of transformer layers exposed by the model."""
        return self._accessor.num_layers

    def _model_id(self) -> str:
        """Best-effort model identifier across MLX and torch configs."""
        return (
            getattr(self.config, "model_id", None)
            or getattr(self.config, "_name_or_path", None)
            or "unknown"
        )

    def _backend_name(self) -> str:
        """Infer the active runtime backend without importing MLX on torch hosts."""
        backend = getattr(self.runtime, "backend", None)
        if backend is not None:
            name = str(backend).lower()
            return "torch" if name == "cuda" else name

        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                first_param = next(parameters())
            except TypeError:
                try:
                    first_param = next(iter(parameters()))
                except (StopIteration, TypeError):
                    first_param = None
            except StopIteration:
                first_param = None

            if first_param is not None and type(first_param).__module__.startswith("torch"):
                return "torch"

        return "mlx"

    def _resolve_torch_layers(self) -> list[Any]:
        """Locate transformer blocks on a torch-backed model."""
        runtime_layers = getattr(self.runtime, "_resolve_layers", None)
        if callable(runtime_layers):
            return list(runtime_layers())

        candidates = [
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            ("layers",),
        ]

        for path in candidates:
            target = self.model
            for step in path[:-1]:
                target = getattr(target, step, None)
                if target is None:
                    break
            if target is None:
                continue

            layers = getattr(target, path[-1], None)
            if layers is None or not hasattr(layers, "__iter__") or not hasattr(layers, "__len__"):
                continue
            try:
                length = len(layers)
            except TypeError:
                continue
            if length:
                return list(layers)

        raise ValueError("Cannot resolve transformer layers for torch activation patching.")

    def _torch_device(self):
        """Resolve the device for torch tokenization and forward passes."""
        runtime_device = getattr(self.runtime, "_device", None)
        if runtime_device is not None:
            return runtime_device

        import torch

        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                first_param = next(self.model.parameters())
            except StopIteration:
                first_param = None
            if first_param is not None:
                return first_param.device

        return torch.device("cpu")

    def _tokenize_torch_prompt(self, prompt: str) -> dict[str, Any]:
        """Tokenize a prompt for torch runtimes, preserving device placement."""
        runtime_tokenize = getattr(self.runtime, "_tokenize_prompt", None)
        if callable(runtime_tokenize):
            return runtime_tokenize(prompt)

        import torch

        device = self._torch_device()
        non_blocking = getattr(device, "type", "") == "cuda"
        if callable(self.tokenizer):
            batch = self.tokenizer(prompt, return_tensors="pt")
            batch = getattr(batch, "data", batch)
            return {
                name: tensor.to(device, non_blocking=non_blocking)
                if hasattr(tensor, "to")
                else torch.as_tensor(tensor, device=device)
                for name, tensor in dict(batch).items()
            }

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        except TypeError:
            input_ids = self.tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        elif hasattr(input_ids, "ndim") and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0).to(device, non_blocking=non_blocking)
        elif hasattr(input_ids, "to"):
            input_ids = input_ids.to(device, non_blocking=non_blocking)
        else:
            input_ids = torch.as_tensor(input_ids, device=device)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

        return {"input_ids": input_ids}

    @staticmethod
    def _extract_hidden_states(output: Any) -> Any:
        """Normalize a layer output into its hidden-state tensor."""
        if hasattr(output, "hidden_states"):
            return output.hidden_states
        if isinstance(output, tuple):
            return output[0]
        return output

    @classmethod
    def _replace_hidden_states(cls, output: Any, hidden_states: Any) -> Any:
        """Write patched hidden states back into a layer output container."""
        if hasattr(output, "hidden_states"):
            output.hidden_states = hidden_states
            return output
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def _run_model(self, model_inputs: dict[str, Any]) -> Any:
        """Run a forward pass across the common torch/MLX model signatures."""
        try:
            return self.model(**model_inputs, use_cache=False)
        except TypeError:
            try:
                return self.model(**model_inputs)
            except TypeError:
                return self.model(model_inputs["input_ids"])

    async def _capture_activation_vector(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Capture a hidden-state vector from the active backend."""
        if self._backend_name() == "torch":
            return self._capture_activation_torch(prompt, layer, position)
        return await self._capture_activation_mlx(prompt, layer, position)

    async def _capture_activation_mlx(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Legacy MLX capture path."""
        import mlx.core as mx

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        captured = await self._accessor.forward_through_layers(
            input_ids,
            layers=[layer],
            capture_hidden_states=True,
        )

        h = captured[layer]
        if position == -1:
            position = h.shape[1] - 1
        activation = h[0, position, :]
        return np.array(activation.astype(mx.float32), copy=False)

    def _capture_activation_torch(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Torch-native activation capture using forward hooks."""
        import torch

        layers = self._resolve_torch_layers()
        block = layers[layer]
        captured: dict[str, Any] = {}

        def capture_hook(_module, _args, output):
            hidden = self._extract_hidden_states(output)
            pos = position if position >= 0 else hidden.shape[1] - 1
            captured["activation"] = hidden[0, pos, :].detach().to("cpu", dtype=torch.float32)

        handle = block.register_forward_hook(capture_hook)
        try:
            model_inputs = self._tokenize_torch_prompt(prompt)
            with torch.inference_mode():
                self._run_model(model_inputs)
        finally:
            handle.remove()

        activation = captured.get("activation")
        if activation is None:
            raise RuntimeError(f"Failed to capture activation for layer {layer}.")
        return activation.numpy()

    def _predict_top_token(self, prompt: str) -> tuple[str, float]:
        """Run a single forward pass and return the top next-token prediction."""
        if self._backend_name() == "torch":
            return self._predict_top_token_torch(prompt)
        return self._predict_top_token_mlx(prompt)

    def _predict_top_token_mlx(self, prompt: str) -> tuple[str, float]:
        """Legacy MLX forward path."""
        import mlx.core as mx

        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        outputs = self.model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        top_idx = int(mx.argmax(probs))
        return self.tokenizer.decode([top_idx]), float(probs[top_idx])

    def _predict_top_token_torch(self, prompt: str) -> tuple[str, float]:
        """Torch-native forward path."""
        import torch

        model_inputs = self._tokenize_torch_prompt(prompt)
        with torch.inference_mode():
            outputs = self._run_model(model_inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_idx = int(torch.argmax(probs).item())
        return self.tokenizer.decode([top_idx]), float(probs[top_idx].item())


class ActivationPatcher(_RuntimeAwareIntrospector):
    """Activation patching for causal intervention experiments.

    Example:
        >>> patcher = ActivationPatcher(model=model, tokenizer=tokenizer, config=config)
        >>> # Capture source activation
        >>> source_activation = await patcher.capture_activation("7*8=", layer=22)
        >>> # Patch into target
        >>> result = await patcher.patch_and_generate(
        ...     target_prompt="7+8=",
        ...     source_activation=source_activation,
        ...     layer=22,
        ...     blend=1.0,
        ... )
    """

    async def capture_activation(
        self,
        prompt: str,
        layer: int,
        position: int = -1,
    ) -> np.ndarray:
        """Capture activation at a specific layer and position.

        Args:
            prompt: The prompt to process
            layer: Layer index to capture
            position: Token position (-1 for last)

        Returns:
            Activation vector as numpy array
        """
        return await self._capture_activation_vector(prompt, layer, position)

    def _create_patched_layer(
        self,
        original_layer: Any,
        source_activation: mx.array,
        blend: float,
        position: int = -1,
    ) -> Any:
        """Create a wrapper layer that patches activations."""
        import mlx.core as mx

        class PatchedLayerWrapper:
            def __init__(self, layer: Any, activation: mx.array, blend: float, pos: int):
                self._wrapped = layer
                self._activation = activation
                self._blend = blend
                self._position = pos
                # Copy attributes for compatibility
                for attr in [
                    "mlp",
                    "attn",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                ]:
                    if hasattr(layer, attr):
                        setattr(self, attr, getattr(layer, attr))

            def __call__(self, h: mx.array, **kwargs) -> Any:
                result = self._wrapped(h, **kwargs)

                # Extract hidden states
                if hasattr(result, "hidden_states"):
                    hs = result.hidden_states
                elif isinstance(result, tuple):
                    hs = result[0]
                else:
                    hs = result

                # Determine position to patch
                pos = self._position if self._position >= 0 else hs.shape[1] - 1

                # Patch: blend original with source activation
                original = hs[:, pos : pos + 1, :]
                patched = (1 - self._blend) * original + self._blend * self._activation.reshape(
                    1, 1, -1
                )
                new_hs = mx.concatenate([hs[:, :pos, :], patched, hs[:, pos + 1 :, :]], axis=1)

                if hasattr(result, "hidden_states"):
                    result.hidden_states = new_hs
                    return result
                elif isinstance(result, tuple):
                    return (new_hs,) + result[1:]
                return new_hs

            def __getattr__(self, name: str) -> Any:
                return getattr(self._wrapped, name)

        return PatchedLayerWrapper(original_layer, source_activation, blend, position)

    async def patch_and_predict(
        self,
        target_prompt: str,
        source_activation: np.ndarray | mx.array,
        layer: int,
        blend: float = 1.0,
        position: int = -1,
    ) -> tuple[str, float]:
        """Patch activation and get top prediction.

        Args:
            target_prompt: Prompt to patch into
            source_activation: Activation to inject
            layer: Layer to patch at
            blend: Blend factor (0=original, 1=full replacement)
            position: Position to patch (-1 for last)

        Returns:
            Tuple of (top_token, probability)
        """
        if self._backend_name() == "torch":
            return await self._patch_and_predict_torch(
                target_prompt,
                source_activation,
                layer,
                blend=blend,
                position=position,
            )

        return await self._patch_and_predict_mlx(
            target_prompt,
            source_activation,
            layer,
            blend=blend,
            position=position,
        )

    async def _patch_and_predict_mlx(
        self,
        target_prompt: str,
        source_activation: np.ndarray | mx.array,
        layer: int,
        blend: float = 1.0,
        position: int = -1,
    ) -> tuple[str, float]:
        """Legacy MLX patch path."""
        import mlx.core as mx

        # Convert to mx.array if needed
        if isinstance(source_activation, np.ndarray):
            source_activation = mx.array(source_activation.astype(np.float32))

        # Save original layer
        original_layer = self._accessor.get_layer(layer)

        # Install patched layer
        patched_layer = self._create_patched_layer(
            original_layer, source_activation, blend, position
        )
        self._accessor.set_layer(layer, patched_layer)

        try:
            # Run forward pass
            input_ids = mx.array(self.tokenizer.encode(target_prompt))[None, :]
            outputs = self.model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Get top prediction
            probs = mx.softmax(logits[0, -1, :], axis=-1)
            top_idx = int(mx.argmax(probs))
            top_prob = float(probs[top_idx])
            top_token = self.tokenizer.decode([top_idx])

            return top_token, top_prob
        finally:
            # Restore original layer
            self._accessor.set_layer(layer, original_layer)

    async def _patch_and_predict_torch(
        self,
        target_prompt: str,
        source_activation: np.ndarray | Any,
        layer: int,
        blend: float = 1.0,
        position: int = -1,
    ) -> tuple[str, float]:
        """Torch-native activation patching using a layer-output hook."""
        import torch

        layers = self._resolve_torch_layers()
        block = layers[layer]

        if isinstance(source_activation, np.ndarray):
            source_tensor = torch.from_numpy(source_activation)
        elif isinstance(source_activation, torch.Tensor):
            source_tensor = source_activation
        else:
            source_tensor = torch.as_tensor(source_activation)

        def patch_hook(_module, _args, output):
            hidden = self._extract_hidden_states(output)
            pos = position if position >= 0 else hidden.shape[1] - 1
            source = source_tensor.to(device=hidden.device, dtype=hidden.dtype)
            source = source.reshape(1, -1).expand(hidden.shape[0], -1)
            patched_hidden = hidden.clone()
            original = patched_hidden[:, pos, :]
            patched_hidden[:, pos, :] = (1 - blend) * original + blend * source
            return self._replace_hidden_states(output, patched_hidden)

        handle = block.register_forward_hook(patch_hook)
        try:
            return self._predict_top_token_torch(target_prompt)
        finally:
            handle.remove()

    async def sweep_layers(
        self,
        target_prompt: str,
        source_prompt: str,
        layers: list[int] | None = None,
        blend: float = 1.0,
        source_answer: str | None = None,
        target_answer: str | None = None,
    ) -> PatchingResult:
        """Sweep patching across multiple layers.

        Args:
            target_prompt: Prompt to patch into
            source_prompt: Prompt to get source activation from
            layers: Layers to test (None = every 10th layer)
            blend: Blend factor
            source_answer: Expected answer from source (for transfer detection)
            target_answer: Expected answer from target

        Returns:
            Complete patching result
        """
        if layers is None:
            num_layers = self.num_layers
            layers = list(range(0, num_layers, max(1, num_layers // 10)))

        # Get baseline prediction
        baseline_token, baseline_prob = self._predict_top_token(target_prompt)

        # Capture source activations at all layers
        source_captured: dict[int, Any]
        if self._backend_name() == "torch":
            source_captured = {
                layer_idx: await self.capture_activation(source_prompt, layer_idx)
                for layer_idx in layers
            }
        else:
            import mlx.core as mx

            source_ids = mx.array(self.tokenizer.encode(source_prompt))[None, :]
            source_captured = await self._accessor.forward_through_layers(source_ids, layers=layers)

        # Test each layer
        layer_results = []
        for layer in layers:
            if self._backend_name() == "torch":
                source_activation = source_captured[layer]
            else:
                source_activation = source_captured[layer][0, -1, :]
            top_token, top_prob = await self.patch_and_predict(
                target_prompt,
                source_activation,
                layer,
                blend,
            )

            # Determine effect
            if top_token == baseline_token:
                effect = PatchEffect.NO_CHANGE
            elif source_answer and source_answer.startswith(top_token.strip()):
                effect = PatchEffect.TRANSFERRED
            elif target_answer and target_answer.startswith(top_token.strip()):
                effect = PatchEffect.STILL_TARGET
            else:
                effect = PatchEffect.CHANGED

            layer_results.append(
                PatchingLayerResult(
                    layer=layer,
                    top_token=top_token,
                    top_prob=top_prob,
                    baseline_token=baseline_token,
                    baseline_prob=baseline_prob,
                    effect=effect,
                )
            )

        return PatchingResult(
            model_id=self._model_id(),
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            source_answer=source_answer,
            target_answer=target_answer,
            blend=blend,
            layers=layers,
            baseline_token=baseline_token,
            baseline_prob=baseline_prob,
            layer_results=layer_results,
        )


class CommutativityAnalyzer(_RuntimeAwareIntrospector):
    """Analyze whether representations respect commutativity (A*B = B*A).

    Example:
        >>> analyzer = CommutativityAnalyzer(model=model, tokenizer=tokenizer, config=config)
        >>> result = await analyzer.analyze(layer=22)
        >>> print(f"Mean similarity: {result.mean_similarity:.4f}")
    """

    async def get_activation(self, prompt: str, layer: int) -> np.ndarray:
        """Get last-token hidden state for a prompt at a given layer."""
        return await self._capture_activation_vector(prompt, layer)

    async def analyze(
        self,
        layer: int | None = None,
        pairs: list[tuple[str, str]] | None = None,
    ) -> CommutativityResult:
        """Analyze commutativity at a specific layer.

        Args:
            layer: Layer to analyze (default: 60% through network)
            pairs: Explicit pairs to test (default: all single-digit multiplication)

        Returns:
            CommutativityResult with similarity statistics
        """
        from .models.patching import CommutativityPair, CommutativityResult

        if layer is None:
            layer = int(self.num_layers * 0.6)

        if pairs is None:
            pairs = []
            for a in range(2, 10):
                for b in range(a + 1, 10):
                    pairs.append((f"{a}*{b}=", f"{b}*{a}="))

        similarities = []
        pair_results = []

        for prompt_a, prompt_b in pairs:
            h_a = await self.get_activation(prompt_a, layer)
            h_b = await self.get_activation(prompt_b, layer)

            # Cosine similarity
            dot = np.dot(h_a, h_b)
            norm_a = np.linalg.norm(h_a)
            norm_b = np.linalg.norm(h_b)
            sim = float(dot / (norm_a * norm_b + 1e-8))

            similarities.append(sim)
            pair_results.append(
                CommutativityPair(
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    similarity=sim,
                )
            )

        return CommutativityResult(
            model_id=self._model_id(),
            layer=layer,
            num_pairs=len(pairs),
            mean_similarity=float(np.mean(similarities)),
            std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            pairs=pair_results,
        )
