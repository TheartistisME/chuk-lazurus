"""
Hook infrastructure for capturing intermediate model states.

MLX doesn't have PyTorch's register_forward_hook, so we use a wrapper
approach that intercepts layer outputs during forward pass.

Example:
    >>> from chuk_lazarus.introspection import ModelHooks, CaptureConfig, LayerSelection
    >>>
    >>> hooks = ModelHooks(model)
    >>> hooks.configure(CaptureConfig(
    ...     layers=[0, 4, 8, 12],
    ...     capture_attention_weights=True,
    ... ))
    >>>
    >>> # Run inference
    >>> output = hooks.forward(input_ids)
    >>>
    >>> # Access captured states
    >>> layer_4_hidden = hooks.state.hidden_states[4]
    >>> layer_4_attn = hooks.state.attention_weights[4]

Lazy-import contract:
- Module-level import must NOT pull ``mlx`` (EWS-0.3b / BACKEND_IN_SCOPE).
- Pydantic models (``CaptureConfig``, ``CapturedState``) and enums
  (``LayerSelection``, ``PositionSelection``) are resolvable without mlx.
- ``ModelHooks`` subclasses nothing and only touches ``mlx.core`` / ``mlx.nn``
  inside method bodies. The class is built eagerly (no ``nn.Module`` base),
  but any mlx usage is deferred until forward/layer access.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ._backend_dispatch import register_hook, to_backend_tensor

if TYPE_CHECKING:  # pragma: no cover - type-checker only; never executed at runtime
    import mlx.core as mx
    import mlx.nn as nn


def _mx():
    """Lazy accessor for ``mlx.core``."""
    import mlx.core as mx  # noqa: PLC0415

    return mx


def _nn():
    """Lazy accessor for ``mlx.nn``."""
    import mlx.nn as nn  # noqa: PLC0415

    return nn


def _get_backend():
    """Resolve the active runtime backend lazily."""
    from chuk_lazarus.models_v2.core import backend as backend_mod

    current = getattr(backend_mod, "_current_backend", None)
    if current is not None:
        return current
    return backend_mod.get_backend()


class LayerSelection(str, Enum):
    """Selection strategy for which layers to capture."""

    ALL = "all"
    """Capture all layers."""


class PositionSelection(str, Enum):
    """Selection strategy for which sequence positions to capture."""

    ALL = "all"
    """Capture all positions."""

    LAST = "last"
    """Capture only the last position (memory efficient)."""


class CaptureConfig(BaseModel):
    """Configuration for what to capture during forward pass."""

    layers: list[int] | LayerSelection = Field(
        default=LayerSelection.ALL,
        description="Which layers to capture. Use LayerSelection.ALL for all layers or a list of indices.",
    )

    capture_hidden_states: bool = Field(
        default=True,
        description="Capture hidden states after each layer.",
    )

    capture_attention_weights: bool = Field(
        default=False,
        description="Capture attention weight matrices. Memory intensive.",
    )

    capture_attention_output: bool = Field(
        default=False,
        description="Capture attention output before residual add.",
    )

    capture_ffn_output: bool = Field(
        default=False,
        description="Capture FFN output before residual add.",
    )

    capture_pre_norm: bool = Field(
        default=False,
        description="Capture states before layer norm (raw residual stream).",
    )

    positions: list[int] | PositionSelection = Field(
        default=PositionSelection.LAST,
        description="Which sequence positions to capture.",
    )

    detach: bool = Field(
        default=True,
        description="Whether to detach captured tensors from computation graph.",
    )

    model_config = ConfigDict(use_enum_values=False)


class CapturedState(BaseModel):
    """Container for captured intermediate states."""

    hidden_states: dict[int, Any] = Field(
        default_factory=dict,
        description="Hidden states after each layer. Shape: [batch, seq, hidden]",
    )

    attention_weights: dict[int, Any] = Field(
        default_factory=dict,
        description="Attention weights per layer. Shape: [batch, heads, seq, seq]",
    )

    attention_outputs: dict[int, Any] = Field(
        default_factory=dict,
        description="Attention outputs before residual. Shape: [batch, seq, hidden]",
    )

    ffn_outputs: dict[int, Any] = Field(
        default_factory=dict,
        description="FFN outputs before residual. Shape: [batch, seq, hidden]",
    )

    pre_norm_states: dict[int, Any] = Field(
        default_factory=dict,
        description="States before layer norm. Shape: [batch, seq, hidden]",
    )

    input_ids: Any | None = Field(
        default=None,
        description="Original input token IDs.",
    )

    embeddings: Any | None = Field(
        default=None,
        description="Token embeddings before first layer.",
    )

    final_hidden: Any | None = Field(
        default=None,
        description="Final hidden state after all layers.",
    )

    logits: Any | None = Field(
        default=None,
        description="Output logits if computed.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def clear(self) -> None:
        """Clear all captured states to free memory."""
        self.hidden_states.clear()
        self.attention_weights.clear()
        self.attention_outputs.clear()
        self.ffn_outputs.clear()
        self.pre_norm_states.clear()
        self.input_ids = None
        self.embeddings = None
        self.final_hidden = None
        self.logits = None

    @property
    def num_layers_captured(self) -> int:
        """Number of layers with captured hidden states."""
        return len(self.hidden_states)

    @property
    def captured_layers(self) -> list[int]:
        """Sorted list of layer indices that were captured."""
        return sorted(self.hidden_states.keys())

    def get_hidden_at_position(self, layer_idx: int, position: int = -1) -> Any | None:
        """
        Get hidden state for a specific layer and sequence position.

        Args:
            layer_idx: Layer index
            position: Sequence position (-1 for last)

        Returns:
            Hidden state vector of shape [hidden_size] or None if not captured
        """
        if layer_idx not in self.hidden_states:
            return None
        hidden = self.hidden_states[layer_idx]
        # Handle both [seq, hidden] and [batch, seq, hidden]
        if hidden.ndim == 2:
            return hidden[position]
        return hidden[0, position]


class ModelHooks:
    """
    Hook manager for capturing intermediate model states.

    This class wraps a model's forward pass to capture intermediate
    activations, attention weights, and other internal states without
    modifying the model code.

    Example:
        >>> hooks = ModelHooks(model)
        >>> hooks.configure(CaptureConfig(
        ...     layers=[0, 4, 8],
        ...     capture_attention_weights=True,
        ... ))
        >>>
        >>> # Wrapped forward pass captures states
        >>> logits = hooks.forward(input_ids)
        >>>
        >>> # Access captured data
        >>> print(hooks.state.hidden_states[4].shape)
        >>> print(hooks.state.attention_weights[4].shape)
    """

    def __init__(
        self,
        model: Any,
        embedding_scale: float | None = None,
        model_config: Any | None = None,
    ):
        """
        Initialize hooks for a model.

        Args:
            model: The model to hook into. Should have a .model.layers attribute.
            embedding_scale: Optional scale factor for embeddings.
                Some models (e.g., Gemma) scale embeddings by sqrt(hidden_size).
                If not provided, will try to detect from model_config or model properties.
            model_config: Optional model config (e.g., GemmaConfig, LlamaConfig).
                Used to auto-detect embedding_scale and other model properties.
        """
        self.model = model
        self.model_config = model_config
        self.config = CaptureConfig()
        self.state = CapturedState()
        self._original_forwards: dict[int, Callable] = {}
        self._hooked = False
        self._embedding_scale_override = embedding_scale

    def _detach_tensor(self, tensor: Any, backend: Any) -> Any:
        """Detach a captured tensor when requested by config."""
        if not self.config.detach:
            return tensor
        return backend.stop_gradient(tensor)

    def _capture_tensor(self, tensor: Any, backend: Any) -> Any:
        """Slice and optionally detach a tensor before storing it."""
        captured = self._maybe_slice_positions(tensor)
        return self._detach_tensor(captured, backend)

    def _extract_hidden_state(self, layer_out: Any) -> Any | None:
        """Extract hidden states from common layer output shapes."""
        if hasattr(layer_out, "hidden_states") and layer_out.hidden_states is not None:
            return layer_out.hidden_states
        if hasattr(layer_out, "last_hidden_state") and layer_out.last_hidden_state is not None:
            return layer_out.last_hidden_state
        if isinstance(layer_out, tuple | list):
            return layer_out[0] if layer_out else None
        return layer_out

    def _extract_attention_weights(self, layer_out: Any) -> Any | None:
        """Extract attention weights when a layer exposes them."""
        if hasattr(layer_out, "attention_weights") and layer_out.attention_weights is not None:
            return layer_out.attention_weights
        attentions = getattr(layer_out, "attentions", None)
        if attentions:
            return attentions[0]
        if isinstance(layer_out, tuple | list) and len(layer_out) > 1:
            candidate = layer_out[1]
            if hasattr(candidate, "ndim"):
                return candidate
        return None

    def _extract_logits(self, model_out: Any) -> Any | None:
        """Extract logits from common model output shapes."""
        if hasattr(model_out, "logits") and model_out.logits is not None:
            return model_out.logits
        if isinstance(model_out, tuple | list):
            return model_out[0] if model_out else None
        return model_out

    def _call_model(self, input_ids: Any, **model_kwargs: Any) -> Any:
        """Call the wrapped model across the common positional/keyword surfaces."""

        def _is_signature_mismatch(exc: TypeError) -> bool:
            msg = str(exc)
            return any(
                marker in msg
                for marker in (
                    "unexpected keyword argument",
                    "required positional argument",
                    "positional argument",
                )
            )

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if model_kwargs:
            attempts.extend(
                [
                    ((), {"input_ids": input_ids, **model_kwargs}),
                    ((input_ids,), model_kwargs),
                ]
            )
        attempts.extend(
            [
                ((), {"input_ids": input_ids}),
                ((input_ids,), {}),
            ]
        )

        last_error: TypeError | None = None
        for args, kwargs in attempts:
            try:
                return self.model(*args, **kwargs)
            except TypeError as exc:
                if not _is_signature_mismatch(exc):
                    raise
                last_error = exc

        if last_error is not None:  # pragma: no cover - defensive fallback
            raise last_error
        raise RuntimeError("Model call failed without raising a TypeError")

    def _forward_torch(
        self,
        input_ids: Any,
        return_logits: bool = True,
    ) -> Any | None:
        """Torch runtime path: capture states via forward hooks and run the real model."""
        backend = _get_backend()

        self.state.clear()

        input_ids = to_backend_tensor(input_ids, backend=backend)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        self.state.input_ids = input_ids

        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()

        if embed is None:
            raise ValueError("Cannot find embedding layer")

        embeddings = embed(input_ids)
        embed_scale = self._get_embedding_scale()
        if embed_scale is not None:
            embeddings = embeddings * embed_scale
        self.state.embeddings = embeddings

        last_hidden: dict[str, Any] = {}
        detachers: list[Callable[[], None]] = []
        last_layer_idx = len(layers) - 1

        def _make_hook(layer_idx: int) -> Callable[..., Any]:
            def _capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
                hidden = self._extract_hidden_state(output)
                if hidden is None:
                    return

                last_hidden["value"] = hidden
                if not self._should_capture_layer(layer_idx):
                    return

                if self.config.capture_pre_norm and inputs:
                    self.state.pre_norm_states[layer_idx] = self._capture_tensor(inputs[0], backend)

                if self.config.capture_hidden_states:
                    self.state.hidden_states[layer_idx] = self._capture_tensor(hidden, backend)

                if self.config.capture_attention_weights:
                    attention = self._extract_attention_weights(output)
                    if attention is not None:
                        self.state.attention_weights[layer_idx] = self._capture_tensor(
                            attention, backend
                        )

            return _capture

        try:
            for layer_idx, layer in enumerate(layers):
                if self._should_capture_layer(layer_idx) or layer_idx == last_layer_idx:
                    detachers.append(register_hook(backend, layer, _make_hook(layer_idx)))

            model_kwargs: dict[str, Any] = {}
            if self.config.capture_attention_weights:
                model_kwargs["output_attentions"] = True

            model_out = self._call_model(input_ids, **model_kwargs)
        finally:
            for detach in reversed(detachers):
                detach()

        final_hidden = last_hidden.get("value")
        if final_hidden is not None and final_norm is not None:
            final_hidden = final_norm(final_hidden)
        self.state.final_hidden = final_hidden

        if not return_logits:
            return None

        logits = self._extract_logits(model_out)
        if logits is not None:
            self.state.logits = logits
            return logits

        if final_hidden is None:
            return None

        lm_head = self._get_lm_head()
        if lm_head is None:
            return None

        head_out = lm_head(final_hidden)
        logits = head_out.logits if hasattr(head_out, "logits") else head_out
        self.state.logits = logits
        return logits

    def configure(self, config: CaptureConfig) -> ModelHooks:
        """
        Configure what to capture.

        Args:
            config: Capture configuration

        Returns:
            Self for chaining
        """
        self.config = config
        return self

    def capture_layers(
        self,
        layers: list[int] | LayerSelection = LayerSelection.ALL,
        capture_attention: bool = False,
    ) -> ModelHooks:
        """
        Convenience method to configure layer capture.

        Args:
            layers: Which layers to capture, or LayerSelection.ALL
            capture_attention: Whether to capture attention weights

        Returns:
            Self for chaining
        """
        self.config.layers = layers
        self.config.capture_attention_weights = capture_attention
        return self

    def _should_capture_layer(self, layer_idx: int) -> bool:
        """Check if a layer should be captured based on config."""
        if self.config.layers == LayerSelection.ALL:
            return True
        if isinstance(self.config.layers, list):
            return layer_idx in self.config.layers
        return False

    def _maybe_slice_positions(self, tensor: Any) -> Any:
        """Slice tensor to only keep configured positions."""
        if self.config.positions == PositionSelection.ALL:
            return tensor
        if self.config.positions == PositionSelection.LAST:
            # Keep only last position
            if tensor.ndim == 2:  # [seq, hidden]
                return tensor[-1:]
            elif tensor.ndim == 3:  # [batch, seq, hidden]
                return tensor[:, -1:]
            elif tensor.ndim == 4:  # [batch, heads, seq, seq] for attention
                return tensor[:, :, -1:, :]
        # Explicit position list
        if isinstance(self.config.positions, list):
            positions = self.config.positions
            if tensor.ndim == 2:
                return tensor[positions]
            elif tensor.ndim == 3:
                return tensor[:, positions, :]
            elif tensor.ndim == 4:
                return tensor[:, :, positions, :]
        return tensor

    def _get_layers(self) -> list[Any]:
        """Get the list of transformer layers from the model."""
        # Try common attribute names
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "layers"):
                return list(inner.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "h"):
                return list(self.model.transformer.h)
            if hasattr(self.model.transformer, "layers"):
                return list(self.model.transformer.layers)
        raise ValueError(
            "Cannot find layers in model. Expected model.model.layers, "
            "model.layers, or model.transformer.h"
        )

    def _get_embed_tokens(self) -> Any | None:
        """Get the embedding layer from the model."""
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "embed_tokens"):
                return inner.embed_tokens
        if hasattr(self.model, "embed_tokens"):
            return self.model.embed_tokens
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                return self.model.transformer.wte
        return None

    def _get_final_norm(self) -> Any | None:
        """Get the final layer norm from the model."""
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "norm"):
                return inner.norm
        if hasattr(self.model, "norm"):
            return self.model.norm
        return None

    def _get_lm_head(self) -> Callable[[Any], Any] | None:
        """Get the LM head or tied embedding projection from the model."""
        # Check for explicit lm_head first
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            return self.model.lm_head

        # Check for tied embeddings (explicit flag)
        if hasattr(self.model, "tie_word_embeddings") and self.model.tie_word_embeddings:
            embed = self._get_embed_tokens()
            if embed is not None and hasattr(embed, "as_linear"):
                return embed.as_linear

        # Fallback: if no lm_head, try to use embedding as_linear (common in mlx-lm models)
        # This handles models that use tied embeddings without setting the flag
        embed = self._get_embed_tokens()
        if embed is not None and hasattr(embed, "as_linear"):
            return embed.as_linear

        return None

    def _get_embedding_scale(self) -> float | None:
        """
        Get embedding scale factor if model uses one.

        Checks in order:
        1. Explicit override provided at init
        2. model_config.embedding_scale (from model families)
        3. Model backbone embedding_scale property
        4. Attached _embedding_scale_for_hooks (for external models)

        Note: External models (e.g., from mlx_lm) may not expose embedding_scale,
        which can cause incorrect logit lens results for models like Gemma.
        Use the model_config or embedding_scale parameter when creating ModelHooks.
        """
        # Use override if provided
        if self._embedding_scale_override is not None:
            return self._embedding_scale_override

        # Check model_config (from model families registry)
        if self.model_config is not None and hasattr(self.model_config, "embedding_scale"):
            scale = self.model_config.embedding_scale
            if scale is not None:
                return scale

        # Check for property on the model backbone (our native models)
        inner = getattr(self.model, "model", self.model)
        if hasattr(inner, "embedding_scale"):
            return inner.embedding_scale

        # Check for attached scale (set by analyzer for quantized models)
        if hasattr(self.model, "_embedding_scale_for_hooks"):
            return self.model._embedding_scale_for_hooks

        return None

    def forward(
        self,
        input_ids: Any,
        cache: Any | None = None,
        return_logits: bool = True,
    ) -> Any | None:
        """
        Run forward pass while capturing intermediate states.

        This manually unrolls the forward pass to capture states at each layer.

        Args:
            input_ids: Input token IDs [batch, seq] or [seq]
            cache: Optional KV cache
            return_logits: Whether to compute and return logits

        Returns:
            Logits if return_logits=True, else None
        """
        backend = _get_backend()
        if backend.name == "torch":
            return self._forward_torch(input_ids, return_logits=return_logits)

        nn = _nn()

        self.state.clear()

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        self.state.input_ids = input_ids

        # Get model components
        layers = self._get_layers()
        embed = self._get_embed_tokens()
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()

        # Embeddings
        if embed is not None:
            # Call the embedding layer - works for both nn.Embedding and wrapped embeddings
            h = embed(input_ids)
            # Apply embedding scale if model uses one (e.g., Gemma)
            embed_scale = self._get_embedding_scale()
            if embed_scale is not None:
                h = h * embed_scale
            self.state.embeddings = h
        else:
            raise ValueError("Cannot find embedding layer")

        # Create causal mask (critical for correct attention)
        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        # Process each layer
        for layer_idx, layer in enumerate(layers):
            should_capture = self._should_capture_layer(layer_idx)

            if should_capture and self.config.capture_pre_norm:
                self.state.pre_norm_states[layer_idx] = self._maybe_slice_positions(h)

            # Run layer forward
            # Layers can return:
            # - BlockOutput (our models) with .hidden_states
            # - Tuple (hidden_states,) or (hidden_states, cache)
            # - Just the tensor directly
            # Try with mask first (for attention layers), fall back without (for SSM layers)
            try:
                layer_out = layer(h, mask=mask, cache=cache)
            except TypeError:
                # Layer doesn't accept mask (e.g., Mamba SSM layers)
                layer_out = layer(h, cache=cache)

            # Extract hidden state from layer output
            if hasattr(layer_out, "hidden_states"):
                # BlockOutput from our models
                h = layer_out.hidden_states
                # Check for attention weights
                if (
                    should_capture
                    and self.config.capture_attention_weights
                    and hasattr(layer_out, "attention_weights")
                    and layer_out.attention_weights is not None
                ):
                    self.state.attention_weights[layer_idx] = self._maybe_slice_positions(
                        layer_out.attention_weights
                    )
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
                # Check for attention weights in output
                if should_capture and self.config.capture_attention_weights and len(layer_out) > 2:
                    self.state.attention_weights[layer_idx] = self._maybe_slice_positions(
                        layer_out[2]
                    )
            else:
                h = layer_out

            # Capture hidden state after layer
            if should_capture and self.config.capture_hidden_states:
                captured = self._maybe_slice_positions(h)
                self.state.hidden_states[layer_idx] = self._detach_tensor(captured, backend)

        # Final norm
        if final_norm is not None:
            h = final_norm(h)

        self.state.final_hidden = h

        # LM head
        if return_logits and lm_head is not None:
            head_out = lm_head(h)
            # Handle HeadOutput from our models
            if hasattr(head_out, "logits"):
                logits = head_out.logits
            else:
                logits = head_out
            self.state.logits = logits
            return logits

        return None

    def forward_to_layer(
        self,
        input_ids: Any,
        target_layer: int,
    ) -> Any:
        """
        Run forward pass up to a specific layer and return hidden state.

        Useful for probing experiments where you only need activations
        up to a certain depth.

        Args:
            input_ids: Input token IDs
            target_layer: Stop after this layer (0-indexed)

        Returns:
            Hidden state after target_layer
        """
        backend = _get_backend()
        if backend.name == "torch":
            input_ids = to_backend_tensor(input_ids, backend=backend)
            if input_ids.ndim == 1:
                input_ids = input_ids[None, :]

            layers = self._get_layers()
            if target_layer >= len(layers):
                raise ValueError(f"Layer {target_layer} out of range for {len(layers)} layers")

            target_hidden: dict[str, Any] = {}
            detach = register_hook(
                backend,
                layers[target_layer],
                lambda _module, _inputs, output: target_hidden.setdefault(
                    "value",
                    self._extract_hidden_state(output),
                ),
            )
            try:
                self._call_model(input_ids)
            finally:
                detach()

            hidden = target_hidden.get("value")
            if hidden is None:
                raise ValueError(f"Failed to capture hidden state for layer {target_layer}")
            return hidden

        # Ensure batch dimension
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        layers = self._get_layers()
        embed = self._get_embed_tokens()

        if embed is not None:
            h = embed(input_ids)
        else:
            raise ValueError("Cannot find embedding layer")

        for layer_idx, layer in enumerate(layers):
            if layer_idx > target_layer:
                break

            layer_out = layer(h)
            # Extract hidden state from layer output
            if hasattr(layer_out, "hidden_states"):
                h = layer_out.hidden_states
            elif isinstance(layer_out, tuple):
                h = layer_out[0]
            else:
                h = layer_out

        return h

    def get_layer_logits(
        self,
        layer_idx: int,
        normalize: bool = True,
    ) -> Any | None:
        """
        Project hidden state from a specific layer to vocabulary logits.

        This is the core of "logit lens" - seeing what the model would
        predict if we stopped at an intermediate layer.

        Args:
            layer_idx: Which layer's hidden state to project
            normalize: Whether to apply final layer norm first

        Returns:
            Logits of shape [batch, seq, vocab] or None if layer not captured
        """
        if layer_idx not in self.state.hidden_states:
            return None

        h = self.state.hidden_states[layer_idx]

        # Apply final norm if requested
        if normalize:
            final_norm = self._get_final_norm()
            if final_norm is not None:
                h = final_norm(h)

        # Project to vocab
        lm_head = self._get_lm_head()
        if lm_head is not None:
            head_out = lm_head(h)
            # Handle HeadOutput from our models
            if hasattr(head_out, "logits"):
                return head_out.logits
            return head_out

        return None

    def __repr__(self) -> str:
        """String representation."""
        try:
            num_layers = len(self._get_layers()) if hasattr(self, "model") else "?"
        except Exception:
            num_layers = "?"
        return (
            f"ModelHooks(layers={self.config.layers}, "
            f"captured={self.state.num_layers_captured}/{num_layers})"
        )
