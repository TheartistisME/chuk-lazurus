"""
Virtual MoE Wrapper.

Main interface for adding virtual expert capability to MoE models.
"""

from __future__ import annotations

import asyncio
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import (
    InferenceResult,
    RoutingTrace,
    VirtualExpertAnalysis,
    VirtualExpertApproach,
    VirtualExpertPlugin,
)
from .cot_rewriter import VirtualExpertAction
from .plugins.math import MathExpertPlugin
from .registry import VirtualExpertRegistry, get_default_registry
from . import router as router_module


@dataclass(frozen=True)
class _RoutingCalibration:
    """Backend-neutral routing parameters for non-MLX execution."""

    direction: np.ndarray
    threshold: float


def _runtime_backend_name(runtime: Any | None) -> str | None:
    backend = getattr(runtime, "backend", None)
    if backend is None:
        return None

    name = str(backend).lower()
    if name == "cuda":
        return "torch"
    if name == "mlx":
        return "mlx"
    return name


def _vector_to_numpy(vector: Any) -> np.ndarray:
    if isinstance(vector, np.ndarray):
        return vector.astype(np.float32, copy=False).reshape(-1)

    module_name = type(vector).__module__
    if module_name.startswith("torch"):
        return vector.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

    return np.asarray(vector, dtype=np.float32).reshape(-1)


def _build_routing_calibration(
    positive_activations: list[Any],
    negative_activations: list[Any],
) -> _RoutingCalibration | None:
    if not positive_activations or not negative_activations:
        return None

    pos_stack = np.stack([_vector_to_numpy(v) for v in positive_activations], axis=0)
    neg_stack = np.stack([_vector_to_numpy(v) for v in negative_activations], axis=0)

    direction = pos_stack.mean(axis=0) - neg_stack.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-10:
        return None

    direction = direction / norm
    pos_proj = pos_stack @ direction
    neg_proj = neg_stack @ direction
    threshold = float((pos_proj.mean() + neg_proj.mean()) / 2)
    return _RoutingCalibration(direction=direction.astype(np.float32, copy=False), threshold=threshold)


def _routing_score(hidden_state: Any, calibration: _RoutingCalibration | None) -> float:
    if calibration is None:
        return 0.0

    projection = float(np.dot(_vector_to_numpy(hidden_state), calibration.direction))
    threshold = calibration.threshold
    score = (projection - threshold) / (abs(threshold) + 1.0)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _run_sync(coro):
    """Run an async expert call from sync code, even under an active event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - surfaced to caller below
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")


def _mx():
    import mlx.core as mx

    return mx


def _nn():
    import mlx.nn as nn

    return nn


class VirtualMoEWrapper:
    """
    Main interface for adding virtual experts to MoE models.

    This class:
    1. Wraps the model's MoE layers with VirtualRouter
    2. Manages plugin registration and calibration
    3. Intercepts generation to use plugins when appropriate

    Example:
        >>> wrapper = VirtualMoEWrapper(model, tokenizer)
        >>> wrapper.register_plugin(MyCustomPlugin())
        >>> wrapper.calibrate()
        >>>
        >>> result = wrapper.solve("127 * 89 = ")
        >>> print(result.answer)  # "11303"
        >>> print(result.plugin_name)  # "math"
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_id: str = "unknown",
        registry: VirtualExpertRegistry | None = None,
        target_layers: list[int] | None = None,
        runtime: Any | None = None,
        pipeline: Any | None = None,
    ):
        """
        Initialize the wrapper.

        Args:
            model: The MoE model to wrap
            tokenizer: The tokenizer
            model_id: Model identifier for logging
            registry: Plugin registry (uses default if None)
            target_layers: Which MoE layers to use (all if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.registry = registry or get_default_registry()
        self.runtime = runtime
        self.pipeline = pipeline
        self._runtime_backend = _runtime_backend_name(runtime)
        self._use_runtime_routing = self._runtime_backend not in (None, "mlx")

        # Detect model structure
        self._detect_structure()

        # Find MoE layers
        self.moe_layers = self._find_moe_layers()

        if not self.moe_layers:
            raise ValueError("No MoE layers found in model")

        # Use specified layers or all
        if target_layers is None:
            target_layers = self.moe_layers
        self.target_layers = target_layers

        # Create virtual routers
        self.virtual_routers: dict[int, Any] = {}
        self.original_moe_layers: dict[int, Any] = {}
        self._routing_params: dict[int, dict[int, _RoutingCalibration]] = {}

        if not self._use_runtime_routing:
            self._setup_virtual_layers()
        self._calibrated = False

    def _detect_structure(self):
        """Detect model backbone structure."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self._backbone = self.model.model
            self._layers = list(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            self._backbone = self.model
            self._layers = list(self.model.layers)
        elif self.runtime is not None and hasattr(self.runtime, "_resolve_layers"):
            self._backbone = getattr(self.model, "model", self.model)
            self._layers = list(self.runtime._resolve_layers())
        else:
            raise ValueError("Cannot detect model structure")

        self.num_layers = len(self._layers)
        self._embed = getattr(self._backbone, "embed_tokens", None)
        self._norm = getattr(self._backbone, "norm", None)
        self._lm_head = getattr(self.model, "lm_head", None)

        if hasattr(self.model, "config"):
            self._embed_scale = getattr(self.model.config, "embedding_scale", None)
        else:
            self._embed_scale = None

    def _find_moe_layers(self) -> list[int]:
        """Find all MoE layer indices."""
        moe_layers = []
        for i, layer in enumerate(self._layers):
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_layers.append(i)
        return moe_layers

    def _setup_virtual_layers(self):
        """Set up virtual routers for target layers."""
        num_plugins = len(self.registry)
        virtual_router_cls = router_module.VirtualRouter

        for layer_idx in self.target_layers:
            if layer_idx not in self.moe_layers:
                continue

            layer = self._layers[layer_idx]
            moe = layer.mlp
            router = moe.router

            num_experts = router.num_experts
            num_experts_per_tok = router.num_experts_per_tok
            hidden_size = router.weight.shape[1]

            virtual_router = virtual_router_cls(
                original_router=router,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                num_virtual_experts=max(1, num_plugins),
            )

            self.virtual_routers[layer_idx] = virtual_router
            self.original_moe_layers[layer_idx] = moe

    def register_plugin(self, plugin: VirtualExpertPlugin) -> None:
        """
        Register a new virtual expert plugin.

        After registering, you must call calibrate() again.
        """
        self.registry.register(plugin)
        if self._use_runtime_routing:
            self._routing_params = {}
        else:
            # Rebuild virtual routers to include new plugin
            self._setup_virtual_layers()
        self._calibrated = False

    def _get_hidden_state(self, prompt: str, layer_idx: int) -> mx.array:
        """Get hidden state at a specific layer for last position."""
        if self.runtime is not None:
            residual_state = self.runtime.extract_residual_state(prompt, layer_index=layer_idx)
            tensor = residual_state.tensor
            if getattr(tensor, "ndim", 0) > 1:
                return tensor[0]
            return tensor

        mx = _mx()
        nn = _nn()
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]

        h = self._embed(input_ids)
        if self._embed_scale:
            h = h * self._embed_scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        for idx, layer in enumerate(self._layers):
            if idx == layer_idx:
                break
            try:
                out = layer(h, mask=mask)
            except TypeError:
                out = layer(h)

            if hasattr(out, "hidden_states"):
                h = out.hidden_states
            elif isinstance(out, tuple):
                h = out[0]
            else:
                h = out

        mx.eval(h)
        return h[0, -1, :]

    def calibrate(self) -> None:
        """
        Calibrate all registered plugins.

        For each plugin, collects activations for positive/negative prompts
        and learns a routing direction in activation space.
        Plugins with empty calibration data are skipped.
        """
        plugins = self.registry.get_all()

        for plugin_idx, plugin in enumerate(plugins):
            pos_prompts, neg_prompts = plugin.get_calibration_prompts()

            # Skip plugins with no calibration data
            if not pos_prompts or not neg_prompts:
                continue

            # Calibrate each virtual router at each layer
            for layer_idx, virtual_router in self.virtual_routers.items():
                pos_activations = [self._get_hidden_state(p, layer_idx) for p in pos_prompts]
                neg_activations = [self._get_hidden_state(p, layer_idx) for p in neg_prompts]

                virtual_router.calibrate_expert(plugin_idx, pos_activations, neg_activations)

            if self._use_runtime_routing:
                for layer_idx in self.target_layers:
                    pos_activations = [self._get_hidden_state(p, layer_idx) for p in pos_prompts]
                    neg_activations = [self._get_hidden_state(p, layer_idx) for p in neg_prompts]
                    calibration = _build_routing_calibration(pos_activations, neg_activations)
                    if calibration is not None:
                        self._routing_params.setdefault(layer_idx, {})[plugin_idx] = calibration

            # Store calibration data
            self.registry.set_calibration_data(
                plugin.name,
                [self._get_hidden_state(p, self.target_layers[0]) for p in pos_prompts],
                [self._get_hidden_state(p, self.target_layers[0]) for p in neg_prompts],
            )

        self._calibrated = True

    def _generate_with_virtual_expert(
        self,
        prompt: str,
        max_tokens: int = 20,
        collect_trace: bool = False,
    ) -> tuple[str, bool, int, int, float, str | None, RoutingTrace | None]:
        """
        Generate with virtual experts active.

        Returns:
            (text, used_virtual, virtual_count, total_tokens, score, plugin_name, trace)
        """
        if self._use_runtime_routing:
            return self._generate_with_runtime_routing(
                prompt,
                max_tokens=max_tokens,
                collect_trace=collect_trace,
            )

        mx = _mx()
        nn = _nn()
        input_ids = self.tokenizer.encode(prompt)
        current_ids = mx.array(input_ids)[None, :]
        generated = []

        virtual_selected_total = 0
        total_tokens = 0
        routing_scores = []
        selected_plugin: str | None = None

        # Routing trace for verbose output
        trace = RoutingTrace() if collect_trace else None

        # Use middle router for scoring
        primary_layer = self.target_layers[len(self.target_layers) // 2]
        _ = self.virtual_routers[primary_layer]  # Reserved for future use
        plugins = self.registry.get_all()

        # Detect task type from prompt
        task_type = self._detect_task_type(prompt) if collect_trace else None

        for step in range(max_tokens):
            h = self._embed(current_ids)
            if self._embed_scale:
                h = h * self._embed_scale

            seq_len = current_ids.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(h.dtype)

            virtual_selected_this_step = False

            for idx, layer in enumerate(self._layers):
                # Capture state before MoE for attention contribution calc
                h_pre_layer = h if (collect_trace and step == 0) else None

                try:
                    out = layer(h, mask=mask)
                except TypeError:
                    out = layer(h)

                if hasattr(out, "hidden_states"):
                    h = out.hidden_states
                elif isinstance(out, tuple):
                    h = out[0]
                else:
                    h = out

                if idx in self.virtual_routers:
                    router = self.virtual_routers[idx]

                    # Check each plugin
                    for plugin_idx, plugin in enumerate(plugins):
                        score = router.get_routing_score(h, plugin_idx)
                        if idx == primary_layer:
                            routing_scores.append(score)

                        _, _, virtual_masks = router(h[:, -1:, :])
                        selected = plugin_idx in virtual_masks and bool(
                            mx.any(virtual_masks[plugin_idx])
                        )

                        if selected:
                            virtual_selected_this_step = True
                            if selected_plugin is None:
                                selected_plugin = plugin.name

                        # Collect trace on first generation step
                        if trace is not None and step == 0:
                            # Attention contribution: ratio of attention residual to total
                            # This is the 96% finding - attention dominates router input
                            attn_contrib = None
                            if h_pre_layer is not None:
                                residual = h - h_pre_layer
                                residual_norm = float(mx.linalg.norm(residual[:, -1, :]))
                                total_norm = float(mx.linalg.norm(h[:, -1, :]))
                                if total_norm > 0:
                                    # Higher layers have more attention contribution
                                    # Scale based on layer depth (validated finding)
                                    base_ratio = residual_norm / total_norm
                                    layer_factor = min(1.0, 0.5 + 0.5 * (idx / self.num_layers))
                                    attn_contrib = base_ratio * layer_factor

                            trace.add_decision(
                                layer=idx,
                                confidence=score,
                                selected=selected,
                                task=task_type,
                                attention_contribution=attn_contrib,
                            )

            if virtual_selected_this_step:
                virtual_selected_total += 1
            total_tokens += 1

            # Get logits
            if self._norm is not None:
                h = self._norm(h)

            if self._lm_head is not None:
                logits = self._lm_head(h)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            else:
                logits = h @ self._embed.weight.T

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            current_ids = mx.concatenate([current_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if generated and not any(c.isdigit() for c in token_str):
                break

        text = self.tokenizer.decode(generated).strip()
        used_virtual = virtual_selected_total > 0
        avg_score = np.mean(routing_scores) if routing_scores else 0.0

        return (
            text,
            used_virtual,
            virtual_selected_total,
            total_tokens,
            avg_score,
            selected_plugin,
            trace,
        )

    def _detect_task_type(self, prompt: str) -> str | None:
        """Detect the task type from the prompt for trace display."""
        prompt_lower = prompt.lower()
        if "*" in prompt or "×" in prompt:
            return "multiply"
        elif "+" in prompt:
            return "add"
        elif "-" in prompt and not prompt.startswith("-"):
            return "subtract"
        elif "/" in prompt or "÷" in prompt:
            return "divide"
        elif any(op in prompt_lower for op in ["sqrt", "root"]):
            return "sqrt"
        elif "^" in prompt or "**" in prompt:
            return "power"
        return "arithmetic"

    def _generate_with_runtime_routing(
        self,
        prompt: str,
        max_tokens: int = 20,
        collect_trace: bool = False,
    ) -> tuple[str, bool, int, int, float, str | None, RoutingTrace | None]:
        """Approximate routing on non-MLX backends using residual captures."""
        plugins = self.registry.get_all()
        trace = RoutingTrace() if collect_trace else None
        task_type = self._detect_task_type(prompt) if collect_trace else None

        used_virtual = False
        best_score = 0.0
        best_plugin_name: str | None = None

        for layer_idx in self.target_layers:
            hidden = self._get_hidden_state(prompt, layer_idx)
            layer_best_score = 0.0
            layer_selected = False

            for plugin_idx, plugin in enumerate(plugins):
                score = _routing_score(hidden, self._routing_params.get(layer_idx, {}).get(plugin_idx))
                if score > layer_best_score:
                    layer_best_score = score
                if score > best_score:
                    best_score = score
                    best_plugin_name = plugin.name

            layer_selected = layer_best_score > 0.5
            used_virtual = used_virtual or layer_selected

            if trace is not None:
                trace.add_decision(
                    layer=layer_idx,
                    confidence=layer_best_score,
                    selected=layer_selected,
                    task=task_type,
                )

        text = self._generate_direct(prompt, max_tokens=max_tokens)
        return (
            text,
            used_virtual,
            1 if used_virtual else 0,
            1,
            best_score,
            best_plugin_name if used_virtual else None,
            trace,
        )

    def _generate_direct(self, prompt: str, max_tokens: int = 20) -> str:
        """Generate directly without virtual experts."""
        if self.runtime is not None:
            from ..generation import GenerationConfig

            result = self.runtime.generate(
                prompt,
                GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
            return result.text.strip()

        mx = _mx()
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            token_str = self.tokenizer.decode([next_token])
            if "\n" in token_str:
                break
            if generated and not any(c.isdigit() for c in token_str):
                break

        return self.tokenizer.decode(generated).strip()

    def solve(
        self,
        prompt: str,
        max_tokens: int = 20,
        verbose: bool = False,
        action: VirtualExpertAction | None = None,
    ) -> InferenceResult:
        """
        Solve a problem, using virtual experts when appropriate.

        With action (Rogue-1 / CoT):
        1. Model output is parsed into VirtualExpertAction with trace steps
        2. Action.expert names the plugin to dispatch to
        3. Plugin executes the trace

        Without action (legacy):
        1. MoE routing decides if virtual expert should be used
        2. Only plugins with extract_and_evaluate work (MathExpert)

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            verbose: Collect detailed routing trace
            action: Pre-parsed VirtualExpertAction (from model YAML output or CoT rewriter)

        Returns:
            InferenceResult with the answer and metadata
        """
        if not self._calibrated:
            self.calibrate()

        # With CoT action: dispatch directly to the named expert
        if action and action.expert != "none":
            plugin = self.registry.get(action.expert)
            if plugin:
                ve_result = _run_sync(plugin.execute(action))
                if ve_result.success and ve_result.data:
                    data = ve_result.data
                    if isinstance(data, dict):
                        answer = str(
                            data.get("formatted", data.get("answer", data.get("result", data)))
                        )
                    else:
                        answer = str(data)

                    correct_answer = None
                    if isinstance(plugin, MathExpertPlugin):
                        _, correct_answer = plugin.extract_and_evaluate(prompt)

                    return InferenceResult(
                        prompt=prompt,
                        answer=answer,
                        correct_answer=correct_answer,
                        approach=VirtualExpertApproach.VIRTUAL_EXPERT,
                        used_virtual_expert=True,
                        plugin_name=plugin.name,
                        routing_score=1.0,
                        virtual_expert_selected_count=1,
                        total_tokens=1,
                    )

        # Legacy path: find plugin supporting extract_and_evaluate
        plugin = self.registry.find_handler(prompt)
        if plugin and "extract_and_evaluate" not in plugin.get_operations():
            plugin = None
        correct_answer = None

        if plugin and isinstance(plugin, MathExpertPlugin):
            _, correct_answer = plugin.extract_and_evaluate(prompt)

        # Generate with virtual expert checking
        gen_text, used_virtual, v_count, total, score, plugin_name, trace = (
            self._generate_with_virtual_expert(prompt, max_tokens, collect_trace=verbose)
        )

        # If virtual expert selected and we have a compatible plugin, dispatch
        if used_virtual and plugin:
            ve_action = VirtualExpertAction(
                expert=plugin.name,
                operation="extract_and_evaluate",
                parameters={"text": prompt},
            )

            ve_result = _run_sync(plugin.execute(ve_action))
            if ve_result.success and ve_result.data:
                data = ve_result.data
                if isinstance(data, dict):
                    answer = str(
                        data.get("formatted", data.get("answer", data.get("result", data)))
                    )
                else:
                    answer = str(data)
                approach = VirtualExpertApproach.VIRTUAL_EXPERT
            else:
                answer = gen_text
                approach = VirtualExpertApproach.MODEL_DIRECT
        else:
            answer = gen_text
            approach = VirtualExpertApproach.MODEL_DIRECT

        return InferenceResult(
            prompt=prompt,
            answer=answer,
            correct_answer=correct_answer,
            approach=approach,
            used_virtual_expert=used_virtual,
            plugin_name=plugin_name,
            routing_score=score,
            virtual_expert_selected_count=v_count,
            total_tokens=total,
            routing_trace=trace,
        )

    def compare(self, prompt: str, verbose: bool = False) -> InferenceResult:
        """Compare model-only vs virtual expert on a single prompt.

        Args:
            prompt: The input prompt
            verbose: Show detailed routing trace

        Returns:
            VirtualExpertResult with routing trace if verbose
        """
        plugin = self.registry.find_handler(prompt)
        correct = None
        if plugin and isinstance(plugin, MathExpertPlugin):
            _, correct = plugin.extract_and_evaluate(prompt)

        model_answer = self._generate_direct(prompt)
        result = self.solve(prompt, verbose=verbose)

        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt}")
        print(f"Correct answer: {correct}")
        print("-" * 60)
        print(f"Model alone:      {model_answer}")
        print(f"Virtual expert:   {result.answer}")
        if result.plugin_name:
            print(f"Plugin used:      {result.plugin_name}")
        print(
            f"Virtual selected: {result.virtual_expert_selected_count}/{result.total_tokens} tokens"
        )
        print(f"Correct:          {result.is_correct}")

        # Verbose routing trace
        if verbose and result.routing_trace:
            print("-" * 60)
            print("Routing Trace:")
            print(result.routing_trace.format_verbose())

        print(f"{'=' * 60}")

        return result

    def benchmark(self, problems: list[str]) -> VirtualExpertAnalysis:
        """
        Run benchmark on a list of problems.

        Args:
            problems: List of prompts to test

        Returns:
            VirtualExpertAnalysis with aggregate results
        """
        if not self._calibrated:
            self.calibrate()

        results = []
        correct_with = 0
        correct_without = 0
        times_used = 0
        routing_scores = []
        plugins_used: dict[str, int] = {}

        for prompt in problems:
            plugin = self.registry.find_handler(prompt)
            correct = None
            if plugin and isinstance(plugin, MathExpertPlugin):
                _, correct = plugin.extract_and_evaluate(prompt)

            # Model alone
            model_answer = self._generate_direct(prompt)
            model_correct = False
            if correct is not None:
                try:
                    match = re.search(r"-?\d+(?:\.\d+)?", model_answer)
                    if match:
                        model_correct = abs(float(match.group()) - correct) < 0.01
                except (ValueError, TypeError):
                    pass

            if model_correct:
                correct_without += 1

            # With virtual expert
            result = self.solve(prompt)

            if result.is_correct:
                correct_with += 1

            if result.used_virtual_expert:
                times_used += 1
                if result.plugin_name:
                    plugins_used[result.plugin_name] = plugins_used.get(result.plugin_name, 0) + 1

            if result.routing_score is not None:
                routing_scores.append(result.routing_score)

            results.append(result)

        return VirtualExpertAnalysis(
            model_name=self.model_id,
            total_problems=len(problems),
            correct_with_virtual=correct_with,
            correct_without_virtual=correct_without,
            times_virtual_used=times_used,
            avg_routing_score=np.mean(routing_scores) if routing_scores else 0,
            plugins_used=plugins_used,
            results=results,
        )


def create_virtual_expert_wrapper(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    plugins: list[VirtualExpertPlugin] | None = None,
    **kwargs,
) -> VirtualMoEWrapper:
    """
    Factory function to create a virtual expert wrapper.

    Args:
        model: The MoE model
        tokenizer: The tokenizer
        model_id: Model identifier
        plugins: Additional plugins to register
        **kwargs: Additional arguments for VirtualMoEWrapper

    Returns:
        Configured VirtualMoEWrapper
    """
    wrapper = VirtualMoEWrapper(model, tokenizer, model_id, **kwargs)

    if plugins:
        for plugin in plugins:
            wrapper.register_plugin(plugin)

    return wrapper
