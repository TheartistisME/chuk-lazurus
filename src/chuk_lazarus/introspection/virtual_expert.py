"""
Virtual Expert Introspection and Demo Tools.

This module re-exports the core virtual expert classes from inference
and provides demo/analysis functions for introspection purposes.

For production use, import directly from chuk_lazarus.inference:

    from chuk_lazarus.inference import (
        VirtualMoEWrapper,
        VirtualExpertPlugin,
        MathExpertPlugin,
    )

For demos and analysis:

    from chuk_lazarus.introspection import (
        demo_virtual_expert,
        demo_all_approaches,
    )

CLI Usage:
    lazarus introspect virtual-expert analyze -m model
    lazarus introspect virtual-expert solve -m model --prompt "2+2="
    lazarus introspect virtual-expert benchmark -m model
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    import mlx.core as mx  # noqa: F401
    import mlx.nn as nn  # noqa: F401

    # Re-export core classes from inference (type-only)
    from chuk_lazarus.inference.virtual_expert import (
        InferenceResult,
        MathExpertPlugin,
        RoutingDecision,
        RoutingTrace,
        SafeMathEvaluator,
        VirtualDenseWrapper,
        VirtualExpertAnalysis,
        VirtualExpertApproach,
        VirtualExpertPlugin,
        VirtualExpertRegistry,
        VirtualExpertResult,
        VirtualMoEWrapper,
        VirtualRouter,
        create_virtual_expert_wrapper,
        get_default_registry,
    )
    from chuk_lazarus.inference.virtual_experts.cot_rewriter import FewShotCoTRewriter


_REEXPORT_NAMES = {
    "InferenceResult": ("chuk_lazarus.inference.virtual_expert", "InferenceResult"),
    "MathExpertPlugin": ("chuk_lazarus.inference.virtual_expert", "MathExpertPlugin"),
    "RoutingDecision": ("chuk_lazarus.inference.virtual_expert", "RoutingDecision"),
    "RoutingTrace": ("chuk_lazarus.inference.virtual_expert", "RoutingTrace"),
    "SafeMathEvaluator": ("chuk_lazarus.inference.virtual_expert", "SafeMathEvaluator"),
    "VirtualDenseWrapper": ("chuk_lazarus.inference.virtual_expert", "VirtualDenseWrapper"),
    "VirtualExpertAnalysis": ("chuk_lazarus.inference.virtual_expert", "VirtualExpertAnalysis"),
    "VirtualExpertApproach": ("chuk_lazarus.inference.virtual_expert", "VirtualExpertApproach"),
    "VirtualExpertPlugin": ("chuk_lazarus.inference.virtual_expert", "VirtualExpertPlugin"),
    "VirtualExpertRegistry": ("chuk_lazarus.inference.virtual_expert", "VirtualExpertRegistry"),
    "VirtualExpertResult": ("chuk_lazarus.inference.virtual_expert", "VirtualExpertResult"),
    "VirtualMoEWrapper": ("chuk_lazarus.inference.virtual_expert", "VirtualMoEWrapper"),
    "VirtualRouter": ("chuk_lazarus.inference.virtual_expert", "VirtualRouter"),
    "create_virtual_expert_wrapper": (
        "chuk_lazarus.inference.virtual_expert",
        "create_virtual_expert_wrapper",
    ),
    "get_default_registry": (
        "chuk_lazarus.inference.virtual_expert",
        "get_default_registry",
    ),
    "FewShotCoTRewriter": (
        "chuk_lazarus.inference.virtual_experts.cot_rewriter",
        "FewShotCoTRewriter",
    ),
}

# Legacy compatibility aliases also resolve lazily via __getattr__
_LEGACY_ALIASES = {
    "ExpertHijacker": "VirtualMoEWrapper",
    "VirtualExpertSlot": "VirtualMoEWrapper",
    "HybridEmbeddingInjector": "VirtualMoEWrapper",
}


def __getattr__(name: str):
    """PEP 562 lazy resolver for re-exported inference classes.

    Defers heavy ``mlx``-backed imports until the caller actually touches one
    of the re-exports.
    """
    if name in _REEXPORT_NAMES:
        import importlib  # noqa: PLC0415

        mod_path, attr = _REEXPORT_NAMES[name]
        mod = importlib.import_module(mod_path)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    if name in _LEGACY_ALIASES:
        target = _LEGACY_ALIASES[name]
        value = __getattr__(target)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class VirtualExpertAction(str, Enum):
    """Available virtual expert CLI actions."""

    ANALYZE = "analyze"
    """Analyze virtual expert behavior across test categories."""

    SOLVE = "solve"
    """Solve a single problem using virtual expert."""

    BENCHMARK = "benchmark"
    """Run benchmark on virtual expert system."""

    COMPARE = "compare"
    """Compare model output with and without virtual expert."""

    INTERACTIVE = "interactive"
    """Run interactive session with virtual expert."""


# Legacy compatibility aliases (ExpertHijacker, VirtualExpertSlot,
# HybridEmbeddingInjector) are resolved lazily via ``__getattr__`` above —
# keeping them as module attributes would force eager mlx import.


def demo_virtual_expert(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    problems: list[str] | None = None,
) -> VirtualExpertAnalysis:
    """
    Demo the virtual expert system.

    Args:
        model: The MoE model
        tokenizer: The tokenizer
        model_id: Model identifier
        problems: List of problems to test (defaults to arithmetic)

    Returns:
        VirtualExpertAnalysis with results
    """
    if problems is None:
        problems = [
            "2 + 2 = ",
            "5 * 5 = ",
            "10 - 3 = ",
            "6 * 7 = ",
            "25 + 17 = ",
            "100 - 37 = ",
            "23 * 17 = ",
            "127 * 89 = ",
            "456 * 78 = ",
            "999 * 888 = ",
        ]

    from chuk_lazarus.inference.virtual_expert import VirtualMoEWrapper  # noqa: PLC0415

    print("\n" + "=" * 70)
    print("VIRTUAL EXPERT DEMO")
    print("=" * 70)

    wrapper = VirtualMoEWrapper(model, tokenizer, model_id)

    print("\nCalibrating virtual expert routing...")
    wrapper.calibrate()
    print("Calibration complete.")

    print("\nRunning benchmark...\n")
    analysis = wrapper.benchmark(problems)

    print(f"{'Prompt':<25} {'Model':<15} {'Virtual':<15} {'Plugin':<10} {'V?':<5}")
    print("-" * 75)

    for result in analysis.results:
        model_answer = wrapper._generate_direct(result.prompt)[:12]
        virtual_answer = result.answer[:12]
        plugin = result.plugin_name or "N/A"
        used = "YES" if result.used_virtual_expert else "no"
        correct = "✓" if result.is_correct else "✗"

        print(
            f"{result.prompt:<25} {model_answer:<15} {virtual_answer:<15} {plugin:<10} {used:<5} {correct}"
        )

    print("\n" + "-" * 75)
    print(f"Model-only accuracy:   {analysis.model_accuracy:.1%}")
    print(f"With virtual expert:   {analysis.virtual_accuracy:.1%}")
    print(f"Improvement:           {analysis.virtual_accuracy - analysis.model_accuracy:+.1%}")
    print(f"Virtual expert used:   {analysis.times_virtual_used}/{analysis.total_problems}")

    if analysis.plugins_used:
        print("Plugins used:")
        for name, count in analysis.plugins_used.items():
            print(f"  - {name}: {count}")

    print("=" * 70)

    return analysis


def demo_all_approaches(
    model: nn.Module,
    tokenizer: Any,
    model_id: str = "unknown",
    problems: list[str] | None = None,
) -> dict[str, VirtualExpertAnalysis]:
    """
    Demo the virtual expert system.

    Note: This now uses the unified plugin-based approach.
    The "approaches" terminology is kept for backwards compatibility.

    Returns:
        Dict with single key "virtual_slot" containing analysis
    """
    analysis = demo_virtual_expert(model, tokenizer, model_id, problems)
    return {"virtual_slot": analysis}


def create_virtual_expert(
    model: nn.Module,
    tokenizer: Any,
    approach: str = "virtual_slot",
    model_id: str = "unknown",
    **kwargs,
) -> VirtualMoEWrapper:
    """
    Factory function for backwards compatibility.

    Note: The 'approach' parameter is ignored - all approaches now use
    the unified VirtualMoEWrapper with plugins.
    """
    from chuk_lazarus.inference.virtual_expert import VirtualMoEWrapper  # noqa: PLC0415

    return VirtualMoEWrapper(model, tokenizer, model_id, **kwargs)


# =============================================================================
# Service Layer for CLI Commands
# =============================================================================


class VirtualExpertConfig(BaseModel):
    """Configuration for virtual expert operations."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model path or name")
    backend: str | None = Field(default=None, description="Runtime backend override")
    device: str | None = Field(default=None, description="Runtime device override")
    layer: int | None = Field(default=None, description="Target layer")
    expert: int | None = Field(default=None, description="Target expert")
    prompt: str | None = Field(default=None, description="Prompt for solve/compare")
    verbose: bool = Field(default=False, description="Show detailed routing trace")
    use_few_shot_rewriter: bool = Field(
        default=False,
        description="Use FewShotCoTRewriter for query normalization (for non-CoT-trained models)",
    )
    test_categories: dict[str, list[str]] | None = Field(
        default=None, description="Test categories for analyze"
    )
    benchmark_problems: list[dict[str, Any]] | None = Field(
        default=None, description="Benchmark problems"
    )


class VirtualExpertServiceResult(BaseModel):
    """Result of virtual expert operation."""

    model_config = ConfigDict(frozen=True)

    action: str = Field(default="")
    results: list[dict[str, Any]] = Field(default_factory=list)
    accuracy: float | None = Field(default=None)
    summary: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = Field(default=None)

    def to_display(self) -> str:
        """Format result for display."""
        lines = [
            f"\n{'=' * 70}",
            f"VIRTUAL EXPERT: {self.action.upper()}",
            f"{'=' * 70}",
        ]

        if self.answer:
            lines.append(f"\nAnswer: {self.answer}")

        if self.accuracy is not None:
            lines.append(f"\nAccuracy: {self.accuracy:.1%}")

        if self.summary:
            # Special handling for routing trace (verbose mode)
            routing_trace = self.summary.pop("routing_trace", None)

            lines.append("\nSummary:")
            for key, value in self.summary.items():
                if key in ("hijack_layer", "hijack_confidence"):
                    continue  # Handled in routing trace
                lines.append(f"  {key}: {value}")

            # Show routing trace prominently if present
            if routing_trace:
                lines.append(f"\n{'-' * 70}")
                lines.append("ROUTING TRACE:")
                lines.append(routing_trace)

        return "\n".join(lines)


class VirtualExpertService:
    """Service for virtual expert operations."""

    @classmethod
    def _load_pipeline(cls, config: VirtualExpertConfig):
        """Load a backend-aware inference pipeline."""
        from ..inference import UnifiedPipeline, UnifiedPipelineConfig

        return UnifiedPipeline.from_pretrained(
            config.model,
            pipeline_config=UnifiedPipelineConfig(
                backend_name=config.backend,
                device=config.device,
            ),
            verbose=False,
        )

    @classmethod
    def _create_wrapper(
        cls,
        model,
        tokenizer,
        model_id: str,
        use_few_shot_rewriter: bool = False,
        *,
        layer: int | None = None,
        runtime: Any | None = None,
        pipeline: Any | None = None,
    ):
        """Create the appropriate wrapper based on model type.

        Auto-detects MoE vs Dense models and creates the right wrapper.

        Args:
            model: The loaded model
            tokenizer: The tokenizer
            model_id: Model identifier
            use_few_shot_rewriter: If True, use FewShotCoTRewriter for query normalization.
                                   Use this for models that are NOT CoT-trained.
                                   Default is False (assumes CoT-trained model).
        """
        from chuk_lazarus.inference.virtual_experts.dense_wrapper import (  # noqa: PLC0415
            VirtualDenseWrapper,
        )
        from chuk_lazarus.inference.virtual_experts.wrapper import (  # noqa: PLC0415
            VirtualMoEWrapper,
        )
        from chuk_lazarus.inference.virtual_experts.cot_rewriter import (  # noqa: PLC0415
            FewShotCoTRewriter,
        )

        # Check if model has MoE layers
        has_moe = False
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = list(model.model.layers)
        elif hasattr(model, "layers"):
            layers = list(model.layers)
        elif runtime is not None and hasattr(runtime, "_resolve_layers"):
            layers = list(runtime._resolve_layers())
        else:
            layers = []

        for model_layer in layers:
            if hasattr(model_layer, "mlp") and hasattr(model_layer.mlp, "router"):
                has_moe = True
                break

        if has_moe:
            # Use MoE wrapper for MoE models
            target_layers = [layer] if layer is not None else None
            return VirtualMoEWrapper(
                model,
                tokenizer,
                model_id,
                target_layers=target_layers,
                runtime=runtime,
                pipeline=pipeline,
            )
        else:
            # Use Dense wrapper for dense models
            rewriter = None
            if use_few_shot_rewriter:
                # Use FewShotCoTRewriter to normalize queries to VirtualExpertAction format
                # This is needed for models that are NOT trained on CoT action format
                rewriter = FewShotCoTRewriter(
                    model,
                    tokenizer,
                    max_examples_per_expert=5,
                    runtime=runtime,
                )

            wrapper = VirtualDenseWrapper(
                model,
                tokenizer,
                model_id,
                target_layer=layer,
                cot_rewriter=rewriter,
                runtime=runtime,
                pipeline=pipeline,
            )
            return wrapper

    @classmethod
    async def analyze(cls, config: VirtualExpertConfig) -> VirtualExpertServiceResult:
        """Analyze virtual expert behavior across test categories.

        Args:
            config: Virtual expert configuration.

        Returns:
            VirtualExpertServiceResult with analysis.
        """
        pipeline = cls._load_pipeline(config)
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Auto-detect MoE vs Dense and use appropriate wrapper
        wrapper = cls._create_wrapper(
            model,
            tokenizer,
            config.model,
            config.use_few_shot_rewriter,
            layer=config.layer,
            runtime=pipeline.runtime,
            pipeline=pipeline,
        )

        # Default test categories if not provided
        test_categories = config.test_categories or {
            "arithmetic": ["2 + 2 = ", "10 * 5 = ", "100 - 37 = "],
            "factual": ["The capital of France is ", "Water boils at "],
        }

        results = []
        summary = {}

        for category, prompts in test_categories.items():
            category_results = []
            for prompt in prompts:
                try:
                    result = wrapper.solve(prompt, max_tokens=10)
                    category_results.append(
                        {
                            "prompt": prompt,
                            "output": result.answer,
                            "plugin_used": result.plugin_name,
                            "used_virtual_expert": result.used_virtual_expert,
                        }
                    )
                except Exception as e:
                    category_results.append(
                        {
                            "prompt": prompt,
                            "error": str(e),
                        }
                    )

            results.extend(category_results)
            summary[category.upper()] = len(category_results)

        return VirtualExpertServiceResult(
            action="analyze",
            results=results,
            summary=summary,
        )

    @classmethod
    async def solve(cls, config: VirtualExpertConfig) -> VirtualExpertServiceResult:
        """Solve a single problem using virtual expert.

        Args:
            config: Virtual expert configuration with prompt.

        Returns:
            VirtualExpertServiceResult with answer.
        """
        if not config.prompt:
            raise ValueError("Prompt required for solve action")

        pipeline = cls._load_pipeline(config)
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Auto-detect MoE vs Dense model and use appropriate wrapper
        wrapper = cls._create_wrapper(
            model,
            tokenizer,
            config.model,
            config.use_few_shot_rewriter,
            layer=config.layer,
            runtime=pipeline.runtime,
            pipeline=pipeline,
        )

        # Solve using virtual expert
        result = wrapper.solve(config.prompt, max_tokens=30, verbose=config.verbose)

        return VirtualExpertServiceResult(
            action="solve",
            answer=result.answer,
            results=[
                {
                    "prompt": config.prompt,
                    "output": result.answer,
                    "plugin_used": result.plugin_name,
                    "used_virtual_expert": result.used_virtual_expert,
                    "is_correct": result.is_correct,
                }
            ],
        )

    @classmethod
    async def benchmark(cls, config: VirtualExpertConfig) -> VirtualExpertServiceResult:
        """Run benchmark on virtual expert system.

        Args:
            config: Virtual expert configuration with benchmark problems.

        Returns:
            VirtualExpertServiceResult with benchmark results.
        """
        pipeline = cls._load_pipeline(config)
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        # Auto-detect MoE vs Dense and use appropriate wrapper
        wrapper = cls._create_wrapper(
            model,
            tokenizer,
            config.model,
            config.use_few_shot_rewriter,
            layer=config.layer,
            runtime=pipeline.runtime,
            pipeline=pipeline,
        )

        # Default benchmark problems if not provided
        problems = config.benchmark_problems or [
            {"prompt": "2 + 2 = ", "expected": "4"},
            {"prompt": "10 * 5 = ", "expected": "50"},
            {"prompt": "100 - 37 = ", "expected": "63"},
            {"prompt": "15 + 27 = ", "expected": "42"},
        ]

        results = []
        correct = 0
        total = len(problems)

        for problem in problems:
            prompt = problem["prompt"]
            expected = problem.get("expected", "")

            try:
                result = wrapper.solve(prompt, max_tokens=10)

                # Check if answer matches
                is_correct = expected in result.answer.strip()
                if is_correct:
                    correct += 1

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "output": result.answer,
                        "correct": is_correct,
                        "plugin_used": result.plugin_name,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "error": str(e),
                        "correct": False,
                    }
                )

        return VirtualExpertServiceResult(
            action="benchmark",
            results=results,
            accuracy=correct / total if total > 0 else 0.0,
            summary={"correct": correct, "total": total},
        )

    @classmethod
    async def compare(cls, config: VirtualExpertConfig) -> VirtualExpertServiceResult:
        """Compare model output with and without virtual expert.

        Args:
            config: Virtual expert configuration with prompt.

        Returns:
            VirtualExpertServiceResult with comparison.
        """
        if not config.prompt:
            raise ValueError("Prompt required for compare action")

        pipeline = cls._load_pipeline(config)
        model = pipeline.model
        tokenizer = pipeline.tokenizer

        wrapper = cls._create_wrapper(
            model,
            tokenizer,
            config.model,
            config.use_few_shot_rewriter,
            layer=config.layer,
            runtime=pipeline.runtime,
            pipeline=pipeline,
        )
        result = wrapper.solve(config.prompt, max_tokens=30, verbose=config.verbose)
        direct_output = wrapper._generate_direct(config.prompt, max_tokens=30)

        # Build result with routing trace if verbose
        summary = {
            "with_expert": result.answer[:50] if result.answer else "",
            "without_expert": direct_output[:50],
        }

        if config.verbose and result.routing_trace:
            summary["routing_trace"] = result.routing_trace.format_verbose()
            if result.routing_trace.hijack_layer is not None:
                summary["hijack_layer"] = result.routing_trace.hijack_layer
                summary["hijack_confidence"] = result.routing_trace.hijack_confidence

        return VirtualExpertServiceResult(
            action="compare",
            results=[
                {
                    "prompt": config.prompt,
                    "with_expert": result.answer,
                    "without_expert": direct_output,
                    "plugin_used": result.plugin_name,
                    "used_virtual_expert": result.used_virtual_expert,
                    "is_correct": result.is_correct,
                    "routing_score": result.routing_score,
                }
            ],
            summary=summary,
        )

    @classmethod
    async def interactive(cls, config: VirtualExpertConfig) -> VirtualExpertServiceResult:
        """Run interactive session with virtual expert.

        Note: Interactive mode is not supported in service context.
        Use CLI directly for interactive mode.

        Args:
            config: Virtual expert configuration.

        Returns:
            VirtualExpertServiceResult indicating interactive mode not supported.
        """
        return VirtualExpertServiceResult(
            action="interactive",
            summary={
                "status": "Interactive mode not supported in service context. Use CLI directly."
            },
        )


__all__ = [
    # Enums
    "VirtualExpertAction",
    # Core classes (re-exported from inference)
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "InferenceResult",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "create_virtual_expert_wrapper",
    "get_default_registry",
    # Routing trace (verbose output)
    "RoutingDecision",
    "RoutingTrace",
    # Legacy aliases
    "ExpertHijacker",
    "VirtualExpertSlot",
    "HybridEmbeddingInjector",
    # Demo functions
    "demo_virtual_expert",
    "demo_all_approaches",
    "create_virtual_expert",
    # Service layer for CLI
    "VirtualExpertConfig",
    "VirtualExpertService",
    "VirtualExpertServiceResult",
]
