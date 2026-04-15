"""
Introspection tools for model analysis.

This package exposes a large public API (analyzer, hooks, logit lens,
ablation, steering, MoE, etc.).  Many of the underlying submodules touch
``mlx.core`` / ``mlx.nn`` at module scope, which would make
``import chuk_lazarus.introspection`` pull ``libmlx.so`` on every platform.

To keep the package backend-neutral under ``CHUK_BACKEND=torch`` we resolve
every re-exported symbol lazily via PEP 562 ``__getattr__``.  The contract:

- ``import chuk_lazarus.introspection`` is cheap and does NOT import
  ``mlx``/``mlx_lm``.
- Accessing any attribute on the package triggers the matching submodule
  import on demand.
- Identical public API surface compared to the previous eager re-exports.

See ``src/chuk_lazarus/models_v2/__init__.py`` for the reference pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    # Re-imports purely to satisfy IDEs / type-checkers.  Not executed at runtime.
    from .ablation import (  # noqa: F401
        AblationConfig,
        AblationResult,
        AblationStudy,
        AblationType,
        ComponentType,
        LayerSweepResult,
        ModelAdapter,
    )
    from .accessor import AsyncModelAccessor, ModelAccessor  # noqa: F401
    from .analyzer import (  # noqa: F401
        AnalysisConfig,
        AnalysisResult,
        AnalyzerService,
        AnalyzerServiceConfig,
        ComparisonResult,
        LayerPredictionResult,
        LayerStrategy,
        LayerTransition,
        ModelAnalyzer,
        ModelInfo,
        ResidualContribution,
        TokenEvolutionResult,
        TokenPrediction,
        TrackStrategy,
        analyze_prompt,
    )
    from .attention import (  # noqa: F401
        AggregationStrategy,
        AttentionAnalyzer,
        AttentionFocus,
        AttentionPattern,
        extract_attention_weights,
    )
    from .hooks import (  # noqa: F401
        CaptureConfig,
        CapturedState,
        LayerSelection,
        ModelHooks,
        PositionSelection,
    )


# -----------------------------------------------------------------------------
# Lazy attribute map: public_name -> (relative submodule path, attribute).
# Kept as a plain dict so PEP 562 ``__getattr__`` can look up in constant time.
# -----------------------------------------------------------------------------

_LAZY: dict[str, tuple[str, str]] = {
    # --- Ablation (subpackage) ---
    "AblationConfig": (".ablation", "AblationConfig"),
    "AblationResult": (".ablation", "AblationResult"),
    "AblationStudy": (".ablation", "AblationStudy"),
    "AblationType": (".ablation", "AblationType"),
    "ComponentType": (".ablation", "ComponentType"),
    "LayerSweepResult": (".ablation", "LayerSweepResult"),
    "ModelAdapter": (".ablation", "ModelAdapter"),
    # --- Accessor ---
    "AsyncModelAccessor": (".accessor", "AsyncModelAccessor"),
    "ModelAccessor": (".accessor", "ModelAccessor"),
    # --- Analyzer service layer ---
    "AnalysisConfig": (".analyzer", "AnalysisConfig"),
    "AnalysisResult": (".analyzer", "AnalysisResult"),
    "AnalyzerService": (".analyzer", "AnalyzerService"),
    "AnalyzerServiceConfig": (".analyzer", "AnalyzerServiceConfig"),
    "ComparisonResult": (".analyzer", "ComparisonResult"),
    "LayerPredictionResult": (".analyzer", "LayerPredictionResult"),
    "LayerStrategy": (".analyzer", "LayerStrategy"),
    "LayerTransition": (".analyzer", "LayerTransition"),
    "ModelAnalyzer": (".analyzer", "ModelAnalyzer"),
    "ModelInfo": (".analyzer", "ModelInfo"),
    "ResidualContribution": (".analyzer", "ResidualContribution"),
    "TokenEvolutionResult": (".analyzer", "TokenEvolutionResult"),
    "TokenPrediction": (".analyzer", "TokenPrediction"),
    "TrackStrategy": (".analyzer", "TrackStrategy"),
    "analyze_prompt": (".analyzer", "analyze_prompt"),
    # --- Attention analysis ---
    "AggregationStrategy": (".attention", "AggregationStrategy"),
    "AttentionAnalyzer": (".attention", "AttentionAnalyzer"),
    "AttentionFocus": (".attention", "AttentionFocus"),
    "AttentionPattern": (".attention", "AttentionPattern"),
    "extract_attention_weights": (".attention", "extract_attention_weights"),
    # --- Circuit ---
    "CircuitCaptureConfig": (".circuit", "CircuitCaptureConfig"),
    "CircuitCaptureResult": (".circuit", "CircuitCaptureResult"),
    "CircuitCompareConfig": (".circuit", "CircuitCompareConfig"),
    "CircuitCompareResult": (".circuit", "CircuitCompareResult"),
    "CircuitDecodeConfig": (".circuit", "CircuitDecodeConfig"),
    "CircuitDecodeResult": (".circuit", "CircuitDecodeResult"),
    "CircuitExportConfig": (".circuit", "CircuitExportConfig"),
    "CircuitExportResult": (".circuit", "CircuitExportResult"),
    "CircuitInvokeConfig": (".circuit", "CircuitInvokeConfig"),
    "CircuitInvokeResult": (".circuit", "CircuitInvokeResult"),
    "CircuitService": (".circuit", "CircuitService"),
    "CircuitTestConfig": (".circuit", "CircuitTestConfig"),
    "CircuitTestResult": (".circuit", "CircuitTestResult"),
    "CircuitViewConfig": (".circuit", "CircuitViewConfig"),
    "CircuitViewResult": (".circuit", "CircuitViewResult"),
    # --- Classifier / Clustering ---
    "ClassifierConfig": (".classifier", "ClassifierConfig"),
    "ClassifierResult": (".classifier", "ClassifierResult"),
    "ClassifierService": (".classifier", "ClassifierService"),
    "ClusteringConfig": (".clustering", "ClusteringConfig"),
    "ClusteringResult": (".clustering", "ClusteringResult"),
    "ClusteringService": (".clustering", "ClusteringService"),
    # --- Enums ---
    "ArithmeticOperator": (".enums", "ArithmeticOperator"),
    "CommutativityLevel": (".enums", "CommutativityLevel"),
    "ComputeStrategy": (".enums", "ComputeStrategy"),
    "ConfidenceLevel": (".enums", "ConfidenceLevel"),
    "CriterionType": (".enums", "CriterionType"),
    "Difficulty": (".enums", "Difficulty"),
    "DirectionMethod": (".enums", "DirectionMethod"),
    "FactType": (".enums", "FactType"),
    "FormatDiagnosis": (".enums", "FormatDiagnosis"),
    "InvocationMethod": (".enums", "InvocationMethod"),
    "MemorizationLevel": (".enums", "MemorizationLevel"),
    "NeuronRole": (".enums", "NeuronRole"),
    "OverrideMode": (".enums", "OverrideMode"),
    "PatchEffect": (".enums", "PatchEffect"),
    "Region": (".enums", "Region"),
    "TestStatus": (".enums", "TestStatus"),
    # --- Generation services ---
    "GenerationConfig": (".generation", "GenerationConfig"),
    "GenerationResult": (".generation", "GenerationResult"),
    "GenerationService": (".generation", "GenerationService"),
    "LogitEvolutionConfig": (".generation", "LogitEvolutionConfig"),
    "LogitEvolutionResult": (".generation", "LogitEvolutionResult"),
    "LogitEvolutionService": (".generation", "LogitEvolutionService"),
    # --- Hooks ---
    "CaptureConfig": (".hooks", "CaptureConfig"),
    "CapturedState": (".hooks", "CapturedState"),
    "LayerSelection": (".hooks", "LayerSelection"),
    "ModelHooks": (".hooks", "ModelHooks"),
    "PositionSelection": (".hooks", "PositionSelection"),
    # --- Counterfactual interventions ---
    "CausalTraceResult": (".interventions", "CausalTraceResult"),
    "ComponentTarget": (".interventions", "ComponentTarget"),
    "CounterfactualIntervention": (".interventions", "CounterfactualIntervention"),
    "FullCausalTrace": (".interventions", "FullCausalTrace"),
    "InterventionConfig": (".interventions", "InterventionConfig"),
    "InterventionResult": (".interventions", "InterventionResult"),
    "InterventionType": (".interventions", "InterventionType"),
    "CounterfactualPatchingResult": (".interventions", "PatchingResult"),
    "patch_activations": (".interventions", "patch_activations"),
    "trace_causal_path": (".interventions", "trace_causal_path"),
    # --- Layer analysis ---
    "AttentionResult": (".layer_analysis", "AttentionResult"),
    "ClusterResult": (".layer_analysis", "ClusterResult"),
    "LayerAnalysisResult": (".layer_analysis", "LayerAnalysisResult"),
    "LayerAnalyzer": (".layer_analysis", "LayerAnalyzer"),
    "RepresentationResult": (".layer_analysis", "RepresentationResult"),
    "analyze_format_sensitivity": (".layer_analysis", "analyze_format_sensitivity"),
    # --- Logit lens ---
    "LayerPrediction": (".logit_lens", "LayerPrediction"),
    "LogitLens": (".logit_lens", "LogitLens"),
    "TokenEvolution": (".logit_lens", "TokenEvolution"),
    "run_logit_lens": (".logit_lens", "run_logit_lens"),
    # --- Memory ---
    "MemoryAnalysisConfig": (".memory", "MemoryAnalysisConfig"),
    "MemoryAnalysisResult": (".memory", "MemoryAnalysisResult"),
    "MemoryAnalysisService": (".memory", "MemoryAnalysisService"),
    # --- Pydantic models ---
    "ArithmeticStats": (".models", "ArithmeticStats"),
    "ArithmeticTestCase": (".models", "ArithmeticTestCase"),
    "ArithmeticTestResult": (".models", "ArithmeticTestResult"),
    "ArithmeticTestSuite": (".models", "ArithmeticTestSuite"),
    "AttractorNode": (".models", "AttractorNode"),
    "CalibrationResult": (".models", "CalibrationResult"),
    "CapitalFact": (".models", "CapitalFact"),
    "CapturedCircuit": (".models", "CapturedCircuit"),
    "CircuitComparisonResult": (".models", "CircuitComparisonResult"),
    "CircuitDirection": (".models", "CircuitDirection"),
    "CircuitEntry": (".models", "CircuitEntry"),
    "CircuitInvocationResult": (".models", "CircuitInvocationResult"),
    "CommutativityPair": (".models", "CommutativityPair"),
    "CommutativityResult": (".models", "CommutativityResult"),
    "ElementFact": (".models", "ElementFact"),
    "Fact": (".models", "Fact"),
    "FactNeighborhood": (".models", "FactNeighborhood"),
    "FactSet": (".models", "FactSet"),
    "MathFact": (".models", "MathFact"),
    "MemoryStats": (".models", "MemoryStats"),
    "MetacognitiveResult": (".models", "MetacognitiveResult"),
    "ParsedArithmeticPrompt": (".models", "ParsedArithmeticPrompt"),
    "PatchingLayerResult": (".models", "PatchingLayerResult"),
    "PatchingResult": (".models", "PatchingResult"),
    "ProbeLayerResult": (".models", "ProbeLayerResult"),
    "ProbeResult": (".models", "ProbeResult"),
    "ProbeTopNeuron": (".models", "ProbeTopNeuron"),
    "RetrievalResult": (".models", "RetrievalResult"),
    "UncertaintyResult": (".models", "UncertaintyResult"),
    # --- MoE ---
    "CategoryActivation": (".moe", "CategoryActivation"),
    "CategoryPrompts": (".moe", "CategoryPrompts"),
    "CoactivationAnalysis": (".moe", "CoactivationAnalysis"),
    "CompressionAnalysis": (".moe", "CompressionAnalysis"),
    "CompressionPlan": (".moe", "CompressionPlan"),
    "ExpertAblationResult": (".moe", "ExpertAblationResult"),
    "ExpertCategory": (".moe", "ExpertCategory"),
    "ExpertIdentity": (".moe", "ExpertIdentity"),
    "ExpertLogitContribution": (".moe", "ExpertLogitContribution"),
    "ExpertPair": (".moe", "ExpertPair"),
    "ExpertProfile": (".moe", "ExpertProfile"),
    "ExpertRole": (".moe", "ExpertRole"),
    "ExpertSimilarity": (".moe", "ExpertSimilarity"),
    "ExpertUtilization": (".moe", "ExpertUtilization"),
    "LayerRoutingSnapshot": (".moe", "LayerRoutingSnapshot"),
    "MoEAblationConfig": (".moe", "MoEAblationConfig"),
    "MoEArchitecture": (".moe", "MoEArchitecture"),
    "MoECaptureConfig": (".moe", "MoECaptureConfig"),
    "MoECapturedState": (".moe", "MoECapturedState"),
    "MoEHooks": (".moe", "MoEHooks"),
    "MoELayerInfo": (".moe", "MoELayerInfo"),
    "MoELogitLens": (".moe", "MoELogitLens"),
    "PromptCategory": (".moe", "PromptCategory"),
    "PromptCategoryGroup": (".moe", "PromptCategoryGroup"),
    "RouterEntropy": (".moe", "RouterEntropy"),
    "ablate_expert": (".moe", "ablate_expert"),
    "ablate_expert_batch": (".moe", "ablate_expert_batch"),
    "analyze_coactivation": (".moe", "analyze_coactivation"),
    "analyze_compression_opportunities": (".moe", "analyze_compression_opportunities"),
    "analyze_expert_vocabulary": (".moe", "analyze_expert_vocabulary"),
    "cluster_experts_by_specialization": (".moe", "cluster_experts_by_specialization"),
    "compare_routing": (".moe", "compare_routing"),
    "compute_expert_similarity": (".moe", "compute_expert_similarity"),
    "compute_routing_diversity": (".moe", "compute_routing_diversity"),
    "create_compression_plan": (".moe", "create_compression_plan"),
    "detect_moe_architecture": (".moe", "detect_moe_architecture"),
    "find_causal_experts": (".moe", "find_causal_experts"),
    "find_generalists": (".moe", "find_generalists"),
    "find_merge_candidates": (".moe", "find_merge_candidates"),
    "find_prune_candidates": (".moe", "find_prune_candidates"),
    "find_specialists": (".moe", "find_specialists"),
    "get_all_prompts": (".moe", "get_all_prompts"),
    "get_category_prompts": (".moe", "get_category_prompts"),
    "get_dominant_experts": (".moe", "get_dominant_experts"),
    "get_grouped_prompts": (".moe", "get_grouped_prompts"),
    "get_moe_layer_info": (".moe", "get_moe_layer_info"),
    "get_moe_layers": (".moe", "get_moe_layers"),
    "get_prompts_by_group": (".moe", "get_prompts_by_group"),
    "get_prompts_flat": (".moe", "get_prompts_flat"),
    "get_rare_experts": (".moe", "get_rare_experts"),
    "identify_all_experts": (".moe", "identify_all_experts"),
    "identify_expert": (".moe", "identify_expert"),
    "is_moe_model": (".moe", "is_moe_model"),
    "print_compression_summary": (".moe", "print_compression_summary"),
    "print_expert_summary": (".moe", "print_expert_summary"),
    "sweep_layer_experts": (".moe", "sweep_layer_experts"),
    "moe_compute_similarity_matrix": (".moe", "compute_similarity_matrix"),
    # --- Patcher ---
    "ActivationPatcher": (".patcher", "ActivationPatcher"),
    "CommutativityAnalyzer": (".patcher", "CommutativityAnalyzer"),
    # --- Probing ---
    "MetacognitiveConfig": (".probing", "MetacognitiveConfig"),
    "MetacognitiveService": (".probing", "MetacognitiveService"),
    "ProbeConfig": (".probing", "ProbeConfig"),
    "ProbeService": (".probing", "ProbeService"),
    "UncertaintyConfig": (".probing", "UncertaintyConfig"),
    "UncertaintyService": (".probing", "UncertaintyService"),
    # --- Steering ---
    "ActivationSteering": (".steering", "ActivationSteering"),
    "LegacySteeringConfig": (".steering", "LegacySteeringConfig"),
    "SteeredGemmaMLP": (".steering", "SteeredGemmaMLP"),
    "SteeringConfig": (".steering", "SteeringConfig"),
    "SteeringHook": (".steering", "SteeringHook"),
    "SteeringMode": (".steering", "SteeringMode"),
    "ToolCallingSteering": (".steering", "ToolCallingSteering"),
    "compare_steering_effects": (".steering", "compare_steering_effects"),
    "format_functiongemma_prompt": (".steering", "format_functiongemma_prompt"),
    "steer_model": (".steering", "steer_model"),
    # --- Utilities ---
    "analyze_orthogonality": (".utils", "analyze_orthogonality"),
    "apply_chat_template": (".utils", "apply_chat_template"),
    "compute_similarity_matrix": (".utils", "compute_similarity_matrix"),
    "cosine_similarity": (".utils", "cosine_similarity"),
    "extract_expected_answer": (".utils", "extract_expected_answer"),
    "find_answer_onset": (".utils", "find_answer_onset"),
    "find_discriminative_neurons": (".utils", "find_discriminative_neurons"),
    "generate_arithmetic_prompts": (".utils", "generate_arithmetic_prompts"),
    "load_external_chat_template": (".utils", "load_external_chat_template"),
    "normalize_number_string": (".utils", "normalize_number_string"),
    "parse_layers_arg": (".utils", "parse_layers_arg"),
    "parse_prompts_from_arg": (".utils", "parse_prompts_from_arg"),
}

# Optional virtual-expert surface.  Only wired if the optional
# ``chuk_virtual_expert`` dependency is installed.
_VIRTUAL_EXPERT_NAMES: tuple[str, ...] = (
    "ExpertHijacker",
    "HybridEmbeddingInjector",
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "VirtualExpertConfig",
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualExpertService",
    "VirtualExpertServiceResult",
    "VirtualExpertSlot",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "create_virtual_expert",
    "create_virtual_expert_wrapper",
    "demo_all_approaches",
    "demo_virtual_expert",
    "get_default_registry",
)
for _ve_name in _VIRTUAL_EXPERT_NAMES:
    _LAZY[_ve_name] = (".virtual_expert", _ve_name)


__all__ = sorted(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name, __name__)
        except ImportError:
            if name in _VIRTUAL_EXPERT_NAMES:
                # Optional dependency missing; mirror the previous behaviour of
                # the eager-try/except block by raising AttributeError.
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r} "
                    f"(optional dependency not installed)"
                ) from None
            raise
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
