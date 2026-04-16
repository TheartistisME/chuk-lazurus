"""
Mixture of Experts (MoE) introspection subpackage.

The package surface is resolved lazily so torch hosts can import
``chuk_lazarus.introspection.moe`` without walking through MLX-sensitive
modules until a concrete symbol is requested.
"""

from __future__ import annotations

import importlib


_LAZY_GROUPS: dict[str, list[str]] = {
    ".ablation": [
        "ablate_expert",
        "ablate_expert_batch",
        "find_causal_experts",
        "sweep_layer_experts",
    ],
    ".ablation_service": [
        "AblationBenchmarkResult",
        "AblationBenchmarkService",
        "BenchmarkProblemResult",
    ],
    ".analysis_service": [
        "AttentionCaptureResult",
        "ExpertWeightInfo",
        "LayerRoutingInfo",
        "MoEAnalysisService",
        "MoEAnalysisServiceConfig",
        "PositionRoutingInfo",
        "TaxonomyExpertMapping",
        "classify_token",
        "get_layer_phase",
        "get_trigram",
    ],
    ".attention_routing_service": [
        "DEFAULT_ATTENTION_CONTEXTS",
        "AttentionRoutingAnalysis",
        "AttentionRoutingService",
        "AttentionSummary",
        "ContextRoutingResult",
        "LayerRoutingResults",
    ],
    ".compression": [
        "ActivationOverlapResult",
        "CompressionAnalysis",
        "ExpertActivationStats",
        "ExpertSimilarity",
        "analyze_compression_opportunities",
        "collect_expert_activations",
        "compute_activation_overlap",
        "compute_expert_similarity",
        "compute_expert_similarity_with_activations",
        "compute_similarity_matrix",
        "compute_similarity_matrix_with_activations",
        "create_compression_plan",
        "find_merge_candidates",
        "find_merge_candidates_with_activations",
        "find_prune_candidates",
        "print_activation_overlap_matrix",
        "print_compression_summary",
    ],
    ".config": ["MoEAblationConfig", "MoECaptureConfig"],
    ".datasets": [
        "CATEGORY_GROUPS",
        "CategoryPrompts",
        "PromptCategory",
        "PromptCategoryGroup",
        "get_all_prompts",
        "get_category_prompts",
        "get_grouped_prompts",
        "get_prompts_by_group",
        "get_prompts_flat",
    ],
    ".detector": [
        "detect_moe_architecture",
        "get_moe_layer_info",
        "get_moe_layers",
        "is_moe_model",
    ],
    ".enums": ["ExpertCategory", "ExpertRole", "MoEAction", "MoEArchitecture", "MoEType"],
    ".expert_router": ["ExpertRouter"],
    ".explore_service": [
        "ComparisonResult",
        "DeepDiveResult",
        "ExploreService",
        "LayerPhaseData",
        "PatternMatch",
        "PositionEvolution",
        "TokenAnalysis",
    ],
    ".hooks": ["MoECapturedState", "MoEHooks"],
    ".identification": [
        "CategoryActivation",
        "ExpertProfile",
        "cluster_experts_by_specialization",
        "find_generalists",
        "find_specialists",
        "identify_all_experts",
        "identify_expert",
        "print_expert_summary",
    ],
    ".logit_lens": [
        "ExpertLogitContribution",
        "ExpertVocabContribution",
        "LayerRoutingSnapshot",
        "LayerVocabAnalysis",
        "MoELogitLens",
        "TokenExpertPreference",
        "VocabExpertMapping",
        "analyze_expert_vocabulary",
        "compute_expert_vocab_contribution",
        "compute_token_expert_mapping",
        "find_expert_specialists",
        "print_expert_vocab_summary",
        "print_token_expert_preferences",
    ],
    ".models": [
        "CoactivationAnalysis",
        "CompressionPlan",
        "ExpertAblationResult",
        "ExpertChatResult",
        "ExpertComparisonResult",
        "ExpertIdentity",
        "ExpertPair",
        "ExpertPattern",
        "ExpertTaxonomy",
        "ExpertUtilization",
        "GenerationStats",
        "LayerDivergenceResult",
        "LayerRouterWeights",
        "LayerRoutingAnalysis",
        "MoELayerInfo",
        "MoEModelInfo",
        "RouterEntropy",
        "RouterWeightCapture",
        "TokenExpertMapping",
        "TopKVariationResult",
        "VocabExpertAnalysis",
    ],
    ".moe_compression": [
        "CompressionConfig",
        "CompressionResult",
        "MoECompressionService",
        "OverlayExperts",
        "OverlayRepresentation",
        "ProjectionOverlay",
        "ReconstructionError",
        "ReconstructionVerification",
        "StorageEstimate",
    ],
    ".moe_type": ["MoETypeAnalysis", "MoETypeService", "ProjectionRankAnalysis"],
    ".overlay_inference": ["OverlayMoEModel"],
    ".router": [
        "analyze_coactivation",
        "compare_routing",
        "compute_routing_diversity",
        "get_dominant_experts",
        "get_rare_experts",
    ],
    ".test_data": [
        "ATTENTION_ROUTING_CONTEXTS",
        "CONTEXT_WINDOW_TESTS",
        "DEFAULT_CONTEXTS",
        "DOMAIN_PROMPTS",
        "TAXONOMY_TEST_PROMPTS",
        "TOKEN_CONTEXTS",
    ],
    ".tracking": [
        "CrossLayerAnalysis",
        "ExpertPipeline",
        "ExpertPipelineNode",
        "LayerAlignmentResult",
        "analyze_cross_layer_routing",
        "compute_expert_activation_profile",
        "compute_layer_alignment",
        "identify_functional_pipelines",
        "print_alignment_matrix",
        "print_pipeline_summary",
        "track_expert_across_layers",
    ],
    ".visualization": [
        "multi_layer_routing_matrix",
        "plot_expert_utilization",
        "plot_multi_layer_heatmap",
        "plot_routing_flow",
        "plot_routing_heatmap",
        "routing_heatmap_ascii",
        "routing_weights_to_matrix",
        "save_routing_heatmap",
        "save_utilization_chart",
        "utilization_bar_ascii",
    ],
}

_LAZY: dict[str, tuple[str, str]] = {
    name: (module, name)
    for module, names in _LAZY_GROUPS.items()
    for name in names
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
