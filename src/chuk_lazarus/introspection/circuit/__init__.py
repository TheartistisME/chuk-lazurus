"""
Circuit analysis tools for mechanistic interpretability.

Public re-exports stay lazy so importing the package does not immediately
pull in MLX-sensitive modules such as ``collector`` on torch hosts.
"""

from __future__ import annotations

import importlib


_LAZY_GROUPS: dict[str, list[str]] = {
    ".collector": [
        "ActivationCollector",
        "CollectedActivations",
        "CollectorConfig",
        "collect_activations",
    ],
    ".dataset": [
        "CircuitDataset",
        "ContrastivePair",
        "LabeledPrompt",
        "PromptCategory",
        "ToolPrompt",
        "ToolPromptDataset",
        "create_arithmetic_dataset",
        "create_binary_dataset",
        "create_code_execution_dataset",
        "create_contrastive_dataset",
        "create_factual_consistency_dataset",
        "create_tool_calling_dataset",
        "create_tool_delegation_dataset",
    ],
    ".directions": [
        "DirectionBundle",
        "DirectionExtractor",
        "DirectionMethod",
        "ExtractedDirection",
        "extract_all_directions",
        "extract_direction",
    ],
    ".export": [
        "CircuitEdge",
        "CircuitGraph",
        "CircuitNode",
        "EdgeType",
        "NodeType",
        "create_circuit_from_ablation",
        "create_circuit_from_directions",
        "export_circuit_to_dot",
        "export_circuit_to_html",
        "export_circuit_to_json",
        "export_circuit_to_mermaid",
        "load_circuit",
        "load_circuit_from_json",
        "save_circuit",
    ],
    ".probes": [
        "ProbeBattery",
        "ProbeDataset",
        "ProbeResult",
        "StratigraphyResult",
        "create_arithmetic_probe",
        "create_code_trace_probe",
        "create_factual_consistency_probe",
        "create_suppression_probe",
        "create_tool_decision_probe",
        "get_default_probe_datasets",
    ],
    ".service": [
        "CircuitCaptureConfig",
        "CircuitCaptureResult",
        "CircuitCompareConfig",
        "CircuitCompareResult",
        "CircuitDecodeConfig",
        "CircuitDecodeResult",
        "CircuitExportConfig",
        "CircuitExportResult",
        "CircuitInvokeConfig",
        "CircuitInvokeResult",
        "CircuitService",
        "CircuitTestConfig",
        "CircuitTestResult",
        "CircuitViewConfig",
        "CircuitViewResult",
    ],
}

_OPTIONAL_LAZY_GROUPS: dict[str, list[str]] = {
    ".geometry": [
        "GeometryAnalyzer",
        "GeometryResult",
        "ProbeType",
        "compute_pca",
        "compute_umap",
        "train_linear_probe",
    ]
}

_OPTIONAL_NAMES = {
    name
    for names in _OPTIONAL_LAZY_GROUPS.values()
    for name in names
}

_LAZY: dict[str, tuple[str, str]] = {
    name: (module, name)
    for groups in (_LAZY_GROUPS, _OPTIONAL_LAZY_GROUPS)
    for module, names in groups.items()
    for name in names
}

__all__ = sorted(name for name in _LAZY if name not in _OPTIONAL_NAMES)


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        try:
            module = importlib.import_module(module_name, __name__)
        except ImportError:
            if name in _OPTIONAL_NAMES:
                raise AttributeError(
                    f"module {__name__!r} has no attribute {name!r} "
                    f"(optional dependency not installed)"
                ) from None
            raise
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _OPTIONAL_NAMES)
