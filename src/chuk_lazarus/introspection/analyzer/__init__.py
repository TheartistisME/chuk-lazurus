"""
Analyzer subpackage for model introspection.

The analyzer surface is re-exported lazily so importing
``chuk_lazarus.introspection.analyzer`` under ``CHUK_BACKEND=torch`` does
not eagerly import MLX-sensitive implementation modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-checker only
    from .config import AnalysisConfig, LayerStrategy, TrackStrategy  # noqa: F401
    from .core import ModelAnalyzer, analyze_prompt  # noqa: F401
    from .loader import (  # noqa: F401
        _is_quantized_model,
        _load_model_sync,
        get_model_hidden_size,
        get_model_num_layers,
        get_model_vocab_size,
    )
    from .models import (  # noqa: F401
        AnalysisResult,
        LayerPredictionResult,
        LayerTransition,
        ModelInfo,
        ResidualContribution,
        TokenEvolutionResult,
        TokenPrediction,
    )
    from .service import (  # noqa: F401
        AnalyzerService,
        AnalyzerServiceConfig,
        ComparisonResult,
    )
    from .utils import (  # noqa: F401
        compute_entropy,
        compute_js_divergence,
        compute_kl_divergence,
        get_layers_to_capture,
    )


_LAZY: dict[str, tuple[str, str]] = {
    "AnalysisConfig": (".config", "AnalysisConfig"),
    "LayerStrategy": (".config", "LayerStrategy"),
    "TrackStrategy": (".config", "TrackStrategy"),
    "AnalysisResult": (".models", "AnalysisResult"),
    "LayerPredictionResult": (".models", "LayerPredictionResult"),
    "LayerTransition": (".models", "LayerTransition"),
    "ModelInfo": (".models", "ModelInfo"),
    "ResidualContribution": (".models", "ResidualContribution"),
    "TokenEvolutionResult": (".models", "TokenEvolutionResult"),
    "TokenPrediction": (".models", "TokenPrediction"),
    "ModelAnalyzer": (".core", "ModelAnalyzer"),
    "analyze_prompt": (".core", "analyze_prompt"),
    "_is_quantized_model": (".loader", "_is_quantized_model"),
    "_load_model_sync": (".loader", "_load_model_sync"),
    "get_model_hidden_size": (".loader", "get_model_hidden_size"),
    "get_model_num_layers": (".loader", "get_model_num_layers"),
    "get_model_vocab_size": (".loader", "get_model_vocab_size"),
    "compute_entropy": (".utils", "compute_entropy"),
    "compute_js_divergence": (".utils", "compute_js_divergence"),
    "compute_kl_divergence": (".utils", "compute_kl_divergence"),
    "get_layers_to_capture": (".utils", "get_layers_to_capture"),
    "AnalyzerService": (".service", "AnalyzerService"),
    "AnalyzerServiceConfig": (".service", "AnalyzerServiceConfig"),
    "ComparisonResult": (".service", "ComparisonResult"),
}

__all__ = [
    "AnalysisConfig",
    "LayerStrategy",
    "TrackStrategy",
    "AnalysisResult",
    "LayerPredictionResult",
    "LayerTransition",
    "ModelInfo",
    "ResidualContribution",
    "TokenEvolutionResult",
    "TokenPrediction",
    "ModelAnalyzer",
    "analyze_prompt",
    "_is_quantized_model",
    "_load_model_sync",
    "get_model_hidden_size",
    "get_model_num_layers",
    "get_model_vocab_size",
    "compute_entropy",
    "compute_js_divergence",
    "compute_kl_divergence",
    "get_layers_to_capture",
    "AnalyzerService",
    "AnalyzerServiceConfig",
    "ComparisonResult",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
