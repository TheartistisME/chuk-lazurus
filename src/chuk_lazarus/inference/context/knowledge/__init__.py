"""Knowledge store — the production path for document understanding."""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "streaming_prefill": (".build", "streaming_prefill"),
    "ArchitectureConfig": (".config", "ArchitectureConfig"),
    "ArchitectureNotCalibrated": (".config", "ArchitectureNotCalibrated"),
    "extract_donor_residual": (".inject", "extract_donor_residual"),
    "generate_with_boundary": (".inject", "generate_with_boundary"),
    "generate_with_injection": (".inject", "generate_with_injection"),
    "generate_with_markov_injection": (".inject", "generate_with_markov_injection"),
    "generate_with_persistent_injection": (".inject", "generate_with_persistent_injection"),
    "inject_1d": (".inject", "inject_1d"),
    "KeywordRouter": (".route", "KeywordRouter"),
    "SparseKeywordIndex": (".route", "SparseKeywordIndex"),
    "TFIDFRouter": (".route", "TFIDFRouter"),
    "_extract_keywords_from_text": (".route", "_extract_keywords_from_text"),
    "extract_window_keywords": (".route", "extract_window_keywords"),
    "InjectionEntry": (".store", "InjectionEntry"),
    "KnowledgeStore": (".store", "KnowledgeStore"),
    "build_knowledge_store_torch": (".torch_build", "build_knowledge_store_torch"),
    "TorchKnowledgeStore": (".torch_store", "TorchKnowledgeStore"),
}

__all__ = sorted(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
