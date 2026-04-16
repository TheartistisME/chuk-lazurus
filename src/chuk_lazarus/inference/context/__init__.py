"""
Context management for stateful inference.

The concrete engines and knowledge paths remain MLX-backed in many cases, so
the package root resolves its public surface lazily via PEP 562 to stay
import-clean under ``CHUK_BACKEND=torch``.
"""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "GemmaBackboneAdapter": (".adapters", "GemmaBackboneAdapter"),
    "GemmaLayerAdapter": (".adapters", "GemmaLayerAdapter"),
    "LlamaBackboneAdapter": (".adapters", "LlamaBackboneAdapter"),
    "LlamaLayerAdapter": (".adapters", "LlamaLayerAdapter"),
    "ArchitectureConfig": (".arch_config", "ArchitectureConfig"),
    "ArchitectureNotCalibrated": (".arch_config", "ArchitectureNotCalibrated"),
    "BoundedKVEngine": (".bounded_engine", "BoundedKVEngine"),
    "Checkpoint": (".bounded_engine", "Checkpoint"),
    "ConversationState": (".bounded_engine", "ConversationState"),
    "GenerationMode": (".bounded_engine", "GenerationMode"),
    "MemoryReport": (".bounded_engine", "MemoryReport"),
    "PathLabel": (".bounded_engine", "PathLabel"),
    "TurnStats": (".bounded_engine", "TurnStats"),
    "CheckpointLibrary": (".checkpoint_library", "CheckpointLibrary"),
    "LibraryFile": (".checkpoint_library", "LibraryFile"),
    "LibraryFormatVersion": (".checkpoint_library", "LibraryFormatVersion"),
    "LibraryManifest": (".checkpoint_library", "LibraryManifest"),
    "WindowMeta": (".checkpoint_library", "WindowMeta"),
    "CheckpointMeta": (".kv_checkpoint", "CheckpointMeta"),
    "ContextCheckpointFile": (".kv_checkpoint", "ContextCheckpointFile"),
    "ContextCheckpointStatus": (".kv_checkpoint", "ContextCheckpointStatus"),
    "KVCheckpoint": (".kv_checkpoint", "KVCheckpoint"),
    "KVDirectGenerator": (".kv_generator", "KVDirectGenerator"),
    "make_kv_generator": (".kv_generator", "make_kv_generator"),
    "ModelBackboneProtocol": (".protocols", "ModelBackboneProtocol"),
    "TransformerLayerProtocol": (".protocols", "TransformerLayerProtocol"),
    "CompiledRSGenerator": (".rs_generator", "CompiledRSGenerator"),
    "SparseIndexEngine": (".sparse_engine", "SparseIndexEngine"),
    "EntityExtractor": (".sparse_index", "EntityExtractor"),
    "FactSpan": (".sparse_index", "FactSpan"),
    "SparseEntry": (".sparse_index", "SparseEntry"),
    "SparseSemanticIndex": (".sparse_index", "SparseSemanticIndex"),
    "SurpriseClassifier": (".sparse_index", "SurpriseClassifier"),
    "extract_content_words": (".sparse_index", "extract_content_words"),
    "CheckpointStore": (".unlimited_engine", "CheckpointStore"),
    "EngineStats": (".unlimited_engine", "EngineStats"),
    "KVGeneratorProtocol": (".unlimited_engine", "KVGeneratorProtocol"),
    "KVStore": (".unlimited_engine", "KVStore"),
    "LibrarySource": (".unlimited_engine", "LibrarySource"),
    "ResidualStore": (".unlimited_engine", "ResidualStore"),
    "TokenArchive": (".unlimited_engine", "TokenArchive"),
    "UnlimitedContextEngine": (".unlimited_engine", "UnlimitedContextEngine"),
    "KV_ROUTE_FILE": (".vec_inject", "KV_ROUTE_FILE"),
    "VEC_INJECT_FILE": (".vec_inject", "VEC_INJECT_FILE"),
    "LocalVecInjectProvider": (".vec_inject", "LocalVecInjectProvider"),
    "VecInjectMatch": (".vec_inject", "VecInjectMatch"),
    "VecInjectMeta": (".vec_inject", "VecInjectMeta"),
    "VecInjectMetaKey": (".vec_inject", "VecInjectMetaKey"),
    "VecInjectProvider": (".vec_inject", "VecInjectProvider"),
    "VecInjectResult": (".vec_inject", "VecInjectResult"),
    "VecInjectWindowKey": (".vec_inject", "VecInjectWindowKey"),
    "vec_inject": (".vec_inject", "vec_inject"),
    "vec_inject_all": (".vec_inject", "vec_inject_all"),
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
