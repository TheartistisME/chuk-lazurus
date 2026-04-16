"""
Inference and text generation utilities.

The public surface is resolved lazily so ``import chuk_lazarus.inference`` stays
backend-neutral on torch/CUDA hosts.
"""

from __future__ import annotations

_LAZY: dict[str, tuple[str, str]] = {
    "ASSISTANT_SUFFIX": (".chat", "ASSISTANT_SUFFIX"),
    "ChatHistory": (".chat", "ChatHistory"),
    "ChatMessage": (".chat", "ChatMessage"),
    "FallbackTemplate": (".chat", "FallbackTemplate"),
    "Role": (".chat", "Role"),
    "format_chat_prompt": (".chat", "format_chat_prompt"),
    "format_history": (".chat", "format_history"),
    "BoundedKVEngine": (".context", "BoundedKVEngine"),
    "Checkpoint": (".context", "Checkpoint"),
    "CheckpointLibrary": (".context", "CheckpointLibrary"),
    "CheckpointMeta": (".context", "CheckpointMeta"),
    "CheckpointStore": (".context", "CheckpointStore"),
    "CompiledRSGenerator": (".context", "CompiledRSGenerator"),
    "ContextCheckpointFile": (".context", "ContextCheckpointFile"),
    "ContextCheckpointStatus": (".context", "ContextCheckpointStatus"),
    "ConversationState": (".context", "ConversationState"),
    "EngineStats": (".context", "EngineStats"),
    "GemmaBackboneAdapter": (".context", "GemmaBackboneAdapter"),
    "GemmaLayerAdapter": (".context", "GemmaLayerAdapter"),
    "GenerationMode": (".context", "GenerationMode"),
    "KVCheckpoint": (".context", "KVCheckpoint"),
    "KVDirectGenerator": (".context", "KVDirectGenerator"),
    "KVGeneratorProtocol": (".context", "KVGeneratorProtocol"),
    "KVStore": (".context", "KVStore"),
    "LibraryFile": (".context", "LibraryFile"),
    "LibraryFormatVersion": (".context", "LibraryFormatVersion"),
    "LibraryManifest": (".context", "LibraryManifest"),
    "LibrarySource": (".context", "LibrarySource"),
    "LlamaBackboneAdapter": (".context", "LlamaBackboneAdapter"),
    "LlamaLayerAdapter": (".context", "LlamaLayerAdapter"),
    "MemoryReport": (".context", "MemoryReport"),
    "ModelBackboneProtocol": (".context", "ModelBackboneProtocol"),
    "PathLabel": (".context", "PathLabel"),
    "TokenArchive": (".context", "TokenArchive"),
    "TransformerLayerProtocol": (".context", "TransformerLayerProtocol"),
    "TurnStats": (".context", "TurnStats"),
    "UnlimitedContextEngine": (".context", "UnlimitedContextEngine"),
    "WindowMeta": (".context", "WindowMeta"),
    "make_kv_generator": (".context", "make_kv_generator"),
    "GenerationConfig": (".generation", "GenerationConfig"),
    "GenerationResult": (".generation", "GenerationResult"),
    "GenerationStats": (".generation", "GenerationStats"),
    "StopReason": (".generation", "StopReason"),
    "generate": (".generation", "generate"),
    "generate_stream": (".generation", "generate_stream"),
    "get_stop_tokens": (".generation", "get_stop_tokens"),
    "DownloadConfig": (".loader", "DownloadConfig"),
    "DownloadResult": (".loader", "DownloadResult"),
    "DType": (".loader", "DType"),
    "HFLoader": (".loader", "HFLoader"),
    "LoadedWeights": (".loader", "LoadedWeights"),
    "StandardWeightConverter": (".loader", "StandardWeightConverter"),
    "WeightConverter": (".loader", "WeightConverter"),
    "IntrospectionResult": (".unified", "IntrospectionResult"),
    "UnifiedPipeline": (".unified", "UnifiedPipeline"),
    "UnifiedPipelineConfig": (".unified", "UnifiedPipelineConfig"),
    "UnifiedPipelineState": (".unified", "UnifiedPipelineState"),
    "MathExpertPlugin": (".virtual_expert", "MathExpertPlugin"),
    "SafeMathEvaluator": (".virtual_expert", "SafeMathEvaluator"),
    "VirtualDenseRouter": (".virtual_expert", "VirtualDenseRouter"),
    "VirtualDenseWrapper": (".virtual_expert", "VirtualDenseWrapper"),
    "VirtualExpertAnalysis": (".virtual_expert", "VirtualExpertAnalysis"),
    "VirtualExpertApproach": (".virtual_expert", "VirtualExpertApproach"),
    "VirtualExpertPlugin": (".virtual_expert", "VirtualExpertPlugin"),
    "VirtualExpertRegistry": (".virtual_expert", "VirtualExpertRegistry"),
    "VirtualExpertResult": (".virtual_expert", "VirtualExpertResult"),
    "VirtualMoEWrapper": (".virtual_expert", "VirtualMoEWrapper"),
    "VirtualRouter": (".virtual_expert", "VirtualRouter"),
    "create_virtual_dense_wrapper": (".virtual_expert", "create_virtual_dense_wrapper"),
    "create_virtual_expert_wrapper": (".virtual_expert", "create_virtual_expert_wrapper"),
    "get_default_registry": (".virtual_expert", "get_default_registry"),
}

_OPTIONAL = {
    "MathExpertPlugin",
    "SafeMathEvaluator",
    "VirtualDenseRouter",
    "VirtualDenseWrapper",
    "VirtualExpertAnalysis",
    "VirtualExpertApproach",
    "VirtualExpertPlugin",
    "VirtualExpertRegistry",
    "VirtualExpertResult",
    "VirtualMoEWrapper",
    "VirtualRouter",
    "create_virtual_dense_wrapper",
    "create_virtual_expert_wrapper",
    "get_default_registry",
}

__all__ = sorted(_LAZY.keys())


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        mod_name, attr = _LAZY[name]
        try:
            mod = importlib.import_module(mod_name, __name__)
        except ImportError:
            if name in _OPTIONAL:
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
    return sorted(list(globals().keys()) + __all__)
