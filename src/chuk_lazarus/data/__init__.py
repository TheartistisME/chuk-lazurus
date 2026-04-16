"""
Data handling for training.

The package root resolves its public surface lazily so importing
``chuk_lazarus.data`` or any submodule like ``chuk_lazarus.data.base_dataset``
does not immediately pull MLX-only dataset implementations on torch hosts.
"""

from __future__ import annotations

_MODULE_EXPORTS: dict[str, list[str]] = {
    ".base_dataset": ["BaseDataset"],
    ".batch_dataset_base": ["BatchDatasetBase"],
    ".batching": [
        "BatchFingerprint",
        "BatchingConfig",
        "BatchingMode",
        "BatchMetrics",
        "BatchPlan",
        "BatchPlanBuilder",
        "BatchPlanMeta",
        "BatchReader",
        "BatchShapeHistogram",
        "BatchSpec",
        "BatchWriter",
        "BucketId",
        "BucketSpec",
        "BucketStats",
        "CollatedBatch",
        "EpochPlan",
        "LengthCache",
        "LengthEntry",
        "MicrobatchSpec",
        "PackedSequence",
        "PackingConfig",
        "PackingMetrics",
        "PackingMode",
        "PadPolicy",
        "SequenceToPack",
        "TokenBudgetBatchSampler",
        "compute_batch_fingerprint",
        "compute_packing_metrics",
        "create_segment_attention_mask",
        "default_collate",
        "load_batch_plan",
        "pack_sequences",
        "pad_sequences",
        "save_batch_plan",
        "verify_batch_fingerprint",
    ],
    ".classification_dataset": [
        "ClassificationDataset",
        "ClassificationSample",
        "load_classification_data",
    ],
    ".generators": [
        "MathProblem",
        "MathProblemGenerator",
        "ProblemType",
        "ToolCallTrace",
        "TrainingSample",
        "generate_lazarus_dataset",
    ],
    ".preference_dataset": [
        "PreferenceDataset",
        "PreferencePair",
        "load_preference_data",
    ],
    ".protocols": [
        "BatchableDataset",
        "ClassificationDatasetProtocol",
        "Dataset",
        "PreferenceDatasetProtocol",
        "SFTDatasetProtocol",
        "TokenizerProtocol",
    ],
    ".rollout_buffer": ["Episode", "RolloutBuffer", "Transition"],
    ".samples": [
        "DatasetFingerprint",
        "DatasetSource",
        "PreferenceSample",
        "Sample",
        "SampleMeta",
        "SampleType",
        "SampleValidationError",
        "compute_dataset_fingerprint",
    ],
    ".sft_dataset": ["SFTDataset", "SFTSample"],
    ".tokenizers": [
        "BoWCharacterTokenizer",
        "BoWTokenizerConfig",
        "CharacterTokenizer",
        "CharacterTokenizerConfig",
        "load_vocabulary",
        "save_vocabulary",
    ],
    ".train_batch_dataset": ["TrainBatchDataset"],
}

_LAZY: dict[str, tuple[str, str]] = {
    export: (module_name, export)
    for module_name, exports in _MODULE_EXPORTS.items()
    for export in exports
}

__all__ = [
    "BaseDataset",
    "Dataset",
    "BatchableDataset",
    "SFTDatasetProtocol",
    "PreferenceDatasetProtocol",
    "ClassificationDatasetProtocol",
    "TokenizerProtocol",
    "BatchDatasetBase",
    "TrainBatchDataset",
    "ClassificationDataset",
    "ClassificationSample",
    "load_classification_data",
    "BucketSpec",
    "BucketId",
    "BucketStats",
    "LengthCache",
    "LengthEntry",
    "TokenBudgetBatchSampler",
    "BatchSpec",
    "BatchMetrics",
    "BatchShapeHistogram",
    "PadPolicy",
    "BatchingMode",
    "BatchingConfig",
    "BatchFingerprint",
    "compute_batch_fingerprint",
    "verify_batch_fingerprint",
    "PackingMode",
    "PackingConfig",
    "PackedSequence",
    "SequenceToPack",
    "pack_sequences",
    "create_segment_attention_mask",
    "PackingMetrics",
    "compute_packing_metrics",
    "BatchPlan",
    "BatchPlanMeta",
    "BatchPlanBuilder",
    "EpochPlan",
    "MicrobatchSpec",
    "save_batch_plan",
    "load_batch_plan",
    "BatchWriter",
    "BatchReader",
    "CollatedBatch",
    "default_collate",
    "pad_sequences",
    "Sample",
    "SampleMeta",
    "SampleType",
    "DatasetSource",
    "PreferenceSample",
    "SampleValidationError",
    "DatasetFingerprint",
    "compute_dataset_fingerprint",
    "MathProblem",
    "MathProblemGenerator",
    "ProblemType",
    "ToolCallTrace",
    "TrainingSample",
    "generate_lazarus_dataset",
    "PreferenceDataset",
    "PreferencePair",
    "load_preference_data",
    "Episode",
    "RolloutBuffer",
    "Transition",
    "SFTDataset",
    "SFTSample",
    "BoWCharacterTokenizer",
    "BoWTokenizerConfig",
    "CharacterTokenizer",
    "CharacterTokenizerConfig",
    "load_vocabulary",
    "save_vocabulary",
]


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
