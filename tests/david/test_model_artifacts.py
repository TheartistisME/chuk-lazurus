from __future__ import annotations

import json
from pathlib import Path

from chuk_lazarus.david.model_artifacts import inspect_vindex_artifact, is_vindex_artifact_path


def test_inspect_vindex_artifact_reports_structured_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "gemma4-e2b.vindex.ple"
    artifact.mkdir()
    (artifact / "index.json").write_text(
        json.dumps(
            {
                "family": "gemma4",
                "model": "fallback-model-path",
                "source": {"huggingface_repo": "google/gemma-4-E2B-it"},
                "hidden_size": 1536,
                "num_layers": 35,
                "vocab_size": 262144,
                "dtype": "f32",
            }
        ),
        encoding="utf-8",
    )
    (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact / "weight_manifest.json").write_text(
        json.dumps(
            [
                {"key": "layers.0.self_attn.q_proj.weight", "file": "attn_weights.bin"},
                {"key": "embed_tokens.weight", "file": "embeddings.bin"},
            ]
        ),
        encoding="utf-8",
    )
    (artifact / "attn_weights.bin").write_bytes(b"attn")
    (artifact / "embeddings.bin").write_bytes(b"embed")

    metadata = inspect_vindex_artifact(artifact)

    assert is_vindex_artifact_path(artifact) is True
    assert metadata.available is True
    assert metadata.family == "gemma4"
    assert metadata.source_hf_path == "google/gemma-4-E2B-it"
    assert metadata.hidden_size == 1536
    assert metadata.layers == 35
    assert metadata.vocab == 262144
    assert metadata.dtype == "f32"
    assert metadata.file_presence["index.json"] is True
    assert metadata.file_presence["attn_weights.bin"] is True
    assert metadata.binary_weight_files == ["attn_weights.bin", "embeddings.bin"]
    assert metadata.manifest_declared_files == ["attn_weights.bin", "embeddings.bin"]
    assert metadata.total_bytes > 0
    assert metadata.missing_files == []
    assert metadata.missing_binary_files == []


def test_inspect_vindex_artifact_fails_closed_when_files_are_missing(tmp_path: Path) -> None:
    artifact = tmp_path / "gemma4-e2b.vindex.ple"
    artifact.mkdir()
    (artifact / "index.json").write_text(
        json.dumps({"family": "gemma4", "hidden_size": 1536, "num_layers": 35}),
        encoding="utf-8",
    )

    metadata = inspect_vindex_artifact(artifact)

    assert metadata.available is False
    assert metadata.file_presence["index.json"] is True
    assert metadata.file_presence["tokenizer.json"] is False
    assert metadata.file_presence["weight_manifest.json"] is False
    assert metadata.missing_files == ["tokenizer.json", "weight_manifest.json"]


def test_inspect_vindex_artifact_reports_missing_manifest_binary_files(tmp_path: Path) -> None:
    artifact = tmp_path / "gemma4-e2b.vindex.ple"
    artifact.mkdir()
    (artifact / "index.json").write_text(
        json.dumps({"family": "gemma4", "hidden_size": 1536, "num_layers": 35}),
        encoding="utf-8",
    )
    (artifact / "tokenizer.json").write_text("{}", encoding="utf-8")
    (artifact / "weight_manifest.json").write_text(
        json.dumps(
            {
                "groups": [
                    {"file": "attn_weights.bin"},
                    {"nested": {"file": "embeddings.bin"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifact / "attn_weights.bin").write_bytes(b"attn")

    metadata = inspect_vindex_artifact(artifact)

    assert metadata.available is False
    assert metadata.manifest_declared_files == ["attn_weights.bin", "embeddings.bin"]
    assert metadata.binary_weight_files == ["attn_weights.bin", "embeddings.bin"]
    assert metadata.file_presence["attn_weights.bin"] is True
    assert metadata.file_presence["embeddings.bin"] is False
    assert metadata.missing_binary_files == ["embeddings.bin"]
    assert "embeddings.bin" in metadata.missing_files
