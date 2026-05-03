from __future__ import annotations

from pathlib import Path

from chuk_lazarus.david import DavidConfig, DavidRuntime
from chuk_lazarus.david.config import AdapterSessionMetadata


def test_materializer_compatibility_comes_from_adapter_metadata(tmp_path: Path) -> None:
    adapter = AdapterSessionMetadata(
        model_id="test/model",
        tokenizer_id="test/tokenizer",
        model_revision="abc",
        adapter_family="test-family",
        boundary_layer=4,
        kv_source_layer=5,
        kv_target_layer=6,
        insertion_family="full_attention",
    )
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state", adapter=adapter))

    result = runtime.run_once("Fix repo patch target")

    assert result.materialized.compatibility["model_id"] == "test/model"
    assert result.materialized.compatibility["kv_source_layer"] == 5
    assert result.decoder.prior_scope["kv_target_layer"] == 6
    assert result.decoder.prior_scope["session_id"] == "default"


def test_decoder_constraints_for_temporal_recall(tmp_path: Path) -> None:
    runtime = DavidRuntime.create(DavidConfig(workspace_root=tmp_path, state_dir=tmp_path / "state"))

    result = runtime.run_once("What was the latest thing I asked you to remember?")

    assert result.method == "temporal_recall"
    assert result.decoder.constraints["ordinal"] == "exact_occurrence_required"
    assert result.decoder.prior_scope["model_id"] == "offline-deterministic"

