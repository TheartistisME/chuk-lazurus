from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chuk_lazarus.david.config import DavidConfig
from chuk_lazarus.david.model_backend import TransformersCausalLMBackend
from chuk_lazarus.david.runtime import DavidRuntime


pytestmark = pytest.mark.slow


def test_real_gemma_e2b_backend_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("DAVID_REAL_BACKEND_SMOKE") != "1":
        pytest.skip("set DAVID_REAL_BACKEND_SMOKE=1 to run the local Gemma E2B smoke")

    model_id = os.environ.get("DAVID_REAL_BACKEND_MODEL", "google/gemma-4-E2B-it")
    device = os.environ.get("DAVID_REAL_BACKEND_DEVICE", "cuda")
    _require_local_backend_prereqs(model_id=model_id, device=device)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()
    validation_report = tmp_path / "accepted-validation.json"
    validation_report.write_text(json.dumps(_accepted_validation_report(model_id)), encoding="utf-8")

    config = DavidConfig(
        workspace_root=workspace,
        state_dir=state_dir,
        model_path=model_id,
        validation_report_path=str(validation_report),
        require_validated_model=True,
        auto_jit_index=False,
        session_id="real-backend-smoke",
        max_route_tokens=256,
    )
    runtime = DavidRuntime.create(config)

    assert isinstance(runtime.backend, TransformersCausalLMBackend)
    assert runtime.backend.local_files_only is True
    runtime.backend.device = device

    result = runtime.run_once("Reply with exactly: DAVID_SMOKE_OK")

    assert result.model_result is not None
    assert result.model_result.ok is True, result.model_result.error
    assert result.model_result.backend == "transformers-causal-lm"
    assert result.model_result.metadata["local_files_only"] is True
    assert result.model_result.text.strip()


def _require_local_backend_prereqs(*, model_id: str, device: str) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")

    try:
        transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        transformers.AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"local-only model load is unavailable for {model_id}: {type(exc).__name__}: {exc}")


def _accepted_validation_report(model_id: str) -> dict[str, object]:
    selected_config = {
        "adapter_config_id": "gemma-e2b-real-smoke",
        "route_layer": 11,
        "route_query_head": 3,
        "route_dimension": 2048,
        "boundary_layer": 17,
        "residual_capture_layer": 17,
        "kv_source_layer": 21,
        "kv_target_layer": 23,
        "injection_layer": 23,
        "projection_producer_layer": 21,
        "behavior_cache_layer": 21,
        "insertion_family": "kv_direct",
        "kv_layout": "bshd",
        "candidate_role": "real-backend-smoke",
    }
    return {
        "schema_name": "lazarus.model_config_validation_report",
        "schema_version": 1,
        "validation_status": "accepted",
        "confidence": "high",
        "validation_level": "behavioral",
        "auto_load_allowed": True,
        "harness_load_policy": "auto",
        "selected_config": selected_config,
        "source_report_summary": {
            "model_identity": model_id,
            "tokenizer_identity": model_id,
            "adapter_family": "gemma4",
            "model_revision_or_hash": "local-smoke",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "model_identity_gate": {
            "model_identity": model_id,
            "tokenizer_identity": model_id,
            "adapter_family": "gemma4",
            "model_revision_or_hash": "local-smoke",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
        },
        "topology_gate": {"accepted": True},
        "projection_gate": {"ranked_candidates": []},
        "behavior_gate": {"accepted": True},
        "report_integrity": {"accepted": True},
        "provenance": {"loader_options": {"model": model_id, "local_files_only": True}},
        "warnings": [],
    }
