from __future__ import annotations

import os
from pathlib import Path

import pytest

from chuk_lazarus.david.config import DavidConfig
from chuk_lazarus.david.model_backend import TransformersCausalLMBackend
from chuk_lazarus.david.runtime import DavidRuntime
from chuk_lazarus.david.torch_backend import TorchRuntimeModelBackend


pytestmark = pytest.mark.slow


def test_real_manual_attested_backend_standard_decode_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.environ.get("DAVID_REAL_ATTESTED_BACKEND_SMOKE") != "1":
        pytest.skip(
            "set DAVID_REAL_ATTESTED_BACKEND_SMOKE=1 to run the local manual-attested model smoke"
        )

    model_path = _required_env("DAVID_REAL_ATTESTED_MODEL")
    validation_report_path = Path(_required_env("DAVID_REAL_ATTESTED_VALIDATION_REPORT"))
    attestation_path = Path(_required_env("DAVID_REAL_ATTESTED_ATTESTATION"))
    backend_selector = os.environ.get("DAVID_REAL_ATTESTED_BACKEND", "transformers")
    device = os.environ.get("DAVID_REAL_ATTESTED_DEVICE", "cuda")
    dtype = os.environ.get("DAVID_REAL_ATTESTED_DTYPE", "auto")
    max_new_tokens = _int_env("DAVID_REAL_ATTESTED_MAX_NEW_TOKENS", default=4)

    assert validation_report_path.is_file(), (
        "DAVID_REAL_ATTESTED_VALIDATION_REPORT must point to an existing validator report"
    )
    assert attestation_path.is_file(), (
        "DAVID_REAL_ATTESTED_ATTESTATION must point to an existing manual attestation"
    )
    _require_local_backend_prereqs(backend_selector=backend_selector, device=device)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    workspace = tmp_path / "workspace"
    state_dir = tmp_path / "state"
    workspace.mkdir()

    runtime = DavidRuntime.create(
        DavidConfig(
            workspace_root=workspace,
            state_dir=state_dir,
            model_path=model_path,
            validation_report_path=str(validation_report_path),
            model_attestation_path=str(attestation_path),
            require_validated_model=True,
            model_backend=backend_selector,
            model_device=device,
            model_dtype=dtype,
            model_max_new_tokens=max_new_tokens,
            auto_jit_index=False,
            session_id="real-attested-backend-smoke",
            max_route_tokens=256,
        )
    )

    readiness = runtime.readiness()
    assert "manual_reviewed" in readiness["model validation"]
    assert "standard_decode_only" in readiness["model validation"]
    assert runtime.model_attestation is not None
    assert runtime.model_attestation.standard_decode_allowed is True

    backend_name = _normalized_backend_name(backend_selector)
    if backend_name == "torch-runtime":
        assert isinstance(runtime.backend, TorchRuntimeModelBackend)
    else:
        assert isinstance(runtime.backend, TransformersCausalLMBackend)

    result = runtime.run_once("Reply with exactly: DAVID_ATTESTED_SMOKE_OK")

    assert result.model_result is not None
    assert result.model_result.ok is True, result.model_result.error
    assert result.model_result.backend == backend_name
    assert result.model_result.text.strip()
    replay = result.model_result.metadata.get("materialization_replay", {})
    if replay:
        assert replay.get("tensor_replay") is False


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        pytest.fail(f"{name} is required when DAVID_REAL_ATTESTED_BACKEND_SMOKE=1")
    return value.strip()


def _int_env(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError:
        pytest.fail(f"{name} must be an integer")
    if parsed < 1:
        pytest.fail(f"{name} must be at least 1")
    return parsed


def _normalized_backend_name(selector: str) -> str:
    normalized = selector.strip().lower()
    if normalized in {"torch-runtime", "torch_runtime", "torch"}:
        return "torch-runtime"
    if normalized in {"transformers", "transformers-causal-lm", "hf"}:
        return "transformers-causal-lm"
    pytest.fail(f"unsupported DAVID_REAL_ATTESTED_BACKEND={selector!r}")


def _require_local_backend_prereqs(*, backend_selector: str, device: str) -> None:
    _normalized_backend_name(backend_selector)
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch")
