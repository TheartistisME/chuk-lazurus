"""Torch prefill sidecar dispatcher for the context prefill CLI."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TORCH_PREFILL_FILE = "torch_prefill.json"
TORCH_PREFILL_VERSION = 1
TORCH_STORE_DIR = "torch_store"


@dataclass(frozen=True)
class TorchPrefillArtifacts:
    """Resolved torch prefill artifacts written under a checkpoint root."""

    metadata_path: Path
    store_path: Path
    actual_device: str
    num_tokens: int
    num_windows: int
    metadata: dict[str, Any]
    reused: bool = False


def _resolve_device(device: str | None) -> str:
    import torch

    requested = device or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _load_model_and_tokenizer(model_id: str, device: str | None) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual_device = _resolve_device(device)
    dtype = torch.bfloat16 if actual_device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(actual_device)
    model.eval()
    return model, tokenizer, actual_device


def _resolve_architecture_config(model_config: Any, *, window_size: int | None):
    from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig

    config = ArchitectureConfig.from_model_config(model_config)
    if window_size is not None:
        object.__setattr__(config, "window_size", window_size)
    return config


def _metadata_to_artifacts(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    *,
    reused: bool,
) -> TorchPrefillArtifacts:
    artifacts = metadata.get("artifacts", {})
    windowing = metadata.get("windowing", {})
    model_info = metadata.get("model", {})

    store_rel = artifacts.get("store_dir", TORCH_STORE_DIR)
    metadata_rel = artifacts.get("sidecar", TORCH_PREFILL_FILE)
    return TorchPrefillArtifacts(
        metadata_path=checkpoint_path / str(metadata_rel),
        store_path=checkpoint_path / str(store_rel),
        actual_device=str(model_info.get("device", "unknown")),
        num_tokens=int(windowing.get("num_tokens", 0)),
        num_windows=int(windowing.get("num_windows", 0)),
        metadata=metadata,
        reused=reused,
    )


def load_existing_torch_prefill(checkpoint_path: Path | str) -> TorchPrefillArtifacts | None:
    """Load an existing complete torch sidecar contract if present."""

    from chuk_lazarus.inference.context.knowledge.torch_store import MANIFEST_FILE

    checkpoint = Path(checkpoint_path)
    metadata_path = checkpoint / TORCH_PREFILL_FILE
    if not metadata_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        return None

    artifacts = _metadata_to_artifacts(checkpoint, metadata, reused=True)
    if not (artifacts.store_path / MANIFEST_FILE).exists():
        return None
    return artifacts


def run_torch_prefill_sidecar(
    *,
    checkpoint_path: Path | str,
    model_id: str,
    input_path: Path | str,
    raw_text: str,
    name: str | None,
    window_size: int | None,
    max_tokens: int | None,
    requested_device: str | None,
    requested_phases: list[str],
    residual_mode: str,
    store_pages_requested: bool,
    store_kv_full_requested: bool,
    export_mode: bool,
    resume: bool,
) -> TorchPrefillArtifacts:
    """Build or reuse the torch-side prefill contract for later generate batches."""

    from chuk_lazarus.inference.context.knowledge.torch_build import build_knowledge_store_torch
    from chuk_lazarus.inference.context.knowledge.torch_store import MANIFEST_FILE, STORE_VERSION

    checkpoint = Path(checkpoint_path)
    checkpoint.mkdir(parents=True, exist_ok=True)

    # Route 'vec_inject' phase to the MLX-schema-parity torch module.
    # When phases == {'vec_inject'} the Apollo-v12 knowledge store is NOT
    # needed — supervisor-decision ve-ins-0mocicx9x0000d9ec75 Decision B.1
    # preserves Apollo-v12 unchanged for every other phase set.
    if set(requested_phases) == {"vec_inject"}:
        from ._vec_inject_torch import (
            VecInjectTorchArtifacts,
            run_vec_inject_prefill_torch,
        )

        vec_artifacts: VecInjectTorchArtifacts = run_vec_inject_prefill_torch(
            checkpoint_path=checkpoint,
            model_id=model_id,
            input_path=input_path,
            raw_text=raw_text,
            name=name,
            window_size=window_size,
            max_tokens=max_tokens,
            requested_device=requested_device,
            residual_mode=residual_mode,
            resume=resume,
        )
        return TorchPrefillArtifacts(
            metadata_path=checkpoint / "torch_vec_inject.json",
            store_path=vec_artifacts.vec_inject_npz_path,
            actual_device=vec_artifacts.actual_device,
            num_tokens=vec_artifacts.num_tokens,
            num_windows=vec_artifacts.num_windows,
            metadata=vec_artifacts.metadata,
            reused=vec_artifacts.reused,
        )

    existing = load_existing_torch_prefill(checkpoint)
    if resume and existing is not None:
        return existing

    store_path = checkpoint / TORCH_STORE_DIR
    if store_path.exists():
        shutil.rmtree(store_path)

    model, tokenizer, actual_device = _load_model_and_tokenizer(model_id, requested_device)
    arch_config = _resolve_architecture_config(model.config, window_size=window_size)
    store = build_knowledge_store_torch(
        model=model,
        tokenizer=tokenizer,
        raw_text=raw_text,
        config=arch_config,
        output_path=store_path,
        max_tokens=max_tokens,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "version": TORCH_PREFILL_VERSION,
        "kind": "torch-prefill",
        "status": "complete",
        "backend": "torch",
        "created_at": created_at,
        "model": {
            "id": model_id,
            "device": actual_device,
        },
        "source": {
            "name": name,
            "input_file": str(Path(input_path)),
            "max_tokens": max_tokens,
        },
        "windowing": {
            "window_size": int(store.config.window_size),
            "num_tokens": int(store.num_tokens),
            "num_windows": int(store.num_windows),
        },
        "arch_config": store.config.to_dict(),
        "requested": {
            "phases": list(requested_phases),
            "residual_mode": residual_mode,
            "store_pages": bool(store_pages_requested),
            "store_kv_full": bool(store_kv_full_requested),
            "mode": "export" if export_mode else "standard",
            "device": requested_device,
        },
        "written": {
            "phases": ["windows"],
            "store_pages": False,
            "store_kv_full": False,
            "mlx_checkpoint_artifacts": False,
            "torch_store_only": True,
        },
        "artifacts": {
            "sidecar": TORCH_PREFILL_FILE,
            "store_dir": TORCH_STORE_DIR,
            "store_layout": "apollo-v12",
            "store_version": STORE_VERSION,
            "manifest": f"{TORCH_STORE_DIR}/{MANIFEST_FILE}",
            "entries": f"{TORCH_STORE_DIR}/entries.npz",
            "window_tokens": f"{TORCH_STORE_DIR}/window_tokens.npz",
            "window_token_lists": f"{TORCH_STORE_DIR}/window_token_lists.npz",
            "idf": f"{TORCH_STORE_DIR}/idf.json",
            "keywords": f"{TORCH_STORE_DIR}/keywords.json",
            "boundaries_dir": f"{TORCH_STORE_DIR}/boundaries",
            "boundary_residual": f"{TORCH_STORE_DIR}/boundary_residual.npy",
        },
        "generate_batch": {
            "backend": "torch",
            "checkpoint_layout": "checkpoint-relative",
            "sidecar": TORCH_PREFILL_FILE,
            "store_dir": TORCH_STORE_DIR,
            "ready": True,
        },
    }

    metadata_path = checkpoint / TORCH_PREFILL_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return _metadata_to_artifacts(checkpoint, metadata, reused=False)


__all__ = [
    "TORCH_PREFILL_FILE",
    "TORCH_PREFILL_VERSION",
    "TORCH_STORE_DIR",
    "TorchPrefillArtifacts",
    "load_existing_torch_prefill",
    "run_torch_prefill_sidecar",
]
