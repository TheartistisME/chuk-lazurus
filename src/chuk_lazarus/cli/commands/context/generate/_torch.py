"""Torch/CUDA checkpoint-sidecar path for ``context generate``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .._types import GenerateConfig


class TorchCheckpointContractError(RuntimeError):
    """Raised when a torch checkpoint is missing its sidecar contract."""


def _default_top_k(value: int | None) -> int:
    return 3 if value is None else int(value)


def _validate_torch_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Path]:
    from ..prefill._torch_sidecar import TORCH_PREFILL_FILE, TORCH_STORE_DIR
    from .....inference.context.knowledge.torch_store import MANIFEST_FILE

    metadata_path = checkpoint_path / TORCH_PREFILL_FILE
    if not metadata_path.exists():
        raise TorchCheckpointContractError(
            f"torch checkpoint is missing {TORCH_PREFILL_FILE}; "
            "re-run `context prefill --backend torch` for this checkpoint."
        )

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is invalid JSON: {metadata_path}"
        ) from exc

    status = str(metadata.get("status", "missing"))
    if status != "complete":
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is incomplete (status={status!r}): {metadata_path}"
        )

    backend = metadata.get("backend")
    if backend not in (None, "torch"):
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar declares backend={backend!r}, expected 'torch'"
        )

    generate_batch = metadata.get("generate_batch", {})
    if generate_batch and not bool(generate_batch.get("ready", False)):
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is not marked ready for generate: {metadata_path}"
        )

    store_rel = metadata.get("artifacts", {}).get("store_dir", TORCH_STORE_DIR)
    store_path = checkpoint_path / str(store_rel)
    if not store_path.is_dir():
        raise TorchCheckpointContractError(
            f"torch checkpoint is missing store directory: {store_path}"
        )

    manifest_path = store_path / MANIFEST_FILE
    if not manifest_path.exists():
        raise TorchCheckpointContractError(
            f"torch checkpoint store is incomplete: missing {manifest_path}"
        )

    return metadata, store_path


def run_torch_checkpoint_generate(
    config: GenerateConfig,
    args,
    *,
    device: str | None,
) -> None:
    """Run checkpoint-backed generation through the torch knowledge-store path."""
    from .....inference.context.knowledge.torch_query import run_torch_query_command

    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        return

    checkpoint = config.checkpoint
    if checkpoint is None:
        raise ValueError("torch checkpoint generate requires config.checkpoint")
    if not checkpoint.exists():
        print(f"Error: library not found: {checkpoint}", file=sys.stderr)
        return

    try:
        metadata, store_path = _validate_torch_checkpoint(checkpoint)
    except TorchCheckpointContractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    model_info = metadata.get("model", {})
    windowing = metadata.get("windowing", {})
    sidecar_model_id = model_info.get("id")

    print(f"Loading torch checkpoint: {checkpoint}", file=sys.stderr)
    print(
        f"  torch_store={store_path.name}  |  {int(windowing.get('num_windows', 0))} windows  "
        f"|  {int(windowing.get('num_tokens', 0)):,} tokens",
        file=sys.stderr,
    )
    if sidecar_model_id and sidecar_model_id != config.model:
        print(
            f"Warning: torch sidecar model_id={sidecar_model_id!r} but loading model={config.model!r}",
            file=sys.stderr,
        )

    run_torch_query_command(
        model_id=config.model,
        prompt=prompt_text,
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=_default_top_k(getattr(args, "top_k", None)),
        device=device,
        store_path=store_path,
        system_prompt=getattr(args, "system_prompt", None),
        no_chat_template=bool(getattr(args, "no_chat_template", False)),
    )


__all__ = [
    "TorchCheckpointContractError",
    "run_torch_checkpoint_generate",
]
