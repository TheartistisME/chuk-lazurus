"""
Residual state serialization helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import LazarusBackend, ResidualState


def save_residual_state(path: str | Path, residual_state: ResidualState) -> Path:
    """Persist a residual state using a backend-appropriate format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if residual_state.backend == LazarusBackend.CUDA:
        import torch
        from safetensors.torch import save_file

        tensor = residual_state.tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to("cpu", non_blocking=True)

        output_path = output_path.with_suffix(".safetensors")
        save_file(
            {"residual": tensor.contiguous()},
            str(output_path),
            metadata={
                "backend": residual_state.backend.value,
                "layer_index": str(residual_state.layer_index),
                "sequence_length": str(residual_state.sequence_length),
                "hidden_size": str(residual_state.hidden_size),
                "dtype": residual_state.dtype,
                "device": residual_state.device,
            },
        )
        return output_path

    output_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        output_path,
        residual=np.asarray(residual_state.tensor),
        backend=residual_state.backend.value,
        layer_index=residual_state.layer_index,
        sequence_length=residual_state.sequence_length,
        hidden_size=residual_state.hidden_size,
        dtype=residual_state.dtype,
        device=residual_state.device,
    )
    return output_path


def load_residual_state(path: str | Path) -> ResidualState:
    """Load a residual state from disk."""
    input_path = Path(path)

    if input_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(str(input_path))
        residual = tensors["residual"]
        # `load_file` does not expose metadata, so read it through safe_open.
        from safetensors import safe_open

        with safe_open(str(input_path), framework="pt", device="cpu") as handle:
            info = handle.metadata()

        return ResidualState(
            backend=LazarusBackend(info["backend"]),
            layer_index=int(info["layer_index"]),
            tensor=residual,
            sequence_length=int(info["sequence_length"]),
            hidden_size=int(info["hidden_size"]),
            dtype=info["dtype"],
            device=info["device"],
        )

    archive = np.load(input_path, allow_pickle=False)
    return ResidualState(
        backend=LazarusBackend(str(archive["backend"])),
        layer_index=int(archive["layer_index"]),
        tensor=archive["residual"],
        sequence_length=int(archive["sequence_length"]),
        hidden_size=int(archive["hidden_size"]),
        dtype=str(archive["dtype"]),
        device=str(archive["device"]),
    )
