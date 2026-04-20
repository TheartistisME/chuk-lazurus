"""
Residual state serialization helpers.

Both backends now serialize to .safetensors. Legacy .npz files are still
readable; new writes always produce .safetensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import LazarusBackend, ResidualState


def save_residual_state(path: str | Path, residual_state: ResidualState) -> Path:
    """Persist a residual state as .safetensors (both backends)."""
    output_path = Path(path).with_suffix(".safetensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "backend": residual_state.backend.value,
        "layer_index": str(residual_state.layer_index),
        "sequence_length": str(residual_state.sequence_length),
        "hidden_size": str(residual_state.hidden_size),
        "dtype": residual_state.dtype,
        "device": residual_state.device,
    }

    if residual_state.backend == LazarusBackend.CUDA:
        import torch
        from safetensors.torch import save_file

        tensor = residual_state.tensor
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to("cpu", non_blocking=True).contiguous()
        save_file({"residual": tensor}, str(output_path), metadata=metadata)
        return output_path

    # MLX or any other backend: write numpy-backed safetensors.
    from safetensors.numpy import save_file as save_file_np

    array = np.asarray(residual_state.tensor)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    array = np.ascontiguousarray(array)
    save_file_np({"residual": array}, str(output_path), metadata=metadata)
    return output_path


def load_residual_state(path: str | Path) -> ResidualState:
    """Load a residual state from .safetensors (preferred) or legacy .npz."""
    input_path = Path(path)

    if input_path.suffix == ".safetensors":
        # Read metadata via safe_open to decide framework.
        from safetensors import safe_open

        with safe_open(str(input_path), framework="pt", device="cpu") as handle:
            info = handle.metadata() or {}

        backend = LazarusBackend(info["backend"])

        if backend == LazarusBackend.CUDA:
            from safetensors.torch import load_file

            tensors = load_file(str(input_path))
            residual = tensors["residual"]
        else:
            from safetensors.numpy import load_file as load_file_np

            tensors = load_file_np(str(input_path))
            residual = tensors["residual"]

        return ResidualState(
            backend=backend,
            layer_index=int(info["layer_index"]),
            tensor=residual,
            sequence_length=int(info["sequence_length"]),
            hidden_size=int(info["hidden_size"]),
            dtype=info["dtype"],
            device=info["device"],
        )

    # Legacy .npz path (read-only; new writes go to .safetensors).
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
