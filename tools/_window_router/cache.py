"""Feature-cache helpers for the window-router pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def dataset_meta_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_suffix(path.suffix + ".meta.json")


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dataset_meta(dataset_path: str | Path) -> dict[str, Any]:
    meta_path = dataset_meta_path(dataset_path)
    if not meta_path.exists():
        return {}
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def encoder_version(*, encoder_type: str, model_id: str | None = None) -> str:
    model_key = str(model_id or "none").strip() or "none"
    if encoder_type == "gemma-embed":
        return f"last-hidden-state-v1::{model_key}"
    if encoder_type == "sentence-transformer":
        return f"sentence-transformer-v1::{model_key}"
    return f"{encoder_type}-v1"


def feature_cache_key(*, encoder_type: str, encoder_version: str, dataset_hash: str) -> str:
    material = f"{encoder_type}|{encoder_version}|{dataset_hash}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def default_feature_cache_path(
    dataset_path: str | Path,
    *,
    encoder_type: str,
    encoder_version: str,
    dataset_hash: str,
) -> Path:
    path = Path(dataset_path)
    cache_key = feature_cache_key(
        encoder_type=encoder_type,
        encoder_version=encoder_version,
        dataset_hash=dataset_hash,
    )
    return path.with_name(
        f"{path.stem}.{encoder_type}.{cache_key[:16]}.features.pt"
    )


def resolve_device(
    requested_device: str,
    *,
    preferred_device: str | None = None,
) -> str:
    import torch  # noqa: PLC0415

    requested = str(requested_device or "auto").strip().lower()
    preferred = str(preferred_device or "").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unknown device {requested_device!r}")
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("cuda requested but torch.cuda.is_available() is false")
        return "cuda"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def write_feature_cache(
    out_path: str | Path,
    *,
    features: Any,
    labels: Any,
    meta: dict[str, Any],
) -> Path:
    import torch  # noqa: PLC0415

    cache_path = Path(out_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": torch.as_tensor(features, dtype=torch.float32).cpu(),
        "labels": torch.as_tensor(labels, dtype=torch.long).cpu(),
        "meta": dict(meta),
    }
    torch.save(payload, cache_path)
    sidecar = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")
    return cache_path


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("feature cache payload is not a dict")
    for key in ("features", "labels", "meta"):
        if key not in payload:
            raise ValueError(f"feature cache missing required key {key!r}")
    meta = payload["meta"]
    if not isinstance(meta, dict):
        raise ValueError("feature cache meta is not a dict")
    return payload


class CachedFeatureDataset:
    """Tensor-backed dataset that keeps feature vectors out of Python lists."""

    def __init__(self, features: Any, labels: Any) -> None:
        import torch  # noqa: PLC0415

        self._features = torch.as_tensor(features, dtype=torch.float32).cpu()
        self._labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        if self._features.shape[0] != self._labels.shape[0]:
            raise ValueError(
                "cached features and labels must have matching leading dimensions"
            )

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self._features[int(index)],
            "label": int(self._labels[int(index)].item()),
        }


__all__ = [
    "CachedFeatureDataset",
    "compute_file_sha256",
    "dataset_meta_path",
    "default_feature_cache_path",
    "encoder_version",
    "feature_cache_key",
    "load_feature_cache",
    "read_dataset_meta",
    "resolve_device",
    "write_feature_cache",
]
