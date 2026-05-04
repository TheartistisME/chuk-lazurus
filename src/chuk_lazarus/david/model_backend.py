"""Model backend seam for David terminal-agent harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping, Protocol, Sequence

from .materialization_replay import ReplayConsumerInput, replay_generation_metadata


@dataclass(frozen=True)
class ModelBackendStatus:
    name: str
    available: bool
    loaded: bool = False
    reason: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBackendResult:
    text: str
    backend: str
    ok: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelBackend(Protocol):
    """Minimal generation contract used by the harness layer."""

    name: str

    def status(self) -> ModelBackendStatus:
        ...

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        stop: Sequence[str] | None = None,
        logits_processor: Any | Sequence[Any] | None = None,
        materialization_plan: Mapping[str, Any] | None = None,
        replay_consumer: ReplayConsumerInput = None,
    ) -> ModelBackendResult:
        ...


class OfflineModelBackend:
    """Deterministic backend for offline tests and startup plumbing."""

    name = "offline-deterministic"

    def __init__(self, *, prefix: str = "offline") -> None:
        self.prefix = prefix

    def status(self) -> ModelBackendStatus:
        return ModelBackendStatus(
            name=self.name,
            available=True,
            loaded=True,
            metadata={"deterministic": True},
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        stop: Sequence[str] | None = None,
        logits_processor: Any | Sequence[Any] | None = None,
        materialization_plan: Mapping[str, Any] | None = None,
        replay_consumer: ReplayConsumerInput = None,
    ) -> ModelBackendResult:
        del logits_processor
        words = prompt.split()
        clipped = " ".join(words[: max(0, max_new_tokens)])
        text = f"{self.prefix}: {clipped}".strip()
        metadata: dict[str, Any] = {"prompt_tokens": len(words), "max_new_tokens": max_new_tokens}
        replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=False,
        )
        if replay_metadata is not None:
            metadata["materialization_replay"] = replay_metadata
        return ModelBackendResult(
            text=_apply_stop(text, stop),
            backend=self.name,
            metadata=metadata,
        )


class TransformersCausalLMBackend:
    """Optional local-only Hugging Face causal-LM backend.

    The backend is deliberately lazy and fail-closed: importing this module never
    imports torch/transformers, and the default loader passes local_files_only so
    construction and tests cannot download model assets.
    """

    name = "transformers-causal-lm"

    def __init__(
        self,
        model_id: str,
        *,
        local_files_only: bool = True,
        device: str | None = None,
        torch_dtype: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.device = device
        self.torch_dtype = torch_dtype
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    def status(self) -> ModelBackendStatus:
        missing = _missing_optional_packages("transformers", "torch")
        if missing:
            return ModelBackendStatus(
                name=self.name,
                available=False,
                loaded=False,
                reason=f"missing optional packages: {', '.join(missing)}",
                metadata=self._metadata(),
            )
        return ModelBackendStatus(
            name=self.name,
            available=self._load_error is None,
            loaded=self._model is not None and self._tokenizer is not None,
            reason=self._load_error or "ready",
            metadata=self._metadata(),
        )

    def load(self) -> ModelBackendStatus:
        status = self.status()
        if not status.available or status.loaded:
            return status
        try:
            transformers = __import__("transformers", fromlist=["AutoModelForCausalLM", "AutoTokenizer"])
            kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.local_files_only)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            if self.device:
                self._model = self._model.to(self.device)
            self._model.eval()
            self._load_error = None
        except Exception as exc:  # pragma: no cover - exercised without model assets by status/result
            self._tokenizer = None
            self._model = None
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self.status()

    @property
    def tokenizer(self) -> Any | None:
        return self._tokenizer

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        stop: Sequence[str] | None = None,
        logits_processor: Any | Sequence[Any] | None = None,
        materialization_plan: Mapping[str, Any] | None = None,
        replay_consumer: ReplayConsumerInput = None,
    ) -> ModelBackendResult:
        status = self.load()
        replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=False,
        )
        if not status.available or not status.loaded:
            metadata = dict(status.metadata)
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text="",
                backend=self.name,
                ok=False,
                error=status.reason,
                metadata=metadata,
            )
        try:
            torch = __import__("torch")
            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            generation_kwargs: dict[str, Any] = {**inputs, "max_new_tokens": max_new_tokens}
            processors = _normalize_logits_processors(logits_processor)
            if processors:
                generation_kwargs["logits_processor"] = processors
            with torch.no_grad():
                output_ids = self._model.generate(**generation_kwargs)
            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            metadata = {
                **self._metadata(),
                "max_new_tokens": max_new_tokens,
                "logits_processor_count": len(processors),
                "stop_count": len(stop or ()),
            }
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text=_apply_stop(text, stop),
                backend=self.name,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive fail-close path
            metadata = self._metadata()
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text="",
                backend=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata,
            )

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "local_files_only": self.local_files_only,
            "device": self.device,
        }


def _missing_optional_packages(*names: str) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def _apply_stop(text: str, stop: Sequence[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for marker in stop:
        if marker:
            index = text.find(marker)
            if index >= 0:
                cut = min(cut, index)
    return text[:cut]


def _normalize_logits_processors(logits_processor: Any | Sequence[Any] | None) -> list[Any]:
    if logits_processor is None:
        return []
    if isinstance(logits_processor, list):
        return logits_processor
    if isinstance(logits_processor, tuple):
        return list(logits_processor)
    return [logits_processor]
