"""Model backend seam for David terminal-agent harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping, Protocol, Sequence

from .config import AdapterSessionMetadata
from .materialization_replay import (
    ReplayConsumerCapabilities,
    ReplayConsumerInput,
    replay_generation_metadata,
)
from .model_artifacts import VindexArtifactMetadata, inspect_vindex_artifact


_SUPPORTED_DTYPE_STRINGS = {"auto", "float16", "bfloat16", "float32", "none"}


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


@dataclass(frozen=True)
class PromptFormatResult:
    prompt: str
    metadata: dict[str, Any]


class ModelBackend(Protocol):
    """Minimal generation contract used by the harness layer."""

    name: str

    def status(self) -> ModelBackendStatus:
        ...

    def replay_consumer_capabilities(
        self,
        adapter: AdapterSessionMetadata,
    ) -> ReplayConsumerCapabilities | None:
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
            metadata={"deterministic": True, "tensor_replay": False},
        )

    def replay_consumer_capabilities(
        self,
        adapter: AdapterSessionMetadata,
    ) -> ReplayConsumerCapabilities | None:
        return _no_tensor_replay_capabilities(self.name, adapter)

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


class VindexArtifactBackend:
    """Readiness-only backend for local .vindex.ple artifacts.

    These artifacts can supply model/memory/materialization evidence, but this
    harness does not yet have a decoder that can run them directly.
    """

    name = "vindex-artifact"

    def __init__(self, artifact_path: str) -> None:
        self.artifact_path = artifact_path
        self.artifact: VindexArtifactMetadata = inspect_vindex_artifact(artifact_path)

    def status(self) -> ModelBackendStatus:
        reason = (
            "available for methodology/materialization evidence; not directly loadable by "
            "transformers-causal-lm"
            if self.artifact.available
            else f"incomplete vindex artifact: missing {', '.join(self.artifact.missing_files) or 'required files'}"
        )
        return ModelBackendStatus(
            name=self.name,
            available=self.artifact.available,
            loaded=False,
            reason=reason,
            metadata={
                **self.artifact.to_dict(),
                "direct_generation": False,
                "loadable_by_transformers_causal_lm": False,
                "tensor_replay": False,
            },
        )

    def replay_consumer_capabilities(
        self,
        adapter: AdapterSessionMetadata,
    ) -> ReplayConsumerCapabilities | None:
        return _no_tensor_replay_capabilities(
            self.name,
            adapter,
            reason=".vindex.ple artifact backend has no decoder or tensor replay hook",
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
        del prompt, max_new_tokens, stop, logits_processor
        status = self.status()
        metadata = dict(status.metadata)
        replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=False,
        )
        if replay_metadata is not None:
            metadata["materialization_replay"] = replay_metadata
        return ModelBackendResult(
            text="",
            backend=self.name,
            ok=False,
            error=status.reason,
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
        torch_dtype: str | Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.device = device
        self.requested_dtype = _normalize_dtype_request(torch_dtype)
        self._resolved_dtype_label: str | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_error: str | None = None

    def status(self) -> ModelBackendStatus:
        dtype_error = self._requested_dtype_error()
        if dtype_error is not None:
            return ModelBackendStatus(
                name=self.name,
                available=False,
                loaded=False,
                reason=dtype_error,
                metadata=self._metadata(),
            )
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
            torch = __import__("torch")
            dtype_ok, resolved_dtype, dtype_reason = self._resolve_torch_dtype(torch)
            if not dtype_ok:
                self._load_error = dtype_reason
                return self.status()
            transformers = __import__("transformers", fromlist=["AutoModelForCausalLM", "AutoTokenizer"])
            kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
            if resolved_dtype is not None:
                kwargs["torch_dtype"] = resolved_dtype
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

    def replay_consumer_capabilities(
        self,
        adapter: AdapterSessionMetadata,
    ) -> ReplayConsumerCapabilities | None:
        return _no_tensor_replay_capabilities(
            self.name,
            adapter,
            reason="transformers causal-LM backend has no tensor replay hook installed",
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
            formatted_prompt = format_prompt_with_chat_template(self._tokenizer, prompt)
            inputs = self._tokenizer(formatted_prompt.prompt, return_tensors="pt")
            if self.device:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            generation_kwargs: dict[str, Any] = {**inputs, "max_new_tokens": max_new_tokens}
            processors = _normalize_logits_processors(logits_processor)
            if processors:
                generation_kwargs["logits_processor"] = processors
            with torch.no_grad():
                output_ids = self._model.generate(**generation_kwargs)
            prompt_token_count = _token_count_from_input_ids(inputs.get("input_ids"))
            generated_ids = _generated_token_ids(output_ids[0], prompt_token_count)
            generated_token_count = _token_count_from_ids(generated_ids)
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
            metadata = {
                **self._metadata(),
                **formatted_prompt.metadata,
                "max_new_tokens": max_new_tokens,
                "logits_processor_count": len(processors),
                "stop_count": len(stop or ()),
            }
            if prompt_token_count is not None:
                metadata["prompt_token_count"] = prompt_token_count
                metadata["input_token_count"] = prompt_token_count
            if generated_token_count is not None:
                metadata["generated_token_count"] = generated_token_count
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text=_apply_stop(text, stop),
                backend=self.name,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive fail-close path
            metadata = self._metadata()
            if self._tokenizer is not None:
                metadata.update(format_prompt_with_chat_template(self._tokenizer, prompt).metadata)
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
            "requested_dtype": _dtype_request_label(self.requested_dtype),
            "resolved_dtype": self._resolved_dtype_label,
            "tensor_replay": False,
        }

    def _requested_dtype_error(self) -> str | None:
        if not isinstance(self.requested_dtype, str):
            return None
        if self.requested_dtype in _SUPPORTED_DTYPE_STRINGS:
            return None
        allowed = ", ".join(sorted(_SUPPORTED_DTYPE_STRINGS))
        return f"invalid torch dtype '{self.requested_dtype}'; expected one of: {allowed}"

    def _resolve_torch_dtype(self, torch: Any) -> tuple[bool, Any | None, str | None]:
        if not isinstance(self.requested_dtype, str):
            self._resolved_dtype_label = type(self.requested_dtype).__name__
            return True, self.requested_dtype, None
        if self.requested_dtype == "none":
            self._resolved_dtype_label = None
            return True, None, None
        if self.requested_dtype == "auto":
            self._resolved_dtype_label = "auto"
            return True, "auto", None
        if not hasattr(torch, self.requested_dtype):
            return False, None, f"torch dtype '{self.requested_dtype}' is unavailable in installed torch"
        self._resolved_dtype_label = self.requested_dtype
        return True, getattr(torch, self.requested_dtype), None


def _missing_optional_packages(*names: str) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def format_prompt_with_chat_template(tokenizer: Any, prompt: str) -> PromptFormatResult:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        return PromptFormatResult(
            prompt=prompt,
            metadata={
                "prompt_format": "raw",
                "prompt_format_fallback_reason": "tokenizer has no apply_chat_template",
            },
        )

    try:
        formatted_prompt = apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception as exc:
        return PromptFormatResult(
            prompt=prompt,
            metadata={
                "prompt_format": "raw",
                "prompt_format_fallback_reason": f"{type(exc).__name__}: {exc}",
            },
        )

    if not isinstance(formatted_prompt, str):
        return PromptFormatResult(
            prompt=prompt,
            metadata={
                "prompt_format": "raw",
                "prompt_format_fallback_reason": (
                    "tokenizer.apply_chat_template returned "
                    f"{type(formatted_prompt).__name__}, expected str"
                ),
            },
        )

    return PromptFormatResult(
        prompt=formatted_prompt,
        metadata={
            "prompt_format": "chat_template",
            "prompt_format_source": "tokenizer.apply_chat_template",
        },
    )


def _no_tensor_replay_capabilities(
    backend_name: str,
    adapter: AdapterSessionMetadata,
    *,
    reason: str | None = None,
) -> ReplayConsumerCapabilities:
    return ReplayConsumerCapabilities(
        consumer_id=f"{backend_name}:no-tensor-replay",
        strategies=(),
        capabilities=(),
        model_id=adapter.model_id,
        tokenizer_id=adapter.tokenizer_id,
        model_revision=adapter.model_revision,
        adapter_family=adapter.adapter_family,
        insertion_families=(adapter.insertion_family,) if adapter.insertion_family else (),
        metadata={
            "supports_tensor_replay": False,
            "reason": reason or f"{backend_name} does not expose a tensor replay consumer",
        },
    )


def _normalize_dtype_request(value: str | Any | None) -> str | Any:
    if value is None:
        return "none"
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or "none"
    return value


def _dtype_request_label(value: str | Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


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


def _token_count_from_input_ids(input_ids: Any) -> int | None:
    if input_ids is None:
        return None
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        return len(input_ids[0])
    except (TypeError, IndexError, KeyError):
        pass
    try:
        return len(input_ids)
    except TypeError:
        return None


def _generated_token_ids(sequence_ids: Any, prompt_token_count: int | None) -> Any:
    if prompt_token_count is None:
        return sequence_ids
    try:
        return sequence_ids[prompt_token_count:]
    except (TypeError, IndexError, KeyError):
        return sequence_ids


def _token_count_from_ids(token_ids: Any) -> int | None:
    shape = getattr(token_ids, "shape", None)
    if shape is not None:
        try:
            return int(shape[-1])
        except (TypeError, ValueError, IndexError):
            pass
    try:
        return len(token_ids)
    except TypeError:
        return None
