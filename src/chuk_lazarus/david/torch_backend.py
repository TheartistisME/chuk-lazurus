"""Optional TorchInferenceRuntime-backed backend for David."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any, Mapping, Sequence

from .config import AdapterSessionMetadata
from .materialization_replay import (
    ReplayConsumerCapabilities,
    ReplayConsumerInput,
    replay_generation_metadata,
)
from .model_backend import ModelBackendResult, ModelBackendStatus, _apply_stop


_SUPPORTED_DTYPE_STRINGS = {"auto", "float16", "bfloat16", "float32", "none"}


@dataclass(frozen=True)
class _TorchRuntimeGeneration:
    config: Any
    text: str
    stats: Any | None
    stop_reason: Any | None
    generation_path: str | None


class TorchRuntimeModelBackend:
    """Local-only product backend using the proven TorchInferenceRuntime.

    This backend intentionally starts with standard decode only. Residual/KV
    tensor replay is left fail-closed until David has an explicit replay
    consumer bridge into TorchInferenceRuntime.
    """

    name = "torch-runtime"

    def __init__(
        self,
        model_id: str,
        *,
        local_files_only: bool = True,
        device: str | None = None,
        torch_dtype: str | Any | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.device = (device or "cuda").strip() or "cuda"
        self.trust_remote_code = trust_remote_code
        self.requested_dtype = _normalize_dtype_request(torch_dtype)
        self._resolved_dtype_label: str | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._runtime: Any | None = None
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
        missing = _missing_optional_packages("torch", "transformers")
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
            loaded=self._runtime is not None,
            reason=self._load_error or "ready",
            metadata=self._metadata(),
        )

    def load(self) -> ModelBackendStatus:
        status = self.status()
        if not status.available or status.loaded:
            return status
        try:
            torch = __import__("torch")
            if self.device.startswith("cuda") and not bool(torch.cuda.is_available()):
                self._load_error = "CUDA device requested but torch.cuda.is_available() is false"
                return self.status()
            dtype_ok, resolved_dtype, dtype_reason = self._resolve_torch_dtype(torch)
            if not dtype_ok:
                self._load_error = dtype_reason
                return self.status()

            transformers = __import__("transformers", fromlist=["AutoModelForCausalLM", "AutoTokenizer"])
            runtime_module = __import__(
                "chuk_lazarus.inference.backends.torch_runtime",
                fromlist=["TorchInferenceRuntime"],
            )

            tokenizer_kwargs = {
                "local_files_only": self.local_files_only,
                "trust_remote_code": self.trust_remote_code,
            }
            model_kwargs: dict[str, Any] = dict(tokenizer_kwargs)
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype

            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)
            self._model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            to_kwargs = {"non_blocking": True} if self.device.startswith("cuda") else {}
            self._model = self._model.to(self.device, **to_kwargs)
            self._model.eval()
            self._runtime = runtime_module.TorchInferenceRuntime(
                self._model,
                self._tokenizer,
                device=self.device,
                engine="standard",
            )
            self._load_error = None
        except Exception as exc:  # pragma: no cover - exercised via fake failures in tests.
            self._tokenizer = None
            self._model = None
            self._runtime = None
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self.status()

    def replay_consumer_capabilities(
        self,
        adapter: AdapterSessionMetadata,
    ) -> ReplayConsumerCapabilities | None:
        return ReplayConsumerCapabilities(
            consumer_id=f"{self.name}:no-tensor-replay",
            strategies=(),
            capabilities=(),
            model_id=adapter.model_id,
            tokenizer_id=adapter.tokenizer_id,
            model_revision=adapter.model_revision,
            adapter_family=adapter.adapter_family,
            insertion_families=(adapter.insertion_family,) if adapter.insertion_family else (),
            metadata={
                "supports_tensor_replay": False,
                "reason": "torch-runtime standard decode backend has no tensor replay hook installed",
            },
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
        processors = _normalize_logits_processors(logits_processor)
        status = self.load()
        replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=False,
        )
        if not status.available or not status.loaded:
            metadata = dict(status.metadata)
            metadata["logits_processor_count"] = len(processors)
            metadata["logits_processor_applied"] = False
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
            generation = self._generate_standard(prompt, max_new_tokens=max_new_tokens)
            metadata = {
                **self._metadata(),
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "use_plugins": False,
                "engine": "standard",
                "generation_path": generation.generation_path,
                "logits_processor_count": len(processors),
                "logits_processor_applied": False,
            }
            stats = _to_jsonable(generation.stats)
            if stats is not None:
                metadata["stats"] = stats
            if generation.stop_reason is not None:
                metadata["stop_reason"] = str(generation.stop_reason)
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text=_apply_stop(generation.text, stop),
                backend=self.name,
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive fail-close path.
            metadata = self._metadata()
            metadata["logits_processor_count"] = len(processors)
            metadata["logits_processor_applied"] = False
            if replay_metadata is not None:
                metadata["materialization_replay"] = replay_metadata
            return ModelBackendResult(
                text="",
                backend=self.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                metadata=metadata,
            )

    def _generate_standard(self, prompt: str, *, max_new_tokens: int) -> _TorchRuntimeGeneration:
        generation_module = __import__(
            "chuk_lazarus.inference.generation",
            fromlist=["GenerationConfig"],
        )
        generation_config = generation_module.GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_plugins=False,
        )
        result = self._runtime.generate(prompt, generation_config)
        return _TorchRuntimeGeneration(
            config=generation_config,
            text=str(getattr(result, "text", "")),
            stats=getattr(result, "stats", None),
            stop_reason=getattr(result, "stop_reason", None),
            generation_path=getattr(self._runtime, "last_generation_path", None),
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "requested_dtype": _dtype_request_label(self.requested_dtype),
            "resolved_dtype": self._resolved_dtype_label,
            "trust_remote_code": self.trust_remote_code,
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


def _normalize_dtype_request(value: str | Any | None) -> str | Any:
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized or "auto"
    return value


def _dtype_request_label(value: str | Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _normalize_logits_processors(logits_processor: Any | Sequence[Any] | None) -> list[Any]:
    if logits_processor is None:
        return []
    if isinstance(logits_processor, list):
        return logits_processor
    if isinstance(logits_processor, tuple):
        return list(logits_processor)
    return [logits_processor]


def _to_jsonable(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value
