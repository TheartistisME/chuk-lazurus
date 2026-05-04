"""Optional TorchInferenceRuntime-backed backend for David."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import AdapterSessionMetadata
from .materialization_replay import (
    REPLAY_STRATEGY_CAPABILITIES,
    ReplayConsumerCapabilities,
    ReplayConsumerInput,
    replay_generation_metadata,
)
from .model_backend import (
    ModelBackendResult,
    ModelBackendStatus,
    _apply_stop,
    format_prompt_with_chat_template,
)
from .residual_replay_bridge import (
    RESIDUAL_SIDECAR_STRATEGY,
    validate_residual_sidecar_replay,
)
from .residual_store import (
    ResidualReference,
    ResidualSidecarManifest,
    ResidualStore,
    ResidualStoreError,
)

_SUPPORTED_DTYPE_STRINGS = {"auto", "float16", "bfloat16", "float32", "none"}
_REQUIRED_TORCH_RUNTIME_PACKAGES = ("torch", "transformers", "pydantic")
_REQUIRED_TORCH_RUNTIME_MODULES = (
    "chuk_lazarus.inference.generation",
    "chuk_lazarus.inference.backends.torch_runtime",
)


@dataclass(frozen=True)
class _TorchRuntimeDependencyReport:
    required_packages: tuple[str, ...]
    required_modules: tuple[str, ...]
    missing_packages: tuple[str, ...]
    import_errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_packages and not self.import_errors

    def reason(self) -> str:
        if self.missing_packages:
            return f"missing torch-runtime dependencies: {', '.join(self.missing_packages)}"
        if self.import_errors:
            return f"torch-runtime dependency import failed: {'; '.join(self.import_errors)}"
        return "ready"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "required_packages": list(self.required_packages),
            "required_modules": list(self.required_modules),
            "missing_packages": list(self.missing_packages),
            "import_errors": list(self.import_errors),
        }


@dataclass(frozen=True)
class _TorchRuntimeGeneration:
    config: Any
    text: str
    stats: Any | None
    stop_reason: Any | None
    generation_path: str | None


@dataclass(frozen=True)
class _ResidualSidecarReplayInput:
    adapter: AdapterSessionMetadata
    sidecar: ResidualSidecarManifest
    ref: ResidualReference
    consumer: ReplayConsumerInput
    memory_family: str


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
        enable_residual_sidecar_replay: bool = False,
        residual_store: ResidualStore | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self.device = (device or "cuda").strip() or "cuda"
        self.trust_remote_code = trust_remote_code
        self.requested_dtype = _normalize_dtype_request(torch_dtype)
        self.enable_residual_sidecar_replay = bool(enable_residual_sidecar_replay)
        self._residual_store = residual_store or ResidualStore()
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
        dependencies = _torch_runtime_dependency_report()
        if not dependencies.ok:
            return ModelBackendStatus(
                name=self.name,
                available=False,
                loaded=False,
                reason=dependencies.reason(),
                metadata=self._metadata(dependencies),
            )
        return ModelBackendStatus(
            name=self.name,
            available=self._load_error is None,
            loaded=self._runtime is not None,
            reason=self._load_error or "ready",
            metadata=self._metadata(dependencies),
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
        if self.enable_residual_sidecar_replay:
            return ReplayConsumerCapabilities(
                consumer_id=f"{self.name}:residual-sidecar",
                strategies=(RESIDUAL_SIDECAR_STRATEGY,),
                capabilities=(REPLAY_STRATEGY_CAPABILITIES[RESIDUAL_SIDECAR_STRATEGY],),
                model_id=adapter.model_id,
                tokenizer_id=adapter.tokenizer_id,
                model_revision=adapter.model_revision,
                adapter_family=adapter.adapter_family,
                insertion_families=(adapter.insertion_family,) if adapter.insertion_family else (),
                metadata={
                    "supports_tensor_replay": True,
                    "strategy": RESIDUAL_SIDECAR_STRATEGY,
                    "reason": "residual-sidecar replay explicitly enabled",
                },
            )
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
        processor_metadata = _logits_processor_refusal_metadata(processors)
        status = self.load()
        replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=False,
        )
        if not status.available or not status.loaded:
            metadata = dict(status.metadata)
            metadata.update(processor_metadata)
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
            formatted_prompt = format_prompt_with_chat_template(self._tokenizer, prompt)
            residual_input, residual_refusal = self._residual_sidecar_replay_input(
                materialization_plan,
                replay_consumer,
            )
            if residual_refusal is not None:
                metadata = {
                    **self._metadata(),
                    **formatted_prompt.metadata,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "use_plugins": False,
                    "engine": "standard",
                    **processor_metadata,
                    "materialization_replay": residual_refusal,
                }
                return ModelBackendResult(
                    text="",
                    backend=self.name,
                    ok=False,
                    error=str(residual_refusal["reason"]),
                    metadata=metadata,
                )
            if residual_input is not None:
                residual_replay_metadata = replay_generation_metadata(
                    materialization_plan,
                    replay_consumer,
                    backend_name=self.name,
                    supports_tensor_replay=True,
                )
                if processors:
                    refusal = _replay_metadata_with_refusal(
                        residual_replay_metadata,
                        "torch-runtime residual-sidecar path cannot apply decoder logits processors",
                    )
                    metadata = {
                        **self._metadata(),
                        **formatted_prompt.metadata,
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                        "use_plugins": False,
                        "engine": "standard",
                        **processor_metadata,
                        "materialization_replay": refusal,
                    }
                    return ModelBackendResult(
                        text="",
                        backend=self.name,
                        ok=False,
                        error=str(refusal["reason"]),
                        metadata=metadata,
                    )
                generation = self._generate_residual_sidecar(
                    formatted_prompt.prompt,
                    residual_input,
                    max_new_tokens=max_new_tokens,
                )
                metadata = {
                    **self._metadata(),
                    **formatted_prompt.metadata,
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                    "use_plugins": False,
                    "engine": "residual_sidecar",
                    "generation_path": generation.generation_path,
                    **processor_metadata,
                }
                stats = _to_jsonable(generation.stats)
                if stats is not None:
                    metadata["stats"] = stats
                if generation.stop_reason is not None:
                    metadata["stop_reason"] = str(generation.stop_reason)
                if residual_replay_metadata is not None:
                    metadata["materialization_replay"] = _replay_metadata_applied(
                        residual_replay_metadata
                    )
                metadata["residual_sidecar_replay"] = generation.config
                return ModelBackendResult(
                    text=_apply_stop(generation.text, stop),
                    backend=self.name,
                    metadata=metadata,
                )

            generation = self._generate_standard(formatted_prompt.prompt, max_new_tokens=max_new_tokens)
            metadata = {
                **self._metadata(),
                **formatted_prompt.metadata,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "use_plugins": False,
                "engine": "standard",
                "generation_path": generation.generation_path,
                **processor_metadata,
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
            if self._tokenizer is not None:
                metadata.update(format_prompt_with_chat_template(self._tokenizer, prompt).metadata)
            metadata.update(processor_metadata)
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

    def _generate_residual_sidecar(
        self,
        prompt: str,
        residual_input: _ResidualSidecarReplayInput,
        *,
        max_new_tokens: int,
    ) -> _TorchRuntimeGeneration:
        generation_module = __import__(
            "chuk_lazarus.inference.generation",
            fromlist=["GenerationConfig"],
        )
        types_module = __import__(
            "chuk_lazarus.inference.backends.types",
            fromlist=["LazarusBackend", "ResidualState"],
        )
        decision = validate_residual_sidecar_replay(
            adapter=residual_input.adapter,
            sidecar=residual_input.sidecar,
            consumer=residual_input.consumer,
            memory_family=residual_input.memory_family,
            layer=residual_input.ref.layer,
        )
        decision_metadata = decision.to_metadata()
        if decision.refused:
            raise ResidualStoreError(decision.reason)

        loaded = self._residual_store.load_tensor(residual_input.sidecar, residual_input.ref)
        _validate_loaded_single_vector_residual(loaded)
        layer = loaded.layer
        if layer is None:
            raise ResidualStoreError("residual sidecar tensor missing layer")
        hidden_size = int(loaded.shape[-1])
        generation_config = generation_module.GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            use_plugins=False,
        )
        residual_state = types_module.ResidualState(
            backend=types_module.LazarusBackend.CUDA,
            layer_index=int(layer),
            tensor=loaded.array,
            sequence_length=1,
            hidden_size=hidden_size,
            dtype=loaded.dtype,
            device="cpu",
        )
        result = self._runtime.generate_with_residual_seeded_at_layer(
            prompt,
            residual_state,
            generation_config,
        )
        decision_metadata.update(
            {
                "tensor_replay_advertised": self.enable_residual_sidecar_replay,
                "tensor_replay_applied": True,
                "tensors_loaded": True,
                "loaded_tensor": {
                    "kind": loaded.kind,
                    "layer": loaded.layer,
                    "dtype": loaded.dtype,
                    "shape": list(loaded.shape),
                    "hidden_size": hidden_size,
                },
            }
        )
        return _TorchRuntimeGeneration(
            config=decision_metadata,
            text=str(getattr(result, "text", "")),
            stats=getattr(result, "stats", None),
            stop_reason=getattr(result, "stop_reason", None),
            generation_path=getattr(self._runtime, "last_generation_path", None),
        )

    def _residual_sidecar_replay_input(
        self,
        materialization_plan: Mapping[str, Any] | None,
        replay_consumer: ReplayConsumerInput,
    ) -> tuple[_ResidualSidecarReplayInput | None, dict[str, Any] | None]:
        if not _residual_sidecar_requested(materialization_plan):
            return None, None
        base_replay_metadata = replay_generation_metadata(
            materialization_plan,
            replay_consumer,
            backend_name=self.name,
            supports_tensor_replay=True,
        )
        try:
            plan = _materialization_plan_mapping(materialization_plan)
            if bool(plan.get("refused")):
                raise ResidualStoreError(
                    f"upstream materialization refused: {plan.get('reason') or 'unknown reason'}"
                )
            adapter = _adapter_from_replay_plan(plan)
            memory_family = _memory_family_from_replay_plan(plan)
            sidecar = _single_sidecar_from_replay_plan(plan, self._residual_store)
            ref = _single_residual_ref(sidecar)
            consumer = _consumer_from_replay_plan(plan, replay_consumer)
            decision = validate_residual_sidecar_replay(
                adapter=adapter,
                sidecar=sidecar,
                consumer=consumer,
                memory_family=memory_family,
                layer=ref.layer,
            )
            if decision.refused:
                raise ResidualStoreError(decision.reason)
            return (
                _ResidualSidecarReplayInput(
                    adapter=adapter,
                    sidecar=sidecar,
                    ref=ref,
                    consumer=consumer,
                    memory_family=memory_family,
                ),
                None,
            )
        except ResidualStoreError as exc:
            return None, _replay_metadata_with_refusal(base_replay_metadata, str(exc))

    def _metadata(
        self,
        dependencies: _TorchRuntimeDependencyReport | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "model_id": self.model_id,
            "local_files_only": self.local_files_only,
            "device": self.device,
            "requested_dtype": _dtype_request_label(self.requested_dtype),
            "resolved_dtype": self._resolved_dtype_label,
            "trust_remote_code": self.trust_remote_code,
            "residual_sidecar_replay_enabled": self.enable_residual_sidecar_replay,
        }
        if dependencies is not None:
            metadata["dependency_check"] = dependencies.to_metadata()
        return metadata

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


def _torch_runtime_dependency_report() -> _TorchRuntimeDependencyReport:
    missing = tuple(_missing_optional_packages(*_REQUIRED_TORCH_RUNTIME_PACKAGES))
    import_errors: list[str] = []
    if not missing:
        for module_name in _REQUIRED_TORCH_RUNTIME_MODULES:
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError as exc:
                import_errors.append(_module_not_found_reason(module_name, exc))
            except Exception as exc:  # pragma: no cover - defensive for broken local installs.
                import_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
    return _TorchRuntimeDependencyReport(
        required_packages=_REQUIRED_TORCH_RUNTIME_PACKAGES,
        required_modules=_REQUIRED_TORCH_RUNTIME_MODULES,
        missing_packages=missing,
        import_errors=tuple(import_errors),
    )


def _module_not_found_reason(module_name: str, exc: ModuleNotFoundError) -> str:
    missing_name = getattr(exc, "name", None)
    if missing_name and missing_name != module_name:
        return f"{module_name}: missing dependency {missing_name}"
    if missing_name == module_name:
        return f"{module_name}: module is not importable"
    return f"{module_name}: {exc}"


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


def _logits_processor_refusal_metadata(processors: Sequence[Any]) -> dict[str, Any]:
    processor_count = len(processors)
    refused = processor_count > 0
    reason = (
        "torch-runtime standard decode backend cannot apply decoder logits processors"
        if refused
        else None
    )
    return {
        "logits_processor_count": processor_count,
        "logits_processor_applied": False,
        "processors_refused": refused,
        "processors_refusal_reason": reason,
        "steering_applied": False,
        "steering_refused_reason": reason,
    }


def _residual_sidecar_requested(materialization_plan: Mapping[str, Any] | None) -> bool:
    if not isinstance(materialization_plan, Mapping):
        return False
    runtime_replay = materialization_plan.get("runtime_replay")
    runtime_contract = runtime_replay if isinstance(runtime_replay, Mapping) else {}
    requested_strategy = str(
        materialization_plan.get("requested_strategy")
        or runtime_contract.get("strategy")
        or materialization_plan.get("strategy")
        or ""
    )
    required_capability = str(runtime_contract.get("required_capability") or "")
    return (
        requested_strategy == RESIDUAL_SIDECAR_STRATEGY
        or required_capability == REPLAY_STRATEGY_CAPABILITIES[RESIDUAL_SIDECAR_STRATEGY]
    )


def _materialization_plan_mapping(materialization_plan: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(materialization_plan, Mapping):
        raise ResidualStoreError("invalid residual sidecar materialization plan")
    return materialization_plan


def _adapter_from_replay_plan(plan: Mapping[str, Any]) -> AdapterSessionMetadata:
    runtime_replay = _mapping(plan.get("runtime_replay"), "runtime_replay")
    adapter_scope = _mapping(
        plan.get("adapter_scope") or runtime_replay.get("adapter_scope"),
        "adapter_scope",
    )
    return AdapterSessionMetadata(
        model_id=_required_string(adapter_scope, "model_id"),
        tokenizer_id=_required_string(adapter_scope, "tokenizer_id"),
        model_revision=_required_string(adapter_scope, "model_revision"),
        adapter_family=_required_string(adapter_scope, "adapter_family"),
        hidden_size=_required_int(adapter_scope, "hidden_size"),
        boundary_layer=_required_int(adapter_scope, "boundary_layer"),
        kv_source_layer=_optional_int(adapter_scope.get("kv_source_layer"), "kv_source_layer"),
        kv_target_layer=_optional_int(adapter_scope.get("kv_target_layer"), "kv_target_layer"),
        insertion_family=_required_string(adapter_scope, "insertion_family"),
    )


def _memory_family_from_replay_plan(plan: Mapping[str, Any]) -> str:
    runtime_replay = plan.get("runtime_replay")
    runtime_contract = runtime_replay if isinstance(runtime_replay, Mapping) else {}
    memory_family = plan.get("memory_family") or runtime_contract.get("memory_family")
    if not isinstance(memory_family, str) or not memory_family:
        raise ResidualStoreError("residual sidecar plan missing memory_family")
    return memory_family


def _single_sidecar_from_replay_plan(
    plan: Mapping[str, Any],
    residual_store: ResidualStore,
) -> ResidualSidecarManifest:
    sidecars = plan.get("sidecars")
    if not isinstance(sidecars, Sequence) or isinstance(sidecars, (str, bytes)):
        raise ResidualStoreError("residual sidecar plan requires exactly one sidecar")
    if len(sidecars) != 1:
        raise ResidualStoreError(f"residual sidecar plan requires exactly one sidecar, got {len(sidecars)}")
    sidecar_payload = sidecars[0]
    if not isinstance(sidecar_payload, Mapping):
        raise ResidualStoreError("residual sidecar plan sidecar must be an object")
    if "schema_version" in sidecar_payload:
        return ResidualSidecarManifest.from_json(sidecar_payload)
    manifest_path = sidecar_payload.get("manifest_path")
    if manifest_path:
        return residual_store.load_manifest(str(manifest_path))
    return ResidualSidecarManifest.from_json(_sidecar_plan_payload(sidecar_payload))


def _sidecar_plan_payload(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(sidecar.get("scope"), "sidecar.scope")
    refs = sidecar.get("refs")
    if not isinstance(refs, list):
        raise ResidualStoreError("residual sidecar plan sidecar requires refs")
    return {
        "schema_version": 1,
        "artifact_id": _required_string(sidecar, "artifact_id"),
        "memory_family": _required_string(sidecar, "memory_family"),
        "model_id": _required_string(scope, "model_id"),
        "tokenizer_id": _required_string(scope, "tokenizer_id"),
        "model_revision": _required_string(scope, "model_revision"),
        "adapter_family": _required_string(scope, "adapter_family"),
        "insertion_family": _required_string(scope, "insertion_family"),
        "boundary_layer": _optional_int(scope.get("boundary_layer"), "boundary_layer"),
        "residual_layer": _optional_int(scope.get("residual_layer"), "residual_layer"),
        "kv_source_layer": _optional_int(scope.get("kv_source_layer"), "kv_source_layer"),
        "kv_target_layer": _optional_int(scope.get("kv_target_layer"), "kv_target_layer"),
        "hidden_size": _required_int(scope, "hidden_size"),
        "refs": refs,
        "provenance": dict(sidecar.get("provenance") or {}),
    }


def _single_residual_ref(sidecar: ResidualSidecarManifest) -> ResidualReference:
    refs = tuple(ref for ref in sidecar.refs if ref.kind in {"boundary_residual", "residual_stream"})
    if len(refs) != 1:
        raise ResidualStoreError(
            f"residual sidecar requires exactly one residual ref, got {len(refs)}"
        )
    ref = refs[0]
    _validate_single_vector_shape(ref.kind, ref.shape)
    return ref


def _validate_loaded_single_vector_residual(loaded: Any) -> None:
    if loaded.kind not in {"boundary_residual", "residual_stream"}:
        raise ResidualStoreError(f"unsupported residual sidecar tensor kind: {loaded.kind}")
    _validate_single_vector_shape(loaded.kind, loaded.shape)
    if not loaded.matches_manifest_hidden_size():
        raise ResidualStoreError(
            "residual sidecar tensor hidden_size mismatch: "
            f"tensor={loaded.shape[-1]} manifest={loaded.manifest.hidden_size}"
        )


def _validate_single_vector_shape(kind: str, shape: Sequence[int]) -> None:
    if not shape:
        raise ResidualStoreError(f"{kind} residual ref requires shape evidence")
    if len(shape) == 1:
        return
    leading = tuple(int(dim) for dim in shape[:-1])
    if any(dim != 1 for dim in leading):
        if kind == "residual_stream":
            raise ResidualStoreError("multi-row residual_stream replay is not supported")
        raise ResidualStoreError(f"{kind} replay requires a single-vector residual")


def _consumer_from_replay_plan(
    plan: Mapping[str, Any],
    replay_consumer: ReplayConsumerInput,
) -> ReplayConsumerInput:
    if replay_consumer is not None:
        return replay_consumer
    runtime_replay = plan.get("runtime_replay")
    if isinstance(runtime_replay, Mapping):
        consumer = runtime_replay.get("consumer")
        if consumer is not None:
            return consumer
    return None


def _replay_metadata_with_refusal(
    replay_metadata: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    metadata = dict(replay_metadata or {})
    refusal_reasons = list(metadata.get("refusal_reasons") or [])
    refusal_reasons.append(reason)
    metadata.update(
        {
            "version": metadata.get("version", 1),
            "backend": metadata.get("backend", "torch-runtime"),
            "refused": True,
            "ignored": True,
            "applied": False,
            "tensor_replay": False,
            "reason": "; ".join(refusal_reasons),
            "refusal_reasons": refusal_reasons,
        }
    )
    return metadata


def _replay_metadata_applied(replay_metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(replay_metadata)
    metadata.update(
        {
            "refused": False,
            "ignored": False,
            "applied": True,
            "tensor_replay": True,
            "reason": "runtime replay applied",
            "refusal_reasons": [],
        }
    )
    return metadata


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualStoreError(f"residual sidecar plan missing {label}")
    return value


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ResidualStoreError(f"residual sidecar plan missing {key}")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResidualStoreError(f"residual sidecar plan missing {key}")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResidualStoreError(f"residual sidecar plan field {key} must be an integer")
    return value


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
