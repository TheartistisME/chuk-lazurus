"""
PyTorch/CUDA-backed inference runtime.
"""

from __future__ import annotations

import inspect
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

from chuk_lazarus.session_retrieval.tier_policy import TierLabel

from .base import InferenceRuntime
from .types import LazarusBackend, ResidualState


class TorchInferenceRuntime(InferenceRuntime):
    """Inference runtime for Hugging Face causal LM models on CUDA."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        engine: str = "standard",
        family_type: Any | None = None,
        hot_budget_mib: int | None = None,
    ):
        super().__init__(model, tokenizer)
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._engine_mode = self._normalize_engine(engine)
        self._family_type = getattr(family_type, "value", family_type)
        self._hot_budget_bytes = (
            int(hot_budget_mib) * 1024 * 1024 if hot_budget_mib is not None else None
        )
        self._last_generation_path: str | None = None
        self._cached_kv_direct_forward_kwargs: dict[str, Any] | None = None
        # Session-keyed WARM residual cache — opt-in, injected by the server
        # engine. Only consulted by the residual_bounded_kv_direct path. When
        # ``None`` the runtime behaves exactly as it did before the cache was
        # introduced.
        self._session_cache: Any | None = None

    def attach_session_cache(self, session_cache: Any | None) -> None:
        """Install (or clear) the WARM residual session cache.

        The server engine is expected to construct the cache lazily once it
        knows the runtime engine mode is ``residual_bounded_kv_direct`` and
        then pin it here. Threading a pipeline attribute instead of a mutable
        module-global keeps the runtime self-contained and easy to exercise
        in isolation.
        """
        self._session_cache = session_cache

    @property
    def session_cache(self) -> Any | None:
        """Currently attached WARM residual session cache (may be ``None``)."""
        return self._session_cache

    @property
    def backend(self) -> LazarusBackend:
        return LazarusBackend.CUDA

    @property
    def engine_mode(self) -> str:
        """Configured generation engine."""
        return self._engine_mode

    @property
    def last_generation_path(self) -> str | None:
        """Observable label for the most recent generation path."""
        return self._last_generation_path

    @staticmethod
    def _normalize_engine(engine: str | None) -> str:
        normalized = str(engine or "standard").strip().lower()
        if normalized in {
            "standard",
            "kv_direct",
            "bounded_kv_direct",
            "residual_bounded_kv_direct",
        }:
            return normalized
        raise ValueError(f"Unsupported torch runtime engine '{engine}'.")

    def _set_generation_path(self, path: str | None) -> None:
        self._last_generation_path = path

    def _resolve_layers(self) -> list[Any]:
        """Locate transformer layers across common Hugging Face architectures.

        Supports plain causal LMs (model.model.layers, model.transformer.h,
        model.gpt_neox.layers, model.layers) and VLM wrappers whose language
        backbone is nested one level deeper (e.g. Gemma 4's
        model.model.language_model.layers, PaliGemma / Llava
        model.language_model.model.layers).
        """
        candidates = [
            # VLM wrappers first — their outer .model container also has .layers,
            # but the real decoder stack lives below .language_model.
            ("model", "language_model", "layers"),
            ("model", "language_model", "model", "layers"),
            ("language_model", "model", "layers"),
            ("language_model", "layers"),
            # Plain causal-LM shapes.
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            (None, "layers"),
        ]

        for path in candidates:
            target = self._model
            *outer, inner_name = path
            ok = True
            for step in outer:
                if step is None:
                    continue
                target = getattr(target, step, None)
                if target is None:
                    ok = False
                    break
            if not ok:
                continue

            layers = getattr(target, inner_name, None)
            if layers is None:
                continue
            # A bare tensor attribute called `.layers` on a leaf module is not
            # what we want — require an iterable container of modules.
            if not hasattr(layers, "__iter__") or not hasattr(layers, "__len__"):
                continue
            try:
                length = len(layers)
            except TypeError:
                continue
            if length == 0:
                continue
            return list(layers)

        raise ValueError(
            "Cannot resolve transformer layers for residual hooks. "
            "Expected model.model.layers, model.model.language_model.layers, "
            "model.language_model.model.layers, model.transformer.h, "
            "gpt_neox.layers, or model.layers."
        )

    def _tokenize_prompt(self, prompt: str) -> dict[str, Any]:
        """Tokenize a prompt and move tensors to CUDA explicitly."""
        if callable(getattr(self._tokenizer, "__call__", None)):
            batch = self._tokenizer(prompt, return_tensors="pt")
            return {
                name: tensor.to(self._device, non_blocking=True) for name, tensor in batch.items()
            }

        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        return {
            "input_ids": input_ids.to(self._device, non_blocking=True),
        }

    def _supports_flash_attention(self) -> bool:
        """Best-effort check for flash-attention compatibility."""
        config = getattr(self._model, "config", None)
        if config is None:
            return False

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_attention_heads", None)
            if hidden_size is not None and num_heads:
                try:
                    if hidden_size % num_heads == 0:
                        head_dim = hidden_size // num_heads
                except TypeError:
                    head_dim = None

        try:
            return head_dim is not None and int(head_dim) <= 256
        except (TypeError, ValueError):
            return False

    def _generation_context(self, total_window_tokens: int):
        """Select the fastest safe SDPA backend set for this model."""
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            return nullcontext()

        if self._device.type != "cuda":
            return nullcontext()

        backends = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
        if total_window_tokens <= 512 and self._supports_flash_attention():
            backends.insert(0, SDPBackend.FLASH_ATTENTION)

        return sdpa_kernel(backends)

    def _resolved_eos_token_ids(self) -> list[int]:
        """Resolve EOS ids, including chat-template-specific turn terminators."""
        eos_attr = getattr(self._tokenizer, "eos_token_id", None)
        eos_token_ids: list[int] = []
        if isinstance(eos_attr, list):
            eos_token_ids.extend(int(token_id) for token_id in eos_attr if token_id is not None)
        elif eos_attr is not None:
            eos_token_ids.append(int(eos_attr))

        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            extra = convert("<turn|>")
            if (
                isinstance(extra, int)
                and extra >= 0
                and extra != getattr(self._tokenizer, "unk_token_id", -1)
                and extra not in eos_token_ids
            ):
                eos_token_ids.append(extra)

        return eos_token_ids

    def _generation_kwargs(self, config, model_inputs: dict[str, Any], *, use_cache: bool) -> dict[str, Any]:
        """Build generate() kwargs while avoiding noisy ignored sampling flags.

        eos_token_id resolution:
          - if tokenizer carries a list-valued ``eos_token_id`` attr it wins;
          - otherwise look for additional Gemma-4 stop tokens (``<turn|>``)
            and build a multi-id list when found;
          - else fall back to the single ``tokenizer.eos_token_id``.
        Without the multi-id list, Gemma-4-it does not halt on its natural
        ``<end_of_turn>`` token and produces token-salad past the natural stop.
        """
        eos_token_ids = self._resolved_eos_token_ids()
        if not eos_token_ids:
            eos_token_id = None
        elif len(eos_token_ids) == 1:
            eos_token_id = eos_token_ids[0]
        else:
            eos_token_id = eos_token_ids
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": use_cache,
        }
        if config.temperature > 0:
            generation_kwargs["temperature"] = config.temperature
            generation_kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                generation_kwargs["top_k"] = config.top_k

        if "attention_mask" in model_inputs:
            generation_kwargs["attention_mask"] = model_inputs["attention_mask"]

        return {
            key: value
            for key, value in generation_kwargs.items()
            if value is not None
        }

    def _qwen_identity(self) -> tuple[str | None, list[str]]:
        config = getattr(self._model, "config", None)
        model_type = getattr(config, "model_type", None)
        raw_architectures = getattr(config, "architectures", None) or []
        architectures = [str(arch).lower() for arch in raw_architectures]
        return (str(model_type).lower() if model_type is not None else None, architectures)

    def _ensure_kv_direct_supported(self) -> None:
        if self._device.type != "cuda":
            raise NotImplementedError(
                "engine=kv_direct on the torch runtime is only implemented for CUDA devices. "
                f"Requested device={self._device}."
            )

        family_type = str(self._family_type).lower() if self._family_type is not None else None
        model_type, architectures = self._qwen_identity()
        qwen_markers = [family_type, model_type, *architectures]
        if not any(marker and "qwen" in marker for marker in qwen_markers):
            arch_desc = ", ".join(architectures) if architectures else "unknown"
            raise NotImplementedError(
                "engine=kv_direct on the torch runtime is currently implemented only for "
                "Qwen-family Hugging Face causal LMs. "
                f"family_type={family_type!r}, model_type={model_type!r}, architectures={arch_desc}."
            )

    def _kv_direct_forward_kwargs(self) -> dict[str, Any]:
        if self._cached_kv_direct_forward_kwargs is None:
            kwargs: dict[str, Any] = {}
            try:
                parameters = inspect.signature(self._model.forward).parameters
            except (TypeError, ValueError, AttributeError):
                parameters = {}
            if "logits_to_keep" in parameters:
                kwargs["logits_to_keep"] = 1
            self._cached_kv_direct_forward_kwargs = kwargs
        return dict(self._cached_kv_direct_forward_kwargs)

    def _extract_logits_and_cache(self, outputs: Any) -> tuple[Any, Any]:
        if hasattr(outputs, "logits"):
            return outputs.logits, getattr(outputs, "past_key_values", None)
        if isinstance(outputs, tuple) and outputs:
            logits = outputs[0]
            cache = outputs[1] if len(outputs) > 1 else None
            return logits, cache
        raise RuntimeError(
            "STRICT: torch kv_direct expected model outputs with logits and past_key_values."
        )

    def _build_text_only_position_ids(
        self, batch_size: int, start: int, length: int, device
    ):
        positions = self._torch.arange(start, start + length, device=device, dtype=self._torch.long)
        text_positions = positions.view(1, -1).expand(batch_size, -1)
        model_name = type(self._model).__name__
        if "Qwen3_5MoeForConditionalGeneration" in model_name:
            return self._torch.stack(
                [text_positions, text_positions, text_positions, text_positions], dim=0
            )
        return text_positions

    def _tensor_nbytes(self, tensor: Any) -> int:
        if not isinstance(tensor, self._torch.Tensor):
            return 0
        return int(tensor.numel() * tensor.element_size())

    def _cache_seq_length(self, past_key_values) -> int:
        if past_key_values is None:
            return 0
        if hasattr(past_key_values, "get_seq_length"):
            seq_len = past_key_values.get_seq_length()
            return int(seq_len.item()) if isinstance(seq_len, self._torch.Tensor) else int(seq_len)
        if isinstance(past_key_values, (list, tuple)) and past_key_values:
            first_layer = past_key_values[0]
            if isinstance(first_layer, (list, tuple)) and first_layer:
                first_tensor = first_layer[0]
                if isinstance(first_tensor, self._torch.Tensor):
                    return int(first_tensor.shape[-2])
        return 0

    def _cache_memory_breakdown(self, past_key_values) -> dict[str, int]:
        seq_length = self._cache_seq_length(past_key_values)
        variable_bytes = 0
        fixed_bytes = 0

        if past_key_values is None:
            return {
                "seq_length": 0,
                "variable_bytes": 0,
                "fixed_bytes": 0,
                "total_bytes": 0,
                "variable_bytes_per_token": 0,
            }

        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                layer_variable = 0
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if isinstance(keys, self._torch.Tensor) and keys.numel() > 0:
                    layer_variable += self._tensor_nbytes(keys)
                if isinstance(values, self._torch.Tensor) and values.numel() > 0:
                    layer_variable += self._tensor_nbytes(values)
                variable_bytes += layer_variable

                for attr in ("conv_states", "recurrent_states"):
                    fixed_bytes += self._tensor_nbytes(getattr(layer, attr, None))
        elif isinstance(past_key_values, (list, tuple)):
            for layer in past_key_values:
                if not isinstance(layer, (list, tuple)) or len(layer) < 2:
                    continue
                variable_bytes += self._tensor_nbytes(layer[0])
                variable_bytes += self._tensor_nbytes(layer[1])

        variable_bytes_per_token = variable_bytes // seq_length if seq_length > 0 else 0
        return {
            "seq_length": seq_length,
            "variable_bytes": variable_bytes,
            "fixed_bytes": fixed_bytes,
            "total_bytes": variable_bytes + fixed_bytes,
            "variable_bytes_per_token": variable_bytes_per_token,
        }

    def _max_hot_tokens_for_budget(self, past_key_values, budget_bytes: int) -> int:
        stats = self._cache_memory_breakdown(past_key_values)
        variable_bytes_per_token = stats["variable_bytes_per_token"]
        if variable_bytes_per_token <= 0:
            raise RuntimeError("Could not derive per-token KV bytes for bounded KV-direct.")
        available_variable_bytes = budget_bytes - stats["fixed_bytes"]
        if available_variable_bytes < variable_bytes_per_token:
            raise RuntimeError(
                "Budget is too small for even one KV token after fixed cache overhead. "
                f"budget_bytes={budget_bytes}, fixed_bytes={stats['fixed_bytes']}, "
                f"variable_bytes_per_token={variable_bytes_per_token}"
            )
        return max(1, available_variable_bytes // variable_bytes_per_token)

    def _set_cache_layer_seq_length(self, layer: Any, keep_last_tokens: int) -> None:
        if hasattr(layer, "cumulative_length"):
            current = getattr(layer, "cumulative_length")
            new_length = min(
                keep_last_tokens,
                int(current.item()) if isinstance(current, self._torch.Tensor) else int(current),
            )
            if isinstance(current, self._torch.Tensor):
                current.fill_(new_length)
            else:
                setattr(layer, "cumulative_length", new_length)
        if hasattr(layer, "cumulative_length_int"):
            layer.cumulative_length_int = min(int(layer.cumulative_length_int), keep_last_tokens)

    def _trim_cache_to_last_tokens(self, past_key_values, keep_last_tokens: int):
        if keep_last_tokens < 1:
            raise ValueError(f"keep_last_tokens must be >= 1, got {keep_last_tokens}")
        if past_key_values is None:
            return past_key_values

        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if (
                    isinstance(keys, self._torch.Tensor)
                    and keys.numel() > 0
                    and keys.shape[-2] > keep_last_tokens
                ):
                    layer.keys = keys[..., -keep_last_tokens:, :].contiguous()
                if (
                    isinstance(values, self._torch.Tensor)
                    and values.numel() > 0
                    and values.shape[-2] > keep_last_tokens
                ):
                    layer.values = values[..., -keep_last_tokens:, :].contiguous()
                self._set_cache_layer_seq_length(layer, keep_last_tokens)
            return past_key_values

        if isinstance(past_key_values, tuple):
            trimmed_layers = []
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    key_states, value_states = layer[:2]
                    trimmed_layers.append(
                        (
                            key_states[..., -keep_last_tokens:, :].contiguous(),
                            value_states[..., -keep_last_tokens:, :].contiguous(),
                        )
                    )
                else:
                    trimmed_layers.append(layer)
            return tuple(trimmed_layers)

        if isinstance(past_key_values, list):
            trimmed_layers = []
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    key_states, value_states = layer[:2]
                    trimmed_layers.append(
                        (
                            key_states[..., -keep_last_tokens:, :].contiguous(),
                            value_states[..., -keep_last_tokens:, :].contiguous(),
                        )
                    )
                else:
                    trimmed_layers.append(layer)
            return trimmed_layers

        raise RuntimeError(
            f"Unsupported past_key_values type for bounded trim: {type(past_key_values).__name__}"
        )

    def _enforce_last_token_budget(self, past_key_values, keep_last_tokens: int):
        current_seq = self._cache_seq_length(past_key_values)
        if current_seq <= keep_last_tokens:
            return past_key_values, 0
        evicted = current_seq - keep_last_tokens
        return self._trim_cache_to_last_tokens(past_key_values, keep_last_tokens), evicted

    def _resolved_hot_budget_bytes(self) -> int:
        if self._hot_budget_bytes is not None:
            return self._hot_budget_bytes
        return 1024 * 1024 * 1024

    def _decoded_suffix(self, generated_token_ids: list[int], decoded_so_far: str) -> str:
        decoded_text = self._tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        if decoded_text.startswith(decoded_so_far):
            return decoded_text[len(decoded_so_far) :]
        return decoded_text

    def _sample_next_token(self, logits, config):
        if config.temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        scores = logits.float() / config.temperature
        vocab_size = int(scores.shape[-1])

        if config.top_k is not None:
            top_k = min(int(config.top_k), vocab_size)
            if top_k > 0 and top_k < vocab_size:
                threshold = self._torch.topk(scores, top_k, dim=-1).values[..., -1, None]
                scores = scores.masked_fill(scores < threshold, float("-inf"))

        if config.top_p < 1.0:
            sorted_scores, sorted_indices = self._torch.sort(scores, descending=True, dim=-1)
            sorted_probs = self._torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = self._torch.cumsum(sorted_probs, dim=-1)
            sorted_to_remove = cumulative_probs > config.top_p
            sorted_to_remove[..., 1:] = sorted_to_remove[..., :-1].clone()
            sorted_to_remove[..., 0] = False
            to_remove = self._torch.zeros_like(scores, dtype=self._torch.bool)
            to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_to_remove)
            scores = scores.masked_fill(to_remove, float("-inf"))

        probs = self._torch.softmax(scores, dim=-1)
        return self._torch.multinomial(probs, num_samples=1)

    def _build_generation_result(self, *, input_length: int, token_ids, start_time: float, stop_reason):
        from ..generation import GenerationResult, GenerationStats

        if hasattr(token_ids, "to"):
            tokens_cpu = token_ids.to("cpu")
            token_list = (
                tokens_cpu[0].tolist() if getattr(tokens_cpu, "ndim", 1) > 1 else tokens_cpu.tolist()
            )
        else:
            token_list = list(token_ids)

        output_length = len(token_list)
        generated_text = self._tokenizer.decode(token_list, skip_special_tokens=True)
        gen_time = time.time() - start_time

        stats = GenerationStats(
            input_tokens=input_length,
            output_tokens=output_length,
            total_time_seconds=gen_time,
            tokens_per_second=output_length / gen_time if gen_time > 0 else 0,
        )
        return GenerationResult(text=generated_text, stats=stats, stop_reason=stop_reason)

    def _generate_standard(self, prompt: str, config):
        from ..generation import StopReason

        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        input_length = int(input_ids.shape[1])
        start_time = time.time()
        generation_kwargs = self._generation_kwargs(config, model_inputs, use_cache=True)

        total_window_tokens = input_length + config.max_new_tokens
        with self._torch.inference_mode(), self._generation_context(total_window_tokens):
            output_ids = self._model.generate(input_ids=input_ids, **generation_kwargs)

        new_tokens = output_ids[:, input_length:]
        output_length = int(new_tokens.shape[1])

        stop_reason = StopReason.MAX_TOKENS
        if output_length < config.max_new_tokens and output_length > 0:
            last_token = int(new_tokens[0, -1].item())
            eos_token_ids = set(self._resolved_eos_token_ids())
            stop_reason = StopReason.EOS if last_token in eos_token_ids else StopReason.STOP_TOKEN

        self._set_generation_path("torch.generate.standard")
        return self._build_generation_result(
            input_length=input_length,
            token_ids=new_tokens,
            start_time=start_time,
            stop_reason=stop_reason,
        )

    def _generate_kv_direct(self, prompt: str, config):
        from ..generation import StopReason

        self._ensure_kv_direct_supported()

        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        input_length = int(input_ids.shape[1])
        start_time = time.time()
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(token_id) for token_id in config.stop_tokens)
        stop_token_ids.update(eos_token_ids)
        generated_token_ids: list[int] = []
        stop_reason = StopReason.MAX_TOKENS

        total_window_tokens = input_length + config.max_new_tokens
        with self._torch.inference_mode(), self._generation_context(total_window_tokens):
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **self._kv_direct_forward_kwargs(),
            )
            logits, past_key_values = self._extract_logits_and_cache(outputs)
            if past_key_values is None:
                raise RuntimeError(
                    "STRICT: torch kv_direct expected the model prefill to return past_key_values."
                )

            next_token = self._sample_next_token(logits[:, -1, :], config)

            for step in range(config.max_new_tokens):
                token_id = int(next_token[0, 0].item())
                generated_token_ids.append(token_id)

                if token_id in stop_token_ids:
                    stop_reason = (
                        StopReason.EOS if token_id in eos_token_ids else StopReason.STOP_TOKEN
                    )
                    break

                if step + 1 >= config.max_new_tokens:
                    break

                attention_mask = self._torch.cat(
                    [
                        attention_mask,
                        self._torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                outputs = self._model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    **self._kv_direct_forward_kwargs(),
                )
                logits, past_key_values = self._extract_logits_and_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError(
                        "STRICT: torch kv_direct expected every decode step to return past_key_values."
                    )
                next_token = self._sample_next_token(logits[:, -1, :], config)

        self._set_generation_path("torch.kv_direct.past_key_values")
        return self._build_generation_result(
            input_length=input_length,
            token_ids=generated_token_ids,
            start_time=start_time,
            stop_reason=stop_reason,
        )

    def _generate_bounded_kv_direct(self, prompt: str, config):
        from ..generation import StopReason

        self._ensure_kv_direct_supported()

        budget_bytes = self._resolved_hot_budget_bytes()
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])
        total_seen_tokens = input_length
        start_time = time.time()
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(token_id) for token_id in config.stop_tokens)
        stop_token_ids.update(eos_token_ids)
        generated_token_ids: list[int] = []
        stop_reason = StopReason.MAX_TOKENS

        total_window_tokens = input_length + config.max_new_tokens
        with self._torch.inference_mode(), self._generation_context(total_window_tokens):
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=self._build_text_only_position_ids(
                    batch_size, 0, input_length, input_ids.device
                ),
                use_cache=True,
                **self._kv_direct_forward_kwargs(),
            )
            logits, past_key_values = self._extract_logits_and_cache(outputs)
            if past_key_values is None:
                raise RuntimeError(
                    "STRICT: bounded kv_direct expected the model prefill to return past_key_values."
                )

            max_hot_tokens = self._max_hot_tokens_for_budget(past_key_values, budget_bytes)
            prefill_hot_limit = max(1, max_hot_tokens - config.max_new_tokens)
            past_key_values, _ = self._enforce_last_token_budget(
                past_key_values, prefill_hot_limit
            )

            next_token = self._sample_next_token(logits[:, -1, :], config)

            for step in range(config.max_new_tokens):
                token_id = int(next_token[0, 0].item())
                generated_token_ids.append(token_id)

                if token_id in stop_token_ids:
                    stop_reason = (
                        StopReason.EOS if token_id in eos_token_ids else StopReason.STOP_TOKEN
                    )
                    break

                if step + 1 >= config.max_new_tokens:
                    break

                hot_seq_len = self._cache_seq_length(past_key_values)
                step_attention_mask = self._torch.ones(
                    (batch_size, hot_seq_len + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                outputs = self._model(
                    input_ids=next_token,
                    attention_mask=step_attention_mask,
                    position_ids=self._build_text_only_position_ids(
                        batch_size, total_seen_tokens, 1, next_token.device
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                    **self._kv_direct_forward_kwargs(),
                )
                logits, past_key_values = self._extract_logits_and_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError(
                        "STRICT: bounded kv_direct expected every decode step to return past_key_values."
                    )

                total_seen_tokens += 1
                past_key_values, _ = self._enforce_last_token_budget(
                    past_key_values, max_hot_tokens
                )
                next_token = self._sample_next_token(logits[:, -1, :], config)

        self._set_generation_path("torch.kv_direct.bounded_past_key_values")
        return self._build_generation_result(
            input_length=input_length,
            token_ids=generated_token_ids,
            start_time=start_time,
            stop_reason=stop_reason,
        )

    def _stream_residual_bounded_kv_direct(
        self,
        prompt: str,
        config,
        *,
        conversation_id: str | None = None,
    ):
        """Streaming wrapper around the residual-bounded generation path.

        Keeps the prefill and decode logic of
        ``_generate_residual_bounded_kv_direct`` intact and emits decoded
        suffix chunks as each new token arrives. Session-cache reuse follows
        the same semantics as the non-streaming path.
        """
        from ._torch_residual_bounded import (
            ResidualBoundedState,
            install_boundary_residual_capture_hook,
        )

        budget_bytes = self._resolved_hot_budget_bytes()
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(token_id) for token_id in config.stop_tokens)
        stop_token_ids.update(eos_token_ids)

        state = ResidualBoundedState(
            cold_token_ids=input_ids[0].tolist(),
            hot_window_start=0,
            hot_token_count=input_length,
        )

        max_hot_tokens_pre = self._estimate_max_hot_tokens(budget_bytes)
        prefill_hot_limit = max(1, max_hot_tokens_pre - config.max_new_tokens)

        # --- Session-cache lookup (mirrors the non-streaming path) ---------
        session_entry, resolved_session_id = self._resolve_session_entry(
            conversation_id, state.cold_token_ids
        )
        cache_hit: Any | None = None
        if session_entry is not None and self._session_cache is not None:
            from ._torch_residual_session import reuse_from_session_cache

            total_window_tokens = input_length + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                cache_hit = reuse_from_session_cache(
                    self,
                    entry=session_entry,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_hot_tokens_pre=max_hot_tokens_pre,
                    prefill_hot_limit=prefill_hot_limit,
                    config=config,
                )
            if cache_hit is None:
                self._session_cache.discard(session_entry.conversation_id)
                session_entry = None

        capture_layer = self._resolve_residual_capture_layer()
        capture_hook, capture_handle = install_boundary_residual_capture_hook(capture_layer)

        generated_token_ids: list[int] = []
        decoded_so_far = ""

        try:
            total_window_tokens = input_length + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                if cache_hit is not None:
                    past_key_values = cache_hit.past_key_values
                    logits = cache_hit.logits
                    state.boundary_residual = cache_hit.boundary_residual
                    state.boundary_residual_position = cache_hit.boundary_residual_position
                    state.hot_window_start = cache_hit.hot_window_start
                    state.hot_token_count = self._cache_seq_length(past_key_values)
                    total_seen_tokens = cache_hit.total_seen_tokens
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.session_reused"
                    )
                elif input_length <= max_hot_tokens_pre:
                    outputs = self._run_prefill_with_hook(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, 0, input_length, input_ids.device
                        ),
                        use_cache=True,
                    )
                    logits, past_key_values = self._extract_logits_and_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct stream: prefill returned no past_key_values."
                        )
                    if capture_hook.tensor is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct stream: capture hook produced no residual."
                        )
                    state.boundary_residual = capture_hook.tensor
                    state.boundary_residual_position = input_length - 1
                    past_key_values, evicted = self._enforce_last_token_budget(
                        past_key_values, prefill_hot_limit
                    )
                    state.hot_window_start = evicted
                    state.hot_token_count = input_length - evicted
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.prefill_hot"
                    )
                else:
                    split_idx = input_length - max_hot_tokens_pre
                    prefix_ids = input_ids[:, :split_idx]
                    prefix_mask = attention_mask[:, :split_idx]
                    self._run_prefill_with_hook(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, 0, split_idx, prefix_ids.device
                        ),
                        use_cache=False,
                    )
                    if capture_hook.tensor is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct stream: split prefill captured no residual."
                        )
                    state.boundary_residual = capture_hook.tensor
                    state.boundary_residual_position = split_idx - 1
                    state.prefill_was_split = True

                    suffix_tokens = input_ids[0, split_idx:].tolist()
                    logits, past_key_values, rebuilt_length = self._residual_seeded_prefill(
                        state=state,
                        boundary_residual=state.boundary_residual,
                        seed_position=split_idx - 1,
                        tokens_to_replay=suffix_tokens,
                        batch_size=batch_size,
                    )
                    past_key_values, evicted = self._enforce_last_token_budget(
                        past_key_values, prefill_hot_limit
                    )
                    state.hot_window_start = input_length - (rebuilt_length - evicted)
                    state.hot_token_count = rebuilt_length - evicted
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.prefill_split_rebuilt"
                    )

                next_token = self._sample_next_token(logits[:, -1, :], config)
                if cache_hit is None:
                    total_seen_tokens = input_length

                for step in range(config.max_new_tokens):
                    token_id = int(next_token[0, 0].item())
                    generated_token_ids.append(token_id)
                    state.cold_token_ids.append(token_id)

                    if token_id in stop_token_ids:
                        break

                    chunk = self._decoded_suffix(generated_token_ids, decoded_so_far)
                    if chunk:
                        decoded_so_far += chunk
                        yield chunk

                    if step + 1 >= config.max_new_tokens:
                        break

                    hot_seq_len = self._cache_seq_length(past_key_values)
                    step_attention_mask = self._torch.ones(
                        (batch_size, hot_seq_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    outputs = self._model(
                        input_ids=next_token,
                        attention_mask=step_attention_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, total_seen_tokens, 1, next_token.device
                        ),
                        past_key_values=past_key_values,
                        use_cache=True,
                        **self._kv_direct_forward_kwargs(),
                    )
                    logits, past_key_values = self._extract_logits_and_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct stream: decode step returned no past_key_values."
                        )

                    if capture_hook.tensor is not None:
                        state.boundary_residual = capture_hook.tensor
                        state.boundary_residual_position = total_seen_tokens

                    total_seen_tokens += 1
                    past_key_values, evicted_this_step = self._enforce_last_token_budget(
                        past_key_values, max_hot_tokens_pre
                    )
                    if evicted_this_step > 0:
                        state.hot_window_start += evicted_this_step
                    state.hot_token_count = self._cache_seq_length(past_key_values)
                    next_token = self._sample_next_token(logits[:, -1, :], config)
        finally:
            capture_handle.remove()

        final_chunk = self._decoded_suffix(generated_token_ids, decoded_so_far)
        if final_chunk:
            yield final_chunk
        self._set_generation_path(
            self._last_generation_path
            or "torch.residual_bounded_kv_direct.past_key_values"
        )

        if self._session_cache is not None and resolved_session_id:
            from ._torch_residual_session import write_entry_from_state
            from chuk_lazarus.server._residual_session_cache import ResidualSessionEntry

            entry = write_entry_from_state(
                entry_cls=ResidualSessionEntry,
                conversation_id=resolved_session_id,
                state=state,
                past_key_values=past_key_values,
                total_seen_tokens=total_seen_tokens,
            )
            self._session_cache.put(entry)

    def stream_generate(self, prompt: str, config, *, conversation_id: str | None = None):
        """Entry point for streaming generation.

        See :meth:`generate` for the ``conversation_id`` semantics.
        """
        if self._engine_mode == "standard":
            raise NotImplementedError("stream_generate is only used for non-standard torch engines.")

        self._ensure_kv_direct_supported()

        if self._engine_mode == "residual_bounded_kv_direct":
            # For streaming, fall back to the full residual-bounded pass and
            # emit decoded-suffix chunks tokenwise.  The residual-seeded
            # prefill is identical to the non-streaming path; streaming only
            # changes how the already-generated tokens are surfaced.
            yield from self._stream_residual_bounded_kv_direct(
                prompt, config, conversation_id=conversation_id
            )
            return

        budget_bytes = (
            self._resolved_hot_budget_bytes() if self._engine_mode == "bounded_kv_direct" else None
        )
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])
        total_seen_tokens = input_length
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(token_id) for token_id in config.stop_tokens)
        stop_token_ids.update(eos_token_ids)
        generated_token_ids: list[int] = []

        total_window_tokens = input_length + config.max_new_tokens
        with self._torch.inference_mode(), self._generation_context(total_window_tokens):
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=self._build_text_only_position_ids(
                    batch_size, 0, input_length, input_ids.device
                ),
                use_cache=True,
                **self._kv_direct_forward_kwargs(),
            )
            logits, past_key_values = self._extract_logits_and_cache(outputs)
            if past_key_values is None:
                raise RuntimeError(
                    "STRICT: torch kv_direct expected the model prefill to return past_key_values."
                )

            if budget_bytes is not None:
                max_hot_tokens = self._max_hot_tokens_for_budget(past_key_values, budget_bytes)
                prefill_hot_limit = max(1, max_hot_tokens - config.max_new_tokens)
                past_key_values, _ = self._enforce_last_token_budget(
                    past_key_values, prefill_hot_limit
                )
            else:
                max_hot_tokens = None

            next_token = self._sample_next_token(logits[:, -1, :], config)
            decoded_so_far = ""

            for step in range(config.max_new_tokens):
                token_id = int(next_token[0, 0].item())
                generated_token_ids.append(token_id)

                if token_id in stop_token_ids:
                    self._set_generation_path(
                        "torch.kv_direct.bounded_past_key_values"
                        if budget_bytes is not None
                        else "torch.kv_direct.past_key_values"
                    )
                    break

                chunk = self._decoded_suffix(generated_token_ids, decoded_so_far)
                if chunk:
                    decoded_so_far += chunk
                    yield chunk

                if step + 1 >= config.max_new_tokens:
                    break

                if budget_bytes is not None:
                    hot_seq_len = self._cache_seq_length(past_key_values)
                    step_attention_mask = self._torch.ones(
                        (batch_size, hot_seq_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                else:
                    step_attention_mask = self._torch.cat(
                        [
                            attention_mask,
                            self._torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
                    attention_mask = step_attention_mask

                outputs = self._model(
                    input_ids=next_token,
                    attention_mask=step_attention_mask,
                    position_ids=self._build_text_only_position_ids(
                        batch_size, total_seen_tokens, 1, next_token.device
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                    **self._kv_direct_forward_kwargs(),
                )
                logits, past_key_values = self._extract_logits_and_cache(outputs)
                if past_key_values is None:
                    raise RuntimeError(
                        "STRICT: torch kv_direct expected every decode step to return past_key_values."
                    )

                total_seen_tokens += 1
                if max_hot_tokens is not None:
                    past_key_values, _ = self._enforce_last_token_budget(
                        past_key_values, max_hot_tokens
                    )
                next_token = self._sample_next_token(logits[:, -1, :], config)

        self._set_generation_path(
            "torch.kv_direct.bounded_past_key_values"
            if budget_bytes is not None
            else "torch.kv_direct.past_key_values"
        )
        final_chunk = self._decoded_suffix(generated_token_ids, decoded_so_far)
        if final_chunk:
            yield final_chunk

    def generate(self, prompt: str, config, *, conversation_id: str | None = None):
        """Entry point for non-streaming generation.

        ``conversation_id`` is consumed only by the
        ``residual_bounded_kv_direct`` engine mode — it keys the session
        cache so multi-turn chats can skip the split-prefill forward after
        the first turn. All other engine modes ignore it.
        """
        self._set_generation_path(None)
        if self._engine_mode == "kv_direct":
            return self._generate_kv_direct(prompt, config)
        if self._engine_mode == "bounded_kv_direct":
            return self._generate_bounded_kv_direct(prompt, config)
        if self._engine_mode == "residual_bounded_kv_direct":
            return self._generate_residual_bounded_kv_direct(
                prompt, config, conversation_id=conversation_id
            )
        return self._generate_standard(prompt, config)

    # ------------------------------------------------------------------
    # Residual-backed bounded KV-direct  (ports the Gemma research
    # bounded_engine.BoundedKVEngine methodology to the torch Qwen path).
    # See chuk_lazarus.inference.backends._torch_residual_bounded for
    # hook / state helpers.
    # ------------------------------------------------------------------

    def _estimate_max_hot_tokens(self, budget_bytes: int) -> int:
        """Config-derived bound on hot tokens for the given byte budget.

        Used *before* the first forward so the prefill path can decide
        whether the prompt alone exceeds the hot budget and trigger the
        residual-seeded split-prefill rebuild.
        """
        from ._torch_residual_bounded import estimate_kv_bytes_per_token

        bytes_per_token = estimate_kv_bytes_per_token(self._model)
        if bytes_per_token <= 0:
            raise RuntimeError(
                "Config-derived KV bytes/token is non-positive — cannot size the hot window."
            )
        return max(1, int(budget_bytes) // int(bytes_per_token))

    def _resolve_residual_capture_layer(self) -> Any:
        """Pick the layer to capture the boundary residual from.

        The *last* transformer layer's post-output hidden state is the
        Markov-sufficient pre-final-norm residual — identical to the
        ``residual_last = h[:, -1:, :]`` capture in MLX ``KVDirectGenerator._prefill_core``.
        """
        layers = self._resolve_layers()
        return layers[-1]

    def _run_prefill_with_hook(
        self,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        use_cache: bool,
        pre_hook_handle: Any | None = None,
    ) -> Any:
        """Single-prefill forward pass returning outputs.

        The boundary-residual capture hook is expected to be installed by the
        caller — this helper only wires the generic HF forward call.
        """
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            **self._kv_direct_forward_kwargs(),
        )

    def _residual_seeded_prefill(
        self,
        state,
        boundary_residual: Any,
        seed_position: int,
        tokens_to_replay: list[int],
        batch_size: int,
    ):
        """Rebuild a bounded KV store anchored on a residual seed.

        Mirrors ``KVDirectGenerator.prefill_from_layer`` (MLX) — the
        boundary residual occupies position 0 of the rebuilt window via a
        layer-0 ``forward_pre_hook`` that overwrites ``hidden_states[:, 0, :]``
        on the multi-position prefill call.  Subsequent positions are the
        last-N cold tokens replayed at absolute positions
        ``seed_position+1 .. seed_position+len(tokens_to_replay)`` so RoPE
        sees contiguous positions continuing from the evicted prefix.

        Returns ``(logits, past_key_values, rebuilt_length)``.
        """
        from ._torch_residual_bounded import install_layer0_residual_seed_hook

        if not tokens_to_replay:
            raise RuntimeError(
                "residual_bounded_kv_direct: cannot rebuild with zero replay tokens."
            )

        layers = self._resolve_layers()
        first_layer = layers[0]

        seed_id = (
            getattr(self._tokenizer, "bos_token_id", None)
            or getattr(self._tokenizer, "pad_token_id", None)
            or 0
        )
        seeded_ids = [seed_id, *tokens_to_replay]
        seq_len = len(seeded_ids)
        device = boundary_residual.device

        input_ids = self._torch.tensor([seeded_ids], dtype=self._torch.long, device=device)
        attention_mask = self._torch.ones((batch_size, seq_len), dtype=self._torch.long, device=device)
        position_ids = self._build_text_only_position_ids(
            batch_size, seed_position, seq_len, device
        )

        _seed_hook, seed_handle = install_layer0_residual_seed_hook(
            first_layer, boundary_residual
        )
        try:
            outputs = self._run_prefill_with_hook(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
        finally:
            seed_handle.remove()

        logits, past_key_values = self._extract_logits_and_cache(outputs)
        if past_key_values is None:
            raise RuntimeError(
                "residual_bounded_kv_direct: seeded prefill did not return past_key_values."
            )
        return logits, past_key_values, seq_len

    def _resolve_session_entry(self, conversation_id: str | None, prompt_token_ids):
        """Look up the session cache entry for ``conversation_id``.

        Falls back to a prefix-match scan when no explicit id is supplied.
        Returns ``(entry, resolved_id)`` where ``resolved_id`` is the
        canonical key that should be used when writing the entry back after
        generation finishes. ``entry`` is ``None`` on miss.
        """
        cache = self._session_cache
        if cache is None:
            return None, None
        from chuk_lazarus.server._residual_session_cache import resolve_conversation_id

        resolved_id = resolve_conversation_id(
            explicit_id=conversation_id,
            prompt_token_ids=prompt_token_ids,
        )
        if resolved_id is None:
            return None, None

        # Explicit id beats prefix-match: if the client gave us a user, honor
        # it even when the prompt text is brand-new (cache miss is fine).
        entry = cache.get(resolved_id)
        if entry is not None:
            return entry, resolved_id

        # Anonymous / prefix-hash fallback — scan every entry for the longest
        # cold-token prefix that matches our prompt.
        if conversation_id is None:
            entry = cache.find_prefix_match(prompt_token_ids)
            if entry is not None:
                return entry, entry.conversation_id
        return None, resolved_id

    def _generate_residual_bounded_kv_direct(
        self,
        prompt: str,
        config,
        *,
        conversation_id: str | None = None,
    ):
        """Bounded KV-direct with residual-backed rebuild on the Qwen path.

        Matches the three-tier semantics from
        ``chuk_lazarus.inference.context.research.bounded_engine``:

          HOT   — ``past_key_values`` bounded to the configured budget.
          WARM  — the ``boundary_residual`` refreshed by the forward hook
                  every step; Markov-sufficient statistic for everything
                  attended so far.
          COLD  — ``state.cold_token_ids``, never evicted.

        Eviction path:
          * If the prompt itself exceeds the hot budget, split prefill into
            ``prefix`` (captures residual via a cache-less forward) and
            ``suffix`` (rebuilt via a residual-seeded prefill).  The final
            hot state holds a window anchored on the residual seed.
          * Each decode step trims ``past_key_values`` to the last
            ``max_hot_tokens`` tokens — but the residual snapshot is kept
            fresh from the capture hook, so the evicted prefix is never
            forgotten; a later ``_residual_seeded_prefill`` call can rebuild
            from it.

        Session-cache reuse (multi-turn speedup):
          * When ``conversation_id`` matches a cached entry whose
            ``cold_token_ids`` is a prefix of the new prompt, we skip the
            cold prefill entirely and only run a forward over the delta
            tokens on top of the cached hot KV. The whole split-prefill
            rebuild is avoided on every turn after the first.

        GPU-only — ``_ensure_kv_direct_supported`` enforces CUDA.
        """
        from ..generation import StopReason
        from ._torch_residual_bounded import (
            ResidualBoundedState,
            install_boundary_residual_capture_hook,
        )

        self._ensure_kv_direct_supported()

        budget_bytes = self._resolved_hot_budget_bytes()
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = self._torch.ones_like(input_ids, dtype=self._torch.long)

        batch_size = int(input_ids.shape[0])
        input_length = int(input_ids.shape[1])
        start_time = time.time()
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(token_id) for token_id in config.stop_tokens)
        stop_token_ids.update(eos_token_ids)

        state = ResidualBoundedState(
            cold_token_ids=input_ids[0].tolist(),
            hot_window_start=0,
            hot_token_count=input_length,
        )

        max_hot_tokens_pre = self._estimate_max_hot_tokens(budget_bytes)
        prefill_hot_limit = max(1, max_hot_tokens_pre - config.max_new_tokens)

        # --- Session-cache lookup -------------------------------------------
        session_entry, resolved_session_id = self._resolve_session_entry(
            conversation_id, state.cold_token_ids
        )
        cache_hit: Any | None = None
        if session_entry is not None and self._session_cache is not None:
            from ._torch_residual_session import reuse_from_session_cache

            total_window_tokens = input_length + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                cache_hit = reuse_from_session_cache(
                    self,
                    entry=session_entry,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_hot_tokens_pre=max_hot_tokens_pre,
                    prefill_hot_limit=prefill_hot_limit,
                    config=config,
                )
            if cache_hit is None:
                # Prefix mismatch — stale entry, drop it so the cold path
                # below repopulates the cache.
                self._session_cache.discard(session_entry.conversation_id)
                session_entry = None

        capture_layer = self._resolve_residual_capture_layer()
        capture_hook, capture_handle = install_boundary_residual_capture_hook(capture_layer)

        generated_token_ids: list[int] = []
        stop_reason = StopReason.MAX_TOKENS

        try:
            total_window_tokens = input_length + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                if cache_hit is not None:
                    # --- Session-cache reuse path --------------------------
                    past_key_values = cache_hit.past_key_values
                    logits = cache_hit.logits
                    state.boundary_residual = cache_hit.boundary_residual
                    state.boundary_residual_position = cache_hit.boundary_residual_position
                    state.hot_window_start = cache_hit.hot_window_start
                    state.hot_token_count = self._cache_seq_length(past_key_values)
                    total_seen_tokens = cache_hit.total_seen_tokens
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.session_reused"
                    )
                elif input_length <= max_hot_tokens_pre:
                    # --- HOT prefill path ----------------------------------
                    outputs = self._run_prefill_with_hook(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, 0, input_length, input_ids.device
                        ),
                        use_cache=True,
                    )
                    logits, past_key_values = self._extract_logits_and_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct: prefill returned no past_key_values."
                        )
                    if capture_hook.tensor is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct: capture hook produced no residual."
                        )
                    state.boundary_residual = capture_hook.tensor
                    state.boundary_residual_position = input_length - 1

                    past_key_values, evicted = self._enforce_last_token_budget(
                        past_key_values, prefill_hot_limit
                    )
                    state.hot_window_start = evicted
                    state.hot_token_count = input_length - evicted
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.prefill_hot"
                    )
                else:
                    # --- SPLIT prefill: residual seed on last-N cold tokens -
                    split_idx = input_length - max_hot_tokens_pre
                    prefix_ids = input_ids[:, :split_idx]
                    prefix_mask = attention_mask[:, :split_idx]
                    # First pass: cache-less forward over the prefix, purely
                    # to capture the boundary residual at the last prefix token.
                    self._run_prefill_with_hook(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, 0, split_idx, prefix_ids.device
                        ),
                        use_cache=False,
                    )
                    if capture_hook.tensor is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct: split prefill captured no residual."
                        )
                    state.boundary_residual = capture_hook.tensor
                    state.boundary_residual_position = split_idx - 1
                    state.prefill_was_split = True

                    suffix_tokens = input_ids[0, split_idx:].tolist()
                    logits, past_key_values, rebuilt_length = self._residual_seeded_prefill(
                        state=state,
                        boundary_residual=state.boundary_residual,
                        seed_position=split_idx - 1,
                        tokens_to_replay=suffix_tokens,
                        batch_size=batch_size,
                    )
                    past_key_values, evicted = self._enforce_last_token_budget(
                        past_key_values, prefill_hot_limit
                    )
                    state.hot_window_start = input_length - (rebuilt_length - evicted)
                    state.hot_token_count = rebuilt_length - evicted
                    self._set_generation_path(
                        "torch.residual_bounded_kv_direct.prefill_split_rebuilt"
                    )

                next_token = self._sample_next_token(logits[:, -1, :], config)

                if cache_hit is None:
                    total_seen_tokens = input_length

                for step in range(config.max_new_tokens):
                    token_id = int(next_token[0, 0].item())
                    generated_token_ids.append(token_id)
                    state.cold_token_ids.append(token_id)

                    if token_id in stop_token_ids:
                        stop_reason = (
                            StopReason.EOS if token_id in eos_token_ids else StopReason.STOP_TOKEN
                        )
                        break

                    if step + 1 >= config.max_new_tokens:
                        break

                    hot_seq_len = self._cache_seq_length(past_key_values)
                    step_attention_mask = self._torch.ones(
                        (batch_size, hot_seq_len + 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    outputs = self._model(
                        input_ids=next_token,
                        attention_mask=step_attention_mask,
                        position_ids=self._build_text_only_position_ids(
                            batch_size, total_seen_tokens, 1, next_token.device
                        ),
                        past_key_values=past_key_values,
                        use_cache=True,
                        **self._kv_direct_forward_kwargs(),
                    )
                    logits, past_key_values = self._extract_logits_and_cache(outputs)
                    if past_key_values is None:
                        raise RuntimeError(
                            "residual_bounded_kv_direct: decode step returned no past_key_values."
                        )

                    # Refresh the residual snapshot BEFORE trimming so it
                    # remains a Markov statistic for everything the model
                    # has just attended to (including soon-to-be-evicted
                    # tokens).  Capture hook already updated ``tensor`` in
                    # the forward pass above.
                    if capture_hook.tensor is not None:
                        state.boundary_residual = capture_hook.tensor
                        state.boundary_residual_position = total_seen_tokens

                    total_seen_tokens += 1

                    before_len = self._cache_seq_length(past_key_values)
                    past_key_values, evicted_this_step = self._enforce_last_token_budget(
                        past_key_values, max_hot_tokens_pre
                    )
                    if evicted_this_step > 0:
                        state.hot_window_start += evicted_this_step
                        state.hot_token_count = self._cache_seq_length(past_key_values)
                    else:
                        state.hot_token_count = before_len

                    next_token = self._sample_next_token(logits[:, -1, :], config)
        finally:
            capture_handle.remove()

        self._set_generation_path(
            self._last_generation_path
            or "torch.residual_bounded_kv_direct.past_key_values"
        )

        # --- Write back to the session cache -------------------------------
        # On both cold-miss and cache-hit paths we refresh the cached entry
        # with the final state so the NEXT turn can take the reuse path.
        if self._session_cache is not None and resolved_session_id:
            from ._torch_residual_session import write_entry_from_state
            from chuk_lazarus.server._residual_session_cache import ResidualSessionEntry

            entry = write_entry_from_state(
                entry_cls=ResidualSessionEntry,
                conversation_id=resolved_session_id,
                state=state,
                past_key_values=past_key_values,
                total_seen_tokens=total_seen_tokens,
            )
            self._session_cache.put(entry)

        return self._build_generation_result(
            input_length=input_length,
            token_ids=generated_token_ids,
            start_time=start_time,
            stop_reason=stop_reason,
        )

    def extract_residual_state(self, prompt: str, layer_index: int) -> ResidualState:
        layers = self._resolve_layers()
        layer = layers[layer_index]
        captured: dict[str, Any] = {}

        def capture_hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["tensor"] = hidden[:, -1, :].detach().to("cpu")

        handle = layer.register_forward_hook(capture_hook)
        try:
            model_inputs = self._tokenize_prompt(prompt)
            with self._torch.inference_mode():
                self._model(**model_inputs, use_cache=False)
        finally:
            handle.remove()

        residual = captured.get("tensor")
        if residual is None:
            raise RuntimeError(f"Failed to capture residual state for layer {layer_index}.")

        return ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=layer_index,
            tensor=residual,
            sequence_length=int(model_inputs["input_ids"].shape[1]),
            hidden_size=int(residual.shape[-1]),
            dtype=str(residual.dtype).replace("torch.", ""),
            device=str(self._device),
        )

    def generate_with_residual(self, prompt: str, residual_state: ResidualState, config):
        layers = self._resolve_layers()
        layer = layers[residual_state.layer_index]
        residual = residual_state.tensor.to(self._device, non_blocking=True)
        injected_once = False

        def inject_hook(_module, args, kwargs):
            nonlocal injected_once
            if injected_once or not args:
                return args, kwargs

            hidden_states = args[0]
            injected_hidden = hidden_states.clone()
            injected_hidden[:, -1, :] = residual
            injected_once = True
            new_args = (injected_hidden, *args[1:])
            return new_args, kwargs

        handle = layer.register_forward_pre_hook(inject_hook, with_kwargs=True)
        try:
            model_inputs = self._tokenize_prompt(prompt)
            input_ids = model_inputs["input_ids"]
            input_length = int(input_ids.shape[1])
            start_time = time.time()
            generation_kwargs = self._generation_kwargs(config, model_inputs, use_cache=False)
            # Disable cache accumulation while the injected hidden state is active.
            generation_kwargs["past_key_values"] = None

            total_window_tokens = input_length + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                output_ids = self._model.generate(input_ids=input_ids, **generation_kwargs)

            new_tokens = output_ids[:, input_length:]
            output_length = int(new_tokens.shape[1])
            new_tokens_cpu = new_tokens.to("cpu")
            generated_text = self._tokenizer.decode(
                new_tokens_cpu[0].tolist(), skip_special_tokens=True
            )
            gen_time = time.time() - start_time
        finally:
            handle.remove()

        from ..generation import GenerationResult, GenerationStats, StopReason

        stop_reason = StopReason.MAX_TOKENS if output_length >= config.max_new_tokens else StopReason.STOP_TOKEN
        stats = GenerationStats(
            input_tokens=input_length,
            output_tokens=output_length,
            total_time_seconds=gen_time,
            tokens_per_second=output_length / gen_time if gen_time > 0 else 0,
        )
        return GenerationResult(text=generated_text, stats=stats, stop_reason=stop_reason)

    def generate_with_residual_prefill_seeded(
        self, prompt: str, residual_state: ResidualState, config
    ):
        """Canonical two-stage prefill: seed position 0 with the boundary residual.

        Mirrors chrishayuk/chuk-lazurus ``prefill_to_layer(initial_residual=boundary)``
        + ``prefill_from_layer``. Canonically the boundary would be prepended
        at the embedding so every subsequent token attends to it through ALL
        transformer layers and the KV cache is anchored on a coherent
        donor-plus-prompt stream — which is what stops cross-session retrieval
        from producing token salad. See MLX mirror at
        ``src/chuk_lazarus/inference/context/kv_generator.py`` lines 182-226.

        PyTorch implementation note: we cannot use ``inputs_embeds=`` on
        Gemma-4 because ``transformers/models/gemma4/modeling_gemma4.py``
        ``get_per_layer_inputs`` broadcasts
        ``inputs_embeds[:, :, None, :]`` against
        ``embed_tokens.weight[None, None, :, :]`` (vocab≈256k, H=8192),
        allocating ~253 GiB on a prefill and deterministic OOM on a 32 GiB GPU.
        Instead we use the ``input_ids`` path: allocate a seed position
        (BOS/pad/0) at the front of the prompt and install a
        ``forward_pre_hook`` on ``layers[0]`` that, on the prefill call
        (``hidden_states.shape[1] > 1``), overwrites
        ``hidden_states[:, 0, :]`` with the boundary residual. This is
        equivalent to prepending at the embedding because the seed token's
        id never influences subsequent computation — its post-embedding
        hidden state is replaced wholesale before layer[0] runs, and the
        causal mask lets every real-prompt position attend to it through
        ALL layers. The hook is a no-op on decode steps
        (``shape[1] == 1``) and will NOT latch on a decode call.
        """
        from ..generation import GenerationResult, GenerationStats, StopReason

        # 1. Tokenise. This gives (1, S) input_ids on device.
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]                  # (1, S)
        input_length = int(input_ids.shape[1])

        # 2. Pick a seed token. The concrete id does NOT matter — our pre-hook
        #    on layers[0] replaces its post-embedding hidden state with the
        #    boundary before layer[0] runs. We prefer BOS so any downstream
        #    tokenizer-aware path (attention_mask auto-build, position_ids)
        #    treats it as a real start token rather than padding.
        seed_id = (
            getattr(self._tokenizer, "bos_token_id", None)
            or getattr(self._tokenizer, "pad_token_id", None)
            or 0
        )
        seed_tensor = self._torch.tensor(
            [[seed_id]], dtype=input_ids.dtype, device=input_ids.device
        )
        seeded_input_ids = self._torch.cat([seed_tensor, input_ids], dim=1)  # (1, S+1)
        S_total = int(seeded_input_ids.shape[1])

        # 3. Coerce boundary to 1-D (H,) on the model's device and dtype.
        #    We store it 1-D because the hook will assign it to
        #    hidden_states[:, 0, :].
        boundary = residual_state.tensor
        while boundary.dim() > 1:
            # Collapse leading singleton dims: (1, H), (1, 1, H), etc.
            if boundary.shape[0] == 1:
                boundary = boundary.squeeze(0)
            else:
                raise RuntimeError(
                    "STRICT: boundary has unexpected leading non-singleton dim "
                    f"{tuple(boundary.shape)}"
                )
        # Resolve model hidden size via the embedding weight (authoritative for
        # Gemma-4).
        embed_weight = self._model.get_input_embeddings().weight
        target_dtype = embed_weight.dtype
        target_device = embed_weight.device
        boundary = boundary.to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        if boundary.shape[-1] != embed_weight.shape[-1]:
            raise RuntimeError(
                "STRICT: boundary hidden size "
                f"{boundary.shape[-1]} != model hidden size {embed_weight.shape[-1]}"
            )

        # 4. Register a forward_pre_hook on layers[0] that, on the first
        #    prefill call (hidden_states.shape[1] > 1), replaces
        #    hidden_states[:, 0, :] with the boundary residual. Subsequent
        #    decode calls (shape[1] == 1) are no-ops and MUST NOT latch the
        #    injected_once flag.
        layers = self._resolve_layers()
        first_layer = layers[0]
        injected_once = False

        def seed_hook(_module, args, kwargs):
            nonlocal injected_once
            if injected_once or not args:
                return args, kwargs
            hidden_states = args[0]
            # Some HF layers pass hidden_states in kwargs; defend against that.
            if not hasattr(hidden_states, "shape") or hidden_states.ndim < 3:
                return args, kwargs
            if hidden_states.shape[1] <= 1:
                # Decode step — do nothing, do NOT latch.
                return args, kwargs
            # Prefill step: replace position 0 with the boundary.
            new_hidden = hidden_states.clone()
            new_hidden[:, 0, :] = boundary
            injected_once = True
            return (new_hidden, *args[1:]), kwargs

        seed_handle = first_layer.register_forward_pre_hook(
            seed_hook, with_kwargs=True
        )

        start_time = time.time()
        try:
            # 5. Build attention_mask covering S_total positions. Pass through
            #    the existing _generation_kwargs helper to keep eos-list
            #    handling for Gemma-4 <turn|> consistent with every other
            #    generate path.
            attention_mask = self._torch.ones(
                (seeded_input_ids.shape[0], S_total),
                dtype=self._torch.long,
                device=seeded_input_ids.device,
            )
            generation_kwargs = self._generation_kwargs(
                config, {"attention_mask": attention_mask}, use_cache=True
            )
            generation_kwargs["attention_mask"] = attention_mask

            total_window_tokens = S_total + config.max_new_tokens
            with self._torch.inference_mode(), self._generation_context(total_window_tokens):
                output_ids = self._model.generate(
                    input_ids=seeded_input_ids,
                    **generation_kwargs,
                )
        finally:
            seed_handle.remove()

        # 6. Slice new tokens. When input_ids is used, HF's generate returns
        #    the full sequence (seeded prompt + generated); strip the seeded
        #    prefix.
        new_tokens = output_ids[:, S_total:]
        output_length = int(new_tokens.shape[1])
        new_tokens_cpu = new_tokens.to("cpu")
        generated_text = self._tokenizer.decode(
            new_tokens_cpu[0].tolist(), skip_special_tokens=True
        )
        gen_time = time.time() - start_time

        # 7. Stop-reason bookkeeping (mirrors generate_with_residual).
        stop_reason = StopReason.MAX_TOKENS
        if output_length < config.max_new_tokens and output_length > 0:
            last_token = int(new_tokens[0, -1].item())
            eos_token_id = self._tokenizer.eos_token_id
            if isinstance(eos_token_id, list) and last_token in eos_token_id:
                stop_reason = StopReason.EOS
            elif eos_token_id is not None and last_token == eos_token_id:
                stop_reason = StopReason.EOS
            else:
                stop_reason = StopReason.STOP_TOKEN

        stats = GenerationStats(
            input_tokens=input_length,
            output_tokens=output_length,
            total_time_seconds=gen_time,
            tokens_per_second=output_length / gen_time if gen_time > 0 else 0.0,
        )
        return GenerationResult(text=generated_text, stats=stats, stop_reason=stop_reason)

    def _kv_direct_patched_forward_factory(
        self,
        fire_counter: dict[str, Any],
        propagation_state: dict[str, Any],
    ):
        """Produce a replacement ``target_self_attn.forward`` that injects the
        archived K/V prefix and applies the axis-4 tier mask on the prefix
        portion of the attention logits.

        Targets **Gemma-4** (KV-sharing architecture). The target layer must
        be a non-shared layer with ``store_full_length_kv=True`` — layer 14
        for Gemma-4-E2B (full-attention master). Combined (archived + fresh)
        K/V is stamped into ``shared_kv_states[target_layer.layer_idx]`` so
        all downstream full-attention layers (19, 24, 29, 34) inherit the
        prefix via Gemma-4's native sharing mechanism.

        Stamped module attributes read per forward:

        * ``_lazarus_archived_K`` — ``(1, n_kv_heads, N, head_dim)``, raw
          ``k_proj(residual)`` output: pre-norm, pre-rotary.
        * ``_lazarus_archived_V`` — ``(1, n_kv_heads, N, head_dim)``, raw
          ``v_proj(residual)`` output: pre-norm.
        * ``_lazarus_mask_inputs`` — :class:`AttentionTierMaskInputs`.

        Semantics:

        * Fresh K/V follow the normal path: ``k_proj`` → ``k_norm`` →
          rotary → transpose → cache update.
        * Archived K/V are aligned with fresh by applying ``k_norm`` /
          ``v_norm`` before concat. Archived keys then receive a
          position-0 rotary transform so the injected prefix re-enters
          the same rotary domain the live layer expects.
        * Combined K/V is written to ``shared_kv_states`` when
          ``store_full_length_kv`` is set, so downstream shared layers
          see the prefix too.
        * Tier mask applies to the ``[0, N)`` slice of attention logits:
          HOT → identity, WARM → subtractive penalty, COLD → ``-inf``.
        """
        torch = self._torch
        nn = torch.nn
        from transformers.models.gemma4.modeling_gemma4 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        def _resolve_shared_kv_states(shared_kv_states, past_key_values):
            if shared_kv_states is not None:
                return shared_kv_states
            if past_key_values is None:
                return None
            if not hasattr(past_key_values, "shared_layers"):
                past_key_values.shared_layers = {}
            return past_key_values.shared_layers

        def _prepare_archived_prefix(module_self, cos, sin, key_states, value_states):
            archived_K_raw = getattr(module_self, "_lazarus_archived_K", None)
            archived_V_raw = getattr(module_self, "_lazarus_archived_V", None)
            if archived_K_raw is None or archived_V_raw is None:
                return None, None

            prefix_attn_module = propagation_state.get("prefix_attn_module", module_self)
            ak = archived_K_raw.to(device=key_states.device, dtype=key_states.dtype)
            av = archived_V_raw.to(device=value_states.device, dtype=value_states.dtype)

            # The archived K/V were materialized through the source attention
            # layer, so each consumer re-enters that layer's normalized /
            # rotary domain before concat.
            ak = prefix_attn_module.k_norm(ak.transpose(1, 2))  # (1, N, Hkv, hd)
            av = prefix_attn_module.v_norm(av.transpose(1, 2))  # (1, N, Hkv, hd)

            N = int(ak.shape[1])
            cos0 = cos[:, 0:1, :].expand(cos.shape[0], N, cos.shape[-1])
            sin0 = sin[:, 0:1, :].expand(sin.shape[0], N, sin.shape[-1])
            ak = apply_rotary_pos_emb(ak, cos0, sin0, unsqueeze_dim=2)
            ak = ak.transpose(1, 2)  # → (1, Hkv, N, hd)
            av = av.transpose(1, 2)  # V gets no rotary
            return ak, av

        def _apply_selected_head_prefix_mask(prefix_slice):
            selected_head_indices = propagation_state.get("selected_head_indices", ())
            if not selected_head_indices:
                return prefix_slice

            num_heads = int(prefix_slice.shape[1])
            selected_mask = torch.zeros(num_heads, dtype=torch.bool, device=prefix_slice.device)
            selected_mask[list(selected_head_indices)] = True
            if bool(selected_mask.all()):
                return prefix_slice

            blocked_prefix = torch.full_like(prefix_slice, torch.finfo(prefix_slice.dtype).min)
            return torch.where(
                selected_mask.view(1, num_heads, 1, 1),
                prefix_slice,
                blocked_prefix,
            )

        def _pad_attention_mask_for_prefix(attention_mask, fresh_kv_len: int, prefix_len: int):
            if attention_mask is None or prefix_len <= 0 or attention_mask.ndim != 4:
                return attention_mask

            expected_kv_len = fresh_kv_len + prefix_len
            current_kv_len = int(attention_mask.shape[-1])
            if current_kv_len == expected_kv_len:
                return attention_mask[:, :, :, :expected_kv_len]

            causal_mask = attention_mask[:, :, :, :fresh_kv_len]
            pad = torch.zeros(
                *causal_mask.shape[:-1],
                prefix_len,
                dtype=causal_mask.dtype,
                device=causal_mask.device,
            )
            return torch.cat([pad, causal_mask], dim=-1)

        def patched_forward(
            module_self,
            hidden_states,
            position_embeddings=None,
            attention_mask=None,
            shared_kv_states=None,
            past_key_values=None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module_self.head_dim)
            module_layer_idx = int(getattr(module_self, "layer_idx", -1))
            is_target_layer = module_layer_idx == int(
                propagation_state.get("target_layer_idx", -1)
            )
            # Gemma-4 strips k_proj/v_proj/k_norm/v_norm from KV-shared (consumer)
            # layers (HF modeling_gemma4.py:1168-1179). Any patched layer that
            # cannot project its own K/V must route through the shared-KV branch.
            module_lacks_kv_projections = not hasattr(module_self, "k_proj")
            module_is_kv_shared_layer = bool(getattr(module_self, "is_kv_shared_layer", False))
            is_shared_follower = (
                module_lacks_kv_projections
                or module_is_kv_shared_layer
                or module_layer_idx in propagation_state.get(
                    "shared_layer_indices_set",
                    frozenset(),
                )
            )

            cos, sin = position_embeddings

            # Q: project → view → q_norm → rotary → transpose
            query_states = module_self.q_proj(hidden_states).view(hidden_shape)
            query_states = module_self.q_norm(query_states)
            query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
            query_states = query_states.transpose(1, 2)

            shared_kv_states = _resolve_shared_kv_states(shared_kv_states, past_key_values)
            if is_shared_follower:
                if past_key_values is None:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: patched shared layer "
                        f"{module_layer_idx} requires past_key_values."
                    )
                shared_source_idx = getattr(module_self, "kv_shared_layer_index", None)
                if (
                    shared_source_idx is None
                    or shared_kv_states is None
                    or shared_source_idx not in shared_kv_states
                ):
                    shared_source_idx = int(propagation_state.get("target_layer_idx", module_layer_idx))
                if shared_source_idx is None:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: patched shared layer "
                        f"{module_layer_idx} has no kv_shared_layer_index."
                    )
                if shared_kv_states is None or shared_source_idx not in shared_kv_states:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: shared K/V source "
                        f"{shared_source_idx} missing for patched layer "
                        f"{module_layer_idx}."
                    )
                key_states, value_states = shared_kv_states[shared_source_idx]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
                fresh_kv_len = self._cache_seq_length(past_key_values)
                if fresh_kv_len <= 0:
                    fresh_kv_len = int(key_states.shape[-2])
            else:
                key_states = module_self.k_proj(hidden_states).view(hidden_shape)
                if module_self.v_proj is not None:
                    value_states = module_self.v_proj(hidden_states).view(hidden_shape)
                else:
                    value_states = key_states  # mirrors Gemma4 edge case

                key_states = module_self.k_norm(key_states)
                key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
                key_states = key_states.transpose(1, 2)

                value_states = module_self.v_norm(value_states)
                value_states = value_states.transpose(1, 2)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, module_self.layer_idx
                    )
                fresh_kv_len = int(key_states.shape[-2])

            fresh_key_states = key_states
            fresh_value_states = value_states

            # ── LAZARUS KV-DIRECT PREFIX INJECTION ──────────────────────────
            mask_inputs_local = getattr(module_self, "_lazarus_mask_inputs", None)
            ak, av = _prepare_archived_prefix(module_self, cos, sin, key_states, value_states)
            if ak is not None and av is not None:
                prefix_already_present = bool(
                    is_shared_follower
                    and propagation_state.get("propagate_shared_prefix", False)
                    and int(key_states.shape[-2]) > int(fresh_kv_len)
                )
                if not prefix_already_present:
                    key_states = torch.cat([ak, key_states], dim=-2)
                    value_states = torch.cat([av, value_states], dim=-2)

            prefix_present = max(0, int(key_states.shape[-2]) - int(fresh_kv_len))
            if prefix_present > 0:
                fire_counter.setdefault("prefix_layers", set()).add(module_layer_idx)

            if shared_kv_states is not None and is_target_layer:
                if propagation_state.get("propagate_shared_prefix", False) and prefix_present > 0:
                    shared_kv_states[module_layer_idx] = (key_states, value_states)
                    # When the target is itself a KV-consumer (e.g. layer 30 →
                    # producer 28 for Gemma-4-E2B-it sliding), also stamp the
                    # prefix-augmented K/V into the producer's slot so downstream
                    # consumers reading from shared_kv_states[producer_idx]
                    # (HF semantic at modeling_gemma4.py:1208) see the prefix.
                    target_kv_shared_source_idx = propagation_state.get("target_kv_shared_source_idx")
                    if target_kv_shared_source_idx is not None:
                        shared_kv_states[int(target_kv_shared_source_idx)] = (
                            key_states,
                            value_states,
                        )
                    fire_counter.setdefault("prefix_layers", set()).update(
                        propagation_state.get("full_attention_layer_indices", [])
                    )
                else:
                    shared_kv_states[module_layer_idx] = (
                        fresh_key_states,
                        fresh_value_states,
                    )

            # ── EAGER ATTENTION WITH TIER MASK ─────────────────────────────
            key_rep = repeat_kv(key_states, module_self.num_key_value_groups)
            value_rep = repeat_kv(value_states, module_self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) * module_self.scaling

            selected_attention_mask = attention_mask
            if propagation_state.get("use_prefixed_full_attention_mask", False) and (
                is_target_layer or is_shared_follower
            ):
                selected_attention_mask = propagation_state.get(
                    "prefixed_full_attention_mask",
                    attention_mask,
                )

            if selected_attention_mask is not None:
                full_mask = _pad_attention_mask_for_prefix(
                    selected_attention_mask,
                    fresh_kv_len=int(fresh_kv_len),
                    prefix_len=prefix_present,
                )
                attn_weights = attn_weights + full_mask

            # Axis-4 tier mask over the [0, N) prefix portion.
            if prefix_present > 0 and (
                mask_inputs_local is not None
                or propagation_state.get("selected_head_indices", ())
            ):
                prefix_slice = attn_weights[..., :prefix_present].contiguous()
                if mask_inputs_local is not None:
                    prefix_slice = apply_tier_attention_mask(prefix_slice, mask_inputs_local)
                    fire_counter["mask_applied"] = fire_counter.get("mask_applied", 0) + 1
                prefix_slice = _apply_selected_head_prefix_mask(prefix_slice)
                attn_weights = torch.cat(
                    [prefix_slice, attn_weights[..., prefix_present:]], dim=-1
                )

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype
            )

            attn_output = torch.matmul(attn_weights, value_rep)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = module_self.o_proj(attn_output)

            return attn_output, attn_weights

        return patched_forward

    def generate_with_kv_direct_materialization(
        self,
        prompt: str,
        config,
        *,
        materialization,
        per_window_token_ranges: dict[int, tuple[int, int]],
        tier_assignments: dict[int, "TierLabel"],
        warm_config,
        source_layer: int,
        warm_scores: dict[int, float] | None = None,
        query_id: str | None = None,
        insertion_family: Literal["full_attention", "sliding"] = "full_attention",
        sliding_layer_indices: tuple[int, ...] | None = None,
        sliding_head_indices: tuple[int, ...] | None = None,
    ) -> "KvDirectGenerationResult":
        """Generate using archived-residual K/V for context, NO archived-text replay.

        Frozen contract: axis-5 ``ve-ins-0mo9pb4f5000004cc67``. Path-A
        prohibition (resolved via ORCH posture 2/3): archived tokens are
        NEVER re-embedded; archived context enters through
        ``materialization`` (the already-projected K/V from
        :func:`materialize_kv_direct`). Fresh-query tokens flow through
        the normal forward path.

        The N archived-residual K/V slots are concatenated as a prefix to
        the live cache at layer ``source_layer + 1`` via a mode-aware
        monkey-patch on Gemma-4 full-attention ``self_attn`` modules.
        ``materialization.K`` / ``materialization.V`` are prepended at the
        source full-attention layer and may also propagate through the
        shared-K/V full-attention followers depending on
        ``LAZARUS_KV_PROPAGATION_MODE``. HOT windows thereby appear
        byte-identical
        in attention logits; COLD windows were already excluded at
        gather-time; WARM penalty is recorded for observability but not
        additively applied here (out-of-scope for leg-3).

        Parameters
        ----------
        prompt:
            Fresh-query prompt to generate from. Tokenized and run
            through the model's normal forward path.
        config:
            :class:`GenerationConfig` — sampling + ``max_new_tokens``.
        materialization:
            :class:`KVDirectMaterialization` carrier for the archived
            K/V tensors. ``.path_a_replay_count`` is forwarded onto the
            result metadata.
        per_window_token_ranges:
            ``window_id -> (slot_start, slot_end)`` on the archived-K/V
            ``N`` axis. Forwarded to the result for axis-6 consumption.
        tier_assignments:
            ``window_id -> TierLabel`` for every included window.
        warm_config:
            :class:`WarmPenaltyConfig`. ``mask_penalty_applied`` is set
            ``True`` iff a non-zero penalty config is provided; the
            subtractive logit scaling is not wired into the live
            attention path in this leg (documentary field).
        source_layer:
            ``injection_layer`` at which residuals were captured;
            ``source_layer + 1`` is the attention layer that receives
            the archived K/V prefix.
        warm_scores:
            Optional ``window_id -> score`` map. Reserved for future
            per-window penalty scaling; not applied here.
        query_id:
            Optional opaque id echoed onto the result metadata.
        insertion_family:
            Runtime K/V insertion family. ``"full_attention"`` preserves
            the existing Gemma-4 propagation path. ``"sliding"`` patches
            only the explicitly selected sliding-attention layers and
            gates the archived prefix to the selected heads.
        sliding_layer_indices:
            Explicit decoder-layer indices for the sliding-family path.
            Ignored when ``insertion_family="full_attention"``.
        sliding_head_indices:
            Explicit attention-head indices allowed to see the archived
            prefix on the sliding-family path. Ignored when
            ``insertion_family="full_attention"``.

        Returns
        -------
        KvDirectGenerationResult
            Carries decoded ``text`` plus the axis-6 metadata bundle
            (``selected_tier``, ``mask_penalty_applied``,
            ``kv_direct_active``, ``vram_peak_mib``, ``vram_delta_mib``,
            ``path_a_replay_count``).
        """
        from ..generation import StopReason
        from .types import LazarusBackend  # local import for clarity
        import types as _pytypes
        import os as _os

        _ = LazarusBackend  # reserved for future path-label correlation

        torch = self._torch
        model = self._model
        insertion_family = str(insertion_family).strip().lower()
        if insertion_family not in {"full_attention", "sliding"}:
            raise RuntimeError(
                "generate_with_kv_direct_materialization: insertion_family "
                "must be one of {'full_attention', 'sliding'}; "
                f"got {insertion_family!r}."
            )
        propagation_mode = str(
            _os.environ.get("LAZARUS_KV_PROPAGATION_MODE", "a")
        ).strip().lower()
        if propagation_mode not in {"a", "b", "ab", "none"}:
            raise RuntimeError(
                "generate_with_kv_direct_materialization: "
                "LAZARUS_KV_PROPAGATION_MODE must be one of "
                "{'a', 'b', 'ab', 'none'}; "
                f"got {propagation_mode!r}."
            )
        propagation_state: dict[str, Any] = {"mode": propagation_mode}

        on_cuda = self._device.type == "cuda" and torch.cuda.is_available()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(self._device)
            vram_start = int(torch.cuda.memory_allocated(self._device))
        else:
            vram_start = 0

        # Tokenize the fresh query. Archived tokens are NEVER fed here.
        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        input_length = int(input_ids.shape[1])
        fresh_attention_mask = model_inputs.get("attention_mask")
        if fresh_attention_mask is None:
            fresh_attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Resolve target layer (materialization lives at source_layer + 1).
        layers = self._resolve_layers()
        target_idx = int(source_layer) + 1
        if target_idx >= len(layers):
            raise RuntimeError(
                "generate_with_kv_direct_materialization: source_layer+1 "
                f"out of range for model with {len(layers)} layers "
                f"(source_layer={source_layer})."
            )
        target_layer = layers[target_idx]
        target_self_attn = getattr(target_layer, "self_attn", None) or getattr(
            target_layer, "attention", None
        )
        if target_self_attn is None:
            raise RuntimeError(
                "generate_with_kv_direct_materialization: target layer has "
                "no .self_attn/.attention module to hook."
            )

        materialized_source_layer_raw = getattr(
            materialization, "materialized_source_layer", None
        )
        materialized_source_layer = (
            None
            if materialized_source_layer_raw is None
            else int(materialized_source_layer_raw)
        )
        if (
            materialized_source_layer is not None
            and int(source_layer) != materialized_source_layer
        ):
            raise RuntimeError(
                "generate_with_kv_direct_materialization: requested "
                f"source_layer={int(source_layer)} disagrees with "
                "materialized_source_layer="
                f"{materialized_source_layer} on the KVDirectMaterialization carrier."
            )
        materialized_target_layer = (
            None
            if materialized_source_layer is None
            else int(materialized_source_layer) + 1
        )
        materialized_insertion_family_raw = getattr(
            materialization, "materialized_insertion_family", None
        )
        materialized_insertion_family = (
            None
            if materialized_insertion_family_raw is None
            else str(materialized_insertion_family_raw).strip().lower()
        )
        materialized_lineage_layer_indices = tuple(
            dict.fromkeys(
                int(layer_idx)
                for layer_idx in (
                    getattr(materialization, "materialized_lineage_layer_indices", ())
                    or ()
                )
            )
        )

        selected_tier_labels = sorted(
            {TierLabel(tier).value for tier in tier_assignments.values()}
        )
        has_warm = any(
            TierLabel(t) == TierLabel.WARM for t in tier_assignments.values()
        )
        penalty_value = float(getattr(warm_config, "penalty_value", 0.0))
        mask_penalty_will_apply = bool(has_warm and penalty_value > 0.0)

        # Build the AttentionTierMaskInputs that the patched forward reads.
        _normalized_tier_map = {
            int(wid): TierLabel(tier) for wid, tier in tier_assignments.items()
        }
        mask_inputs = AttentionTierMaskInputs(
            per_window_token_ranges={
                int(wid): (int(a), int(b))
                for wid, (a, b) in per_window_token_ranges.items()
            },
            tier_assignments=_normalized_tier_map,
            warm_config=warm_config,
            warm_scores=(
                None if warm_scores is None
                else {int(wid): float(s) for wid, s in warm_scores.items()}
            ),
        )

        archived_K = materialization.K
        archived_V = materialization.V
        n_archived = int(archived_K.shape[-2])
        propagation_state["n_archived"] = n_archived
        # KV-consumer (shared-KV) target resolution. Gemma-4 strips k_proj/
        # v_proj/k_norm/v_norm on KV-shared (consumer) layers (HF
        # transformers/models/gemma4/modeling_gemma4.py:1168-1179). When the
        # requested source_layer+1 resolves to a consumer (e.g. Gemma-4-E2B-it
        # layers 29..34), prefix renormalisation must use the producer module
        # at self_attn.kv_shared_layer_index (which retains k_norm/v_norm).
        # See ve-ins-0moe02apx00003fe3b6 (failure-finding) and
        # ve-ins-0moe0c0jb0000475268 (lead manifest).
        target_is_kv_consumer = bool(
            getattr(target_self_attn, "is_kv_shared_layer", False)
        ) or not hasattr(target_self_attn, "k_proj")
        target_kv_shared_source_idx: int | None = None
        if target_is_kv_consumer:
            producer_idx_raw = getattr(target_self_attn, "kv_shared_layer_index", None)
            if producer_idx_raw is None:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: target layer "
                    f"{target_idx} is a KV-shared consumer (no k_proj) but "
                    "exposes no kv_shared_layer_index — cannot resolve "
                    "producer module for prefix normalisation."
                )
            producer_idx = int(producer_idx_raw)
            if producer_idx < 0 or producer_idx >= len(layers):
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: target layer "
                    f"{target_idx} kv_shared_layer_index={producer_idx} out "
                    f"of range for model with {len(layers)} layers."
                )
            producer_self_attn = getattr(layers[producer_idx], "self_attn", None) or getattr(
                layers[producer_idx], "attention", None
            )
            if producer_self_attn is None:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: producer layer "
                    f"{producer_idx} (resolved from KV-consumer target "
                    f"{target_idx}) has no .self_attn/.attention module."
                )
            missing_attrs = [
                attr_name
                for attr_name in ("k_norm", "v_norm")
                if not hasattr(producer_self_attn, attr_name)
            ]
            if missing_attrs:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: producer layer "
                    f"{producer_idx} (resolved from KV-consumer target "
                    f"{target_idx}) is missing prefix-normalisation attribute(s): "
                    f"{missing_attrs!r}."
                )
            propagation_state["prefix_attn_module"] = producer_self_attn
            target_kv_shared_source_idx = producer_idx
        else:
            propagation_state["prefix_attn_module"] = target_self_attn
        propagation_state["target_is_kv_consumer"] = target_is_kv_consumer
        propagation_state["target_kv_shared_source_idx"] = target_kv_shared_source_idx
        propagation_state["target_layer_idx"] = target_idx

        def _resolve_self_attn(layer) -> Any | None:
            return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)

        def _normalize_index_tuple(
            values: tuple[int, ...] | None,
            *,
            name: str,
        ) -> tuple[int, ...] | None:
            if values is None:
                return None
            normalized = tuple(dict.fromkeys(int(value) for value in values))
            if not normalized:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: "
                    f"{name} cannot be empty when provided."
                )
            if any(value < 0 for value in normalized):
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: "
                    f"{name} must contain only non-negative indices."
                )
            return normalized

        text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        layer_types = list(getattr(text_config, "layer_types", []))
        target_layer_type = (
            getattr(target_self_attn, "layer_type", None)
            or (layer_types[target_idx] if target_idx < len(layer_types) else None)
        )

        def _layer_type_for(layer_idx: int, layer_self_attn: Any | None = None) -> str | None:
            if layer_self_attn is None:
                layer_self_attn = _resolve_self_attn(layers[layer_idx])
            layer_type = getattr(layer_self_attn, "layer_type", None)
            if layer_type is not None:
                return str(layer_type)
            if layer_idx < len(layer_types):
                return str(layer_types[layer_idx])
            return None

        def _is_sliding_layer(layer_idx: int, layer_self_attn: Any | None = None) -> bool:
            if layer_self_attn is None:
                layer_self_attn = _resolve_self_attn(layers[layer_idx])
            if layer_self_attn is None:
                return False
            if bool(getattr(layer_self_attn, "is_sliding", False)):
                return True
            return _layer_type_for(layer_idx, layer_self_attn) == "sliding_attention"

        full_attention_layer_indices: list[int] = []
        patch_targets: list[Any] = []
        selected_head_indices: tuple[int, ...] = ()
        selected_layer_indices: tuple[int, ...] = ()
        if insertion_family == "full_attention":
            full_attention_layer_indices = [target_idx]
            for layer_idx in range(target_idx + 1, len(layers)):
                layer_self_attn = _resolve_self_attn(layers[layer_idx])
                if layer_self_attn is None:
                    continue
                same_layer_type = _layer_type_for(layer_idx, layer_self_attn) == target_layer_type
                shares_from_target = getattr(layer_self_attn, "kv_shared_layer_index", None) == target_idx
                if same_layer_type or shares_from_target:
                    full_attention_layer_indices.append(layer_idx)
            full_attention_layer_indices = list(dict.fromkeys(full_attention_layer_indices))
            full_attention_self_attn_modules = [
                _resolve_self_attn(layers[layer_idx]) for layer_idx in full_attention_layer_indices
            ]
            full_attention_self_attn_modules = [
                module for module in full_attention_self_attn_modules if module is not None
            ]
            patch_targets = (
                full_attention_self_attn_modules
                if propagation_mode in {"a", "b", "ab"}
                else [target_self_attn]
            )
            selected_layer_indices = tuple(full_attention_layer_indices)
            shared_layer_indices = full_attention_layer_indices[1:]
        else:
            if (
                materialized_insertion_family is not None
                and materialized_insertion_family != "sliding"
            ):
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: insertion_family='sliding' "
                    "requires sliding-origin materialization; got "
                    f"materialized_insertion_family={materialized_insertion_family!r} "
                    f"from source_layer={materialized_source_layer!r} "
                    f"(target layer {materialized_target_layer!r})."
                )
            if target_layer_type != "sliding_attention" and not bool(
                getattr(target_self_attn, "is_sliding", False)
            ):
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: insertion_family='sliding' "
                    "requires source_layer+1 to resolve to a sliding-attention layer; "
                    f"got layer {target_idx} type {target_layer_type!r}."
                )

            normalized_sliding_layer_indices = _normalize_index_tuple(
                sliding_layer_indices,
                name="sliding_layer_indices",
            )
            normalized_sliding_head_indices = _normalize_index_tuple(
                sliding_head_indices,
                name="sliding_head_indices",
            )
            if normalized_sliding_layer_indices is None or normalized_sliding_head_indices is None:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: insertion_family='sliding' "
                    "requires both sliding_layer_indices and sliding_head_indices."
                )
            if materialized_lineage_layer_indices:
                invalid_lineage_indices = tuple(
                    layer_idx
                    for layer_idx in normalized_sliding_layer_indices
                    if layer_idx not in materialized_lineage_layer_indices
                )
                if invalid_lineage_indices:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: sliding_layer_indices "
                        f"{invalid_lineage_indices!r} are incompatible with materialized "
                        "sliding lineage "
                        f"{materialized_lineage_layer_indices!r} rooted at target layer "
                        f"{materialized_target_layer!r}."
                    )

            q_proj = getattr(target_self_attn, "q_proj", None)
            q_proj_out_features = getattr(q_proj, "out_features", None)
            if q_proj_out_features is None:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: target layer has no q_proj "
                    "out_features to validate sliding_head_indices."
                )
            target_num_attention_heads = int(q_proj_out_features) // int(target_self_attn.head_dim)
            invalid_head_indices = [
                head_idx
                for head_idx in normalized_sliding_head_indices
                if head_idx >= target_num_attention_heads
            ]
            if invalid_head_indices:
                raise RuntimeError(
                    "generate_with_kv_direct_materialization: sliding_head_indices "
                    f"{tuple(invalid_head_indices)!r} out of range for {target_num_attention_heads} "
                    "attention heads."
                )

            shared_layer_indices = []
            for layer_idx in normalized_sliding_layer_indices:
                if layer_idx >= len(layers):
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: sliding_layer_indices "
                        f"contains out-of-range layer {layer_idx} for model with {len(layers)} layers."
                    )
                layer_self_attn = _resolve_self_attn(layers[layer_idx])
                if layer_self_attn is None:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: sliding layer "
                        f"{layer_idx} has no .self_attn/.attention module to hook."
                    )
                if not _is_sliding_layer(layer_idx, layer_self_attn):
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: sliding_layer_indices "
                        f"must contain only sliding-attention layers; layer {layer_idx} "
                        f"resolved to type {_layer_type_for(layer_idx, layer_self_attn)!r}."
                    )
                patch_targets.append(layer_self_attn)
                if getattr(layer_self_attn, "kv_shared_layer_index", None) is not None:
                    shared_layer_indices.append(layer_idx)
            selected_layer_indices = normalized_sliding_layer_indices
            selected_head_indices = normalized_sliding_head_indices
            full_attention_self_attn_modules = patch_targets

        patch_targets = list(dict.fromkeys(patch_targets))
        propagation_state["full_attention_layer_indices"] = full_attention_layer_indices
        propagation_state["shared_layer_indices"] = shared_layer_indices
        propagation_state["shared_layer_indices_set"] = frozenset(shared_layer_indices)
        propagation_state["selected_head_indices"] = selected_head_indices
        propagation_state["selected_layer_indices"] = selected_layer_indices
        propagation_state["use_prefixed_full_attention_mask"] = (
            insertion_family == "full_attention" and propagation_mode in {"a", "ab"}
        )
        propagation_state["propagate_shared_prefix"] = (
            insertion_family == "full_attention" and propagation_mode in {"a", "ab"}
        )

        def _build_prefixed_full_attention_mask(
            current_input_ids,
            current_fresh_attention_mask,
        ):
            q_len = int(current_input_ids.shape[1])
            fresh_total_len = int(current_fresh_attention_mask.shape[1])
            q_offset = fresh_total_len - q_len
            mask_dtype = target_self_attn.q_proj.weight.dtype
            min_value = torch.finfo(mask_dtype).min
            device = current_fresh_attention_mask.device

            query_positions = torch.arange(
                q_offset,
                q_offset + q_len,
                device=device,
                dtype=torch.long,
            )
            key_positions = torch.arange(
                fresh_total_len,
                device=device,
                dtype=torch.long,
            )
            causal_ok = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            causal_ok = causal_ok.view(1, 1, q_len, fresh_total_len)
            valid_keys = current_fresh_attention_mask.to(device=device, dtype=torch.bool)[
                :, None, None, :
            ]
            fresh_mask = torch.where(
                causal_ok & valid_keys,
                torch.zeros(
                    (current_fresh_attention_mask.shape[0], 1, q_len, fresh_total_len),
                    dtype=mask_dtype,
                    device=device,
                ),
                torch.full(
                    (current_fresh_attention_mask.shape[0], 1, q_len, fresh_total_len),
                    min_value,
                    dtype=mask_dtype,
                    device=device,
                ),
            )
            prefix_mask = torch.zeros(
                (current_fresh_attention_mask.shape[0], 1, q_len, n_archived),
                dtype=mask_dtype,
                device=device,
            )
            return torch.cat([prefix_mask, fresh_mask], dim=-1)

        def _build_generation_attention_mask(current_input_ids, current_fresh_attention_mask):
            if propagation_state.get("use_prefixed_full_attention_mask", False):
                propagation_state["prefixed_full_attention_mask"] = (
                    _build_prefixed_full_attention_mask(
                        current_input_ids,
                        current_fresh_attention_mask,
                    )
                )
            else:
                propagation_state["prefixed_full_attention_mask"] = None
            return current_fresh_attention_mask

        # Install the patched forward on the target attention module.
        # Fire counter records truthful axis-6 observability.
        fire_counter: dict[str, Any] = {"prefix_layers": set(), "mask_applied": 0}
        installed_patches: list[tuple[str, Any, Any]] = []
        patched = self._kv_direct_patched_forward_factory(fire_counter, propagation_state)
        for module in patch_targets:
            current_forward = module.forward
            forward_func = getattr(current_forward, "__func__", None)
            forward_kwdefaults = getattr(forward_func, "__kwdefaults__", None)
            if isinstance(forward_kwdefaults, dict) and "__original" in forward_kwdefaults:
                new_kwdefaults = dict(forward_kwdefaults)
                new_kwdefaults["__original"] = _pytypes.MethodType(patched, module)
                installed_patches.append(("kwdefaults", forward_func, dict(forward_kwdefaults)))
                forward_func.__kwdefaults__ = new_kwdefaults
            else:
                installed_patches.append(("forward", module, current_forward))
                module.forward = _pytypes.MethodType(patched, module)  # type: ignore[assignment]
            module._lazarus_archived_K = archived_K  # type: ignore[attr-defined]
            module._lazarus_archived_V = archived_V  # type: ignore[attr-defined]
            module._lazarus_mask_inputs = mask_inputs  # type: ignore[attr-defined]

        stop_reason = StopReason.MAX_TOKENS
        generated_token_ids: list[int] = []
        start_time = time.time()
        eos_token_ids = set(self._resolved_eos_token_ids())
        stop_token_ids = set(int(t) for t in config.stop_tokens) | eos_token_ids

        try:
            total_window_tokens = input_length + int(config.max_new_tokens)
            with torch.inference_mode(), self._generation_context(total_window_tokens):
                # Prefill — patched forward sees q_len > 1, prepends prefix,
                # applies tier mask on [0, N) of attention logits.
                caller_attention_mask = _build_generation_attention_mask(
                    input_ids,
                    fresh_attention_mask,
                )
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=caller_attention_mask,
                    use_cache=True,
                )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                if past_key_values is None:
                    raise RuntimeError(
                        "generate_with_kv_direct_materialization: prefill did "
                        "not return past_key_values; cannot continue decoding."
                    )

                next_token = self._sample_next_token(logits[:, -1, :], config)

                # Debug: inspect prefill logits distribution.
                import os as _os
                if _os.environ.get("LAZARUS_KV_DIRECT_DEBUG"):
                    _last = logits[:, -1, :].float()
                    _top = _last.topk(5, dim=-1)
                    _top_ids = _top.indices[0].tolist()
                    _top_vals = [round(v, 3) for v in _top.values[0].tolist()]
                    _top_toks = [
                        self._tokenizer.decode([int(i)], skip_special_tokens=False)
                        for i in _top_ids
                    ]
                    print(
                        f"[kv_direct.debug] prefill top5 ids={_top_ids} "
                        f"vals={_top_vals} toks={_top_toks} "
                        f"prefix_forwards={len(fire_counter.get('prefix_layers', set()))}",
                        flush=True,
                    )

                for step in range(int(config.max_new_tokens)):
                    token_id = int(next_token[0, 0].item())
                    generated_token_ids.append(token_id)

                    if _os.environ.get("LAZARUS_KV_DIRECT_DEBUG") and step < 5:
                        print(
                            f"[kv_direct.debug] step={step} token_id={token_id} "
                            f"tok={self._tokenizer.decode([token_id], skip_special_tokens=False)!r} "
                            f"in_stop={token_id in stop_token_ids}",
                            flush=True,
                        )

                    if token_id in stop_token_ids:
                        stop_reason = (
                            StopReason.EOS if token_id in eos_token_ids
                            else StopReason.STOP_TOKEN
                        )
                        break

                    if step + 1 >= int(config.max_new_tokens):
                        break

                    # Extend the fresh-token mask by one and rebuild any
                    # caller-side full-attention override for this decode step.
                    fresh_attention_mask = torch.cat(
                        [
                            fresh_attention_mask,
                            torch.ones(
                                (fresh_attention_mask.shape[0], 1),
                                dtype=fresh_attention_mask.dtype,
                                device=fresh_attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
                    caller_attention_mask = _build_generation_attention_mask(
                        next_token,
                        fresh_attention_mask,
                    )
                    outputs = model(
                        input_ids=next_token,
                        attention_mask=caller_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = outputs.logits
                    past_key_values = outputs.past_key_values
                    next_token = self._sample_next_token(logits[:, -1, :], config)
        finally:
            for patch_kind, patch_target, original_value in reversed(installed_patches):
                if patch_kind == "kwdefaults":
                    patch_target.__kwdefaults__ = original_value
                else:
                    patch_target.forward = original_value  # type: ignore[assignment]
            for module in patch_targets:
                for attr in tuple(name for name in vars(module).keys() if name.startswith("_lazarus_")):
                    try:
                        delattr(module, attr)
                    except AttributeError:
                        pass
            for module in patch_targets:
                leaked_attrs = [
                    name for name in vars(module).keys() if name.startswith("_lazarus_")
                ]
                assert not leaked_attrs, (
                    "generate_with_kv_direct_materialization: leaked lazarus attrs "
                    f"on layer {getattr(module, 'layer_idx', '?')}: {leaked_attrs}"
                )

        gen_time = time.time() - start_time

        # Decode the generated text.
        generated_text = self._tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )
        output_length = len(generated_token_ids)

        if on_cuda:
            vram_peak = int(torch.cuda.max_memory_allocated(self._device))
            vram_peak_mib = int(vram_peak // (1024 * 1024))
            vram_delta_mib = int(max(0, vram_peak - vram_start) // (1024 * 1024))
        else:
            vram_peak_mib = 0
            vram_delta_mib = 0

        # TRUTHFUL axis-6 observability.
        prefix_forwards = int(len(fire_counter.get("prefix_layers", set())))
        mask_applied_count = int(fire_counter.get("mask_applied", 0))
        kv_direct_active = prefix_forwards > 0
        mask_penalty_applied = bool(mask_penalty_will_apply and mask_applied_count > 0)

        self._set_generation_path("torch.kv_direct.materialized_residuals")

        metadata: dict[str, Any] = {
            "selected_tier": selected_tier_labels,
            "mask_penalty_applied": mask_penalty_applied,
            "kv_direct_active": kv_direct_active,
            "insertion_family": insertion_family,
            "vram_peak_mib": int(vram_peak_mib),
            "vram_delta_mib": int(vram_delta_mib),
            "path_a_replay_count": int(
                getattr(materialization, "path_a_replay_count", 0)
            ),
            "query_id": query_id,
            "n_archived_slots": int(n_archived),
            "propagation_mode": propagation_mode,
            "sliding_layer_indices": [int(layer_idx) for layer_idx in selected_layer_indices]
            if insertion_family == "sliding"
            else [],
            "sliding_head_indices": [int(head_idx) for head_idx in selected_head_indices]
            if insertion_family == "sliding"
            else [],
            "prefix_forwards_observed": prefix_forwards,
            "mask_applied_count": mask_applied_count,
            "per_window_token_ranges": {
                int(wid): (int(a), int(b))
                for wid, (a, b) in per_window_token_ranges.items()
            },
            "hot_budget_mib_observed": int(
                getattr(materialization, "hot_budget_mib_observed", 0)
            ),
            "materialization_mode": str(
                getattr(
                    materialization,
                    "materialization_mode",
                    "project_through_W_K_W_V_at_injection_layer",
                )
            ),
            "gen_time_seconds": float(gen_time),
            "input_tokens": int(input_length),
            "output_tokens": int(output_length),
            "stop_reason": stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason),
        }

        return KvDirectGenerationResult(text=generated_text, metadata=metadata)

    def clear_cache(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()


@dataclass
class KvDirectGenerationResult:
    """Return value for :meth:`TorchInferenceRuntime.generate_with_kv_direct_materialization`.

    Mirrors the minimal surface of :class:`GenerationResult` (just
    ``text``) while carrying the axis-6-facing ``metadata`` bundle that
    downstream consumers read. A dataclass is used because
    :class:`GenerationResult` is a frozen pydantic model and cannot be
    extended with arbitrary fields.
    """

    text: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Axis-4 (mute-compress-mask) pure-tensor surface.
#
# Frozen contract: ``ve-ins-0mo9p9ke3000060c0c7``. This block is standalone,
# CPU-runnable, and MUST NOT be wired into any existing HF model path in this
# module (follow-up mission). It only provides dataclasses, the pure-tensor
# :func:`apply_tier_attention_mask` entry point, JSON round-trip helpers, and
# the per-query transparency observation builder.
#
# Tier semantics (contract-literal):
#   * HOT  — identity: logits are byte-identical to the unmasked baseline
#   * WARM — subtractive logit penalty (``logit -= penalty_value``), optional
#            per-window scaling by ``(1 - score)`` and optional ``clamp_min``
#   * COLD — pre-softmax exclusion: slice set to ``-inf`` in the input dtype
# ---------------------------------------------------------------------------


ATTENTION_TIER_MASK_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class WarmPenaltyConfig:
    """Subtractive attention-logit penalty for WARM-tier windows.

    Contract: ``ve-ins-0mo9p9ke3000060c0c7`` (axis-4 mute-compress-mask).
    Penalty is subtractive in logit units: ``logit -= penalty_value``.
    Multiplicative penalties are forbidden by the contract.
    """

    penalty_value: float = 4.0
    per_warm_uniform: bool = True
    clamp_min: float | None = None


@dataclass(frozen=True)
class AttentionTierMaskInputs:
    """Per-query inputs for :func:`apply_tier_attention_mask`.

    ``per_window_token_ranges`` maps ``window_id -> (start, end)`` half-open
    intervals on the ``kv_len`` axis of the attention-logit tensor.
    ``tier_assignments`` maps ``window_id -> TierLabel``. Every window id in
    ``per_window_token_ranges`` must appear in ``tier_assignments`` and vice
    versa; mismatched keys raise ``ValueError`` at apply time.
    ``warm_scores`` is optional; it maps ``window_id -> normalized router
    score in [0.0, 1.0]`` and is consulted only when
    ``warm_config.per_warm_uniform is False`` to scale the warm penalty by
    ``(1 - score)``.
    """

    per_window_token_ranges: dict[int, tuple[int, int]]
    tier_assignments: dict[int, "TierLabel"]
    warm_config: WarmPenaltyConfig
    warm_scores: dict[int, float] | None = None


def apply_tier_attention_mask(attn_logits, inputs: AttentionTierMaskInputs):
    """Apply axis-4 tier mask to pre-softmax attention logits.

    Contract: ``ve-ins-0mo9p9ke3000060c0c7``. Operates on the last axis of
    ``attn_logits`` (the ``kv_len`` axis); any leading dimensions are
    preserved as-is. Returns a new tensor (never modifies in place).

    * HOT windows: slices are untouched on the cloned tensor → byte-identity
      with the unmasked baseline.
    * COLD windows: slices are set to ``-inf`` (dtype-stable) — pre-softmax
      exclusion removes them from the effective attention budget.
    * WARM windows: slices have ``warm_config.penalty_value`` subtracted.
      If ``warm_config.per_warm_uniform is False``, the penalty is scaled
      per window by ``(1 - inputs.warm_scores[window_id])``. If
      ``warm_config.clamp_min is not None``, WARM-touched logits are clamped
      from below AFTER the subtraction.

    Raises
    ------
    ValueError
        If the window-id sets of ``per_window_token_ranges`` and
        ``tier_assignments`` differ, if any ``(start, end)`` is degenerate
        or out of bounds on the last axis, or if ``per_warm_uniform`` is
        ``False`` and a WARM window is missing from ``warm_scores`` (or
        ``warm_scores`` is ``None``).
    """

    range_keys = set(inputs.per_window_token_ranges.keys())
    tier_keys = set(inputs.tier_assignments.keys())
    if range_keys != tier_keys:
        raise ValueError(
            "apply_tier_attention_mask: per_window_token_ranges and "
            "tier_assignments must share the same window-id key set; "
            f"range-only={sorted(range_keys - tier_keys)!r} "
            f"tier-only={sorted(tier_keys - range_keys)!r}"
        )

    kv_len = int(attn_logits.shape[-1])
    for window_id, (start, end) in inputs.per_window_token_ranges.items():
        if end <= start:
            raise ValueError(
                f"apply_tier_attention_mask: window {window_id} has "
                f"degenerate range (start={start}, end={end}); end must be > start."
            )
        if start < 0 or end > kv_len:
            raise ValueError(
                f"apply_tier_attention_mask: window {window_id} range "
                f"({start}, {end}) is out of bounds for kv_len={kv_len}."
            )

    warm_config = inputs.warm_config
    if not warm_config.per_warm_uniform:
        warm_window_ids = [
            wid
            for wid, tier in inputs.tier_assignments.items()
            if tier == TierLabel.WARM
        ]
        if warm_window_ids and inputs.warm_scores is None:
            raise ValueError(
                "apply_tier_attention_mask: warm_config.per_warm_uniform is "
                "False but warm_scores is None; per-window scaling requires "
                "a score for every WARM window."
            )
        if inputs.warm_scores is not None:
            missing = [
                wid for wid in warm_window_ids if wid not in inputs.warm_scores
            ]
            if missing:
                raise ValueError(
                    "apply_tier_attention_mask: per_warm_uniform is False but "
                    f"WARM windows {sorted(missing)!r} have no entry in "
                    "warm_scores."
                )

    out = attn_logits.clone()
    dtype = out.dtype
    neg_inf = float("-inf")

    for window_id, tier in inputs.tier_assignments.items():
        start, end = inputs.per_window_token_ranges[window_id]
        if tier == TierLabel.HOT:
            # Identity: leave the cloned slice untouched for byte-identity.
            continue
        if tier == TierLabel.COLD:
            # Pre-softmax exclusion via additive -inf on the input dtype.
            fill_value = out.new_full((), neg_inf, dtype=dtype)
            out[..., start:end] = fill_value
            continue
        if tier == TierLabel.WARM:
            if warm_config.per_warm_uniform:
                penalty = float(warm_config.penalty_value)
            else:
                # warm_scores presence already validated above.
                score = float(inputs.warm_scores[window_id])  # type: ignore[index]
                penalty = float(warm_config.penalty_value) * (1.0 - score)
            warm_slice = out[..., start:end] - penalty
            if warm_config.clamp_min is not None:
                warm_slice = warm_slice.clamp_min(float(warm_config.clamp_min))
            out[..., start:end] = warm_slice.to(dtype)
            continue
        raise ValueError(
            f"apply_tier_attention_mask: unknown tier {tier!r} for window {window_id}."
        )

    return out


def attention_tier_mask_inputs_to_dict(
    inputs: AttentionTierMaskInputs,
) -> dict[str, Any]:
    """Serialize :class:`AttentionTierMaskInputs` to a JSON-compatible dict.

    Envelope format (schema version :data:`ATTENTION_TIER_MASK_SCHEMA_VERSION`)::

        {
          "schema_version": 1,
          "per_window_token_ranges": {"<wid>": [start, end], ...},
          "tier_assignments": {"<wid>": "hot"|"warm"|"cold", ...},
          "warm_config": {"penalty_value": float,
                          "per_warm_uniform": bool,
                          "clamp_min": float|null},
          "warm_scores": {"<wid>": score, ...} | null,
        }
    """

    ranges_dict = {
        str(wid): [int(start), int(end)]
        for wid, (start, end) in inputs.per_window_token_ranges.items()
    }
    tier_dict = {
        str(wid): TierLabel(tier).value for wid, tier in inputs.tier_assignments.items()
    }
    warm_config_dict: dict[str, Any] = {
        "penalty_value": float(inputs.warm_config.penalty_value),
        "per_warm_uniform": bool(inputs.warm_config.per_warm_uniform),
        "clamp_min": (
            None
            if inputs.warm_config.clamp_min is None
            else float(inputs.warm_config.clamp_min)
        ),
    }
    if inputs.warm_scores is None:
        warm_scores_dict: dict[str, float] | None = None
    else:
        warm_scores_dict = {
            str(wid): float(score) for wid, score in inputs.warm_scores.items()
        }
    return {
        "schema_version": ATTENTION_TIER_MASK_SCHEMA_VERSION,
        "per_window_token_ranges": ranges_dict,
        "tier_assignments": tier_dict,
        "warm_config": warm_config_dict,
        "warm_scores": warm_scores_dict,
    }


def attention_tier_mask_inputs_from_dict(
    data: dict[str, Any],
) -> AttentionTierMaskInputs:
    """Deserialize :class:`AttentionTierMaskInputs` from the canonical dict.

    Validates ``schema_version`` and the ``TierLabel`` enum strings; raises
    ``ValueError`` on any mismatch.
    """

    schema_version = data.get("schema_version")
    if schema_version != ATTENTION_TIER_MASK_SCHEMA_VERSION:
        raise ValueError(
            "attention_tier_mask_inputs_from_dict: schema_version mismatch "
            f"(expected {ATTENTION_TIER_MASK_SCHEMA_VERSION!r}, "
            f"got {schema_version!r})."
        )

    raw_ranges = data.get("per_window_token_ranges")
    if not isinstance(raw_ranges, dict):
        raise ValueError(
            "attention_tier_mask_inputs_from_dict: per_window_token_ranges "
            "must be a dict."
        )
    per_window_token_ranges: dict[int, tuple[int, int]] = {}
    for wid_str, pair in raw_ranges.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                "attention_tier_mask_inputs_from_dict: per_window_token_ranges "
                f"entry for {wid_str!r} must be a [start, end] pair."
            )
        per_window_token_ranges[int(wid_str)] = (int(pair[0]), int(pair[1]))

    raw_tiers = data.get("tier_assignments")
    if not isinstance(raw_tiers, dict):
        raise ValueError(
            "attention_tier_mask_inputs_from_dict: tier_assignments must be a dict."
        )
    tier_assignments: dict[int, TierLabel] = {}
    for wid_str, tier_str in raw_tiers.items():
        try:
            tier = TierLabel(tier_str)
        except ValueError as exc:
            raise ValueError(
                "attention_tier_mask_inputs_from_dict: unknown TierLabel "
                f"{tier_str!r} for window {wid_str!r}."
            ) from exc
        tier_assignments[int(wid_str)] = tier

    raw_warm_config = data.get("warm_config")
    if not isinstance(raw_warm_config, dict):
        raise ValueError(
            "attention_tier_mask_inputs_from_dict: warm_config must be a dict."
        )
    clamp_min_raw = raw_warm_config.get("clamp_min", None)
    warm_config = WarmPenaltyConfig(
        penalty_value=float(raw_warm_config["penalty_value"]),
        per_warm_uniform=bool(raw_warm_config["per_warm_uniform"]),
        clamp_min=None if clamp_min_raw is None else float(clamp_min_raw),
    )

    raw_warm_scores = data.get("warm_scores", None)
    if raw_warm_scores is None:
        warm_scores: dict[int, float] | None = None
    else:
        if not isinstance(raw_warm_scores, dict):
            raise ValueError(
                "attention_tier_mask_inputs_from_dict: warm_scores must be a "
                "dict or null."
            )
        warm_scores = {
            int(wid_str): float(score) for wid_str, score in raw_warm_scores.items()
        }

    return AttentionTierMaskInputs(
        per_window_token_ranges=per_window_token_ranges,
        tier_assignments=tier_assignments,
        warm_config=warm_config,
        warm_scores=warm_scores,
    )


def attention_tier_mask_inputs_to_json(inputs: AttentionTierMaskInputs) -> str:
    """Byte-deterministic JSON serialization of the axis-4 inputs."""

    return json.dumps(
        attention_tier_mask_inputs_to_dict(inputs),
        sort_keys=True,
        separators=(",", ":"),
    )


def attention_tier_mask_inputs_from_json(raw: str) -> AttentionTierMaskInputs:
    """Inverse of :func:`attention_tier_mask_inputs_to_json`."""

    return attention_tier_mask_inputs_from_dict(json.loads(raw))


def build_tier_mask_observation(
    query_id: str,
    policy_version: str,
    inputs: AttentionTierMaskInputs,
    warm_per_window_penalties: dict[int, float],
    hot_byte_identity_check: bool,
) -> dict[str, Any]:
    """Emit the per-query transparency observation (contract-literal schema).

    Keys: ``query_id``, ``policy_version``, ``window_count_total``,
    ``tier_counts`` (``{"hot": N, "warm": N, "cold": N}``),
    ``warm_penalty_value``, ``warm_per_window_penalties``,
    ``cold_window_ids`` (sorted), ``hot_byte_identity_check``.
    """

    tier_counts = {
        TierLabel.HOT.value: 0,
        TierLabel.WARM.value: 0,
        TierLabel.COLD.value: 0,
    }
    cold_window_ids: list[int] = []
    for window_id, tier in inputs.tier_assignments.items():
        tier_counts[TierLabel(tier).value] += 1
        if tier == TierLabel.COLD:
            cold_window_ids.append(int(window_id))
    cold_window_ids.sort()

    return {
        "query_id": str(query_id),
        "policy_version": str(policy_version),
        "window_count_total": len(inputs.tier_assignments),
        "tier_counts": tier_counts,
        "warm_penalty_value": float(inputs.warm_config.penalty_value),
        "warm_per_window_penalties": {
            int(wid): float(penalty)
            for wid, penalty in warm_per_window_penalties.items()
        },
        "cold_window_ids": cold_window_ids,
        "hot_byte_identity_check": bool(hot_byte_identity_check),
    }
