"""
PyTorch/CUDA-backed inference runtime.
"""

from __future__ import annotations

import inspect
import time
from contextlib import nullcontext
from typing import Any

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
    ):
        super().__init__(model, tokenizer)
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._engine_mode = self._normalize_engine(engine)
        self._family_type = getattr(family_type, "value", family_type)
        self._last_generation_path: str | None = None
        self._cached_kv_direct_forward_kwargs: dict[str, Any] | None = None

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
        if normalized in {"standard", "kv_direct"}:
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

    def generate(self, prompt: str, config):
        self._set_generation_path(None)
        if self._engine_mode == "kv_direct":
            return self._generate_kv_direct(prompt, config)
        return self._generate_standard(prompt, config)

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

    def clear_cache(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
