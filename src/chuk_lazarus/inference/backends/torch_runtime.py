"""
PyTorch/CUDA-backed inference runtime.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from .base import InferenceRuntime
from .types import LazarusBackend, ResidualState


class TorchInferenceRuntime(InferenceRuntime):
    """Inference runtime for Hugging Face causal LM models on CUDA."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        super().__init__(model, tokenizer)
        import torch

        self._torch = torch
        self._device = torch.device(device)

    @property
    def backend(self) -> LazarusBackend:
        return LazarusBackend.CUDA

    def _resolve_layers(self) -> list[Any]:
        """Locate transformer layers across common Hugging Face architectures."""
        candidates = [
            ("model", "layers"),
            ("transformer", "h"),
            ("gpt_neox", "layers"),
            (None, "layers"),
        ]

        for outer_name, inner_name in candidates:
            target = self._model
            if outer_name is not None:
                target = getattr(target, outer_name, None)
            if target is None:
                continue

            layers = getattr(target, inner_name, None)
            if layers is not None:
                return list(layers)

        raise ValueError(
            "Cannot resolve transformer layers for residual hooks. "
            "Expected model.model.layers, model.transformer.h, gpt_neox.layers, or model.layers."
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

    def _generation_kwargs(self, config, model_inputs: dict[str, Any], *, use_cache: bool) -> dict[str, Any]:
        """Build generate() kwargs while avoiding noisy ignored sampling flags."""
        generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
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

    def generate(self, prompt: str, config):
        from ..generation import GenerationResult, GenerationStats, StopReason

        model_inputs = self._tokenize_prompt(prompt)
        input_ids = model_inputs["input_ids"]
        input_length = int(input_ids.shape[1])
        start_time = time.time()
        generation_kwargs = self._generation_kwargs(config, model_inputs, use_cache=True)

        total_window_tokens = input_length + config.max_new_tokens
        with self._torch.inference_mode(), self._generation_context(total_window_tokens):
            output_ids = self._model.generate(input_ids=input_ids, **generation_kwargs)

        new_tokens = output_ids[:, input_length:]
        new_tokens_cpu = new_tokens.to("cpu")
        generated_text = self._tokenizer.decode(new_tokens_cpu[0].tolist(), skip_special_tokens=True)
        output_length = int(new_tokens.shape[1])
        gen_time = time.time() - start_time

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
            tokens_per_second=output_length / gen_time if gen_time > 0 else 0,
        )
        return GenerationResult(text=generated_text, stats=stats, stop_reason=stop_reason)

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

    def clear_cache(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
