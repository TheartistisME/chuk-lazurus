"""Torch/CUDA maintained knowledge query and chat helpers."""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import shorten
from typing import Any

from ...backends import LazarusBackend, ResidualState, TorchInferenceRuntime
from ...generation import GenerationConfig
from .torch_store import TorchKnowledgeStore

_SYSTEM_PROMPT = (
    "You answer using only the retrieved knowledge-store context. "
    "Be concise and say when the store context is insufficient."
)


@dataclass(frozen=True)
class TorchKnowledgeResponse:
    text: str
    mode: str
    routing_mode: str
    window_ids: list[int]
    expansion_terms: list[str]
    stats_summary: str
    residual_reason: str
    boundary_shape: tuple[int, ...] | None = None
    context_preview: str = ""


def _load_torch_runtime(model_id: str, device: str | None) -> tuple[TorchInferenceRuntime, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or "cuda"
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(resolved_device)
    model.eval()
    runtime = TorchInferenceRuntime(model, tokenizer, device=resolved_device)
    return runtime, tokenizer


def _encode_token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def _as_cpu_torch_tensor(value: Any):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu")
    else:
        tensor = torch.as_tensor(np.asarray(value), device="cpu")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(1, -1)
    elif tensor.ndim == 2 and tensor.shape[0] != 1:
        tensor = tensor[:1]

    return tensor.contiguous()


def _infer_model_hidden_size(runtime: TorchInferenceRuntime) -> int | None:
    model = getattr(runtime, "_model", None)
    if model is None:
        return None

    # Build candidate sources. VLM wrappers (e.g. Gemma 4) keep the text-backbone
    # dimensions under config.text_config, not on the top-level config.
    candidates: list[Any] = []
    config = getattr(model, "config", None)
    candidates.append(config)
    if config is not None:
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            candidates.append(text_config)
    candidates.append(model)

    for source in candidates:
        if source is None:
            continue
        for attr in ("hidden_size", "n_embd", "d_model", "dim", "hidden_dim"):
            value = getattr(source, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    # Fall back to the embedding matrix if configs hide the value.
    try:
        embeddings = model.get_input_embeddings()
    except (AttributeError, TypeError):
        embeddings = None
    if embeddings is not None:
        dim = getattr(embeddings, "embedding_dim", None)
        if isinstance(dim, int) and dim > 0:
            return dim
        weight = getattr(embeddings, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape is not None and len(shape) >= 2:
            try:
                return int(shape[-1])
            except (TypeError, ValueError):
                pass
    return None


def _count_model_layers(runtime: TorchInferenceRuntime) -> int | None:
    try:
        return len(runtime._resolve_layers())
    except Exception:
        return None


def _residual_is_compatible(
    runtime: TorchInferenceRuntime,
    boundary_tensor,
    layer_index: int,
) -> tuple[bool, str]:
    layer_count = _count_model_layers(runtime)
    if layer_count is None:
        return False, "layer count unavailable"
    if layer_index < 0 or layer_index >= layer_count:
        return False, f"layer index {layer_index} is outside the model's {layer_count} layers"

    hidden_size = _infer_model_hidden_size(runtime)
    if hidden_size is None:
        return False, "model hidden size unavailable"

    width = int(boundary_tensor.shape[-1])
    if width != hidden_size:
        return False, f"boundary width {width} does not match model hidden size {hidden_size}"

    return True, "boundary and model shapes are compatible"


def _render_prompt(
    tokenizer: Any,
    *,
    question: str,
    context_blocks: list[str],
    window_keywords: list[str],
    mode: str,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    use_chat_template: bool = True,
) -> str:
    system_text = system_prompt or _SYSTEM_PROMPT
    messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]
    if history:
        messages.extend(history)

    user_sections: list[str] = []
    if context_blocks:
        user_sections.append("Knowledge store context:\n" + "\n\n---\n\n".join(context_blocks))
    if window_keywords:
        user_sections.append("Retrieved keywords: " + ", ".join(window_keywords[:12]))
    user_sections.append(f"Execution mode: {mode}")
    user_sections.append(f"Question: {question}")
    messages.append({"role": "user", "content": "\n\n".join(user_sections)})

    if use_chat_template and callable(getattr(tokenizer, "apply_chat_template", None)):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    parts = []
    for message in messages:
        parts.append(f"{message['role'].upper()}:\n{message['content']}")
    parts.append("ASSISTANT:\n")
    return "\n\n".join(parts)


def _extract_expansion_terms(expansion_text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", expansion_text):
        lowered = match.lower()
        if len(lowered) < 2 or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
    return terms


def _expand_query(
    query_text: str,
    tokenizer: Any,
    runtime: TorchInferenceRuntime,
    *,
    n_tokens: int = 30,
) -> tuple[list[int], list[str]]:
    prompt = (
        f"Q: {query_text}\n"
        "A: The specific distinctive words and phrases from this event are:"
    )
    result = runtime.generate(
        prompt,
        GenerationConfig(max_new_tokens=n_tokens, temperature=0.0, top_p=1.0),
    )
    terms = _extract_expansion_terms(result.text)

    expansion_ids: list[int] = []
    seen_ids: set[int] = set()
    for term in terms:
        for variant in (term, f" {term}", term.title(), f" {term.title()}"):
            for token_id in _encode_token_ids(tokenizer, variant, add_special_tokens=False):
                if token_id not in seen_ids:
                    seen_ids.add(token_id)
                    expansion_ids.append(token_id)

    return expansion_ids, terms


def _merge_keywords(store: TorchKnowledgeStore, window_ids: list[int]) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for window_id in window_ids:
        for keyword in store.keywords.get(window_id, []):
            value = keyword.strip()
            lowered = value.lower()
            if value and lowered not in seen:
                seen.add(lowered)
                keywords.append(value)
    return keywords


def _prepare_store_response(
    *,
    store: TorchKnowledgeStore,
    runtime: TorchInferenceRuntime,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    history: list[dict[str, str]] | None = None,
    system_prompt: str | None = None,
    no_chat_template: bool = False,
) -> TorchKnowledgeResponse:
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    expansion_ids, expansion_terms = _expand_query(question, tokenizer, runtime)
    window_ids = store.route_top_k(question, tokenizer, k=top_k, expansion_ids=expansion_ids)
    routing_mode = "tfidf"
    if not window_ids:
        fallback_window = store.route(question, tokenizer=tokenizer, method="auto")
        if fallback_window is not None:
            window_ids = [fallback_window]
            routing_mode = "auto"

    if not window_ids:
        prompt = _render_prompt(
            tokenizer,
            question=question,
            context_blocks=[],
            window_keywords=[],
            mode="plain",
            history=history,
            system_prompt=system_prompt,
            use_chat_template=not no_chat_template,
        )
        result = runtime.generate(prompt, generation_config)
        return TorchKnowledgeResponse(
            text=result.text,
            mode="plain",
            routing_mode="miss",
            window_ids=[],
            expansion_terms=expansion_terms,
            stats_summary=result.stats.summary,
            residual_reason="no matching store window",
        )

    context_blocks = [store.get_window_text(window_id, tokenizer) for window_id in window_ids]
    window_keywords = _merge_keywords(store, window_ids)
    boundary_shape: tuple[int, ...] | None = None
    residual_reason = "boundary unavailable"
    mode = "prompt-context"
    prompt_mode = "context-replay"

    prompt = _render_prompt(
        tokenizer,
        question=question,
        context_blocks=context_blocks,
        window_keywords=window_keywords,
        mode=prompt_mode,
        history=history,
        system_prompt=system_prompt,
        use_chat_template=not no_chat_template,
    )

    try:
        boundary = store.load_boundary(window_ids[0], device="cpu")
        boundary_tensor = _as_cpu_torch_tensor(boundary)
        boundary_shape = tuple(boundary_tensor.shape)
        crystal_layer = int(store.config.crystal_layer)
        residual_compatible, residual_reason = _residual_is_compatible(
            runtime,
            boundary_tensor,
            crystal_layer,
        )
        if residual_compatible:
            prompt_mode = "markov-boundary"
            prompt = _render_prompt(
                tokenizer,
                question=question,
                context_blocks=context_blocks,
                window_keywords=window_keywords,
                mode=prompt_mode,
                history=history,
                system_prompt=system_prompt,
                use_chat_template=not no_chat_template,
            )
            residual_state = ResidualState(
                backend=LazarusBackend.CUDA,
                layer_index=crystal_layer,
                tensor=boundary_tensor,
                sequence_length=int(
                    len(store.window_token_lists.get(window_ids[0], [])) or store.num_tokens
                ),
                hidden_size=int(boundary_tensor.shape[-1]),
                dtype=str(boundary_tensor.dtype).replace("torch.", ""),
                device="cpu",
            )
            result = runtime.generate_with_residual(prompt, residual_state, generation_config)
            mode = "residual"
        else:
            result = runtime.generate(prompt, generation_config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        residual_reason = str(exc)
        result = runtime.generate(prompt, generation_config)

    preview = shorten(
        " ".join(block.replace("\n", " ") for block in context_blocks),
        width=180,
        placeholder="...",
    )
    return TorchKnowledgeResponse(
        text=result.text,
        mode=mode,
        routing_mode=routing_mode,
        window_ids=window_ids,
        expansion_terms=expansion_terms,
        stats_summary=result.stats.summary,
        residual_reason=residual_reason,
        boundary_shape=boundary_shape,
        context_preview=preview,
    )


def run_torch_query_command(
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str | None = None,
    store_path: str | Path | None = None,
    system_prompt: str | None = None,
    no_chat_template: bool = False,
    stdout=sys.stdout,
    stderr=sys.stderr,
) -> TorchKnowledgeResponse:
    runtime, tokenizer = _load_torch_runtime(model_id, device)
    if store_path is None:
        question_prompt = _render_prompt(
            tokenizer,
            question=prompt,
            context_blocks=[],
            window_keywords=[],
            mode="plain",
            system_prompt=system_prompt,
            use_chat_template=not no_chat_template,
        )
        result = runtime.generate(
            question_prompt,
            GenerationConfig(max_new_tokens=max_new_tokens, temperature=temperature, top_p=1.0),
        )
        print("No knowledge store - plain torch inference.", file=stderr)
        stdout.write(result.text.rstrip() + "\n")
        stdout.flush()
        return TorchKnowledgeResponse(
            text=result.text,
            mode="plain",
            routing_mode="plain",
            window_ids=[],
            expansion_terms=[],
            stats_summary=result.stats.summary,
            residual_reason="no knowledge store",
        )

    store = TorchKnowledgeStore.load(store_path)
    print(f"Loading knowledge store: {Path(store_path)}", file=stderr)
    store.log_stats(file=stderr)

    start_time = time.monotonic()
    response = _prepare_store_response(
        store=store,
        runtime=runtime,
        tokenizer=tokenizer,
        question=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        system_prompt=system_prompt,
        no_chat_template=no_chat_template,
    )
    elapsed_ms = (time.monotonic() - start_time) * 1000

    if response.expansion_terms:
        print(
            f"  Query expansion: {', '.join(response.expansion_terms[:15])}",
            file=stderr,
        )
    if response.window_ids:
        print(
            f"  Routed to windows {response.window_ids} via {response.routing_mode}",
            file=stderr,
        )
        print(
            f"  Context mode: {response.mode} ({response.residual_reason})",
            file=stderr,
        )
        if response.boundary_shape is not None:
            print(f"  Boundary shape: {response.boundary_shape}", file=stderr)
        if response.context_preview:
            print(f"  Window preview: {response.context_preview}", file=stderr)
    else:
        print("  No matching windows - plain torch inference.", file=stderr)

    print(f"  {response.stats_summary} ({elapsed_ms:.0f} ms total)", file=stderr)
    stdout.write(response.text.rstrip() + "\n")
    stdout.flush()
    runtime.clear_cache()
    return response


def run_torch_chat_command(
    *,
    model_id: str,
    store_path: str | Path,
    max_new_tokens: int,
    temperature: float,
    device: str | None = None,
    input_fn=input,
    stdout=sys.stdout,
    stderr=sys.stderr,
) -> int:
    runtime, tokenizer = _load_torch_runtime(model_id, device)
    store = TorchKnowledgeStore.load(store_path)
    print(f"Loading knowledge store: {Path(store_path)}", file=stderr)
    store.log_stats(file=stderr)
    print(file=stderr)

    history: list[dict[str, str]] = []
    while True:
        try:
            prompt_text = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            print(file=stderr)
            break

        if not prompt_text.strip():
            continue

        response = _prepare_store_response(
            store=store,
            runtime=runtime,
            tokenizer=tokenizer,
            question=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=3,
            history=history,
        )

        stdout.write(response.text.rstrip() + "\n")
        stdout.flush()
        history.append({"role": "user", "content": prompt_text})
        history.append({"role": "assistant", "content": response.text.strip()})

    runtime.clear_cache()
    return 0


__all__ = [
    "TorchKnowledgeResponse",
    "run_torch_chat_command",
    "run_torch_query_command",
]
