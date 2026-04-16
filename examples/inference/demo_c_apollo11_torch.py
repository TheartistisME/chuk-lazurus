#!/usr/bin/env python3
"""
Experiment C - Apollo 11 torch/CUDA demo.

This is the narrow Apollo experiment path for Lazarus on CUDA:
route a single question through the Apollo store, load the chosen window
boundary, and generate through TorchInferenceRuntime with residual injection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import shorten
from typing import Any

from chuk_lazarus.inference.backends import LazarusBackend, ResidualState, TorchInferenceRuntime
from chuk_lazarus.inference.generation import GenerationConfig


DEFAULT_STORE = "/tmp/apollo-demo/apollo11_store"
DEFAULT_DEVICE = "cuda"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apollo 11 torch/CUDA demo")
    parser.add_argument("--store", default=DEFAULT_STORE, help="Apollo store directory")
    parser.add_argument(
        "--model",
        required=True,
        help="Local model path or Hugging Face model ID",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device to use")
    parser.add_argument("--question", required=True, help="Apollo 11 question to answer")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        dest="max_new_tokens",
        help="Maximum number of generated tokens",
    )
    return parser


def load_apollo_store(store_path: str | Path):
    from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore

    return TorchKnowledgeStore.load(Path(store_path))


def _load_torch_runtime(model_id: str, device: str) -> tuple[TorchInferenceRuntime, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    runtime = TorchInferenceRuntime(model, tokenizer, device=device)
    return runtime, tokenizer


def _as_cpu_torch_tensor(value):
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


# Use the canonical helpers from torch_query so this demo benefits from the
# wrapper-config unwrap (Gemma 4 keeps hidden_size under config.text_config).
from chuk_lazarus.inference.context.knowledge.torch_query import (  # noqa: E402
    _count_model_layers,
    _infer_model_hidden_size,
    _residual_is_compatible,
)


def _render_prompt(tokenizer: Any, question: str, window_text: str, window_keywords: list[str], mode: str) -> str:
    system = (
        "You answer Apollo 11 questions using only the Apollo store context "
        "provided by Lazarus. Be concise and quote exact wording when useful."
    )
    context_parts = []
    if window_text:
        context_parts.append(f"Transcript excerpt:\n{window_text}")
    if window_keywords:
        context_parts.append(
            "Apollo store keywords: " + ", ".join(window_keywords[:8])
        )
    context_parts.append(f"Execution mode: {mode}")
    context_parts.append(f"Question: {question}")
    user_content = "\n\n".join(context_parts)

    if callable(getattr(tokenizer, "apply_chat_template", None)):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        "<start_of_turn>system\n"
        f"{system}<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_content}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def run_apollo_demo(
    *,
    store_path: str | Path,
    model_id: str,
    device: str,
    question: str,
    max_new_tokens: int,
) -> int:
    store = load_apollo_store(store_path)
    runtime, tokenizer = _load_torch_runtime(model_id, device)

    window_id = store.route(question, method="keyword")
    routing_mode = "keyword"
    if window_id is None:
        window_ids = store.route_top_k(question, tokenizer, k=1)
        if not window_ids:
            raise RuntimeError("Apollo store could not route the question to any window")
        window_id = window_ids[0]
        routing_mode = "tfidf"

    window_text = store.get_window_text(window_id, tokenizer)
    window_keywords = list(store.keywords.get(window_id, [])) if getattr(store, "keywords", None) else []
    boundary = store.load_boundary(window_id, device="cpu")
    boundary_tensor = _as_cpu_torch_tensor(boundary)

    crystal_layer = int(store.config.crystal_layer)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    residual_compatible, residual_reason = _residual_is_compatible(
        runtime, boundary_tensor, crystal_layer
    )
    mode = "residual" if residual_compatible else "prompt-context"
    prompt = _render_prompt(tokenizer, question, window_text, window_keywords, mode)

    if residual_compatible:
        residual_state = ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=crystal_layer,
            tensor=boundary_tensor,
            sequence_length=int(
                len(store.window_token_lists.get(window_id, [])) or store.num_tokens
            ),
            hidden_size=int(boundary_tensor.shape[-1]),
            dtype=str(boundary_tensor.dtype).replace("torch.", ""),
            device="cpu",
        )
        result = runtime.generate_with_residual(prompt, residual_state, generation_config)
    else:
        result = runtime.generate(prompt, generation_config)

    preview = shorten(window_text.replace("\n", " "), width=180, placeholder="...")
    print("Apollo 11 torch/CUDA demo")
    print(f"Store: {Path(store_path)}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Backend: {runtime.backend.value}")
    print(f"Routing: {routing_mode}")
    print(f"Mode: {mode} ({residual_reason})")
    print(f"Question: {question}")
    print(f"Routed window: {window_id}")
    print(f"Window preview: {preview}")
    print(f"Boundary: layer={crystal_layer}, shape={tuple(boundary_tensor.shape)}, dtype={boundary_tensor.dtype}")
    print("Answer:")
    print(result.text.strip())
    print(f"Stats: {result.stats.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_apollo_demo(
        store_path=args.store,
        model_id=args.model,
        device=args.device,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
