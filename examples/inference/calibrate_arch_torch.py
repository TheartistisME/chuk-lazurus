#!/usr/bin/env python3
"""
Torch/CUDA port of copy-head discovery for ArchitectureConfig calibration.

Mirrors the MLX script examples/inference/calibrate_arch.py using the
transformers AutoModelForCausalLM + forward-hook mechanism so it works
on any model loadable via transformers (Gemma 3/4, Llama, SmolLM, ...).

Two-part behavioral calibration:

  Part A — Injection layer scan:
    For each layer L in the top third of the stack, forward the fact document
    and capture the residual at the answer-token position. Compute
        c_L = dot(h_L[answer_pos], embed(answer))
    averaged over probes. The peak layer is the injection layer;
    retrieval_layer = injection_layer - 1.

  Part B — Causal head ablation:
    At retrieval_layer, for each query head H register a pre-forward hook on
    self_attn.o_proj that zeroes the slice [H*head_dim : (H+1)*head_dim] of
    the input (the concatenated per-head attention output). Measure
        delta_p = P(answer | baseline) - P(answer | head_H_zeroed)
    on the repeat-completion cloze prefix. The head with the highest
        score = mean(delta) * coverage
    is the copy head.

Usage
-----
    uv run python examples/inference/calibrate_arch_torch.py \
        --model /path/to/gemma-4-E2B-it --device cuda
    uv run python examples/inference/calibrate_arch_torch.py \
        --model /path/to/gemma-4-E2B-it --device cuda --top-layers 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

# Same probes as the MLX calibrator — entity-copy cloze completions.
PROBES = [
    {
        "doc": (
            "Zarkov Industries was established in the mid-1990s as a pioneering manufacturer "
            "of industrial filtration systems. Its headquarters, built on a former industrial "
            "lot, became a landmark of Voltara's commercial district. Today Zarkov Industries "
            "employs over 2,400 people.\n\n"
            "What city was Zarkov Industries founded in?"
        ),
        "cloze_prefix": (
            "Zarkov Industries is based in Voltara. "
            "Zarkov Industries is based in"
        ),
        "answer": "Voltara",
    },
    {
        "doc": (
            "Nexaris Corporation began as a small software consultancy before pivoting to "
            "enterprise data management. The founding team worked out of a converted warehouse "
            "in Cerulion's technology corridor.\n\n"
            "What city was Nexaris Corporation founded in?"
        ),
        "cloze_prefix": (
            "Nexaris Corporation is headquartered in Cerulion. "
            "Nexaris Corporation is headquartered in"
        ),
        "answer": "Cerulion",
    },
    {
        "doc": (
            "Helion Systems traces its origins to a university spin-out that relocated from "
            "campus to leased office space in Dravenport's innovation quarter in 2003.\n\n"
            "What city was Helion Systems founded in?"
        ),
        "cloze_prefix": (
            "Helion Systems is located in Dravenport. "
            "Helion Systems is located in"
        ),
        "answer": "Dravenport",
    },
    {
        "doc": (
            "Keltara Dynamics was incorporated in Solmere following a management buyout.\n\n"
            "What city was Keltara Dynamics founded in?"
        ),
        "cloze_prefix": (
            "Keltara Dynamics is incorporated in Solmere. "
            "Keltara Dynamics is incorporated in"
        ),
        "answer": "Solmere",
    },
    {
        "doc": (
            "Joe Namath was approached by Fabergé Inc. regarding a promotional campaign. "
            "After negotiations, Namath agreed to endorse Brut cologne.\n\n"
            "What did Joe Namath agree to do?"
        ),
        "cloze_prefix": (
            "Joe Namath agreed to endorse Brut cologne. "
            "Joe Namath agreed to"
        ),
        "answer": "endorse",
    },
    {
        "doc": (
            "Sylvia Marchand, a retired art dealer based in Geneva, followed advice from "
            "her estate lawyers. She agreed to sell her painting through a major auction house.\n\n"
            "What did Sylvia Marchand agree to do?"
        ),
        "cloze_prefix": (
            "Sylvia Marchand agreed to sell her painting. "
            "Sylvia Marchand agreed to"
        ),
        "answer": "sell",
    },
]


def _resolve_layers(model):
    """Same resolution order as TorchInferenceRuntime._resolve_layers."""
    candidates = [
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        (None, "layers"),
    ]
    for path in candidates:
        target = model
        *outer, inner = path
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
        layers = getattr(target, inner, None)
        if layers is None or not hasattr(layers, "__len__"):
            continue
        if len(layers) == 0:
            continue
        return list(layers)
    raise RuntimeError("Could not locate transformer layers in model")


def _first_answer_token_id(tokenizer, answer: str) -> int:
    """Return the id of the first sub-token that represents the answer word.

    We try both ' answer' (leading space, common BPE form) and 'answer'.
    """
    for variant in (f" {answer}", answer):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            return int(ids[0])
    raise ValueError(f"Tokenizer produced no tokens for answer={answer!r}")


def _locate_answer_position(tokenizer, doc_text: str, answer: str) -> tuple[list[int], int]:
    """Tokenize the full prompt and return (token_ids, last_answer_token_pos).

    We find the last occurrence of the answer-token in the tokenized prompt;
    the residual at that position is what Part A measures.
    """
    # Full instruct-style prompt for document question.
    prompt = (
        f"{doc_text}\n\nAnswer: The answer is {answer}."
    )
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_tid = _first_answer_token_id(tokenizer, answer)
    positions = [i for i, t in enumerate(ids) if int(t) == answer_tid]
    if not positions:
        # Fall back to the non-space variant.
        alt = tokenizer.encode(answer, add_special_tokens=False)
        if alt:
            answer_tid = int(alt[0])
            positions = [i for i, t in enumerate(ids) if int(t) == answer_tid]
    if not positions:
        raise RuntimeError(f"Answer token for {answer!r} not found in tokenized prompt")
    return list(ids), positions[-1]


def _capture_layer_residuals(model, layers, token_ids, device):
    """Forward once, capture every layer's output tensor at all positions."""
    import torch

    captures: dict[int, "torch.Tensor"] = {}
    handles = []
    for idx, layer in enumerate(layers):
        def make_hook(i):
            def hook(_mod, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[i] = hidden.detach()
            return hook
        handles.append(layer.register_forward_hook(make_hook(idx)))

    input_ids = torch.as_tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    try:
        with torch.inference_mode():
            try:
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            except TypeError:
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    return captures


def _embed_vector(model, token_id: int):
    emb = model.get_input_embeddings()
    import torch

    weight = emb.weight
    return weight[token_id].detach().to(dtype=torch.float32, device=weight.device)


def scan_injection_layers(model, tokenizer, probes, first_layer: int, device) -> tuple[int, int]:
    """Part A — find peak layer for dot(h_L, embed(answer))."""
    import torch

    layers = _resolve_layers(model)
    num_layers = len(layers)
    per_layer_scores: dict[int, list[float]] = {i: [] for i in range(first_layer, num_layers)}

    for probe in probes:
        token_ids, ans_pos = _locate_answer_position(tokenizer, probe["doc"], probe["answer"])
        # We want the residual at position ans_pos-1, because that is where the
        # model must *produce* the answer — same convention as the MLX script.
        target_pos = max(0, ans_pos - 1)
        ans_tid = _first_answer_token_id(tokenizer, probe["answer"])
        embed_vec = _embed_vector(model, ans_tid)

        captures = _capture_layer_residuals(model, layers, token_ids, device)
        for L in range(first_layer, num_layers):
            h = captures[L][0, target_pos, :].to(dtype=torch.float32)
            score = float(torch.dot(h, embed_vec).item())
            per_layer_scores[L].append(score)

    avg: dict[int, float] = {L: sum(s) / len(s) for L, s in per_layer_scores.items() if s}
    inject_layer = max(avg, key=lambda L: avg[L])
    retrieval_layer = max(0, inject_layer - 1)

    print(f"\n{BOLD}Part A — Injection layer scan{RESET}")
    for L in sorted(avg):
        marker = "  <-- peak" if L == inject_layer else ""
        print(f"  L{L:3d}: avg_dot = {avg[L]:+.3f}{marker}")
    print(f"  => retrieval_layer = {retrieval_layer}, injection_layer = {inject_layer}")
    return inject_layer, retrieval_layer


def _softmax_prob(model, tokenizer, prompt: str, answer: str, device) -> float:
    """Return P(first answer token | prompt)."""
    import torch

    ans_tid = _first_answer_token_id(tokenizer, answer)
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.as_tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        try:
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1, :].to(dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[ans_tid].item())


def _head_ablation_context(attn_module, head_index: int, head_dim: int):
    """Return a handle that zeroes head_index's o_proj input slice while active."""

    def pre_hook(_mod, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        # x shape: (B, S, num_heads * head_dim) just before o_proj.
        ablated = x.clone()
        start = head_index * head_dim
        end = start + head_dim
        ablated[..., start:end] = 0
        return (ablated, *inputs[1:])

    return attn_module.o_proj.register_forward_pre_hook(pre_hook)


def scan_routing_heads(
    model,
    tokenizer,
    probes,
    retrieval_layer: int,
    device,
) -> tuple[int, float, int]:
    """Part B — causal head ablation at retrieval_layer.

    Returns (best_head, score, num_heads_considered).
    """
    layers = _resolve_layers(model)
    attn = layers[retrieval_layer].self_attn
    # Infer head structure from o_proj input width and attn attributes.
    o_in_dim = attn.o_proj.weight.shape[1]
    num_heads = int(getattr(attn, "num_heads", getattr(attn.config if hasattr(attn, "config") else object(), "num_attention_heads", 0)) or 0)
    if num_heads == 0:
        # Fall back to model config num_attention_heads from text_config if present.
        cfg = getattr(model, "config", None)
        text_cfg = getattr(cfg, "text_config", None) if cfg is not None else None
        for src in (text_cfg, cfg):
            if src is None:
                continue
            n = getattr(src, "num_attention_heads", 0)
            if n:
                num_heads = int(n)
                break
    head_dim = o_in_dim // num_heads if num_heads else getattr(attn, "head_dim", 0)
    assert num_heads * head_dim == o_in_dim, (
        f"num_heads * head_dim ({num_heads} * {head_dim}) != o_proj in dim ({o_in_dim})"
    )

    print(f"\n{BOLD}Part B — Causal head ablation at L{retrieval_layer}{RESET}")
    print(f"  num_heads={num_heads}, head_dim={head_dim}, o_proj_in={o_in_dim}")

    # Baselines.
    baselines = []
    for probe in probes:
        baselines.append(_softmax_prob(model, tokenizer, probe["cloze_prefix"], probe["answer"], device))

    scores = {}
    for H in range(num_heads):
        handle = _head_ablation_context(attn, H, head_dim)
        try:
            deltas = []
            for probe, p_base in zip(probes, baselines, strict=True):
                p_abl = _softmax_prob(model, tokenizer, probe["cloze_prefix"], probe["answer"], device)
                deltas.append(p_base - p_abl)
        finally:
            handle.remove()
        mean_d = sum(deltas) / len(deltas)
        coverage = sum(1 for d in deltas if d >= 0.05) / len(deltas)
        score = mean_d * coverage
        scores[H] = (score, mean_d, coverage, deltas)
        print(
            f"  H{H}: mean_delta={mean_d:+.3f}  coverage={coverage:.2f}  "
            f"score={score:+.4f}  deltas=[{', '.join(f'{d:+.3f}' for d in deltas)}]"
        )

    best_head = max(scores, key=lambda H: scores[H][0])
    best_score = scores[best_head][0]
    print(f"  => query_head = {best_head} (score={best_score:+.4f})")
    return best_head, best_score, num_heads


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model path or HF id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top-layers", type=int, default=None, help="Scan only the top N layers")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} (dtype={args.dtype}, device={args.device})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, low_cpu_mem_usage=True)
    model.to(args.device)
    model.eval()

    text_cfg = getattr(config, "text_config", None) or config
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
    hidden_size = int(getattr(text_cfg, "hidden_size", 0))
    model_type = str(getattr(config, "model_type", "?")).lower()
    text_model_type = str(getattr(text_cfg, "model_type", model_type)).lower()

    print(
        f"{BOLD}Model:{RESET} wrapper={model_type}, text={text_model_type}, "
        f"num_layers={num_layers}, hidden={hidden_size}"
    )

    first_layer = (num_layers * 2) // 3
    if args.top_layers is not None:
        first_layer = max(0, num_layers - args.top_layers)

    inject_layer, retrieval_layer = scan_injection_layers(
        model, tokenizer, PROBES, first_layer, args.device
    )
    best_head, score, num_heads = scan_routing_heads(
        model, tokenizer, PROBES, retrieval_layer, args.device
    )

    # Normalisation for the registry key.
    norm_type = text_model_type
    if norm_type in ("gemma", "gemma2", "gemma3", "gemma3_text"):
        norm_type = "gemma"
    elif norm_type in ("gemma4", "gemma4_text"):
        norm_type = "gemma4"
    elif norm_type in ("mistral", "codellama", "smollm", "llama4"):
        norm_type = "llama"

    summary = {
        "model": args.model,
        "wrapper_model_type": model_type,
        "text_model_type": text_model_type,
        "registry_model_type": norm_type,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "retrieval_layer": retrieval_layer,
        "query_head": best_head,
        "injection_layer": inject_layer,
        "routing_score": score,
    }

    print(f"\n{'═' * 60}")
    print(f"{BOLD}Result:{RESET}")
    for k, v in summary.items():
        print(f"  {k} = {v}")

    print(f"\n{BOLD}Add to arch_config.py:{RESET}")
    print(
        f'\n  ArchitectureConfig._KNOWN[("{norm_type}", {num_layers})] = ArchitectureConfig(\n'
        f"      retrieval_layer={retrieval_layer},\n"
        f"      query_head={best_head},\n"
        f"      injection_layer={inject_layer},\n"
        f"  )\n"
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote JSON summary to {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
