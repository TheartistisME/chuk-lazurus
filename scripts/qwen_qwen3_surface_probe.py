#!/usr/bin/env python3
"""
Qwen3 KV-direct surface probe.

This is the operator-facing entrypoint for reproducible Qwen3 evidence.

It provides:
  - model load via UnifiedPipeline
  - standard generate via pipeline.generate()
  - Qwen3 KV-direct generate via a local backbone adapter
  - bounded hot-window behavior via KVDirectGenerator.slide()

Truthful limitations:
  - UnifiedPipeline.make_engine() is currently not Qwen3-aware in core.
  - The core BoundedKVEngine is Gemma-specific, so this script implements
    the bounded hot-window demo locally instead of claiming native support.

Usage:
    uv run python scripts/qwen_qwen3_surface_probe.py --help
    uv run python scripts/qwen_qwen3_surface_probe.py load
    uv run python scripts/qwen_qwen3_surface_probe.py generate --prompt "Explain KV-direct"
    uv run python scripts/qwen_qwen3_surface_probe.py bounded --budget-mb 2
    uv run python scripts/qwen_qwen3_surface_probe.py all
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.inference._lazy_mlx import mx, nn
from chuk_lazarus.inference.context import KVDirectGenerator


MODEL_ALIASES = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


@dataclass(frozen=True)
class CapabilityReport:
    model_id: str
    family: str
    loaded: bool
    native_make_engine_supported: bool | None = None
    native_make_engine_reason: str | None = None


class Qwen3LayerAdapter:
    """Adapter that exposes Qwen3 blocks through the KV-direct protocol."""

    def __init__(self, block: Any):
        self._block = block

    def pre_attn_norm(self, h: Any) -> Any:
        return self._block.input_layernorm(h)

    def project_qkv(self, x: Any, B: int, S: int, offset: int) -> tuple[Any, Any, Any]:
        attn = self._block.self_attn
        q, k, v = self._project_qkv(x, B, S, attn)
        q = attn.rope(q, offset=offset)
        k = attn.rope(k, offset=offset)
        return q, k, v

    def project_qkv_pre_rope(self, x: Any, B: int, S: int) -> tuple[Any, Any, Any]:
        attn = self._block.self_attn
        return self._project_qkv(x, B, S, attn)

    def _project_qkv(self, x: Any, B: int, S: int, attn: Any) -> tuple[Any, Any, Any]:
        q = attn.q_proj(x).reshape(B, S, attn.num_heads, attn.head_dim)
        k = attn.k_proj(x).reshape(B, S, attn.num_kv_heads, attn.head_dim)
        v = attn.v_proj(x).reshape(B, S, attn.num_kv_heads, attn.head_dim)
        q = attn.q_norm(q).transpose(0, 2, 1, 3)
        k = attn.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        return q, k, v

    def apply_rope(self, x: Any, offset: int) -> Any:
        return self._block.self_attn.rope(x, offset=offset)

    def output_project(self, attn_result: Any) -> Any:
        return self._block.self_attn.o_proj(attn_result)

    def residual_add_attn(self, h: Any, attn_out: Any) -> Any:
        return h + attn_out

    def pre_ffn_norm(self, h: Any) -> Any:
        return self._block.post_attention_layernorm(h)

    def ffn(self, x: Any) -> Any:
        return self._block.mlp(x)

    def residual_add_ffn(self, h: Any, ffn_out: Any) -> Any:
        return h + ffn_out

    @property
    def num_heads(self) -> int:
        return self._block.self_attn.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._block.self_attn.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._block.self_attn.head_dim

    @property
    def n_rep(self) -> int:
        return self._block.self_attn.n_rep

    @property
    def attn_scale(self) -> float:
        return self._block.self_attn.scale

    def head_output_projection(self, head_out: Any, head_idx: int) -> Any:
        o_weight = self._block.self_attn.o_proj.weight
        start = head_idx * self.head_dim
        stop = start + self.head_dim
        return mx.matmul(head_out, o_weight[:, start:stop].T)


class Qwen3BackboneAdapter:
    """Adapter for Qwen3ForCausalLM that satisfies ModelBackboneProtocol."""

    def __init__(self, model: Any):
        self._model = model
        self._backbone = model.model
        self._adapted = [Qwen3LayerAdapter(block) for block in self._backbone.layers]

    @property
    def adapted_layers(self) -> list[Qwen3LayerAdapter]:
        return self._adapted

    def embed(self, input_ids: Any) -> Any:
        return self._backbone.embed_tokens(input_ids)

    def unembed(self, h: Any) -> Any:
        return self._model.lm_head(h)

    def final_norm(self, h: Any) -> Any:
        return self._backbone.norm(h)

    def prefill_mask(self, layer_idx: int, h: Any) -> Any | None:
        _, seq_len, _ = h.shape
        if seq_len <= 1:
            return None
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        return mask.astype(h.dtype)

    def is_global_layer(self, layer_idx: int) -> bool:
        return getattr(self._model.config, "sliding_window", None) is None

    @property
    def sliding_window(self) -> int | None:
        return getattr(self._model.config, "sliding_window", None)

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    @property
    def embed_matrix(self) -> Any:
        return self._backbone.embed_tokens.weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 KV-direct surface probe")
    parser.add_argument("--model", choices=sorted(MODEL_ALIASES), default="qwen3-0.6b")
    parser.add_argument("--model-id", help="Override the HuggingFace model ID")
    subparsers = parser.add_subparsers(dest="command", required=False)

    load_parser = subparsers.add_parser("load", help="Load the model and report capabilities")
    load_parser.add_argument("--prompt", default="Describe KV-direct in one sentence.")

    gen_parser = subparsers.add_parser(
        "generate", help="Run standard generate and local KV-direct generate"
    )
    gen_parser.add_argument("--prompt", default="Explain how KV-direct differs from standard KV.")
    gen_parser.add_argument("--max-new-tokens", type=int, default=32)
    gen_parser.add_argument("--temperature", type=float, default=0.0)

    bounded_parser = subparsers.add_parser(
        "bounded", help="Run a bounded hot-window simulation using KV-direct sliding"
    )
    bounded_parser.add_argument("--budget-mb", type=float, default=2.0)
    bounded_parser.add_argument("--turns", type=int, default=4)
    bounded_parser.add_argument("--turn-input-tokens", type=int, default=48)
    bounded_parser.add_argument("--turn-gen-tokens", type=int, default=16)
    bounded_parser.add_argument(
        "--seed-text",
        default="Keep the state bounded while continuing the conversation.",
    )

    all_parser = subparsers.add_parser("all", help="Run load, generate, and bounded probes")
    all_parser.add_argument("--prompt", default="Explain how KV-direct differs from standard KV.")
    all_parser.add_argument("--max-new-tokens", type=int, default=32)
    all_parser.add_argument("--temperature", type=float, default=0.0)
    all_parser.add_argument("--budget-mb", type=float, default=2.0)
    all_parser.add_argument("--turns", type=int, default=4)
    all_parser.add_argument("--turn-input-tokens", type=int, default=48)
    all_parser.add_argument("--turn-gen-tokens", type=int, default=16)
    all_parser.add_argument(
        "--seed-text",
        default="Keep the state bounded while continuing the conversation.",
    )

    return parser.parse_args()


def resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return MODEL_ALIASES[args.model]


def encode_tokens(tokenizer: Any, text: str, target_tokens: int | None = None) -> list[int]:
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if not tokens:
        fallback = tokenizer.eos_token_id or tokenizer.bos_token_id or 1
        tokens = [int(fallback)]
    if target_tokens is None:
        return tokens
    if len(tokens) >= target_tokens:
        return tokens[:target_tokens]
    repeated: list[int] = []
    while len(repeated) < target_tokens:
        repeated.extend(tokens)
    return repeated[:target_tokens]


def load_pipeline(model_id: str) -> UnifiedPipeline:
    config = UnifiedPipelineConfig(backend_name="mlx")
    return UnifiedPipeline.from_pretrained(model_id, pipeline_config=config, verbose=True)


def probe_native_make_engine(pipeline: UnifiedPipeline) -> tuple[bool, str | None]:
    try:
        _engine = pipeline.make_engine()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def build_kv_generator(model: Any) -> KVDirectGenerator:
    return KVDirectGenerator(Qwen3BackboneAdapter(model))


def run_load_probe(pipeline: UnifiedPipeline, model_id: str) -> CapabilityReport:
    supported, reason = probe_native_make_engine(pipeline)
    print("\n[load]")
    print(f"model_id={model_id}")
    print(f"family={pipeline.family_type.value}")
    print(f"runtime={pipeline.runtime.__class__.__name__}")
    print(f"loaded={pipeline.state.is_loaded if pipeline.state else False}")
    print(f"tensor_count={pipeline.state.tensor_count if pipeline.state else 'unknown'}")
    print(f"hidden_size={pipeline.config.hidden_size}")
    print(f"num_layers={pipeline.config.num_hidden_layers}")
    print(f"native_make_engine={supported}")
    if reason:
        print(f"native_make_engine_reason={reason}")
    print("native_bounded_engine=False")
    print("native_bounded_engine_reason=core BoundedKVEngine is Gemma-specific")
    return CapabilityReport(
        model_id=model_id,
        family=pipeline.family_type.value,
        loaded=bool(pipeline.state and pipeline.state.is_loaded),
        native_make_engine_supported=supported,
        native_make_engine_reason=reason,
    )


def kv_direct_generate(
    tokenizer: Any,
    kv_gen: KVDirectGenerator,
    prompt_ids: Any,
    max_new_tokens: int,
) -> tuple[str, float, list[int]]:
    logits, kv_store = kv_gen.prefill(prompt_ids)
    seq_len = prompt_ids.shape[1]
    generated_ids: list[int] = []

    start = time.perf_counter()
    for _ in range(max_new_tokens):
        next_token = int(mx.argmax(logits[0, -1, :]))
        generated_ids.append(next_token)
        logits, kv_store = kv_gen.step_uncompiled(mx.array([[next_token]]), kv_store, seq_len)
        seq_len += 1
        mx.eval(logits)
    elapsed = time.perf_counter() - start

    return tokenizer.decode(generated_ids, skip_special_tokens=True), elapsed, generated_ids


def run_generate_probe(
    pipeline: UnifiedPipeline,
    kv_gen: KVDirectGenerator,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> None:
    print("\n[generate]")
    prompt_ids = mx.array(pipeline.tokenizer.encode(prompt, return_tensors="np"))
    standard = pipeline.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    standard_ids = pipeline.model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    mx.eval(standard_ids)
    kv_text, kv_elapsed, kv_ids = kv_direct_generate(
        pipeline.tokenizer,
        kv_gen,
        prompt_ids,
        max_new_tokens,
    )
    print(f"prompt={prompt!r}")
    print(f"standard_text={standard.text!r}")
    print(f"kv_direct_text={kv_text!r}")
    print(f"token_match={standard_ids[0, prompt_ids.shape[1]:].tolist() == kv_ids}")
    print(f"standard_toks_per_s={standard.stats.tokens_per_second:.2f}")
    print(f"kv_direct_toks_per_s={(max_new_tokens / kv_elapsed) if kv_elapsed > 0 else 0:.2f}")


def run_bounded_probe(
    pipeline: UnifiedPipeline,
    kv_gen: KVDirectGenerator,
    budget_mb: float,
    turns: int,
    turn_input_tokens: int,
    turn_gen_tokens: int,
    seed_text: str,
) -> None:
    tokenizer = pipeline.tokenizer
    budget_bytes = int(budget_mb * 1024 * 1024)
    per_token_bytes = kv_gen.kv_bytes(1)
    max_hot_tokens = budget_bytes // per_token_bytes
    if max_hot_tokens < 1:
        raise ValueError(
            f"budget_mb={budget_mb} is too small for one KV token; "
            f"need at least {per_token_bytes / 1024 / 1024:.3f} MB"
        )

    print("\n[bounded]")
    print(f"budget_bytes={budget_bytes}")
    print(f"kv_bytes_per_token={per_token_bytes}")
    print(f"max_hot_tokens={max_hot_tokens}")
    print(f"turns={turns}")
    print(f"turn_input_tokens={turn_input_tokens}")
    print(f"turn_gen_tokens={turn_gen_tokens}")

    kv_store: list[tuple[Any, Any]] | None = None
    total_tokens = 0
    window_start = 0

    print(
        "turn | path  | input | gen | hot_tokens | window_start | hot_bytes | budget_pct | evicted"
    )
    print("-" * 92)

    for turn in range(1, turns + 1):
        turn_text = f"Turn {turn}: {seed_text}"
        turn_ids = encode_tokens(tokenizer, turn_text, turn_input_tokens)
        turn_array = mx.array([turn_ids])

        if kv_store is None:
            logits, kv_store = kv_gen.prefill(turn_array)
        else:
            logits, kv_store = kv_gen.extend(turn_array, kv_store, abs_start=total_tokens)
        total_tokens += len(turn_ids)

        start = time.perf_counter()
        generated_ids: list[int] = []
        for _ in range(turn_gen_tokens):
            next_token = int(mx.argmax(logits[0, -1, :]))
            generated_ids.append(next_token)
            logits, kv_store = kv_gen.step_uncompiled(mx.array([[next_token]]), kv_store, total_tokens)
            total_tokens += 1
        gen_elapsed = time.perf_counter() - start

        hot_tokens = total_tokens - window_start
        evicted = 0
        if hot_tokens > max_hot_tokens:
            evicted = hot_tokens - max_hot_tokens
            kv_store = kv_gen.slide(kv_store, evicted)
            window_start += evicted
            hot_tokens -= evicted

        hot_bytes = kv_gen.kv_bytes(hot_tokens)
        budget_pct = hot_bytes * 100.0 / budget_bytes
        path = "grow" if window_start == 0 else "full"
        print(
            f"{turn:>4} | {path:<4} | {len(turn_ids):>5} | {len(generated_ids):>3} "
            f"| {hot_tokens:>10} | {window_start:>12} | {hot_bytes:>9} "
            f"| {budget_pct:>9.1f}% | {evicted:>7}"
        )
        print(
            f"     generated={tokenizer.decode(generated_ids, skip_special_tokens=True)!r} "
            f"tok_per_s={(turn_gen_tokens / gen_elapsed) if gen_elapsed > 0 else 0:.2f}"
        )

        if hot_tokens > max_hot_tokens:
            raise AssertionError("bounded hot state exceeded budget")

    print("\nbounded_state=exists")
    print(f"final_total_tokens={total_tokens}")
    print(f"final_window_start={window_start}")
    print(f"final_hot_tokens={total_tokens - window_start}")


def main() -> int:
    args = parse_args()
    model_id = resolve_model_id(args)

    try:
        pipeline = load_pipeline(model_id)
        kv_gen = build_kv_generator(pipeline.model)
    except Exception as exc:
        print(f"ERROR loading Qwen3 surface: {type(exc).__name__}: {exc}")
        return 2

    command = args.command or "all"
    if command in ("load", "all"):
        run_load_probe(pipeline, model_id)
    if command in ("generate", "all"):
        prompt = getattr(args, "prompt", "Explain how KV-direct differs from standard KV.")
        max_new_tokens = getattr(args, "max_new_tokens", 32)
        temperature = getattr(args, "temperature", 0.0)
        run_generate_probe(pipeline, kv_gen, prompt, max_new_tokens, temperature)
    if command in ("bounded", "all"):
        budget_mb = getattr(args, "budget_mb", 2.0)
        turns = getattr(args, "turns", 4)
        turn_input_tokens = getattr(args, "turn_input_tokens", 48)
        turn_gen_tokens = getattr(args, "turn_gen_tokens", 16)
        seed_text = getattr(args, "seed_text", "Keep the state bounded while continuing the conversation.")
        run_bounded_probe(
            pipeline,
            kv_gen,
            budget_mb=budget_mb,
            turns=turns,
            turn_input_tokens=turn_input_tokens,
            turn_gen_tokens=turn_gen_tokens,
            seed_text=seed_text,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
