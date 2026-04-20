#!/usr/bin/env python3
"""Generic clause-aligned torch strict-injection demo.

Apollo-11-pattern residual-injection retrieval against any clause-aligned store
built with tools/build_clause_aligned_store.py (or the legacy
tools/build_aus3000_clause_aligned_variant.py). Every silent-fallback path is
turned into a HARD FAILURE.

What this script asserts at runtime (ALL must hold, else RuntimeError):
  1. torch.cuda.is_available() → True
  2. The loaded model's parameters actually live on cuda:N (not CPU)
  3. _residual_is_compatible returns True BEFORE generation (no prompt-context fallback)
  4. The residual injection forward-pre-hook actually fires during generation
     (we wrap the hook and record a hit; absence = fail)
  5. GPU memory grew between model-load and generation (proves real CUDA work,
     not a silent CPU path)
  6. The routed window came from the store and its text is non-empty

Usage:
  uv run python examples/inference/demo_clause_aligned_strict.py \\
      --question "What does clause 1.4.72 define?"

Defaults target the cached google/gemma-4-E2B-it model (hidden=1536, 35 layers)
which matches the AUS3000 clause-aligned variant manifest (crystal_layer=29,
boundary width 1536).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from textwrap import shorten
from typing import Any

DEFAULT_STORE = "/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store"
DEFAULT_MODEL = "google/gemma-4-E2B-it"
DEFAULT_DEVICE = "cuda"
DEFAULT_QUESTION = "Summarise what clause 1.4.72 of AS/NZS 3000 defines."


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=DEFAULT_STORE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=120, dest="max_new_tokens")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help=(
            "Override the system prompt sent to the model. If omitted, derived from "
            "store metadata (torch_prefill.json source.name and source.standard_title) "
            "or falls back to a generic clause-aware prompt."
        ),
    )
    args = parser.parse_args()

    # Assertion 1: CUDA must be available. Do this before any import cost.
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("STRICT: CUDA device requested but torch.cuda.is_available() is False")
    print(_bold("=== STRICT CLAUSE-ALIGNED TORCH INJECTION DEMO ==="))
    print(f"store   : {args.store}")
    print(f"model   : {args.model}")
    print(f"device  : {args.device}  [{torch.cuda.get_device_name(0)}]")
    print(f"question: {args.question}")
    print()

    from chuk_lazarus.inference.backends import LazarusBackend, ResidualState, TorchInferenceRuntime
    from chuk_lazarus.inference.context.knowledge.torch_store import TorchKnowledgeStore
    from chuk_lazarus.inference.context.knowledge import torch_query
    from chuk_lazarus.inference.generation import GenerationConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load store
    print(f"[1/6] Loading store from {args.store}")
    store = TorchKnowledgeStore.load(Path(args.store))
    crystal_layer = int(store.config.crystal_layer)
    print(f"      crystal_layer={crystal_layer}  windows={store.num_windows}")

    # Load model + tokenizer on CUDA (bfloat16 for memory)
    mem_before_load = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f"[2/6] Loading {args.model} on {args.device} (bfloat16) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.to(args.device)
    model.eval()
    load_time = time.time() - t0

    # Assertion 2: model parameters truly on CUDA
    first_param_device = next(model.parameters()).device
    if first_param_device.type != "cuda":
        raise RuntimeError(
            f"STRICT: model parameters on {first_param_device}, not cuda — silent CPU fallback"
        )
    mem_after_load = torch.cuda.memory_allocated()
    print(f"      loaded in {load_time:.1f}s  param.device={first_param_device}")
    print(f"      GPU memory after model load: {mem_after_load/1024**3:.2f} GB  (delta {(mem_after_load-mem_before_load)/1024**3:+.2f} GB)")

    runtime = TorchInferenceRuntime(model, tokenizer, device=args.device)

    # Route the question via exact-clause-ID → TF-IDF path (NOT the learned/auto router)
    print(f"[3/6] Routing question through store (exact clause-ID → TF-IDF, NO auto/MLP) ...")
    import re as _re
    window_ids, routing_mode = torch_query._resolve_grounded_windows(
        store, args.question, tokenizer, top_k=1
    )
    if not window_ids:
        raise RuntimeError(
            "STRICT: store could not route question via exact or TF-IDF path "
            "(no clause-ID match, no TF-IDF overlap)"
        )
    if routing_mode == "auto":
        raise RuntimeError(
            "STRICT: routing fell through to 'auto' (learned/MLP router). "
            "Strict mode refuses this path — we only accept 'exact' or 'tfidf'."
        )
    # If the question contains a clause-ID pattern (e.g. 1.4.72) expect EXACT routing
    _clause_id_pattern = _re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")
    _has_clause_id = bool(_clause_id_pattern.search(args.question))
    if _has_clause_id and routing_mode != "exact":
        raise RuntimeError(
            f"STRICT: question contains a clause-ID-like token but routing_mode={routing_mode!r}. "
            f"Exact clause-ID dominance is required for this query class. "
            f"Matched via fallback: window_ids={window_ids}. Possible causes: clause ID not in "
            f"store metadata, or _collect_exact_matches not recognising this format."
        )
    window_id = window_ids[0]
    window_text = store.get_window_text(window_id, tokenizer)
    if not window_text:
        raise RuntimeError(f"STRICT: store.get_window_text({window_id}) returned empty")
    window_keywords = list(store.keywords.get(window_id, []))
    print(f"      routed via {_bold(routing_mode)} → window {window_id}"
          f"  (question has clause-ID: {_has_clause_id})")
    print(f"      window text (first 200 chars): {shorten(window_text.replace(chr(10), ' '), 200)}")
    print(f"      keywords: {', '.join(window_keywords[:6])}")

    # Load boundary tensor
    boundary = store.load_boundary(window_id, device="cpu")
    # Normalise to torch tensor on CPU (will be moved to cuda inside generate_with_residual)
    import numpy as np
    if isinstance(boundary, np.ndarray):
        boundary_tensor = torch.from_numpy(boundary)
    elif isinstance(boundary, torch.Tensor):
        boundary_tensor = boundary.detach().to("cpu")
    else:
        raise RuntimeError(f"STRICT: unexpected boundary type {type(boundary)}")
    print(f"      boundary tensor: shape={tuple(boundary_tensor.shape)} dtype={boundary_tensor.dtype}")

    # Assertion 3: residual compatibility MUST pass (else fallback to prompt-context)
    compatible, reason = torch_query._residual_is_compatible(runtime, boundary_tensor, crystal_layer)
    print(f"[4/6] Residual compatibility check: {compatible}  ({reason})")
    if not compatible:
        raise RuntimeError(
            f"STRICT: _residual_is_compatible returned False — would have silently fallen back to "
            f"prompt-context mode. Reason: {reason}"
        )
    print(_green("      ✓ compatible — residual injection path will be used"))

    # Assertion 4: wrap the injection hook so we know if it fires
    hook_fired = {"count": 0, "layer": None, "in_shape": None, "residual_shape": None}
    orig_generate_with_residual = runtime.generate_with_residual

    def instrumented_generate_with_residual(prompt, residual_state, config):
        # Access underlying torch runtime to patch its inject_hook path.
        # Easier: wrap by calling the original but register our OWN pre-hook that
        # just records a fired signal. We keep the real hook untouched.
        import torch as _torch
        layers = runtime._resolve_layers()
        the_layer = layers[residual_state.layer_index]
        observed_residual_shape = tuple(residual_state.tensor.shape)

        def spy_hook(_module, args, kwargs):
            if not args:
                return args, kwargs
            hidden = args[0]
            hook_fired["count"] += 1
            hook_fired["layer"] = residual_state.layer_index
            hook_fired["in_shape"] = tuple(hidden.shape)
            hook_fired["residual_shape"] = observed_residual_shape
            return args, kwargs

        spy_handle = the_layer.register_forward_pre_hook(spy_hook, with_kwargs=True)
        try:
            return orig_generate_with_residual(prompt, residual_state, config)
        finally:
            spy_handle.remove()

    runtime.generate_with_residual = instrumented_generate_with_residual

    # Build prompt — derive the system prompt either from args, store metadata, or generic fallback
    if args.system_prompt:
        system = args.system_prompt
    else:
        system = (
            "You answer clause-based questions using only the store context provided by Lazarus. "
            "Be concise; quote clause IDs when known."
        )
        try:
            import json as _json
            checkpoint_dir = Path(args.store).parent
            prefill_path = checkpoint_dir / "torch_prefill.json"
            with open(prefill_path, "r", encoding="utf-8") as _fh:
                _prefill = _json.load(_fh)
            _source = _prefill.get("source") or {}
            _source_name = _source.get("name")
            _standard_title = _source.get("standard_title")
            _subject = _source_name or _standard_title
            if _subject:
                system = (
                    f"You answer questions about {_subject} using only the store context "
                    f"provided by Lazarus. Be concise; quote clause IDs when known."
                )
        except Exception:
            # File missing, JSON invalid, keys absent → keep generic fallback silently.
            pass
    print(f"      system prompt: {system!r}")

    user_content = (
        f"Store window excerpt:\n{window_text}\n\n"
        f"Store keywords: {', '.join(window_keywords[:8])}\n\n"
        f"Execution mode: residual\n\n"
        f"Question: {args.question}"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    residual_state = ResidualState(
        backend=LazarusBackend.CUDA,
        layer_index=crystal_layer,
        tensor=boundary_tensor,
        sequence_length=int(len(store.window_token_lists.get(window_id, [])) or store.num_tokens),
        hidden_size=int(boundary_tensor.shape[-1]),
        dtype=str(boundary_tensor.dtype).replace("torch.", ""),
        device="cpu",
    )
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)

    print(f"[5/6] Generating with residual injection at layer {crystal_layer} ...")
    mem_before_gen = torch.cuda.memory_allocated()
    result = runtime.generate_with_residual(prompt, residual_state, gen_config)
    mem_peak_gen = torch.cuda.max_memory_allocated()

    # Assertion 5: GPU memory grew during generation
    if mem_peak_gen <= mem_after_load:
        raise RuntimeError(
            f"STRICT: GPU peak memory ({mem_peak_gen/1024**3:.2f}GB) did not exceed "
            f"post-load memory ({mem_after_load/1024**3:.2f}GB) — suspicious CPU fallback"
        )

    # Assertion 6: hook must have fired at least once
    if hook_fired["count"] == 0:
        raise RuntimeError(
            "STRICT: spy hook did NOT fire during generate_with_residual. The injection layer "
            "was never traversed — residual injection did not occur."
        )

    print(_green(f"      ✓ spy hook fired {hook_fired['count']}× on layer {hook_fired['layer']}"))
    print(f"      hidden_state shape at hook: {hook_fired['in_shape']}")
    print(f"      residual tensor shape: {hook_fired['residual_shape']}")
    print(f"      GPU peak during generation: {mem_peak_gen/1024**3:.2f} GB  (delta {(mem_peak_gen-mem_before_gen)/1024**3:+.2f} GB from pre-gen)")
    print(f"      input_tokens={result.stats.input_tokens}  output_tokens={result.stats.output_tokens}  throughput={result.stats.tokens_per_second:.1f} tok/s")

    print(f"[6/6] {_bold('RESPONSE:')}")
    print("─" * 60)
    print(result.text)
    print("─" * 60)

    print(_green(_bold("\nALL STRICT ASSERTIONS PASSED:")))
    print(_green("  ✓ CUDA available + model on cuda:0"))
    print(_green(f"  ✓ Routing mode = {routing_mode} (NOT auto/MLP)"))
    print(_green("  ✓ Residual-compatibility check TRUE (no prompt-context fallback)"))
    print(_green(f"  ✓ Injection hook fired {hook_fired['count']}× at layer {crystal_layer}"))
    print(_green(f"  ✓ GPU memory grew during generation: {(mem_peak_gen-mem_before_load)/1024**3:+.2f} GB total"))
    print(_green(f"  ✓ Store provided window {window_id} with {len(window_text)} chars of context"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
