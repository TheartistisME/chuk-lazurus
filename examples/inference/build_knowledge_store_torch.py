#!/usr/bin/env python3
"""Build a torch-native Apollo knowledge store from raw text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig
from chuk_lazarus.inference.context.knowledge.torch_build import build_knowledge_store_torch

DEFAULT_DEVICE = "auto"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a torch-native Apollo knowledge store from raw text"
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--input", required=True, help="Path to the raw text corpus")
    parser.add_argument("--output", required=True, help="Directory for the output store")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device to use: auto, cuda, or cpu",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Optional maximum number of tokens to read from the corpus",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        dest="window_size",
        help="Override the architecture window size",
    )
    parser.add_argument(
        "--entries-per-window",
        type=int,
        default=None,
        dest="entries_per_window",
        help="Override the number of entries extracted per window",
    )
    return parser


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        return "cpu"
    return device


def _load_model_and_tokenizer(model_id: str, device: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    actual_device = _resolve_device(device)
    dtype = torch.bfloat16 if actual_device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(actual_device)
    model.eval()
    return model, tokenizer, actual_device


def _resolve_architecture_config(
    model_config: Any,
    *,
    window_size: int | None,
    entries_per_window: int | None,
) -> ArchitectureConfig:
    config = ArchitectureConfig.from_model_config(model_config)
    if window_size is not None:
        object.__setattr__(config, "window_size", window_size)
    if entries_per_window is not None:
        object.__setattr__(config, "entries_per_window", entries_per_window)
    return config


def _load_raw_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path.read_text(encoding="utf-8")


def run_build(
    *,
    model_id: str,
    input_path: Path,
    output_path: Path,
    device: str,
    max_tokens: int | None,
    window_size: int | None,
    entries_per_window: int | None,
) -> int:
    text = _load_raw_text(input_path)
    model, tokenizer, actual_device = _load_model_and_tokenizer(model_id, device)
    config = _resolve_architecture_config(
        model.config,
        window_size=window_size,
        entries_per_window=entries_per_window,
    )

    store = build_knowledge_store_torch(
        model=model,
        tokenizer=tokenizer,
        raw_text=text,
        config=config,
        output_path=output_path,
        max_tokens=max_tokens,
    )

    print("Torch knowledge-store build")
    print(f"Model: {model_id}")
    print(f"Device: {actual_device}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Corpus chars: {len(text)}")
    print(f"Store version: 12")
    print(f"Windows: {store.num_windows}")
    print(f"Tokens: {store.num_tokens}")
    print(
        "Architecture: "
        f"retrieval=L{store.config.retrieval_layer}, "
        f"query_head=H{store.config.query_head}, "
        f"injection=L{store.config.injection_layer}, "
        f"crystal_layer=L{store.config.crystal_layer}, "
        f"window_size={store.config.window_size}, "
        f"entries_per_window={store.config.entries_per_window}"
    )
    print(f"Store ready for the read-only torch runner contract at: {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_build(
        model_id=args.model,
        input_path=Path(args.input),
        output_path=Path(args.output),
        device=args.device,
        max_tokens=args.max_tokens,
        window_size=args.window_size,
        entries_per_window=args.entries_per_window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
