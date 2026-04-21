#!/usr/bin/env python3
"""Bisect the clause-aligned builder gibberish bug by testing the Apollo
flat-corpus build path on our 5-conversation cross-session corpus.

We take the synthetic ChatLoopSessions from
``chuk_lazarus.cross_session_demo.session_generator.DEFAULT_PLANS``,
concatenate them into a single raw-text corpus with clear session/turn
separators, and feed that corpus directly into the v12 Apollo store
builder (``build_knowledge_store_torch``) with NO JSON clause detour.

If this path runs clean where the clause-aligned builder produces
gibberish on the same content, the bug lives in the clause builder.
If it also gibberishes, the bug is in the synthetic prose itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES: tuple[str, ...] = (
    "manifest.json",
    "boundary_residual.npy",
    "entries.npz",
    "window_tokens.npz",
    "idf.json",
    "keywords.json",
)
REQUIRED_DIRS: tuple[str, ...] = ("boundaries",)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a v12 Apollo knowledge store from the 5-session "
            "synthetic cross-session corpus (raw-text path, no JSON)."
        )
    )
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--output",
        default="/tmp/apollo-memory/store",
        help="Output directory for the Apollo v12 store",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Torch device to use",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Optional cap on corpus tokens fed to the builder",
    )
    p.add_argument(
        "--max-new-tokens-reserved",
        type=int,
        default=None,
        dest="max_new_tokens_reserved",
        help="(Ignored; not relevant to build.)",
    )
    p.add_argument(
        "--session-separator",
        default="\n\n=== SESSION BOUNDARY ===\n\n",
        dest="session_separator",
        help="Separator string placed between sessions in the raw corpus",
    )
    p.add_argument(
        "--turn-separator",
        default="\n---\n",
        dest="turn_separator",
        help="Separator string placed between turns within a session",
    )
    p.add_argument(
        "--corpus-dump",
        default=None,
        dest="corpus_dump",
        help="Optional path to also write the concatenated corpus for debugging",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        return "cpu"
    return device


def _build_corpus(
    *,
    session_separator: str,
    turn_separator: str,
    verbose: bool,
) -> str:
    """Concatenate DEFAULT_PLANS sessions into a single raw-text corpus."""
    from chuk_lazarus.cross_session_demo.session_generator import (
        DEFAULT_PLANS,
        generate_session,
    )

    session_blocks: list[str] = []
    for plan in DEFAULT_PLANS:
        session = generate_session(plan)
        header = (
            f"[BEGIN SESSION {session.session_id} "
            f"topic={plan.topic} "
            f"planted_phrase={plan.planted_phrase}]"
        )
        turn_lines: list[str] = []
        for turn in session.turns:
            # turn.role is a Role enum (str, Enum); .value is e.g. "user"
            role_name = turn.role.value.upper()
            turn_lines.append(f"{role_name}: {turn.text}")

        body = turn_separator.join(turn_lines)
        block = f"{header}{turn_separator}{body}"
        session_blocks.append(block)

        total_chars = sum(len(t.text) for t in session.turns)
        approx_tokens = total_chars // 4
        if verbose:
            print(
                f"[corpus] session={session.session_id[:12]} "
                f"topic={plan.topic} "
                f"turns={len(session.turns)} "
                f"chars={total_chars} "
                f"~tokens={approx_tokens}"
            )

    raw_text = session_separator.join(session_blocks)
    print(f"[corpus] total_chars={len(raw_text)} ~tokens={len(raw_text) // 4}")
    return raw_text


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


def _verify_layout(output_path: Path) -> tuple[bool, list[str]]:
    """Return (ok, bullet-list lines) describing the store layout."""
    lines: list[str] = []
    ok = True

    for rel in REQUIRED_FILES:
        p = output_path / rel
        if not p.exists() or not p.is_file():
            ok = False
            lines.append(f"- MISSING file: {rel}")
            continue
        lines.append(f"- {rel} ({p.stat().st_size} bytes)")

    for rel in REQUIRED_DIRS:
        d = output_path / rel
        if not d.exists() or not d.is_dir():
            ok = False
            lines.append(f"- MISSING dir: {rel}")
            continue
        window_files = sorted(d.glob("window_*.npy"))
        if not window_files:
            ok = False
            lines.append(f"- {rel}/ EMPTY (no window_*.npy)")
            continue
        total = sum(f.stat().st_size for f in window_files)
        lines.append(
            f"- {rel}/ ({len(window_files)} window files, {total} bytes total)"
        )

    return ok, lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Build the raw corpus from the 5 synthetic sessions.
    raw_text = _build_corpus(
        session_separator=args.session_separator,
        turn_separator=args.turn_separator,
        verbose=args.verbose,
    )

    if args.corpus_dump:
        dump_path = Path(args.corpus_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(raw_text, encoding="utf-8")
        print(f"[corpus] dump written: {dump_path}")

    # 2. Load model + tokenizer (Apollo reference pattern).
    model, tokenizer, actual_device = _load_model_and_tokenizer(
        args.model, args.device
    )
    print(f"[model] loaded {args.model} on {actual_device}")

    # 3. Measure exact token count (no truncation).
    tokens_total = len(
        tokenizer(raw_text, add_special_tokens=False).input_ids
    )
    print(f"[tokens] tokens_total={tokens_total}")
    if tokens_total < 128_000:
        print(
            f"WARNING: tokens_total={tokens_total} is below the 128k axis-5 "
            f"gate; corpus may be undersized for a realistic build.",
            file=sys.stderr,
        )

    # 4. Architecture config from the real model config.
    from chuk_lazarus.inference.context.knowledge.config import ArchitectureConfig

    config = ArchitectureConfig.from_model_config(model.config)

    # 5. Build the v12 Apollo store directly from raw text.
    from chuk_lazarus.inference.context.knowledge.torch_build import (
        build_knowledge_store_torch,
    )

    store = build_knowledge_store_torch(
        model=model,
        tokenizer=tokenizer,
        raw_text=raw_text,
        config=config,
        output_path=output_path,
        max_tokens=args.max_tokens,
    )

    # 6. Print build summary.
    manifest_path = output_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("[build] summary:")
    print(f"  version            = {manifest.get('version')}")
    print(f"  num_entries        = {manifest.get('num_entries')}")
    print(f"  num_windows        = {manifest.get('num_windows')}")
    print(f"  num_tokens         = {manifest.get('num_tokens')}")
    print(f"  entries_per_window = {manifest.get('entries_per_window')}")
    print(f"  crystal_layer      = {manifest.get('crystal_layer')}")

    # 7. Verify the on-disk layout.
    ok, lines = _verify_layout(output_path)
    print("[layout] store contents:")
    for line in lines:
        print(f"  {line}")
    if not ok:
        print("[layout] FAIL: required files/dirs missing", file=sys.stderr)
        return 2

    # 8. Final summary line.
    summary = {
        "build_path": str(output_path),
        "tokens": tokens_total,
        "num_windows": manifest.get("num_windows"),
        "num_entries": manifest.get("num_entries"),
        "version": manifest.get("version"),
    }
    print(f"[summary] {json.dumps(summary)}")

    # Keep a reference to ``store`` to satisfy linters / guarantee load.
    _ = store
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
