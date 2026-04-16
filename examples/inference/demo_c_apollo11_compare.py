#!/usr/bin/env python3
"""
Interactive plain-vs-Apollo comparison for Apollo 11 questions.

This script loads one local torch model, answers the question plainly first,
waits for Enter, and then re-runs the same model through the Apollo store
with residual injection when the boundary geometry matches.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_STORE = "/tmp/apollo-demo/apollo11_store"
DEFAULT_DEVICE = "cuda"
HEADER_WIDTH = 88


class Ansi:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.reset = "\033[0m" if enabled else ""
        self.bold = "\033[1m" if enabled else ""
        self.dim = "\033[2m" if enabled else ""
        self.cyan = "\033[96m" if enabled else ""
        self.green = "\033[92m" if enabled else ""
        self.magenta = "\033[95m" if enabled else ""
        self.yellow = "\033[93m" if enabled else ""

    def paint(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        prefix = "".join(styles)
        return f"{prefix}{text}{self.reset}"


STYLE = Ansi(sys.stdout.isatty())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive plain-vs-Apollo comparison for Apollo 11 questions"
    )
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face model ID")
    parser.add_argument(
        "--store",
        default=DEFAULT_STORE,
        help="Apollo store directory",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device to use")
    parser.add_argument("--question", required=True, help="Apollo 11 question to ask")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        dest="max_new_tokens",
        help="Maximum number of generated tokens",
    )
    return parser


def _plain_prompt(tokenizer: Any, question: str) -> str:
    system = "You answer Apollo 11 questions clearly and concisely."
    user_content = f"Question: {question}"

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


def _terminal_width(default: int = HEADER_WIDTH) -> int:
    columns = shutil.get_terminal_size((default, 24)).columns
    return max(72, min(100, columns - 2))


def _wrap_block(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.rstrip().splitlines() or [""]:
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(
            wrap(
                raw_line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    return lines


def _print_rule(width: int, accent: str) -> None:
    print(STYLE.paint("+" + "-" * width + "+", accent))


def _print_panel(title: str, body_lines: list[str], accent: str, width: int) -> None:
    _print_rule(width, accent)
    title_text = title[: width - 2].ljust(width - 2)
    print(STYLE.paint(f"|{STYLE.paint(title_text, STYLE.bold, accent)}|", accent))
    _print_rule(width, accent)
    for line in body_lines:
        print(STYLE.paint(f"| {line[: width - 2].ljust(width - 2)} |", accent))
    _print_rule(width, accent)


def _print_kv_panel(title: str, items: list[tuple[str, str]], accent: str, width: int) -> None:
    lines = [f"{key}: {value}" for key, value in items]
    _print_panel(title, lines, accent, width)


def _apollo_answer(
    *,
    store_path: str | Path,
    runtime,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
) -> tuple[str, dict[str, str], str]:
    from chuk_lazarus.inference.backends import LazarusBackend, ResidualState
    from chuk_lazarus.inference.generation import GenerationConfig
    from examples.inference.demo_c_apollo11_torch import (
        _as_cpu_torch_tensor,
        _render_prompt,
        _residual_is_compatible,
        load_apollo_store,
    )

    store = load_apollo_store(store_path)

    window_id = store.route(question, method="keyword")
    routing_mode = "keyword"
    if window_id is None:
        window_ids = store.route_top_k(question, tokenizer, k=1)
        if not window_ids:
            raise RuntimeError("Apollo store could not route the question to any window")
        window_id = window_ids[0]
        routing_mode = "tfidf"

    window_text = store.get_window_text(window_id, tokenizer)
    boundary = store.load_boundary(window_id, device="cpu")
    boundary_tensor = _as_cpu_torch_tensor(boundary)

    crystal_layer = int(store.config.crystal_layer)
    residual_ok, residual_reason = _residual_is_compatible(runtime, boundary_tensor, crystal_layer)
    mode = "residual" if residual_ok else "prompt-context"

    keywords_map = getattr(store, "keywords", None) or {}
    window_keywords = list(keywords_map.get(window_id, []))
    prompt = _render_prompt(tokenizer, question, window_text, window_keywords, mode)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    if residual_ok:
        residual_state = ResidualState(
            backend=LazarusBackend.CUDA,
            layer_index=crystal_layer,
            tensor=boundary_tensor,
            sequence_length=int(len(store.window_token_lists.get(window_id, [])) or store.num_tokens),
            hidden_size=int(boundary_tensor.shape[-1]),
            dtype=str(boundary_tensor.dtype).replace("torch.", ""),
            device="cpu",
        )
        result = runtime.generate_with_residual(prompt, residual_state, generation_config)
    else:
        result = runtime.generate(prompt, generation_config)

    metadata = {
        "routing": routing_mode,
        "mode": mode,
        "reason": residual_reason,
        "window": str(window_id),
        "layer": str(crystal_layer),
        "boundary": str(tuple(boundary_tensor.shape)),
        "dtype": str(boundary_tensor.dtype),
    }
    return result.text, metadata, result.stats.summary


def run_compare(
    *,
    store_path: str | Path,
    model_id: str,
    device: str,
    question: str,
    max_new_tokens: int,
) -> int:
    width = _terminal_width()
    from chuk_lazarus.inference.generation import GenerationConfig
    from examples.inference.demo_c_apollo11_torch import _load_torch_runtime

    runtime, tokenizer = _load_torch_runtime(model_id, device)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    header_lines = [
        f"Model   : {shorten(model_id, width=width - 12, placeholder='...')}",
        f"Store   : {shorten(str(store_path), width=width - 12, placeholder='...')}",
        f"Device  : {device}",
        f"Question: {question}",
    ]
    _print_panel("APOLLO COMPARISON", header_lines, STYLE.cyan, width)

    plain_prompt = _plain_prompt(tokenizer, question)
    plain_started = time.perf_counter()
    plain_result = runtime.generate(plain_prompt, config)
    plain_elapsed = time.perf_counter() - plain_started

    _print_kv_panel(
        "PLAIN",
        [
            ("Mode", "plain LLM"),
            ("Elapsed", f"{plain_elapsed:.2f}s"),
            ("Stats", plain_result.stats.summary),
        ],
        STYLE.green,
        width,
    )
    _print_panel(
        "PLAIN ANSWER",
        _wrap_block(plain_result.text.strip() or "(no text returned)", width - 4),
        STYLE.green,
        width,
    )

    try:
        input(STYLE.paint("\nPress Enter to continue to the Apollo-assisted answer... ", STYLE.bold, STYLE.yellow))
    except EOFError:
        print(STYLE.paint("\nNo interactive input available; continuing automatically.", STYLE.dim, STYLE.yellow))

    apollo_started = time.perf_counter()
    apollo_text, metadata, apollo_summary = _apollo_answer(
        store_path=store_path,
        runtime=runtime,
        tokenizer=tokenizer,
        question=question,
        max_new_tokens=max_new_tokens,
    )
    apollo_elapsed = time.perf_counter() - apollo_started

    _print_kv_panel(
        "APOLLO",
        [
            ("Routing", metadata["routing"]),
            ("Mode", metadata["mode"]),
            ("Reason", metadata["reason"]),
            ("Window", metadata["window"]),
            ("Layer", metadata["layer"]),
            ("Boundary", f'{metadata["boundary"]} / {metadata["dtype"]}'),
            ("Elapsed", f"{apollo_elapsed:.2f}s"),
            ("Stats", apollo_summary),
        ],
        STYLE.magenta,
        width,
    )
    _print_panel(
        "APOLLO ANSWER",
        _wrap_block(apollo_text.strip() or "(no text returned)", width - 4),
        STYLE.magenta,
        width,
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_compare(
        store_path=args.store,
        model_id=args.model,
        device=args.device,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    raise SystemExit(main())
