"""Shared helpers for knowledge CLI commands."""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager


def _resolve_backend_device(args: Namespace | None) -> tuple[str | None, str | None]:
    """Read the optional backend/device pair off an argparse namespace."""
    if args is None:
        return None, None
    return getattr(args, "backend", None), getattr(args, "device", None)


@contextmanager
def knowledge_backend_env(
    backend: str | None = None,
    device: str | None = None,
) -> Iterator[None]:
    """Temporarily surface CLI backend flags through the existing env contract."""
    prev_backend = os.environ.get("CHUK_BACKEND")
    prev_device = os.environ.get("CHUK_DEVICE")

    if backend is not None:
        os.environ["CHUK_BACKEND"] = backend
    if device is not None:
        os.environ["CHUK_DEVICE"] = device

    try:
        yield
    finally:
        if prev_backend is None:
            os.environ.pop("CHUK_BACKEND", None)
        else:
            os.environ["CHUK_BACKEND"] = prev_backend

        if prev_device is None:
            os.environ.pop("CHUK_DEVICE", None)
        else:
            os.environ["CHUK_DEVICE"] = prev_device


def _build_pipeline_config(*, backend: str | None = None, device: str | None = None):
    """Best-effort pipeline config wiring that tolerates partially landed refactors."""
    if backend is None and device is None:
        return None

    from ....inference import UnifiedPipelineConfig

    fields = getattr(UnifiedPipelineConfig, "model_fields", {})
    kwargs: dict[str, object] = {}

    if backend is not None:
        if "backend_name" in fields:
            kwargs["backend_name"] = backend
        if "backend" in fields:
            from ....inference.backends import LazarusBackend

            kwargs["backend"] = LazarusBackend.CUDA if backend == "torch" else LazarusBackend.MLX

    if device is not None and "device" in fields:
        kwargs["device"] = device

    if not kwargs:
        return None

    return UnifiedPipelineConfig(**kwargs)


def _mlx_core():
    from ....inference._lazy_mlx import mx

    return mx


def load_model(
    model_id: str,
    *,
    backend: str | None = None,
    device: str | None = None,
):
    """Load model, create KV generator, warm compute graph.

    Returns (pipeline, kv_gen, tokenizer).
    """
    from ....inference import UnifiedPipeline
    from ....inference.context.kv_generator import make_kv_generator

    print(f"Loading model: {model_id}", file=sys.stderr)
    pipeline_kwargs = {"verbose": False}
    pipeline_config = _build_pipeline_config(backend=backend, device=device)
    if pipeline_config is not None:
        pipeline_kwargs["pipeline_config"] = pipeline_config

    with knowledge_backend_env(backend=backend, device=device):
        pipeline = UnifiedPipeline.from_pretrained(model_id, **pipeline_kwargs)

    kv_gen = make_kv_generator(pipeline.model, pipeline.config)
    tokenizer = pipeline.tokenizer

    # Warm compute graph
    mx = _mlx_core()
    _ = kv_gen.prefill(mx.array([[1, 2, 3]]))

    return pipeline, kv_gen, tokenizer


def prepare_prompt(tokenizer, prompt_text: str) -> list[int]:
    """Wrap prompt in chat template if tokenizer supports it."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return tokenizer.encode(prompt_text, add_special_tokens=True)


def stop_token_ids(tokenizer) -> set[int]:
    """Return the set of stop token IDs for the tokenizer."""
    if tokenizer.eos_token_id is not None:
        return {tokenizer.eos_token_id}
    return set()


def generate_plain(
    kv_gen,
    prompt_ids: list[int],
    max_tokens: int,
    stop_ids: set[int],
    *,
    stream: bool = False,
    tokenizer=None,
) -> list[int]:
    """Plain generation without injection. Returns generated token IDs.

    When stream=True, writes each token to stdout as it's generated
    (requires tokenizer).
    """
    mx = _mlx_core()
    q_ids = mx.array(prompt_ids)[None]
    logits, kv_store = kv_gen.prefill(q_ids)
    mx.eval(logits)

    generated: list[int] = []
    seq_len = q_ids.shape[1]

    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        if token in stop_ids:
            break
        generated.append(token)
        if stream and tokenizer is not None:
            sys.stdout.write(tokenizer.decode([token], skip_special_tokens=True))
            sys.stdout.flush()
        logits, kv_store = kv_gen.step_uncompiled(mx.array([[token]]), kv_store, seq_len=seq_len)
        seq_len += 1

    return generated
