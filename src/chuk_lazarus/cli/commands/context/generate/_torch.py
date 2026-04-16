"""Torch/CUDA checkpoint-sidecar path for ``context generate``."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from textwrap import shorten
from typing import Any

from .._types import GenerateConfig


class TorchCheckpointContractError(RuntimeError):
    """Raised when a torch checkpoint is missing its sidecar contract."""


_UNSUPPORTED_REPLAY_MODES = {"accumulated", "compressed", "explore", "inject", "kv", "sparse"}


def _default_top_k(value: int | None, *, strategy: str | None) -> int:
    if value is not None:
        return int(value)
    if strategy == "unified":
        return 10
    return 3


def _uses_chat_template(tokenizer: Any, no_chat_template: bool) -> bool:
    return (not no_chat_template) and callable(getattr(tokenizer, "apply_chat_template", None))


def _render_torch_context_prompt(
    tokenizer: Any,
    *,
    question: str,
    context_blocks: list[str],
    window_keywords: list[str],
    mode: str,
    system_prompt: str | None = None,
    no_chat_template: bool = False,
) -> str:
    from .....inference.context.knowledge.torch_query import _render_prompt

    return _render_prompt(
        tokenizer,
        question=question,
        context_blocks=context_blocks,
        window_keywords=window_keywords,
        mode=mode,
        system_prompt=system_prompt,
        use_chat_template=_uses_chat_template(tokenizer, no_chat_template),
        no_chat_template=no_chat_template,
    )


class _TorchReplayLibraryView:
    """Minimal CheckpointLibrary-compatible view over a torch knowledge store."""

    def __init__(self, store: Any, tokenizer: Any) -> None:
        self._store = store
        self._tokenizer = tokenizer

    @property
    def num_windows(self) -> int:
        return int(self._store.num_windows)

    def find_window_for_term(self, term: str, _tokenizer: Any) -> int | None:
        term_lower = term.lower()
        for window_id in range(self.num_windows):
            text = self._store.get_window_text(window_id, self._tokenizer)
            if term_lower in text.lower():
                return window_id
        return None


def _prepare_explicit_store_response(
    *,
    store: Any,
    runtime: Any,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
    temperature: float,
    replay_window_ids: list[int],
    routing_mode: str,
    system_prompt: str | None = None,
    no_chat_template: bool = False,
):
    from .....inference.backends import LazarusBackend, ResidualState
    from .....inference.context.knowledge.torch_query import (
        TorchKnowledgeResponse,
        _as_cpu_torch_tensor,
        _merge_keywords,
        _residual_is_compatible,
    )
    from .....inference.generation import GenerationConfig

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=1.0,
    )

    context_blocks = [store.get_window_text(window_id, tokenizer) for window_id in replay_window_ids]
    window_keywords = _merge_keywords(store, replay_window_ids)
    boundary_shape: tuple[int, ...] | None = None
    residual_reason = "boundary unavailable"
    mode = "prompt-context"
    prompt_mode = "context-replay"
    attempt_residual = len(replay_window_ids) == 1
    prompt = _render_torch_context_prompt(
        tokenizer,
        question=question,
        context_blocks=context_blocks,
        window_keywords=window_keywords,
        mode=prompt_mode,
        system_prompt=system_prompt,
        no_chat_template=no_chat_template,
    )

    try:
        if attempt_residual:
            boundary = store.load_boundary(replay_window_ids[0], device="cpu")
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
                prompt = _render_torch_context_prompt(
                    tokenizer,
                    question=question,
                    context_blocks=context_blocks,
                    window_keywords=window_keywords,
                    mode=prompt_mode,
                    system_prompt=system_prompt,
                    no_chat_template=no_chat_template,
                )
                residual_state = ResidualState(
                    backend=LazarusBackend.CUDA,
                    layer_index=crystal_layer,
                    tensor=boundary_tensor,
                    sequence_length=int(
                        len(store.window_token_lists.get(replay_window_ids[0], [])) or store.num_tokens
                    ),
                    hidden_size=int(boundary_tensor.shape[-1]),
                    dtype=str(boundary_tensor.dtype).replace("torch.", ""),
                    device="cpu",
                )
                result = runtime.generate_with_residual(prompt, residual_state, generation_config)
                mode = "residual"
            else:
                result = runtime.generate(prompt, generation_config)
        else:
            if replay_window_ids:
                residual_reason = "multiple replay windows selected"
            else:
                residual_reason = "no explicit replay windows selected"
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
        window_ids=replay_window_ids,
        expansion_terms=[],
        stats_summary=result.stats.summary,
        residual_reason=residual_reason,
        boundary_shape=boundary_shape,
        context_preview=preview,
    )


def _run_torch_store_generate(
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str | None,
    store_path: Path,
    replay_arg: list[str] | None,
    find_term: str | None,
    system_prompt: str | None,
    no_chat_template: bool,
) -> None:
    from .....inference.context.knowledge.torch_query import (
        _expand_query,
        _load_torch_runtime,
        _prepare_store_response,
    )
    from .....inference.context.knowledge.torch_store import TorchKnowledgeStore
    from ._resolve import _resolve_replay

    runtime, tokenizer = _load_torch_runtime(model_id, device)
    try:
        store = TorchKnowledgeStore.load(store_path)
        print(f"Loading knowledge store: {store_path}", file=sys.stderr)
        store.log_stats(file=sys.stderr)

        start_time = time.monotonic()
        replay_view = _TorchReplayLibraryView(store, tokenizer)
        replay_ids = _resolve_replay(replay_view, tokenizer, replay_arg, find_term)

        special_mode = (
            replay_ids[0]
            if isinstance(replay_ids, list) and len(replay_ids) == 1 and isinstance(replay_ids[0], str)
            else None
        )
        if special_mode in _UNSUPPORTED_REPLAY_MODES:
            print(
                f"  Warning: torch checkpoint generate does not support --replay {special_mode!r}; "
                "using auto store routing",
                file=sys.stderr,
            )
            replay_ids = None

        if replay_ids is None:
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
        else:
            response = _prepare_explicit_store_response(
                store=store,
                runtime=runtime,
                tokenizer=tokenizer,
                question=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                replay_window_ids=replay_ids,
                routing_mode="find" if find_term else "manual",
                system_prompt=system_prompt,
                no_chat_template=no_chat_template,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        if response.expansion_terms:
            print(
                f"  Query expansion: {', '.join(response.expansion_terms[:15])}",
                file=sys.stderr,
            )
        if response.window_ids:
            print(
                f"  Routed to windows {response.window_ids} via {response.routing_mode}",
                file=sys.stderr,
            )
            print(
                f"  Context mode: {response.mode} ({response.residual_reason})",
                file=sys.stderr,
            )
            if response.boundary_shape is not None:
                print(f"  Boundary shape: {response.boundary_shape}", file=sys.stderr)
            if response.context_preview:
                print(f"  Window preview: {response.context_preview}", file=sys.stderr)
        else:
            print("  No matching windows - plain torch inference.", file=sys.stderr)
        print(f"  {response.stats_summary} ({elapsed_ms:.0f} ms total)", file=sys.stderr)
        sys.stdout.write(response.text.rstrip() + "\n")
        sys.stdout.flush()
    finally:
        runtime.clear_cache()


def _validate_torch_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Path]:
    from .....inference.context.knowledge.torch_store import MANIFEST_FILE
    from ..prefill._torch_sidecar import TORCH_PREFILL_FILE, TORCH_STORE_DIR

    metadata_path = checkpoint_path / TORCH_PREFILL_FILE
    if not metadata_path.exists():
        raise TorchCheckpointContractError(
            f"torch checkpoint is missing {TORCH_PREFILL_FILE}; "
            "re-run `context prefill --backend torch` for this checkpoint."
        )

    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as exc:
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is invalid JSON: {metadata_path}"
        ) from exc

    status = str(metadata.get("status", "missing"))
    if status != "complete":
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is incomplete (status={status!r}): {metadata_path}"
        )

    backend = metadata.get("backend")
    if backend not in (None, "torch"):
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar declares backend={backend!r}, expected 'torch'"
        )

    generate_batch = metadata.get("generate_batch", {})
    if generate_batch and not bool(generate_batch.get("ready", False)):
        raise TorchCheckpointContractError(
            f"torch checkpoint sidecar is not marked ready for generate: {metadata_path}"
        )

    store_rel = metadata.get("artifacts", {}).get("store_dir", TORCH_STORE_DIR)
    store_path = checkpoint_path / str(store_rel)
    if not store_path.is_dir():
        raise TorchCheckpointContractError(
            f"torch checkpoint is missing store directory: {store_path}"
        )

    manifest_path = store_path / MANIFEST_FILE
    if not manifest_path.exists():
        raise TorchCheckpointContractError(
            f"torch checkpoint store is incomplete: missing {manifest_path}"
        )

    return metadata, store_path


def run_torch_checkpoint_generate(
    config: GenerateConfig,
    args,
    *,
    device: str | None,
) -> None:
    """Run checkpoint-backed generation through the torch knowledge-store path."""
    from .....inference.context.knowledge.torch_query import run_torch_query_command

    prompt_text = config.prompt_text
    if not prompt_text:
        print("Error: no prompt specified. Use --prompt or --prompt-file.", file=sys.stderr)
        return

    checkpoint = config.checkpoint
    if checkpoint is None:
        raise ValueError("torch checkpoint generate requires config.checkpoint")
    if not checkpoint.exists():
        print(f"Error: library not found: {checkpoint}", file=sys.stderr)
        return

    try:
        metadata, store_path = _validate_torch_checkpoint(checkpoint)
    except TorchCheckpointContractError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    model_info = metadata.get("model", {})
    windowing = metadata.get("windowing", {})
    sidecar_model_id = model_info.get("id")

    print(f"Loading torch checkpoint: {checkpoint}", file=sys.stderr)
    print(
        f"  torch_store={store_path.name}  |  {int(windowing.get('num_windows', 0))} windows  "
        f"|  {int(windowing.get('num_tokens', 0)):,} tokens",
        file=sys.stderr,
    )
    if sidecar_model_id and sidecar_model_id != config.model:
        print(
            f"Warning: torch sidecar model_id={sidecar_model_id!r} but loading model={config.model!r}",
            file=sys.stderr,
        )

    strategy = getattr(args, "strategy", None)
    top_k = _default_top_k(getattr(args, "top_k", None), strategy=strategy)
    no_chat_template = bool(getattr(args, "no_chat_template", False))
    replay_arg = getattr(args, "replay", None)
    find_term = getattr(args, "find", None)
    system_prompt = getattr(args, "system_prompt", None)

    if no_chat_template or replay_arg is not None or find_term is not None:
        _run_torch_store_generate(
            model_id=config.model,
            prompt=prompt_text,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_k=top_k,
            device=device,
            store_path=store_path,
            replay_arg=replay_arg,
            find_term=find_term,
            system_prompt=system_prompt,
            no_chat_template=no_chat_template,
        )
        return

    run_torch_query_command(
        model_id=config.model,
        prompt=prompt_text,
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        top_k=top_k,
        device=device,
        store_path=store_path,
        system_prompt=system_prompt,
        no_chat_template=False,
    )


__all__ = [
    "_TorchReplayLibraryView",
    "_default_top_k",
    "_render_torch_context_prompt",
    "TorchCheckpointContractError",
    "run_torch_checkpoint_generate",
]
