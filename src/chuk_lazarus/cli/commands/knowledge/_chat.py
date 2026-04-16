"""knowledge chat — Interactive multi-turn conversation with a knowledge store."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


async def knowledge_chat_cmd(args: Namespace) -> None:
    """Interactive chat loop with persistent 1D injection from a knowledge store."""
    from ._backend import load_store_backend, resolve_backend_selection

    selection = resolve_backend_selection(args)
    if selection.effective == "torch":
        from ....inference.context.knowledge.torch_query import run_torch_chat_command

        run_torch_chat_command(
            model_id=args.model,
            store_path=args.store,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device=selection.device,
        )
        return

    from ._common import _mlx_core, generate_plain, load_model, prepare_prompt, stop_token_ids

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"Error: knowledge store not found: {store_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load model + store ────────────────────────────────────────────
    _, kv_gen, tokenizer = load_model(
        args.model,
        backend=selection.requested,
        device=selection.device,
    )
    print(f"Loading knowledge store: {store_path}", file=sys.stderr)
    store_cls = load_store_backend(selection.effective)
    store = store_cls.load(store_path)
    store.log_stats()
    print(file=sys.stderr)

    max_tokens = args.max_tokens
    temperature = args.temperature
    stop_ids = stop_token_ids(tokenizer)

    # ── Chat loop ─────────────────────────────────────────────────────
    while True:
        try:
            prompt_text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break

        if not prompt_text.strip():
            continue

        prompt_ids = prepare_prompt(tokenizer, prompt_text)
        window_id = store.route(prompt_text, tokenizer=tokenizer)

        if window_id is None:
            generated = generate_plain(
                kv_gen,
                prompt_ids,
                max_tokens,
                stop_ids,
                stream=True,
                tokenizer=tokenizer,
            )
        else:
            from ....inference.context.knowledge import generate_with_injection

            mx = _mlx_core()
            entries = store.get_entries_for_query(window_id, prompt_text, tokenizer)
            generated = generate_with_injection(
                kv_gen=kv_gen,
                prompt_ids=mx.array(prompt_ids)[None],
                entries=entries,
                config=store.config,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_ids=stop_ids,
            )
            sys.stdout.write(tokenizer.decode(generated, skip_special_tokens=True))

        sys.stdout.write("\n")
        sys.stdout.flush()
