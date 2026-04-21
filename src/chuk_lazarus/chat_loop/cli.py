"""Local REPL CLI for the turn-aligned chat loop.

Runs a Gemma-4-E2B-it chat REPL that streams assistant output to stdout
and chunks the stream into 512-token windows in real time (during the
stream, not post-hoc). Each turn is recorded in a ChatLoopSession so
axis-2 can later pull raw records via TranscriptHandoff.

Heavy dependencies (torch, transformers) are imported inside function
bodies so `import chuk_lazarus.chat_loop.cli` stays cheap.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Callable

from chuk_lazarus.chat_loop.handoff import TranscriptHandoff
from chuk_lazarus.chat_loop.session import ChatLoopSession, ChunkBoundary, TurnRecord
from chuk_lazarus.chat_loop.streaming import StreamingWindower
from chuk_lazarus.inference.chat import ChatHistory, Role, format_history

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


GEMMA_LOCAL_SNAPSHOT = (
    "/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/"
    "snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
)
GEMMA_HUB_ID = "google/gemma-4-E2B-it"


def load_gemma(
    model_path: str | None = None,
    *,
    device: str | None = None,
) -> tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    """Load the Gemma tokenizer + model, preferring the local snapshot.

    If the local snapshot path does not exist on disk, falls back to the
    canonical Hugging Face hub id.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = model_path or GEMMA_LOCAL_SNAPSHOT
    if not os.path.isdir(resolved):
        resolved = GEMMA_HUB_ID

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForCausalLM.from_pretrained(resolved)
    model.to(device)
    model.eval()
    return tokenizer, model


def stream_assistant_reply(
    *,
    model: Any,
    tokenizer: Any,
    history: ChatHistory,
    windower: StreamingWindower,
    session: ChatLoopSession,
    turn: TurnRecord,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    on_chunk: Callable[[ChunkBoundary, str], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    """Stream an assistant reply, windowing on the hot path.

    Returns the full accumulated reply string. Each streamed text delta
    is fed into the windower; any emitted boundaries are appended to the
    given turn. On stream end, a final tail boundary (if any) is flushed
    and the turn is marked finished.
    """
    from transformers import TextIteratorStreamer

    prompt = format_history(tokenizer, history, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "streamer": streamer,
    }

    errors: list[BaseException] = []

    def _run_generate() -> None:
        try:
            model.generate(**generation_kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()

    accumulated: list[str] = []
    for text_delta in streamer:
        if not text_delta:
            continue
        accumulated.append(text_delta)
        if on_text_delta is not None:
            on_text_delta(text_delta)
        boundaries = windower.feed_text(text_delta)
        for boundary in boundaries:
            session.append_chunk(turn, boundary)
            if on_chunk is not None:
                decoded = tokenizer.decode(
                    windower._token_ids[  # noqa: SLF001 - intentional read
                        boundary.start_token_offset : boundary.end_token_offset
                    ],
                    skip_special_tokens=True,
                )
                on_chunk(boundary, decoded)

    tail = windower.flush()
    if tail is not None:
        session.append_chunk(turn, tail)
        if on_chunk is not None:
            decoded = tokenizer.decode(
                windower._token_ids[  # noqa: SLF001
                    tail.start_token_offset : tail.end_token_offset
                ],
                skip_special_tokens=True,
            )
            on_chunk(tail, decoded)

    session.finish_turn(turn)

    thread.join()
    if errors:
        raise errors[0]

    return "".join(accumulated)


def run_repl(
    *,
    system_prompt: str | None = None,
    model_path: str | None = None,
    max_new_tokens: int = 512,
) -> ChatLoopSession:
    """Run the interactive REPL until EOF / /quit, return the session."""
    tokenizer, model = load_gemma(model_path)

    history = ChatHistory()
    if system_prompt:
        history.add_system(system_prompt)

    session = ChatLoopSession()

    while True:
        try:
            user_text = input("you> ")
        except EOFError:
            print()
            break

        stripped = user_text.strip()
        if not stripped:
            break
        if stripped == "/quit":
            break

        history.add_user(user_text)
        user_turn = session.begin_turn(Role.USER, user_text)
        session.finish_turn(user_turn)

        windower = StreamingWindower(tokenizer)
        assistant_turn = session.begin_turn(Role.ASSISTANT, "")

        sys.stdout.write("gemma> ")
        sys.stdout.flush()

        def _echo(delta: str) -> None:
            sys.stdout.write(delta)
            sys.stdout.flush()

        full_reply = stream_assistant_reply(
            model=model,
            tokenizer=tokenizer,
            history=history,
            windower=windower,
            session=session,
            turn=assistant_turn,
            max_new_tokens=max_new_tokens,
            on_text_delta=_echo,
        )

        assistant_turn.text = full_reply
        sys.stdout.write("\n")
        sys.stdout.flush()

        history.add_assistant(full_reply)

    total_chunks = sum(len(t.chunk_boundaries) for t in session.turns)
    print(f"Transcript length: {len(session.turns)} turns, {total_chunks} chunks")
    return session


def main() -> None:
    """Entrypoint for `python -m chuk_lazarus.chat_loop.cli`."""
    parser = argparse.ArgumentParser(
        description="Turn-aligned Gemma chat REPL with streaming 512-token windowing.",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model path (default: local Gemma snapshot, fallback to hub id).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens per assistant turn (default: 512).",
    )
    args = parser.parse_args()

    session = run_repl(
        system_prompt=args.system,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
    )

    handoff = TranscriptHandoff(session)
    turns = handoff.snapshot()
    total_chunks = sum(len(t.chunk_boundaries) for t in turns)
    print(
        f"Handoff ready: session_id={session.session_id} "
        f"turns={len(turns)} chunks={total_chunks}"
    )


if __name__ == "__main__":
    main()
