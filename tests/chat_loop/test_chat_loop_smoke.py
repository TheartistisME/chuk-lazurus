"""CUDA-gated end-to-end smoke test against real Gemma."""

from __future__ import annotations

import pytest


try:
    import torch

    _HAS_CUDA = bool(torch.cuda.is_available())
except Exception:
    _HAS_CUDA = False


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available"),
]


def test_one_turn_real_gemma_streaming_and_windowing() -> None:
    # Skip cleanly if heavy deps aren't installed in this env
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from chuk_lazarus.chat_loop import ChatLoopSession, StreamingWindower
    from chuk_lazarus.chat_loop.cli import load_gemma, stream_assistant_reply
    from chuk_lazarus.inference.chat import ChatHistory, Role

    tokenizer, model = load_gemma()

    history = ChatHistory()
    history.add_system("You are concise.")
    history.add_user("Describe a forest in exactly two sentences.")

    session = ChatLoopSession()
    windower = StreamingWindower(tokenizer)
    assistant_turn = session.begin_turn(Role.ASSISTANT, "")

    reply = stream_assistant_reply(
        model=model,
        tokenizer=tokenizer,
        history=history,
        windower=windower,
        session=session,
        turn=assistant_turn,
        max_new_tokens=64,
    )

    assert isinstance(reply, str)
    assert reply != ""

    # stream_assistant_reply is expected to call session.finish_turn(turn)
    assert assistant_turn.finished_at is not None
    assert windower.total_tokens > 0
    # Two sentences under 64 new tokens will almost never hit the 512-token
    # window threshold, but we allow 0 or 1 full emission.
    assert windower.emitted_chunks in (0, 1)
