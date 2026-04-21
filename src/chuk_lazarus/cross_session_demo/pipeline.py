"""Per-session 3-step chain (axis-1 -> axis-2 -> axis-3).

One call to :func:`run_session` runs a single plan through:

1. ``session_generator.generate_session`` (axis-1 ChatLoopSession).
2. ``session_close.emit_from_handoff`` (axis-2 AUS3000 records to disk).
3. ``session_store.invoke_build`` (axis-3 torch_store build via subprocess).

Non-zero returncode from axis-3 raises ``RuntimeError`` with the
stderr tail so the caller can fail fast.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chuk_lazarus.chat_loop import TranscriptHandoff
from chuk_lazarus.session_close import emit_from_handoff
from chuk_lazarus.session_store import invoke_build

from chuk_lazarus.cross_session_demo.session_generator import (
    SessionPlan,
    generate_session,
    plant_location,
)


def _concatenate_session_text(session) -> str:
    return "\n".join(turn.text for turn in session.turns)


def run_session(
    plan: SessionPlan,
    inputs_root: Path,
    checkpoint_root: Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one plan through axes 1-3 and return a result dict.

    Returned keys:
      - ``session_id``: 32-char hex id assigned by axis-1
      - ``plan``: the input plan (model_dump)
      - ``planted_handle``: dotted handle of the planted turn
      - ``planted_phrase``: verbatim phrase
      - ``emitted_paths``: list[Path] from axis-2
      - ``checkpoint``: Path to the per-session checkpoint
      - ``full_text``: concatenation of every turn's text
      - ``returncode``: axis-3 subprocess returncode
      - ``stderr``: axis-3 captured stderr
    """
    inputs_root = Path(inputs_root)
    checkpoint_root = Path(checkpoint_root)
    inputs_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    session = generate_session(plan)
    handoff = TranscriptHandoff(session)
    emitted_paths = emit_from_handoff(handoff, inputs_root)
    planted_handle = plant_location(session, plan)
    full_text = _concatenate_session_text(session)

    result = invoke_build(
        session_id=session.session_id,
        input_dir=inputs_root / session.session_id,
        checkpoint_root=checkpoint_root,
        device=device,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").splitlines()[-40:]
        raise RuntimeError(
            "axis-3 build failed for session "
            f"{session.session_id}: rc={result.returncode}\n"
            + "\n".join(tail)
        )

    return {
        "session_id": session.session_id,
        "plan": plan.model_dump(),
        "planted_handle": planted_handle,
        "planted_phrase": plan.planted_phrase,
        "emitted_paths": [str(p) for p in emitted_paths],
        "checkpoint": str(result.checkpoint),
        "full_text": full_text,
        "returncode": result.returncode,
        "stderr": result.stderr,
    }


__all__ = ["run_session"]
