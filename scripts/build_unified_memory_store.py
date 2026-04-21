"""Build ONE unified clause-aligned store from all 5 conversations.

Mirrors the AUS3000 build pattern exactly: per-clause JSON files in a
single input directory, fed to tools/build_clause_aligned_store.py
(unmodified). Produces ONE torch_store containing every turn from
every conversation.

Per-clause JSON schema (AUS3000-format):
{
  "standard_id":    "Conversational Memory",            # CLEAN, not SHA256
  "standard_title": "Five themed conversations: ...",
  "clause_id":      "<session_idx>.<turn_idx>",         # dotted-numeric
  "clause_title":   "Turn <N> of <topic>",
  "clause_content": "<turn text with planted phrase planted ONCE>"
}

Session→clause_id mapping (deterministic, human-readable):
  1.* = dubai-trip
  2.* = alice-project
  3.* = sydney-conference
  4.* = rust-migration
  5.* = pottery-class

After build, query with:
  bash scripts/run_clause_aligned_demo.sh \
    --store /tmp/unified-memory/checkpoint/torch_store \
    --question "What does clause 1.11 define?"
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from chuk_lazarus.cross_session_demo.session_generator import (
    DEFAULT_PLANS,
    PLANTED_ASSISTANT_TURN_INDEX,
    generate_session,
)

INPUT_DIR = Path("/tmp/unified-memory/inputs")
CHECKPOINT_DIR = Path("/tmp/unified-memory/checkpoint")
STANDARD_ID = "Conversational Memory"
STANDARD_TITLE = (
    "Five themed conversations: dubai-trip, alice-project, "
    "sydney-conference, rust-migration, pottery-class"
)


def main() -> int:
    # Wipe + recreate
    shutil.rmtree("/tmp/unified-memory", ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[builder] generating {len(DEFAULT_PLANS)} conversations...")
    total_turns = 0
    planted_handles: list[tuple[str, str]] = []  # (handle, phrase)

    for session_idx, plan in enumerate(DEFAULT_PLANS, start=1):
        print(f"  session {session_idx}: {plan.topic}")
        session = generate_session(plan)
        for turn in session.turns:
            turn_idx = turn.turn_index
            clause_id = f"{session_idx}.{turn_idx}"
            speaker = "user" if turn.role.value == "user" else "assistant"
            clause_title = (
                f"Turn {turn_idx} of {plan.topic} ({speaker})"
            )
            clause_content = turn.text
            record = {
                "standard_id": STANDARD_ID,
                "standard_title": STANDARD_TITLE,
                "clause_id": clause_id,
                "clause_title": clause_title,
                "clause_content": clause_content,
            }
            # Filename: zero-padded for natural sort
            fname = f"s{session_idx:02d}_t{turn_idx:03d}.json"
            (INPUT_DIR / fname).write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            total_turns += 1
            if turn_idx == PLANTED_ASSISTANT_TURN_INDEX and speaker == "assistant":
                planted_handles.append((clause_id, plan.planted_phrase))

    print(f"[builder] wrote {total_turns} per-turn JSON files to {INPUT_DIR}")
    print(f"[builder] planted phrases (handle -> phrase):")
    for handle, phrase in planted_handles:
        print(f"  {handle}  ->  {phrase!r}")

    # Run the canonical (UNMODIFIED) builder against the input dir.
    print(f"\n[builder] invoking tools/build_clause_aligned_store.py...")
    cmd = [
        sys.executable,
        "tools/build_clause_aligned_store.py",
        "--input-dir", str(INPUT_DIR),
        "--checkpoint", str(CHECKPOINT_DIR),
        "--device", "cuda",
        "--force",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"[builder] FAIL: builder returned {result.returncode}")
        return result.returncode

    print(f"\n[builder] done.")
    print(f"  store     : {CHECKPOINT_DIR}/torch_store")
    print(f"  prefill   : {CHECKPOINT_DIR}/torch_prefill.json")
    print(f"  manifest  : {CHECKPOINT_DIR}/clause_aligned_build_manifest.json")
    print(f"\n  query example:")
    print(f"    bash scripts/run_clause_aligned_demo.sh \\")
    print(f"      --store {CHECKPOINT_DIR}/torch_store \\")
    print(f"      --question 'What did clause {planted_handles[0][0]} say?'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
