"""Multi-probe at 120 max_new_tokens, reusing existing /tmp/csd-multi checkpoints.

Skips the ~10-min build phase by using checkpoints already on disk from a
prior run. Runs 15 queries (5 sessions × 3 modes) with LAZARUS_MAX_NEW_TOKENS
defaulted to 120 (the original axis-5 baseline).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("LAZARUS_MAX_NEW_TOKENS", "120")

from chuk_lazarus.cross_session_demo.cli import DEFAULT_PLANS
from chuk_lazarus.cross_session_demo.session_generator import generate_session, plant_location
from chuk_lazarus.cross_session_demo.verification import (
    _TOPICAL_QUESTIONS,
    _ENTITY_QUESTIONS,
)
from chuk_lazarus.session_retrieval import SessionRetriever

INPUTS_ROOT = Path("/tmp/csd-multi/inputs")
CHECKPOINTS_ROOT = Path("/tmp/csd-multi/checkpoints")


def main() -> int:
    print(f"[probe-120] LAZARUS_MAX_NEW_TOKENS={os.environ['LAZARUS_MAX_NEW_TOKENS']}")
    if not CHECKPOINTS_ROOT.exists() or not any(CHECKPOINTS_ROOT.iterdir()):
        print(f"ERROR: {CHECKPOINTS_ROOT} empty — run multi_probe_test.py first")
        return 1

    # Reconstruct sessions deterministically (same plans → same UUIDs/handles).
    sessions = []
    for plan in DEFAULT_PLANS:
        gen = generate_session(plan)
        handle = plant_location(gen, plan)
        sessions.append({
            "plan": plan.model_dump(),
            "session_id": gen.session_id,
            "planted_handle": handle,
            "planted_phrase": plan.planted_phrase,
        })
        print(f"  session: {plan.topic}  id={gen.session_id[:8]}  handle={handle}")

    print(f"[probe-120] loading retriever across existing checkpoints…")
    retriever = SessionRetriever.from_checkpoint_root(
        CHECKPOINTS_ROOT,
        original_input_root=INPUTS_ROOT,
        device="cuda",
        model_id="google/gemma-4-E2B-it",
    )
    print(f"[probe-120] retriever loaded; crystal_layer={retriever.crystal_layer}")
    print(f"[probe-120] handles: {[h.session_id[:8] for h in retriever.handles]}")

    results = []
    for i, s in enumerate(sessions):
        topic = s["plan"]["topic"]
        print(f"\n=== session {i}: {topic} ({s['session_id'][:8]}) ===")
        print(f"  planted: {s['planted_phrase']!r}")

        # exact-ID
        print(f"  [exact-ID] {s['planted_handle']}")
        r = retriever.query_exact_id(s["planted_handle"])
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        sm = r.source_session == s["session_id"]
        print(f"    hit={hit}  src_match={sm}  win={r.window_id}")
        print(f"    answer[:240]={r.generated_answer[:240]!r}")
        results.append({"session": topic, "mode": "exact", "hit": hit,
                        "source_match": sm, "answer": r.generated_answer})

        # topical
        q = _TOPICAL_QUESTIONS[topic]
        print(f"  [topical]  {q}")
        r = retriever.query_topical(q)
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        sm = r.source_session == s["session_id"]
        print(f"    hit={hit}  src_match={sm}  win={r.window_id}  score={r.routing_score}")
        print(f"    answer[:240]={r.generated_answer[:240]!r}")
        results.append({"session": topic, "mode": "topical", "hit": hit,
                        "source_match": sm, "answer": r.generated_answer})

        # entity
        q = _ENTITY_QUESTIONS[topic]
        print(f"  [entity]   {q}")
        r = retriever.query_entity_mention(q)
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        sm = r.source_session == s["session_id"]
        print(f"    hit={hit}  src_match={sm}  win={r.window_id}  score={r.routing_score}")
        print(f"    answer[:240]={r.generated_answer[:240]!r}")
        results.append({"session": topic, "mode": "entity", "hit": hit,
                        "source_match": sm, "answer": r.generated_answer})

    print("\n" + "=" * 70)
    print(f"TALLY (max_new_tokens={os.environ['LAZARUS_MAX_NEW_TOKENS']})")
    print("=" * 70)
    print(f"total queries: {len(results)}")
    print(f"verbatim hits: {sum(1 for r in results if r['hit'])} / {len(results)}")
    print(f"source matches (routing): {sum(1 for r in results if r['source_match'])} / {len(results)}")
    print()
    print("by mode:")
    for mode in ("exact", "topical", "entity"):
        rows = [r for r in results if r["mode"] == mode]
        hits = sum(1 for r in rows if r["hit"])
        rts = sum(1 for r in rows if r["source_match"])
        print(f"  {mode:8s}: verbatim {hits}/{len(rows)}  routing {rts}/{len(rows)}")

    out = Path("/tmp/csd-multi/multi_probe_results_120.json")
    out.write_text(json.dumps({
        "max_new_tokens": int(os.environ["LAZARUS_MAX_NEW_TOKENS"]),
        "results": results,
    }, indent=2, default=str))
    print(f"\n[probe-120] saved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
