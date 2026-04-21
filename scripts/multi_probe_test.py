"""Multi-probe cross-session retrieval test with extended output budget.

Builds 5 themed session checkpoints, then runs 15 queries across them:
  - 5 exact-ID (one per session)
  - 5 topical (one per session)
  - 5 entity-mention (one per session)

Output token budget is controlled by LAZARUS_MAX_NEW_TOKENS env var
(default 120; set to 512 for this test).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("LAZARUS_MAX_NEW_TOKENS", "512")

from chuk_lazarus.cross_session_demo.cli import DEFAULT_PLANS, run_session
from chuk_lazarus.cross_session_demo.token_budget import measure_total_tokens
from chuk_lazarus.cross_session_demo.verification import (
    _TOPICAL_QUESTIONS,
    _ENTITY_QUESTIONS,
)
from chuk_lazarus.session_retrieval import SessionRetriever

INPUTS_ROOT = Path("/tmp/csd-multi/inputs")
CHECKPOINTS_ROOT = Path("/tmp/csd-multi/checkpoints")


def main() -> int:
    shutil.rmtree("/tmp/csd-multi", ignore_errors=True)
    INPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[multi-probe] LAZARUS_MAX_NEW_TOKENS={os.environ['LAZARUS_MAX_NEW_TOKENS']}")
    print(f"[multi-probe] building {len(DEFAULT_PLANS)} sessions...")

    sessions = []
    for plan in DEFAULT_PLANS:
        print(f"  - building session: {plan.topic}")
        result = run_session(plan, INPUTS_ROOT, CHECKPOINTS_ROOT, device="cuda")
        sessions.append(result)
        print(f"    session_id={result['session_id']}")
        print(f"    planted_handle={result['planted_handle']}")
        print(f"    planted_phrase={result['planted_phrase']!r}")

    total_tokens = measure_total_tokens([s["full_text"] for s in sessions])
    print(f"[multi-probe] total_tokens={total_tokens} (budget_met={total_tokens > 128000})")

    print(f"[multi-probe] loading retriever across {len(sessions)} checkpoints...")
    retriever = SessionRetriever.from_checkpoint_root(
        CHECKPOINTS_ROOT,
        original_input_root=INPUTS_ROOT,
        device="cuda",
        model_id="google/gemma-4-E2B-it",
    )
    print(f"[multi-probe] retriever loaded; crystal_layer={retriever.crystal_layer}")
    print(f"[multi-probe] handles: {[h.session_id[:8] for h in retriever.handles]}")

    results = []
    for i, s in enumerate(sessions):
        topic = s["plan"]["topic"]
        print(f"\n=== session {i}: {topic} ({s['session_id'][:8]}) ===")
        print(f"  planted phrase: {s['planted_phrase']!r}")

        # 1. exact-ID
        print(f"  [exact-ID] {s['planted_handle']}")
        r = retriever.query_exact_id(s["planted_handle"])
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        print(f"    hit={hit} | src={r.source_session[:8]} win={r.window_id}")
        print(f"    answer[:200]={r.generated_answer[:200]!r}")
        results.append({"session": topic, "mode": "exact", "hit": hit,
                        "source_match": r.source_session == s["session_id"],
                        "answer": r.generated_answer})

        # 2. topical
        q = _TOPICAL_QUESTIONS[topic]
        print(f"  [topical] {q}")
        r = retriever.query_topical(q)
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        print(f"    hit={hit} | src={r.source_session[:8]} win={r.window_id} score={r.routing_score}")
        print(f"    answer[:200]={r.generated_answer[:200]!r}")
        results.append({"session": topic, "mode": "topical", "hit": hit,
                        "source_match": r.source_session == s["session_id"],
                        "answer": r.generated_answer})

        # 3. entity-mention
        q = _ENTITY_QUESTIONS[topic]
        print(f"  [entity]   {q}")
        r = retriever.query_entity_mention(q)
        hit = s["planted_phrase"].lower() in (r.generated_answer or "").lower()
        print(f"    hit={hit} | src={r.source_session[:8]} win={r.window_id} score={r.routing_score}")
        print(f"    answer[:200]={r.generated_answer[:200]!r}")
        results.append({"session": topic, "mode": "entity", "hit": hit,
                        "source_match": r.source_session == s["session_id"],
                        "answer": r.generated_answer})

    # Tally
    print("\n" + "=" * 70)
    print("TALLY")
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

    out_path = Path("/tmp/csd-multi/multi_probe_results.json")
    out_path.write_text(json.dumps({
        "max_new_tokens": int(os.environ["LAZARUS_MAX_NEW_TOKENS"]),
        "total_tokens": total_tokens,
        "results": results,
    }, indent=2, default=str))
    print(f"\n[multi-probe] results saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
