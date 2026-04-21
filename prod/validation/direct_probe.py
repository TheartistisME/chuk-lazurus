"""Bypass multi_probe_query_only.py's UUID regeneration; hit the real on-disk
handles directly to force generate_with_residual_prefill_seeded to execute."""
import os, sys, traceback
from pathlib import Path

os.environ.setdefault("LAZARUS_MAX_NEW_TOKENS", "120")

from chuk_lazarus.session_retrieval import SessionRetriever

CHECKPOINTS = Path("/tmp/csd-multi/checkpoints")

# Real handles from on-disk manifest (enumerate session dirs, use '.1.0' of each).
real_session_uuids = sorted(p.name for p in CHECKPOINTS.iterdir() if p.is_dir())
print(f"[direct] found {len(real_session_uuids)} on-disk sessions: {real_session_uuids}")

retriever = SessionRetriever.from_checkpoint_root(
    CHECKPOINTS,
    device="cuda",
    model_id="google/gemma-4-E2B-it",
)
print(f"[direct] retriever loaded; crystal_layer={retriever.crystal_layer}")

results = []
for i, sid in enumerate(real_session_uuids[:3]):   # 3 probes for speed
    handle = f"{sid}.1.0"
    print(f"\n=== probe {i}: {handle} ===")
    try:
        r = retriever.query_exact_id(handle)
        print(f"  source_session={r.source_session[:8]}  window_id={r.window_id}")
        print(f"  strict: {r.strict_assertions}")
        print(f"  answer[:400]={r.generated_answer[:400]!r}")
        results.append({"handle": handle, "ok": True, "answer_head": r.generated_answer[:400]})
    except Exception as exc:
        print(f"  FAILED with {type(exc).__name__}: {exc}")
        traceback.print_exc()
        results.append({"handle": handle, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

import json
with open("/tmp/validation-logs/direct_probe_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n[direct] done; wrote /tmp/validation-logs/direct_probe_results.json")
