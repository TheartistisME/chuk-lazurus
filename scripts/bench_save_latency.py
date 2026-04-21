#!/usr/bin/env python3
"""Benchmark: end-to-end wall-clock of REPL /save (steady-state).

AC-1 primary for the release-gate-ac4 lead.

Semantics
---------
This benchmark measures STEADY-STATE ``MemoryChat.save_current_session``
wall-clock — i.e. the regime the user experiences once the retriever
has already been lazy-loaded. Because the FIRST ``/save`` against an
empty store triggers a one-time ``chat.maybe_load_retriever()`` cold
load of Gemma (~40s on a typical CUDA host) that is orthogonal to the
3-loads-to-1 architecture change AC-1 actually validates, the bench
runs ONE warmup ``/save`` up-front and records it separately as
``warmup_save_seconds``. The ``--runs`` timed loop then executes inside
the SAME ``MemoryChat`` instance against the SAME store_root, so each
timed save exercises the hot ``refresh_handles`` path. AC-1 is evaluated
against the timed saves only (p50/p95/pass come from ``per_run_seconds``,
NOT the warmup). Use ``--no-warmup`` to include the cold retriever load
in the first timed run (diagnostic / cold-store mode). The acceptance
target is **< 2 s on a 6-turn session**.

How it runs
-----------
* We construct a real ``MemoryChat`` against a fresh store root. The model
  is loaded exactly once per bench process so the first run amortizes the
  cold model-load across subsequent runs (same "hot" condition the REPL
  enjoys after the first /save).
* ``chat.maybe_load_retriever`` is deliberately skipped — we start with an
  empty store so the save path exercises the "first save lazy-load" branch
  when ``rebuild_retriever=True``. This is the path the user actually hits
  the first time they press /save, so it is the correct benchmark.
* For every run we pre-seed a fresh session with ``session_turns`` calls to
  ``plain_chat_turn`` (short canned prompts, rotated from a small fixed
  list so the benchmark is deterministic), then take ``t0 =
  perf_counter()`` (with ``torch.cuda.synchronize`` when on cuda), call
  ``save_current_session``, then take ``t1`` (again with synchronize), and
  record ``t1 - t0`` into ``per_run_seconds``.
* p50 / p95 are computed over ``per_run_seconds``. Pass = all runs under
  ``target_seconds`` AND every save returned True.

Output
------
A single JSON object on stdout. INFO/trace lines go to stderr so stdout
stays parseable by consumers such as the release-gate-ac4 lead.

Exit codes
----------
* ``0`` — pass OR the cuda-absent skip path was taken (unless --strict).
* ``1`` — at least one run exceeded 2.0s or ``save_current_session``
          returned False.
* ``2`` — environmental error (model load failed, import error, etc.).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Ensure the repo root is importable BEFORE we try to import the chat module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Canned user lines for deterministic seeding. We rotate through the list so
# identical turns don't collapse into duplicate windows.
CANNED_PROMPTS = [
    "Hi! Tell me a two-sentence story about a robot.",
    "Now tell me a two-sentence story about a dragon.",
    "What about a tiny story concerning an astronaut?",
    "One more, please — two sentences about a librarian.",
    "And another two-sentence tale about a carpenter.",
    "Final one: two sentences featuring a lighthouse keeper.",
]


def _log(msg: str) -> None:
    """INFO line to stderr — keeps stdout clean for JSON output."""
    print(f"[bench_save_latency] {msg}", file=sys.stderr, flush=True)


def _iso_utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _percentile(values: list[float], pct: float) -> float:
    """Simple linear-interpolation percentile (no numpy dependency)."""
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    xs = sorted(values)
    k = (len(xs) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def _emit(obj: dict[str, Any]) -> None:
    """Pretty-print the final JSON record to stdout."""
    print(json.dumps(obj, indent=2, sort_keys=True), flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_save_latency.py",
        description=(
            "Measure REPL /save wall-clock end-to-end (target <2s on a "
            "6-turn session, AC-1 for release-gate-ac4)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of independent /save timings to collect (default 5).",
    )
    parser.add_argument(
        "--session-turns",
        type=int,
        default=6,
        help="Number of chat turns to pre-seed per run before /save (default 6).",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path(f"/tmp/bench-save-latency-{os.getpid()}"),
        help="Base store root (each run gets a unique suffix under it).",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("LAZARUS_MODEL"),
        help="Override model path (default: env LAZARUS_MODEL or local Gemma snapshot).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default cuda).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens per assistant turn during seeding (default 64; keeps bench fast).",
    )
    parser.add_argument(
        "--prefill-text",
        default="Hi! Tell me a two-sentence story.",
        help="Cheap user turn used when CANNED_PROMPTS is exhausted.",
    )
    parser.add_argument(
        "--skip-if-no-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set (default), exit 0 with pass=false when CUDA is unavailable "
            "so structural runs don't block on GPU-less hosts. Use --strict to override."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Override --skip-if-no-cuda: treat CUDA absence as a fatal "
            "environmental error (exit 2)."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help=(
            "Skip the single warmup save that initializes the retriever "
            "(cold-store mode). Default is False because AC-1 evaluates "
            "steady-state latency; --no-warmup will include the one-time "
            "retriever cold load in the first run's timing and typically "
            "fail the 2s target."
        ),
    )
    return parser.parse_args(argv)


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433 (deferred import is intentional)

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _cuda_sync(device: str) -> None:
    """Drain pending CUDA kernels so wall-clock timing is accurate.

    No-op when not running on cuda or when torch is not importable.
    """
    if not device.startswith("cuda"):
        return
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        _log(f"cuda.synchronize failed (non-fatal): {exc!r}")


def _build_skip_record(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "benchmark": "save_latency",
        "session_turns": int(args.session_turns),
        "runs": int(args.runs),
        "per_run_seconds": [],
        "p50_seconds": float("nan"),
        "p95_seconds": float("nan"),
        "target_seconds": 2.0,
        "pass": False,
        "model_id": args.model_path or "google/gemma-4-E2B-it",
        "device": args.device,
        "timestamp": _iso_utc_now(),
        "host": socket.gethostname(),
        "notes": reason,
        "warmup_save_seconds": None,
        "warmup_skipped": bool(getattr(args, "no_warmup", False)),
    }


def _build_error_record(
    args: argparse.Namespace, reason: str, detail: str
) -> dict[str, Any]:
    return {
        "benchmark": "save_latency",
        "session_turns": int(args.session_turns),
        "runs": int(args.runs),
        "per_run_seconds": [],
        "p50_seconds": float("nan"),
        "p95_seconds": float("nan"),
        "target_seconds": 2.0,
        "pass": False,
        "model_id": args.model_path or "google/gemma-4-E2B-it",
        "device": args.device,
        "timestamp": _iso_utc_now(),
        "host": socket.gethostname(),
        "notes": f"{reason}: {detail}",
        "error": True,
        "warmup_save_seconds": None,
        "warmup_skipped": bool(getattr(args, "no_warmup", False)),
    }


def _seed_session(chat: Any, session_turns: int, prefill_text: str) -> None:
    """Run ``session_turns`` plain chat turns on ``chat`` using canned lines."""
    for i in range(session_turns):
        user_text = (
            CANNED_PROMPTS[i] if i < len(CANNED_PROMPTS) else prefill_text
        )
        with contextlib.redirect_stdout(sys.stderr):
            chat.plain_chat_turn(user_text)


def _timed_save_cycle(
    *,
    chat: Any,
    device: str,
    session_turns: int,
    prefill_text: str,
) -> tuple[float, bool, str]:
    """Execute one seed-then-save cycle against an already-constructed chat.

    Returns ``(elapsed_seconds, save_ok, note)``. The chat must already have
    its model loaded; this function only drives ``start_new_session`` +
    seeding + ``save_current_session`` and times the save in isolation.
    """
    with contextlib.redirect_stdout(sys.stderr):
        chat.start_new_session()
    _seed_session(chat, session_turns, prefill_text)

    # Drain kernels BEFORE timing so seeding work doesn't leak into t0.
    _cuda_sync(device)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        ok = chat.save_current_session()
    _cuda_sync(device)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    note = "" if ok else "save_current_session returned False"
    return elapsed, bool(ok), note


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # CUDA skip / strict-fail early so we don't import heavy deps unless we
    # actually intend to run.
    if args.device.startswith("cuda") and not _cuda_available():
        if args.strict:
            record = _build_error_record(
                args,
                "environmental_error",
                "cuda-unavailable-and-strict-mode",
            )
            _emit(record)
            return 2
        if args.skip_if_no_cuda:
            record = _build_skip_record(args, "cuda-unavailable-skipped")
            _emit(record)
            return 0

    # Import MemoryChat only after the CUDA skip gate — importing pulls in
    # transformers/torch, which is expensive on CPU-only hosts.
    try:
        from scripts.interactive_memory_chat import MemoryChat  # type: ignore
    except Exception as exc:  # noqa: BLE001
        record = _build_error_record(
            args,
            "import_error",
            f"could not import MemoryChat: {exc!r}",
        )
        traceback.print_exc(file=sys.stderr)
        _emit(record)
        return 2

    args.store_root.mkdir(parents=True, exist_ok=True)
    _log(
        f"starting bench: runs={args.runs} session_turns={args.session_turns} "
        f"device={args.device} store_root={args.store_root} "
        f"no_warmup={bool(args.no_warmup)}"
    )

    per_run_seconds: list[float] = []
    notes: list[str] = []
    all_ok = True
    warmup_save_seconds: float | None = None
    warmup_skipped: bool = bool(args.no_warmup)

    try:
        # Construct ONE MemoryChat that owns the entire bench. Warmup and
        # all timed runs share this instance so the retriever is lazy-
        # loaded exactly once (during the warmup save) and every timed
        # save exercises the steady-state ``refresh_handles`` path.
        chat = MemoryChat(
            store_root=args.store_root,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            memory_mode="off",  # no recall needed for the save bench
            device=args.device,
        )
        with contextlib.redirect_stdout(sys.stderr):
            chat.load_model()

        if not args.no_warmup:
            _log("warmup: seeding + /save (one-time retriever cold load) ...")
            w_elapsed, w_ok, w_note = _timed_save_cycle(
                chat=chat,
                device=args.device,
                session_turns=args.session_turns,
                prefill_text=args.prefill_text,
            )
            warmup_save_seconds = float(w_elapsed)
            if not w_ok:
                notes.append(f"warmup:{w_note}")
            _log(
                f"  warmup /save took {w_elapsed:.3f}s ok={w_ok} "
                f"(excluded from p50/p95)"
            )
        else:
            _log(
                "warmup: SKIPPED (--no-warmup); first timed run will "
                "include the one-time retriever cold load."
            )

        for run_index in range(args.runs):
            _log(f"run {run_index + 1}/{args.runs}: seeding + /save ...")
            elapsed, ok, note = _timed_save_cycle(
                chat=chat,
                device=args.device,
                session_turns=args.session_turns,
                prefill_text=args.prefill_text,
            )
            per_run_seconds.append(float(elapsed))
            if not ok:
                all_ok = False
                notes.append(f"run{run_index}:{note}")
            _log(f"  run {run_index + 1} /save took {elapsed:.3f}s ok={ok}")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        record = _build_error_record(
            args,
            "runtime_error",
            f"{type(exc).__name__}: {exc}",
        )
        record["per_run_seconds"] = per_run_seconds
        record["warmup_save_seconds"] = warmup_save_seconds
        record["warmup_skipped"] = warmup_skipped
        _emit(record)
        return 2

    target = 2.0
    over_target = [s for s in per_run_seconds if s >= target]
    if over_target:
        all_ok = False
        notes.append(
            f"{len(over_target)}/{len(per_run_seconds)} run(s) exceeded target {target}s"
        )

    p50 = _percentile(per_run_seconds, 50.0)
    p95 = _percentile(per_run_seconds, 95.0)

    note_str = "; ".join(notes) if notes else "all runs under target"
    record = {
        "benchmark": "save_latency",
        "session_turns": int(args.session_turns),
        "runs": int(args.runs),
        "per_run_seconds": [float(x) for x in per_run_seconds],
        "p50_seconds": float(p50),
        "p95_seconds": float(p95),
        "target_seconds": float(target),
        "pass": bool(all_ok),
        "model_id": args.model_path or "google/gemma-4-E2B-it",
        "device": args.device,
        "timestamp": _iso_utc_now(),
        "host": socket.gethostname(),
        "notes": note_str,
        "warmup_save_seconds": (
            float(warmup_save_seconds)
            if warmup_save_seconds is not None
            else None
        ),
        "warmup_skipped": bool(warmup_skipped),
    }
    # A tiny extra: mean is useful for sanity checks but not part of the
    # contract — include it if we have data.
    if per_run_seconds:
        try:
            record["mean_seconds"] = float(statistics.fmean(per_run_seconds))
        except Exception:  # noqa: BLE001
            pass

    _emit(record)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # Last-ditch: ensure we always emit a parseable JSON on stdout even
        # if something upstream of main() exploded.
        traceback.print_exc(file=sys.stderr)
        _emit(
            {
                "benchmark": "save_latency",
                "session_turns": 0,
                "runs": 0,
                "per_run_seconds": [],
                "p50_seconds": float("nan"),
                "p95_seconds": float("nan"),
                "target_seconds": 2.0,
                "pass": False,
                "model_id": "",
                "device": "",
                "timestamp": _iso_utc_now(),
                "host": socket.gethostname(),
                "notes": f"fatal_unhandled: {type(exc).__name__}: {exc}",
                "error": True,
                "warmup_save_seconds": None,
                "warmup_skipped": False,
            }
        )
        sys.exit(2)
