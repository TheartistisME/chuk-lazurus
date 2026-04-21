#!/usr/bin/env python3
"""Benchmark: per-turn chat-decode latency regression under live indexing.

AC-2 primary for the release-gate-ac4 lead.

Semantics
---------
This benchmark measures the wall-clock cost of running chat decode with
LiveIndexer active (capturing residuals and spooling window tuples onto a
low-priority CUDA stream + background daemon thread) versus a quiescent
baseline in which the indexer is detached and no capture work is spawned.
The acceptance target is **< 10% p50 regression** on per-turn decode
latency.

Two-phase measurement
---------------------
Both phases run against the SAME in-process Gemma model (loaded once) on
the SAME canned-prompt rotation, so the only variable between phases is
whether the live indexer closure is firing during each ``plain_chat_turn``.

* PHASE 1 — ``active_indexing``:
    ``chat.start_new_session()`` allocates a fresh ``LiveIndexer`` and the
    ``on_window`` closure enqueues tuples on every window boundary. We
    record per-turn wall-clock (with ``torch.cuda.synchronize`` on CUDA)
    for ``warmup_turns + turns`` turns and keep only the trailing ``turns``
    for percentile computation. After the loop we drain the indexer via
    ``stop()`` (which does NOT run compaction and does NOT write a
    manifest) and detach it. This keeps phase 2 clean without paying the
    full ``/save`` cost that AC-1 measures separately.

* PHASE 2 — ``quiescent``:
    ``chat.start_new_session()`` again (fresh session id), then we set
    ``chat.indexer = None`` so the ``_make_on_window`` closure's
    ``self.indexer is None`` guard short-circuits to a no-op. Same
    canned-prompt rotation, same warmup + timing loop.

We then compute ``p50_regression_pct = (active.p50 / quiescent.p50 - 1)
* 100`` (and likewise for p95) and pass iff p50_regression_pct ≤ 10.0.

Output
------
A single JSON object on stdout. INFO lines go to stderr so stdout stays
parseable by consumers such as the release-gate-ac4 lead.

Exit codes
----------
* ``0`` — pass OR the cuda-absent skip path was taken (unless --strict).
* ``1`` — p50 regression exceeded 10%.
* ``2`` — environmental error (model load failed, import error, etc.).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
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


# Canned user lines rotated through both phases. Keeping inputs varied
# prevents the model from collapsing into a cache-friendly degenerate path,
# but keeping the list short-and-deterministic means the two phases see
# IDENTICAL prompt sequences (modulo session_id) — which is the comparison
# we want.
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
    print(f"[bench_chat_latency] {msg}", file=sys.stderr, flush=True)


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
        prog="bench_chat_latency.py",
        description=(
            "Measure per-turn chat-decode latency regression incurred by "
            "active LiveIndexer vs a quiescent baseline (target <10% p50, "
            "AC-2 for release-gate-ac4)."
        ),
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Timed turns per phase after warmup (default 10).",
    )
    parser.add_argument(
        "--warmup-turns",
        type=int,
        default=2,
        help="Warmup turns dropped from percentiles per phase (default 2).",
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
        help="Max new tokens per assistant turn (default 64; keeps bench fast).",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path(f"/tmp/bench-chat-latency-{os.getpid()}"),
        help="Store root directory (only used during phase 1 indexer writes).",
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
        "--seed",
        type=int,
        default=1234,
        help="Seed for the deterministic prompt rotation (default 1234).",
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


def _empty_phase() -> dict[str, Any]:
    return {
        "per_turn_seconds": [],
        "p50_seconds": float("nan"),
        "p95_seconds": float("nan"),
    }


def _build_skip_record(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "benchmark": "chat_latency",
        "turns": int(args.turns),
        "model_id": args.model_path or "google/gemma-4-E2B-it",
        "device": args.device,
        "timestamp": _iso_utc_now(),
        "host": socket.gethostname(),
        "active_indexing": _empty_phase(),
        "quiescent": _empty_phase(),
        "p50_regression_pct": float("nan"),
        "p95_regression_pct": float("nan"),
        "target_pct": 10.0,
        "pass": False,
        "notes": reason,
    }


def _build_error_record(
    args: argparse.Namespace, reason: str, detail: str
) -> dict[str, Any]:
    rec = _build_skip_record(args, f"{reason}: {detail}")
    rec["error"] = True
    return rec


def _pick_prompts(total: int, rng: random.Random) -> list[str]:
    """Produce a deterministic rotation of ``total`` prompts.

    We always start at a shuffled offset so repeated runs with the same seed
    produce the same sequence, but back-to-back phases within one run see
    the SAME prompt list so the comparison is apples-to-apples.
    """
    base = list(CANNED_PROMPTS)
    rng.shuffle(base)
    out: list[str] = []
    for i in range(total):
        out.append(base[i % len(base)])
    return out


def _run_phase(
    *,
    chat: Any,
    prompts: list[str],
    warmup_turns: int,
    turns: int,
    device: str,
    phase_label: str,
) -> list[float]:
    """Execute one warmup+timed loop; return per-turn seconds for TIMED turns only."""
    per_turn: list[float] = []
    total = warmup_turns + turns
    if len(prompts) < total:
        raise RuntimeError(
            f"{phase_label}: prompt list shorter ({len(prompts)}) than "
            f"required total turns ({total})"
        )

    for i in range(total):
        user_text = prompts[i]
        # Drain kernels BEFORE timing so prior work doesn't leak into t0.
        _cuda_sync(device)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            chat.plain_chat_turn(user_text)
        _cuda_sync(device)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        tag = "warmup" if i < warmup_turns else "timed"
        _log(f"  {phase_label} turn {i + 1}/{total} ({tag}) {elapsed:.3f}s")
        if i >= warmup_turns:
            per_turn.append(float(elapsed))
    return per_turn


def _detach_indexer(chat: Any) -> None:
    """Drain the phase-1 indexer WITHOUT compaction, then detach it.

    We call ``stop()`` (not ``flush_and_close``) so no manifest is written —
    AC-2 only measures chat latency, never /save. If ``stop()`` is missing
    or raises, we fall back to a plain detach; the daemon thread is marked
    daemon=True so it will be torn down on process exit anyway.
    """
    indexer = getattr(chat, "indexer", None)
    if indexer is None:
        return
    stop_fn = getattr(indexer, "stop", None)
    if callable(stop_fn):
        try:
            stop_fn()
        except Exception as exc:  # noqa: BLE001 - best-effort drain
            _log(f"indexer.stop() raised (non-fatal): {exc!r}")
    setattr(chat, "indexer", None)


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
    total_turns = args.warmup_turns + args.turns
    rng = random.Random(args.seed)
    prompts = _pick_prompts(total_turns, rng)

    _log(
        f"starting bench: turns={args.turns} warmup={args.warmup_turns} "
        f"device={args.device} store_root={args.store_root}"
    )

    try:
        chat = MemoryChat(
            store_root=args.store_root,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            memory_mode="off",  # no recall needed for the chat-latency bench
            device=args.device,
        )
        with contextlib.redirect_stdout(sys.stderr):
            chat.load_model()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        record = _build_error_record(
            args,
            "model_load_error",
            f"{type(exc).__name__}: {exc}",
        )
        _emit(record)
        return 2

    # ------------------------------------------------------------------
    # PHASE 1 — active indexing.
    # ------------------------------------------------------------------
    active_per_turn: list[float] = []
    try:
        with contextlib.redirect_stdout(sys.stderr):
            chat.start_new_session()
        if getattr(chat, "indexer", None) is None:
            _log(
                "WARNING: phase 1 indexer is None (model/tokenizer missing or "
                "init failed); phase 1 will still measure chat latency but the "
                "active/quiescent delta will collapse to ~0."
            )
        active_per_turn = _run_phase(
            chat=chat,
            prompts=prompts,
            warmup_turns=args.warmup_turns,
            turns=args.turns,
            device=args.device,
            phase_label="active",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        record = _build_error_record(
            args,
            "phase1_runtime_error",
            f"{type(exc).__name__}: {exc}",
        )
        _emit(record)
        return 2
    finally:
        # Drain the worker WITHOUT compaction/manifest, then detach.
        try:
            _detach_indexer(chat)
        except Exception as exc:  # noqa: BLE001
            _log(f"detach_indexer raised (non-fatal): {exc!r}")

    # ------------------------------------------------------------------
    # PHASE 2 — quiescent.
    # ------------------------------------------------------------------
    quiescent_per_turn: list[float] = []
    try:
        with contextlib.redirect_stdout(sys.stderr):
            chat.start_new_session()
        # Force-quiesce: the on_window closure short-circuits when indexer is None.
        setattr(chat, "indexer", None)
        quiescent_per_turn = _run_phase(
            chat=chat,
            prompts=prompts,
            warmup_turns=args.warmup_turns,
            turns=args.turns,
            device=args.device,
            phase_label="quiescent",
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        record = _build_error_record(
            args,
            "phase2_runtime_error",
            f"{type(exc).__name__}: {exc}",
        )
        record["active_indexing"] = {
            "per_turn_seconds": active_per_turn,
            "p50_seconds": _percentile(active_per_turn, 50.0),
            "p95_seconds": _percentile(active_per_turn, 95.0),
        }
        _emit(record)
        return 2

    # ------------------------------------------------------------------
    # Stats + record assembly.
    # ------------------------------------------------------------------
    active_p50 = _percentile(active_per_turn, 50.0)
    active_p95 = _percentile(active_per_turn, 95.0)
    quiescent_p50 = _percentile(quiescent_per_turn, 50.0)
    quiescent_p95 = _percentile(quiescent_per_turn, 95.0)

    notes_list: list[str] = []

    def _pct(active: float, quiescent: float) -> float:
        if quiescent <= 0.0 or quiescent != quiescent:  # zero / NaN guard
            notes_list.append("quiescent baseline is zero/NaN; regression undefined")
            return float("nan")
        return (active / quiescent - 1.0) * 100.0

    p50_reg = _pct(active_p50, quiescent_p50)
    p95_reg = _pct(active_p95, quiescent_p95)
    target = 10.0
    # pass iff p50 regression is within target; NaN regression => fail.
    passed = (p50_reg == p50_reg) and (p50_reg <= target)

    if not passed and (p50_reg == p50_reg):
        notes_list.append(
            f"p50 regression {p50_reg:.2f}% exceeds target {target:.2f}%"
        )
    if not notes_list:
        notes_list.append("regression within target")

    record: dict[str, Any] = {
        "benchmark": "chat_latency",
        "turns": int(args.turns),
        "model_id": args.model_path or "google/gemma-4-E2B-it",
        "device": args.device,
        "timestamp": _iso_utc_now(),
        "host": socket.gethostname(),
        "active_indexing": {
            "per_turn_seconds": [float(x) for x in active_per_turn],
            "p50_seconds": float(active_p50),
            "p95_seconds": float(active_p95),
        },
        "quiescent": {
            "per_turn_seconds": [float(x) for x in quiescent_per_turn],
            "p50_seconds": float(quiescent_p50),
            "p95_seconds": float(quiescent_p95),
        },
        "p50_regression_pct": float(p50_reg),
        "p95_regression_pct": float(p95_reg),
        "target_pct": float(target),
        "pass": bool(passed),
        "notes": "; ".join(notes_list),
    }
    # Non-contract extras useful for sanity checks.
    try:
        if active_per_turn:
            record["active_indexing"]["mean_seconds"] = float(
                statistics.fmean(active_per_turn)
            )
        if quiescent_per_turn:
            record["quiescent"]["mean_seconds"] = float(
                statistics.fmean(quiescent_per_turn)
            )
    except Exception:  # noqa: BLE001
        pass

    _emit(record)
    return 0 if passed else 1


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
                "benchmark": "chat_latency",
                "turns": 0,
                "model_id": "",
                "device": "",
                "timestamp": _iso_utc_now(),
                "host": socket.gethostname(),
                "active_indexing": {
                    "per_turn_seconds": [],
                    "p50_seconds": float("nan"),
                    "p95_seconds": float("nan"),
                },
                "quiescent": {
                    "per_turn_seconds": [],
                    "p50_seconds": float("nan"),
                    "p95_seconds": float("nan"),
                },
                "p50_regression_pct": float("nan"),
                "p95_regression_pct": float("nan"),
                "target_pct": 10.0,
                "pass": False,
                "notes": f"fatal_unhandled: {type(exc).__name__}: {exc}",
                "error": True,
            }
        )
        sys.exit(2)
