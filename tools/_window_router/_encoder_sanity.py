"""Gate 0.2 — encoder cos-sim ranking sanity probe.

Builds an :class:`HFSentenceEncoder`, encodes five reference prompts, and
asserts four relative cosine-similarity orderings. The probe is relative
only (no absolute thresholds) so it is robust to normalization choices
and per-model scale differences.

Only the sentence-vector slice of the encoder output is used — any
clause-hierarchy one-hot tail is discarded via ``encoder.dim -
encoder.clause_feature_dim``. For the five reference prompts the clause
vocab is trivially empty, but the slice keeps the probe correct if the
encoder is ever fit on a larger corpus first.
"""

from __future__ import annotations

import argparse
import math
import time
from itertools import combinations
from typing import Any

from tools._window_router.cache import resolve_device
from tools._window_router.encoder import HFSentenceEncoder

Q1 = "What does clause 1.4.2 mean by accessible, in plain language?"
T1 = "Accessible"
T2 = "Basic protection"
Q2 = (
    "What mix of training, qualifications, knowledge, and experience "
    "makes someone a competent person under clause 1.4.34?"
)
T3 = "Competent person"

REFERENCE_PROMPTS: dict[str, str] = {
    "Q1": Q1,
    "T1": T1,
    "T2": T2,
    "Q2": Q2,
    "T3": T3,
}

COS_SIM_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("Q1", "T1", "Q1", "T2"),
    ("Q2", "T3", "Q2", "T2"),
    ("Q1", "T1", "Q2", "T1"),
    ("Q2", "T3", "Q1", "T3"),
)


def _cosine(u: list[float], v: list[float]) -> float:
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v, strict=True):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (math.sqrt(nu) * math.sqrt(nv))


def compute_cosine_similarities(encoder: Any) -> dict[tuple[str, str], float]:
    """Encode the five reference prompts once and return all pairwise cos sims.

    Uses only the first ``encoder.dim - encoder.clause_feature_dim``
    columns (the sentence-vector slice) so any clause-hierarchy tail is
    ignored.
    """
    keys = list(REFERENCE_PROMPTS.keys())
    texts = [REFERENCE_PROMPTS[k] for k in keys]

    matrix = encoder.encode_many_tensor(texts, batch_size=len(texts))
    rows = [[float(x) for x in row] for row in matrix.tolist()]

    sentence_dim = int(encoder.dim) - int(encoder.clause_feature_dim)
    sliced = [row[:sentence_dim] for row in rows]

    sims: dict[tuple[str, str], float] = {}
    for (i, ki), (j, kj) in combinations(enumerate(keys), 2):
        value = _cosine(sliced[i], sliced[j])
        sims[(ki, kj)] = value
        sims[(kj, ki)] = value
    return sims


def _lookup(sims: dict[tuple[str, str], float], a: str, b: str) -> float:
    if (a, b) in sims:
        return sims[(a, b)]
    if (b, a) in sims:
        return sims[(b, a)]
    raise KeyError(f"no cosine similarity for pair ({a!r}, {b!r})")


def run_sanity(
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "auto",
) -> dict[str, Any]:
    """Build the encoder, run the probe, and return a structured result."""
    resolved_device = resolve_device(device) if device == "auto" else device
    start = time.perf_counter()

    encoder = HFSentenceEncoder(model_id=model_id, device=resolved_device)
    encoder.fit(list(REFERENCE_PROMPTS.values()))

    sims = compute_cosine_similarities(encoder)

    pair_results: list[dict[str, Any]] = []
    all_passed = True
    for a, b, c, d in COS_SIM_PAIRS:
        cos_ab = _lookup(sims, a, b)
        cos_cd = _lookup(sims, c, d)
        passed = cos_ab > cos_cd
        if not passed:
            all_passed = False
        pair_results.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "cos_ab": cos_ab,
                "cos_cd": cos_cd,
                "pass": passed,
            }
        )

    elapsed = time.perf_counter() - start
    return {
        "similarities": {f"{a}|{b}": v for (a, b), v in sims.items()},
        "pair_results": pair_results,
        "all_passed": all_passed,
        "elapsed_seconds": elapsed,
    }


def _format_row(result: dict[str, Any]) -> str:
    mark = "PASS" if result["pass"] else "FAIL"
    return (
        f"[{mark}] cos({result['a']},{result['b']})={result['cos_ab']:+.4f} "
        f"> cos({result['c']},{result['d']})={result['cos_cd']:+.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate 0.2 cos-sim ranking sanity probe."
    )
    parser.add_argument(
        "--model-id",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model id for the sentence encoder.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device ('auto', 'cpu', or 'cuda').",
    )
    args = parser.parse_args(argv)

    report = run_sanity(model_id=args.model_id, device=args.device)
    print(f"model_id: {args.model_id}")
    print(f"device:   {args.device}")
    print(f"elapsed:  {report['elapsed_seconds']:.3f}s")
    print("pair results:")
    for row in report["pair_results"]:
        print(f"  {_format_row(row)}")
    print("all_passed:", report["all_passed"])
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
