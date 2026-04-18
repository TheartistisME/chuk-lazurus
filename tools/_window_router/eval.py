"""Window-router evaluation.

Given a trained :class:`TorchMLPClassifier`, an encoder, a knowledge
store, and a benchmark fixture, compute legacy case-level top-1 / top-3
/ MRR alongside Batch 3 breakdowns:

- single-clause top-1 accuracy
- per-clause recall@k for prompts with multiple ``primary_clause_ids``

The evaluator resolves every primary clause id to its canonical window
via ``store.window_metadata`` and runs a TF-IDF baseline via
``store.route_top_k`` on the same cases for side-by-side comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class _StoreLike(Protocol):
    window_metadata: dict[int, dict[str, Any]]

    def route_top_k(
        self,
        query_text: str,
        tokenizer: Any,
        k: int = 3,
        expansion_ids: list[int] | None = None,
    ) -> list[int]: ...


def resolve_expected_window_id(
    store: _StoreLike, clause_id: str
) -> int | None:
    """Find the primary window whose metadata ``clause_id`` matches.

    Prefers the window with the lowest ``part_index`` (default 1), then
    the lowest window id. Returns ``None`` when no window matches.
    """
    target = str(clause_id).strip()
    if not target:
        return None
    matches: list[tuple[int, int]] = []
    for window_id, metadata in store.window_metadata.items():
        candidate = str(metadata.get("clause_id", "")).strip()
        if candidate == target:
            try:
                part_index = int(metadata.get("part_index", 1) or 1)
            except (TypeError, ValueError):
                part_index = 1
            matches.append((part_index, int(window_id)))
    if not matches:
        return None
    matches.sort()
    return matches[0][1]


def resolve_expected_window_ids(
    store: _StoreLike, clause_ids: list[str]
) -> tuple[list[int], list[str]]:
    """Resolve each primary clause id to a unique expected window id."""
    resolved: list[int] = []
    missing: list[str] = []
    seen: set[int] = set()
    for clause_id in clause_ids:
        window_id = resolve_expected_window_id(store, clause_id)
        if window_id is None:
            missing.append(str(clause_id))
            continue
        window_int = int(window_id)
        if window_int not in seen:
            seen.add(window_int)
            resolved.append(window_int)
    return resolved, missing


def score_predictions(
    pred_windows: list[list[int]], expected: list[int]
) -> dict[str, float | int]:
    """Compute top-1 accuracy, top-3 accuracy, and mean reciprocal rank."""
    n = len(expected)
    if n == 0:
        return {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "mrr": 0.0, "n": 0}
    top1 = 0
    top3 = 0
    rr_sum = 0.0
    for preds, gold in zip(pred_windows, expected):
        preds_int = [int(p) for p in preds]
        if preds_int and preds_int[0] == int(gold):
            top1 += 1
        if int(gold) in preds_int[:3]:
            top3 += 1
        for rank, pred in enumerate(preds_int, start=1):
            if pred == int(gold):
                rr_sum += 1.0 / rank
                break
    return {
        "top_1_accuracy": top1 / n,
        "top_3_accuracy": top3 / n,
        "mrr": rr_sum / n,
        "n": n,
    }


def score_ranked_predictions(
    pred_windows: list[list[int]], expected_groups: list[list[int]]
) -> dict[str, float | int]:
    """Compute case-level top-1 / top-3 / MRR with multi-relevant support."""
    n = len(expected_groups)
    if n == 0:
        return {"top_1_accuracy": 0.0, "top_3_accuracy": 0.0, "mrr": 0.0, "n": 0}

    top1 = 0
    top3 = 0
    rr_sum = 0.0
    for preds, golds in zip(pred_windows, expected_groups):
        preds_int = [int(p) for p in preds]
        gold_set = {int(gold) for gold in golds}
        if preds_int and preds_int[0] in gold_set:
            top1 += 1
        if any(pred in gold_set for pred in preds_int[:3]):
            top3 += 1
        for rank, pred in enumerate(preds_int, start=1):
            if pred in gold_set:
                rr_sum += 1.0 / rank
                break

    return {
        "top_1_accuracy": top1 / n,
        "top_3_accuracy": top3 / n,
        "mrr": rr_sum / n,
        "n": n,
    }


def score_single_clause_top1(
    pred_windows: list[list[int]], expected_groups: list[list[int]]
) -> dict[str, float | int]:
    """Compute top-1 accuracy on single-clause prompts only."""
    single_preds: list[list[int]] = []
    single_expected: list[int] = []
    for preds, golds in zip(pred_windows, expected_groups):
        if len(golds) != 1:
            continue
        single_preds.append(preds)
        single_expected.append(int(golds[0]))

    metrics = score_predictions(single_preds, single_expected)
    return {
        "single_clause_top_1_accuracy": float(metrics["top_1_accuracy"]),
        "single_clause_n": int(metrics["n"]),
    }


def _recall_at_k(preds: list[int], expected: list[int], k: int) -> float:
    gold_set = {int(gold) for gold in expected}
    if not gold_set:
        return 0.0
    pred_set = {int(pred) for pred in preds[: int(k)]}
    hits = len(gold_set & pred_set)
    return hits / len(gold_set)


def score_multi_clause_recall(
    pred_windows: list[list[int]], expected_groups: list[list[int]], k: int
) -> dict[str, float | int]:
    """Compute micro-averaged per-clause recall@k on multi-clause prompts."""
    total_hits = 0.0
    total_expected = 0
    case_count = 0

    for preds, golds in zip(pred_windows, expected_groups):
        gold_set = list(dict.fromkeys(int(gold) for gold in golds))
        if len(gold_set) <= 1:
            continue
        case_count += 1
        total_expected += len(gold_set)
        total_hits += _recall_at_k(preds, gold_set, k) * len(gold_set)

    recall = 0.0 if total_expected == 0 else total_hits / total_expected
    return {
        "multi_clause_recall_at_k": recall,
        "multi_clause_n": case_count,
        "multi_clause_clause_count": total_expected,
    }


def _model_topk(model: Any, features: list[float], k: int) -> list[int]:
    import torch  # noqa: PLC0415

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    effective_k = min(int(k), int(logits.shape[-1]))
    if effective_k <= 0:
        return []
    topk = torch.topk(logits, k=effective_k)
    return [int(i) for i in topk.indices.tolist()]


def _baseline_topk(
    store: _StoreLike, prompt: str, tokenizer: Any, k: int
) -> list[int]:
    try:
        windows = store.route_top_k(prompt, tokenizer=tokenizer, k=int(k))
    except TypeError:
        windows = store.route_top_k(prompt, tokenizer, int(k))
    except (AttributeError, ValueError):
        if tokenizer is None:
            return []
        raise
    return [int(w) for w in windows]


def evaluate(
    *,
    model: Any,
    encoder: Any,
    store: _StoreLike,
    cases: list[dict[str, Any]],
    tokenizer: Any | None = None,
    k: int = 3,
) -> dict[str, Any]:
    """Run the classifier and the TF-IDF baseline against a benchmark.

    Cases without ``primary_clause_ids`` or whose primary clauses do not
    all resolve to window ids are skipped with a note in ``skipped``.
    """
    model_preds: list[list[int]] = []
    baseline_preds: list[list[int]] = []
    expected_groups: list[list[int]] = []
    per_case: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case in cases:
        prompt = str(case.get("prompt") or "")
        primary_ids = [str(x) for x in (case.get("primary_clause_ids") or ())]
        if not prompt or not primary_ids:
            skipped.append(
                {
                    "name": case.get("name"),
                    "reason": "missing prompt or primary_clause_ids",
                }
            )
            continue
        expected_windows, missing_clause_ids = resolve_expected_window_ids(
            store, primary_ids
        )
        if missing_clause_ids:
            skipped.append(
                {
                    "name": case.get("name"),
                    "reason": (
                        "no window for clauses "
                        + ", ".join(repr(clause_id) for clause_id in missing_clause_ids)
                    ),
                }
            )
            continue

        features = list(encoder.encode(prompt))
        model_top = _model_topk(model, features, k=k)
        baseline_top = _baseline_topk(store, prompt, tokenizer, k=k)

        model_preds.append(model_top)
        baseline_preds.append(baseline_top)
        expected_groups.append(expected_windows)
        per_case.append(
            {
                "name": case.get("name"),
                "category": case.get("category"),
                "prompt": prompt,
                "primary_clause_id": primary_ids[0],
                "primary_clause_ids": primary_ids,
                "expected_window_id": int(expected_windows[0]),
                "expected_window_ids": expected_windows,
                "is_multi_clause": len(expected_windows) > 1,
                "model_top3": model_top,
                "baseline_top3": baseline_top,
                "model_recall_at_k": _recall_at_k(model_top, expected_windows, k),
                "baseline_recall_at_k": _recall_at_k(
                    baseline_top, expected_windows, k
                ),
            }
        )

    model_metrics = score_ranked_predictions(model_preds, expected_groups)
    model_metrics.update(score_single_clause_top1(model_preds, expected_groups))
    model_metrics.update(score_multi_clause_recall(model_preds, expected_groups, k))

    baseline_metrics = score_ranked_predictions(baseline_preds, expected_groups)
    baseline_metrics.update(score_single_clause_top1(baseline_preds, expected_groups))
    baseline_metrics.update(
        score_multi_clause_recall(baseline_preds, expected_groups, k)
    )

    return {
        "model": dict(model_metrics),
        "tfidf_baseline": dict(baseline_metrics),
        "cases": per_case,
        "skipped": skipped,
        "k": int(k),
    }


def _format_metric_block(label: str, metrics: dict[str, Any], *, k: int) -> list[str]:
    lines = [f"## {label}"]
    lines.append(f"- n: {metrics.get('n', 0)}")
    for count_key in ("single_clause_n", "multi_clause_n", "multi_clause_clause_count"):
        if count_key in metrics:
            lines.append(f"- {count_key}: {metrics.get(count_key, 0)}")
    metric_labels = (
        ("top_1_accuracy", "top_1_accuracy"),
        ("top_3_accuracy", "top_3_accuracy"),
        ("single_clause_top_1_accuracy", "single_clause_top_1_accuracy"),
        ("multi_clause_recall_at_k", f"multi_clause_recall@{int(k)}"),
        ("mrr", "mrr"),
    )
    for key, display_name in metric_labels:
        if key not in metrics:
            continue
        value = metrics.get(key, 0.0)
        lines.append(f"- {display_name}: {float(value):.4f}")
    lines.append("")
    return lines


def write_report(out_dir: str | Path, report: dict[str, Any]) -> Path:
    """Persist ``report.json`` and ``report.md`` side-by-side. Returns the dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = ["# Window Router Evaluation", ""]
    report_k = int(report.get("k", 3) or 3)
    lines.extend(
        _format_metric_block("Model (MLP)", report.get("model", {}), k=report_k)
    )
    lines.extend(
        _format_metric_block(
            "TF-IDF Baseline", report.get("tfidf_baseline", {}), k=report_k
        )
    )
    skipped = report.get("skipped", [])
    if skipped:
        lines.append("## Skipped")
        for row in skipped:
            lines.append(f"- {row.get('name')}: {row.get('reason')}")
        lines.append("")
    (out_path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


__all__ = ["evaluate", "resolve_expected_window_id", "score_predictions", "write_report"]
