# WS-5 — AUS3000 End-to-End Eval Report

**Mission:** chuk-lazurus-n7k  **Batch:** 4  **Depends on:** WS-4  **Owner:** single teammate

## Scope
Run the WS-4 pipeline against the real AUS3000 clause-aligned `torch_store` and benchmark
fixture. Commit a concise results markdown. Infrastructure is the product; the number is
informational.

## Exclusive file ownership (edit only these)
- `docs/learned_router/eval/aus3000_eval.md`                 (new — ≤ 150 lines)
- `docs/learned_router/eval/aus3000_report.json`             (new — raw metrics)

No source-code edits in this workstream. No new tests. Use the WS-4 CLI exactly as shipped.

## Inputs
- Store path: `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store`
- Benchmark fixture: `tests/fixtures/aus3000/benchmark/epic1_v1.json`
- Encoder: `bow` (primary run, offline, deterministic). `gemma-embed` is **optional** — include
  only if the model weights are already cached at
  `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/`. If not cached, skip `gemma-embed`
  and say so in the report. Do not trigger a HF download from this workstream.

## Commands
```
# 1. Build dataset
uv run python tools/train_window_router.py build-dataset \
    --store-path "<store>" --out-jsonl artifacts/router/aus3000_ds.jsonl

# 2. Train (bow, CPU, <= 20 epochs)
uv run python tools/train_window_router.py train \
    --dataset artifacts/router/aus3000_ds.jsonl --encoder bow \
    --out-ckpt artifacts/router/aus3000_bow.pt --hidden 256 --epochs 20 --device cpu

# 3. Eval
uv run python tools/train_window_router.py eval \
    --ckpt artifacts/router/aus3000_bow.pt \
    --benchmark-fixture tests/fixtures/aus3000/benchmark/epic1_v1.json \
    --store-path "<store>" \
    --out-report docs/learned_router/eval/
```

## Report template (`aus3000_eval.md`)
```
# AUS3000 Learned Router — Epic-1 Results
date: 2026-04-17   fixture: epic1_v1   cases: 23
| metric | Learned MLP (bow) | TF-IDF baseline |
|--------|-------------------|-----------------|
| top-1 accuracy | X.XXX | X.XXX |
| top-3 accuracy | X.XXX | X.XXX |
| MRR            | X.XXX | X.XXX |
| train time (s) | X     | —     |

## Per-case table
| case_name | primary_clause_ids | predicted (top-1) | tfidf (top-1) | match |

## Verdict
One sentence: {beats | ties | loses to} TF-IDF baseline on top-1. Infrastructure lands regardless.

## Follow-ups
- bullet list of ideas for v2 (gemma-embed encoder, data augmentation, multi-clause handling).
```

## Quality gate before closing
- `tests/fixtures/aus3000/benchmark/epic1_v1.json` untouched.
- `uv run python tools/evaluate_aus3000_variant.py --mode single_pass_gate --device cpu --max-cases 5` still passes.
- The written markdown is ≤ 150 lines.

## Deliverable format
- `vee record completion --title "WS-5 AUS3000 eval" --body "<top-1 learned vs tfidf, verdict, path to report>" --tag chuk-lazurus-n7k --tag ws-5`
- Then `vee session close`.
