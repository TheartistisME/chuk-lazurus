# WS-4 — Generic Window Router Tool

**Mission:** chuk-lazurus-n7k  **Batch:** 3  **Depends on:** WS-3  **Owner:** single teammate

## Scope
Build a generic CLI that trains an MLP window router against **any**
`TorchKnowledgeStore`-compatible store and **any** benchmark fixture. AUS3000 is a single CLI
invocation, not a code path. No import of MLX.

## Exclusive file ownership (edit only these)
- `tools/train_window_router.py`                           (new — argparse entry point)
- `tools/_window_router/__init__.py`                       (new)
- `tools/_window_router/dataset.py`                        (new)
- `tools/_window_router/encoder.py`                        (new)
- `tools/_window_router/eval.py`                           (new)
- `tests/tools/window_router/__init__.py`                  (new, empty)
- `tests/tools/window_router/test_dataset.py`              (new)
- `tests/tools/window_router/test_encoder.py`              (new)
- `tests/tools/window_router/test_eval.py`                 (new)
- `tests/tools/window_router/test_pipeline_smoke.py`       (new)

**DO NOT TOUCH** `tools/evaluate_aus3000_variant.py`, `src/chuk_lazarus/inference/context/knowledge/*`, or anything under `src/`.

## CLI contract
```
uv run python tools/train_window_router.py build-dataset \
    --store-path <store> --out-jsonl <path> [--paraphrases N]
uv run python tools/train_window_router.py train \
    --dataset <jsonl> --encoder bow|gemma-embed --out-ckpt <path> \
    [--hidden 256] [--epochs 10] [--batch-size 32] [--lr 1e-3] [--device cpu|cuda]
    [--model-id <hf-id>]   # required when --encoder gemma-embed
uv run python tools/train_window_router.py eval \
    --ckpt <path> --benchmark-fixture <json> --store-path <store> \
    --out-report <dir>  [--encoder-cache <path>]  [--model-id <hf-id>]
```

## Module responsibilities
- **`dataset.py`** — `build_router_dataset(store: TorchKnowledgeStore, paraphrase_templates)`
  produces `(text, window_id)` pairs from: `clause_title`, `clause_id + title`, templated
  paraphrases (`"Define {title}"`, `"What is {title}?"`, `"Clause {clause_id}: {title}"`,
  `"Tell me about {title}"`), plus first-120-token excerpts of the window text decoded by the
  tokenizer. The builder must not assume any metadata field beyond those present in
  `window_metadata` (missing fields → skip that template gracefully). Write JSONL:
  `{"text": "...", "window_id": int}`. Add `{"num_windows": int}` header line optional —
  persisted separately in a sidecar `.meta.json`.
- **`encoder.py`** — two encoders, same `.encode(text) -> list[float]` / `.vocab_size` / `.dim`
  surface:
  - `BowCharacterEncoder` — pure-Python character n-gram bag, no external weights; `fit(texts)`
    populates vocab.
  - `GemmaEmbedEncoder(model_id, device)` — loads Gemma via `transformers.AutoModel`, mean-pools
    the last hidden state; fails loudly on ImportError with a clear message about `--encoder bow`
    fallback. Keep lazy import so tests without network access never trigger the download.
- **`eval.py`** — given ckpt + store + fixture:
  - loads `TorchMLPClassifier` (WS-2) weights, encoder, and store.
  - for each benchmark case with `primary_clause_ids`, resolves expected window ids via
    `TorchKnowledgeStore.window_metadata` → primary window.
  - computes top-1, top-3, MRR; runs TF-IDF baseline via `store.route_top_k(..., k=3, tokenizer)`
    on the same cases.
  - writes `report.json` and `report.md` side-by-side.
- **`train_window_router.py`** — argparse dispatch; reuses `TorchClassificationTrainer` (WS-3);
  saves `TorchMLPClassifier` weights + `{"encoder": "...", "num_labels": N, "input_size": D}`.

## Tests
CPU only, offline.
- `test_dataset.py` — given a hand-crafted 4-window metadata dict, `build_router_dataset`
  produces ≥ 1 sample per window_id and at least 4 templates fire for windows with titles.
- `test_encoder.py` — `BowCharacterEncoder.fit([...]).encode("x")` deterministic; `vocab_size`
  stable; `dim == vocab_size`.
- `test_eval.py` — synthetic store stub + fake ckpt; `top_1_accuracy` and `mrr` computed
  correctly on a 3-case synthetic benchmark.
- `test_pipeline_smoke.py` — builds a 4-window synthetic store fixture in `tmp_path` (manifest
  + window_tokens.npz + window_metadata.json), runs `build-dataset` → `train` (bow, 2 epochs) →
  `eval` on a tiny fixture, asserts the run produces `report.json` and the checkpoint file. No
  network, no HF downloads.

## Quality gate before closing
```
uv run pytest tests/tools/window_router/ -q
uv run ruff check tools/train_window_router.py tools/_window_router/
uv run pytest tests/tools/test_evaluate_aus3000_variant.py tests/inference/context/ -q   # unchanged
```

## Deliverable format
- `vee record insight --title "WS-4 generic window router tool" --tag chuk-lazurus-n7k --tag ws-4`
- `vee session close --handoff`
