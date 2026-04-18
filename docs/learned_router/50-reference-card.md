# Torch Parity + Learned Router Reference Card

## Goal

Ship a torch-parity classifier/training stack and a generic `tools/train_window_router.py` CLI that trains an MLP window router over any `(TorchKnowledgeStore, benchmark_fixture)` pair, with AUS3000 as the proving ground. Zero edits to MLX or frozen route/store files; AUS3000 `single_pass_gate` stays `23/23`.

## Current Status

- Status: `complete`
- Learned-router headline: top-1 `0.765` (`13/17`), top-3 `0.824`, MRR `0.794`
- TF-IDF baseline in the shipped CLI: top-1 `0.000`, top-3 `0.000`, MRR `0.000`
- Benchmark scope: 23 cases total, 17 evaluated, 6 skipped by CLI
- Run metadata: `encoder=bow`, `hidden=256`, `epochs=20`, `device=cpu`, `dataset=7218` samples
- AUS3000 route/store/query/evaluator regression gate remains `23/23 PASS`

## What Has Been Completed

- WS-1: Completed `torch_backend.py`, registered the `torch`/`pytorch` backend alias, and added backend plus registry coverage.
- WS-2: Added torch-native classifier parity modules: `TorchLinear`, `TorchMLPClassifier`, and `TorchTokenEmbedding`, each with tests.
- WS-3: Added `src/chuk_lazarus/training/torch/` with `TorchBaseTrainer` and `TorchClassificationTrainer`, including checkpointing and synthetic-smoke coverage.
- WS-4: Built the generic router CLI with `build-dataset`, `train`, and `eval`, plus reusable helpers under `tools/_window_router/`.
- WS-5: Ran the AUS3000 end-to-end eval against the clause-aligned store and wrote `docs/learned_router/eval/aus3000_eval.md` plus `docs/learned_router/eval/aus3000_report.json`.

## Primary Artifacts

- Training dataset: [artifacts/router/aus3000_ds.jsonl](../../artifacts/router/aus3000_ds.jsonl) (`7218` samples)
- Trained checkpoint: [artifacts/router/aus3000_bow.pt](../../artifacts/router/aus3000_bow.pt)
- Human report: [eval/aus3000_eval.md](eval/aus3000_eval.md)
- Machine report: [eval/aus3000_report.json](eval/aus3000_report.json)
- AUS3000 benchmark fixture: [tests/fixtures/aus3000/benchmark/epic1_v1.json](../../tests/fixtures/aus3000/benchmark/epic1_v1.json)
- Read-only store input: `/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store`

## Key Code Paths

- `src/chuk_lazarus/models_v2/core/backend/torch_backend.py`, `src/chuk_lazarus/models_v2/core/backend/registry.py`
- `src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py`, `torch_mlp.py`, `torch_token_embedding.py`
- `src/chuk_lazarus/training/torch/torch_base_trainer.py`, `torch_classification_trainer.py`
- `tools/train_window_router.py`, `tools/_window_router/dataset.py`, `tools/_window_router/encoder.py`, `tools/_window_router/eval.py`
- `src/chuk_lazarus/inference/context/knowledge/torch_store.py`, `src/chuk_lazarus/inference/context/knowledge/route.py`

## Key Test Coverage

- `tests/models_v2/core/backend/` for the torch backend and registry alias surface
- `tests/models_v2/models/classifiers/` for the torch classifier parity modules
- `tests/training/torch/` for the torch trainer stack and synthetic linearly separable smoke
- `tests/tools/window_router/` for the dataset, train, and eval pipeline on a synthetic 4-window store
- `tests/inference/context/` and `tests/tools/test_evaluate_aus3000_variant.py` for the frozen AUS3000 routing gate

## Commands to Reproduce

```bash
STORE=/mnt/c/Users/jehma/Desktop/AI/TradeGuru/lewisnjehmal/downloads/standards/embeddings/AS_NZS_3000-2018/gemma4_aus3000_clause_aligned_variant/torch_store

uv run python tools/train_window_router.py build-dataset \
  --store-path "$STORE" \
  --out-jsonl artifacts/router/aus3000_ds.jsonl

uv run python tools/train_window_router.py train \
  --dataset artifacts/router/aus3000_ds.jsonl \
  --encoder bow \
  --out-ckpt artifacts/router/aus3000_bow.pt \
  --epochs 20 \
  --device cpu

uv run python tools/train_window_router.py eval \
  --ckpt artifacts/router/aus3000_bow.pt \
  --benchmark-fixture tests/fixtures/aus3000/benchmark/epic1_v1.json \
  --store-path "$STORE" \
  --out-report docs/learned_router/eval/
```

## Implementation Snapshot

- The learned-router pipeline is generic over `--store-path` and `--benchmark-fixture`; AUS3000 is one invocation, not a special case.
- The shipped AUS3000 run used `bow` because Gemma weights were not cached locally.
- The evaluation outputs live under `docs/learned_router/eval/`, with the training artifact under `artifacts/router/`.
- The current CLI report still shows an all-zero TF-IDF baseline, which is part of the follow-up work below.

## Next Exact Moves

- Swap `bow` for `gemma-embed` once Gemma weights are cached locally; this is the most likely way to close the 4-case umbrella/adjacent-clause gap.
- Extend eval to score per-clause recall for prompts that span more than one primary clause id.
- Investigate the 6 benchmark rows the CLI skipped and surface them in the report so the skip rate is auditable.
- Reinvestigate the TF-IDF baseline path in `tools/train_window_router.py eval`; the all-zero baseline suggests the comparator is not wired through correctly or the indexed corpus is empty.
- Add paraphrase-heavy data augmentation per clause to improve robustness on comparison/rule-style prompts.
