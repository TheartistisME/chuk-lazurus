# Training Guide

`tools/train_window_router.py` is the end-to-end entry point for any `TorchKnowledgeStore` plus benchmark-fixture pair. The flow is store-agnostic: `build-dataset` reads window metadata from the store, `train` fits a `TorchMLPClassifier` with `TorchClassificationTrainer`, and `eval` scores the trained router against the benchmark and the store's TF-IDF baseline.

## Inputs

- Store path: a directory loadable by `TorchKnowledgeStore.load(<STORE_PATH>)`.
- Benchmark fixture: a JSON file with a `cases` array; `eval` keeps only cases that have `primary_clause_ids`.
- Training data: JSONL records with `text` and `window_id`.
- Encoders: `bow` is pure Python; `gemma-embed` lazy-loads `transformers` and requires `--model-id`.

## `build-dataset`

Example:

```bash
uv run python tools/train_window_router.py build-dataset \
  --store-path <STORE_PATH> \
  --out-jsonl <DATASET_JSONL> \
  --tokenizer-model-id <TOKENIZER_MODEL_ID>
```

Expected outputs:

- Writes `<DATASET_JSONL>` containing one JSON object per line with `text` and `window_id`.
- Writes `<DATASET_JSONL>.meta.json` with `num_windows`, `num_samples`, `num_templates`, and `store_path`.
- Prints `wrote N samples to ...` and `wrote sidecar meta to ...`.

Key knobs:

- `--store-path` selects the source store.
- `--out-jsonl` selects the training-set destination.
- `--tokenizer-model-id` is optional; when present, excerpt samples are added and the TF-IDF baseline path becomes available later.
- `--paraphrases` caps the default paraphrase templates.

## `train`

Example:

```bash
uv run python tools/train_window_router.py train \
  --dataset <DATASET_JSONL> \
  --encoder bow \
  --out-ckpt <CKPT_PATH> \
  --epochs 10 \
  --device cpu \
  --hidden 256
```

Expected outputs:

- Writes `<CKPT_PATH>` as a `torch.save` checkpoint.
- The checkpoint payload contains `state_dict` and `meta`.
- `meta` includes `encoder`, `num_labels`, `input_size`, `hidden_size`, and `model_id`.
- `bow` checkpoints also store `encoder_vocab` and `encoder_n`.
- Prints `saved checkpoint ... (num_labels=..., dim=...)`.

Key knobs:

- `--encoder bow|gemma-embed` selects the feature encoder.
- `--epochs` controls `TorchTrainerConfig.num_epochs`.
- `--device` selects `cpu` or `cuda`.
- `--hidden` is the router hidden-size knob; in guide terminology this is the `--hidden-size` setting, and the checkpoint stores it as `hidden_size`.
- `--model-id` is required with `--encoder gemma-embed`.

## `eval`

Example:

```bash
uv run python tools/train_window_router.py eval \
  --ckpt <CKPT_PATH> \
  --benchmark-fixture <BENCHMARK_FIXTURE> \
  --store-path <STORE_PATH> \
  --out-report <REPORT_DIR> \
  --tokenizer-model-id <TOKENIZER_MODEL_ID>
```

Expected outputs:

- Writes `<REPORT_DIR>/report.json` and `<REPORT_DIR>/report.md`.
- Prints model metrics as `top1`, `top3`, `mrr`, and `n`.
- Prints TF-IDF baseline metrics when `--tokenizer-model-id` is supplied.
- Skips benchmark cases without `primary_clause_ids`.

Key knobs:

- `--ckpt` selects the trained router checkpoint.
- `--benchmark-fixture` selects the benchmark file.
- `--store-path` loads the store used to resolve expected window ids and baseline routing.
- `--out-report` selects the output directory for the report pair.
- `--tokenizer-model-id` enables the TF-IDF baseline path and excerpt decoding.
- `--model-id` is only needed when the checkpoint was trained with `gemma-embed`.

## End-to-End

- Build a dataset from the target store.
- Train with the desired encoder, epochs, device, and hidden size.
- Evaluate the checkpoint against the benchmark fixture and compare the learned router to the TF-IDF baseline.
