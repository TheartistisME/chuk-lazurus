# Extending the Learned Router to a New Corpus

Use this checklist when wiring `tools/train_window_router.py` to a new `(store, benchmark fixture)` pair.

## Required inputs

- `TorchKnowledgeStore` directory with `manifest.json`, `window_tokens.npz`, `window_token_lists.npz`, `idf.json`, `keywords.json`, `window_metadata.json`, and `boundaries/window_###.npy`.
- `manifest.json` may point `window_metadata` at another filename; `TorchKnowledgeStore.load()` follows that path.
- `window_metadata.json` may be either a dict keyed by window id or a list of objects with `window_id`.
- Each window record should carry `clause_id` and `clause_title`; `part_index` is used when a clause spans multiple windows, and `part_index=1` is treated as primary.
- Use `tests/fixtures/aus3000/benchmark/epic1_v1.json` as the concrete fixture-shape reference.
- Each benchmark case needs `name`, `prompt`, and `primary_clause_ids`; optional fields such as `category`, `support_clause_ids`, `required_answer_anchors`, `required_literal_groups`, `expect_insufficient`, and `expect_no_electrical_bleed` are preserved by the evaluator.

## Checklist

1. Build the store and its sidecar. Finish the corpus build first; the router CLI only consumes a completed store directory. Verify the store opens with `TorchKnowledgeStore.load(<STORE>)` and that `window_metadata` contains the expected clause ids and titles.
2. Build the dataset and its sidecar. Run `uv run python tools/train_window_router.py build-dataset --store-path <STORE> --out-jsonl <OUT>.jsonl`. The command writes `<OUT>.jsonl` plus `<OUT>.jsonl.meta.json`, and samples come from `window_metadata.json` plus optional excerpts decoded from `window_token_lists.npz`.
3. Train the router. Run `uv run python tools/train_window_router.py train --dataset <OUT>.jsonl --encoder bow --out-ckpt <CKPT>.pt`. Use `bow` first; `gemma-embed` is optional and only works when the model id is available locally. The checkpoint stores `state_dict`, `meta`, and encoder vocabulary for bow runs.
4. Evaluate against the fixture. Run `uv run python tools/train_window_router.py eval --ckpt <CKPT>.pt --benchmark-fixture tests/fixtures/<corpus>/benchmark/<epoch>.json --store-path <STORE> --out-report docs/learned_router/eval/`. The evaluator resolves the expected window from `primary_clause_ids[0]`, then reports model top-1, top-3, and MRR alongside the TF-IDF baseline.
5. Interpret the results. Read `report.json` for machine consumption and `report.md` for the summary. Compare learned-router top-1 first, then top-3 and MRR. If the baseline shows `—`, the baseline tokenizer was not supplied, so the comparison is incomplete. Investigate skipped cases before drawing a corpus-level conclusion.
6. Keep the corpus generic. Do not add corpus-specific branches to the CLI. New corpus adoption should only change `--store-path`, `--benchmark-fixture`, and output paths.

