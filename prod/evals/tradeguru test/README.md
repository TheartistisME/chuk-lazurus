# TradeGuru Fault-Finding Memory Eval

This directory is the production home for the TradeGuru electrical fault-finding A/B eval.

The important bit: the memory store is permanent and local to this directory:

```text
prod/evals/tradeguru test/memory_store/tradeguru_fault_memory
```

Do not build this store in `/tmp`. Do not delete it during cleanup. The runner uses append mode by default, so an interrupted import can be resumed and complete document sessions are skipped.

Generated folders such as `logs/`, caches, virtual environments, and build output are excluded by default. This keeps large run logs like `logs/*/chat.json` out of the knowledge store and avoids long single-document `LiveIndexer` drain timeouts. Only opt into generated files intentionally with `--include-generated` on `scripts/import_tradeguru_memory.py`.

## What This Runs

The eval compares the same Gemma 4 runtime and same fault-finding questions across:

```text
memory_off_empty_store
memory_on_tradeguru_store
memory_on_tradeguru_plus_noise
memory_off_with_store_present
```

The memory-on path routes the question through the TradeGuru checkpoint store, assigns HOT/WARM/COLD windows, materializes selected residual streams into K/V, generates an answer, and records evidence telemetry. The scoring rubric rewards correct fault types, test order, safety controls, no unsafe live-work advice, document-specific details, clean telemetry, and relevant evidence windows.

## Agent Preflight

Run from WSL:

```bash
cd /mnt/c/Users/jehma/Desktop/lazarus/chuk-lazurus
bd prime
git status --short --branch
```

Confirm the TradeGuru sources exist:

```bash
ls "/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/electricalhowto"
ls "/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/electricalhowto - 2"
ls "/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/fault-finding-guide"
```

Run a non-model dry run:

```bash
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" dry-run
```

## Build The Permanent Store

```bash
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" import
```

Outputs:

```text
prod/evals/tradeguru test/memory_store/tradeguru_fault_memory/
prod/evals/tradeguru test/logs/import_<timestamp>.log
```

Expected supervision checks:

```bash
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" inspect
find "prod/evals/tradeguru test/memory_store/tradeguru_fault_memory" -path "*/torch_store/manifest.json" | wc -l
```

Every complete document session should contain:

```text
<session_id>/torch_store/manifest.json
<session_id>/torch_store/boundaries/
<session_id>/torch_store/residual_streams/
<session_id>/torch_store/window_token_lists.npz
<session_id>/torch_store/window_metadata.json
```

If the import is interrupted, rerun the same `import` command. Append mode is the default.

Only use a destructive rebuild when explicitly intended:

```bash
TRADEGURU_IMPORT_MODE=force bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" import
```

## Run The Eval

```bash
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" eval
```

Outputs:

```text
prod/evals/tradeguru test/results/eval_results_<timestamp>.jsonl
prod/evals/tradeguru test/results/eval_results_<timestamp>.jsonl.summary.json
prod/evals/tradeguru test/logs/eval_<timestamp>.log
```

To import and evaluate in one supervised run:

```bash
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" all
```

## Supervision Checklist

Watch for:

```text
memory_used=false for memory_off_empty_store
memory_used=false for memory_off_with_store_present
memory_used=true for memory_on_tradeguru_store
memory_used=true for memory_on_tradeguru_plus_noise
evidence_support_count > 0 in memory-on rows
no unsafe live-work advice failures
summary mean_score improves from OFF to ON
```

In another WSL terminal, GPU monitoring is useful during import/eval:

```bash
nvidia-smi -l 5
```

If `memory_on_tradeguru_plus_noise` warns that it reused the normal store, that condition is not a true noise test yet. To build a true noise store, import TradeGuru plus an unrelated source into a separate permanent root, then pass it to the eval:

```bash
TRADEGURU_STORE_ROOT="$PWD/prod/evals/tradeguru test/memory_store/tradeguru_fault_memory_plus_noise" \
TRADEGURU_EXTRA_SOURCE="/mnt/c/path/to/noise/docs" \
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" import

TRADEGURU_NOISE_STORE_ROOT="$PWD/prod/evals/tradeguru test/memory_store/tradeguru_fault_memory_plus_noise" \
bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" eval
```

## Useful Overrides

```bash
TRADEGURU_MODEL=/path/or/hf/model
TRADEGURU_DEVICE=cuda
TRADEGURU_MAX_NEW_TOKENS=240
TRADEGURU_HOT_BUDGET_MIB=512
TRADEGURU_CONDITIONS=memory_off,memory_on,memory_on_noise,off_store
TRADEGURU_STORE_ROOT="$PWD/prod/evals/tradeguru test/memory_store/tradeguru_fault_memory"
TRADEGURU_RESULTS_ROOT="$PWD/prod/evals/tradeguru test/results"
TRADEGURU_LOG_ROOT="$PWD/prod/evals/tradeguru test/logs"
```

## Handoff

Record the result summary path and any warnings in the relevant bead. Do not remove the permanent memory store after the run. If code changed, follow the repo closeout workflow: quality gates, `bd sync`, commit, pull/rebase, push, and verify the branch is up to date.
