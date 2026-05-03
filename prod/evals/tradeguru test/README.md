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

The eval defaults to the production memory selector:

```text
selector_policy=utility-v2
dense_scoring=deterministic
selector_budget=K_HOT + K_WARM
```

Use `TRADEGURU_SELECTOR_POLICY=rank-v1` only when intentionally comparing against the frozen legacy router.

### Catalog/TOC-index window penalty

The `run_tradeguru_fault_eval.sh` shell script defaults `LAZARUS_ROUTER_TOC_INDEX_PENALTY=5.0` (override via `TRADEGURU_TOC_INDEX_PENALTY`). This demotes catalog-style "Example Queries" windows that begin with phrases like `## Example Queries This video answers questions like:`. They TF-IDF-match keyword-rich fault questions but inject low-value index content into KV memory and drag memory_on answers below memory_off (eval `eval_results_20260428T092804Z.jsonl` showed all 4 top-rank hot windows for the kettle question were such TOC pages). Set `TRADEGURU_TOC_INDEX_PENALTY=0` to disable the penalty for an A/B comparison. The penalty applies before tier assignment so HOT slots go to procedural windows ("Step-by-Step", "Key Lessons", "Safety Notes", "Pro Tips") instead.

### Caller-supplied system prompt on the semantic-prefix decode path

The KV-direct memory-on path includes a "semantic prefix" decoder
(`SessionRetriever._generate_with_semantic_token_prefix` in
`src/chuk_lazarus/session_retrieval/retriever.py`) used when
`_synthesize_memory_laws_answer` / `_synthesize_website_color_scheme_answer` /
`_synthesize_dirty_store_domain_answer` all return empty — which is the
default for fault-finding questions. Historically the decoder hardcoded a
"You answer from the context that appears before this chat… be concise."
system prompt authored for chat_loop value-extraction tests. For procedural
electrical fault-finding answers this destroys the ordered test sequence
and skips the safety preamble: grade
`tradeguru_grade_20260428T115132Z.json` showed `correct_test_order` 1/8 and
`safety_controls_present` 2/8 on memory-on (vs 6/8 and 8/8 on memory-off
for the same eight questions, with answer lengths of 65–354 chars vs
2500+ chars).

`run_tradeguru_fault_eval.sh` defaults
`LAZARUS_KV_SEMANTIC_PREFIX_USE_CALLER_SYSTEM_PROMPT=1`, which routes the
decoder through the same SYSTEM_PROMPT that
`scripts/evaluate_tradeguru_fault_memory.py` configures on the retriever
(safety-first electrical context, isolation/lockout/PPE, ordered
fault-finding sequence, licensed-electrician escalation). The chat_loop
value-extraction default ("be concise") is preserved when the variable is
unset or `0`. Set `LAZARUS_KV_SEMANTIC_PREFIX_USE_CALLER_SYSTEM_PROMPT=0`
to A/B against the legacy decoder behaviour.

### Document-grounding directive on the semantic-prefix decode path

Iteration-2's caller-prompt fix restored safety language on memory-on
but grade `tradeguru_grade_20260428T124342Z.json` still showed
`memory_on_mean=7.125` vs `memory_off_mean=7.375` (lift=`-0.25`). The
remaining failures were `correct_test_order` 5/8 and
`uses_document_specific_details` 6/8 — every memory-on answer truncated
mid-sentence in the safety preamble (4 enumerated bullets on
PPE / LOTO / isolate / prove dead) and never reached the diagnostic
content within `max_new_tokens=240`.

`run_tradeguru_fault_eval.sh` defaults
`LAZARUS_KV_SEMANTIC_PREFIX_GROUND_IN_DOCUMENT=1`, which prepends a
small RAG-best-practice directive to the caller-supplied system prompt
on the prefix-decode path. The directive asks the model to ground its
answer in the document terminology already provided as KV-direct
context (component names, fault types, test instruments) and to keep
any preamble brief (1-2 sentences). The directive is task-agnostic
(no rubric vocabulary), preserves the caller's safety semantics
verbatim, and is silent unless both
`LAZARUS_KV_SEMANTIC_PREFIX_USE_CALLER_SYSTEM_PROMPT=1` *and*
`LAZARUS_KV_SEMANTIC_PREFIX_GROUND_IN_DOCUMENT=1`. Set
`LAZARUS_KV_SEMANTIC_PREFIX_GROUND_IN_DOCUMENT=0` to A/B against the
iteration-2 caller-verbatim behaviour.

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
TRADEGURU_SELECTOR_POLICY=utility-v2
TRADEGURU_DENSE_SCORING=deterministic
TRADEGURU_RRF_K=60
TRADEGURU_MMR_LAMBDA=0.75
TRADEGURU_SELECTOR_BUDGET=12
TRADEGURU_CONDITIONS=memory_off,memory_on,memory_on_noise,off_store
TRADEGURU_STORE_ROOT="$PWD/prod/evals/tradeguru test/memory_store/tradeguru_fault_memory"
TRADEGURU_RESULTS_ROOT="$PWD/prod/evals/tradeguru test/results"
TRADEGURU_LOG_ROOT="$PWD/prod/evals/tradeguru test/logs"
```

## Memory-Off Comparator Is Required For Grade Pass

`scripts/tradeguru_meta.py grade` treats a missing memory-off comparator as a **hard failure** by default. Any benefit-lift claim about the TradeGuru router needs both memory-on and memory-off rows in the eval JSONL — without a memory-off baseline, `memory_lift_vs_off` cannot be measured and the grade is not actionable evidence for router changes.

In practice that means an eval run for a meta-loop grade must include at least one of:

```text
memory_off_empty_store
memory_off_with_store_present
```

alongside the memory-on conditions. Set `TRADEGURU_CONDITIONS` accordingly when invoking `run_tradeguru_fault_eval.sh`.

For smoke runs that intentionally skip the comparator, pass `--allow-missing-memory-off`:

```bash
python3 scripts/tradeguru_meta.py grade <eval.jsonl> --allow-missing-memory-off
```

That demotes `missing_memory_off_comparator` from FAIL back to WARN. Smoke grades produced this way are not evidence of router improvement and must not be used to justify router or selector edits.

## Handoff

Record the result summary path and any warnings in the relevant bead. Do not remove the permanent memory store after the run. If code changed, follow the repo closeout workflow: quality gates, `bd sync`, commit, pull/rebase, push, and verify the branch is up to date.
