# Coding Eval

This eval tests whether Lazarus interactive memory improves real coding-agent
work on hidden-test fixtures.

## Architecture

Use `--agent pi-memory` for the real eval.

That path is intentionally:

1. `evaluate_memory_coding_task.py` creates hidden-test coding fixtures and
   planted memory stores.
2. It starts `pi_interactive_memory_bridge.py` for each fixture/condition.
3. The bridge loads `scripts/interactive_memory_chat.py::MemoryChat`.
4. Pi talks to the bridge through a temporary OpenAI-compatible provider.
5. Pi still owns its normal coding tools: `read`, `write`, `edit`, `bash`.
6. Every model turn is produced by the same interactive chat architecture used
   by the REPL, including `memory_mode`, `memory_profile`, store routing, and
   memory telemetry.

Do not use plain `--agent pi` when the goal is to validate Lazarus memory. Plain
Pi uses Pi's normal provider stack and bypasses `MemoryChat`.

## Important Safety

`--agent pi-memory` loads the Gemma/Lazarus interactive model. Do not start it
while another GPU eval is running unless you explicitly want concurrent model
loads.

Static checks and the mock runner do not load Gemma.

## Static Checks

From the repo root:

```bash
uv run ruff check prod/evals/coding-eval/evaluate_memory_coding_task.py \
  prod/evals/coding-eval/pi_interactive_memory_bridge.py \
  scripts/evaluate_memory_coding_task.py

python3 -m py_compile \
  prod/evals/coding-eval/evaluate_memory_coding_task.py \
  prod/evals/coding-eval/pi_interactive_memory_bridge.py \
  scripts/evaluate_memory_coding_task.py
```

## Mock Baseline

This verifies fixture generation, visible/hidden tests, scoring, reports, and
telemetry without loading Gemma:

```bash
python3 prod/evals/coding-eval/evaluate_memory_coding_task.py \
  --conditions A,B,C,D \
  --require-telemetry
```

Expected pattern:

- A: empty store, memory off, visible tests pass, hidden project rules mostly miss.
- B: helpful store, memory auto, hidden rules improve.
- C: helpful plus noisy store, memory auto, hidden rules still improve without noise leakage.
- D: helpful store, memory off, matches A and proves store alone does nothing.

## Real Pi + MemoryChat Eval

Run this only when the GPU/model slot is free:

```bash
python3 prod/evals/coding-eval/evaluate_memory_coding_task.py \
  --agent pi-memory \
  --conditions A,B,C,D \
  --require-telemetry \
  --timeout-s 900 \
  --memory-bridge-start-timeout-s 1200
```

Useful overrides:

```bash
--memory-model-path google/gemma-4-E2B-it
--memory-device cuda
--memory-generation-engine standard
--memory-max-new-tokens 800
--pi-bin ~/.pi/agent/node_modules/.bin/pi
```

Each task directory contains:

- `solution.py`: Pi's edited solution.
- `visible_tests.py`: visible test file Pi may run.
- `hidden_tests.py`: hidden scoring file, not described in the prompt.
- `agent_prompt.md`: prompt sent to Pi.
- `memory_store/`: planted lessons for the condition.
- `pi_memory_bridge.log`: bridge startup/model/memory logs.
- `memory_eval_telemetry.jsonl`: per-turn MemoryChat telemetry.

The run report is written under `/tmp/lazarus-memory-coding-eval/run-*/report.json`
unless `--output-root` is supplied.

## Supervising The Run

1. Confirm no other Gemma/GPU eval is active.
2. Run the real command above.
3. Watch for `MEMORY_CODING_EVAL_PASS` or a specific `MEMORY_CODING_EVAL_FAIL`.
4. If a condition fails, inspect that condition's `task_dir` in the report.
5. Check `pi_memory_bridge.log` first for MemoryChat/model issues.
6. Check `memory_eval_telemetry.jsonl` for `memory_used`, `mode`,
   `kv_direct_active`, and `no_silent_fallback`.
7. Preserve the report path and relevant task directories in the handoff.

## Compatibility Entry Point

`scripts/evaluate_memory_coding_task.py` remains as a wrapper so older commands
still work, but the canonical eval lives in this directory.
