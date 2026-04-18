# Complex Task — Torch Parity + Learned Window Router

<Metadata>
- Project: chuk-lazarus
- Branch base: `main` at commit `30e5f04` (AUS3000 single_pass_gate at 23/23, soak_gate at 579/579)
- Date: 2026-04-17
- Author: orchestrator (this session) → handed to claude-opus-4.7 Lead
- Environment: WSL2 Linux, Python 3.13.11, managed via `uv`
- Runtime deps: `torch==2.10.0+cu128` (CUDA 12.8 available), `numpy`, `transformers` (Gemma tokenizer), `pytest==9`, `pytest-asyncio`
- Target hardware: CUDA GPU primary (RTX-class, bf16 capable), CPU fallback, Apple Silicon MLX path preserved
- MLX present in codebase: yes, must stay untouched
- Orchestration: `vee` CLI (v0.1.0) — `vee agent spawn claude …` only from orchestrator
- Team coordination in Lead pane: `Team:*` skills (claude teams), mandatory
</Metadata>

# Task

Extend the model/training stack so the existing MLX-bound tinymlp classifier family can **also** train and infer on CUDA/PyTorch, following the repo's established "torch-native parity file" pattern (the same pattern used by `torch_store.py` / `torch_query.py` / `torch_runtime.py`). Then train an MLP **window router** on the AUS3000 clause-aligned store as the first concrete use case, measuring its routing accuracy against the current TF-IDF baseline on the 23-case benchmark.

**Broader goal (non-negotiable):** the new torch parity stack and the window-router plumbing must be **generic over any Markov/Lazarus store + benchmark fixture pair**. AUS3000 is the first caller, not a hard-coded special case.

## Success criteria

- All existing MLX tests still pass (`tests/models_v2/**`, `tests/training/**`).
- All 25 AUS3000 route/store/query/evaluator tests still pass; `single_pass_gate` stays at `23/23 PASS` on the same command.
- New torch-native modules have tests; all pass on CPU; pass on CUDA when available.
- A window-router model trains end-to-end using only the new torch stack (no MLX at runtime on Linux).
- Trained router produces top-1 and top-3 accuracy on the 23-case AUS3000 benchmark, reported alongside TF-IDF baseline.
- Dataset builder, trainer, and eval are parameterized on `--store-path` and `--benchmark-fixture`; AUS3000 is one invocation.

## Constraints (must not affect)

- **Zero edits** to `mlx_backend.py`, `models_v2/models/classifiers/*.py` (MLX), `training/classification_trainer.py`, or any other file that `import mlx`.
- **Zero edits** to `route.py`, `torch_store.py`, `torch_query.py`, `torch_runtime.py`. The AUS3000 exact-routing fix is frozen.
- The reference card `docs/aus3000_accuracy_program/04-reference-card.md` stays showing `23/23`.
- New torch classifier files must live **alongside** MLX files (`torch_mlp.py` next to `mlp.py`), never replace them.
- Final doc set ≤ 150 lines per file, flat structure under `docs/learned_router/`.

# Project Deliverables

- **D1 — Torch backend completion.** Finish `models_v2/core/backend/torch_backend.py` to implement every abstract method in `base.py`; register in `registry.py`; add unit tests.
- **D2 — Torch classifier parity modules.** `torch_linear.py`, `torch_mlp.py`, `torch_token_embedding.py` next to their MLX twins; identical constructor signatures and output shapes; each module self-contained, < 120 lines.
- **D3 — Torch training stack.** `torch_base_trainer.py` and `torch_classification_trainer.py` under `src/chuk_lazarus/training/torch/`; cross-entropy, Adam/AdamW, gradient clipping, checkpoint save/load via `torch.save`; same high-level contract as the MLX `ClassificationTrainer`.
- **D4 — Generic window-router tool.** `tools/train_window_router.py` with subcommands `build-dataset`, `train`, `eval`. Input: `--store-path`, `--benchmark-fixture`, `--model-id` (Gemma or similar). Output: trained checkpoint + eval report JSON.
- **D5 — Tests.** For every new module a matching `tests/**/test_*.py`. All tests use the real torch CPU path (no mocks for tensor ops); CUDA tests guarded with `@pytest.mark.skipif(not torch.cuda.is_available())`.
- **D6 — AUS3000 validation report.** Run `train_window_router` on the clause-aligned store, produce `docs/learned_router/eval/aus3000_eval.md` with top-1 / top-3 accuracy vs TF-IDF baseline. Commit only if top-1 is reported; verdict (beats / ties / loses to baseline) is informational — the infrastructure lands either way.

## Workflow

- Run `vee --help` to confirm CLI surface, then `vee init` (if workspace not already initialized) and `vee mission onboard` to set up tracking.
- Understand the task: the *why* (creator flagged routing as the open problem; we want learned routing), the *how* (torch-parity pattern mirroring `torch_store.py`), the *where* (`models_v2/` classifiers + `training/` + new `tools/train_window_router.py`).
- Skip the pre-build documentation round. Do **epics in reverse**: build first (Epic 1), document after (Epic 2).
- Run through every phase until success criteria are met. The app must train a router end-to-end on AUS3000 from a single `uv run python tools/train_window_router.py ...` invocation when done.

# Non-negotiables

- Entire process completed end to end.
- All code dependencies installed; `uv sync` + `uv run pytest` must pass out of the box with no manual steps.
- `chuk_lazarus` usable for window-router training out of the box after changes.
- Industry standard code practices: type hints, module docstrings, no emoji in source, `ruff`-clean.
- All code inserted MUST have its equivalent test(s).
- All tests passing smoke / dry-run / real.
- MLX path must remain 100% intact — the MLX-only tests stay green.

# Epics

## Epic-1 — Build & Train (we start here; reverse order)

### E1 Phase 1 — Orchestrator spawns Lead

*Orchestrator action:*
```
vee agent spawn claude --pane lead-b --prompt "<onboarding prompt pointing at this spec>"
```
- Lead is Opus 4.7 via `claude --dangerously-skip-permissions`.
- Orchestrator then enters sleep/monitor cadence (`vee agent check-in --pane lead-b`).

*Lead onboarding (delivered in the spawn prompt):*
1. `vee session open --pane lead-b --role lead --mission router-backend`
2. `vee mission create "Torch parity + learned router" --type epic --priority 1` → capture mission id
3. Read this spec file (`docs/learned_router/00-task-spec.md`) in full.
4. Read the "reference pattern" files the orchestrator has flagged (see **Reference Pattern** below).
5. Ask no clarification questions — proceed with best judgement; orchestrator will not answer Lead questions.

### E1 Phase 2 — Lead delegates via claude teams

Lead's orchestration rules (see **Lead Rules** below) require it to use the `Team:*` skill family — no direct code writing.

Lead workflow:
1. `/Team:plan-reviewed` — produce a minimal spec per workstream (≤ 80 lines each), saved under `docs/learned_router/specs/`.
2. `/Team:do-teams` — hand out workstreams in the batch order below.
3. Each teammate runs `/Team:review-loop` (sonnet reviewers, waves of 5) before its PR is considered green.
4. When all workstreams report PASS, Lead runs the final integration step: the AUS3000 training + eval end-to-end, writes `docs/learned_router/eval/aus3000_eval.md`, then `vee session close` and reports back to orchestrator.

### Workstreams (parallelisable)

| ID | Owner files (exclusive) | Depends on |
|----|-------------------------|------------|
| **WS-1 Torch backend** | `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` (complete), `registry.py` additions, `tests/models_v2/core/backend/test_torch_backend.py` | — |
| **WS-2 Torch classifiers** | `src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py`, `torch_mlp.py`, `torch_token_embedding.py`, matching tests under `tests/models_v2/models/classifiers/` | — (torch-native; does not need WS-1) |
| **WS-3 Torch training** | `src/chuk_lazarus/training/torch/__init__.py`, `torch_base_trainer.py`, `torch_classification_trainer.py`, matching tests under `tests/training/torch/` | WS-2 |
| **WS-4 Window router tool** | `tools/train_window_router.py`, `tools/_window_router/__init__.py`, `tools/_window_router/dataset.py`, `tools/_window_router/encoder.py`, `tools/_window_router/eval.py`, tests under `tests/tools/window_router/` | WS-3 |
| **WS-5 AUS3000 eval run** | `docs/learned_router/eval/aus3000_eval.md`, trained checkpoint artifact path | WS-4 |

No workstream shares ownership of any file. Lead MUST enforce this before dispatching `/Team:do-teams`.

### Reference Pattern (Lead must read these verbatim before spec)

- **MLX reference (do-not-touch twins):**
  - `src/chuk_lazarus/models_v2/models/classifiers/mlp.py`
  - `src/chuk_lazarus/models_v2/models/classifiers/linear.py`
  - `src/chuk_lazarus/training/classification_trainer.py`
  - `src/chuk_lazarus/training/base_trainer.py`
- **Torch-parity-style reference (same repo, same pattern):**
  - `src/chuk_lazarus/inference/backends/torch_runtime.py`
  - `src/chuk_lazarus/inference/context/knowledge/torch_store.py`
  - `src/chuk_lazarus/inference/context/knowledge/torch_query.py`
- **Already-started file (complete it):**
  - `src/chuk_lazarus/models_v2/core/backend/torch_backend.py`
- **Route fix (must keep green):**
  - `src/chuk_lazarus/inference/context/knowledge/route.py`
  - the 25 tests under `tests/inference/context/` and `tests/tools/`

### Checklist

**WS-1 — Torch backend**
- [ ] All abstract methods from `base.py` are implemented in `torch_backend.py`.
- [ ] `TorchBackend` honors `--device cuda|cpu|mps`; CUDA-unavailable falls back to CPU without crashing.
- [ ] `registry.py` accepts `"torch"` (and alias `"pytorch"`) and returns a working `TorchBackend`.
- [ ] Unit test for every tensor op method; `test_torch_backend.py` green on CPU.
- [ ] CUDA-guarded test confirms bf16 on SM 80+ and fp16 below.
- [ ] No file under `models_v2/core/backend/` lost any existing behaviour (MLX import still works).

**WS-2 — Torch classifiers**
- [ ] `torch_linear.py` exposes `TorchLinearClassifier(input_size, num_labels, bias)`; signature mirrors MLX `LinearClassifier`.
- [ ] `torch_mlp.py` exposes `TorchMLPClassifier(input_size, hidden_size, num_labels, activation, bias)`.
- [ ] `torch_token_embedding.py` exposes `TorchTokenEmbedding(vocab_size, hidden_size)` with weight tying support.
- [ ] Gradients flow (autograd tests).
- [ ] Shape parity with MLX twins for the same constructor args (tested on a fixed fixture).

**WS-3 — Torch training**
- [ ] `TorchBaseTrainer` / `TorchClassificationTrainer` implement `train(dataset, num_epochs)`.
- [ ] Optimizer: `AdamW`; LR sched: optional cosine; grad clip: optional `max_norm`.
- [ ] Checkpoint save/load uses `torch.save` and survives round-trip.
- [ ] `compute_loss` returns `(loss_tensor, metrics_dict)` with `loss` and `accuracy`; identical contract to MLX.
- [ ] Minimal trainer test trains a 2-class MLP on a synthetic linearly-separable dataset and reaches ≥ 99% acc.

**WS-4 — Window router tool**
- [ ] `tools/train_window_router.py build-dataset --store-path ... --out-jsonl ...` produces (text, window_id) JSONL from `window_metadata.json` + corpus (if present) + templated paraphrases ("Define {title}", "What is {title}?", "Clause {id}: {title}", clause-content excerpts).
- [ ] `tools/train_window_router.py train --dataset ... --encoder bow|gemma-embed --out-ckpt ...` trains `TorchMLPClassifier` to convergence on CPU or CUDA.
- [ ] `tools/train_window_router.py eval --ckpt ... --benchmark-fixture ... --store-path ...` reports top-1 / top-3 / MRR per case, with TF-IDF baseline alongside, in JSON + markdown.
- [ ] No AUS3000-specific logic in code — only in CLI args.
- [ ] Smoke test under `tests/tools/window_router/test_pipeline_smoke.py` runs build→train(2 epochs)→eval on a tiny synthetic 4-window store without downloading weights.

**WS-5 — AUS3000 eval**
- [ ] Build dataset from real AUS3000 store (clause-aligned variant).
- [ ] Train one checkpoint, ≤ 20 epochs, CPU or CUDA.
- [ ] Run eval on `tests/fixtures/aus3000/benchmark/epic1_v1.json`.
- [ ] Commit a markdown report: numbers, verdict vs TF-IDF, recommendation.

### Mandatory Batch Protocol inside Epic-1

- **Batch 1 (parallel):** WS-1, WS-2.
- **Batch 2 (after both Batch-1 PASS):** WS-3.
- **Batch 3 (after WS-3 PASS):** WS-4.
- **Batch 4 (after WS-4 PASS):** WS-5.
- Quality gate after every batch: `uv run pytest -q` must pass AND `uv run pytest tests/inference/context/ tests/tools/test_evaluate_aus3000_variant.py -q` must pass AND the full AUS3000 `single_pass_gate` smoke (`--max-cases 5 --device cpu` or equivalent fast path) must not regress on those 5 cases.

## Epic-2 — Document (only after Epic-1 is green)

### E2 Phase 1 — Orchestrator spawns docs Lead

*Orchestrator action:*
```
vee agent spawn claude --pane lead-docs --prompt "<doc spec pointing at completed code>"
```

Docs Lead orchestration rules identical to Epic-1: delegate only, `Team:*` skills, review loops required.

### Documents to produce

All under `docs/learned_router/`, all ≤ 150 lines each:

- `10-architecture.md` — torch parity pattern, registry, files-touched map.
- `20-training-guide.md` — how to build a dataset from any store, how to train, how to eval.
- `30-aus3000-results.md` — consolidated results + the TF-IDF comparison story.
- `40-extending-to-new-stores.md` — checklist for wiring the learned router to a new corpus.
- `50-reference-card.md` — final status card, mirroring `docs/aus3000_accuracy_program/04-reference-card.md` style.

### Orchestration Protocols

- Claude code lead does NOT write code or docs itself. Lead delegates per-file work to teammates via the `Team:*` skill.
- Lead must dispatch the correct workstreams to keep ownership exclusive.
- Every teammate must run quality gates after every change (`uv run pytest <their test file>`, `ruff check <their files>`), otherwise the change is rejected.
- Lead completes all stages start-to-finish before reporting back. No mid-stream questions to orchestrator.

### Lead Rules

- **LR1 — No code, only delegation.** Lead cannot write source files. Lead writes/edits specs, workstream assignments, and quality-gate verdicts only.
- **LR2 — Claude teams mandatory.** Lead must use `/Team:plan-reviewed`, `/Team:do-teams`, `/Team:review-loop` at the phases above. Ad-hoc agent invocations are forbidden.
- **LR3 — No cross-workstream file ownership.** If two workstreams share a file, Lead must split it first.
- **LR4 — Preserve MLX path.** Before closing any workstream, Lead verifies `uv run pytest tests/ -q` includes zero MLX regressions. Any MLX-touching diff is a hard reject.
- **LR5 — No orchestrator pings.** Lead does not DM the orchestrator with questions. If truly stuck, Lead records the blocker via `vee record --type blocker` and waits for an orchestrator check-in.
- **LR6 — Record learnings.** Every completed workstream must `vee record --type insight` with what worked and what didn't, under mission id.

### Sub-Agent Rules

- **SR1 — Own only assigned files.** No editing outside the workstream manifest.
- **SR2 — Test with every change.** Every diff ships with its test; `uv run pytest <test>` green before pushing status.
- **SR3 — Torch-native, not MLX-converted.** Never copy-then-edit an MLX file; write torch-native from scratch following the `torch_store.py` pattern.
- **SR4 — Ask Lead, not Orchestrator.** Clarifications go up one level only.
- **SR5 — No scope creep.** Deliver exactly what the workstream spec lists. Extras go into a follow-up mission item.
- **SR6 — Use vee.** `vee session open` at start; `vee record` at finish; `vee session close --handoff` when handing review to the review-loop reviewer.

## Batch Protocols

- **BP1 — Quality-gate after every batch.** `uv run pytest -q` must be green on `main` HEAD after every batch merge.
- **BP2 — AUS3000 smoke after every batch.** Run a 5-case `single_pass_gate` subset on CPU; any regression blocks the next batch.
- **BP3 — Memory discipline.** Every teammate records a `vee record --type decision` explaining non-obvious design choices. Every Lead ticks `vee mission update --status in_progress/completed` as phases turn.

## Workflow (repeat at end per template)

- Run `vee --help` to confirm the CLI surface.
- Understand task: why (learned routing is the open problem), how (torch-parity files), where (`models_v2/`, `training/torch/`, `tools/train_window_router.py`).
- Begin Epic-1 (build first, per agreed reverse order).
- Continue through every phase and epic until success criteria are met. End state: a single `uv run python tools/train_window_router.py ...` command trains a router from any store + fixture pair, and the AUS3000 eval report is committed.

# Non-negotiables (restated)

- Entire process completed end to end.
- All code dependencies installed; `uv sync` + `uv run pytest` passes out of the box.
- chuk_lazarus trains routers out of the box after changes, for AUS3000 *and* any other store/fixture pair.
- Industry standard code practices.
- All code inserted MUST have its equivalent test(s).
- All tests passing smoke / dry-run / real.
- MLX path preserved verbatim.

# Critical

AUS3000 is the first use case but it is **not** the point. The point is: wire up chuk-lazarus so that for any (corpus → store → benchmark fixture) triple, we can train a small MLP to route queries to the right Markov window and read the stored residual + window tokens for a single-forward-pass answer. This is the missing leg of the pseudo-infinite memory stack (store + Markov state + *learned* routing). Build it generic, prove it on AUS3000, ship it.
