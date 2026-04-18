# Plan: Torch Parity + Learned Window Router — Master Orchestration

## Task Description
Build torch-native parity alongside the MLX classifier + training stack in `chuk_lazarus`, then
train an MLP **window router** on the AUS3000 clause-aligned knowledge store as the first
concrete use case — generic over any `(store, benchmark_fixture)` pair. MLX path must stay 100%
intact; AUS3000 `single_pass_gate` must remain 23/23.

Five per-workstream specs already exist and are approved by the Lead:
- `docs/learned_router/specs/ws-1-torch-backend.md`
- `docs/learned_router/specs/ws-2-torch-classifiers.md`
- `docs/learned_router/specs/ws-3-torch-training.md`
- `docs/learned_router/specs/ws-4-window-router-tool.md`
- `docs/learned_router/specs/ws-5-aus3000-eval.md`

This master plan binds them together: dispatch order, dependencies, team members, quality gates.

## Objective
Ship a `uv run python tools/train_window_router.py build-dataset|train|eval` CLI that trains a
TorchMLPClassifier over any `TorchKnowledgeStore`, with AUS3000 as the proving ground. Deliver a
committed `docs/learned_router/eval/aus3000_eval.md` report with top-1/top-3/MRR numbers next to
the TF-IDF baseline, without touching any MLX file or any frozen torch-store/route file.

## Problem Statement
Current routing is TF-IDF or clause-exact only. There is no learned router, and the existing
torch classifier stack is incomplete: `torch_backend.py` is half-written, torch-native classifier
modules and a torch trainer do not exist. That blocks every "small MLP in front of the Markov
store" experiment. We also lack a generic CLI — AUS3000 evaluation must not be hard-coded.

## Solution Approach
Parallelise two foundation workstreams (WS-1 backend, WS-2 classifier primitives) because
neither depends on the other. Then gate-serialise: WS-3 trainer uses WS-2 classes; WS-4 CLI uses
WS-3 trainer; WS-5 runs WS-4 CLI on the real AUS3000 store. Finally Epic-2 produces docs only
after Epic-1 is green. Every workstream owns an exclusive file set — no two workstreams touch
the same file.

## Relevant Files
Existing files the teammates READ but never edit (frozen):
- `src/chuk_lazarus/models_v2/core/backend/base.py` — abstract backend interface
- `src/chuk_lazarus/models_v2/models/classifiers/{mlp.py,linear.py,sequence.py,token.py,factory.py}` — MLX twins
- `src/chuk_lazarus/training/{base_trainer.py,classification_trainer.py}` — MLX trainer
- `src/chuk_lazarus/inference/backends/torch_runtime.py` — torch-parity reference pattern
- `src/chuk_lazarus/inference/context/knowledge/{torch_store.py,torch_query.py,route.py}` — route fix (frozen)
- `tests/fixtures/aus3000/benchmark/epic1_v1.json` — AUS3000 benchmark fixture
- `tools/evaluate_aus3000_variant.py` — existing AUS3000 evaluator (DO NOT EDIT)

### New Files (created by workstreams)
See each workstream spec for the authoritative file list. Summary:
- WS-1: expand `torch_backend.py` + `registry.py`; expand existing torch backend/registry tests
- WS-2: `torch_linear.py`, `torch_mlp.py`, `torch_token_embedding.py` + tests
- WS-3: `src/chuk_lazarus/training/torch/{__init__.py,torch_base_trainer.py,torch_classification_trainer.py}` + tests
- WS-4: `tools/train_window_router.py`, `tools/_window_router/{__init__.py,dataset.py,encoder.py,eval.py}` + tests
- WS-5: `docs/learned_router/eval/aus3000_eval.md`, `aus3000_report.json`

## Implementation Phases
### Phase 1: Foundation (parallel)
WS-1 and WS-2. No shared files. Complete before any Phase 2 start.
### Phase 2: Torch training
WS-3. Consumes WS-2 classifier classes.
### Phase 3: Integration & Polish
WS-4 (generic CLI), then WS-5 (AUS3000 eval), then Epic-2 documentation.

## Team Orchestration

The Lead (this agent) holds the mission id `chuk-lazurus-n7k`, never writes code, and enforces
exclusive file ownership. Teammates spawn via `/Team:do-teams` + `/Team:team-implement-spec`, and
each closes its workstream with `/Team:review-loop` before Lead accepts the PR.

### Team Members
- Builder
  - Name: `ws-1-builder`
  - Role: WS-1 Torch backend completion + registry alias + tests
  - Agent Type: builder
  - Mode: builder
- Builder
  - Name: `ws-2-builder`
  - Role: WS-2 torch classifiers (linear, mlp, token embedding) + tests
  - Agent Type: builder
  - Mode: builder
- Builder
  - Name: `ws-3-builder`
  - Role: WS-3 torch training stack + tests
  - Agent Type: builder
  - Mode: builder
- Builder
  - Name: `ws-4-builder`
  - Role: WS-4 generic `tools/train_window_router.py` CLI + tests
  - Agent Type: builder
  - Mode: builder
- Builder
  - Name: `ws-5-runner`
  - Role: WS-5 execute AUS3000 eval end-to-end and write the markdown report
  - Agent Type: builder
  - Mode: builder
- Reviewer
  - Name: `sonnet-reviewer-wave`
  - Role: parallel independent reviewers (wave of 5 via `/Team:review-loop`)
  - Agent Type: reviewer
  - Mode: lead

## Step by Step Tasks

### 1. ws-1-torch-backend
- **Task ID**: ws-1-torch-backend
- **Depends On**: none
- **Assigned To**: ws-1-builder
- **Role**: builder
- **Mode**: builder
- **Parallel**: true
- Read `docs/learned_router/specs/ws-1-torch-backend.md` and implement exactly as specified
- Run the scoped pytest and `ruff` gate listed in the spec; both must be green
- `/Team:review-loop` before PR is considered green
- `vee record insight --title "WS-1 torch backend" --tag chuk-lazurus-n7k --tag ws-1`

### 2. ws-2-torch-classifiers
- **Task ID**: ws-2-torch-classifiers
- **Depends On**: none
- **Assigned To**: ws-2-builder
- **Role**: builder
- **Mode**: builder
- **Parallel**: true
- Read `docs/learned_router/specs/ws-2-torch-classifiers.md` and implement exactly as specified
- Run the scoped pytest and `ruff` gate in the spec; both green
- `/Team:review-loop` before PR is considered green
- `vee record insight --title "WS-2 torch classifiers" --tag chuk-lazurus-n7k --tag ws-2`

### 3. ws-3-torch-training
- **Task ID**: ws-3-torch-training
- **Depends On**: ws-1-torch-backend, ws-2-torch-classifiers
- **Assigned To**: ws-3-builder
- **Role**: builder
- **Mode**: builder
- **Parallel**: false
- Read `docs/learned_router/specs/ws-3-torch-training.md` and implement exactly as specified
- Trains `TorchMLPClassifier` on synthetic 2-class blobs to ≥ 99% accuracy in smoke test
- `/Team:review-loop` before PR is considered green
- `vee record insight --title "WS-3 torch training" --tag chuk-lazurus-n7k --tag ws-3`

### 4. ws-4-window-router-tool
- **Task ID**: ws-4-window-router-tool
- **Depends On**: ws-3-torch-training
- **Assigned To**: ws-4-builder
- **Role**: builder
- **Mode**: builder
- **Parallel**: false
- Read `docs/learned_router/specs/ws-4-window-router-tool.md`
- Build the 3-subcommand CLI generic over `(store, benchmark)`
- Smoke test runs offline on a synthetic 4-window store fixture
- `/Team:review-loop` before PR is green
- `vee record insight --title "WS-4 window router tool" --tag chuk-lazurus-n7k --tag ws-4`

### 5. ws-5-aus3000-eval
- **Task ID**: ws-5-aus3000-eval
- **Depends On**: ws-4-window-router-tool
- **Assigned To**: ws-5-runner
- **Role**: builder
- **Mode**: builder
- **Parallel**: false
- Run the 3 CLI commands from `docs/learned_router/specs/ws-5-aus3000-eval.md`
- Commit `docs/learned_router/eval/aus3000_eval.md` + `aus3000_report.json`
- `vee record completion --title "WS-5 AUS3000 eval" --tag chuk-lazurus-n7k --tag ws-5`

### 6. validate-all
- **Task ID**: validate-all
- **Depends On**: ws-1-torch-backend, ws-2-torch-classifiers, ws-3-torch-training, ws-4-window-router-tool, ws-5-aus3000-eval
- **Assigned To**: Lead
- **Role**: reviewer
- **Mode**: lead
- **Parallel**: false
- Run every command in the Validation Commands section below
- Confirm AUS3000 `single_pass_gate` still 23/23 and no new pytest collection errors
- `vee record completion` with final summary; `vee session close`

## Acceptance Criteria
- All 5 workstream specs implemented per their exclusive-file lists.
- New torch tests (backend, classifiers, trainer, window_router) are fully green on CPU.
- CUDA-guarded tests skip cleanly on CUDA-less hosts, pass on CUDA hosts.
- MLX-only tests that were green in baseline are still green (zero new regressions).
- AUS3000 `single_pass_gate` stays at 23/23; `docs/aus3000_accuracy_program/04-reference-card.md` unchanged.
- `uv run python tools/train_window_router.py ... eval ...` produces `aus3000_eval.md` with top-1/top-3/MRR numbers vs TF-IDF baseline.
- Zero edits to `mlx_backend.py`, `mlp.py` (MLX), `linear.py` (MLX), `classification_trainer.py`, `base_trainer.py`, `route.py`, `torch_store.py`, `torch_query.py`, `torch_runtime.py`.

## Validation Commands
```
# Per-workstream gates (run after each batch)
uv run pytest tests/models_v2/core/backend/test_torch_backend.py tests/models_v2/core/backend/test_registry.py -q
uv run pytest tests/models_v2/models/classifiers/test_torch_linear.py tests/models_v2/models/classifiers/test_torch_mlp.py tests/models_v2/models/classifiers/test_torch_token_embedding.py -q
uv run pytest tests/training/torch/ -q
uv run pytest tests/tools/window_router/ -q

# Integration gates (run after WS-3 and again at end)
uv run pytest tests/inference/context/ tests/tools/test_evaluate_aus3000_variant.py -q
uv run python tools/evaluate_aus3000_variant.py --mode single_pass_gate --device cpu --max-cases 5

# Lint
uv run ruff check src/chuk_lazarus/models_v2/core/backend/torch_backend.py \
                  src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py \
                  src/chuk_lazarus/models_v2/models/classifiers/torch_mlp.py \
                  src/chuk_lazarus/models_v2/models/classifiers/torch_token_embedding.py \
                  src/chuk_lazarus/training/torch/ \
                  tools/train_window_router.py tools/_window_router/

# End-to-end (WS-5)
uv run python tools/train_window_router.py build-dataset --store-path <STORE> --out-jsonl artifacts/router/aus3000_ds.jsonl
uv run python tools/train_window_router.py train --dataset artifacts/router/aus3000_ds.jsonl --encoder bow --out-ckpt artifacts/router/aus3000_bow.pt --epochs 20 --device cpu
uv run python tools/train_window_router.py eval --ckpt artifacts/router/aus3000_bow.pt --benchmark-fixture tests/fixtures/aus3000/benchmark/epic1_v1.json --store-path <STORE> --out-report docs/learned_router/eval/
```

## Notes
- Baseline: 7 pre-existing pytest collection errors on Linux from MLX imports (introspection,
  experts, cli/test_base_flags.py, models_v2/families/llama4/test_config.py) — do NOT treat as
  regressions. 29 pre-existing MLX-specific runtime failures in the backend/registry tests are
  also baseline. New torch tests must be fully green; MLX baseline must not grow.
- No new pip dependencies required; `torch` is already in dev-extras, `transformers` already
  present.
- Single_pass_gate at 23/23 is non-negotiable. Run the 5-case smoke after every batch.
- Lead only writes specs and quality-gate verdicts — NEVER source code.

APPROVED — dispatch Batch 1.
