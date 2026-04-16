description: "Filled vee-first orchestration brief for end-to-end Lazarus CUDA completion on RTX 5090"
argument-hint: "Complete Lazarus so it is fully usable on my RTX 5090 with the torch/CUDA backend"

# Vee Orchestrate

## Task Input

Task text: `Complete Lazarus so it is fully usable on my RTX 5090 with the torch/CUDA backend. Continue from the current Epic 2 state, use the existing dual-backend CUDA spec/workstream docs, finish every remaining blocker to end-to-end usability, and do not stop until all success criteria are achieved and the branch is pushed.`

Derived sections:
- `#Task` is the exact completion of Lazarus CUDA usability on the user's RTX 5090.
- `checklist items` are derived from the current repo/runtime state, existing Epic 2 docs, and open beads tasks.
- `success criteria` are derived from the user's requirement to start using Lazarus on the RTX 5090 now.
- `epic names` are derived from the remaining blocker chain and validation gates.
- `document requirements` are partially already satisfied by the existing `docs/refactor/dual-backend-cuda*` artifacts; Epic 1 below is a reconcile/finalize phase, not greenfield document authoring.
- `implementation constraints` are derived from `AGENTS.md`, the existing Epic 2 workstream boundaries, and the current dirty worktree.

Use the task text as the authoritative source for:
- `#Task`
- checklist items
- success criteria
- epic names
- document requirements
- implementation constraints

## Workflow

- Run `vee --help` to understand how we will use vee in our agentic lifecycle.
- Understand task, the why, the how and the where.
- Begin document reconciliation phase.
- Continue through each phase and each epic until we have completed our task in all success criteria.

## Non-negotiables

- Entire process completed end to end.
- All code dependencies installed so the user can run Lazarus out of the box.
- Ready to use out of the box after changes.
- Industry-standard code practices.
- All code inserted must have equivalent tests.
- All tests passing at smoke, dry-run, and real-exec levels where the repo/spec requires them.
- Work is not complete until `git pull --rebase`, `bd sync`, `git push`, and `git status` confirms up-to-date with origin.

# Task

Complete Lazarus so it is fully usable on the user's RTX 5090 with the torch/CUDA backend.

## Success Criteria

- `CHUK_BACKEND=torch PYTHONPATH=src python3 -c "import chuk_lazarus"` succeeds.
- `CHUK_BACKEND=torch PYTHONPATH=src python3 -c "import chuk_lazarus.inference.loader"` succeeds without importing `mlx` or `mlx_lm`.
- `CHUK_BACKEND=torch PYTHONPATH=src python3 -c "import chuk_lazarus.introspection"` succeeds without importing `mlx` or `mlx_lm`.
- `CHUK_BACKEND=torch PYTHONPATH=src python3 -c "import chuk_lazarus.cli"` succeeds without importing `mlx` or `mlx_lm`.
- `CHUK_BACKEND=torch PYTHONPATH=src python3 -c "import chuk_lazarus; chuk_lazarus.LlamaForCausalLM"` succeeds.
- `uv run python -m pytest tests/ci/test_no_top_level_mlx.py -x` passes.
- High-level `train {sft,dpo,grpo,ppo,dual_reward}` command paths run on torch/CUDA where supported, and any intentional unsupported mode hard-fails with precise scoped errors rather than MLX crashes.
- High-level `infer`, `serve`, `knowledge`, `context prefill`, `context generate`, and `introspect` command paths are import-clean and execute on torch/CUDA where Epic 2 scope requires real support.
- CUDA smoke passes on the RTX 5090 for the primary user-facing workflows the repo claims to support now.
- The remaining optional-dependency and compatibility gaps are either completed or filed explicitly as follow-up beads issues with acceptance criteria.
- All completed work is committed, pushed, and synced in beads.

## Current Verified Runtime State

These are the live torch import gates verified on the current branch/worktree:

- `import chuk_lazarus` -> `OK`
- `import chuk_lazarus.inference.loader` -> `FAIL` at `src/chuk_lazarus/inference/context/adapters/gemma_adapter.py:17`
- `import chuk_lazarus.introspection` -> `FAIL` at `src/chuk_lazarus/introspection/hooks.py:30`
- `import chuk_lazarus.cli` -> `FAIL` via `src/chuk_lazarus/cli/commands/_constants.py` -> `introspection`
- `resolve chuk_lazarus.LlamaForCausalLM` -> `FAIL` via eager `models_v2` subpackage surfaces

## Existing Source Of Truth Documents

Do not restart planning from scratch. Start from these existing artifacts and reconcile them with the live worktree:

- `docs/refactor/dual-backend-cuda-all-buckets/00-README.md`
- `docs/refactor/dual-backend-cuda-all-buckets/01-command-matrix.md`
- `docs/refactor/dual-backend-cuda-all-buckets/02-implementation-spec.md`
- `docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md`
- `docs/refactor/dual-backend-cuda-all-buckets/04-validation-matrix.md`
- `docs/refactor/dual-backend-cuda-all-buckets/05-epic2-progress.md`
- `docs/refactor/dual-backend-cuda-epic2/00-scope.md`
- `docs/refactor/dual-backend-cuda-epic3/00-scope.md`

## Active / Open Beads Work

- `chuk-lazurus-40v` — lazy-init CLI/introspection constants chain for train commands on torch
- `chuk-lazurus-818` — make `models_v2` loader and LoRA stack torch-safe for high-level `trainer.run`
- `chuk-lazurus-x4i.1` — lazy-init `inference/context` package exports for torch runtime
- `chuk-lazurus-x4i.3` — stale description; root import is now fixed, but symbol-level package/runtime cleanup still remains for `models_v2`
- `chuk-lazurus-x4i.4` — reconcile legacy direct `chuk_virtual_expert` imports outside `virtual_experts`

## Epics

- `Epic-1` Reconcile and finalize the execution plan against the live worktree and open issues
- `Epic-2` Clear the remaining torch runtime/import gates
- `Epic-3` Finish high-level CUDA usability for training/model-loading paths
- `Epic-4` Finish end-to-end CUDA usability for inference, serve, knowledge, context, and introspection buckets
- `Epic-5` Run the full RTX 5090 validation matrix, close follow-ups, and land the branch

### Epic-1 Documents

Create or update the following documents/checklists to complete this checklist:

```md
- Reconcile `05-epic2-progress.md` with the current git history and worktree.
- Identify every already-landed workstream vs still-local-only work.
- Produce an exact blocker matrix from live import probes and open beads tasks.
- Segment the remaining code changes into non-conflicting workstreams.
- Define the final validation gates for RTX 5090 usability.
```

### Documents

- Update existing spec/workstream docs rather than replacing them.
- Produce any additional short-lived coordinator notes only if they reduce merge risk.
- Keep all new workstream planning anchored to exact file paths, tests, and commands.

### Document Preparation Workflow

*Pre-Phase* [Necessary vee commands used]
- Run `vee --help`.
- Open a vee session for this workspace.
- Prime and doctor the vee workspace.
- Review `bd ready`, open issues, and the Epic 2 progress log.

*Phase 1* [Necessary vee commands used]
- Spawn a Claude lead using vee.
- Claude lead must not author code first. It must synthesize the exact remaining file paths, blockers, acceptance criteria, and workstream ownership boundaries from the existing docs and live repo state.
- Claude lead may delegate document reconciliation/research to Claude teammates if needed, but should reuse the existing Epic 2 docs rather than requesting fresh greenfield specs.
- Once the live blocker matrix and remaining workstreams are reconciled, Claude lead reports the plan and begins Epic 2 execution orchestration.

*Phase 2* [Necessary vee commands used]
- Claude lead spawns one teammate per disjoint workstream/write scope where parallelism is safe.
- Every worker must run tests or equivalent quality gates after each change.
- Claude lead must continuously rebalance ownership so that no two agents touch the same write set.
- Once the remaining docs/checklists are green, move to code execution epics below.

- `Epic-2` Remaining torch runtime/import gates

*E2 Phase1* [Necessary vee commands used]
Spawn or coordinate Claude workers for these non-conflicting blocker groups:

- `E2-A` `inference/context` package init and adapters
  - `src/chuk_lazarus/inference/context/adapters/__init__.py`
  - `src/chuk_lazarus/inference/context/adapters/gemma_adapter.py`
  - `src/chuk_lazarus/inference/context/adapters/llama_adapter.py`
  - any directly-related `inference/context/__init__.py` lazy-export path
  - acceptance: `import chuk_lazarus.inference.loader` works under torch

- `E2-B` introspection root and CLI constants chain
  - `src/chuk_lazarus/introspection/hooks.py`
  - `src/chuk_lazarus/introspection/__init__.py`
  - `src/chuk_lazarus/cli/commands/_constants.py`
  - `src/chuk_lazarus/cli/__init__.py`
  - parser/command package init files only as needed
  - acceptance: `import chuk_lazarus.introspection` and `import chuk_lazarus.cli` work under torch

- `E2-C` remaining `models_v2` lazy surfaces for concrete symbol access
  - `src/chuk_lazarus/models_v2/families/__init__.py`
  - `src/chuk_lazarus/models_v2/models/__init__.py`
  - `src/chuk_lazarus/models_v2/blocks/__init__.py`
  - `src/chuk_lazarus/models_v2/losses/__init__.py`
  - downstream package leaves only as required by symbol access
  - acceptance: `chuk_lazarus.LlamaForCausalLM`, `Model`, `Block`, and `compute_lm_loss` resolve under torch

- `Epic-3` High-level training/model-loading usability

*E3 Phase1* [Necessary vee commands used]
- Reconcile the newly-landed trainer work (`c78ca06`) with the remaining high-level blockers.
- Finish `models_v2` loader + LoRA + high-level trainer run path on torch/CUDA.
- Validate high-level `train sft`, `train dpo`, `train grpo`, `train ppo`, and `train dual_reward`.
- Any intentionally unsupported mode must fail with a precise scoped error, not an MLX import crash.

- `Epic-4` End-to-end user-facing bucket completion

*E4 Phase1* [Necessary vee commands used]
- Finish real CUDA usability, not just parser/config plumbing, for:
  - `infer`
  - `serve` / `lazarus-serve`
  - `knowledge`
  - `context prefill`
  - `context generate`
  - `introspect`
  - any bucket still marked partial/deferred in `05-epic2-progress.md`
- Where Epic 3 scope intentionally owns a feature, document that boundary explicitly and either complete it or file/refresh the exact follow-up issue.

- `Epic-5` RTX 5090 validation and landing the plane

*E5 Phase1* [Necessary vee commands used]
- Run the final validation matrix on the RTX 5090 host.
- Run all required smoke/dry-run/real tests from the completion checklist.
- Reconcile beads statuses with actual landed work.
- File any remaining follow-up issues with exact acceptance criteria.
- `git pull --rebase`
- `bd sync`
- `git push`
- verify `git status` shows up to date with origin

### Orchestration Protocols

- Claude code lead does not take overlapping write scopes with its workers.
- Claude lead must delegate exact file ownership and test gates.
- Claude lead may write only minimal coordinator artifacts; implementation should be delegated across disjoint workstreams.
- Each worker must respect the existing dirty worktree and must not revert unrelated edits.
- Quality gates must be run by every agent after every change.
- Claude lead must continue until the success criteria above are achieved, not merely until one issue closes.

## Workflow

- Run `vee --help` to understand how vee will be used in the lifecycle.
- Understand task, why it matters, how to finish it, and where the remaining blockers live.
- Begin document reconciliation phase.
- Continue through each phase and each epic until all success criteria are met.

## Non-negotiables

- Entire process completed end to end.
- All code dependencies installed so Lazarus runs out of the box.
- Ready to use out of the box.
- Industry standard code practices.
- All code inserted must have equivalent tests.
- All tests passing smoke/dry-run/real.
