# Refactor Documentation Index

This directory holds active refactor planning and execution docs. For the current CUDA work, the all-buckets stack is the primary navigation path.

## Current CUDA Workstream

Read these in order:

1. [dual-backend-cuda-all-buckets/00-README.md](dual-backend-cuda-all-buckets/00-README.md)
2. [dual-backend-cuda-all-buckets/01-command-matrix.md](dual-backend-cuda-all-buckets/01-command-matrix.md)
3. [dual-backend-cuda-all-buckets/02-implementation-spec.md](dual-backend-cuda-all-buckets/02-implementation-spec.md)
4. [dual-backend-cuda-all-buckets/03-workstreams.md](dual-backend-cuda-all-buckets/03-workstreams.md)
5. [dual-backend-cuda-all-buckets/04-validation-matrix.md](dual-backend-cuda-all-buckets/04-validation-matrix.md)
6. [dual-backend-cuda-all-buckets/05-epic2-progress.md](dual-backend-cuda-all-buckets/05-epic2-progress.md)

## CUDA and RTX 5090 Supporting Docs

| Doc | Role |
|---|---|
| [dual-backend-cuda/01-implementation-spec.md](dual-backend-cuda/01-implementation-spec.md) | Epic 1 foundation contract for backend selection, lazy imports, and torch/CUDA bring-up. |
| [dual-backend-cuda/02-workstreams.md](dual-backend-cuda/02-workstreams.md) | Epic 1 parallel workstream split. |
| [dual-backend-cuda-epic2/00-scope.md](dual-backend-cuda-epic2/00-scope.md) | Epic 2 scope anchors for deferred CUDA parity work. |
| [dual-backend-cuda-epic3/00-scope.md](dual-backend-cuda-epic3/00-scope.md) | Epic 3 scope anchors for portable checkpoints and multi-GPU. |
| [dual-backend-cuda-epic3/01-post-epic2-handoff-spec.md](dual-backend-cuda-epic3/01-post-epic2-handoff-spec.md) | Current post-Epic-2 handoff/execution brief. |
| [rtx5090-cuda-vee-orchestrate.md](rtx5090-cuda-vee-orchestrate.md) | Vee orchestration brief for end-to-end RTX 5090 completion. |

## Other Refactor Docs

| Doc | Scope |
|---|---|
| [cli-introspect-refactor-roadmap.md](cli-introspect-refactor-roadmap.md) | Refactor plan for the CLI introspection surface. |
| [../introspection-refactoring-roadmap.md](../introspection-refactoring-roadmap.md) | Broader introspection package refactor roadmap stored at the docs root. |
| [../moe-refactoring-roadmap.md](../moe-refactoring-roadmap.md) | MoE-specific refactor roadmap stored at the docs root. |

## Canonical Usage

- Use this directory for execution plans, ownership boundaries, validation gates, and handoff context.
- Use the topic docs outside `docs/refactor/` for behavior, APIs, and subsystem design.
