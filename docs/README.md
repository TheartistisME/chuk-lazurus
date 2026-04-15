# Lazarus Documentation Hub

This directory has a mix of user guides, architecture specs, research notes, and active refactor planning. Use this page as the navigation layer; the topic docs linked here remain the canonical source for their subject.

## Start Here

| Area | Canonical docs | Use this when |
|---|---|---|
| Project entry | [getting-started.md](getting-started.md), [cli.md](cli.md), [api-reference.md](api-reference.md) | You need installation, the main CLI surface, or the public Python API. |
| Inference and serving | [inference.md](inference.md), [server.md](server.md), [models.md](models.md) | You are loading models, generating text, or running the HTTP server. |
| Context system | [context/README.md](context/README.md), [SPEC_PREFILL.md](SPEC_PREFILL.md), [SPEC_V7.md](SPEC_V7.md) | You are working on unlimited context, prefill, routing, or the knowledge-store architecture. |
| Introspection tools | [introspection.md](introspection.md), [tools/README.md](tools/README.md), [tool_calling_circuit.md](tool_calling_circuit.md) | You need the mechanistic interpretability surface or a specific introspection command. |
| Training and data | [training.md](training.md), [batching.md](batching.md), [roadmap-batching.md](roadmap-batching.md) | You are training models or working on batch planning and data flow. |
| Refactor and CUDA work | [refactor/README.md](refactor/README.md) | You need active implementation plans, workstreams, validation, or handoff docs. |

## Major Areas

### Product and API docs

- [getting-started.md](getting-started.md) is the quickest entry point for install and first run.
- [cli.md](cli.md) is the top-level CLI reference.
- [api-reference.md](api-reference.md) is the public Python API reference.
- [inference.md](inference.md), [models.md](models.md), and [server.md](server.md) cover runtime usage by subsystem.
- [training.md](training.md) and [batching.md](batching.md) are the canonical training/data guides.

### Context and knowledge-store architecture

- [context/README.md](context/README.md) is the index for prefill phases, routing strategies, and multi-step routing.
- [SPEC.md](SPEC.md), [SPEC_V7.md](SPEC_V7.md), and [SPEC_PREFILL.md](SPEC_PREFILL.md) are the architecture/spec stack behind the context system.

### Introspection and tool docs

- [introspection.md](introspection.md) is the overview for the introspection surface.
- [tools/README.md](tools/README.md) groups the per-command docs under `docs/tools/`.
- [tool_calling_circuit.md](tool_calling_circuit.md) is a standalone deep-dive analysis doc.
- [virtual_experts.md](virtual_experts.md), [virtual-math-expert.md](virtual-math-expert.md), [expert-compression.md](expert-compression.md), [gemma_alignment_circuits.md](gemma_alignment_circuits.md), and [refusal_direction_findings.md](refusal_direction_findings.md) are topic-specific research notes.

### Roadmaps and refactors

- [refactor/README.md](refactor/README.md) is the entry point for active refactor execution docs.
- [introspection-refactoring-roadmap.md](introspection-refactoring-roadmap.md), [moe-refactoring-roadmap.md](moe-refactoring-roadmap.md), [roadmap-introspection-moe.md](roadmap-introspection-moe.md), and [roadmap-batching.md](roadmap-batching.md) are roadmap docs that live at the docs root.

## Current CUDA Workstream

If you are trying to understand the live CUDA/refactor effort, start in this order:

1. [refactor/README.md](refactor/README.md)
2. [refactor/dual-backend-cuda-all-buckets/00-README.md](refactor/dual-backend-cuda-all-buckets/00-README.md)
3. [refactor/dual-backend-cuda-all-buckets/03-workstreams.md](refactor/dual-backend-cuda-all-buckets/03-workstreams.md)
4. [refactor/dual-backend-cuda-all-buckets/04-validation-matrix.md](refactor/dual-backend-cuda-all-buckets/04-validation-matrix.md)
5. [refactor/dual-backend-cuda-all-buckets/05-epic2-progress.md](refactor/dual-backend-cuda-all-buckets/05-epic2-progress.md)

## Directory Indexes

- [context/README.md](context/README.md)
- [tools/README.md](tools/README.md)
- [refactor/README.md](refactor/README.md)
