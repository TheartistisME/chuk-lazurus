# Deprecated Docs Register

Last reviewed: 2026-04-15

This register tracks docs retained for history but no longer treated as canonical.

| Document | Why deprecated | Canonical replacement |
| --- | --- | --- |
| [SPEC.md](SPEC.md) | Early decoupled-attention architecture draft superseded by the finalized knowledge-store architecture and current context docs. | [SPEC_V7.md](SPEC_V7.md), [context/README.md](context/README.md), [context/prefill/vec_inject.md](context/prefill/vec_inject.md) |
| [SPEC_PREFILL.md](SPEC_PREFILL.md) | Refactor-gap analysis written before residual chaining and persisted final residual support landed. | [SPEC_V7.md](SPEC_V7.md), [context/README.md](context/README.md), [context/prefill/vec_inject.md](context/prefill/vec_inject.md) |
| [roadmap-batching.md](roadmap-batching.md) | Historical implementation roadmap; maintained batching and training behavior now lives in the current module guides. | [batching.md](batching.md), [training.md](training.md) |
| [introspection-refactoring-roadmap.md](introspection-refactoring-roadmap.md) | Older introspection split plan superseded by the current package layout and the active combined roadmap. | [introspection.md](introspection.md), [roadmap-introspection-moe.md](roadmap-introspection-moe.md) |
| [moe-refactoring-roadmap.md](moe-refactoring-roadmap.md) | Earlier MoE-specific refactor plan superseded by the `introspection/moe/` package and the active combined roadmap. | [introspection.md](introspection.md), [roadmap-introspection-moe.md](roadmap-introspection-moe.md), [expert-compression.md](expert-compression.md) |
| [virtual-math-expert.md](virtual-math-expert.md) | Narrative experiment write-up that uses legacy approach names; the maintained user-facing guide is the unified virtual experts doc. | [virtual_experts.md](virtual_experts.md), [introspection.md](introspection.md) |

## Still Canonical

- [SPEC_V7.md](SPEC_V7.md)
- [virtual_experts.md](virtual_experts.md)
