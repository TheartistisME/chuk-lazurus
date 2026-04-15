# Tools Documentation Index

This directory is the command-level reference for the introspection and circuit tooling. The topic docs themselves stay canonical; this page is the navigation layer.

## Canonical Entry Points

| Need | Start here |
|---|---|
| Overview of the introspection surface | [../introspection.md](../introspection.md) |
| Per-command docs for `lazarus introspect ...` | This directory |
| Standalone `circuit` console script | [circuit-cli.md](circuit-cli.md) |
| Introspection refactor planning | [../refactor/README.md](../refactor/README.md), [../refactor/cli-introspect-refactor-roadmap.md](../refactor/cli-introspect-refactor-roadmap.md) |

## Command Groups

### Core inspection

- [introspect-analyze.md](introspect-analyze.md)
- [introspect-compare.md](introspect-compare.md)
- [introspect-generate.md](introspect-generate.md)
- [introspect-layer.md](introspect-layer.md)
- [introspect-embedding.md](introspect-embedding.md)
- [introspect-early-layers.md](introspect-early-layers.md)
- [introspect-hooks.md](introspect-hooks.md)
- [introspect-logit-lens.md](introspect-logit-lens.md)
- [introspect-neurons.md](introspect-neurons.md)

### Causal testing and interventions

- [introspect-ablate.md](introspect-ablate.md)
- [introspect-patch.md](introspect-patch.md)
- [introspect-steer.md](introspect-steer.md)
- [introspect-directions.md](introspect-directions.md)
- [introspect-operand-directions.md](introspect-operand-directions.md)
- [introspect-activation-diff.md](introspect-activation-diff.md)
- [introspect-weight-diff.md](introspect-weight-diff.md)

### Probing, classifiers, and diagnostics

- [introspect-probe.md](introspect-probe.md)
- [introspect-classifier.md](introspect-classifier.md)
- [introspect-metacognitive.md](introspect-metacognitive.md)
- [introspect-uncertainty.md](introspect-uncertainty.md)
- [introspect-dual-reward.md](introspect-dual-reward.md)
- [introspect-format-sensitivity.md](introspect-format-sensitivity.md)
- [introspect-activation-cluster.md](introspect-activation-cluster.md)
- [introspect-commutativity.md](introspect-commutativity.md)
- [introspect-arithmetic.md](introspect-arithmetic.md)

### Memory and MoE analysis

- [introspect-memory.md](introspect-memory.md)
- [introspect-moe-expert.md](introspect-moe-expert.md)

## Notes

- If you need the full command surface first, read [../introspection.md](../introspection.md).
- If you are tracking active CLI/introspection restructuring work, use [../refactor/README.md](../refactor/README.md) instead of inferring status from these command docs.
