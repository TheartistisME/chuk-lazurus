# Dual-Backend CUDA — All Buckets (Epic 1 Planning)

Planning docs for making every Lazarus CLI command bucket work on Nvidia CUDA while preserving the MLX/Metal path on Apple Silicon.

## Documents

| # | File | Purpose |
|---|------|---------|
| 01 | [01-command-matrix.md](01-command-matrix.md) | Full CLI inventory across every bucket with current vs target CUDA status and MLX-preservation note per row. Includes chokepoint table C1–C8 and the `circuit` standalone console script. |
| 02 | [02-implementation-spec.md](02-implementation-spec.md) | Exact file-level edits, pinned deps (`torch==2.9.1`, `transformers==4.56.0`, `accelerate==1.5.2`), per-trainer mixed-precision matrix, `.torchckpt` schema, backend dispatch templates, streaming decode contract, and anticipated test files. Cross-references Epic 2/3 stubs at `docs/refactor/dual-backend-cuda-epic2/00-scope.md` and `…-epic3/00-scope.md`. |
| 03 | [03-workstreams.md](03-workstreams.md) | 16 non-conflicting workstreams (EWS-0…15) with file ownership, Wave 0 → 0.5 → A1 → A2 → B merge order, quality gates (unit tests + MLX regression + CUDA smoke), fixtures and baselines owned by EWS-0, and deterministic rollback cascades. |
| 04 | [04-validation-matrix.md](04-validation-matrix.md) | Per-subcommand smoke/dry-run/real-exec matrix with CUDA acceptance, MLX regression, numeric tolerances, fixtures manifest, offline-cache policy, perf baseline storage, and CUDA determinism invariants (Appendix O). |

## Buckets covered

infer (standard + kv_direct) · context prefill · context generate · knowledge (build/query/chat) · introspect (full surface including circuit capture/invoke/decode/test/compare/view/export, virtual-expert, moe-expert [30 handlers], classifier, logit-lens) · serve · lazarus-serve (streaming + non-streaming) · train (sft/dpo/grpo/ppo/dual_reward) · generate (cross-reference) · data · tokenizer · gym · experiment · bench.

Standalone `circuit` console script is covered separately (see 01 §14, owned by EWS-6).

## MLX preservation

Every torch-target row preserves the MLX path via `get_backend("mlx")` dispatch. Every backend-agnostic row is explicitly marked "No MLX path exists; backend-agnostic." MLX regression is a mandatory merge gate on every workstream.

## Review history

Each doc went through adversarial review rounds before approval.

| Doc | R1 | R2 | R3 | R4 |
|-----|----|----|----|----|
| 01 command-matrix | REVISION | ACCEPT-minor | **APPROVED** | — |
| 02 implementation-spec | REVISION | REVISION | **APPROVED** | — |
| 03 workstreams | REVISION | REVISION | **APPROVED** | — |
| 04 validation-matrix | REVISION | REVISION | REVISION | **APPROVED** |

## Next step

Epic 1 planning is complete. Epic 2 execution begins with EWS-0 (seeds parser splits, fixture harness, Epic 2/3 scope stubs, `cuda_exemption_auditor.yml`, then transfers split-file ownership to downstream streams).
