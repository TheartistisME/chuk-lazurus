# Epic 3: Portable Checkpoints & Multi-GPU — Scope (STUB)

Status: Stub (created by Epic 1b EWS-0; fleshed out in Epic 3 kickoff)
Parent: `docs/refactor/dual-backend-cuda/01-implementation-spec.md`
Sibling: `docs/refactor/dual-backend-cuda-all-buckets/02-implementation-spec.md`

This stub exists so that Epic 1b `NotImplementedError` messages can point to
stable anchors. Each anchor below MUST remain addressable; flesh out under
the anchor, do not rename.

## Scope anchors (do not rename)

### ckpt-convert
CLI `lazarus context convert --from mlxckpt --to torchckpt` (and reverse).
Implements tensor-level conversion between `.mlxckpt` and `.torchckpt` v1
formats. Schema reference: Epic 1b spec §5.2.

### weights-convert
CLI `lazarus weights convert --from mlx --to hf <path>` (and reverse).
Converts MLX-layout safetensors to HF AutoModelForCausalLM-loadable layout
(parameter renames + shard map). Detection: Epic 1b spec §5.4.

### multi-gpu
Enable `device_map="auto"` and explicit multi-device shard maps. Integrates
`accelerate` big-model inference. Rejected at config ingress in Epic 1b
(spec §4, §10 rejection rule).

### training-resume
Cross-backend training checkpoint resume (MLX-saved → torch-resumed and
vice versa). Requires a portable optimiser-state format; today each backend
uses native formats (Epic 1b §14 hard-errors on cross-backend resume).

## Out of scope for Epic 3
- Feature parity items → Epic 2.
