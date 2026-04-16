# Epic 2: CUDA Feature Parity — Scope (STUB)

Status: Stub (created by Epic 1b EWS-0; fleshed out in Epic 2 kickoff)
Parent: `docs/refactor/dual-backend-cuda/01-implementation-spec.md`
Sibling: `docs/refactor/dual-backend-cuda-all-buckets/02-implementation-spec.md`

This stub exists so that Epic 1b `NotImplementedError` messages can point to
stable anchors. Each anchor below MUST remain addressable; flesh out under
the anchor, do not rename.

## Scope anchors (do not rename)

### residual-inject
Port vec-inject residual extraction + injection to torch. Owner: TBD.
Entry files: `inference/context/research/vec_inject/_primitives.py`,
`cli/commands/context/prefill/_vec_inject.py`,
`inference/backends/torch_runtime.py::extract_residual_state`.

### kv-direct
Port the KV-direct generator (`inference/context/kv_generator.py` +
`inference/unified.py:583` gate) to torch. Owner: TBD.

### quantisation
Add bitsandbytes / AWQ / GPTQ quantised loading for torch. Owner: TBD.
Entry files: `inference/loader.py` torch arm, new `inference/quant/`.
Not in Epic 1b.

## Out of scope for Epic 2
- Cross-backend checkpoint conversion → Epic 3.
- Multi-GPU / FSDP → Epic 3.
