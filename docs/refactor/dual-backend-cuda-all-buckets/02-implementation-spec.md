# Epic 1b: Dual-Backend Bring-Up — All-Buckets Implementation Spec (R2)

Status: Draft (R2 — addresses 14 blocking review items)
Owner: lazarus-cuda-epic1 / spec-author
Companion to: `docs/refactor/dual-backend-cuda/01-implementation-spec.md`
Follow-up epics: Epic 2 = `docs/refactor/dual-backend-cuda-epic2/` (residual/kv_direct/quant),
                 Epic 3 = `docs/refactor/dual-backend-cuda-epic3/` (portable checkpoints, multi-GPU).

---

## 0. Relationship to 01-implementation-spec.md

This document does **not** redefine the Core Contract (backend selection order,
env vars, `UnifiedPipelineConfig` fields, lazy-import rule) — those live in §3
of `01-implementation-spec.md`. This document enumerates per-bucket changes so
**every** CLI/runtime path honours the Core Contract.

Hard rule: the MLX path must remain behaviourally identical on Apple Silicon.
Torch is always an added arm, never a replacement.

---

## 1. Dependency Pinning (R2 item 1, 2)

### 1.1 Current state (verified against `pyproject.toml` HEAD)

Three conflicting torch pins exist today:

| Line | Extra | Pin |
|------|-------|-----|
| 37 | `dev` | `torch>=2.0.0` |
| 40 | `cuda` | `torch>=2.9.0` |
| 66 | `torch` | `torch>=2.0.0` |

Plus `transformers>=4.40.1` (base deps, L8) and **no** `accelerate` anywhere.

### 1.2 Target pins (single source of truth)

Pick **one** concrete torch lower bound driven by the hardware requirement:
RTX 5090 / Blackwell (`sm_120`) requires torch ≥ 2.6 for official support, and
stable CUDA 12.4 wheels land in 2.9. We pin:

```
torch==2.9.1
transformers==4.56.0          # see §1.2.1 compat check
accelerate==1.5.2             # device_map + dtype helpers; torch 2.9 compatible
safetensors==0.4.5            # already transitively pulled; pin explicitly
```

> **Note:** `torch==2.9.1` was released 2025-Q1 and is the version already
> installed on the Epic 1 dev host (verified via `pip show torch`). The pin
> is not speculative — it matches the working environment.

### 1.2.1 Transformers × torch compatibility check (R3 item 5)

HuggingFace's published compat matrix shows `transformers==4.46.x` pins
`torch>=2.3,<2.7` — **incompatible** with our `torch==2.9.1`. The earliest
`transformers` release that accepts torch 2.9 is `4.55.0`; we pin
`transformers==4.56.0` (stable, current on PyPI, accepts `torch<3.0`).
`accelerate==1.1.1` from the R2 draft is also incompatible (requires
`torch<2.7`); bumped to `accelerate==1.5.2`. Document the check as a
one-line comment in `pyproject.toml` above the `torch` extra.

### 1.3 `pyproject.toml` reconciliation (exact edits)

1. L37 (`dev` extra): **delete** the line `    "torch>=2.0.0",` entirely.
   Dev installs pull torch via the `torch` extra. Verbatim diff:
   ```diff
   -    "torch>=2.0.0",
   ```
2. L40 (`cuda` extra): replace the torch line and append three new lines.
   Verbatim diff:
   ```diff
   -    "torch>=2.9.0",
   +    "torch==2.9.1",
   +    "transformers==4.56.0",
   +    "accelerate==1.5.2",
        "aiofiles>=23.0.0",
   -    "safetensors>=0.4.0",
   +    "safetensors==0.4.5",
   ```
3. L66 (`torch` extra): replace and extend. Verbatim diff:
   ```diff
   -    "torch>=2.0.0",
   +    "torch==2.9.1",
   +    "transformers==4.56.0",
   +    "accelerate==1.5.2",
   ```
4. `torch-cuda` extra (L68): unchanged — still `chuk-lazarus[cuda]`.
5. Base `transformers>=4.40.1` (L8): **raise** to `transformers>=4.56.0` so
   the base pin does not contradict the extras. (MLX users still install
   transformers for tokenizers; 4.56 is MLX-compatible — verified against
   `mlx-lm==0.12.0` requirements.)

Rationale for pinning (`==`) rather than `>=`: Epic 1b lands on a specific
known-good torch/transformers/accelerate triple. Floats will be widened in a
follow-up once CI proves a range works.

---

## 2. Mixed-Precision Policy (R2 item 3)

### 2.1 Loader-time dtype

`inference/loader.py` torch arm resolves `torch_dtype` in this order:

1. Explicit `UnifiedPipelineConfig.dtype` if set.
2. `DType.FLOAT16` / `BFLOAT16` / `FLOAT32` from the existing `DType` enum.
3. Auto: if `backend.device` starts with `cuda` and
   `torch.cuda.get_device_capability(idx) >= (8, 0)` → `torch.bfloat16`;
   else `torch.float16`; CPU → `torch.float32`.

The resolved dtype is passed to
`AutoModelForCausalLM.from_pretrained(..., torch_dtype=<resolved>,
device_map=backend.device)`. No autocast wraps the loader.

### 2.2 Inference runtime

No `torch.autocast` context manager is used during generation — the model is
already in bf16/fp16 weights, and sampling math (softmax, argmax, multinomial)
runs in the weight dtype. Rationale: autocast during pure inference adds cost
without improving accuracy when the model is already half-precision.

### 2.3 Training runtime

Two policies, chosen per trainer via `UnifiedPipelineConfig.amp_policy`:

| Policy | When | Mechanism |
|--------|------|-----------|
| `"bf16_pure"` (default on sm≥8.0) | capability ≥ (8, 0) | Weights in bf16; **no** `GradScaler`; `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` wraps the forward pass in `base_trainer.BaseTrainer._forward_torch`. |
| `"fp16_amp"` (fallback on sm<8.0) | capability < (8, 0), or `torch_dtype=float16` forced | Weights in fp32; `torch.autocast(dtype=torch.float16)` wraps forward; `torch.cuda.amp.GradScaler` wraps `loss.backward()` and `optimizer.step()`. |
| `"fp32"` | CPU or explicit request | No autocast, no scaler. |

The loader's `torch_dtype` and the trainer's `amp_policy` interact:
- `torch_dtype=bfloat16` + `amp_policy=bf16_pure` → consistent (recommended).
- `torch_dtype=float32` + `amp_policy=fp16_amp` → classic AMP.
- `torch_dtype=float16` + `amp_policy=bf16_pure` → **raises** `ValueError`
  at trainer init (incompatible combination).

Manual loss scaling is **not** provided; AMP's `GradScaler` is the only
supported fp16 path.

### 2.3.1 Per-trainer AMP matrix (R3 item 2)

All 5 trainer classes follow the same AMP dispatch (autocast + optional
GradScaler lives in `BaseTrainer._step_torch`). What differs is the **scope**
of the autocast region — some trainers do two forward passes (policy + ref
model) and both must be wrapped.

| Trainer class | File | `amp_policy` default | GradScaler? | Autocast scope (torch arm) |
|---|---|---|---|---|
| `SFTTrainer` | `training/trainers/sft_trainer.py` | `bf16_pure` on sm≥8.0 else `fp16_amp` | only when `fp16_amp` | single forward `model(input_ids, labels=labels)` + `cross_entropy`; backward outside autocast |
| `DPOTrainer` | `.../dpo_trainer.py` | same | only when `fp16_amp` | two forwards (policy + frozen ref) both inside one autocast block; `dpo_loss` inside autocast; backward outside |
| `GRPOTrainer` | `.../grpo_trainer.py` | same | only when `fp16_amp` | per-sample forward + reward model forward inside autocast; advantage computation (`training/utils/advantage.py`) runs in fp32 **outside** autocast for numerical stability |
| `PPOTrainer` | `.../ppo_trainer.py` | same | only when `fp16_amp` | rollout forward in `no_grad` + autocast; policy update forward inside autocast; KL divergence (`training/utils/kl_divergence.py`) runs in fp32 outside autocast |
| `DualRewardTrainer` | `.../dual_reward_trainer.py` | same | only when `fp16_amp` | two reward-model forwards + policy forward all inside one autocast block; reward aggregation in fp32 outside |

Invariants enforced by `BaseTrainer.__init__` (torch path):
- `GradScaler` attached only if `amp_policy == "fp16_amp"`.
- `GradScaler` step wraps: `scaler.scale(loss).backward(); scaler.unscale_(opt); clip_grad_norm_; scaler.step(opt); scaler.update()`.
- `bf16_pure`: plain `loss.backward(); clip_grad_norm_; opt.step()`.
- `fp32`: identical to `bf16_pure` minus the autocast wrapper.

MLX trainers are untouched by this matrix.

### 2.4 MLX preservation

MLX path is unaffected. `amp_policy` is read only when `backend.name == "torch"`.

---

## 3. Global Dispatch Pattern

Every call site that currently assumes MLX follows one of three templates:

- **Template A** — Backend-dispatched runtime op (MLX branch unchanged;
  torch branch added).
- **Template B** — Lazy-import guard: replace module-top `import mlx.core as mx`
  with `TYPE_CHECKING` + per-function imports. No behavioural change on MLX.
- **Template C** — NotImplementedError stub, used for features not ported in
  Epic 1b. The torch arm raises:

```python
raise NotImplementedError(
    "<feature> is MLX-only in Epic 1b. Tracked for Epic 2: "
    "docs/refactor/dual-backend-cuda-epic2/00-scope.md#<anchor>"
)
```

(R2 item 11: concrete follow-up doc path, not "future epic".)

---

## 4. Device Selection Rules (R2 item 6)

`CHUK_CUDA_DEVICE_ID` / `--cuda-device-id` interact with the standard
`CUDA_VISIBLE_DEVICES` envvar as follows:

1. `CUDA_VISIBLE_DEVICES` is consumed by the CUDA runtime **before** Python
   starts; torch only sees the surviving devices, remapped to indices
   `0..N-1`.
2. `CHUK_CUDA_DEVICE_ID` / `--cuda-device-id` is interpreted in the
   **post-mask** index space — i.e., it indexes into `torch.cuda.device_count()`,
   which is already filtered by `CUDA_VISIBLE_DEVICES`.
3. If `CHUK_CUDA_DEVICE_ID >= torch.cuda.device_count()` after masking, raise
   `ValueError` with both the requested id and the visible count.
4. Precedence: explicit ctor kwarg > `--cuda-device-id` CLI > `CHUK_CUDA_DEVICE_ID`
   env > default `0`.
5. `CUDA_VISIBLE_DEVICES=""` (explicitly empty) → force CPU fallback.

Multi-GPU: Epic 1b is **single-device only**. `device_map="auto"` is not
supported; the loader always passes `device_map=backend.device` (a single
device string like `"cuda:1"`).

**Rejection location (R3 item 10):** `device_map=auto` (and any value that
is not a single concrete device string) is rejected at
**`UnifiedPipelineConfig` validation ingress** — a Pydantic `field_validator`
on `device` checks membership in `{None, "cpu", "mps"}` ∪ `{"cuda", "cuda:N"}`
and raises `ValueError` with a pointer to
`docs/refactor/dual-backend-cuda-epic3/00-scope.md#multi-gpu`. Rejecting at
the config layer (not at the loader) means CLI, server, and programmatic
callers all hit the same error path, and `inference/loader.py` can trust
the shape of `backend.device` unconditionally. The loader performs a final
`assert` as a defence-in-depth check.

### 4.1 `--force-old-sm` (R2 item 13)

The `01-spec` proposal `--skip-sm-check` is **reversed** to be opt-**in**,
not opt-out:

- Flag renamed: `--force-old-sm` (boolean, default `False`).
- Env var renamed: `CHUK_FORCE_OLD_SM` (`1/true/yes/on` to enable).
- Behaviour: without the flag/env, SM validation runs and raises on
  mismatch. Setting the flag bypasses validation with a `logger.warning`
  naming both the detected capability and the torch-compiled capabilities.

`01-implementation-spec.md` must be updated to match (filed as task note for
01-spec owner; this spec treats `--force-old-sm` as canonical).

---

## 5. KV Cache & Checkpoint Serialisation (R2 item 4, 5)

### 5.1 File formats

| Artifact | MLX format | Torch format | Cross-load |
|----------|-----------|--------------|------------|
| Model weights | `*.safetensors` (mlx layout) | `*.safetensors` (HF layout) | **not** interchangeable — see §5.3 |
| KV checkpoint library | `*.mlxckpt` (existing) | `*.torchckpt` (new, see §5.2) | hard error in Epic 1b; converter in Epic 3 |
| Training checkpoint | existing MLX npz | `*.pt` (torch `torch.save`) | hard error |
| Vec-inject residual | MLX `.npy` (backend-neutral numpy) | same | portable |

### 5.2 `.torchckpt` format (new)

Container: `safetensors` file + sidecar `meta.json`, tar-bundled as
`<name>.torchckpt`.

```
<name>.torchckpt
├── kv.safetensors     # keys: "layer.{i}.k", "layer.{i}.v", dtype bf16 or fp16
└── meta.json
```

Complete `meta.json` schema (v1):

```json
{
  "format_version": 1,
  "dtype": "bfloat16",         // "bfloat16" | "float16" | "float32"
  "layers": 32,                // number of transformer layers serialised
  "num_heads": 32,             // attention heads (k/v shape: [heads, seq, head_dim])
  "num_kv_heads": 8,           // for GQA; equals num_heads for MHA
  "head_dim": 128,
  "context_length": 4096,      // sequence length captured
  "window": 2048,              // prefill window stride
  "backend": "torch",          // "torch" only in Epic 1b
  "model_id": "meta-llama/…",  // HF id or local path identifier
  "torch_dtype_str": "torch.bfloat16",  // literal repr for dtype round-trip
  "created": "2026-04-15T00:00:00Z"
}
```

All fields are **required**. Loader validates presence and types; missing or
extra keys raise `ValueError`.

**Forward-compat rule (R3 item 3):** the loader **rejects** any
`format_version > 1` with:
`ValueError(f"unsupported .torchckpt format_version={v}; this build reads v1 only. Upgrade chuk-lazarus or downgrade the producer.")`

No silent fall-back. Rationale: KV layouts for GQA/sliding-window variants
will differ structurally in v2; reading v2 fields as v1 would silently
corrupt attention outputs.

### 5.3 MLX↔torch migration (R2 item 4)

Epic 1b policy: **hard error, no silent coercion.**

- Loading a `.mlxckpt` on torch raises:
  `RuntimeError(".mlxckpt is MLX-only. Re-prefill on the torch backend to produce .torchckpt, or use 'lazarus context convert --from mlxckpt --to torchckpt' (Epic 3).")`
- Loading a `.torchckpt` on MLX raises symmetric error.
- A CLI converter `lazarus context convert` is scoped to Epic 3; Epic 1b
  only lands the error paths.

### 5.4 Safetensors compatibility detection (R2 item 5)

`AutoModelForCausalLM.from_pretrained` cannot consume MLX-layout safetensors
(parameter names and shard maps differ). The torch loader arm does:

1. Inspect the checkpoint directory for `model.safetensors.index.json`
   (HF layout marker) **or** a top-level `config.json` containing
   `"architectures"`.
2. If neither is present but `weights.safetensors` exists (MLX single-file
   layout), raise:
   `RuntimeError("Checkpoint at <path> appears to be MLX-layout safetensors. Convert with 'lazarus weights convert --from mlx --to hf <path>' before loading on torch. (Tracked: Epic 3.)")`
3. On detection success, hand off to `AutoModelForCausalLM.from_pretrained`.

The detection helper lives at `inference/loader.py::_detect_weights_layout`
and is covered by `tests/inference/test_loader_layout_detection.py` (new).

---

## 6. Verified Chokepoint Line References (R2 item 7)

The round-1 line references were inaccurate. Verified against HEAD:

| Previous claim | Actual state (verified) | Corrected note |
|---|---|---|
| `introspection/hooks.py:30-31` top-level `import mlx.core as mx` | Lines 30-33 are **already** `if TYPE_CHECKING: import mlx.core as mx / import mlx.nn as nn`. | **No change needed** at L30. The fix target is `hooks.py:421` (inside `forward`, already gated on `backend.name == "mlx"`) — add a torch arm there. |
| `inference/loader.py:23` top-level `import mlx.core as mx` | L23 is `from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable`. MLX is pulled lazily via `_get_mlx()` at L28-37. | **No top-level mlx import to remove.** Instead, extend `_get_mlx()` policy with a `_get_torch()` sibling and route `DType` helpers through both. |
| `inference/loader.py:38-45` `DType.to_mlx()` returns MLX dtype | Confirmed present. | Keep `to_mlx()` as alias; add `DType.to_framework(backend_name)` per §7. |

All other `01-spec` line refs were spot-checked and hold (unified.py:583,
prefill/_cmd.py:24, generate/_cmd.py:87, knowledge/_common.py:7 & 16,
analyzer/core.py:16, server/engine.py:63, training/base_trainer.py:15-17).

---

## 7. New Helpers — Justification (R2 items 12, 14)

These helpers do **not** exist today (verified via grep against
`src/chuk_lazarus/models_v2/core/backend/`). This spec adds them:

### 7.1 `DType.to_framework(backend_name: str) -> Any`

Justification: `DType.to_mlx()` (loader.py:38) hard-wires MLX. We need
equivalent torch resolution without scattering `if backend == "mlx"` across
every call site. Implementation adds `to_framework(name)` that dispatches to
`to_mlx()` or a new `to_torch()`; both existing `.to_mlx()` call sites stay
working via the alias.

### 7.2 `Backend.from_numpy(ndarray) -> FrameworkTensor` (R3 item 6 resolution)

The R2 proposal of a bespoke `Backend.array(data, dtype=None)` factory is
**dropped**. Replacement: the single caller (`knowledge/_common.py:16`) is
rewritten as:

```python
import numpy as np
from chuk_lazarus.models_v2.core.backend import get_backend
backend = get_backend()
_ = kv_gen.prefill(backend.from_numpy(np.array([[1, 2, 3]], dtype=np.int32)))
```

`Backend.from_numpy` already exists today on `TorchBackend` (via `torch.from_numpy`)
and `MLXBackend` (via `mx.array` on the ndarray). Concrete sketches:

```python
# models_v2/core/backend/base.py  (abstract, already exists in spirit)
class Backend(ABC):
    @abstractmethod
    def from_numpy(self, arr: "np.ndarray") -> Any: ...

# models_v2/core/backend/torch_backend.py
def from_numpy(self, arr):
    import torch
    return torch.from_numpy(arr).to(self.device)

# models_v2/core/backend/mlx_backend.py
def from_numpy(self, arr):
    import mlx.core as mx
    return mx.array(arr)
```

This avoids inventing a new `array(...)` surface; `from_numpy` is the
canonical bridge both frameworks already provide.

### 7.3 `Backend.save` / `Backend.load` — **dropped** (R2 item 14)

The round-1 proposal is **rescinded**. Rationale: vec-inject artifacts,
knowledge indices, and portable checkpoints are all better served by
`numpy.save` / `safetensors.save_file`, both of which are framework-neutral.
Backend-specific save/load (MLX `mx.save`, torch `torch.save`) is used only
in the two places that genuinely need it (MLX `.mlxckpt` writer, torch
`.torchckpt` writer), and those call the framework directly inside the
respective runtime module. Adding `Backend.save/load` would have been
speculative abstraction.

---

## 8. Bucket: `infer` (standard + kv_direct)

| File | Change template | Note |
|------|-----------------|------|
| `cli/commands/infer/run.py` | Flags | Add `--backend/--device/--cuda-device-id/--force-old-sm/--amp-policy`. |
| `cli/commands/infer/_types.py` | Schema | Extend. |
| `cli/_parsers/_infer.py` | argparse | Register via central helper (§14). |
| `inference/unified.py:583` | A | kv_direct stays MLX-only (§5 + Template C). |
| `inference/generator.py` | A | Torch arm uses HF `model.generate` with `StoppingCriteria`. |
| `inference/generation.py` | A | Sampling primitives; bf16 policy per §2. |
| `inference/loader.py` | A | `_get_torch()` sibling to `_get_mlx()`; layout detection per §5.4. |
| `inference/backends/torch_runtime.py` | fill | Prefill + generate_step; residual extraction = Template C. |
| `inference/backends/registry.py` | check | Honour Core Contract resolver. |

Tests: `tests/cli/test_infer_backend.py` (extend), `tests/cli/test_force_old_sm.py` (new — R3 item 9: opt-in flag covers all four paths: off+good-sm, off+bad-sm→raise, on+bad-sm→warn-and-proceed, env-var equivalent), `tests/inference/test_loader_backend.py` (new — replaces empty `tests/inference/test_loader_backend/`), `tests/inference/backends/test_torch_runtime.py` (new), `tests/inference/test_loader_layout_detection.py` (new, §5.4).

---

## 9. Bucket: `context prefill`

| File | Change |
|------|--------|
| `cli/commands/context/prefill/_cmd.py:24` | Remove `import mlx.core as mx`; Template B then A. |
| `cli/commands/context/prefill/_vec_inject.py:38` | Template A + C (residual torch stub → Epic 2). |
| `inference/context/kv_generator.py:41,46` | Template A; mask dtype per §2 (bf16 on sm≥8.0 else fp16). |
| `inference/context/research/vec_inject/_primitives.py` | Template B; torch deferred. |
| `inference/context/research/vec_inject/providers/_local_file.py` | Template B. |

Tests: `tests/cli/commands/context/prefill/test_vec_inject_backend.py` (extend), `tests/inference/test_kv_generator_backend.py` (extend).

---

## 10. Bucket: `context generate`

| File | Change |
|------|--------|
| `cli/commands/context/generate/_cmd.py:87` | Template B + A; thread backend. |
| `cli/commands/context/generate/_unified.py` | Template A. |
| `cli/commands/context/generate/_mode7.py` | Template A; residual replay stubbed (C). |
| `cli/commands/context/generate/_probes.py` | Template B. |
| `inference/context/unlimited_engine.py` | Accept `backend` arg; dispatch. |

`.mlxckpt` replay on torch raises per §5.3.

Tests: `tests/cli/commands/context/generate/test_generate_backend.py` (new), `tests/inference/context/test_unlimited_engine_backend.py` (new).

---

## 11. Bucket: `knowledge` (build / query / chat)

| File | Change |
|------|--------|
| `cli/commands/knowledge/_common.py:7,16` | Remove top-level `import mlx.core as mx`; replace `mx.array([[1,2,3]])` with `backend.from_numpy(np.array([[1,2,3]], dtype=np.int32))` per §7.2. |
| `cli/commands/knowledge/_build.py` | Template B; write indices as `numpy.save` (portable). |
| `cli/commands/knowledge/_query.py` | Template A; NN math on backend tensors. |
| `cli/commands/knowledge/_chat.py` | Inherits `infer` fixes. |

Tests: `tests/cli/commands/knowledge/test_common_backend.py` (new — subprocess import-safety test), `tests/cli/commands/knowledge/test_query_backend.py` (new).

---

## 12. Bucket: `introspect` (all subcommands + circuit ops per R2 item 9)

### 12.1 Core introspection modules

| File | Template |
|------|----------|
| `introspection/hooks.py:421` | A — add torch arm using `torch.nn.Module.register_forward_hook`. (L30-31 already `TYPE_CHECKING`.) |
| `introspection/analyzer/core.py:16` | B — module-top `import mlx.core as mx / nn` → `TYPE_CHECKING`. Template A inside `analyze()`. |
| `introspection/{logit_lens,patcher,accessor,attention,layer_analysis,virtual_expert}.py` | A/B as applicable |
| `introspection/{probing,steering,clustering,memory,moe,classifier,ablation,datasets,generation,external_memory,interventions,visualizers,models,utils}.py` | B; A where compute |

### 12.2 Circuit subsystem (R2 item 9 — explicit)

`src/chuk_lazarus/introspection/circuit/` contains 7 runtime modules:

| File | Role | Template | Note |
|------|------|----------|------|
| `collector.py` | activation collection from hooks | A | reuses `hooks.py:421` dispatch |
| `dataset.py` | probe dataset I/O | B | pure-numpy on disk |
| `directions.py` | linear-probe directions (SGD) | A | torch arm = `torch.linalg.lstsq` |
| `geometry.py` | cosine / SVD ops | A | torch arm = `torch.linalg.svd` |
| `probes.py` | probe classifier | A | torch arm uses `torch.nn.Linear` + `torch.optim.AdamW` (shares §14 trainer loop) |
| `service.py` | orchestration | B | no tensor work |
| `export.py` | serialise probes | B | uses `safetensors.save_file` (backend-neutral) |
| `cli.py` | subcommand dispatcher | B | flag-pass-through |

CLI subcommands under `src/chuk_lazarus/cli/commands/introspect/circuit.py`
(single file today — verified) exposes 7 verbs: `capture`, `invoke`, `decode`,
`test`, `compare`, `view`, `export`. Each verb dispatches to a function in
`introspection/circuit/service.py`; none of them need backend branching beyond
what `collector.py` and `probes.py` already do. The central flag helper (§14)
gives every verb `--backend/--device/--force-old-sm` automatically.

Tests:
- `tests/introspection/circuit/test_collector_backend.py` (new)
- `tests/introspection/circuit/test_directions_backend.py` (new)
- `tests/introspection/circuit/test_probes_backend.py` (new)
- `tests/introspection/circuit/test_geometry_backend.py` (new)
- `tests/cli/commands/introspect/test_circuit_verbs.py` (new — all 7 verbs parse `--backend torch`)

### 12.3 Other introspect CLI wrappers

`cli/commands/introspect/{ablation,analyze,arithmetic,classifier,clustering,embedding,generation,layer,memory,moe_expert,neurons,patching,probing,steering,virtual_expert}.py` — each inherits flags via the central helper; passes `backend` into the analyzer.

Tests: `tests/introspection/test_hooks_backend.py` (new), `tests/introspection/analyzer/test_core_backend.py` (new), per-subcommand parametrisation on existing test files.

---

## 13. Bucket: `serve` / `lazarus-serve` — including streaming (R2 item 10)

### 13.1 Files

| File | Change |
|------|--------|
| `server/engine.py:63` | Accept `UnifiedPipelineConfig` backend fields; log resolved backend/device at startup. |
| `server/app.py` | Thread backend flags; propagate into pipeline load. |
| `server/cli.py` | `lazarus-serve` flags via central helper. |
| `server/routers/openai.py` | **Streaming**: SSE `ChatCompletionChunk` generator currently iterates `pipeline.stream_tokens`; dispatch inside that generator on `backend.name`. Torch arm uses `TextIteratorStreamer` from `transformers`. |
| `server/routers/anthropic.py` | Same SSE pattern; uses Anthropic event schema (`message_start`, `content_block_delta`, …). Backend dispatch at token source. |
| `server/routers/ollama.py` | Newline-delimited JSON stream; same token-source dispatch. |
| `server/schemas/*.py` | No change (wire schemas are framework-agnostic). |
| `client/*` | No change. |

### 13.2 Streaming contract

All three routers funnel into `ModelEngine.astream()` which already bridges
sync generator → async queue (`server/engine.py` class docstring). Engine's
sync `_stream_tokens` must dispatch on backend:

- MLX: existing generator, unchanged.
- Torch: spawn `TextIteratorStreamer` in a thread; yield tokens as they
  arrive. Cancellation maps to `streamer.on_finalized_text` + thread join.

Heartbeat, keepalive, and disconnect handling at the ASGI layer are unchanged.

**Tokenizer decode responsibility (R3 item 8):** on the torch arm,
`TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)`
owns the decode step internally — the engine yields **already-decoded
strings**, not token ids, so routers never call `tokenizer.decode` on torch
streamed output. MLX path keeps its current behaviour (engine yields ids;
router decodes). The asymmetry is deliberate and documented in
`ModelEngine.astream` docstring.

**Cancellation mechanism:** disconnection from the HTTP client sets a
`threading.Event` (`engine._cancel_event`) observed by the generation worker
thread, which breaks out of its `for tok in streamer` loop and calls
`streamer.end()`. We do **not** use `TextIteratorStreamer.on_finalized_text`
for cancellation — that hook fires only on normal completion. The event-based
path also covers MLX (the MLX generator loop checks the same event between
tokens).

Tests:
- `tests/server/test_engine_backend.py` (new)
- `tests/server/test_cli_backend.py` (new)
- `tests/server/routers/test_openai_stream_backend.py` (new)
- `tests/server/routers/test_anthropic_stream_backend.py` (new)
- `tests/server/routers/test_ollama_stream_backend.py` (new)

---

## 14. Bucket: `train` (sft / dpo / grpo / **ppo**) + `datagen` (R2 item 8)

### 14.1 Files

| File | Template |
|------|----------|
| `training/base_trainer.py:15-17` | B — top-level `mlx.core/nn/optimizers` → `TYPE_CHECKING`. Add `_forward_mlx/_forward_torch/_step_mlx/_step_torch` hooks. |
| `training/epoch_processor.py` | A |
| `training/epoch_processor_utils.py` | B |
| `training/batch_processor.py` | B (numpy-native) |
| `training/schedulers.py` | B |
| `training/classification_trainer.py` | A |
| `training/trainers/{sft,dpo,grpo,ppo,dual_reward}_trainer.py` | A — torch arm uses `torch.optim.AdamW` + `torch.nn.utils.clip_grad_norm_` + AMP per §2.3 |
| `training/losses/{sft,dpo,grpo,ppo,dual_reward}_loss.py` | A |
| `training/utils/{log_probs,kl_divergence,advantage}.py` | A |
| `utils/{optimizer_loader,optimizer_adapter,model_adapter}.py` | A |
| `cli/commands/train/{sft,dpo,grpo,datagen,_types}.py` | flags via §15 |

Snapshot resume across backends: **not supported in Epic 1b** (raise on
attempt). Converter tracked in Epic 3.

### 14.2 Tests (now includes PPO)

- `tests/training/test_base_trainer_backend.py` (new)
- `tests/training/trainers/test_sft_backend.py` (new)
- `tests/training/trainers/test_dpo_backend.py` (new)
- `tests/training/trainers/test_grpo_backend.py` (new)
- `tests/training/trainers/test_ppo_backend.py` (new — **added per R2 item 8**)
- `tests/training/trainers/test_dual_reward_backend.py` (new)
- `tests/training/losses/test_losses_backend.py` (new — covers all 5 loss fns)
- `tests/utils/test_optimizer_adapter_backend.py` (new)

---

## 15. Cross-Cutting Work Items

1. **Central flag helper** — `cli/commands/_base.py::add_backend_flags(parser)`;
   every subcommand parser mounts it via `cli/_parsers/__init__.py`.
   `cli/main.py` resolves flags → env overrides before dispatch.
2. **`DType.to_framework`** — see §7.1.
3. **`Backend.array`** — see §7.2.
4. **`pyproject.toml` pin reconciliation** — see §1.3.
5. **CI matrix** — `tests/ci/test_cuda_host_import_safety.py` (new) — iterates
   every `chuk_lazarus.*` submodule and asserts clean import under a simulated
   CUDA-only host (`CHUK_BACKEND=torch`, MLX uninstalled via pytest monkeypatch
   that injects a `sys.modules["mlx"] = None` sentinel).
6. **Docs** — update `README.md` install section (three extras: `mlx`,
   `torch`, `cuda`).
7. **01-spec sync** — forward rename note: `--skip-sm-check` → `--force-old-sm`,
   env `CHUK_SKIP_SM_CHECK` → `CHUK_FORCE_OLD_SM`.

---

## 16. MLX Preservation Checklist (applies per bucket)

- Apple Silicon default path (no env, no flags) lands on MLX.
- No behavioural change to MLX math: same kernels, dtypes, ordering.
- Existing golden tests pass unchanged.
- Module imports succeed on CUDA-only host (CI job §15.5).
- Un-ported features raise `NotImplementedError` with an Epic-2 or Epic-3
  doc anchor.

---

## 17. Deferred to Later Epics (with anchors)

| Feature | Epic | Anchor |
|---------|------|--------|
| Residual injection on torch | Epic 2 | `docs/refactor/dual-backend-cuda-epic2/00-scope.md#residual-inject` |
| kv_direct generator on torch | Epic 2 | `…#kv-direct` |
| Quantisation (bitsandbytes, AWQ) | Epic 2 | `…#quantisation` |
| `.mlxckpt` ↔ `.torchckpt` converter | Epic 3 | `docs/refactor/dual-backend-cuda-epic3/00-scope.md#ckpt-convert` |
| MLX-layout → HF safetensors converter | Epic 3 | `…#weights-convert` |
| Multi-GPU / FSDP / tensor-parallel | Epic 3 | `…#multi-gpu` |
| Cross-backend training resume | Epic 3 | `…#training-resume` |

Epic 2 / Epic 3 scope stub docs are a **deliverable of this epic (EWS-0)**,
landed at:
- `docs/refactor/dual-backend-cuda-epic2/00-scope.md` — anchors:
  `#residual-inject`, `#kv-direct`, `#quantisation`.
- `docs/refactor/dual-backend-cuda-epic3/00-scope.md` — anchors:
  `#ckpt-convert`, `#weights-convert`, `#multi-gpu`, `#training-resume`.

Both stubs are created at PR time for Epic 1b so that every `NotImplementedError`
message in Epic 1b code resolves to a live doc anchor. Later epics extend the
stubs in place (anchors are load-bearing — do not rename).

---

## 18. R3 Revision Cross-Reference

| R3 item | Addressed in |
|---------|--------------|
| 1 torch release/dev-host note | §1.2 note box |
| 2 per-trainer AMP matrix | §2.3.1 |
| 3 `.torchckpt` meta schema + forward-compat | §5.2 (full schema + reject rule) |
| 4 pyproject L37 verbatim diff | §1.3 (now shows diff blocks for all three extras) |
| 5 transformers × torch 2.9 compat check | §1.2.1 (bumped to 4.56.0 + accelerate 1.5.2) |
| 6 `Backend.array` → `from_numpy` | §7.2 (rewritten) + §11 table row |
| 7 Epic 2/3 stub deliverable | §17 + new `epic2/00-scope.md`, `epic3/00-scope.md` |
| 8 streaming decode + cancellation | §13.2 (appended) |
| 9 `test_force_old_sm.py` | §8 tests line |
| 10 `device_map=auto` rejection site | §4 (ingress = `UnifiedPipelineConfig` validator) |

## 19. R2 Revision Cross-Reference

| R2 item | Addressed in |
|---------|--------------|
| 1 torch pin | §1.2, §1.3 |
| 2 transformers/accelerate location & pins | §1.2, §1.3 |
| 3 mixed-precision policy | §2 |
| 4 KV serialisation & migration | §5.1–5.3 |
| 5 safetensors compat detection | §5.4 |
| 6 `CUDA_VISIBLE_DEVICES` vs `--cuda-device-id`, device_map | §4 |
| 7 line-ref verification | §6 |
| 8 PPO test path | §14.2 |
| 9 circuit ops | §12.2 |
| 10 server streaming | §13.2 |
| 11 NotImpl concrete anchors | §3 (Template C), §17 |
| 12 `to_framework` / `Backend.array` existence | §7 (marked NEW) |
| 13 `--force-old-sm` opt-in | §4.1 |
| 14 `Backend.save/load` justification | §7.3 (dropped) |
