# Epic: Dual-Backend CUDA — All-Buckets Command Matrix

## 0. Purpose & Reading Guide

This document is an **exhaustive inventory** of every user-visible CLI subcommand
exposed by the `chuk-lazarus` / `lazarus` / `lazarus-serve` / `circuit` console
scripts, classified by current and target CUDA readiness under the dual-backend
refactor.

### Console script map

| Script | Entry | Relationship |
|---|---|---|
| `chuk-lazarus` | `chuk_lazarus.cli:main` | Canonical top-level CLI. |
| `lazarus` | `chuk_lazarus.cli:main` | **Alias** — identical entry point to `chuk-lazarus`. Every `chuk-lazarus X` row below is callable as `lazarus X`. |
| `lazarus-serve` | `chuk_lazarus.server.cli:main` | **Distinct** script — OpenAI-compatible HTTP server. §6 covers it. |
| `circuit` | `chuk_lazarus.introspection.circuit.cli:main` | **Distinct** standalone script; overlaps with `introspect circuit *` but exposes its own parser tree. §14 enumerates it. |

Scope extension vs. the original Epic 1 spec (`../dual-backend-cuda/01-implementation-spec.md`,
`../dual-backend-cuda/02-workstreams.md`): Epic 1 covered `infer run` and the
`context prefill` / `context generate` hot paths only. This "all-buckets" epic
expands coverage to every command group while **preserving the MLX/Metal path
on every row** — no bucket is Torch-only; every row is `{mlx | torch}`-selectable
via `CHUK_BACKEND` + `--backend`.

### Status legend

| Status | Meaning |
|---|---|
| **works** | Already backend-agnostic today (no MLX import on hot path, or already gated behind `get_backend()`). |
| **MLX-hard-coded** | Module has top-level `import mlx` / `import mlx_lm` or calls `mlx.core.*` / `mx.array(...)` directly on the execution path — **must** be lifted before the torch path works. |
| **partial** | Most of the path is backend-agnostic but at least one chokepoint still forces MLX (e.g. a single `mx.array()` conversion, a Metal-only kernel, an MLX-specific cache class). |
| **torch-target** | Target state after this epic — same behaviour on MLX, and a working RTX 50-series CUDA path. |

### Authoritative chokepoints (verified against source at Epic 2 R2)

Each line below was opened and read before inclusion. Symptoms below match the
actual source text, not the original reviewer brief (the brief's summaries had
drifted against the current code).

| # | File | Line(s) | Actual content (verbatim / paraphrase) | Downstream impact |
|---|---|---:|---|---|
| C1 | `src/chuk_lazarus/inference/unified.py` | 583 | `if self._backend != LazarusBackend.MLX:` — explicit backend guard inside `UnifiedPipeline.make_engine()` that raises `NotImplementedError` on non-MLX. Not a tensor conversion; a **feature gate**. | Blocks `infer run --kv-direct`, `context generate --mode kv-inject`, `knowledge *` (they all call `make_engine()`). The torch path has no `KVDirectGenerator`; either implement one or gate-with-fallback to `extract_residual_state()` / `generate_with_residual()` (comment at L585-587 suggests the latter). |
| C2 | `src/chuk_lazarus/cli/commands/context/prefill/_cmd.py` | 24 | `import mlx.core as mx` inside `context_prefill_cmd()` (method-local, but unconditional). | Blocks `context prefill *` on non-MLX platforms because the import fires before any backend check. Fix: wrap in backend dispatch — `if backend.name == "mlx": import mlx.core as mx`. |
| C3 | `src/chuk_lazarus/cli/commands/context/generate/_cmd.py` | 87 | `import mlx.core as mx` inside `context_generate_cmd()` (method-local, unconditional). Same pattern as C2. | Blocks `context generate *`. Same fix. |
| C4a | `src/chuk_lazarus/cli/commands/knowledge/_common.py` | 7 | **Top-level** `import mlx.core as mx`. | Blocks `knowledge build/query/chat` at module import — even `--help` on a torch-only install fails. |
| C4b | `src/chuk_lazarus/cli/commands/knowledge/_common.py` | 24 | `_ = kv_gen.prefill(mx.array([[1, 2, 3]]))` inside `load_model()` (warm-up call, uses `mx` from the top-level import). | Same fix surface; move both to backend-dispatched helper. |
| C5 | `src/chuk_lazarus/introspection/hooks.py` | 421-427 | `backend = get_backend()` (L421) → `if backend.name != "mlx": raise NotImplementedError(...)` (L422-425) → method-local `import mlx.core as mx; import mlx.nn as nn` (L426-427) inside `ModelHooks.forward()`. Backend guard is **already present**; the TODO is to add the torch branch. | Blocks every `introspect *` command that exercises hooks (≈all of them). Fix: replace the `NotImplementedError` with a torch dispatch that calls backend-neutral tensor ops. |
| C6 | `src/chuk_lazarus/introspection/analyzer/core.py` | 16-17 | **Top-level** `import mlx.core as mx` + `import mlx.nn as nn`. | Blocks `introspect analyze/compare/hooks/ablate/weight-diff/activation-diff/layer/format` at module load. |
| C7 | `src/chuk_lazarus/server/engine.py` | 63 | `import mlx.core as mx` method-local inside the streaming generator, followed by `input_ids = mx.array(input_ids)` at L68. | Blocks `lazarus-serve` streaming + non-streaming (shared generator). |
| C8 | `src/chuk_lazarus/training/base_trainer.py` | 15-17 | **Top-level** `import mlx.core as mx` + `import mlx.nn as nn` + `import mlx.optimizers as optim`. | Blocks `train sft/dpo/grpo` (all trainers import the base). |

Every row below that cites a C# reference inherits its fix from that workstream.
The **MLX/Metal path is preserved on every row** by keeping MLX imports
method-local and dispatching through `models_v2.core.backend.get_backend()`.

### MLX-preservation convention (canonical phrasing)

To keep every row's MLX-preservation claim unambiguous, **exactly one** of the
following two sentences appears in each row's Notes cell:

1. **`MLX path preserved via get_backend("mlx") dispatch.`** — used on every
   row with Current status `MLX-hard-coded` or `partial` (i.e. Target
   `torch-target`). The existing MLX code path is wrapped, not rewritten.
2. **`No MLX path exists; backend-agnostic.`** — used on every `works → works`
   row. There is nothing to preserve because the command never touches MLX.

Per-row chokepoint references (e.g. "C5"), audit findings, and fix hints appear
alongside the canonical sentence in the same cell.

**Default rule (applies to every row in this document):** any `torch-target`
row whose Notes cell does NOT explicitly contain sentence (2) inherits
sentence (1) by default — the chokepoint fix for that row preserves the MLX
path by wrapping the existing MLX code behind `get_backend("mlx")` rather
than rewriting it. This default is stated explicitly so that tables with many
rows sharing the same C-reference (notably §5 and §5.1) do not need to repeat
the canonical sentence in every cell. Readers can verify works-row coverage
by grepping for sentence (2).

---

## 1. `infer`

Console script: `chuk-lazarus infer …` (parser: `cli/_parsers/_infer.py`).

| Subcommand | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `infer run` (standard) | `cli/commands/infer/run.py` | partial | torch-target | Epic 1 WS-4/WS-6 plumbs `--backend {mlx,torch}` + `--device` through `UnifiedPipelineConfig`. Non-KV path is backend-agnostic once `inference/unified.py` tensor conversions (outside C1) are lazy. MLX path preserved via `get_backend("mlx")` dispatch. |
| `infer run --kv-direct` (kv_direct mode) | `cli/commands/infer/run.py` (kv_direct branch) → `inference/unified.py::make_engine()` + `inference/context/kv_generator.py` | MLX-hard-coded | torch-target | Hits chokepoint **C1** — `make_engine()` raises on non-MLX. Target: implement a torch `KVDirectGenerator` adapter OR document the `extract_residual_state()` / `generate_with_residual()` alternative that L585-587 already points to. MLX path untouched. |

---

## 2. `context prefill`

Console script: `chuk-lazarus context prefill …` (parser: `cli/_parsers/_context.py:17`).

| Subcommand | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `context prefill` (dispatcher / all submodes) | `cli/commands/context/prefill/_cmd.py` | MLX-hard-coded | torch-target | Chokepoint **C2**. Lift the `mx` import behind a backend check. MLX path preserved via `get_backend("mlx")` dispatch. |
| `context prefill` (vector injection submode) | `cli/commands/context/prefill/_vec_inject.py` | MLX-hard-coded | torch-target | Epic 1 WS-5 already owned this file; reuse the WS-5 backend-dispatch pattern. MLX path preserved. |
| `context prefill` helpers — `_sparse.py`, `_compass.py`, `_darkspace.py`, `_interval.py`, `_surprise.py`, `_pages.py`, `_checkpoints.py`, `_restore.py`, `_save.py`, `_npz.py`, `_mode7_calibrate.py`, `_kv_route.py`, `_progress.py` | Respective `cli/commands/context/prefill/_*.py` | audited — see below | torch-target | Per-file audit: all 13 helpers use numpy / stdlib only; **no direct `mx.*` calls**. They inherit MLX only transitively through `_cmd.py`'s imported `mx` handle. Once C2 is lifted they fall through unchanged. MLX path preserved via `get_backend("mlx")` dispatch (helpers don't touch tensors directly). |

---

## 3. `context generate`

Console script: `chuk-lazarus context generate …` (parser: `cli/_parsers/_context.py:99`).

| Subcommand / mode | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `context generate` (dispatcher) | `cli/commands/context/generate/_cmd.py` | MLX-hard-coded | torch-target | Chokepoint **C3** (method-local unconditional `import mlx.core as mx`). Lift behind backend check. MLX path preserved via `get_backend("mlx")` dispatch. |
| `context generate --mode standard` | `_modes/_standard.py` | partial | torch-target | Audit: module has no `import mlx`; calls `pipeline.generate(...)` only. Pure pass-through once C1/C3 are fixed. |
| `context generate --mode broad` | `_modes/_broad.py` | partial | torch-target | Audit: no `mx.*` calls; pure pipeline wrapper. |
| `context generate --mode explore` | `_modes/_explore.py` | partial | torch-target | Audit: no `mx.*` calls. |
| `context generate --mode compressed` | `_modes/_compressed.py` | partial | torch-target | Audit: no `mx.*` calls. |
| `context generate --mode accumulated` | `_modes/_accumulated.py` | partial | torch-target | Audit: no `mx.*` calls. |
| `context generate --mode inject` | `_modes/_inject.py` | partial | torch-target | Audit: calls `pipeline.extract_residual_state` + `pipeline.generate_with_residual` — both already backend-aware (`unified.py:592-600`). Clean once C3 fires. |
| `context generate --mode kv-inject` | `_modes/_kv_inject.py` | MLX-hard-coded | torch-target | Calls `pipeline.make_engine()` → hits **C1**. Shares fix with `infer --kv-direct`. |
| `context generate --mode sparse` / `sparse-twopass` | `_modes/_sparse.py`, `_modes/_sparse_twopass.py` | partial | torch-target | Audit: numpy-only index ops; no `mx.*`. Clean after C3. |
| `context generate` (probe-driven / probe-rerank / plain / grounding / iterative) | `_probe_driven.py`, `_probe_rerank.py`, `_plain.py`, `_grounding.py`, `_iterative.py` | partial | torch-target | Audit: no direct MLX imports. Inherit C3 only. |
| `context generate` (mode 7 unified path) | `_unified.py` + `_mode7.py` + `_probes.py` | partial | torch-target | Mode-7 uses the unified pipeline generation loop; depends on C1 only for kv-inject sub-mode, clean otherwise. MLX path preserved via `get_backend("mlx")` dispatch. |
| `context calibrate-frames` | `cli/commands/context/calibrate_frames.py` | partial | torch-target | Calls `make_engine()` → inherits C1 when invoked with `--mode kv-direct`; backend-agnostic for the residual-state path. |

---

## 4. `knowledge`

Console script: `chuk-lazarus knowledge …` (parser: `cli/_parsers/_knowledge.py`).

| Subcommand | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `knowledge build` | `cli/commands/knowledge/_build.py` | MLX-hard-coded | torch-target | Imports `_common.load_model` → inherits **C4a** (top-level `import mlx.core as mx`) + **C4b** (`mx.array([[1,2,3]])` warm-up). Fix: move top-level import behind `get_backend()` dispatch in `_common.py`; warm-up becomes `backend.array(...)`. MLX path preserved via `get_backend("mlx")` dispatch. |
| `knowledge query` | `cli/commands/knowledge/_query.py` | MLX-hard-coded | torch-target | Same chokepoints. |
| `knowledge chat` | `cli/commands/knowledge/_chat.py` | MLX-hard-coded | torch-target | Same chokepoints + `pipeline.make_engine()` call path → also inherits **C1** if the chat loop uses KVDirect. |

---

## 5. `introspect`

Console script: `chuk-lazarus introspect …` (parser: `cli/_parsers/_introspect/`).

All rows below inherit **C5** (`introspection/hooks.py:421-427`) because every
handler eventually constructs a `ModelHooks` instance. Rows that also inherit
**C6** (`analyzer/core.py:16-17`) are called out because those modules fail at
import time, not call time.

| Subcommand | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `introspect analyze` | `cli/commands/introspect/analyze.py` | MLX-hard-coded | torch-target | C5 + C6. |
| `introspect compare` | `cli/commands/introspect/analyze.py` (compare branch) | MLX-hard-coded | torch-target | C5 + C6. |
| `introspect hooks` | `cli/commands/introspect/analyze.py` (hooks branch) | MLX-hard-coded | torch-target | C5 + C6. |
| `introspect ablate` | `cli/commands/introspect/ablation.py` | MLX-hard-coded | torch-target | C5. Weight manipulation uses `mx.array`. |
| `introspect weight-diff` | `cli/commands/introspect/ablation.py` (weight-div) | MLX-hard-coded | torch-target | C5. |
| `introspect activation-diff` | `cli/commands/introspect/ablation.py` (activation-div) | MLX-hard-coded | torch-target | C5. |
| `introspect layer` | `cli/commands/introspect/layer.py` | MLX-hard-coded | torch-target | C5. |
| `introspect format` | `cli/commands/introspect/layer.py` (format branch) | MLX-hard-coded | torch-target | C5. |
| `introspect generate` | `cli/commands/introspect/generation.py` | MLX-hard-coded | torch-target | C5 + C1. |
| `introspect metacog` | `cli/commands/introspect/generation.py` (metacog) | MLX-hard-coded | torch-target | C5 + C1. |
| `introspect steer` | `cli/commands/introspect/steering.py` | MLX-hard-coded | torch-target | C5. |
| `introspect arithmetic` | `cli/commands/introspect/arithmetic.py` | MLX-hard-coded | torch-target | C5. |
| `introspect uncertainty` | `cli/commands/introspect/arithmetic.py` (uncertainty) | MLX-hard-coded | torch-target | C5. |
| `introspect probe` | `cli/commands/introspect/probing.py` | MLX-hard-coded | torch-target | C5. |
| `introspect neurons` | `cli/commands/introspect/probing.py` (neurons) | MLX-hard-coded | torch-target | C5. |
| `introspect cluster` | `cli/commands/introspect/clustering.py` | MLX-hard-coded | torch-target | C5. |
| `introspect memory` | `cli/commands/introspect/memory.py` | MLX-hard-coded | torch-target | C5. |
| `introspect inject` | `cli/commands/introspect/memory.py` (inject) | MLX-hard-coded | torch-target | C5. |
| `introspect directions` | Parser `_introspect/_directions.py`; handler `introspect/embedding.py` | MLX-hard-coded | torch-target | C5. |
| `introspect operand-directions` | Same | MLX-hard-coded | torch-target | C5. |
| `introspect embedding` | `cli/commands/introspect/embedding.py` | MLX-hard-coded | torch-target | C5. |
| `introspect commutativity` | `cli/commands/introspect/embedding.py` (commutativity) | MLX-hard-coded | torch-target | C5. |
| `introspect early-layers` | `cli/commands/introspect/embedding.py` (early-layers) | MLX-hard-coded | torch-target | C5. |
| `introspect patch` | `cli/commands/introspect/patching.py` | MLX-hard-coded | torch-target | C5. |
| `introspect circuit capture` | `cli/commands/introspect/circuit.py` (capture) | MLX-hard-coded | torch-target | C5. |
| `introspect circuit invoke` | `cli/commands/introspect/circuit.py` (invoke) | MLX-hard-coded | torch-target | C5 + C1. |
| `introspect circuit decode` | `cli/commands/introspect/circuit.py` (decode) | MLX-hard-coded | torch-target | C5. |
| `introspect circuit test` | `cli/commands/introspect/circuit.py` (test) | MLX-hard-coded | torch-target | C5. |
| `introspect circuit compare` | `cli/commands/introspect/circuit.py` (compare) | MLX-hard-coded | torch-target | C5. |
| `introspect circuit view` | `cli/commands/introspect/circuit.py` (view) | partial | torch-target | Rendering/JSON only; clean once upstream tensors land as numpy via backend dispatch. |
| `introspect circuit export` | `cli/commands/introspect/circuit.py` (export) | partial | torch-target | Serialization; backend-agnostic. |
| `introspect virtual-expert` | `cli/commands/introspect/virtual_expert.py` | MLX-hard-coded | torch-target | C5 + shares `inference/virtual_experts/*` MLX plumbing. |
| `introspect classifier` | `cli/commands/introspect/classifier.py` | MLX-hard-coded | torch-target | C5. |
| `introspect logit-lens` | `cli/commands/introspect/classifier.py` (logit-lens) | MLX-hard-coded | torch-target | C5. |

### 5.1 `introspect moe-expert …` handlers (enumerated)

The `moe-expert` dispatcher fans out to **30 handler files** (plus
`__init__.py`) under `cli/commands/introspect/moe_expert/handlers/`. The table
below enumerates all 30. All share chokepoint **C5**
because every handler instantiates `ModelHooks`; most additionally use
`mx.array` / `mx.*` directly. Table below enumerates each.

| Subcommand | Handler file | Current | Target | Notes |
|---|---|---|---|---|
| `introspect moe-expert ablate` | `moe_expert/handlers/ablate.py` | MLX-hard-coded | torch-target | C5 + direct `mx.*` weight ops. |
| `introspect moe-expert analyze` | `moe_expert/handlers/analyze.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert attention-pattern` | `moe_expert/handlers/attention_pattern.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert attention-prediction` | `moe_expert/handlers/attention_prediction.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert attention-routing` | `moe_expert/handlers/attention_routing.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert chat` | `moe_expert/handlers/chat.py` | MLX-hard-coded | torch-target | C5 + C1. |
| `introspect moe-expert cold-experts` | `moe_expert/handlers/cold_experts.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert compare` | `moe_expert/handlers/compare.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert context-attention-routing` | `moe_expert/handlers/context_attention_routing.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert context-test` | `moe_expert/handlers/context_test.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert context-window` | `moe_expert/handlers/context_window.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert domain-test` | `moe_expert/handlers/domain_test.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert expert-circuits` | `moe_expert/handlers/expert_circuits.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert expert-interference` | `moe_expert/handlers/expert_interference.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert expert-merging` | `moe_expert/handlers/expert_merging.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert explore` | `moe_expert/handlers/explore.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert full-taxonomy` | `moe_expert/handlers/full_taxonomy.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert generation-dynamics` | `moe_expert/handlers/generation_dynamics.py` | MLX-hard-coded | torch-target | C5 + C1. |
| `introspect moe-expert heatmap` | `moe_expert/handlers/heatmap.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-overlay-compress` | `moe_expert/handlers/moe_overlay_compress.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-overlay-compute` | `moe_expert/handlers/moe_overlay_compute.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-overlay-estimate` | `moe_expert/handlers/moe_overlay_estimate.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-overlay-verify` | `moe_expert/handlers/moe_overlay_verify.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-type-analyze` | `moe_expert/handlers/moe_type_analyze.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert moe-type-compare` | `moe_expert/handlers/moe_type_compare.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert routing-manipulation` | `moe_expert/handlers/routing_manipulation.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert task-prediction` | `moe_expert/handlers/task_prediction.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert token-routing` | `moe_expert/handlers/token_routing.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert trace` | `moe_expert/handlers/trace.py` | MLX-hard-coded | torch-target | C5. |
| `introspect moe-expert weights` | `moe_expert/handlers/weights.py` | MLX-hard-coded | torch-target | C5 + direct `mx.*` weight ops. |

**Scope note:** the `introspection/**` subtree outside `hooks.py` is ~100 files
still carrying top-level `import mlx`. The matrix above lists user-visible
commands; the implementation spec (Task #6) partitions the internal files into
lazy-load workstreams. MLX/Metal path preserved on every row.

---

## 6. `lazarus-serve`

Console script: `lazarus-serve` (entry: `chuk_lazarus.server.cli:main`). **Not**
a subcommand of `chuk-lazarus`; it is its own script. No aliases.

| Route / mode | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `POST /v1/completions` (non-streaming) | `src/chuk_lazarus/server/engine.py` (shared generator) | MLX-hard-coded | torch-target | Chokepoint **C7**. Dispatch token loop via `get_backend().generate(...)`. MLX path preserved via `get_backend("mlx")` dispatch (calls the current MLX loop when `backend.name == "mlx"`). |
| `POST /v1/chat/completions` (streaming SSE) | `src/chuk_lazarus/server/engine.py` (stream branch) + `server/routers/*.py` | MLX-hard-coded | torch-target | Same C7 (stream branch uses same generator). |
| `GET /v1/models`, `/healthz`, admin routes | `src/chuk_lazarus/server/routers/*.py` | works | works | **No MLX path exists; backend-agnostic** (no tensor ops, only metadata + FastAPI plumbing). |

---

## 7. `train`

Console script: `chuk-lazarus train …` (parser: `cli/_parsers/_train.py`).

| Subcommand | Entry file | Current CUDA status | Target CUDA status | Notes |
|---|---|---|---|---|
| `train sft` | `cli/commands/train/sft.py` | MLX-hard-coded | torch-target | Inherits **C8** (`training/base_trainer.py:15-17` top-level `mx`/`nn`/`optim`). Needs backend-dispatched optimizer + loss-step. MLX optimizer path preserved via backend dispatch. |
| `train dpo` | `cli/commands/train/dpo.py` | MLX-hard-coded | torch-target | C8. |
| `train grpo` | `cli/commands/train/grpo.py` | MLX-hard-coded | torch-target | C8 + sampling path through `unified.py` (may also hit C1 when KVDirect is used for rollouts). |
| `train datagen` | `cli/commands/train/datagen.py` | partial | torch-target | Mostly data pipelining; the sample-generation step calls `UnifiedPipeline.generate()` → backend-agnostic once the non-KV generation path is torch-clean. |

---

## 8. `generate` (cross-reference)

There is **no top-level `generate` subcommand**; the brief's label maps to
`train datagen` (dataset generation, §7) and `context generate` (context
generation, §3). Listed here as N/A — see those sections.

---

## 9. `data`

Console script: `chuk-lazarus data …` (parser: `cli/_parsers/_data.py`).

| Subcommand | Entry file | Current | Target | MLX-preservation note |
|---|---|---|---|---|
| `data lengths build` | `cli/commands/data/lengths/build.py` | works | works | **No MLX path exists; backend-agnostic** (tokenizer + numpy). |
| `data lengths stats` | `cli/commands/data/lengths/stats.py` | works | works | Same. |
| `data batchplan build` | `cli/commands/data/batchplan/build.py` | works | works | Same (pure data). |
| `data batchplan info` | `cli/commands/data/batchplan/info.py` | works | works | Same. |
| `data batchplan verify` | `cli/commands/data/batchplan/verify.py` | works | works | Same. |
| `data batchplan shard` | `cli/commands/data/batchplan/shard.py` | works | works | Same. |
| `data batching analyze/histogram/suggest/generate` | `cli/commands/data/batching/*.py` | works | works | Same. |
| `data batch generate` | `cli/commands/data/batching/generate.py` | works | works | Same. |

---

## 10. `tokenizer`

Console script: `chuk-lazarus tokenizer …` (parser: `cli/_parsers/_tokenizer.py`).

| Subcommand | Entry file | Current | Target | MLX-preservation note |
|---|---|---|---|---|
| `tokenizer encode` | `tokenizer/core/encode.py` | works | works | **No MLX path exists; backend-agnostic** (HF/sentencepiece only). |
| `tokenizer decode` | `tokenizer/core/decode.py` | works | works | Same. |
| `tokenizer vocab` | `tokenizer/core/vocab.py` | works | works | Same. |
| `tokenizer compare` | `tokenizer/core/compare.py` | works | works | Same. |
| `tokenizer doctor` | `tokenizer/health/doctor.py` | works | works | Same. |
| `tokenizer fingerprint` | `tokenizer/health/fingerprint.py` | works | works | Same. |
| `tokenizer benchmark` | `tokenizer/health/benchmark.py` | works | works | Same. |
| `tokenizer analyze coverage/diff/efficiency/entropy/fit-score/vocab-suggest` | `tokenizer/analyze/*.py` | works | works | Same. |
| `tokenizer curriculum length-buckets/reasoning` | `tokenizer/curriculum/*.py` | works | works | Same. |
| `tokenizer instrument histogram/oov/vocab-diff/waste` | `tokenizer/instrument/*.py` | works | works | Same. |
| `tokenizer regression run` | `tokenizer/regression/run.py` | works | works | Same. |
| `tokenizer research embeddings` | `tokenizer/research/embeddings.py` | partial | torch-target | Triggered specifically by `tokenizer research embeddings --model <id>` (model-loading path); calls into `UnifiedPipeline.from_pretrained()` + embedding extraction. Inherits generation path (non-KV, no C1). MLX path preserved via backend dispatch. The `--model` flag is the trigger; without `--model` the command operates on a precomputed embedding file and remains **works**. |
| `tokenizer research morph` | `tokenizer/research/morph.py` | works | works | **No MLX path exists; backend-agnostic** (morphology analysis, no model loading). |

---

## 11. `gym`

Console script: `chuk-lazarus gym …` (parser: `cli/_parsers/_gym.py`).

| Subcommand | Entry file | Current | Target | Notes |
|---|---|---|---|---|
| `gym run` | `cli/commands/gym/run.py` | partial | torch-target | Wraps the unified pipeline; non-KV path clean, KV path hits C1. MLX path preserved via `get_backend("mlx")` dispatch. |
| `gym info` | `cli/commands/gym/info.py` | works | works | **No MLX path exists; backend-agnostic** (metadata only). |
| `gym benchmark` | `cli/commands/gym/benchmark.py` | partial | torch-target | Same as `gym run`. |

---

## 12. `experiment`

Console script: `chuk-lazarus experiment …` (parser: `cli/_parsers/_experiment.py`).

| Subcommand | Entry file | Current | Target | Notes |
|---|---|---|---|---|
| `experiment list` | `cli/commands/experiment/handlers.py` (list) | works | works | **No MLX path exists; backend-agnostic** (discovery). |
| `experiment info` | `cli/commands/experiment/handlers.py` (info) | works | works | Same (metadata). |
| `experiment run` | `cli/commands/experiment/handlers.py` (run) | partial | torch-target | Dispatches to any other bucket; backend selection flows through the called command. MLX path preserved via `get_backend("mlx")` dispatch (per called command). |
| `experiment status` | `cli/commands/experiment/handlers.py` (status) | works | works | **No MLX path exists; backend-agnostic** (file-system state). |

---

## 13. `bench`

Console script: `chuk-lazarus bench …` (parser: `cli/_parsers/_bench.py`). No
`cli/commands/bench/` tree — the parser calls inference directly.

| Subcommand | Entry | Current | Target | Notes |
|---|---|---|---|---|
| `bench` (throughput / latency sweep) | `cli/_parsers/_bench.py` → `inference/unified.py` | partial | torch-target | Non-KV path clean; KV path hits C1. Must add `--backend` flag, mirroring `infer run`. MLX path preserved via `get_backend("mlx")` dispatch. |

---

## 14. `circuit` (standalone console script)

Console script: `circuit` → `chuk_lazarus.introspection.circuit.cli:main` (parser
tree defined inline in `introspection/circuit/cli.py`). **Distinct** from the
`chuk-lazarus introspect circuit *` family: the standalone `circuit` script is
an older interface kept for `probe_datasets/` workflows. All rows inherit
chokepoint **C5** (every handler uses `ModelHooks` via `collector.py`) and most
inherit top-level MLX imports in the service / collector modules.

| Subcommand | Entry file | Current | Target | Notes |
|---|---|---|---|---|
| `circuit dataset create` | `introspection/circuit/cli.py` (`dataset create` branch) → `dataset.py` | MLX-hard-coded | torch-target | Dataset module imports MLX for tokenization shim; lift lazily. MLX path preserved via `get_backend("mlx")` dispatch. |
| `circuit dataset show` | `introspection/circuit/cli.py` (`dataset show` branch) → `dataset.py` | works | works | **No MLX path exists on the show branch; backend-agnostic** (pretty-print JSONL). |
| `circuit collect` | `introspection/circuit/cli.py` (collect) → `collector.py` | MLX-hard-coded | torch-target | C5 (constructs `ModelHooks`) + direct `mx.array` in collector. |
| `circuit analyze` | `introspection/circuit/cli.py` (analyze) → `geometry.py` | MLX-hard-coded | torch-target | Uses MLX for covariance / PCA; replace with backend-neutral numpy or `backend.linalg`. |
| `circuit directions` | `introspection/circuit/cli.py` (directions) → `directions.py` | MLX-hard-coded | torch-target | MLX linear algebra; same fix pattern. |
| `circuit visualize` | `introspection/circuit/cli.py` (visualize) → `export.py` | partial | torch-target | Rendering only; backend-agnostic once upstream arrays are numpy. |
| `circuit steer` | `introspection/circuit/cli.py` (steer, experimental) → `service.py` | MLX-hard-coded | torch-target | C5 + generation path (C1 when KV-direct used). |
| `circuit probes run` | `introspection/circuit/cli.py` (`probes run`) → `probes.py` | MLX-hard-coded | torch-target | C5. |
| `circuit probes init` | `introspection/circuit/cli.py` (`probes init`) → `probes.py` (dataset scaffolding) | works | works | **No MLX path exists; backend-agnostic** (writes JSON templates from `probe_datasets/`). |

---

## 15. Summary: bucket → primary fix

| Bucket | Primary chokepoint(s) | Workstream owner (forward ref to Task #7 workstream doc) |
|---|---|---|
| `infer run` standard | (non-KV generation path in `unified.py`, outside C1) | Epic 1 WS-4 (already scoped) |
| `infer run --kv-direct`, `context generate --mode kv-inject`, `calibrate-frames` | **C1** `unified.py:583` (backend guard in `make_engine()`) | Epic 2 WS-A |
| `context prefill *` | **C2** `prefill/_cmd.py:24` | Epic 1 WS-5 (already scoped) |
| `context generate *` | **C3** `generate/_cmd.py:87` (+ C1 for kv-inject) | Epic 2 WS-A |
| `knowledge *` | **C4a/C4b** `knowledge/_common.py:7,24` | Epic 2 WS-B |
| `introspect *`, `introspect moe-expert *`, standalone `circuit *` | **C5** `hooks.py:421-427` + **C6** `analyzer/core.py:16-17` | Epic 2 WS-C (largest stream) |
| `lazarus-serve` | **C7** `server/engine.py:63` | Epic 2 WS-D |
| `train {sft,dpo,grpo,datagen}` | **C8** `training/base_trainer.py:15-17` | Epic 2 WS-E |
| `bench`, `gym`, `tokenizer research embeddings --model …`, `experiment run` | Transitive on non-KV generation path | Close out automatically with Epic 1 WS-4. |
| `data *`, most `tokenizer *`, admin server routes, `experiment list/info/status`, `circuit probes init`, `circuit dataset show` | N/A — no MLX path exists | No change. |
| `generate` (brief term) | N/A — not a real command; maps to `context generate` / `train datagen` | See §8. |

**MLX/Metal path invariant:** every "torch-target" row above is a dual-backend
row — MLX selection via `CHUK_BACKEND=mlx` (or `--backend mlx`) must continue
to produce byte-identical behaviour to `main` on macOS post-refactor. The
workstream documents (Task #7) and validation matrix (Task #8) encode this as
a mandatory regression gate.
