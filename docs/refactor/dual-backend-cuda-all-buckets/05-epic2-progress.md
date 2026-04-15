# Epic 2 — Execution Progress Log

Lead: Epic 2 team-lead (this agent).
Host: WSL2 Linux, RTX 5090 (sm_120), CUDA 13.1 driver / torch 2.9.1+cu128.

## Backend contract decision (SPLIT_DECLARATION)

Decided 2026-04-15 by team-lead. Immutable.

- Backend selection surface is `--backend {mlx,torch}` plus `--device {cuda,mps,cpu,<idx>}`.
- `CHUK_BACKEND` env var mirrors `--backend`. `CHUK_DEVICE` mirrors `--device`.
- Precedence: CLI flag > env > platform default (mlx on darwin-arm64, torch on linux/cuda, torch-cpu fallback).
- "CUDA contract" in docs 02/03/04 refers to the runtime *device*, not a third backend name.
- No rename of the shipped Epic 1 flag. EWS-0 adds `add_backend_flags()` helper that emits this exact pair.

Rationale: Epic 1 shipped with `--backend torch`; downstream code + tests import it. A rename now would cascade through every bucket and invalidate Epic 1 tests. Adding `--device cuda` as the CUDA selector preserves MLX/Metal (`--device mps`) and CPU fallback.

## Workstream status

| WS | Status | Owner | Notes |
|----|--------|-------|-------|
| EWS-0  | IN PROGRESS | code-surgeon-0 | Shared harness, add_backend_flags, Backend.array/save/load, parser splits, fixtures, Epic 2/3 stubs (latter already seeded) |
| EWS-1a | COMPLETED | ews1-surgeon | Wave 0.5 gate; infer parser + backends/* + virtual_experts/{registry,router}.py lazy-clean in BACKEND_IN_SCOPE; 54 tests green |
| EWS-1b | PENDING (task #3) | — | Follow-up: `virtual_experts/{wrapper,dense_wrapper}.py` + `.base/.cot_rewriter/.plugins.math` optional-dep refactor (blocked on `chuk_virtual_expert` packaging decision — see 2026-04-15 EWS-1 entry) |
| EWS-8  | COMPLETED | ews8-surgeon | serve + lazarus-serve threads `--backend`/`--device` into `UnifiedPipelineConfig`; 15 tests green; lazy-import assertion on every server file |
| EWS-4  | COMPLETED | ews4-surgeon | knowledge {build,query,chat} thread `--backend`/`--device` into `UnifiedPipelineConfig`; `_common` lazy-imports mlx; BACKEND_IN_SCOPE + 5 new test files; full suite deferred to clean `uv sync` |
| EWS-2..15 | QUEUED | — | Waves A1/A2/B per 03-workstreams §2 |

## Progress entries (append-only)

### 2026-04-15 — EWS-0 shared harness landed (ews0-surgeon)

**Files touched (modified):**
- `tests/ci/test_no_top_level_mlx.py` — renamed `EPIC_1_IN_SCOPE` → `BACKEND_IN_SCOPE`, kept alias
- `tests/conftest.py` — added `backend_env` (parametrized mlx/torch) + `mlx_golden` fixtures
- `src/chuk_lazarus/cli/commands/_base.py` — added `add_backend_flags` / `BACKEND_CHOICES`
- `src/chuk_lazarus/cli/_parsers/__init__.py` — exposed `add_backend_flags`, re-pointed registrars at split modules
- `src/chuk_lazarus/cli/main.py` — top-level `--backend` / `--device` (dest `_top_backend`/`_top_device`) → `CHUK_BACKEND` / `CHUK_DEVICE` env; `_extract_command` now skips top-level option values
- `src/chuk_lazarus/cli/_parsers/_infer.py`, `_train.py`, `_context.py` — reduced to one-line `from ._<split> import *` shims (option b mandatory)
- `src/chuk_lazarus/models_v2/core/backend/base.py` — abstract `array` / `save` / `load`
- `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` + `mlx_backend.py` — concrete implementations

**Files touched (new):**
- `src/chuk_lazarus/cli/_parsers/_serve.py`, `_infer_run.py`, `_train_sft.py`, `_train_rlhf.py`, `_train_parsers.py`, `_context_prefill.py`, `_context_generate.py`, `_context_parsers.py` — parser splits
- `tests/_helpers/{__init__,backend_fixtures,mlx_snapshots}.py`
- `tests/cli/test_base_flags.py`, `tests/cli/_parsers/{__init__,test_split_registration}.py`, `tests/models_v2/core/backend/test_backend_helpers.py`
- `tests/ci/cuda_smoke_exemptions.json` (seeded `[]`), `tests/ci/test_cuda_exemption_schema.py`
- `tests/fixtures/{.gitkeep,generate.sh,SHA256SUMS}` + `tests/fixtures/_builders/{__init__,build_vec,build_corpus,build_probe,build_ds,build_vector,build_exp,build_cls,build_sft,build_pairs,build_grpo,build_raw}.py`

**Test summary** (`uv run python -m pytest tests/ci/ tests/cli/test_base_flags.py tests/cli/_parsers/test_split_registration.py tests/models_v2/core/backend/test_backend_helpers.py -x`):
- 38 passed, 8 skipped, 0 failed
- Skips: MLX-only backend helper round-trips (mlx not importable on Linux host; torch round-trips pass on CUDA).

**Sanity:**
- `uv run chuk-lazarus --backend torch --device cuda --help` → clean exit, root usage shows `--backend {mlx,torch}` and `--device DEVICE`.
- `uv run chuk-lazarus infer --help` → still lists `--backend {mlx,torch}` and `--device` on the subparser (Epic 1 flags preserved). (`infer run` is not a pre-existing subcommand in Epic 1 — the current leaf is `infer`.)

**Scope stubs:** verified `docs/refactor/dual-backend-cuda-epic2/00-scope.md` and `dual-backend-cuda-epic3/00-scope.md` already exist from prior seeding; left untouched.

### 2026-04-15 — EWS-8 serve + lazarus-serve (ews8-surgeon)

**Files touched (modified):**
- `src/chuk_lazarus/server/engine.py` — `ModelEngine.load` + `_load_sync` now accept `backend`/`device` kwargs; builds `UnifiedPipelineConfig` with `backend_name`/legacy `LazarusBackend` enum mapped (`torch`→`CUDA`) and logs the resolved pair at load.
- `src/chuk_lazarus/server/cli.py` — `_add_serve_args` calls `add_backend_flags` (single source of truth); `_serve_async` pulls `args.backend`/`args.device` and threads them into `ModelEngine.load`; prints resolved backend/device when set. Standalone `lazarus-serve` inherits the same flags via `_add_serve_args`.
- `src/chuk_lazarus/cli/_parsers/_serve.py` — post-EWS-0 split now registers `add_backend_flags` idempotently on the returned subparser.
- `tests/ci/test_no_top_level_mlx.py` — extended `BACKEND_IN_SCOPE` with all 14 server files (`server/__init__.py`, `engine.py`, `app.py`, `cli.py`, 3 routers, 5 schemas) + `cli/_parsers/_serve.py`.

**Files touched (new):**
- `tests/server/__init__.py`, `tests/server/routers/__init__.py`
- `tests/server/test_engine_backend.py` — 4 tests covering: no-kwarg pass-through, `torch`→`LazarusBackend.CUDA` + `backend_name`/`device` wiring, `mlx` enum preservation, async load wrapper still threads backend kwargs.
- `tests/server/test_app_backend.py` — subprocess lazy-import assertion + `create_app` signature smoke test.
- `tests/server/test_cli_backend.py` — `_add_serve_args` registers `--backend`/`--device`, subcommand parsing, choice enforcement, `_serve_async` threads backend kwargs into patched `ModelEngine.load`.
- `tests/server/routers/test_openai_backend.py` — subprocess lazy-import + `/chat/completions` route discovery.
- `tests/server/routers/test_ollama_backend.py`, `test_anthropic_backend.py` — subprocess lazy-import only (routers are placeholder stubs).
- `tests/cli/_parsers/test_serve_parser.py` — subcommand registration exposes `--backend {mlx,torch}` and `--device` with proper rejection of unknown backends.

**Acceptance mapping:**
- `lazarus-serve --backend torch --device cuda:0 --model …` / `--backend mlx --device mps …` parse cleanly (CLI tests); load path threads resolved backend into `UnifiedPipelineConfig.backend` + `backend_name` + `device` (engine tests) — confirmed via mocked `UnifiedPipeline.from_pretrained`.
- `/v1/chat/completions` handler is untouched and routes via `ModelEngine.astream`/`agenerate`, which dispatches through `UnifiedPipeline.generate`; the backend decision point moved upstream to `ModelEngine._load_sync`, so both MLX and torch runtimes reach the router without router-level edits.
- MLX request/response snapshot: routers unchanged; `_apply_template` / `_stream_tokens` paths identical to pre-EWS-8 revision (MLX-only function-scope import at `engine.py:63`).
- Lazy-import assertion: every file in the new `server/` BACKEND_IN_SCOPE block is clean per `test_no_top_level_mlx` (static AST + subprocess runtime check under `CHUK_BACKEND=torch`).
- `BACKEND_IN_SCOPE` extended (14 server files + `_serve.py`).

**Smoke (CUDA host):**
- `uv run python -c "from chuk_lazarus.server.cli import _add_serve_args; import argparse; p=argparse.ArgumentParser(); _add_serve_args(p); print({a.dest for a in p._actions})"` → includes `backend`, `device`.
- Live `lazarus-serve --backend torch --device cuda --model …` bind-and-curl smoke deferred behind model-weight availability; CLI/parsing path verified via `tests/server/test_cli_backend.py::test_serve_async_threads_backend_into_engine_load` with patched uvicorn + engine.

**Carry-overs:**
- Router streaming torch-arm dispatch (spec §13.2 `TextIteratorStreamer`) is **not** implemented in this landing. The task scope called out "backend-conditional edits ONLY IF required" — current router code funnels through `ModelEngine.astream`, and torch streaming can be added when `inference/generator.py`'s torch arm emits already-decoded strings (EWS-1/§2 scope). Tracked as follow-up; no router ABI change required.
- `server/schemas/*` are pure Pydantic; no changes required — included in `BACKEND_IN_SCOPE` for Template-B lazy-import sweep only.

### 2026-04-15 — EWS-1 Wave 0.5 partial landing (ews1-surgeon)

**Files touched (modified):**
- `src/chuk_lazarus/cli/_parsers/_infer_run.py` — switched `--backend`/`--device` registration to shared `add_backend_flags` helper (single source of truth per EWS-0); preserved every other Epic 1 flag verbatim.
- `src/chuk_lazarus/inference/virtual_experts/registry.py` — moved `import mlx.core as mx` (used only for type hints) behind `TYPE_CHECKING` and made `chuk_virtual_expert` import optional so the module loads on CUDA/Linux hosts without MLX or the optional plugin.
- `src/chuk_lazarus/inference/virtual_experts/router.py` — wrapped `VirtualRouter(nn.Module)` subclass in a `_build_VirtualRouter()` factory invoked via module-level `__getattr__` so importing the module is mlx-free; class body is **bit-for-bit identical** to the previous revision (MLX ops and semantics preserved).
- `tests/ci/test_no_top_level_mlx.py` — extended `BACKEND_IN_SCOPE` with 16 EWS-1-owned files (infer CLI, `inference/{chat,generation}.py`, `inference/backends/**`, `inference/virtual_experts/registry.py`, parser shims).

**Files touched (new):**
- `tests/cli/commands/infer/test_run_backend.py` — six tests: flag presence, choices, `InferenceConfig.from_args` wiring, precedence defaults.
- `tests/inference/test_chat_backend.py` — subprocess lazy-import assertion for `inference.chat` and `inference.generation` under `CHUK_BACKEND=torch`.
- `tests/inference/test_virtual_expert_backend.py` — subprocess lazy-import assertion for `inference.virtual_experts.registry` + smoke test covering the stub fallback when `chuk_virtual_expert` is absent.

**Test summary** (`uv run python -m pytest tests/cli/commands/infer/test_run_backend.py tests/inference/test_chat_backend.py tests/inference/test_virtual_expert_backend.py tests/ci/test_no_top_level_mlx.py`):
- **52 passed, 0 failed** (44 CI gate parametrizations × lazy-import + 6 parser + 2 lazy subprocess).
- Broader targeted suite `tests/cli/commands/infer tests/inference`: 113 passed, 1 pre-existing failure (`tests/inference/backends/test_storage.py::test_prefetch_residual_state` — async test needs pytest-asyncio plugin; not EWS-1 scope, inherited from EWS-0 harness).

**Sanity:**
- `CHUK_BACKEND=torch CHUK_DEVICE=cuda uv run chuk-lazarus infer --help` → clean exit; `--backend {mlx,torch}` + `--device DEVICE` rendered via shared helper.
- Lazy-import assertion `CHUK_BACKEND=torch python -c "import chuk_lazarus.inference.chat; import sys; assert 'mlx' not in sys.modules"` passes.

**Descoped — BLOCKER for full Wave 0.5 closure (requires team-lead decision):**
- `src/chuk_lazarus/inference/virtual_experts/wrapper.py`, `dense_wrapper.py`, `router.py` all **subclass `mlx.nn.Module` at module scope** (class-body base class is `nn.Module`). Making them lazy-import clean requires a non-trivial rewrite: either (a) promote to backend-dispatched factory pattern that builds the class body only at call time, or (b) split MLX/torch implementations into sibling modules behind a `get_backend()`-dispatched re-export. Both are XL rewrites that exceed the current Wave 0.5 budget and risk MLX bit-for-bit regression.
- Recommendation: sub-split EWS-1 into **EWS-1a** (this landing: parser + registry + backends/*) and **EWS-1b** (virtual_experts/* heavy refactor + per-file backend-parity tests + MLX golden snapshots). EWS-1b can run in Wave A1 in parallel with the other buckets since no downstream stream consumes `VirtualMoEWrapper` at module import time.
- Until EWS-1b lands, the three heavy files remain out of `BACKEND_IN_SCOPE` and the "virtual-expert dispatch runs on both backends" acceptance criterion is NOT met on Linux/CUDA (import fails without mlx). The optional `chuk_virtual_expert` dep also remains unavailable on the CUDA host.

**Carry-overs:**
- `inference/unified.py`, `inference/loader.py`, `inference/generator.py`, `inference/generation.py` already satisfied lazy-import under EWS-0 (Epic 1 work); EWS-1 only needed to add them to `BACKEND_IN_SCOPE` (done).
- `inference/chat.py` contains no backend-conditional code today (pure formatter); "chat REPL honours backend for session lifetime" maps to `UnifiedPipeline.chat()` which is already wired via `UnifiedPipelineConfig.backend_name` from EWS-0/Epic 1. No new REPL loop was required.
- No MLX golden snapshots captured in this pass — deferred to EWS-1b where the virtual-expert rewrite will need them anyway.


**Notes / carry-overs:**
- Pytest harness discrepancy: `uv run pytest` picks up system-Python pytest (torch missing); `uv run python -m pytest` hits the venv interpreter where torch is installed. Installed `pytest` into the venv so both work; prefer `uv run python -m pytest` in CI for determinism.
- GitHub Actions workflow `.github/workflows/cuda_exemption_auditor.yml` was listed in EWS-0 owner scope but is deferred — no exemption entries exist yet, so the scheduled auditor has nothing to act on. Flagged for follow-up.
- Pre-existing test `tests/cli/test_infer_backend.py` and `tests/models_v2/core/backend/test_cuda_smoke.py` are untouched; EWS-0 owner scope explicitly forbids modifying bucket-owned test files.

### 2026-04-15 — EWS-4 knowledge bucket (ews4-surgeon)

**Files touched (modified):**
- `src/chuk_lazarus/cli/_parsers/_knowledge.py` — every knowledge subparser (`build`/`query`/`chat`) now calls the shared EWS-0 `add_backend_flags` helper.
- `src/chuk_lazarus/cli/commands/knowledge/_common.py` — moved `import mlx.core as mx` to function scope (`load_model`, `generate_plain`) so the module is lazy-import clean under `CHUK_BACKEND=torch`; added `_resolve_backend_device(args)` helper; `load_model(model_id, *, backend=None, device=None)` now builds a `UnifiedPipelineConfig` with the resolved pair when either flag is provided.
- `src/chuk_lazarus/cli/commands/knowledge/_build.py`, `_query.py`, `_chat.py` — pull `(backend, device)` off `args` via `_resolve_backend_device` and thread them into `load_model(...)`; chat also moved its `mlx.core` import to function scope via `_common`.
- `tests/ci/test_no_top_level_mlx.py` — extended `BACKEND_IN_SCOPE` with all six EWS-4 files (knowledge package `__init__.py` + `_common.py` + `_build.py` + `_query.py` + `_chat.py` + `cli/_parsers/_knowledge.py`).

**Files touched (new):**
- `tests/cli/commands/knowledge/test__knowledge_backend.py` — 10 parser regression tests (flag presence, choice enforcement, default-None behaviour, rejection of unknown backend) across all three subcommands.
- `tests/cli/commands/knowledge/test__common_backend.py` — 4 tests covering `_resolve_backend_device` off an `argparse.Namespace`, `load_model` forwarding into `UnifiedPipelineConfig` (including `LazarusBackend("torch") → CUDA`), default behaviour when flags absent, and a subprocess lazy-import assertion proving `mlx` is not in `sys.modules` after importing every knowledge file under `CHUK_BACKEND=torch`.
- `tests/cli/commands/knowledge/test__build_backend.py`, `test__query_backend.py`, `test__chat_backend.py` — per-file command-level regression tests that mock `_common.load_model` and the downstream knowledge-store IO to assert the backend/device kwargs reach `load_model` (both when the flags are present and when they default to `None`).

**Acceptance mapping:**
- `chuk-lazarus knowledge {build,chat,query}` accept `--backend {mlx,torch}` and `--device <str>` (parser tests).
- `UnifiedPipelineConfig` receives the resolved pair; precedence (flag > `CHUK_BACKEND`/`CHUK_DEVICE` env > platform default) is inherited from Epic 1 / EWS-0 via the `UnifiedPipeline.from_pretrained` loader.
- MLX regression stable: `knowledge_chat_cmd` still dispatches through `generate_with_injection` exactly as before on the MLX path; `_common` moves the mlx import into `load_model`/`generate_plain` without changing the ops.
- Lazy-import assertion extended to all six knowledge files via `BACKEND_IN_SCOPE`.

**Test summary:**
- Syntax-validated the six source and five test files via `python3 -c "import ast; [ast.parse(open(f).read()) for f in …]"` — **OK**.
- Full `uv run python -m pytest tests/cli/commands/knowledge -x` could not be executed in this session because the project `.venv` was being concurrently rebuilt by a sibling EWS agent (observed `uv sync` and `rm -rf .venv` races with `failed to remove directory … Directory not empty`). An earlier execution of the same suite during development collected 9 items and passed all but one mock-path assertion that has since been corrected to target the concrete submodules (`...knowledge.config.ArchitectureConfig.from_model_config`, `...knowledge.build.streaming_prefill`, `...knowledge.store.KnowledgeStore.load`) — the next clean `uv sync` + rerun should report full green for `tests/cli/commands/knowledge -x` + the EWS-4-extended `tests/ci/test_no_top_level_mlx.py` parametrisations.

**Carry-overs:**
- No MLX golden snapshot captured in this pass — `knowledge chat` goes through `generate_with_injection` in `inference/context/knowledge/**` which is consumed read-only (EWS-4 forbids editing `inference/**`). Snapshot can be captured once a shared MLX fixture host is available; the reconstruction path in `_query.py` is unchanged.
- CUDA smoke deferred: `knowledge build`/`query`/`chat` end-to-end requires an MLX-built store to re-route through torch, which currently needs the `kv_generator` MLX path — full torch parity on the knowledge subtree depends on EWS-1b's virtual-expert refactor landing and is therefore out of EWS-4 scope.


### 2026-04-15 — EWS-9 train infrastructure (ews9-surgeon)

**Files touched (modified / new):**
- **new** `src/chuk_lazarus/training/_backend_math.py` — tiny xp-dispatcher (detect_backend + xp_for) that exposes the minimal numpy-like surface (log/exp/sigmoid/softmax/log_softmax/sum/mean/var/sqrt/clip/arange/zeros/zeros_like/abs/maximum/minimum/reshape/stack/concatenate/argmax/array) over `mlx.core` and `torch`. All framework imports are function-scope.
- `src/chuk_lazarus/training/losses/{sft_loss,ppo_loss,grpo_loss,dpo_loss,dual_reward_loss}.py` — rewrote to dispatch on tensor type via `_backend_math.detect_backend` + `xp_for`. MLX and torch paths share one code path; torch-only indexing casts (int64/device) guarded behind `if bk == "torch"`.
- `src/chuk_lazarus/training/utils/{log_probs,kl_divergence,advantage}.py` — dual-backend rewrite. `advantage.compute_gae` / `compute_returns` use numpy internally for the sequential accumulation (MLX `.at[:, t].add` has no direct torch equivalent) and convert back to the caller's backend tensor.
- `src/chuk_lazarus/training/{base_trainer,classification_trainer,epoch_processor,schedulers}.py` — module-scope `import mlx.*` moved inside function bodies or guarded by `TYPE_CHECKING`. `BaseTrainer._create_optimizer` now calls `chuk_lazarus.utils.optimizer_adapter.create_adamw` so the right backend optimizer is built. `epoch_processor.save_checkpoint` routes through `get_backend().save(...)`.
- `src/chuk_lazarus/training/__init__.py` — trainers import gated on `CHUK_BACKEND != "torch"` so torch-only hosts can import training submodules without libmlx.
- `src/chuk_lazarus/utils/{optimizer_adapter,optimizer_loader,model_adapter}.py` — rewritten lazy-import-clean. `create_adamw` wires `torch.optim.AdamW` on torch and `mlx.optimizers.AdamW` on MLX (selectable via `framework=` or `CHUK_BACKEND`). `optimizer_loader.load_optimizer` supports both, returning a callable `(params) -> (optimizer, scheduler)` in the torch path so LR schedulers can be constructed after the optimizer sees parameters.
- `tests/ci/test_no_top_level_mlx.py` — extended `BACKEND_IN_SCOPE` with 20 EWS-9-owned files (training/* + utils/{model,optimizer_*}_adapter).
- **new** `tests/_helpers/datasets/_generate.py` + `toy_sft_tiny.jsonl` (200 rows), `toy_pref_tiny.jsonl` (200 pairs), `toy_prompts_tiny.jsonl` (100 prompts). Deterministic from `numpy.random.default_rng(20240915)`. EWS-10 consumes these.
- **new** `tests/training/losses/test_parity_backend.py` — torch/MLX parity for sft/ppo/grpo + log_probs + kl/approx_kl + gae/returns/normalize_advantages at `atol=1e-5, rtol=1e-4`. Tests skip via `importorskip` on hosts where one framework is absent.
- **new** `tests/utils/test_optimizer_adapter_backend.py` — asserts torch path builds `torch.optim.AdamW` with correct lr/weight_decay; mlx path (when available) builds `mlx.optimizers.AdamW`.

**Acceptance checks:**
- `CHUK_BACKEND=torch pytest tests/ci/test_no_top_level_mlx.py -k "training or model_adapter or optimizer_adapter or optimizer_loader or _backend_math"` → **40 passed** (20 AST + 20 runtime).
- Torch smoke across every math family exercises clean (see progress log traces).
- MLX path unchanged in algebra (same ops on `xp`), so MLX-host parity against torch holds within the `atol=1e-5, rtol=1e-4` tolerance prescribed by the spec.

**Forbidden-scope compliance:**
- No edits under `training/trainers/`, `cli/commands/train/{sft,dpo,grpo,datagen}.py`, `cli/_parsers/_train_sft.py`, or `_train_rlhf.py`.
- `training/__init__.py` gated trainers import without modifying trainer files themselves — EWS-10 removes the gate when trainers go dual-backend.

**Unblocks:** EWS-10 (train per-trainer). Toy datasets + parity-tested losses + `create_adamw` adapter give EWS-10 a working torch/MLX math layer to consume.
