# Epic 1 — Dual-Backend Validation Matrix (All CLI Buckets)

Status: R3 (addresses round-2 review blockers — see §O)
Owner: validation-author (team lazarus-cuda-epic1)
Scope: every `chuk-lazurus` CLI subcommand, validated on BOTH backends
(MLX on Apple Silicon, PyTorch/CUDA on NVIDIA including RTX 5090 `sm_120`).

Read [`01-implementation-spec.md`](../dual-backend-cuda/01-implementation-spec.md)
and [`02-workstreams.md`](../dual-backend-cuda/02-workstreams.md) first for
backend-selection semantics, dtype policy, and `UnifiedPipelineConfig.backend`
field contract.

---

## 0. How to read this matrix

Every bucket-subcommand row has **five executable columns** plus two
acceptance columns:

| Column | Definition |
|--------|------------|
| `bucket` | Top-level CLI group (`infer`, `context prefill`, …). |
| `subcommand` | Sub-invocation under the bucket. |
| `smoke cmd` | Minimal `--help` / trivial invocation that must exit 0 with no MLX import on CUDA-only hosts and no torch import on Apple-only hosts. Proves the command is wired and the lazy-import gating holds. |
| `dry-run cmd` | Pipeline constructed, config validated, but no forward pass (uses `--dry-run` / `--validate-only` where available, else `--max-new-tokens 0`). |
| `real-exec cmd` | Smallest end-to-end run that produces real tokens/logits/artifacts. Must be reproducible on CI tier-1 nodes. |
| `CUDA acceptance` | What "pass" means on an RTX 5090 / sm_120 box (or mocked-CUDA CI). |
| `MLX regression` | What must still hold on Apple Silicon — **required for every row**. No row may regress MLX. |

**Conventions**

- `$MODEL` = `Qwen/Qwen2.5-0.5B-Instruct` for standard rows,
  `Qwen/Qwen2.5-1.5B-Instruct` for rows that exercise KV cache or experts.
- `$PROMPT` = `"Say hello in one short sentence."`.
- Backend selection: `--backend cuda` or `--backend mlx`. When omitted the
  env var `CHUK_LAZARUS_BACKEND` is honored; auto-select falls back to
  `platform.system()` per spec §3.1.
- `--device cuda:0` is only legal when `--backend cuda`.
- "Mocked-CUDA CI" = a Linux runner without a GPU where
  `torch.cuda.is_available()` is monkey-patched true and the capability is
  stubbed to `(12, 0)`. Used for smoke + dry-run only.

---

## 1. Environment setup

### 1.1 CUDA host (RTX 5090 target)

```bash
# one-time
uv venv --python 3.11 .venv-cuda
source .venv-cuda/bin/activate
uv pip install -e '.[torch-cuda]'           # no mlx, no mlx-lm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible"
maj, mn = torch.cuda.get_device_capability(0)
assert (maj, mn) >= (8, 0), f"bf16 policy requires sm>=8.0, got {maj}.{mn}"
print(f"device=cuda:0 sm={maj}{mn} ok")
PY

export CHUK_LAZARUS_BACKEND=cuda
export CUDA_VISIBLE_DEVICES=0
```

Failure modes that must be caught by `get_backend(..., check_sm=True)`:

- `sm < 8.0` → `BackendSMError`, command exits non-zero with clear message.
- Missing CUDA toolkit for the compiled torch wheel (e.g. torch built for
  cu121 on a cu118 host) → `BackendToolkitError`.

### 1.2 Apple Silicon host (MLX baseline)

```bash
uv venv --python 3.11 .venv-mlx
source .venv-mlx/bin/activate
uv pip install -e '.[mlx]'
unset CHUK_LAZARUS_BACKEND           # let auto-select resolve to mlx on Darwin
```

`mlx` must NOT be imported at module load on Linux. `torch` must NOT be
imported at module load on Darwin unless the user passed `--backend cuda`.

### 1.3 Mocked-CUDA CI (no GPU)

```bash
uv pip install -e '.[torch]'         # CPU torch, no CUDA wheels
export CHUK_LAZARUS_BACKEND=cuda
export CHUK_LAZARUS_MOCK_CUDA=1      # enables the test shim in registry.py
```

Only the **smoke** and **dry-run** columns are expected to pass here. Any
row whose `real-exec cmd` requires a GPU is marked `real-exec: GPU-only`.

---

## 2. Matrix — `infer` bucket

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| infer | run (standard) | `chuk-lazarus infer run --help` | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend cuda --device cuda:0 --max-new-tokens 0 --validate-only` | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend cuda --device cuda:0 --max-new-tokens 16 --dtype bfloat16` | exits 0; generated text non-empty; logs `backend=cuda device=cuda:0 dtype=bfloat16`; peak VRAM < 4 GB for 0.5B | same command with `--backend mlx` on Darwin: identical token count shape, dtype reported as `mlx.bfloat16`, no torch import |
| infer | run (kv_direct path) | `chuk-lazarus infer run --kv-direct --help` | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend cuda --kv-direct --max-new-tokens 0` | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend cuda --kv-direct --max-new-tokens 32` | KV cache built on CUDA; mask dtype = `torch.bfloat16`; no `mx.bfloat16` references in stack trace | KV cache built on MLX; mask dtype = `mx.bfloat16`; bit-identical first-token logits vs prior release within 1e-3 |

## 3. Matrix — `context prefill` bucket

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| context prefill | vec-inject | `chuk-lazarus context prefill vec-inject --help` | `chuk-lazarus context prefill vec-inject --model $MODEL --source local:./fixtures/vec.npz --backend cuda --validate-only` | `chuk-lazarus context prefill vec-inject --model $MODEL --source local:./fixtures/vec.npz --backend cuda --device cuda:0 --out out.kv` | injected tensors on `cuda:0`; `out.kv` loadable; provider `_local_file` uses lazy torch path | same with `--backend mlx` on Darwin: `out.kv` byte-identical dtype metadata; no top-level `import mlx.core` triggered before `--backend` resolved |
| context prefill | kv-build (alias) | `chuk-lazarus context prefill kv-build --help` | `chuk-lazarus context prefill kv-build --model $MODEL --prompt "$PROMPT" --backend cuda --validate-only` | `chuk-lazarus context prefill kv-build --model $MODEL --prompt "$PROMPT" --backend cuda --device cuda:0 --out kv.bin` | KV tensor shapes match model layer count; persisted as fp16/bf16 per dtype policy | MLX path produces same shapes; round-trip load/save parity |

## 4. Matrix — `context generate` bucket

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| context generate | unified | `chuk-lazarus context generate unified --help` | `chuk-lazarus context generate unified --model $MODEL --prompt "$PROMPT" --backend cuda --validate-only` | `chuk-lazarus context generate unified --model $MODEL --prompt "$PROMPT" --backend cuda --max-new-tokens 24` | unified pipeline selects torch backend; probe hooks attach to torch modules | MLX unified generation parity (token count, dtype) |
| context generate | mode7 | `chuk-lazarus context generate mode7 --help` | `chuk-lazarus context generate mode7 --model $MODEL --prompt "$PROMPT" --backend cuda --validate-only` | `chuk-lazarus context generate mode7 --model $MODEL --prompt "$PROMPT" --backend cuda --max-new-tokens 24` | mode-7 residual stream probe runs on CUDA; no MLX import | MLX mode7 baseline generation matches prior release |
| context generate | probes | `chuk-lazarus context generate probes --help` | `chuk-lazarus context generate probes --model $MODEL --prompt "$PROMPT" --backend cuda --validate-only` | `chuk-lazarus context generate probes --model $MODEL --prompt "$PROMPT" --backend cuda --layers 0,1` | per-layer probe tensors on `cuda:0`; dtype bf16 | MLX probes produce matching shapes and dtype policy |

## 5. Matrix — `knowledge` bucket

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| knowledge | build | `chuk-lazarus knowledge build --help` | `chuk-lazarus knowledge build --model $MODEL --corpus fixtures/corpus.jsonl --backend cuda --validate-only` | `chuk-lazarus knowledge build --model $MODEL --corpus fixtures/corpus.jsonl --backend cuda --out kb.parquet` | embedding batches use `cuda:0`; no host-side OOM at batch=16 | MLX embedding build parity; parquet schema identical |
| knowledge | query | `chuk-lazarus knowledge query --help` | `chuk-lazarus knowledge query --kb kb.parquet --q "hello" --backend cuda --validate-only` | `chuk-lazarus knowledge query --kb kb.parquet --q "hello" --backend cuda --k 5` | top-5 neighbors deterministic under fixed seed | MLX top-5 neighbors match within cosine tol 1e-3 |
| knowledge | chat | `chuk-lazarus knowledge chat --help` | `chuk-lazarus knowledge chat --kb kb.parquet --model $MODEL --backend cuda --validate-only` | `chuk-lazarus knowledge chat --kb kb.parquet --model $MODEL --backend cuda --turns 1 --prompt "$PROMPT"` | end-to-end RAG produces answer; retrieval + generation both on CUDA | MLX RAG answer non-empty; no torch import on Darwin |

## 6. Matrix — `introspect` bucket

Introspection is the largest surface; MLX coupling in `introspection/hooks.py`
is the main risk. Every row below MUST pass `--backend mlx` unchanged.

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| introspect | analyze | `chuk-lazarus introspect analyze --help` | `chuk-lazarus introspect analyze --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect analyze --model $MODEL --prompt "$PROMPT" --backend cuda` | hook registrations are torch `forward_hook`s; no `mx.eval` calls | MLX hook path unchanged; prior JSON report diff == 0 |
| introspect | compare | `chuk-lazarus introspect compare --help` | `chuk-lazarus introspect compare --model $MODEL --baseline $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect compare --model $MODEL --baseline $MODEL --prompt "$PROMPT" --backend cuda` | identical-model compare yields ~0 diff on CUDA | MLX compare ~0 diff; no cross-backend leakage |
| introspect | hooks | `chuk-lazarus introspect hooks --help` | `chuk-lazarus introspect hooks --model $MODEL --backend cuda --list` | `chuk-lazarus introspect hooks --model $MODEL --prompt "$PROMPT" --backend cuda --capture residual` | residual capture tensor on `cuda:0`, bf16 | MLX capture shape & dtype identical |
| introspect | ablate | `chuk-lazarus introspect ablate --help` | `chuk-lazarus introspect ablate --model $MODEL --layers 0 --backend cuda --validate-only` | `chuk-lazarus introspect ablate --model $MODEL --prompt "$PROMPT" --layers 0 --backend cuda` | ablated logits differ from baseline; tensor ops on CUDA | MLX ablation delta matches prior snapshot |
| introspect | weight-diff | `chuk-lazarus introspect weight-diff --help` | `chuk-lazarus introspect weight-diff --a $MODEL --b $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect weight-diff --a $MODEL --b $MODEL --backend cuda --top 8` | norm ~0 for identical ckpts; runs on CUDA | MLX norm ~0; same top-8 names |
| introspect | activation-diff | `chuk-lazarus introspect activation-diff --help` | `chuk-lazarus introspect activation-diff --a $MODEL --b $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect activation-diff --a $MODEL --b $MODEL --prompt "$PROMPT" --backend cuda` | diff computed per-layer on CUDA; memory released between layers | MLX diff matches prior release |
| introspect | layer | `chuk-lazarus introspect layer --help` | `chuk-lazarus introspect layer --model $MODEL --layer 0 --backend cuda --validate-only` | `chuk-lazarus introspect layer --model $MODEL --prompt "$PROMPT" --layer 0 --backend cuda` | per-layer stats JSON; CUDA tensors | MLX per-layer stats identical |
| introspect | format | `chuk-lazarus introspect format --help` | `chuk-lazarus introspect format --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect format --model $MODEL --backend cuda --out fmt.json` | format inspection works when only torch installed | same on MLX-only install |
| introspect | generate | `chuk-lazarus introspect generate --help` | `chuk-lazarus introspect generate --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect generate --model $MODEL --prompt "$PROMPT" --backend cuda --max-new-tokens 16 --capture residual` | generation + residual capture both on CUDA | MLX parity |
| introspect | metacog | `chuk-lazarus introspect metacog --help` | `chuk-lazarus introspect metacog --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect metacog --model $MODEL --prompt "$PROMPT" --backend cuda` | metacognitive probes emit on CUDA | MLX metacog unchanged |
| introspect | steer | `chuk-lazarus introspect steer --help` | `chuk-lazarus introspect steer --model $MODEL --vector fixtures/v.npy --backend cuda --validate-only` | `chuk-lazarus introspect steer --model $MODEL --vector fixtures/v.npy --prompt "$PROMPT" --backend cuda` | steering vec moved to `cuda:0`; generation reflects steer | MLX steer unchanged |
| introspect | arithmetic | `chuk-lazarus introspect arithmetic --help` | `chuk-lazarus introspect arithmetic --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect arithmetic --model $MODEL --expr "king-man+woman" --backend cuda` | vector arithmetic on CUDA | MLX identical top-k |
| introspect | uncertainty | `chuk-lazarus introspect uncertainty --help` | `chuk-lazarus introspect uncertainty --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect uncertainty --model $MODEL --prompt "$PROMPT" --backend cuda --samples 4` | MC samples on CUDA; determinism w/ seed | MLX uncertainty mean/std match |
| introspect | probe | `chuk-lazarus introspect probe --help` | `chuk-lazarus introspect probe --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect probe --model $MODEL --dataset fixtures/probe.jsonl --backend cuda` | linear probe trained on CUDA; accuracy logged | MLX accuracy within 1% |
| introspect | neurons | `chuk-lazarus introspect neurons --help` | `chuk-lazarus introspect neurons --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect neurons --model $MODEL --layer 0 --top 8 --backend cuda` | top-8 neuron list on CUDA | MLX top-8 list identical |
| introspect | cluster | `chuk-lazarus introspect cluster --help` | `chuk-lazarus introspect cluster --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect cluster --model $MODEL --k 4 --backend cuda` | k-means on CUDA tensors | MLX labels match (up to permutation) |
| introspect | memory | `chuk-lazarus introspect memory --help` | `chuk-lazarus introspect memory --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect memory --model $MODEL --prompt "$PROMPT" --backend cuda` | VRAM profile reported | MLX unified-memory profile reported |
| introspect | inject | `chuk-lazarus introspect inject --help` | `chuk-lazarus introspect inject --model $MODEL --vector fixtures/v.npy --backend cuda --validate-only` | `chuk-lazarus introspect inject --model $MODEL --vector fixtures/v.npy --prompt "$PROMPT" --backend cuda` | injected activations take effect; tensors on CUDA | MLX injection parity |
| introspect | directions | `chuk-lazarus introspect directions --help` | `chuk-lazarus introspect directions --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect directions --model $MODEL --dataset fixtures/ds.jsonl --backend cuda` | direction vectors emitted, saved as fp32 | MLX directions match within 1e-3 cosine |
| introspect | operand-directions | `chuk-lazarus introspect operand-directions --help` | `chuk-lazarus introspect operand-directions --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect operand-directions --model $MODEL --op add --backend cuda` | operand directions computed on CUDA | MLX parity |
| introspect | embedding | `chuk-lazarus introspect embedding --help` | `chuk-lazarus introspect embedding --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect embedding --model $MODEL --text "$PROMPT" --backend cuda` | embedding tensor on `cuda:0`, dtype per policy | MLX embedding matches bitwise shape/dtype |
| introspect | commutativity | `chuk-lazarus introspect commutativity --help` | `chuk-lazarus introspect commutativity --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect commutativity --model $MODEL --backend cuda` | commutativity diagnostics run on CUDA | MLX metric identical |
| introspect | early-layers | `chuk-lazarus introspect early-layers --help` | `chuk-lazarus introspect early-layers --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect early-layers --model $MODEL --prompt "$PROMPT" --backend cuda` | first-N-layer stats on CUDA | MLX stats unchanged |
| introspect | patch | `chuk-lazarus introspect patch --help` | `chuk-lazarus introspect patch --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect patch --model $MODEL --src "$PROMPT" --dst "$PROMPT" --layer 0 --backend cuda` | activation patching delta non-zero | MLX delta identical |
| introspect | circuit capture | `chuk-lazarus introspect circuit capture --help` | `chuk-lazarus introspect circuit capture --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect circuit capture --model $MODEL --prompt "$PROMPT" --backend cuda --out ckt.json` | circuit JSON has torch dtype tags | MLX JSON identical keys/values (modulo backend tag) |
| introspect | circuit invoke | `chuk-lazarus introspect circuit invoke --help` | `chuk-lazarus introspect circuit invoke --ckt ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit invoke --ckt ckt.json --backend cuda` | replay runs on CUDA; determinism w/ seed | MLX replay parity |
| introspect | circuit decode | `chuk-lazarus introspect circuit decode --help` | `chuk-lazarus introspect circuit decode --ckt ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit decode --ckt ckt.json --backend cuda` | decoded tokens on CUDA | MLX tokens identical |
| introspect | circuit test | `chuk-lazarus introspect circuit test --help` | `chuk-lazarus introspect circuit test --ckt ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit test --ckt ckt.json --backend cuda` | test harness green on CUDA | MLX harness green |
| introspect | circuit compare | `chuk-lazarus introspect circuit compare --help` | `chuk-lazarus introspect circuit compare --a ckt.json --b ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit compare --a ckt.json --b ckt.json --backend cuda` | self-compare diff ~0 | MLX self-compare ~0 |
| introspect | circuit view | `chuk-lazarus introspect circuit view --help` | `chuk-lazarus introspect circuit view --ckt ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit view --ckt ckt.json --backend cuda --out ckt.svg` | renderer backend-agnostic; no torch import if `--format text` | MLX renderer output unchanged |
| introspect | circuit export | `chuk-lazarus introspect circuit export --help` | `chuk-lazarus introspect circuit export --ckt ckt.json --backend cuda --validate-only` | `chuk-lazarus introspect circuit export --ckt ckt.json --backend cuda --out ckt.pt` | exports torch `state_dict`-shaped blob | MLX export emits `.npz`; MLX path unchanged |
| introspect | virtual-expert | `chuk-lazarus introspect virtual-expert --help` | `chuk-lazarus introspect virtual-expert --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect virtual-expert --model $MODEL --prompt "$PROMPT" --backend cuda` | virtual-expert routing runs on CUDA | MLX routing unchanged (feeds AAES metrics) |
| introspect | moe-expert | `chuk-lazarus introspect moe-expert --help` | `chuk-lazarus introspect moe-expert --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect moe-expert --model $MODEL --prompt "$PROMPT" --backend cuda --top-k 2` | expert selection counts match prior CPU torch run | MLX expert selection identical |
| introspect | classifier | `chuk-lazarus introspect classifier --help` | `chuk-lazarus introspect classifier --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect classifier --model $MODEL --dataset fixtures/cls.jsonl --backend cuda` | classifier head trains on CUDA; accuracy logged | MLX accuracy within 1% |
| introspect | logit-lens | `chuk-lazarus introspect logit-lens --help` | `chuk-lazarus introspect logit-lens --model $MODEL --backend cuda --validate-only` | `chuk-lazarus introspect logit-lens --model $MODEL --prompt "$PROMPT" --backend cuda` | per-layer logit decode on CUDA; top-1 tokens logged | MLX top-1 token sequence identical |

## 7. Matrix — serving buckets

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| serve | (root) | `chuk-lazarus serve --help` | `chuk-lazarus serve --model $MODEL --backend cuda --port 8123 --validate-only` | `chuk-lazarus serve --model $MODEL --backend cuda --port 8123 &` then `curl -X POST :8123/v1/completions -d '{"prompt":"$PROMPT","max_tokens":16}'` | HTTP 200; JSON completion non-empty; process binds to `cuda:0` | same on MLX; HTTP contract byte-identical (keys, types) |
| lazarus-serve | (root) | `chuk-lazarus lazarus-serve --help` | `chuk-lazarus lazarus-serve --model $MODEL --backend cuda --port 8124 --validate-only` | `chuk-lazarus lazarus-serve --model $MODEL --backend cuda --port 8124 &` + websocket ping | WS handshake OK; streamed tokens flush; CUDA memory stable | MLX WS streaming unchanged |

## 8. Matrix — `train` bucket

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| train | sft | `chuk-lazarus train sft --help` | `chuk-lazarus train sft --model $MODEL --data fixtures/sft.jsonl --backend cuda --steps 0 --validate-only` | `chuk-lazarus train sft --model $MODEL --data fixtures/sft.jsonl --backend cuda --steps 2 --bs 1 --dtype bfloat16` | loss finite; optimizer step succeeds; bf16 autocast active on sm>=8 | MLX SFT loss finite; 2-step trajectory matches prior release within 1% |
| train | dpo | `chuk-lazarus train dpo --help` | `chuk-lazarus train dpo --model $MODEL --pairs fixtures/pairs.jsonl --backend cuda --steps 0 --validate-only` | `chuk-lazarus train dpo --model $MODEL --pairs fixtures/pairs.jsonl --backend cuda --steps 2 --bs 1` | DPO loss finite; ref-model loaded on CUDA | MLX DPO 2-step loss within 1% of prior release |
| train | grpo | `chuk-lazarus train grpo --help` | `chuk-lazarus train grpo --model $MODEL --data fixtures/grpo.jsonl --backend cuda --steps 0 --validate-only` | `chuk-lazarus train grpo --model $MODEL --data fixtures/grpo.jsonl --backend cuda --steps 2 --group-size 2` | GRPO reward computed on CUDA; advantages finite | MLX GRPO 2-step parity |

## 9. Matrix — utility buckets

| bucket | subcommand | smoke cmd | dry-run cmd | real-exec cmd | CUDA acceptance | MLX regression |
|---|---|---|---|---|---|---|
| generate | (root) | `chuk-lazarus generate --help` | `chuk-lazarus generate --model $MODEL --prompt "$PROMPT" --backend cuda --max-new-tokens 0` | `chuk-lazarus generate --model $MODEL --prompt "$PROMPT" --backend cuda --max-new-tokens 16` | tokens generated; same output format as `infer run` | MLX generation unchanged |
| data | prep | `chuk-lazarus data prep --help` | `chuk-lazarus data prep --in fixtures/raw.jsonl --out prep.jsonl --validate-only` | `chuk-lazarus data prep --in fixtures/raw.jsonl --out prep.jsonl` | backend-neutral; must not import torch or mlx at module load | same on MLX install; no torch import |
| data | shard | `chuk-lazarus data shard --help` | `chuk-lazarus data shard --in prep.jsonl --out shards/ --n 2 --validate-only` | `chuk-lazarus data shard --in prep.jsonl --out shards/ --n 2` | shard files written; checksum stable | identical on MLX |
| tokenizer | inspect | `chuk-lazarus tokenizer inspect --help` | `chuk-lazarus tokenizer inspect --model $MODEL --validate-only` | `chuk-lazarus tokenizer inspect --model $MODEL --text "$PROMPT"` | backend-neutral; runs without CUDA | runs without MLX on Linux |
| tokenizer | encode | `chuk-lazarus tokenizer encode --help` | `chuk-lazarus tokenizer encode --model $MODEL --text "" --validate-only` | `chuk-lazarus tokenizer encode --model $MODEL --text "$PROMPT"` | id list deterministic | MLX id list identical |
| gym | run | `chuk-lazarus gym run --help` | `chuk-lazarus gym run --env dummy --model $MODEL --backend cuda --episodes 0 --validate-only` | `chuk-lazarus gym run --env dummy --model $MODEL --backend cuda --episodes 1 --max-steps 4` | rollout policy evaluates on CUDA; reward finite | MLX rollout parity |
| experiment | run | `chuk-lazarus experiment run --help` | `chuk-lazarus experiment run --config fixtures/exp.yaml --backend cuda --validate-only` | `chuk-lazarus experiment run --config fixtures/exp.yaml --backend cuda` | experiment driver selects CUDA; results json written | MLX experiment driver unchanged |
| bench | infer | `chuk-lazarus bench infer --help` | `chuk-lazarus bench infer --model $MODEL --backend cuda --iters 0 --validate-only` | `chuk-lazarus bench infer --model $MODEL --backend cuda --iters 10 --batch 1` | tokens/s reported; no NaN latencies; CUDA warm-up excluded | MLX tokens/s within ±5% of prior release on same hardware class |
| bench | train | `chuk-lazarus bench train --help` | `chuk-lazarus bench train --model $MODEL --backend cuda --iters 0 --validate-only` | `chuk-lazarus bench train --model $MODEL --backend cuda --iters 5 --bs 1` | step/s reported; loss finite | MLX step/s within ±5% |

---

## 10. Global invariants (every row)

1. **Lazy imports:** Running any `--help` on a Linux host WITHOUT `mlx`
   installed must succeed. Running on Darwin without `torch` installed must
   succeed for rows not explicitly marked `--backend cuda`.
2. **No implicit fallback:** if `--backend cuda` is passed on a host without
   CUDA, the command must exit non-zero with `BackendUnavailableError`,
   never silently fall back to CPU or MLX.
3. **Dtype policy:** on sm>=8.0 the default dtype is `bfloat16`; on sm<8.0
   the command must refuse with a clear error unless `--dtype float16` is
   passed explicitly. On MLX the default remains `mx.bfloat16` on M-series.
4. **Determinism:** `--seed 0` + `--sampler greedy` must yield identical
   token ids across backends for the *smoke* and *dry-run* columns; the
   *real-exec* column may differ within the documented numeric tolerance
   but token-1 argmax MUST agree.
5. **MLX regression:** every row has an MLX regression check. A PR that
   improves CUDA but regresses any MLX row is rejected.

---

## 11. CI wiring sketch

- `tests/ci/matrix_smoke.yaml` — runs every row's smoke cmd on
  mocked-CUDA Linux + Darwin.
- `tests/ci/matrix_dryrun.yaml` — runs every dry-run on the same two
  runners.
- `tests/ci/matrix_real_cuda.yaml` — runs every `real-exec` row on the
  self-hosted RTX 5090 runner (`runs-on: [self-hosted, cuda, sm120]`).
- `tests/ci/matrix_real_mlx.yaml` — runs every `real-exec` row on the
  self-hosted M-series runner.
- Each job asserts `echo $IMPORTED_MLX` / `$IMPORTED_TORCH` sentinels set
  by the test shim in `registry.py` to catch accidental eager imports.

---

---

## Appendix A — Fixtures manifest (R2 blocker #1)

All fixtures live under `tests/fixtures/` and are produced by a single
generator: `tests/fixtures/generate.sh` (bash) which dispatches to per-file
Python builders under `tests/fixtures/_builders/`. Generator is
deterministic (`PYTHONHASHSEED=0`, `numpy.random.default_rng(0)`).

| Fixture | Schema | Size bound | Generator | Notes |
|---|---|---|---|---|
| `fixtures/vec.npz` | `{ "vectors": float32[N=32, D=hidden], "token_ids": int32[N] }` | ≤ 1 MiB | `_builders/build_vec.py` | hidden auto-detected from `$MODEL` config.json; stored as fp32 for cross-backend parity |
| `fixtures/corpus.jsonl` | `{"id": str, "text": str}` per line | ≤ 256 KiB, N=64 docs | `_builders/build_corpus.py` | text pulled from Wikipedia dump snapshot pinned by SHA |
| `fixtures/probe.jsonl` | `{"text": str, "label": int∈{0,1}}` | ≤ 128 KiB, N=128 rows | `_builders/build_probe.py` | balanced labels |
| `fixtures/ds.jsonl` | `{"positive": str, "negative": str}` | ≤ 128 KiB, N=64 pairs | `_builders/build_ds.py` | direction-finding dataset |
| `fixtures/v.npy` | `float32[hidden]` | ≤ 64 KiB | `_builders/build_vector.py` | unit-norm steering vector |
| `fixtures/exp.yaml` | experiment config (see `chuk_lazarus.experiment.schema`) | ≤ 4 KiB | `_builders/build_exp.py` | pins `$MODEL` and seed |
| `fixtures/cls.jsonl` | `{"text": str, "label": int∈{0..3}}` | ≤ 128 KiB, N=128 | `_builders/build_cls.py` | 4-way classifier training data |
| `fixtures/sft.jsonl` | `{"messages": [{"role":..,"content":..}]}` | ≤ 256 KiB, N=32 | `_builders/build_sft.py` | chat-format SFT |
| `fixtures/pairs.jsonl` | `{"prompt": str, "chosen": str, "rejected": str}` | ≤ 256 KiB, N=32 | `_builders/build_pairs.py` | DPO preference pairs |
| `fixtures/grpo.jsonl` | `{"prompt": str, "reward_fn": str}` | ≤ 128 KiB, N=16 | `_builders/build_grpo.py` | `reward_fn` is a name resolvable in `chuk_lazarus.train.grpo.rewards` |
| `fixtures/raw.jsonl` | `{"text": str}` | ≤ 512 KiB, N=256 | `_builders/build_raw.py` | pre-prep corpus |

Regenerate with:

```bash
bash tests/fixtures/generate.sh --model "$MODEL" --out tests/fixtures/
sha256sum tests/fixtures/*.{npz,npy,jsonl,yaml} > tests/fixtures/SHA256SUMS
```

CI verifies `SHA256SUMS` before the matrix runs; any drift fails the job.

---

## Appendix B — Row arg fill-ins (R2 blocker #2)

The "real-exec" column placeholders in §3–§6 are replaced as follows. These
override the original cells (treat the original cells as `…` templates).

| Row | Real-exec command (concrete) |
|---|---|
| ctx prefill vec-inject | `chuk-lazarus context prefill vec-inject --model $MODEL --source local:tests/fixtures/vec.npz --layer 12 --backend cuda --device cuda:0 --out out/vec.kv` |
| ctx prefill kv-build | `chuk-lazarus context prefill kv-build --model $MODEL --prompt "$PROMPT" --backend cuda --device cuda:0 --layers 0-23 --out out/kv.bin` |
| knowledge build | `chuk-lazarus knowledge build --model $MODEL --corpus tests/fixtures/corpus.jsonl --backend cuda --device cuda:0 --batch 8 --out out/kb.parquet` |
| introspect ablate | `chuk-lazarus introspect ablate --model $MODEL --prompt "$PROMPT" --layers 0,6,12 --heads 0-3 --backend cuda --device cuda:0 --out out/ablate.json` |
| introspect steer | `chuk-lazarus introspect steer --model $MODEL --vector tests/fixtures/v.npy --layer 12 --alpha 2.0 --prompt "$PROMPT" --backend cuda --device cuda:0 --out out/steer.json` |
| introspect arithmetic | `chuk-lazarus introspect arithmetic --model $MODEL --expr "king - man + woman" --top-k 8 --backend cuda --device cuda:0 --out out/arith.json` |
| introspect probe | `chuk-lazarus introspect probe --model $MODEL --dataset tests/fixtures/probe.jsonl --layer 12 --epochs 2 --backend cuda --device cuda:0 --out out/probe.json` |
| introspect directions | `chuk-lazarus introspect directions --model $MODEL --dataset tests/fixtures/ds.jsonl --method diff-mean --layer 12 --backend cuda --device cuda:0 --out out/dirs.npz` |
| introspect operand-directions | `chuk-lazarus introspect operand-directions --model $MODEL --op add --operands "2,3" --layer 12 --backend cuda --device cuda:0 --out out/op_dirs.npz` |
| introspect circuit capture | `chuk-lazarus introspect circuit capture --model $MODEL --prompt "$PROMPT" --layers 0-23 --backend cuda --device cuda:0 --out out/ckt.json` |
| introspect circuit invoke | `chuk-lazarus introspect circuit invoke --ckt out/ckt.json --prompt "$PROMPT" --backend cuda --device cuda:0 --out out/ckt_invoke.json` |
| introspect circuit decode | `chuk-lazarus introspect circuit decode --ckt out/ckt.json --top-k 5 --backend cuda --device cuda:0 --out out/ckt_decode.json` |
| introspect circuit test | `chuk-lazarus introspect circuit test --ckt out/ckt.json --suite smoke --backend cuda --device cuda:0 --out out/ckt_test.json` |
| introspect circuit compare | `chuk-lazarus introspect circuit compare --a out/ckt.json --b out/ckt.json --metric cosine --backend cuda --device cuda:0 --out out/ckt_cmp.json` |
| introspect circuit view | `chuk-lazarus introspect circuit view --ckt out/ckt.json --format svg --backend cuda --out out/ckt.svg` |
| introspect circuit export | `chuk-lazarus introspect circuit export --ckt out/ckt.json --backend cuda --device cuda:0 --out out/ckt.pt` |
| introspect classifier | `chuk-lazarus introspect classifier --model $MODEL --dataset tests/fixtures/cls.jsonl --layer 12 --epochs 2 --backend cuda --device cuda:0 --out out/cls.json` |

All `out/*` paths are under the test run's temp dir (`$PYTEST_TMPDIR` or
`mktemp -d`); cleanup is handled by the pytest fixture `tmp_run_dir`.

---

## Appendix C — Numeric tolerances for introspect rows (R2 blocker #3)

Every introspect row that emits a tensor-shaped artifact must compare
against the MLX reference (captured once on an M-series runner and stored
per Appendix J). Tolerances are applied to the flattened float output:

| Row | Metric | atol | rtol | cosine min |
|---|---|---|---|---|
| analyze / compare | per-layer activation means | 1e-4 | 1e-3 | 0.9999 |
| hooks (residual capture) | residual stream | 1e-4 | 1e-3 | 0.9999 |
| ablate | ablated-vs-baseline delta | 1e-3 | 1e-2 | 0.999 |
| weight-diff | per-tensor L2 norm | 1e-5 | 1e-4 | n/a |
| activation-diff | per-layer L2 | 1e-3 | 1e-2 | 0.999 |
| layer | per-layer stats (mean, std, max) | 1e-4 | 1e-3 | n/a |
| generate (+capture) | residual capture | 1e-3 | 1e-2 | 0.999 |
| metacog | probe logits | 1e-3 | 1e-2 | 0.999 |
| steer | steered-vs-baseline delta | 1e-3 | 1e-2 | 0.999 |
| arithmetic | top-k vector | 1e-4 | 1e-3 | 0.9999 |
| uncertainty | sample mean / std | 5e-3 | 1e-2 | n/a |
| probe | trained weights | 1e-2 | 1e-2 | 0.995 |
| neurons | top-k activation values | 1e-4 | 1e-3 | n/a |
| cluster | centroid coords (after Hungarian match) | 1e-2 | 1e-2 | 0.99 |
| inject | injected-vs-baseline delta | 1e-3 | 1e-2 | 0.999 |
| directions | direction vectors | 1e-4 | 1e-3 | 0.9999 |
| operand-directions | operand direction vectors | 1e-4 | 1e-3 | 0.9999 |
| embedding | embedding tensor | 1e-4 | 1e-3 | 0.9999 |
| commutativity | commutator magnitude | 1e-4 | 1e-3 | n/a |
| early-layers | per-layer stats | 1e-4 | 1e-3 | n/a |
| patch | patched-vs-baseline delta | 1e-3 | 1e-2 | 0.999 |
| circuit capture | attention pattern tensors | 1e-3 | 1e-2 | 0.999 |
| circuit invoke / decode | replay logits | 1e-3 | 1e-2 | 0.999 |
| logit-lens | per-layer logits | 1e-3 | 1e-2 | 0.999 |
| classifier | head weights | 1e-2 | 1e-2 | 0.995 |
| virtual-expert / moe-expert | expert weights | 1e-3 | 1e-2 | 0.999 |

Implemented as `tests/ci/_compare.py::assert_close(ref, actual, spec)`.

---

## Appendix D — Negative-test rows (R2 blocker #4)

Every row that accepts `--device` also gets a negative-test counterpart.
All MUST exit non-zero with the listed error class.

| Negative case | Command | Expected error | Exit |
|---|---|---|---|
| MLX + cuda device | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend mlx --device cuda:0` | `BackendDeviceMismatchError` | 2 |
| CUDA + mps device | `chuk-lazarus infer run --model $MODEL --prompt "$PROMPT" --backend cuda --device mps` | `BackendDeviceMismatchError` | 2 |
| CUDA on no-GPU host | `chuk-lazarus infer run --model $MODEL --backend cuda` (no CUDA) | `BackendUnavailableError` | 3 |
| sm<8.0 + bf16 | `CHUK_LAZARUS_FORCE_SM=75 chuk-lazarus infer run --model $MODEL --backend cuda --dtype bfloat16` | `BackendSMError` | 4 |
| Unknown backend | `chuk-lazarus infer run --model $MODEL --backend rocm` | `BackendUnknownError` | 2 |

Mirror rows exist for: `infer run`, `infer run --kv-direct`,
`context prefill vec-inject`, `context prefill kv-build`,
`context generate unified/mode7/probes`, `knowledge build/query/chat`,
every `introspect` subcommand, `serve`, `lazarus-serve`,
`train {sft,dpo,grpo}`, `generate`, `gym run`, `experiment run`,
`bench {infer,train}`. Enforced by `tests/ci/matrix_negative.yaml`.

---

## Appendix E — Bench correctness (R2 blocker #5)

`bench infer` and `bench train` are **perf+correctness**, not perf-only.

| Check | Threshold |
|---|---|
| Throughput (tokens/s or step/s) | within ±5% of baseline (Appendix J) |
| Final-logits L2 vs dry-run baseline (bench infer) | atol 1e-3, rtol 1e-2 |
| Step-10 loss vs dry-run baseline (bench train) | atol 1e-3, rtol 1e-2 |
| NaN / Inf count | 0 |

`bench` rows fail if either perf OR correctness regresses.

---

## Appendix F — Circuit cross-backend round-trip (R2 blocker #6)

MLX exports `.npz`, CUDA exports `.pt`. The export formats are **not
bit-compatible** by design, but the *logical* circuit (keys, shapes,
dtypes) must round-trip.

Additional rows:

| Row | Command | Criterion |
|---|---|---|
| circuit cross-rt (CUDA→MLX) | `chuk-lazarus introspect circuit export --ckt out/ckt.json --backend cuda --out out/ckt.pt && chuk-lazarus introspect circuit invoke --ckt out/ckt.pt --backend mlx --out out/rt_mlx.json` | Dual gate (per O.4): (a) logits L2 vs native-backend invoke: atol 1e-3, rtol 1e-2; (b) top-1 token agreement 100% across first 16 positions |
| circuit cross-rt (MLX→CUDA) | symmetric, `.npz` → CUDA | Same dual gate as above |

Importer lives at `src/chuk_lazarus/introspection/circuits/_portable.py`
(to be created under WS-5). If that importer is not in scope for Epic 1,
this row is **blocker-deferred** and must be filed as an Epic-2 ticket
before the matrix is signed off.

---

## Appendix G — CI mocked-CUDA shim reference (R2 blocker #7)

The shim is concrete:

- Code: `src/chuk_lazarus/models_v2/core/backend/registry.py::_mock_cuda_shim()`
  activated when `CHUK_LAZARUS_MOCK_CUDA=1`. Sets
  `torch.cuda.is_available` → `lambda: True`,
  `torch.cuda.get_device_capability` → `lambda i=0: (12, 0)`,
  `torch.cuda.current_device` → `lambda: 0`.
- Test wiring: `tests/conftest.py::pytest_configure` reads the env var
  and installs the shim session-wide when set.
- Import sentinels: `tests/ci/_sentinels.py` exposes `IMPORTED_MLX`
  and `IMPORTED_TORCH` booleans, set by import hooks installed in
  `conftest.py` via `sys.meta_path`. The per-row assertions in
  `tests/ci/matrix_smoke.yaml` call
  `python -c "from tests.ci._sentinels import IMPORTED_MLX; assert not IMPORTED_MLX"`.

---

## Appendix H — Perf baselines (R2 blocker #8)

- Storage: GitHub Actions artifact `perf-baseline-<sha>.json`, retained
  90 days; long-term in S3 at
  `s3://chuk-lazarus-ci/baselines/{arch}/{backend}/{yyyy-mm-dd}.json`
  with a `latest.json` alias under the same prefix (see O.7). `{arch}`
  ∈ `{sm120, sm90, sm86, m-series}`, `{backend}` ∈ `{cuda, mlx}`.
  Lifecycle: Standard → Glacier at 180 days, expire at 2 years.
- Schema: `{ "row_id": {"tokens_per_s": float, "step_per_s": float,
  "loss_step10": float, "backend": "cuda|mlx", "sm": int, "sha": str,
  "hw": str, "timestamp": iso8601 } }`.
- Detection: `tests/ci/_perf_gate.py compare --current run.json
  --baseline baseline.json --tolerance 0.05`. Exits non-zero on
  regression; emits a markdown summary to `$GITHUB_STEP_SUMMARY`.
- Baseline promotion: only main-branch green runs promote to the
  baseline via `tests/ci/_perf_gate.py promote`.

---

## Appendix I — Flakiness controls (R2 blocker #9)

| Row | Policy |
|---|---|
| uncertainty (`--samples 4`) | Fixed seed `--seed 0`; retry ≤2 on mismatch; variance tolerance on sample std: atol 5e-3 |
| cluster (k-means) | Hungarian-matched centroid comparison; retry ≤2 (per O.5); init via k-means++ with fixed seed |
| grpo (stochastic reward) | `--seed 0`; retry ≤2; loss trajectory MAE tolerance 5e-2 |
| dpo | retry ≤2 (per O.5); loss atol 1e-2 |
| probe / classifier training | fixed seed; retry ≤2 (per O.5); accuracy tolerance ±1% |
| serve / lazarus-serve | retry ≤3 on transport errors (port bind, handshake); no retry on bad payload |

Retry harness: pytest-rerunfailures with explicit per-row markers in
`tests/ci/matrix_real_*.yaml`.

---

## Appendix J — Artifact size bounds (R2 blocker #10)

| Artifact | Max size |
|---|---|
| `out/vec.kv`, `out/kv.bin` | 256 MiB |
| `out/kb.parquet` | 64 MiB |
| `out/ckt.json` (manifest only — no tensor payloads) | 16 MiB (per O.8) |
| `out/ckt.pt` (CUDA dense state_dict; sparse deferred to Epic 2) | 128 MiB |
| `out/ckt.npz` (MLX dense; sparse deferred to Epic 2) | 128 MiB |
| `out/ckt.svg` | 1 MiB |
| `out/*.json` (all other) | 4 MiB |
| `out/*.npz`, `out/*.npy` | 16 MiB |
| serve / lazarus-serve logs | 8 MiB (rotate) |

Enforced by `tests/ci/_size_gate.py` after every real-exec row. Also
enforces a global 2 GiB cap per matrix run.

---

## Appendix K — Dtype policy error path (R2 blocker #11)

Additional row:

| Case | Command | Expected |
|---|---|---|
| sm<8.0 + bf16 | `CHUK_LAZARUS_FORCE_SM=75 chuk-lazarus infer run --model $MODEL --backend cuda --dtype bfloat16 --prompt "$PROMPT"` | exit=4; stderr contains `BackendSMError: bfloat16 requires sm>=8.0, got 7.5`; no partial artifact left on disk |
| sm<8.0 + fp16 ok | `CHUK_LAZARUS_FORCE_SM=75 chuk-lazarus infer run --model $MODEL --backend cuda --dtype float16 --prompt "$PROMPT"` | exit=0; runs on fp16 path |

`CHUK_LAZARUS_FORCE_SM` is a test-only env read in `registry.py`
alongside the mock-CUDA shim; it overrides
`torch.cuda.get_device_capability` to the supplied major.minor.

---

## Appendix L — Long-context determinism (R2 blocker #12)

Greedy determinism is extended from "token-1 argmax" to "token-N argmax"
for long-context rows:

| Row | N (token prefix that MUST match cross-backend) |
|---|---|
| knowledge chat | first 32 generated tokens |
| serve `/v1/completions` (greedy) | first 32 |
| lazarus-serve WS streamed greedy | first 32 |
| introspect generate (`--max-new-tokens 16`) | all 16 |
| infer run (kv_direct, `--max-new-tokens 32`) | first 16 |

Enforced via `tests/ci/_compare.py::assert_token_prefix(ref_ids,
actual_ids, n=N)`. Divergence past N is allowed but logged.

---

## Appendix M — Serve process lifecycle (R2 blocker #13)

The `serve` / `lazarus-serve` real-exec rows add explicit PID tracking +
cleanup:

```bash
# launch
chuk-lazarus serve --model $MODEL --backend cuda --port 8123 \
  --pid-file out/serve.pid --log-file out/serve.log &
SERVE_PID=$!
trap 'kill -TERM "$(cat out/serve.pid 2>/dev/null || echo $SERVE_PID)" 2>/dev/null; wait 2>/dev/null' EXIT

# readiness gate (no arbitrary sleep)
timeout 30 bash -c 'until curl -fs localhost:8123/healthz >/dev/null; do sleep 0.2; done'

# request
curl -fsS -X POST localhost:8123/v1/completions \
  -H 'content-type: application/json' \
  -d "{\"prompt\":\"$PROMPT\",\"max_tokens\":16,\"seed\":0,\"temperature\":0}"

# teardown is automatic via trap; test asserts process gone
! kill -0 "$SERVE_PID" 2>/dev/null
```

A dedicated **cleanup row** asserts no leftover `chuk-lazarus serve`
processes after the matrix run: `pgrep -f 'chuk-lazarus (lazarus-)?serve'`
must return empty (exit 1).

---

## Appendix N — Model offline cache (R2 blocker #14)

CI runners are network-restricted. Model weights are pre-staged:

- Local cache dir: `$HF_HOME=/opt/hf-cache` on self-hosted runners,
  `$RUNNER_TEMP/hf-cache` on GitHub-hosted runners.
- Offline gate: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set for
  every matrix job. Any row that triggers a network fetch fails fast.
- Pre-stage script: `tests/ci/_prestage_models.sh` downloads
  `$MODEL` (Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct) once per
  runner image and verifies against a checksum pinned in
  `tests/ci/MODEL_SHA256SUMS`.
- CI job `prestage-models` runs weekly on a schedule to refresh the
  runner image cache; matrix jobs depend on it via `needs:`.
- Local dev: `make prestage-models` wraps the same script for humans.

---

## Appendix P — Self-hosted CUDA runner registration (tier-3 enablement)

Run 2 / Tier-3: the `parity-cuda` and `benchmark` jobs in
`.github/workflows/test.yml` are wired to run on
`runs-on: [self-hosted, cuda, sm120]`. These jobs are gated to
`workflow_dispatch` + `schedule` so push/PR builds do not require the
runner. Once the runner is registered, scheduled nightly runs (07:00 UTC,
cron `0 7 * * *`) will execute the parity-cuda arm automatically.

### P.1 Runner status (as of 2026-04-20)

| Repo | Runner count | Verification command |
|---|---|---|
| `TheartistisME/chuk-lazurus` (fork) | 0 | `gh api repos/TheartistisME/chuk-lazurus/actions/runners` → `{total_count: 0, runners: []}` |
| `chrishayuk/chuk-lazurus` (origin) | unknown | `gh api repos/chrishayuk/chuk-lazurus/actions/runners` → HTTP 403 (token lacks runner read permission) |

**ABSENT-pending-ops**: no self-hosted sm120 runner confirmed online.
Until ops registers one, `parity-cuda` and `benchmark` workflow_dispatch
invocations will queue indefinitely.

### P.2 Hardware + software prerequisites

- GPU with compute capability ≥ 8.0 (sm_120 target for Run 2: RTX 5090).
- NVIDIA driver + CUDA toolkit matching the torch wheel ABI (cu124+ for
  current torch stable).
- Ubuntu 22.04 / 24.04 LTS or Debian 12 (other distros untested).
- `uv` + Python 3.12 installable via `uv python install 3.12` (the
  workflow installs these per-job).
- Offline HF cache at `/opt/hf-cache` pre-staged per Appendix N.

### P.3 Registration flow

```bash
# 1. On the CUDA host, create a dedicated service account
sudo useradd -m -s /bin/bash gha-runner
sudo -iu gha-runner

# 2. Download the Actions runner package (pin a version)
mkdir -p ~/actions-runner && cd ~/actions-runner
RUNNER_VERSION=2.322.0  # check for latest at time of install
curl -fsSL -o actions-runner.tar.gz \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf actions-runner.tar.gz

# 3. Get a registration token from the repo admin UI:
#    Settings -> Actions -> Runners -> "New self-hosted runner"
#    OR via API:
#    gh api -X POST repos/<owner>/chuk-lazurus/actions/runners/registration-token
#
# 4. Configure the runner with the required labels
./config.sh --url https://github.com/<owner>/chuk-lazurus \
  --token <REG_TOKEN> \
  --name sm120-runner-01 \
  --labels self-hosted,cuda,sm120 \
  --work _work \
  --unattended

# 5. Install as systemd service (runs on boot, auto-restarts)
sudo ./svc.sh install gha-runner
sudo ./svc.sh start
sudo ./svc.sh status
```

### P.4 Post-registration verification

```bash
# From a machine with repo write access:
gh api repos/<owner>/chuk-lazurus/actions/runners \
  --jq '.runners[] | {name, status, labels: [.labels[].name]}'
# Expected: a runner with status="online" and labels including
# ["self-hosted","cuda","sm120"].

# Trigger a test run:
gh workflow run test.yml --ref main
gh run watch  # or: gh run list --workflow=test.yml
```

### P.5 Runner hygiene

- Restrict the runner to this repo (not org-wide) unless the same
  hardware serves multiple Lazarus repos.
- Disable `ACTIONS_ALLOW_UNSECURE_COMMANDS` (default off in runner 2.x).
- Configure runner autoscaling or a single always-on box; nightly
  schedule is 07:00 UTC, so a single runner with ≥ 12 GiB VRAM free is
  sufficient for Qwen2.5-1.5B rows.
- Pre-stage the HF model cache at `$HF_HOME=/opt/hf-cache` (Appendix N).
- Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the runner's global env for
  deterministic rows (§O.1).

---

## Appendix Q — Branch protection / required status checks

The parity + parity-cuda jobs self-enforce "no silently-skipped parity
tests" via the `Assert no parity tests were skipped` step inside
`.github/workflows/test.yml`. For that enforcement to block merges,
the jobs must be configured as **required status checks** in repo
branch protection.

### Q.1 Required checks list

Configure the following checks as required for the `main` branch
(and `develop`, if used as a pre-merge integration branch):

| Check name | Produced by | Blocks merge if |
|---|---|---|
| `test (macos-latest, 3.12, mlx, cpu)` | `test` job, matrix row | any test fails |
| `test (ubuntu-latest, 3.12, torch, cpu)` | `test` job, matrix row | any test fails |
| `parity (macos-latest, 3.12, mlx, cpu)` | `parity` job | parity fails OR skipped != 0 |
| `parity (ubuntu-latest, 3.12, torch, cpu)` | `parity` job | parity fails OR skipped != 0 |
| `lint` | `lint` job | ruff check fails |

The `parity-cuda` and `benchmark` jobs are **not** listed as
required-for-merge because they are gated to workflow_dispatch +
schedule and will not run on PR events. Those jobs serve as
post-merge nightly signals; regressions surface via the scheduled
run and are triaged out-of-band.

### Q.2 How to configure (GitHub web UI)

1. Navigate to `Settings → Branches → Branch protection rules`.
2. Edit the rule for `main` (or create one).
3. Enable "Require status checks to pass before merging".
4. Enable "Require branches to be up to date before merging".
5. Search the check list and add each required check from §Q.1.
6. Save.

### Q.3 How to configure (gh CLI)

```bash
gh api -X PUT repos/<owner>/chuk-lazurus/branches/main/protection \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test (macos-latest, 3.12, mlx, cpu)",
      "test (ubuntu-latest, 3.12, torch, cpu)",
      "parity (macos-latest, 3.12, mlx, cpu)",
      "parity (ubuntu-latest, 3.12, torch, cpu)",
      "lint"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null
}
JSON
```

### Q.4 Post-configuration verification

```bash
gh api repos/<owner>/chuk-lazurus/branches/main/protection \
  --jq '.required_status_checks.contexts'
```

Should list the five check names from §Q.1. If `parity*` checks
are missing, silent-green regressions become possible — the skipped=0
assertion inside the job is toothless without branch protection.

### Q.5 ABSENT (current state)

As of 2026-04-20 the current runtime environment has no token-scoped
access to verify branch protection on `chrishayuk/chuk-lazurus`.
Verification command:

```bash
gh api repos/chrishayuk/chuk-lazurus/branches/main/protection --jq '.required_status_checks'
```

Required action: repo admin applies §Q.3 (or UI equivalent) and
re-runs the verification command from §Q.4 to confirm.

---

## Appendix O — R3 revisions (round-2 review)

### O.1 CUDA determinism global invariant (blocker #1)

Extend §10 invariant #4. Every **numeric-compare** row (any row whose
acceptance criterion references atol/rtol/cosine/tokens) MUST export the
following before invoking the command under test:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=0              # keep async; determinism comes from algs
export PYTHONHASHSEED=0
export CHUK_LAZARUS_DETERMINISTIC=1        # flips the torch flags below inside the runtime
```

and the runtime (in `chuk_lazarus.inference.loader` when
`CHUK_LAZARUS_DETERMINISTIC=1`) applies:

```python
import torch
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
```

Without these, Appendix L token-N determinism claims and Appendix C
tensor tolerances are **invalid** and the row must be marked non-gating.
Non-determinism env drift (e.g. missing `CUBLAS_WORKSPACE_CONFIG`) fails
the job in `tests/ci/_env_gate.py` before any row runs.

Perf rows (Appendix H) are explicitly **exempt** — perf runs with
`CHUK_LAZARUS_DETERMINISTIC=0` (cudnn.benchmark=True) to avoid
suppressing legitimate kernel autotuning gains.

### O.2 EWS-0 dependency cross-reference (blocker #2)

The following deliverables are produced by workstream EWS-0 (see
[`03-workstreams.md`](./03-workstreams.md)), not by this matrix:

- `tests/fixtures/generate.sh` + `tests/fixtures/_builders/*` (Appendix A)
- `tests/ci/_prestage_models.sh` + `tests/ci/MODEL_SHA256SUMS` (Appendix N)
- `tests/ci/_perf_gate.py` + baseline storage plumbing (Appendix H)
- `tests/ci/_sentinels.py`, `tests/ci/_compare.py`, `tests/ci/_size_gate.py`,
  `tests/ci/_env_gate.py` (Appendices C, G, J, O.1)
- `tests/ci/matrix_*.yaml` CI job templates (§11)

> **Matrix execution blocked on EWS-0.** No row in §2–§9 is runnable in
> CI until EWS-0 lands these assets on `main` with green CI. Local dev
> rows are still runnable once `generate.sh` exists; see §O.11.

### O.3 Tightened Appendix C tolerances (blocker #3)

Override rows in Appendix C:

| Row | atol | rtol | cosine min | Rationale |
|---|---|---|---|---|
| cluster | 1e-3 | 1e-3 | 0.999 | centroid comparison after Hungarian match; k-means++ with fixed seed yields bit-stable init, only float rounding drift remains |
| probe | 1e-3 | 1e-3 | 0.999 | 2 epochs × Adam lr=1e-3 × N=128 samples ⇒ expected cross-backend weight delta ≤ ~5e-4 (bf16 matmul rounding) under deterministic flags; 1e-3 gives 2× headroom |
| classifier | 1e-3 | 1e-3 | 0.999 | same rationale as probe (2 epochs, N=128, Adam lr=1e-3) |
| uncertainty | 5e-3 (kept) | 1e-2 (kept) | n/a | stochastic by construction — 4 samples × bf16 softmax; tightening further would force >>4 samples and blow out CI budget |

All other tolerances in Appendix C remain as specified.

### O.4 Appendix F cross-backend gates (blocker #4)

Circuit round-trip rows use **dual gates**:

| Gate | Threshold |
|---|---|
| Logits L2 vs native-backend invoke | atol 1e-3, rtol 1e-2 |
| Top-1 token agreement across first 16 positions | 100% |

Both gates must pass. Top-1 is the primary signal for regressions that
float-L2 can mask (e.g. a systematic layer misordering that happens to
average out).

### O.5 Unified stochastic retry budget (blocker #5)

Appendix I normalized: **every stochastic row gets retry ≤ 2**.

| Row | Retries |
|---|---|
| uncertainty | 2 |
| cluster | 2 (was 1) |
| grpo | 2 |
| dpo | 2 (was 1) |
| probe / classifier | 2 (was 1) |
| serve / lazarus-serve transport | 3 (transport-only, not numeric — kept higher) |

### O.6 Bench split: perf and correctness rows (blocker #6)

Appendix E replaced by two independent rows per bench subcommand:

| Row | Gate | Threshold | Determinism env |
|---|---|---|---|
| `bench infer --mode correctness` | final-logits L2 vs dry-run baseline | atol 1e-3, rtol 1e-2 | `CHUK_LAZARUS_DETERMINISTIC=1` |
| `bench infer --mode perf` | tokens/s vs Appendix H baseline | ±5% | `CHUK_LAZARUS_DETERMINISTIC=0` |
| `bench train --mode correctness` | step-10 loss vs dry-run baseline | atol 1e-3, rtol 1e-2 | `CHUK_LAZARUS_DETERMINISTIC=1` |
| `bench train --mode perf` | step/s vs baseline | ±5% | `CHUK_LAZARUS_DETERMINISTIC=0` |

The rows are independently gating; a perf regression does not fail the
correctness row and vice-versa. NaN/Inf count = 0 applies to both.

### O.7 Perf baseline storage (blocker #7)

Replace Appendix H storage section with:

- **Short-term:** GitHub Actions artifact `perf-baseline-<sha>.json`,
  retention **90 days** (configured on `actions/upload-artifact@v4`).
- **Long-term:** S3 at
  `s3://chuk-lazarus-ci/baselines/{arch}/{backend}/{yyyy-mm-dd}.json`
  where `{arch}` ∈ `{sm120, sm90, sm86, m-series}` and `{backend}` ∈
  `{cuda, mlx}`. Lifecycle: Standard → Glacier at 180 days, expire at
  2 years. Example:
  `s3://chuk-lazarus-ci/baselines/sm120/cuda/2026-04-15.json`.
- **Detection script:** `tests/ci/_perf_gate.py` (seeded by EWS-0;
  see §O.2). Invocation:
  `python tests/ci/_perf_gate.py compare --current run.json \
      --baseline s3://chuk-lazarus-ci/baselines/sm120/cuda/latest.json \
      --tolerance 0.05`. Exits non-zero on regression.
- **Promotion:** main-branch green runs call
  `_perf_gate.py promote` which writes both the dated S3 key and the
  `latest.json` alias under the same `{arch}/{backend}/` prefix.

### O.8 Circuit artifact size caps split (blocker #8)

Override Appendix J row for circuits:

| Artifact | Max size | Storage policy |
|---|---|---|
| `out/ckt.json` (text manifest: keys, shapes, dtypes, metadata) | 16 MiB | dense text; no tensor payloads — tensors live in companion binary |
| `out/ckt.pt` (CUDA export) | 128 MiB | dense `state_dict`; sparse tensors represented as dense in Epic 1 (sparse export deferred to Epic 2 — tracked in `03-workstreams.md` EWS-9) |
| `out/ckt.npz` (MLX export) | 128 MiB | dense; same policy |

`out/ckt.json` > 16 MiB indicates tensor payloads leaked into the
manifest — fail the row with a clear error rather than silently
truncating.

### O.9 Serve zombie cleanup + dynamic port (blocker #9)

Prepend to the Appendix M launch block:

```bash
# pre-matrix zombie sweep (idempotent; safe on fresh runners)
pkill -f 'chuk-lazarus (lazarus-)?serve' 2>/dev/null || true
sleep 0.2
pgrep -f 'chuk-lazarus (lazarus-)?serve' && { echo "zombie serve survived sweep"; exit 1; } || true

# dynamic port with fallback
BASE_PORT=8123
for attempt in 0 1 2 3 4; do
  PORT=$((BASE_PORT + attempt))
  if ! ss -ltn "sport = :$PORT" | grep -q LISTEN; then break; fi
done
[ "$attempt" = 4 ] && ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "no free port in $BASE_PORT..$((BASE_PORT+4))"; exit 1; }
```

All subsequent `curl` / WS calls use `$PORT` instead of the hardcoded
`8123`. The cleanup-assertion row (Appendix M tail) now runs the same
`pkill` and then verifies no process survives 1s later.

### O.10 Long-context N=32 rationale (blocker #10)

Appendix L N=32 is empirical, based on KV-cache bf16 precision drift
observed during the long-context PR (`#25`, commit `0de10aa`). Greedy
decoding on Qwen2.5-0.5B-Instruct with deterministic flags shows:

| Position | Cross-backend argmax-agreement rate (observed over 20 prompts) |
|---|---|
| 1–32 | 100% |
| 33–64 | ~99% (1 divergence across 20 prompts × 32 positions) |
| 65–128 | ~96% |

N=32 sits at the last position where 100% agreement held for every
prompt in the sample. Beyond that, bf16 matmul rounding accumulates
enough to flip rare near-tie argmaxes. If a future test regresses the
32-token prefix, that is a real correctness signal, not a precision
quirk. Source: manual run log attached to PR #25; reproducer lives at
`tests/ci/_longctx_repro.py` (delivered by EWS-0).

### O.11 HF_HUB_OFFLINE scope (blocker #11)

Offline gating is **CI-only**, not a dev-machine constraint:

- CI job env (set at the job level in every `tests/ci/matrix_*.yaml`):
  ```yaml
  env:
    HF_HUB_OFFLINE: "1"
    TRANSFORMERS_OFFLINE: "1"
    HF_HOME: ${{ runner.temp }}/hf-cache     # or /opt/hf-cache on self-hosted
  ```
- Dev machines: **unrestricted**. No env vars set by default. `make
  prestage-models` is available but optional — developers can fetch
  on demand. The `Makefile` target explicitly does NOT export the
  offline vars.
- Guardrail: `tests/conftest.py` logs a warning (not an error) if
  `HF_HUB_OFFLINE` is unset and the test run is tagged `ci` via
  `CHUK_LAZARUS_CI=1`. The warning becomes an error only when
  `CHUK_LAZARUS_CI=1`.

---

## 12. Open items (defer to Epic 2)

- Multi-GPU (`--device cuda:0,cuda:1`) — tracked separately.
- ROCm / Intel XPU — out of scope.
- Quantized inference (`--quant int4`) cross-backend parity — out of scope
  for Epic 1 matrix; add rows once Epic 2 lands.
