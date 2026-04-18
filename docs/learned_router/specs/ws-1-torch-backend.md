# WS-1 — Torch Backend Completion

**Mission:** chuk-lazurus-n7k  **Batch:** 1 (parallel with WS-2)  **Owner:** single teammate

## Scope
Complete the existing `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` against the abstract
interface in `base.py`, extend registry aliasing, and back every public method with a CPU test.
MLX path stays untouched.

## Exclusive file ownership (edit only these)
- `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` (complete)
- `src/chuk_lazarus/models_v2/core/backend/registry.py` (aliasing additions only — no behaviour change for MLX paths)
- `tests/models_v2/core/backend/test_torch_backend.py` (expand)
- `tests/models_v2/core/backend/test_registry.py` (add torch-alias cases only; do not touch MLX cases)

**DO NOT TOUCH** any other file, especially `base.py`, `types.py`, `mlx_backend.py`, or `__init__.py`.

## Requirements
1. `TorchBackend` must implement every abstract method declared in `base.py`. Audit the current
   file — it is already close; verify each abstract method has a concrete implementation that
   matches the signature (`name`, `device`, `zeros`, `ones`, `randn`, `arange`, `from_numpy`,
   `to_numpy`, `matmul`, `softmax`, `relu`, `silu`, `gelu`, `tanh`, `sigmoid`, `layer_norm`,
   `rms_norm`, `reshape`, `transpose`, `concatenate`, `split`, `scaled_dot_product_attention`,
   `create_causal_mask`, `stop_gradient`, `eval`).
2. `TorchBackend(device=...)`:
   - `"cuda"`  → uses `cuda:0` if available, silently falls back to `"cpu"` otherwise.
   - `"cpu"`   → always resolves to `"cpu"`.
   - `"mps"`   → uses MPS if `torch.backends.mps.is_available()`, else `"cpu"`.
3. Dtype policy:
   - CUDA SM ≥ 8 and `check_sm=True` → `torch.bfloat16`; SM < 8 → `torch.float16`.
   - CPU or MPS → `torch.float32`.
4. Registry:
   - `get_backend("torch")` and `get_backend("pytorch")` must BOTH return a working `TorchBackend`.
   - Aliasing: `BackendType("pytorch")` is not valid (enum only has `"torch"`). Do alias mapping
     inside `get_backend` / `set_backend` — normalise `"pytorch"` → `BackendType.TORCH` before
     `BackendType(name)` is called. Document it with a short comment.
   - No change to MLX auto-detection logic.

## Tests (CPU-only by default; CUDA tests skipped when unavailable)
Expand `tests/models_v2/core/backend/test_torch_backend.py` to cover every abstract method and the
device-resolution branches. Add three CUDA-guarded tests:
- `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`
- one for SM ≥ 8 → bfloat16 (skip if capability major < 8)
- one for SM < 8 → float16  (skip if capability major >= 8)
- one for `TorchBackend(device="cuda")` falling back to CPU when CUDA absent (patch
  `torch.cuda.is_available` to `False`)

Add `tests/models_v2/core/backend/test_registry.py::TestTorchAlias` with:
- `get_backend("torch")` returns `TorchBackend`
- `get_backend("pytorch")` returns `TorchBackend`
- set_backend flow with both names does not raise

## Quality gate before closing
```
uv run pytest tests/models_v2/core/backend/test_torch_backend.py tests/models_v2/core/backend/test_registry.py -q
uv run ruff check src/chuk_lazarus/models_v2/core/backend/torch_backend.py src/chuk_lazarus/models_v2/core/backend/registry.py
```
Both must be green. Also run the AUS3000 5-case smoke (shared smoke, not WS-1-specific):
```
uv run python tools/evaluate_aus3000_variant.py --mode single_pass_gate --device cpu --max-cases 5
```
No regression vs baseline.

## Deliverable format
- Diff touches only the 4 files above.
- Every new test function has a docstring stating the invariant under test.
- `vee record insight --title "WS-1 torch backend completion" --body "<learnings>" --tag chuk-lazurus-n7k --tag ws-1`
- `vee session close --handoff` for review-loop hand-off.
