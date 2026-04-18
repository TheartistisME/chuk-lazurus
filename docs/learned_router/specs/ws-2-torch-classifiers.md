# WS-2 — Torch Classifier Parity Modules

**Mission:** chuk-lazurus-n7k  **Batch:** 1 (parallel with WS-1)  **Owner:** single teammate

## Scope
Provide torch-native parity for the three MLX classifier primitives. Mirror the MLX constructor
signatures exactly so downstream tools can swap backends by file selection. Each module is
torch-native (written from scratch against `torch.nn`), not an MLX port.

## Exclusive file ownership (edit only these)
- `src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py`          (new)
- `src/chuk_lazarus/models_v2/models/classifiers/torch_mlp.py`             (new)
- `src/chuk_lazarus/models_v2/models/classifiers/torch_token_embedding.py` (new)
- `tests/models_v2/models/classifiers/test_torch_linear.py`          (new)
- `tests/models_v2/models/classifiers/test_torch_mlp.py`             (new)
- `tests/models_v2/models/classifiers/test_torch_token_embedding.py` (new)

**DO NOT TOUCH** MLX twins (`mlp.py`, `linear.py`, `sequence.py`, `token.py`, `factory.py`) or the
package `__init__.py`. These torch classes are importable by absolute path only — no re-export.

## Required signatures (mirror MLX)
```python
class TorchLinearClassifier(torch.nn.Module):
    def __init__(self, input_size: int, num_labels: int = 1, bias: bool = True): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...  # (B, input_size) -> (B, num_labels)

class TorchMLPClassifier(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_labels: int = 1,
                 activation: str | ActivationType = "gelu", bias: bool = True): ...
    # structure: Linear(input->hidden) -> act -> Linear(hidden->input) -> Linear(input->num_labels)
    # (mirrors the chuk-mlx MLP component: hidden_size -> intermediate -> hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class TorchTokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, tie_weights: bool = False): ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...    # (B, T) -> (B, T, hidden)
    def as_output_projection(self) -> torch.nn.Linear: ...  # when tie_weights=True, shares weight
```
- `activation` accepts `ActivationType.{GELU,RELU,SILU}` (from `models_v2.core.enums`) OR the
  lowercase string. Use `torch.nn.functional.{gelu,relu,silu}`.
- Default dtype: `torch.float32`. No device placement inside the module; caller decides.
- No emoji, full type hints, module docstring, ≤ 120 lines each.

## Tests (CPU only, all mandatory)
Each module: basic init, output shape for (B,F) or (B,T), gradient flow (autograd `backward()`
populates `.grad` on every trainable param), activation variants for MLP, `bias=False` for
Linear/MLP, `tie_weights=True` for TokenEmbedding (output projection weight is the embedding
weight). Add one shape-parity test per module comparing the torch output `shape` to the MLX
twin's output shape under identical constructor args — guard the import with
`pytest.importorskip("mlx.core")` so Linux CI doesn't fail on missing MLX.

## Quality gate before closing
```
uv run pytest tests/models_v2/models/classifiers/test_torch_linear.py \
              tests/models_v2/models/classifiers/test_torch_mlp.py \
              tests/models_v2/models/classifiers/test_torch_token_embedding.py -q
uv run ruff check src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py \
                  src/chuk_lazarus/models_v2/models/classifiers/torch_mlp.py \
                  src/chuk_lazarus/models_v2/models/classifiers/torch_token_embedding.py
```
Both green. MLX twin tests remain untouched — run them to prove no regression.

## Deliverable format
- `vee record insight --title "WS-2 torch classifiers" --tag chuk-lazurus-n7k --tag ws-2`
- `vee session close --handoff` for review-loop hand-off.
