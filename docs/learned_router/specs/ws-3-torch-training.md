# WS-3 — Torch Training Stack

**Mission:** chuk-lazurus-n7k  **Batch:** 2  **Depends on:** WS-2  **Owner:** single teammate

## Scope
Build a torch-native classification training stack that mirrors the MLX
`BaseTrainer` / `ClassificationTrainer` contract without modifying either. Trainer lives in its
own subpackage so import order cannot trigger MLX.

## Exclusive file ownership (edit only these)
- `src/chuk_lazarus/training/torch/__init__.py`                    (new — re-export classes below)
- `src/chuk_lazarus/training/torch/torch_base_trainer.py`          (new)
- `src/chuk_lazarus/training/torch/torch_classification_trainer.py` (new)
- `tests/training/torch/__init__.py`                               (new, empty)
- `tests/training/torch/test_torch_base_trainer.py`                (new)
- `tests/training/torch/test_torch_classification_trainer.py`      (new)

**DO NOT TOUCH** `src/chuk_lazarus/training/base_trainer.py` or `classification_trainer.py`. No
edits to `src/chuk_lazarus/training/__init__.py` — keep the torch subpackage invisible to the MLX
import path.

## Required contract
```python
@dataclass
class TorchTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    batch_size: int = 32
    num_epochs: int = 1
    log_interval: int = 10
    checkpoint_interval: int = 500
    checkpoint_dir: str = "./checkpoints"
    device: str = "cpu"          # "cuda"|"cpu"|"mps"
    cosine_schedule: bool = False

class TorchBaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, config: TorchTrainerConfig): ...
    @abstractmethod
    def compute_loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]: ...
    @abstractmethod
    def get_train_batches(self, dataset) -> Iterator[dict]: ...
    def train(self, dataset, num_epochs=None, callback=None) -> list[dict]: ...
    def save_checkpoint(self, name: str) -> Path: ...   # torch.save(state_dict + config)
    def load_checkpoint(self, path: str | Path) -> None: ...
    def clip_gradients(self, max_norm: float) -> None:  # torch.nn.utils.clip_grad_norm_

class TorchClassificationTrainer(TorchBaseTrainer):
    # encoder: optional callable(text) -> list[float]; features field used if present
    def __init__(self, model, encoder, config: TorchTrainerConfig): ...
    def compute_loss(self, batch): # CrossEntropyLoss; returns (loss, {"loss": .., "accuracy": ..})
    def get_train_batches(self, dataset): # stacks features -> (B, F) float32 on config.device
```
- Optimizer: `torch.optim.AdamW(model.parameters(), lr, weight_decay)`.
- LR sched: `torch.optim.lr_scheduler.CosineAnnealingLR` when `cosine_schedule=True`.
- Grad clip: `torch.nn.utils.clip_grad_norm_(..., max_norm)` before `optimizer.step()`.
- Checkpoint file: `{checkpoint_dir}/{name}.pt` containing `{"state_dict": ..., "config": ...}`.
- `accuracy` is computed on the current batch.

## Tests (CPU only)
Mandatory cases, all on real torch tensors (no mocks for tensor ops):
- **Smoke:** train a 2-class `TorchMLPClassifier` (from WS-2) on a synthetic linearly-separable
  2-D dataset (`make_blobs` equiv via numpy) for ~50 steps → final batch accuracy ≥ 0.99.
- **Checkpoint round-trip:** train 5 steps, save, load into a fresh `TorchMLPClassifier`, verify
  parameter tensors equal (use `torch.allclose`).
- **Grad clip:** pass a degenerate batch that explodes gradients; verify the clipped global norm
  is ≤ `max_grad_norm + 1e-6`.
- **Encoder path:** trainer accepts samples with `text` + encoder; trainer accepts samples with
  `features` and no encoder.
- **Metrics contract:** `compute_loss` returns `(tensor, {"loss": float, "accuracy": float})`.

## Quality gate before closing
```
uv run pytest tests/training/torch/ -q
uv run ruff check src/chuk_lazarus/training/torch/
uv run pytest tests/models_v2/core/backend/test_torch_backend.py tests/models_v2/models/classifiers/ -q  # WS-1/WS-2 unchanged
```
All green. MLX training tests stay green.

## Deliverable format
- `vee record insight --title "WS-3 torch training stack" --tag chuk-lazurus-n7k --tag ws-3`
- `vee session close --handoff`
