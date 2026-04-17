# Learned Window Router Architecture

## Overview

Torch parity was added as a side-by-side stack so the learned window router can train and infer on PyTorch without replacing the existing MLX implementation. The torch path is selected through the backend registry, while the MLX path stays intact for Apple Silicon and other existing callers. This mirrors the repo's frozen torch-parity pattern in `torch_store.py` and `torch_query.py`, where new torch-native code lives alongside the existing knowledge-store routing code instead of rewriting it.

## Why the torch files sit next to the MLX files

The repository already separates backend selection from model and trainer implementation. The registry resolves `"torch"`, `"pytorch"`, or `"mlx"` and returns the matching backend, so the caller chooses the stack at construction time. Keeping `torch_*.py` beside the MLX twins preserves that split: MLX imports remain untouched, torch code can stay torch-native, and the two stacks can share the same higher-level contract without sharing source files.

## Torch-parity pattern

The learned router follows the same pattern already used by `torch_store.py` and `torch_query.py`: a torch-native twin is introduced for each MLX-facing layer, with the same role and adjacent placement.

| Layer | MLX file | Torch file |
|---|---|---|
| Backend | `src/chuk_lazarus/models_v2/core/backend/mlx_backend.py` | `src/chuk_lazarus/models_v2/core/backend/torch_backend.py` |
| Linear classifier | `src/chuk_lazarus/models_v2/models/classifiers/linear.py` | `src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py` |
| MLP classifier | `src/chuk_lazarus/models_v2/models/classifiers/mlp.py` | `src/chuk_lazarus/models_v2/models/classifiers/torch_mlp.py` |
| Token classifier | `src/chuk_lazarus/models_v2/models/classifiers/token.py` | `src/chuk_lazarus/models_v2/models/classifiers/torch_token_embedding.py` |
| Base trainer | `src/chuk_lazarus/training/base_trainer.py` | `src/chuk_lazarus/training/torch/torch_base_trainer.py` |
| Classification trainer | `src/chuk_lazarus/training/classification_trainer.py` | `src/chuk_lazarus/training/torch/torch_classification_trainer.py` |

The same separation holds for knowledge routing: `route.py`, `torch_store.py`, and `torch_query.py` define the frozen routing behavior that the learned router builds on, but does not replace.

## Registry and backend selection

`src/chuk_lazarus/models_v2/core/backend/registry.py` owns backend selection. It normalizes `"pytorch"` to `"torch"`, auto-detects the default backend from platform and device, and constructs `TorchBackend` or `MLXBackend` accordingly. That registry is the coupling point between backend choice and the rest of the stack, so the torch classifier and trainer modules can remain independent of MLX imports.

## Route and store boundary

The learned router does not change the exact-routing code. `src/chuk_lazarus/inference/context/knowledge/route.py`, `torch_store.py`, and `torch_query.py` remain the routing and store/query reference points, and the new torch training stack consumes their outputs. The architecture assumes the store can be loaded through `TorchKnowledgeStore` and routed through the existing exact, TF-IDF, and keyword logic before any learned model is applied.

## Files-touched map

The required new files are:

- `src/chuk_lazarus/models_v2/core/backend/torch_backend.py`
- `src/chuk_lazarus/models_v2/models/classifiers/torch_linear.py`
- `src/chuk_lazarus/models_v2/models/classifiers/torch_mlp.py`
- `src/chuk_lazarus/models_v2/models/classifiers/torch_token_embedding.py`
- `src/chuk_lazarus/training/torch/torch_base_trainer.py`
- `src/chuk_lazarus/training/torch/torch_classification_trainer.py`
- `tools/train_window_router.py`
- `tools/_window_router/`

The orchestration plan splits the work into WS-1 through WS-5: backend, classifier primitives, torch training, generic window-router tooling, and AUS3000 evaluation. The window-router tool is generic over any store and benchmark fixture pair, with AUS3000 as the first concrete caller rather than a hard-coded special case.

## Resulting flow

`tools/train_window_router.py` exposes `build-dataset`, `train`, and `eval`. It loads a `TorchKnowledgeStore`, builds router training pairs, trains a torch MLP classifier through the torch trainer stack, and evaluates against a benchmark fixture with TF-IDF baseline reporting. The same flow is intended to work for any compatible store and benchmark pair, not just AUS3000.
