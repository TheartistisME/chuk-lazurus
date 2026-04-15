"""CI gate: backend-dispatch in-scope modules must not import mlx/mlx_lm at
top level.

See `docs/refactor/dual-backend-cuda/02-workstreams.md` §9 (Epic 1) and
`docs/refactor/dual-backend-cuda-all-buckets/03-workstreams.md` §EWS-0
(Epic 2 rename of the shared in-scope list).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

BACKEND_IN_SCOPE: list[str] = [
    "src/chuk_lazarus/inference/loader.py",
    "src/chuk_lazarus/inference/unified.py",
    "src/chuk_lazarus/inference/__init__.py",
    "src/chuk_lazarus/inference/generator.py",
    "src/chuk_lazarus/inference/context/kv_generator.py",
    # EWS-1 (Wave 0.5) — infer bucket extensions
    "src/chuk_lazarus/inference/chat.py",
    "src/chuk_lazarus/inference/generation.py",
    "src/chuk_lazarus/inference/backends/__init__.py",
    "src/chuk_lazarus/inference/backends/base.py",
    "src/chuk_lazarus/inference/backends/registry.py",
    "src/chuk_lazarus/inference/backends/torch_runtime.py",
    "src/chuk_lazarus/inference/backends/mlx_runtime.py",
    "src/chuk_lazarus/inference/backends/types.py",
    "src/chuk_lazarus/inference/backends/storage.py",
    "src/chuk_lazarus/inference/backends/prefetch.py",
    "src/chuk_lazarus/inference/virtual_experts/registry.py",
    "src/chuk_lazarus/inference/virtual_experts/router.py",
    "src/chuk_lazarus/cli/commands/infer/__init__.py",
    "src/chuk_lazarus/cli/commands/infer/run.py",
    "src/chuk_lazarus/cli/commands/infer/_types.py",
    "src/chuk_lazarus/cli/_parsers/_infer.py",
    "src/chuk_lazarus/cli/_parsers/_infer_run.py",
    # EWS-4 — knowledge bucket
    "src/chuk_lazarus/cli/commands/knowledge/__init__.py",
    "src/chuk_lazarus/cli/commands/knowledge/_common.py",
    "src/chuk_lazarus/cli/commands/knowledge/_build.py",
    "src/chuk_lazarus/cli/commands/knowledge/_query.py",
    "src/chuk_lazarus/cli/commands/knowledge/_chat.py",
    "src/chuk_lazarus/cli/_parsers/_knowledge.py",
    # EWS-8 — serve bucket
    "src/chuk_lazarus/server/__init__.py",
    "src/chuk_lazarus/server/engine.py",
    "src/chuk_lazarus/server/app.py",
    "src/chuk_lazarus/server/cli.py",
    "src/chuk_lazarus/server/routers/__init__.py",
    "src/chuk_lazarus/server/routers/openai.py",
    "src/chuk_lazarus/server/routers/ollama.py",
    "src/chuk_lazarus/server/routers/anthropic.py",
    "src/chuk_lazarus/server/schemas/__init__.py",
    "src/chuk_lazarus/server/schemas/internal.py",
    "src/chuk_lazarus/server/schemas/openai.py",
    "src/chuk_lazarus/server/schemas/ollama.py",
    "src/chuk_lazarus/server/schemas/anthropic.py",
    "src/chuk_lazarus/cli/_parsers/_serve.py",
    # EWS-2 — context prefill bucket
    "src/chuk_lazarus/cli/commands/context/prefill/__init__.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_cmd.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_checkpoints.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_compass.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_darkspace.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_interval.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_kv_route.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_mode7_calibrate.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_npz.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_pages.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_progress.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_restore.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_save.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_sparse.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_surprise.py",
    "src/chuk_lazarus/cli/commands/context/prefill/_vec_inject.py",
    "src/chuk_lazarus/cli/_parsers/_context_prefill.py",
    "src/chuk_lazarus/inference/context/research/vec_inject/_primitives.py",
    "src/chuk_lazarus/inference/context/research/vec_inject/providers/__init__.py",
    "src/chuk_lazarus/inference/context/research/vec_inject/providers/_index_format.py",
    "src/chuk_lazarus/inference/context/research/vec_inject/providers/_local_file.py",
    # EWS-11 — train datagen + generate top-level
    "src/chuk_lazarus/cli/commands/train/datagen.py",
    "src/chuk_lazarus/cli/commands/data/batching/generate.py",
    "src/chuk_lazarus/datasets/benchmarks.py",
    "src/chuk_lazarus/data/generators/__init__.py",
    "src/chuk_lazarus/data/generators/math_generator.py",
    "src/chuk_lazarus/data/generators/types.py",
    # EWS-3 — context generate bucket
    "src/chuk_lazarus/cli/commands/context/generate/__init__.py",
    "src/chuk_lazarus/cli/commands/context/generate/_cmd.py",
    "src/chuk_lazarus/cli/commands/context/generate/_grounding.py",
    "src/chuk_lazarus/cli/commands/context/generate/_iterative.py",
    "src/chuk_lazarus/cli/commands/context/generate/_mode7.py",
    "src/chuk_lazarus/cli/commands/context/generate/_plain.py",
    "src/chuk_lazarus/cli/commands/context/generate/_probe_driven.py",
    "src/chuk_lazarus/cli/commands/context/generate/_probe_rerank.py",
    "src/chuk_lazarus/cli/commands/context/generate/_probes.py",
    "src/chuk_lazarus/cli/commands/context/generate/_resolve.py",
    "src/chuk_lazarus/cli/commands/context/generate/_unified.py",
    "src/chuk_lazarus/cli/commands/context/calibrate_frames.py",
    "src/chuk_lazarus/cli/_parsers/_context_generate.py",
    "src/chuk_lazarus/inference/context/unlimited_engine.py",
    # EWS-13 — tokenizer bucket (CLI-side only; data/tokenizers/** transitively
    # imports ``chuk_lazarus.data.__init__`` which currently loads ``mlx.core``
    # via ``base_dataset.py`` — that is EWS-12 scope, not EWS-13's.)
    "src/chuk_lazarus/cli/_parsers/_tokenizer.py",
    "src/chuk_lazarus/cli/commands/tokenizer/__init__.py",
    "src/chuk_lazarus/cli/commands/tokenizer/_types.py",
    "src/chuk_lazarus/cli/commands/tokenizer/_utils.py",
    "src/chuk_lazarus/cli/commands/tokenizer/analyze/__init__.py",
    "src/chuk_lazarus/cli/commands/tokenizer/core/__init__.py",
    "src/chuk_lazarus/cli/commands/tokenizer/curriculum/__init__.py",
    "src/chuk_lazarus/cli/commands/tokenizer/health/__init__.py",
    "src/chuk_lazarus/cli/commands/tokenizer/health/benchmark.py",
    "src/chuk_lazarus/cli/commands/tokenizer/health/doctor.py",
    "src/chuk_lazarus/cli/commands/tokenizer/health/fingerprint.py",
    # EWS-12 — data bucket (batchplan, batching, lengths; batching/generate.py is EWS-11)
    "src/chuk_lazarus/cli/_parsers/_data.py",
    "src/chuk_lazarus/cli/commands/data/__init__.py",
    "src/chuk_lazarus/cli/commands/data/_types.py",
    "src/chuk_lazarus/cli/commands/data/_utils.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/__init__.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/_types.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/build.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/info.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/shard.py",
    "src/chuk_lazarus/cli/commands/data/batchplan/verify.py",
    "src/chuk_lazarus/cli/commands/data/batching/__init__.py",
    "src/chuk_lazarus/cli/commands/data/batching/_types.py",
    "src/chuk_lazarus/cli/commands/data/batching/analyze.py",
    "src/chuk_lazarus/cli/commands/data/batching/histogram.py",
    "src/chuk_lazarus/cli/commands/data/batching/suggest.py",
    "src/chuk_lazarus/cli/commands/data/lengths/__init__.py",
    "src/chuk_lazarus/cli/commands/data/lengths/_types.py",
    "src/chuk_lazarus/cli/commands/data/lengths/build.py",
    "src/chuk_lazarus/cli/commands/data/lengths/stats.py",
    # EWS-5 — introspect framework core
    "src/chuk_lazarus/introspection/hooks.py",
    "src/chuk_lazarus/introspection/analyzer/core.py",
    "src/chuk_lazarus/introspection/logit_lens.py",
    "src/chuk_lazarus/introspection/patcher.py",
    "src/chuk_lazarus/introspection/accessor.py",
    "src/chuk_lazarus/introspection/attention.py",
    "src/chuk_lazarus/introspection/layer_analysis.py",
    "src/chuk_lazarus/introspection/virtual_expert.py",
    "src/chuk_lazarus/introspection/_backend_dispatch.py",
    # EWS-9 — train infrastructure (base + losses + utils + optim)
    "src/chuk_lazarus/training/_backend_math.py",
    "src/chuk_lazarus/training/base_trainer.py",
    "src/chuk_lazarus/training/batch_processor.py",
    "src/chuk_lazarus/training/classification_trainer.py",
    "src/chuk_lazarus/training/epoch_processor.py",
    "src/chuk_lazarus/training/epoch_processor_utils.py",
    "src/chuk_lazarus/training/schedulers.py",
    "src/chuk_lazarus/training/losses/__init__.py",
    "src/chuk_lazarus/training/losses/dpo_loss.py",
    "src/chuk_lazarus/training/losses/dual_reward_loss.py",
    "src/chuk_lazarus/training/losses/grpo_loss.py",
    "src/chuk_lazarus/training/losses/ppo_loss.py",
    "src/chuk_lazarus/training/losses/sft_loss.py",
    "src/chuk_lazarus/training/utils/__init__.py",
    "src/chuk_lazarus/training/utils/advantage.py",
    "src/chuk_lazarus/training/utils/kl_divergence.py",
    "src/chuk_lazarus/training/utils/log_probs.py",
    "src/chuk_lazarus/utils/model_adapter.py",
    "src/chuk_lazarus/utils/optimizer_adapter.py",
    "src/chuk_lazarus/utils/optimizer_loader.py",
]

# Backwards-compatible alias.  Epic 2 workstreams append to
# ``BACKEND_IN_SCOPE``; pre-Epic-2 imports of ``EPIC_1_IN_SCOPE`` keep working.
EPIC_1_IN_SCOPE = BACKEND_IN_SCOPE


def _path_to_module(rel: str) -> str:
    assert rel.startswith("src/")
    mod = rel[len("src/") :].replace("/", ".")
    if mod.endswith("/__init__.py") or mod.endswith(".__init__.py"):
        mod = mod.rsplit(".__init__", 1)[0]
    if mod.endswith(".py"):
        mod = mod[:-3]
    return mod


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _try_has_import_error(node: ast.Try) -> bool:
    for h in node.handlers:
        t = h.type
        if t is None:
            return True
        if isinstance(t, ast.Name) and t.id == "ImportError":
            return True
        if isinstance(t, ast.Tuple):
            for elt in t.elts:
                if isinstance(elt, ast.Name) and elt.id == "ImportError":
                    return True
    return False


def _iter_toplevel_imports(body: list[ast.stmt]):
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If) and _is_type_checking_guard(node):
            continue
        elif isinstance(node, ast.Try) and _try_has_import_error(node):
            continue
        elif isinstance(node, ast.ClassDef):
            yield from _iter_toplevel_imports(node.body)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    yield child


def _import_names(node):
    if isinstance(node, ast.Import):
        return [(alias.name, node.lineno) for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "", node.lineno)]
    return []


@pytest.mark.parametrize("path", BACKEND_IN_SCOPE)
def test_no_mlx_ast_imports(path: str) -> None:
    src_path = REPO_ROOT / path
    src = src_path.read_text()
    tree = ast.parse(src)
    for node in _iter_toplevel_imports(tree.body):
        for name, lineno in _import_names(node):
            root = name.split(".")[0]
            assert root not in {"mlx", "mlx_lm"}, (
                f"{path}: top-level import of {name!r} at line {lineno}"
            )


@pytest.mark.parametrize("path", BACKEND_IN_SCOPE)
def test_no_mlx_runtime_imports(path: str) -> None:
    module = _path_to_module(path)
    env = os.environ.copy()
    env["CHUK_BACKEND"] = "torch"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys, importlib\n"
        f"importlib.import_module({module!r})\n"
        "bad = [k for k in sys.modules if k == 'mlx' or k == 'mlx_lm']\n"
        "assert not bad, bad\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, (
        f"{path}: runtime import of mlx/mlx_lm\nstdout={res.stdout}\nstderr={res.stderr}"
    )
