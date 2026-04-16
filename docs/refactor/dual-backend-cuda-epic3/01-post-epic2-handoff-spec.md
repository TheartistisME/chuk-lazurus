You are taking over post-Epic-2 CUDA enablement for Lazarus in:

/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus

Read AGENTS.md and use bd. Do not revert unrelated changes. The worktree is dirty and the branch is already ahead. Assume other agent work may already be present; work with it.

Current baseline
- Epic-2 import hygiene is done.
- This gate is green:
  uv run python -m pytest tests/ci/test_no_top_level_mlx.py -q
  Result: 324 passed
- Recent green suites:
  - uv run python -m pytest tests/cli/commands/context/prefill/test_cmd_backend.py tests/cli/commands/context/generate/test_cmd_backend.py tests/cli/commands/context/test_calibrate_frames_backend.py -q
    Result: 20 passed
  - uv run python -m pytest tests/cli/commands/train/test_import_chain_backend.py tests/training/trainers/test_sft_backend.py tests/training/trainers/test_dpo_backend.py tests/training/trainers/test_grpo_backend.py tests/training/trainers/test_dual_reward_backend.py -q
    Result: 21 passed
  - uv run python -m pytest tests/introspection/analyzer/test_init_backend.py tests/introspection/test_analyzer_core_backend.py -q
    Result: 4 passed
  - uv run python -m pytest tests/server/test_optional_fastapi_backend.py -q
    Result: 3 passed
  - CHUK_BACKEND=torch PYTHONPATH=src uv run python - <<'PY'
    import importlib
    mods = [
      'chuk_lazarus.server',
      'chuk_lazarus.server.engine',
      'chuk_lazarus.server.app',
      'chuk_lazarus.server.cli',
      'chuk_lazarus.server.routers',
      'chuk_lazarus.server.routers.openai',
      'chuk_lazarus.server.routers.ollama',
      'chuk_lazarus.server.routers.anthropic',
      'chuk_lazarus.server.schemas',
      'chuk_lazarus.server.schemas.internal',
      'chuk_lazarus.server.schemas.openai',
      'chuk_lazarus.server.schemas.ollama',
      'chuk_lazarus.server.schemas.anthropic',
    ]
    for name in mods:
        importlib.import_module(name)
    print('ok')
    PY
    Result: ok
- Core package status:
  - parser imports are torch-clean
  - server imports are torch-clean under uv/venv
  - analyzer imports are torch-clean
  - knowledge/prefill/context import gates are torch-clean

Important current changes already landed in worktree
- parser package root and introspect parser lazy dispatch were fixed
- analyzer package root is lazy
- knowledge and prefill leaf modules no longer import mlx at top level
- context prefill `_cmd.py` is import-clean and now returns early on torch instead of crashing at import time
- server package is now lazy/optional-dep safe:
  - src/chuk_lazarus/server/__init__.py
  - src/chuk_lazarus/server/_compat.py
  - src/chuk_lazarus/server/app.py
  - src/chuk_lazarus/server/cli.py
  - src/chuk_lazarus/server/routers/openai.py
  - src/chuk_lazarus/server/routers/ollama.py
  - src/chuk_lazarus/server/routers/anthropic.py

Environment facts
- torch is installed: 2.10.0+cu128
- transformers is installed: 4.46.3
- safetensors is installed
- peft is NOT installed
- repo venv does NOT have fastapi by default, so actual serve runtime needs optional server deps installed if you want full serve end-to-end validation

What remains
1. Turn the current torch placeholders/blockers into real functionality.
   - src/chuk_lazarus/cli/commands/context/generate/_cmd.py currently raises NotImplementedError on torch
   - src/chuk_lazarus/cli/commands/context/calibrate_frames.py currently raises NotImplementedError on torch
   - src/chuk_lazarus/cli/commands/context/prefill/_cmd.py is now import-clean, but torch execution is still a gated early-return path, not real functionality
   - Some current tests lock in blocker behavior; update tests when implementing real torch support

2. Finish high-level torch training with LoRA.
   - These still explicitly block CHUK_BACKEND=torch with use_lora=True:
     - src/chuk_lazarus/training/trainers/sft_trainer.py
     - src/chuk_lazarus/training/trainers/dpo_trainer.py
     - src/chuk_lazarus/training/trainers/grpo_trainer.py
   - Root cause is the models_v2 loader + adapters/lora stack is still MLX-centric for high-level training use
   - peft is not available, so do not assume PEFT; either implement Lazarus-native torch LoRA or a minimal internal torch LoRA path
   - Need real model load, adapter apply, adapter save/load, and checkpoint resume on torch/CUDA

3. Decide and implement the torch path for DualRewardTrainer, or explicitly scope it out.
   - Current blocker is intentional and tests currently assert that blocker
   - File:
     - src/chuk_lazarus/training/trainers/dual_reward_trainer.py
   - Test to update if implemented:
     - tests/training/trainers/test_dual_reward_backend.py

4. Validate and finish real CUDA workflows on the RTX 5090.
   - infer
   - serve
   - knowledge build/query/chat
   - context prefill
   - context generate
   - introspect
   - sft
   - dpo
   - grpo
   - ppo
   - dual_reward if enabled
   - These must be real cuda:0 smokes, not just import or AST gates

5. Clean remaining old-package leakage outside core src package.
   - No remaining direct chuk_virtual_expert imports exist under src/chuk_lazarus/** outside the proper virtual_experts surfaces
   - Remaining old-path references are in experiments, tests, and docs:
     - experiments/csp_cot_gsm8k/train_gsm8k_yaml.py
     - experiments/csp_cot_gsm8k/scripts/diagnose_gsm8k.py
     - experiments/cot_standardization_extended/trace/verifier.py
     - experiments/csp_cot_gsm8k/scripts/template_expander.py
     - tests/introspection/test_virtual_expert.py
     - tests/conftest.py
     - docs/virtual_experts.md
   - Either reconcile these or make the compatibility decision explicit

Known next real blockers
- `context generate` and `calibrate_frames` are still intentionally torch-disabled
- `context prefill` is import-safe now, but not actually implemented for torch
- high-level SFT/DPO/GRPO LoRA-on-torch is still blocked
- DualRewardTrainer torch support is still blocked
- Real 5090 end-to-end validation is still thin

Suggested execution order
1. Re-run the known-green baselines above to confirm your starting state
2. Implement real torch execution for context generate / calibrate-frames / prefill, and update tests that currently assert blocker behavior
3. Implement the high-level torch LoRA path for SFT/DPO/GRPO
4. Decide and implement or defer DualRewardTrainer torch support
5. Run targeted CUDA smokes on the actual 5090
6. Run broader end-to-end validation for infer/serve/knowledge/context/introspect/training
7. Reconcile beads, close or update follow-ups, bd sync, git push

Minimum validation sequence
- CHUK_BACKEND=torch PYTHONPATH=src python -c "import chuk_lazarus"
- CHUK_BACKEND=torch PYTHONPATH=src python -c "import chuk_lazarus.inference.loader"
- CHUK_BACKEND=torch PYTHONPATH=src python -c "import chuk_lazarus.introspection"
- CHUK_BACKEND=torch PYTHONPATH=src python -c "import chuk_lazarus.cli"
- CHUK_BACKEND=torch PYTHONPATH=src python -c "import chuk_lazarus; chuk_lazarus.LlamaForCausalLM"
- uv run python -m pytest tests/ci/test_no_top_level_mlx.py -q
- uv run python -m pytest tests/server tests/cli/commands/knowledge/test__build_backend.py tests/cli/commands/knowledge/test__query_backend.py tests/cli/commands/knowledge/test__chat_backend.py tests/introspection/test_analyzer_core_backend.py -q
- uv run python -m pytest tests/cli/commands/context/prefill/test_cmd_backend.py tests/cli/commands/context/generate/test_cmd_backend.py tests/cli/commands/context/test_calibrate_frames_backend.py -q
- uv run python -m pytest tests/cli/commands/train/test_import_chain_backend.py tests/training/trainers/test_sft_backend.py tests/training/trainers/test_dpo_backend.py tests/training/trainers/test_grpo_backend.py tests/training/trainers/test_dual_reward_backend.py -q
- CHUK_CUDA_SMOKE=1 uv run python -m pytest tests/models_v2/core/backend/test_cuda_smoke.py tests/inference/test_unified_cuda.py tests/inference/backends/test_torch_runtime.py tests/inference/backends/test_storage.py -q

Success criteria
- No top-level MLX gate stays green
- Core CLI/inference/introspection/server imports stay green under torch
- Context commands have real torch/CUDA execution, not placeholder NotImplementedError or graceful blocker returns
- SFT/DPO/GRPO high-level run paths work on torch with LoRA
- Dual-reward is either working on torch or explicitly deferred and tracked
- Real CUDA smokes pass on the RTX 5090
- Remaining beads/issues are reconciled
- Session is not complete until bd sync + git push succeed
