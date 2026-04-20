#!/usr/bin/env bash
# Run the strict-mode clause-aligned + Gemma + CUDA demo using the Linux-native HF cache.
# Generic wrapper — forwards ALL args verbatim to the demo, so you can pass
#   --store, --question, --model, --device, --max-new-tokens, --system-prompt freely.
#
# Usage:
#   bash scripts/run_clause_aligned_demo.sh                                # all defaults
#   bash scripts/run_clause_aligned_demo.sh --question "What does clause 1.4.72 define?"
#   bash scripts/run_clause_aligned_demo.sh --store /path/to/other/clause_aligned/torch_store \
#                                           --question "Summarise clause 3.2.1"

set -euo pipefail

REPO_ROOT="/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus"
LINUX_HF_CACHE="/home/jehmal/.cache/huggingface"

export HF_HOME="$LINUX_HF_CACHE"
export TRANSFORMERS_CACHE="$LINUX_HF_CACHE/hub"
export HF_HUB_CACHE="$LINUX_HF_CACHE/hub"
unset XDG_CACHE_HOME || true

MODEL_DIR="$LINUX_HF_CACHE/hub/models--google--gemma-4-E2B-it"
if [ ! -d "$MODEL_DIR" ]; then
  echo "[clause-aligned] ERROR: expected model not in Linux cache: $MODEL_DIR"
  exit 1
fi

WEIGHT=$(ls "$MODEL_DIR/blobs/"* 2>/dev/null | grep -v "\.incomplete$" | head -1 || true)
if [ -z "$WEIGHT" ]; then
  echo "[clause-aligned] ERROR: no complete weight blob in Linux cache"
  exit 1
fi
echo "[clause-aligned] using Linux HF cache at $LINUX_HF_CACHE"
echo "[clause-aligned] found blob: $(basename "$WEIGHT")  ($(du -sh "$WEIGHT" | cut -f1))"

cd "$REPO_ROOT"
echo "[clause-aligned] starting demo with HF_HOME=$HF_HOME"
echo

exec uv run python examples/inference/demo_clause_aligned_strict.py "$@"
