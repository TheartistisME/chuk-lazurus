#!/usr/bin/env bash
# Run the strict-mode clause-aligned + Gemma + CUDA demo using the Linux-native HF cache
# (which already has gemma-4-E2B-it fully downloaded — instant load, no re-fetch).
#
# This wrapper now defaults to the clause-aligned variant store (the generic demo at
# examples/inference/demo_clause_aligned_strict.py already defaults --store to the
# aus3000 clause-aligned variant), and forwards ALL args verbatim to the demo so
# you can freely pass --store, --question, --model, --device, --max-new-tokens, etc.
#
# For backwards compatibility, a single bare positional arg (no leading --) is still
# treated as the --question value, matching the old usage.
#
# Usage:
#   bash scripts/run_aus3000_demo_fast.sh                                      # default question, aus3000 clause-aligned store
#   bash scripts/run_aus3000_demo_fast.sh "What does clause 1.4.72 define?"    # legacy: lone positional -> --question
#   bash scripts/run_aus3000_demo_fast.sh --question "..." --max-new-tokens 256
#   bash scripts/run_aus3000_demo_fast.sh --store /path/to/clause_aligned/torch_store --question "..."

set -euo pipefail

REPO_ROOT="/mnt/c/users/jehma/desktop/lazarus/chuk-lazurus"
LINUX_HF_CACHE="/home/jehmal/.cache/huggingface"

# Override any inherited (slow) HF cache pointer
export HF_HOME="$LINUX_HF_CACHE"
export TRANSFORMERS_CACHE="$LINUX_HF_CACHE/hub"
export HF_HUB_CACHE="$LINUX_HF_CACHE/hub"
# Neutralise XDG_CACHE_HOME in case a wrapper set it to the Windows bundle
unset XDG_CACHE_HOME || true

# Pre-flight: verify the model is actually present in the Linux cache
MODEL_DIR="$LINUX_HF_CACHE/hub/models--google--gemma-4-E2B-it"
if [ ! -d "$MODEL_DIR" ]; then
  echo "[fast-run] ERROR: expected model not in Linux cache: $MODEL_DIR"
  echo "[fast-run] Either HF_HOME is wrong or you actually need to download."
  exit 1
fi

# Verify there's at least one non-.incomplete weight blob
WEIGHT=$(ls "$MODEL_DIR/blobs/"* 2>/dev/null | grep -v "\.incomplete$" | head -1 || true)
if [ -z "$WEIGHT" ]; then
  echo "[fast-run] ERROR: no complete weight blob in Linux cache — only .incomplete files"
  exit 1
fi
echo "[fast-run] using Linux HF cache at $LINUX_HF_CACHE"
echo "[fast-run] found blob: $(basename "$WEIGHT")  ($(du -sh "$WEIGHT" | cut -f1))"

# Forward ALL args verbatim to the generic demo. The generic demo already
# defaults --store to the aus3000 clause-aligned variant, so running this
# wrapper with no args still works for the end-to-end aus3000 test.
# If the caller passes a single positional arg (no flags), treat it as --question
# for backwards compatibility with the old usage:
#   bash scripts/run_aus3000_demo_fast.sh "What does clause 1.4.72 define?"

if [ $# -eq 1 ] && [[ "$1" != --* ]]; then
  set -- --question "$1"
fi

cd "$REPO_ROOT"
echo "[fast-run] starting demo with HF_HOME=$HF_HOME"
echo

exec uv run python examples/inference/demo_clause_aligned_strict.py "$@"
