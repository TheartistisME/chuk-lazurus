#!/usr/bin/env bash
# Idempotent fixture dispatcher for Validation matrix Appendix A.
#
# Usage:
#   bash tests/fixtures/generate.sh [--model MODEL] [--out DIR] [--seed N]
#
# Runs every builder under tests/fixtures/_builders/, writes artifacts to
# DIR (defaults to tests/fixtures/), and regenerates SHA256SUMS.  Safe to
# re-run; builders overwrite their outputs deterministically.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR=""
SEED=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --out)
            OUT_DIR="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}}"
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

BUILDERS=(
    build_vec
    build_corpus
    build_probe
    build_ds
    build_vector
    build_exp
    build_cls
    build_sft
    build_pairs
    build_grpo
    build_raw
)

for b in "${BUILDERS[@]}"; do
    echo "[generate.sh] running ${b}"
    python - "${OUT_DIR}" "${MODEL}" "${SEED}" <<PYEOF
import sys
from pathlib import Path
sys.path.insert(0, "${SCRIPT_DIR}/..")
from fixtures._builders import ${b} as builder

out_dir = Path(sys.argv[1])
model = sys.argv[2] or None
seed = int(sys.argv[3])
builder.build(out_dir, model=model, seed=seed)
PYEOF
done

# Regenerate the checksum manifest (stable sort order for diffing).
(
    cd "${OUT_DIR}"
    : > SHA256SUMS
    # shellcheck disable=SC2012
    for f in $(ls -1 | grep -v '^SHA256SUMS$' | grep -v '^_builders$' | grep -v '^\.gitkeep$' | grep -v '^generate\.sh$' | sort); do
        if [[ -f "$f" ]]; then
            sha256sum "$f" >> SHA256SUMS
        fi
    done
)

echo "[generate.sh] done — artifacts in ${OUT_DIR}"
