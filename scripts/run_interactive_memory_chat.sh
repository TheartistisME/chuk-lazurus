#!/usr/bin/env bash
# run_interactive_memory_chat.sh -- ergonomic launcher for manual REPL testing.
#
# Resolves the repo root from this script's location, picks a fresh default
# store dir under artifacts/manual/, and forwards extra flags to the Python
# REPL entry point.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

has_store_root=0
has_model_path=0
for arg in "$@"; do
    case "${arg}" in
        --store-root | --store-root=*)
            has_store_root=1
            ;;
        --model-path | --model-path=*)
            has_model_path=1
            ;;
    esac
done

if [[ ${has_store_root} -eq 0 && -z "${LAZARUS_STORE_DIR:-}" ]]; then
    utcstamp="$(date -u +%Y%m%dT%H%M%SZ)"
    default_store="${repo_root}/artifacts/manual/repl_loop_${utcstamp}"
    if [[ -e "${default_store}" ]]; then
        default_store="${default_store}_$$"
    fi
    mkdir -p "${default_store}"
    export LAZARUS_STORE_DIR="${default_store}"
fi

launch_args=(
    --device cuda
    --memory-mode topical
    --max-new-tokens 180
)

if [[ ${has_model_path} -eq 0 && -z "${LAZARUS_MODEL:-}" ]]; then
    launch_args+=(--model-path google/gemma-4-E2B-it)
fi

exec python scripts/interactive_memory_chat.py "${launch_args[@]}" "$@"
