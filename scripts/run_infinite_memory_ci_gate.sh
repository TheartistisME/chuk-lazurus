#!/usr/bin/env bash
# Run the infinite-memory + bounded-KV REPL proof as a CI-friendly gate.
#
# Full mode is the default and uses the Python harness defaults:
# 100 sessions x 100 turns. Smoke mode is available for quick operator checks.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

mode="full"
fresh=1
output_root="${repo_root}/prod/validation/repl-autoverify"
extra_args=()
seeded_residual_doc="${repo_root}/prod/the-bible/apollo-gemma4-zero-injection-token-salad-fix-lesson.md"
verifier_lesson_doc="${repo_root}/prod/the-bible/infinite-memory-bounded-kv-repl-verification-fix-lesson.md"

usage() {
    cat <<'EOF'
Usage: bash scripts/run_infinite_memory_ci_gate.sh [options] [-- extra harness args]

Options:
  --full              Run the full proof workload (default).
  --smoke             Run a quick smoke workload.
  --fresh             Delete the selected harness store before running (default).
  --no-fresh          Reuse the selected harness store.
  --output-root PATH  Harness artifact root.
  --help              Show this help.

Any other option is forwarded to scripts/auto_verify_memory_repl.py.

Examples:
  bash scripts/run_infinite_memory_ci_gate.sh --smoke
  bash scripts/run_infinite_memory_ci_gate.sh --full --model-path /models/gemma
  bash scripts/run_infinite_memory_ci_gate.sh --full -- --vram-ceiling-mib 30000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)
            mode="full"
            shift
            ;;
        --smoke)
            mode="smoke"
            shift
            ;;
        --fresh)
            fresh=1
            shift
            ;;
        --no-fresh)
            fresh=0
            shift
            ;;
        --output-root)
            output_root="$2"
            shift 2
            ;;
        --output-root=*)
            output_root="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            extra_args+=("$@")
            break
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${repo_root}/prod/validation"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_path="${repo_root}/prod/validation/infinite-memory-ci-gate-${stamp}.log"

for required_path in \
    "${repo_root}/scripts/auto_verify_memory_repl.py" \
    "${seeded_residual_doc}" \
    "${verifier_lesson_doc}"
do
    if [[ ! -f "${required_path}" ]]; then
        echo "[ci-gate] ERROR missing required file: ${required_path}" | tee -a "${log_path}"
        exit 2
    fi
done

harness_args=(--output-root "${output_root}")
if [[ "${fresh}" -eq 1 ]]; then
    harness_args+=(--fresh)
fi
if [[ "${mode}" == "smoke" ]]; then
    harness_args+=(--smoke)
else
    harness_args+=(--sessions 100 --turns-per-session 100)
fi
harness_args+=("${extra_args[@]}")

if command -v uv >/dev/null 2>&1; then
    cmd=(uv run --extra dev python scripts/auto_verify_memory_repl.py)
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    cmd=("${HOME}/.local/bin/uv" run --extra dev python scripts/auto_verify_memory_repl.py)
elif command -v python3 >/dev/null 2>&1; then
    cmd=(python3 scripts/auto_verify_memory_repl.py)
else
    cmd=(python scripts/auto_verify_memory_repl.py)
fi

echo "[ci-gate] repo_root=${repo_root}"
echo "[ci-gate] mode=${mode}"
echo "[ci-gate] output_root=${output_root}"
echo "[ci-gate] log=${log_path}"
echo "[ci-gate] seeded_residual_lesson=${seeded_residual_doc}"
echo "[ci-gate] verifier_lesson=${verifier_lesson_doc}"
echo "[ci-gate] command=${cmd[*]} ${harness_args[*]}"

set +e
"${cmd[@]}" "${harness_args[@]}" 2>&1 | tee "${log_path}"
status=${PIPESTATUS[0]}
set -e

if [[ "${status}" -eq 0 ]]; then
    echo "[ci-gate] PASS"
else
    echo "[ci-gate] FAIL status=${status}"
fi
echo "[ci-gate] wrapper_log=${log_path}"
exit "${status}"
