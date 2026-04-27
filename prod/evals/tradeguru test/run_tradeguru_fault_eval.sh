#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

STORE_ROOT="${TRADEGURU_STORE_ROOT:-${SCRIPT_DIR}/memory_store/tradeguru_fault_memory}"
NOISE_STORE_ROOT="${TRADEGURU_NOISE_STORE_ROOT:-}"
RESULTS_ROOT="${TRADEGURU_RESULTS_ROOT:-${SCRIPT_DIR}/results}"
LOG_ROOT="${TRADEGURU_LOG_ROOT:-${SCRIPT_DIR}/logs}"

MODEL="${TRADEGURU_MODEL:-google/gemma-4-E2B-it}"
DEVICE="${TRADEGURU_DEVICE:-auto}"
CONDITIONS="${TRADEGURU_CONDITIONS:-memory_off,memory_on,memory_on_noise,off_store}"
IMPORT_MODE="${TRADEGURU_IMPORT_MODE:-append}"

SOURCE_ELECTRICALHOWTO="${TRADEGURU_SOURCE_ELECTRICALHOWTO:-/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/electricalhowto}"
SOURCE_ELECTRICALHOWTO_2="${TRADEGURU_SOURCE_ELECTRICALHOWTO_2:-/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/electricalhowto - 2}"
SOURCE_FAULT_GUIDE="${TRADEGURU_SOURCE_FAULT_GUIDE:-/mnt/c/Users/jehma/Desktop/TradeGuru/vector-store-knowledge/fault-finding-guide}"
EXTRA_SOURCE="${TRADEGURU_EXTRA_SOURCE:-}"

MAX_NEW_TOKENS="${TRADEGURU_MAX_NEW_TOKENS:-240}"
HOT_BUDGET_MIB="${TRADEGURU_HOT_BUDGET_MIB:-512}"
CANDIDATE_POOL="${TRADEGURU_CANDIDATE_POOL:-32}"
K_HOT="${TRADEGURU_K_HOT:-4}"
K_WARM="${TRADEGURU_K_WARM:-8}"

usage() {
  cat <<'USAGE'
Usage:
  bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" dry-run
  bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" import
  bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" eval
  bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" all
  bash "prod/evals/tradeguru test/run_tradeguru_fault_eval.sh" inspect

Environment overrides:
  TRADEGURU_STORE_ROOT         Permanent memory store root.
  TRADEGURU_NOISE_STORE_ROOT   Optional true TradeGuru-plus-noise store root.
  TRADEGURU_MODEL              HF model id or local model path.
  TRADEGURU_DEVICE             auto, cuda, or cpu.
  TRADEGURU_IMPORT_MODE        append (default) or force.
  TRADEGURU_EXTRA_SOURCE       Optional fourth source directory for noise builds.
USAGE
}

run_id() {
  date -u +"%Y%m%dT%H%M%SZ"
}

source_args() {
  printf '%s\0' \
    --source "${SOURCE_ELECTRICALHOWTO}" \
    --source "${SOURCE_ELECTRICALHOWTO_2}" \
    --source "${SOURCE_FAULT_GUIDE}"
  if [[ -n "${EXTRA_SOURCE}" ]]; then
    printf '%s\0' --source "${EXTRA_SOURCE}"
  fi
}

check_sources() {
  local missing=0
  for source in "${SOURCE_ELECTRICALHOWTO}" "${SOURCE_ELECTRICALHOWTO_2}" "${SOURCE_FAULT_GUIDE}"; do
    if [[ ! -e "${source}" ]]; then
      echo "[tradeguru] MISSING source: ${source}" >&2
      missing=1
    fi
  done
  if [[ -n "${EXTRA_SOURCE}" && ! -e "${EXTRA_SOURCE}" ]]; then
    echo "[tradeguru] MISSING extra source: ${EXTRA_SOURCE}" >&2
    missing=1
  fi
  return "${missing}"
}

import_store() {
  check_sources
  mkdir -p "${STORE_ROOT}" "${LOG_ROOT}"
  local stamp
  stamp="$(run_id)"
  local mode_arg=(--append)
  if [[ "${IMPORT_MODE}" == "force" ]]; then
    mode_arg=(--force)
  elif [[ "${IMPORT_MODE}" != "append" ]]; then
    echo "[tradeguru] TRADEGURU_IMPORT_MODE must be append or force" >&2
    return 2
  fi

  local args=()
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(source_args)

  (
    cd "${REPO_ROOT}"
    PYTHONUNBUFFERED=1 uv run --extra dev python scripts/import_tradeguru_memory.py \
      --store-root "${STORE_ROOT}" \
      "${args[@]}" \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      "${mode_arg[@]}"
  ) 2>&1 | tee "${LOG_ROOT}/import_${stamp}.log"
}

eval_store() {
  mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
  local stamp output
  stamp="$(run_id)"
  output="${RESULTS_ROOT}/eval_results_${stamp}.jsonl"

  local noise_args=()
  if [[ -n "${NOISE_STORE_ROOT}" ]]; then
    noise_args=(--noise-store-root "${NOISE_STORE_ROOT}")
  fi

  (
    cd "${REPO_ROOT}"
    PYTHONUNBUFFERED=1 uv run --extra dev python scripts/evaluate_tradeguru_fault_memory.py \
      --store-root "${STORE_ROOT}" \
      "${noise_args[@]}" \
      --output "${output}" \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --conditions "${CONDITIONS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --hot-budget-mib "${HOT_BUDGET_MIB}" \
      --candidate-pool "${CANDIDATE_POOL}" \
      --k-hot "${K_HOT}" \
      --k-warm "${K_WARM}"
  ) 2>&1 | tee "${LOG_ROOT}/eval_${stamp}.log"
}

dry_run() {
  check_sources
  mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
  local args=()
  while IFS= read -r -d '' item; do
    args+=("${item}")
  done < <(source_args)
  (
    cd "${REPO_ROOT}"
    uv run --extra dev python scripts/import_tradeguru_memory.py \
      --dry-run \
      --limit-docs 3 \
      --store-root "${STORE_ROOT}" \
      "${args[@]}"
    uv run --extra dev python scripts/evaluate_tradeguru_fault_memory.py \
      --dry-run \
      --store-root "${STORE_ROOT}" \
      --conditions "${CONDITIONS}"
  )
}

inspect_store() {
  echo "[tradeguru] eval root    : ${SCRIPT_DIR}"
  echo "[tradeguru] store root   : ${STORE_ROOT}"
  echo "[tradeguru] results root : ${RESULTS_ROOT}"
  echo "[tradeguru] logs root    : ${LOG_ROOT}"
  if [[ -d "${STORE_ROOT}" ]]; then
    echo "[tradeguru] store size   : $(du -sh "${STORE_ROOT}" | awk '{print $1}')"
    echo "[tradeguru] sessions     : $(find "${STORE_ROOT}" -path '*/torch_store/manifest.json' | wc -l)"
    find "${STORE_ROOT}" -maxdepth 2 -name tradeguru_import_manifest.json -print
  else
    echo "[tradeguru] store missing"
  fi
  if [[ -d "${RESULTS_ROOT}" ]]; then
    find "${RESULTS_ROOT}" -maxdepth 1 -type f -print | sort
  fi
}

cmd="${1:-help}"
case "${cmd}" in
  dry-run)
    dry_run
    ;;
  import)
    import_store
    ;;
  eval)
    eval_store
    ;;
  all)
    import_store
    eval_store
    ;;
  inspect)
    inspect_store
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
