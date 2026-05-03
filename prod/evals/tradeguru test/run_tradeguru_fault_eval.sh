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
SELECTOR_POLICY="${TRADEGURU_SELECTOR_POLICY:-utility-v2}"
DENSE_SCORING="${TRADEGURU_DENSE_SCORING:-deterministic}"
RRF_K="${TRADEGURU_RRF_K:-60}"
MMR_LAMBDA="${TRADEGURU_MMR_LAMBDA:-0.75}"
SELECTOR_BUDGET="${TRADEGURU_SELECTOR_BUDGET:-$((K_HOT + K_WARM))}"

# Router penalty for catalog/"Example Queries" TOC-index windows. These match
# keyword-rich queries via TF-IDF but inject low-value context into KV memory.
# Penalising them surfaces procedural windows ("Step-by-Step", "Key Lessons",
# "Safety Notes", "Pro Tips") instead. Override to 0 to disable. Default ON
# because the 20260428T092804Z eval showed top-rank TOC-index hits driving
# memory_on regression vs memory_off.
TRADEGURU_TOC_INDEX_PENALTY="${TRADEGURU_TOC_INDEX_PENALTY:-5.0}"
export LAZARUS_ROUTER_TOC_INDEX_PENALTY="${LAZARUS_ROUTER_TOC_INDEX_PENALTY:-${TRADEGURU_TOC_INDEX_PENALTY}}"
# TOC-index detection requires decoded window text during scoring; force decode
# so the penalty applies before tier assignment.
export LAZARUS_ASI_DECODE_FOR_SELECTOR="${LAZARUS_ASI_DECODE_FOR_SELECTOR:-1}"
# When KV-direct memory_on falls through the synthesise_* paths (no color/value
# extraction match), the retriever's _generate_with_semantic_token_prefix path
# defaults to a hardcoded "be concise" system prompt that was authored for
# value-extraction tests. For procedural / safety-critical electrical
# fault-finding answers this destroys ordered test sequences and skips safety
# preambles, regressing memory_on vs memory_off. Honour the caller-supplied
# system prompt (passed by scripts/evaluate_tradeguru_fault_memory.py via
# SessionRetriever.system_prompt = SYSTEM_PROMPT) so the safety-first task
# semantics survive into the prefix decode. Diagnosis: grade
# 20260428T115132Z showed memory_on correct_test_order=1/8 and
# safety_controls_present=2/8 while memory_off was 6/8 and 8/8 with the same
# question set. Override to 0 to restore the legacy "be concise" prefix prompt.
export LAZARUS_KV_SEMANTIC_PREFIX_USE_CALLER_SYSTEM_PROMPT="${LAZARUS_KV_SEMANTIC_PREFIX_USE_CALLER_SYSTEM_PROMPT:-1}"
# Iteration-2 honoured the caller's safety-first system prompt on the prefix
# decode path, but iteration-3 grade 20260428T124342Z showed memory_on still
# trailing memory_off (-0.25 lift) because the model was spending the entire
# max_new_tokens=240 budget on a verbose 4-bullet safety preamble (PPE / LOTO
# / isolate / prove dead) and never reaching the diagnostic content. All
# memory_on answers truncated mid-sentence in the "Phase 1" header,
# regressing both correct_test_order (5/8) and uses_document_specific_details
# (6/8) versus memory_off (6/8 and 7/8). Prepend a short RAG directive to the
# caller-supplied prompt that asks the model to ground in document
# terminology and keep any preamble brief (1-2 sentences). The directive is
# task-agnostic (no rubric vocabulary), preserves the caller's safety
# semantics verbatim, and is opt-in via env flag. Override to 0 to restore
# the iteration-2 behaviour (caller prompt verbatim with no directive).
export LAZARUS_KV_SEMANTIC_PREFIX_GROUND_IN_DOCUMENT="${LAZARUS_KV_SEMANTIC_PREFIX_GROUND_IN_DOCUMENT:-1}"

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
  TRADEGURU_SELECTOR_POLICY    utility-v2 (default), rank-v1, or hybrid-v2.
  TRADEGURU_DENSE_SCORING      deterministic (default), off, auto, or provided.
  TRADEGURU_SELECTOR_BUDGET    Active HOT/WARM selector budget, default K_HOT+K_WARM.
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
      --k-warm "${K_WARM}" \
      --selector-policy "${SELECTOR_POLICY}" \
      --dense-scoring "${DENSE_SCORING}" \
      --rrf-k "${RRF_K}" \
      --mmr-lambda "${MMR_LAMBDA}" \
      --selector-budget "${SELECTOR_BUDGET}"
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
      --conditions "${CONDITIONS}" \
      --selector-policy "${SELECTOR_POLICY}" \
      --dense-scoring "${DENSE_SCORING}" \
      --rrf-k "${RRF_K}" \
      --mmr-lambda "${MMR_LAMBDA}" \
      --selector-budget "${SELECTOR_BUDGET}"
  )
}

inspect_store() {
  echo "[tradeguru] eval root    : ${SCRIPT_DIR}"
  echo "[tradeguru] store root   : ${STORE_ROOT}"
  echo "[tradeguru] results root : ${RESULTS_ROOT}"
  echo "[tradeguru] logs root    : ${LOG_ROOT}"
  echo "[tradeguru] selector     : ${SELECTOR_POLICY} dense=${DENSE_SCORING} budget=${SELECTOR_BUDGET}"
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
