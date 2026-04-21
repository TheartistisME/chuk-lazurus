#!/usr/bin/env bash
# run_apollo_memory_demo.sh -- turn-aligned-apollo run-1 driver.
#
# Builds the Apollo-style session-memory store from the 5-session synthetic
# cross-session corpus via scripts/build_apollo_style_memory_store.py, then
# runs three strict-mode queries through
# examples/inference/demo_c_apollo11_torch.py. All stdout/stderr is captured
# into artifacts/apollo_memory/ for forensic review by the LEAD.
#
# Scope: turn-aligned-apollo / build-and-test-apollo-path / run-1.
#
# Usage:
#   scripts/run_apollo_memory_demo.sh [flags]
#
# Flags:
#   --model VAL           HF model id (default: google/gemma-4-E2B-it)
#   --device VAL          torch device (default: auto)
#   --store VAL           store output dir (default: /tmp/apollo-memory/store)
#   --max-tokens N        cap tokens passed to the builder (default: unset)
#   --max-new-tokens N    decoder max_new_tokens per query (default: 64)
#   --artifacts VAL       artifact directory (default: artifacts/apollo_memory)
#   --skip-build          skip build step, run queries only
#   -h | --help           show this help and exit

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (env overrides honored; flags take precedence)
# ---------------------------------------------------------------------------
MODEL="${MODEL:-google/gemma-4-E2B-it}"
DEVICE="${DEVICE:-cuda}"   # was 'auto' — apollo demo passes string straight to torch; pin to cuda to prevent silent CPU fallback
STORE_DIR="${STORE_DIR:-/tmp/apollo-memory/store}"
MAX_TOKENS="${MAX_TOKENS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ART_DIR="${ART_DIR:-artifacts/apollo_memory}"
SKIP_BUILD=0

# ---------------------------------------------------------------------------
# Force offline HF + use the locally-cached Gemma snapshot.
# Without these, every invocation does an unauthenticated Hub probe (~10s
# stall + warning). The model is already cached; no network needed.
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-/home/jehmal/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# PYTHONPATH ensures `import chuk_lazarus...` resolves from src/ without an editable install.
export PYTHONPATH="${PYTHONPATH:-src}"

usage() {
    sed -n '2,25p' "$0"
}

# ---------------------------------------------------------------------------
# Manual arg parse (no getopt)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2;;
        --device)
            DEVICE="$2"; shift 2;;
        --store)
            STORE_DIR="$2"; shift 2;;
        --max-tokens)
            MAX_TOKENS="$2"; shift 2;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"; shift 2;;
        --artifacts)
            ART_DIR="$2"; shift 2;;
        --skip-build)
            SKIP_BUILD=1; shift;;
        -h|--help)
            usage; exit 0;;
        *)
            echo "[apollo-demo] unknown flag: $1" >&2
            usage >&2
            exit 64;;
    esac
done

ts() { date -u +%FT%TZ; }

# ---------------------------------------------------------------------------
# 1. Pre-flight
# ---------------------------------------------------------------------------
mkdir -p "${ART_DIR}"

echo "[apollo-demo] $(ts) starting run goal=turn-aligned-apollo"
echo "[apollo-demo] resolved flags:"
echo "[apollo-demo]   MODEL           = ${MODEL}"
echo "[apollo-demo]   DEVICE          = ${DEVICE}"
echo "[apollo-demo]   STORE_DIR       = ${STORE_DIR}"
echo "[apollo-demo]   MAX_TOKENS      = ${MAX_TOKENS:-<unset>}"
echo "[apollo-demo]   MAX_NEW_TOKENS  = ${MAX_NEW_TOKENS}"
echo "[apollo-demo]   ART_DIR         = ${ART_DIR}"
echo "[apollo-demo]   SKIP_BUILD      = ${SKIP_BUILD}"

# ---------------------------------------------------------------------------
# 2. Build step
# ---------------------------------------------------------------------------
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    LOG="${ART_DIR}/build.log"
    DUMP="${ART_DIR}/corpus.txt"
    echo "[apollo-demo] $(ts) build: store=${STORE_DIR} log=${LOG}"

    BUILD_CMD=(
        python3 scripts/build_apollo_style_memory_store.py
        --model "${MODEL}"
        --device "${DEVICE}"
        --output "${STORE_DIR}"
        --corpus-dump "${DUMP}"
        --verbose
    )
    if [[ -n "${MAX_TOKENS}" ]]; then
        BUILD_CMD+=(--max-tokens "${MAX_TOKENS}")
    fi

    "${BUILD_CMD[@]}" 2>&1 | tee "${LOG}"
    BUILD_RC="${PIPESTATUS[0]}"
    if [[ "${BUILD_RC}" -ne 0 ]]; then
        echo "[apollo-demo] BUILD FAILED rc=${BUILD_RC}"
        exit "${BUILD_RC}"
    fi
    echo "[apollo-demo] $(ts) build ok"
else
    echo "[apollo-demo] $(ts) build: skipped (--skip-build)"
fi

# ---------------------------------------------------------------------------
# 3. Query step (strict-mode, three planted-phrase probes)
# ---------------------------------------------------------------------------
QUESTIONS=(
    "What planted phrase was mentioned about the coral reef at Fujairah?"
    "Recall the exact planted phrase about Alice's mauve refactor."
    "What did the keynote say about quokka benchmarks in Gosford?"
)

STATUS_TSV="${ART_DIR}/query_status.tsv"
: > "${STATUS_TSV}"

for i in "${!QUESTIONS[@]}"; do
    idx=$((i + 1))
    q="${QUESTIONS[$i]}"
    QLOG="${ART_DIR}/query_${idx}.log"
    echo "[apollo-demo] $(ts) query ${idx}: ${q}"

    python3 examples/inference/demo_c_apollo11_torch.py \
        --store "${STORE_DIR}" \
        --model "${MODEL}" \
        --device "${DEVICE}" \
        --question "${q}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        2>&1 | tee "${QLOG}" || true
    QRC="${PIPESTATUS[0]}"

    printf 'query\t%s\t%s\t%s\n' "${idx}" "${QRC}" "${q}" >> "${STATUS_TSV}"
    echo "[apollo-demo] query ${idx} rc=${QRC} log=${QLOG}"
done

# ---------------------------------------------------------------------------
# 4. Text-level verdict shortcut (extraction only; LEAD decides)
# ---------------------------------------------------------------------------
extract_answer() {
    local log="$1"
    # Prefer a line beginning with "Answer:" (case-insensitive); fall back
    # to a "[generated]" marker, then to a loose "answer:" match. Tolerate
    # format drift by taking the first non-empty hit.
    local hit
    hit="$(grep -m1 -iE '^[[:space:]]*answer:' "${log}" 2>/dev/null || true)"
    if [[ -z "${hit}" ]]; then
        hit="$(grep -m1 -F '[generated]' "${log}" 2>/dev/null || true)"
    fi
    if [[ -z "${hit}" ]]; then
        hit="$(grep -m1 -iE 'answer:' "${log}" 2>/dev/null || true)"
    fi
    # Collapse whitespace and truncate to 300 chars.
    printf '%s' "${hit}" | tr -s '[:space:]' ' ' | cut -c1-300
}

for i in "${!QUESTIONS[@]}"; do
    idx=$((i + 1))
    q="${QUESTIONS[$i]}"
    QLOG="${ART_DIR}/query_${idx}.log"
    ans="$(extract_answer "${QLOG}")"
    echo "=== Query ${idx}: ${q} ==="
    if [[ -n "${ans}" ]]; then
        echo "${ans}"
    else
        echo "<no answer line detected in ${QLOG}>"
    fi
done

# ---------------------------------------------------------------------------
# 5. Final summary
# ---------------------------------------------------------------------------
echo "[apollo-demo] $(ts) done"
echo "[apollo-demo] artifacts: ${ART_DIR}"
ls -la "${ART_DIR}"
