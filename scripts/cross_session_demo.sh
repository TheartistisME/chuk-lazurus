#!/usr/bin/env bash
# cross_session_demo.sh -- terminal axis-5 runner.
#
# Creates <output_root>/inputs and <output_root>/checkpoints and invokes
# the Python CLI. The Python CLI is the source of truth for the pipeline;
# this wrapper only exists for ergonomic terminal use.
#
# Usage:
#   scripts/cross_session_demo.sh <output_root> [extra CLI flags...]
#
# The VerificationReport JSON is written to <output_root>/report.json.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output_root> [extra flags...]" >&2
    exit 64
fi

OUTPUT_ROOT="$1"
shift

INPUTS_ROOT="${OUTPUT_ROOT}/inputs"
CHECKPOINTS_ROOT="${OUTPUT_ROOT}/checkpoints"
REPORT_OUT="${OUTPUT_ROOT}/report.json"

mkdir -p "${INPUTS_ROOT}" "${CHECKPOINTS_ROOT}"

exec python -m chuk_lazarus.cross_session_demo.cli \
    --inputs-root "${INPUTS_ROOT}" \
    --checkpoints-root "${CHECKPOINTS_ROOT}" \
    --report-out "${REPORT_OUT}" \
    "$@"
