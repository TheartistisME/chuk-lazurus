#!/usr/bin/env bash
# build_session_store.sh -- convenience wrapper for axis-3 session store.
#
# This script is a thin shim that delegates to the authoritative Python CLI:
#   python -m chuk_lazarus.session_store.cli
#
# The Python CLI is the source of truth for flags and behavior. This shell
# wrapper exists only to provide ergonomic positional invocation from a
# terminal or makefile.
#
# Usage:
#   scripts/build_session_store.sh <session_id> <input_dir> <checkpoint_root> [extra CLI flags...]
#
# Examples:
#   scripts/build_session_store.sh sess-001 out/sessions/sess-001 checkpoints/sessions
#   scripts/build_session_store.sh sess-001 out/sess-001 ckpts --force --verify
#   scripts/build_session_store.sh sess-001 out/sess-001 ckpts --dry-run
#
# Exits with the CLI's returncode (which is the build subprocess's returncode
# on success, or 2 on post-build verification failure).

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <session_id> <input_dir> <checkpoint_root> [extra flags...]" >&2
    exit 64
fi

SESSION_ID="$1"
INPUT_DIR="$2"
CHECKPOINT_ROOT="$3"
shift 3

exec python -m chuk_lazarus.session_store.cli \
    --session-id "${SESSION_ID}" \
    --input-dir "${INPUT_DIR}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    "$@"
