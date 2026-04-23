#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
log_path="${repo_root}/prod/validation/kvd_role_smoke.log"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/kvd_role_smoke.XXXXXX")"
default_snapshot="/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
seed_phrase="${LAZARUS_KV_ROLE_SMOKE_SEED_PHRASE:-the saffron sidecar remembers hexagon gullies at sunrise.}"
max_new_tokens="${LAZARUS_KV_ROLE_SMOKE_MAX_NEW_TOKENS:-32}"
session="kvd_role_smoke_$$"
pane_log="${tmp_root}/role_smoke.pane.log"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}" || true
    rm -rf "${tmp_root}"
}

trap cleanup EXIT

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

wait_for_capture() {
    local target_session="$1"
    local pattern="$2"
    local timeout_seconds="$3"
    local started_at
    started_at="$(date +%s)"

    while (( "$(date +%s)" - started_at < timeout_seconds )); do
        tmux has-session -t "${target_session}" 2>/dev/null || return 1
        if tmux capture-pane -pt "${target_session}" | rg -q "${pattern}"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_for_prompt() {
    local target_session="$1"
    local timeout_seconds="$2"
    local started_at
    started_at="$(date +%s)"

    while (( "$(date +%s)" - started_at < timeout_seconds )); do
        tmux has-session -t "${target_session}" 2>/dev/null || return 1
        if tmux capture-pane -pt "${target_session}" \
            | awk 'NF { last = $0 } END { print last }' \
            | rg -q '^you> ?$'; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_for_file_pattern() {
    local file_path="$1"
    local pattern="$2"
    local timeout_seconds="$3"
    local started_at
    started_at="$(date +%s)"

    while (( "$(date +%s)" - started_at < timeout_seconds )); do
        if [[ -f "${file_path}" ]] && rg -q "${pattern}" "${file_path}"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

require_cmd tmux
require_cmd python
require_cmd rg

python - <<'PY' >/dev/null
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for kv_direct_role_smoke.sh")
PY

if [[ -n "${LAZARUS_MODEL:-}" ]]; then
    model_path="${LAZARUS_MODEL}"
elif [[ -d "${default_snapshot}" ]]; then
    model_path="${default_snapshot}"
else
    die "set LAZARUS_MODEL or install the local Gemma-4-E2B-it snapshot at ${default_snapshot}"
fi

mkdir -p "$(dirname -- "${log_path}")"
store_root="$(mktemp -d "${tmp_root}/store.XXXXXX")"

{
    printf 'kvd_role_smoke started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "${repo_root}"
    printf 'model_path=%s\n' "${model_path}"
    printf 'seed_phrase=%s\n' "${seed_phrase}"
    printf 'max_new_tokens=%s\n' "${max_new_tokens}"
    printf '\n'
} > "${log_path}"

printf -v launch_cmd '%q ' \
    env \
    PYTHONUNBUFFERED=1 \
    LAZARUS_KV_CANDIDATE_POOL=8 \
    LAZARUS_KV_K_HOT=1 \
    LAZARUS_KV_K_WARM=4 \
    python "${repo_root}/scripts/interactive_memory_chat.py" \
    --device cuda \
    --memory-mode topical \
    --max-new-tokens "${max_new_tokens}" \
    --store-root "${store_root}" \
    --model-path "${model_path}"

tmux new-session -d -s "${session}" "cd ${repo_root@Q} && ${launch_cmd}"
tmux set-option -t "${session}" remain-on-exit on >/dev/null
tmux pipe-pane -o -t "${session}" "cat > ${pane_log@Q}"

wait_for_capture "${session}" 'interactive memory chat ready' 180 \
    || die 'repl did not reach ready state'

tmux send-keys -t "${session}" \
    "Reply with exactly this sentence and nothing else, with no quotation marks: ${seed_phrase}" \
    C-m
wait_for_capture "${session}" 'gemma>' 180 \
    || die 'seed turn did not emit assistant output'
wait_for_prompt "${session}" 180 \
    || die 'seed turn did not complete'

tmux send-keys -t "${session}" "/save" C-m
wait_for_capture "${session}" 'retriever ready:' 180 \
    || die '/save did not refresh retriever'
wait_for_prompt "${session}" 180 \
    || die '/save did not complete'

tmux send-keys -t "${session}" "/new" C-m
wait_for_capture "${session}" 'fresh session started' 180 \
    || die '/new did not start a fresh session'
wait_for_prompt "${session}" 180 \
    || die '/new did not complete'

tmux send-keys -t "${session}" "/kv_query ${seed_phrase}" C-m
wait_for_capture "${session}" 'KV-DIRECT TIER SELECTION' 180 \
    || die '/kv_query did not print the inspection block'
wait_for_capture "${session}" 'kv_direct>' 180 \
    || die '/kv_query did not emit kv_direct output'
wait_for_prompt "${session}" 180 \
    || die '/kv_query did not return to prompt'

wait_for_file_pattern "${pane_log}" 'role=user +turn= *0' 180 \
    || die 'inspection block did not include user role + turn index'
wait_for_file_pattern "${pane_log}" 'role=assistant +turn= *1' 180 \
    || die 'inspection block did not include assistant role + turn index'

cat "${pane_log}" >> "${log_path}"

tmux send-keys -t "${session}" "/quit" C-m || true
sleep 1
tmux kill-session -t "${session}" || true

printf 'kvd_role_smoke PASS\n'
printf 'log=%s\n' "${log_path}"
