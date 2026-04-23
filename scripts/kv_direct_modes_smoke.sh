#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
log_path="${repo_root}/prod/validation/kvd_modes_smoke.log"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/kvd_modes_smoke.XXXXXX")"
wrapper_path="${tmp_root}/interactive_memory_chat_wrapper.py"
default_snapshot="/home/jehmal/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
seed_phrase="${LAZARUS_KV_SMOKE_SEED_PHRASE:-the saffron sidecar remembers hexagon gullies at sunrise.}"
max_new_tokens="${LAZARUS_KV_SMOKE_MAX_NEW_TOKENS:-32}"

declare -a active_sessions=()
declare -a summary_lines=()

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    local session
    for session in "${active_sessions[@]:-}"; do
        tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}" || true
    done
    rm -rf "${tmp_root}"
}

trap cleanup EXIT

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

wait_for_capture() {
    local session="$1"
    local pattern="$2"
    local timeout_seconds="$3"
    local started_at
    started_at="$(date +%s)"

    while (( "$(date +%s)" - started_at < timeout_seconds )); do
        tmux has-session -t "${session}" 2>/dev/null || return 1
        if tmux capture-pane -pt "${session}" | rg -q "${pattern}"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_for_prompt() {
    local session="$1"
    local timeout_seconds="$2"
    local started_at
    started_at="$(date +%s)"

    while (( "$(date +%s)" - started_at < timeout_seconds )); do
        tmux has-session -t "${session}" 2>/dev/null || return 1
        if tmux capture-pane -pt "${session}" \
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

dump_mode_log() {
    local mode="$1"
    local pane_log="$2"
    {
        printf '===== MODE %s =====\n' "${mode}"
        if [[ -f "${pane_log}" ]]; then
            cat "${pane_log}"
        else
            printf '[missing pane log]\n'
        fi
        printf '\n'
    } >> "${log_path}"
}

abort_mode() {
    local mode="$1"
    local pane_log="$2"
    local reason="$3"
    {
        printf 'ASSERT mode=%s status=FAIL reason=%s\n' "${mode}" "${reason}"
    } >> "${log_path}"
    dump_mode_log "${mode}" "${pane_log}"
    die "mode ${mode} failed: ${reason}"
}

write_wrapper() {
    cat > "${wrapper_path}" <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["KVD_SMOKE_REPO_ROOT"])
os.chdir(repo_root)
module_path = repo_root / "scripts" / "interactive_memory_chat.py"

spec = importlib.util.spec_from_file_location(
    "interactive_memory_chat_wrapped",
    module_path,
)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime

original = TorchInferenceRuntime.generate_with_kv_direct_materialization


def wrapped(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    payload = {
        "kv_direct_active": result.metadata.get("kv_direct_active"),
        "n_archived_slots": result.metadata.get("n_archived_slots"),
        "prefix_forwards_observed": result.metadata.get(
            "prefix_forwards_observed"
        ),
        "propagation_mode": result.metadata.get("propagation_mode"),
        "selected_tier": result.metadata.get("selected_tier"),
    }
    print(
        "KVD_SMOKE_METADATA=" + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return result


TorchInferenceRuntime.generate_with_kv_direct_materialization = wrapped
raise SystemExit(module.main())
PY
}

parse_and_assert_metadata() {
    local pane_log="$1"
    local expected_mode="$2"
    local expected_prefix="$3"

    python - "${pane_log}" "${expected_mode}" "${expected_prefix}" <<'PY'
import json
import sys

pane_log, expected_mode, expected_prefix = sys.argv[1], sys.argv[2], int(sys.argv[3])
metadata_lines = []
with open(pane_log, encoding="utf-8", errors="replace") as handle:
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        if "KVD_SMOKE_METADATA=" in line:
            metadata_lines.append(line.split("KVD_SMOKE_METADATA=", 1)[1])

if not metadata_lines:
    raise SystemExit("missing KVD_SMOKE_METADATA marker")

payload = json.loads(metadata_lines[-1])
actual_mode = payload.get("propagation_mode")
actual_prefix = payload.get("prefix_forwards_observed")
if actual_mode != expected_mode:
    raise SystemExit(
        f"propagation_mode mismatch: expected {expected_mode!r}, got {actual_mode!r}"
    )
if actual_prefix != expected_prefix:
    raise SystemExit(
        f"prefix_forwards_observed mismatch: expected {expected_prefix}, got {actual_prefix!r}"
    )
if payload.get("kv_direct_active") is not True:
    raise SystemExit(
        f"kv_direct_active must be True, got {payload.get('kv_direct_active')!r}"
    )
print(json.dumps(payload, sort_keys=True))
PY
}

run_mode() {
    local mode="$1"
    local expected_prefix="$2"
    local session="kvd_smoke_${mode}_$$"
    local pane_log="${tmp_root}/${mode}.pane.log"
    local store_root
    store_root="$(mktemp -d "${tmp_root}/store_${mode}.XXXXXX")"
    local repo_root_q pane_log_q launch_cmd metadata_json

    active_sessions+=("${session}")
    : > "${pane_log}"

    printf -v repo_root_q '%q' "${repo_root}"
    printf -v pane_log_q '%q' "${pane_log}"
    printf -v launch_cmd '%q ' \
        env \
        PYTHONUNBUFFERED=1 \
        KVD_SMOKE_REPO_ROOT="${repo_root}" \
        LAZARUS_KV_PROPAGATION_MODE="${mode}" \
        python "${wrapper_path}" \
        --device cuda \
        --memory-mode topical \
        --max-new-tokens "${max_new_tokens}" \
        --store-root "${store_root}" \
        --model-path "${model_path}"

    tmux new-session -d -s "${session}" "cd ${repo_root_q} && ${launch_cmd}"
    tmux set-option -t "${session}" remain-on-exit on >/dev/null
    tmux pipe-pane -o -t "${session}" "cat > ${pane_log_q}"

    wait_for_capture "${session}" 'interactive memory chat ready' 180 \
        || abort_mode "${mode}" "${pane_log}" 'repl did not reach ready state'

    tmux send-keys -t "${session}" \
        "Reply with exactly this sentence and nothing else, with no quotation marks: ${seed_phrase}" \
        C-m
    wait_for_capture "${session}" 'gemma>' 180 \
        || abort_mode "${mode}" "${pane_log}" 'seed turn did not emit assistant output'
    wait_for_prompt "${session}" 180 \
        || abort_mode "${mode}" "${pane_log}" 'seed turn did not complete'

    tmux send-keys -t "${session}" "/save" C-m
    wait_for_capture "${session}" 'retriever ready:' 180 \
        || abort_mode "${mode}" "${pane_log}" '/save did not refresh retriever'
    wait_for_prompt "${session}" 180 \
        || abort_mode "${mode}" "${pane_log}" '/save did not complete'

    tmux send-keys -t "${session}" "/new" C-m
    wait_for_capture "${session}" 'fresh session started' 180 \
        || abort_mode "${mode}" "${pane_log}" '/new did not start a fresh session'
    wait_for_prompt "${session}" 180 \
        || abort_mode "${mode}" "${pane_log}" '/new did not complete'

    tmux send-keys -t "${session}" "/kv_query ${seed_phrase}" C-m
    wait_for_file_pattern "${pane_log}" 'KVD_SMOKE_METADATA=' 180 \
        || abort_mode "${mode}" "${pane_log}" '/kv_query did not emit metadata marker'
    wait_for_capture "${session}" 'kv_direct>' 180 \
        || abort_mode "${mode}" "${pane_log}" '/kv_query did not emit kv_direct output'
    wait_for_prompt "${session}" 180 \
        || abort_mode "${mode}" "${pane_log}" '/kv_query did not return to prompt'

    if ! metadata_json="$(parse_and_assert_metadata "${pane_log}" "${mode}" "${expected_prefix}")"; then
        abort_mode "${mode}" "${pane_log}" 'metadata assertion failed'
    fi

    summary_lines+=("mode=${mode} propagation_mode=${mode} prefix_forwards_observed=${expected_prefix}")
    {
        printf 'ASSERT mode=%s status=PASS expected_prefix=%s metadata=%s\n' \
            "${mode}" "${expected_prefix}" "${metadata_json}"
    } >> "${log_path}"
    dump_mode_log "${mode}" "${pane_log}"

    tmux send-keys -t "${session}" "/quit" C-m || true
    sleep 1
    tmux kill-session -t "${session}" || true
}

require_cmd tmux
require_cmd python
require_cmd rg

python - <<'PY' >/dev/null
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for kv_direct_modes_smoke.sh")
PY

if [[ -n "${LAZARUS_MODEL:-}" ]]; then
    model_path="${LAZARUS_MODEL}"
elif [[ -d "${default_snapshot}" ]]; then
    model_path="${default_snapshot}"
else
    die "set LAZARUS_MODEL or install the local Gemma-4-E2B-it snapshot at ${default_snapshot}"
fi

mkdir -p "$(dirname -- "${log_path}")"

{
    printf 'kvd_modes_smoke started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "${repo_root}"
    printf 'model_path=%s\n' "${model_path}"
    printf 'seed_phrase=%s\n' "${seed_phrase}"
    printf 'max_new_tokens=%s\n' "${max_new_tokens}"
    printf '\n'
} > "${log_path}"

write_wrapper
run_mode "a" "5"
run_mode "b" "5"
run_mode "ab" "5"
run_mode "none" "1"

{
    printf 'SUMMARY\n'
    printf '%s\n' "${summary_lines[@]}"
} >> "${log_path}"

printf 'kvd_modes_smoke PASS\n'
printf 'log=%s\n' "${log_path}"
