#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
log_path="${repo_root}/prod/validation/kvd_hot_boost_smoke.log"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/kvd_hot_boost_smoke.XXXXXX")"
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

dump_boost_log() {
    local boost="$1"
    local pane_log="$2"
    {
        printf '===== HOT BOOST %s =====\n' "${boost}"
        if [[ -f "${pane_log}" ]]; then
            cat "${pane_log}"
        else
            printf '[missing pane log]\n'
        fi
        printf '\n'
    } >> "${log_path}"
}

abort_boost() {
    local boost="$1"
    local pane_log="$2"
    local reason="$3"
    {
        printf 'ASSERT hot_boost=%s status=FAIL reason=%s\n' "${boost}" "${reason}"
    } >> "${log_path}"
    dump_boost_log "${boost}" "${pane_log}"
    die "hot_boost=${boost} failed: ${reason}"
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
        "hot_boost_applied": result.metadata.get("hot_boost_applied"),
        "hot_boost_value": result.metadata.get("hot_boost_value"),
    }
    print(
        "KVD_HOT_BOOST_METADATA=" + json.dumps(payload, sort_keys=True),
        flush=True,
    )
    return result


TorchInferenceRuntime.generate_with_kv_direct_materialization = wrapped
raise SystemExit(module.main())
PY
}

parse_and_assert_metadata() {
    local pane_log="$1"
    local expected_boost="$2"

    python - "${pane_log}" "${expected_boost}" <<'PY'
import json
import math
import sys

pane_log = sys.argv[1]
expected_boost = float(sys.argv[2])
expected_applied = expected_boost > 0.0
metadata_lines = []
with open(pane_log, encoding="utf-8", errors="replace") as handle:
    for raw_line in handle:
        line = raw_line.rstrip("\n")
        if "KVD_HOT_BOOST_METADATA=" in line:
            metadata_lines.append(line.split("KVD_HOT_BOOST_METADATA=", 1)[1])

if not metadata_lines:
    raise SystemExit("missing KVD_HOT_BOOST_METADATA marker")

payload = json.loads(metadata_lines[-1])
if payload.get("kv_direct_active") is not True:
    raise SystemExit(
        f"kv_direct_active must be True, got {payload.get('kv_direct_active')!r}"
    )
if payload.get("hot_boost_applied") is not expected_applied:
    raise SystemExit(
        "hot_boost_applied mismatch: "
        f"expected {expected_applied!r}, got {payload.get('hot_boost_applied')!r}"
    )
actual_boost = float(payload.get("hot_boost_value", 0.0))
if not math.isclose(actual_boost, expected_boost, rel_tol=0.0, abs_tol=1e-6):
    raise SystemExit(
        f"hot_boost_value mismatch: expected {expected_boost}, got {actual_boost!r}"
    )
print(json.dumps(payload, sort_keys=True))
PY
}

run_boost() {
    local boost="$1"
    local session="kvd_hot_boost_${boost//./_}_$$"
    local pane_log="${tmp_root}/boost_${boost}.pane.log"
    local store_root
    store_root="$(mktemp -d "${tmp_root}/store_${boost//./_}.XXXXXX")"
    local repo_root_q pane_log_q launch_cmd metadata_json

    active_sessions+=("${session}")
    : > "${pane_log}"

    printf -v repo_root_q '%q' "${repo_root}"
    printf -v pane_log_q '%q' "${pane_log}"
    printf -v launch_cmd '%q ' \
        env \
        PYTHONUNBUFFERED=1 \
        KVD_SMOKE_REPO_ROOT="${repo_root}" \
        LAZARUS_KV_HOT_BOOST="${boost}" \
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
        || abort_boost "${boost}" "${pane_log}" 'repl did not reach ready state'

    tmux send-keys -t "${session}" \
        "Reply with exactly this sentence and nothing else, with no quotation marks: ${seed_phrase}" \
        C-m
    wait_for_capture "${session}" 'gemma>' 180 \
        || abort_boost "${boost}" "${pane_log}" 'seed turn did not emit assistant output'
    wait_for_prompt "${session}" 180 \
        || abort_boost "${boost}" "${pane_log}" 'seed turn did not complete'

    tmux send-keys -t "${session}" "/save" C-m
    wait_for_capture "${session}" 'retriever ready:' 180 \
        || abort_boost "${boost}" "${pane_log}" '/save did not refresh retriever'
    wait_for_prompt "${session}" 180 \
        || abort_boost "${boost}" "${pane_log}" '/save did not complete'

    tmux send-keys -t "${session}" "/new" C-m
    wait_for_capture "${session}" 'fresh session started' 180 \
        || abort_boost "${boost}" "${pane_log}" '/new did not start a fresh session'
    wait_for_prompt "${session}" 180 \
        || abort_boost "${boost}" "${pane_log}" '/new did not complete'

    tmux send-keys -t "${session}" "/kv_query ${seed_phrase}" C-m
    wait_for_file_pattern "${pane_log}" 'KVD_HOT_BOOST_METADATA=' 180 \
        || abort_boost "${boost}" "${pane_log}" '/kv_query did not emit metadata marker'
    wait_for_capture "${session}" 'kv_direct>' 180 \
        || abort_boost "${boost}" "${pane_log}" '/kv_query did not emit kv_direct output'
    wait_for_prompt "${session}" 180 \
        || abort_boost "${boost}" "${pane_log}" '/kv_query did not return to prompt'

    if ! metadata_json="$(parse_and_assert_metadata "${pane_log}" "${boost}")"; then
        abort_boost "${boost}" "${pane_log}" 'metadata assertion failed'
    fi

    summary_lines+=("hot_boost=${boost} metadata=${metadata_json}")
    {
        printf 'ASSERT hot_boost=%s status=PASS metadata=%s\n' \
            "${boost}" "${metadata_json}"
    } >> "${log_path}"
    dump_boost_log "${boost}" "${pane_log}"

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
    raise SystemExit("CUDA is required for kv_direct_hot_boost_smoke.sh")
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
    printf 'kvd_hot_boost_smoke started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repo_root=%s\n' "${repo_root}"
    printf 'model_path=%s\n' "${model_path}"
    printf 'seed_phrase=%s\n' "${seed_phrase}"
    printf 'max_new_tokens=%s\n' "${max_new_tokens}"
    printf '\n'
} > "${log_path}"

write_wrapper
run_boost "0"
run_boost "2"
run_boost "4"

{
    printf 'SUMMARY\n'
    printf '%s\n' "${summary_lines[@]}"
} >> "${log_path}"

printf 'kvd_hot_boost_smoke PASS\n'
printf 'log=%s\n' "${log_path}"
