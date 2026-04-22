#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Run the local Qwen manual KV-direct test against chuk-lazurus.

Defaults:
  - model snapshot: local Infatoshi Qwen3.6-35B-A3B cache
  - prompt file: prompts/qwen_kv_direct_manual_test.txt
  - device: cuda:0
  - max new tokens: 220
  - temperature: 0.0
  - chat template: enabled
  - repeat count: 1

Usage:
  scripts/run_qwen_kv_direct_manual_test.sh

Overrides via env:
  MODEL_PATH=/path/to/model
  PROMPT_PATH=/path/to/prompt.txt
  DEVICE=cuda:0
  MAX_NEW_TOKENS=220
  TEMPERATURE=0.0
  SYSTEM_PROMPT="You are a precise coding assistant."
  USE_CHAT_TEMPLATE=1
  REPEAT_DOC_COUNT=1
  OUT_FILE=/path/to/output.log

Examples:
  scripts/run_qwen_kv_direct_manual_test.sh
  REPEAT_DOC_COUNT=200 scripts/run_qwen_kv_direct_manual_test.sh
  MODEL_PATH=/models/my-qwen DEVICE=cuda:1 scripts/run_qwen_kv_direct_manual_test.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_MODEL_ROOT="/home/jehmal/.cache/huggingface/hub/models--Infatoshi--Qwen3.6-35B-A3B-NVFP4-FP8"
RAW_MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_ROOT}"
PROMPT_PATH="${PROMPT_PATH:-$ROOT_DIR/prompts/qwen_kv_direct_manual_test.txt}"
DEVICE="${DEVICE:-cuda:0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a precise local coding assistant. Follow the user document exactly and do not skip details.}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
REPEAT_DOC_COUNT="${REPEAT_DOC_COUNT:-1}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/qwen_kv_direct}"

mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_FILE="${OUT_FILE:-$OUT_DIR/manual_test_${STAMP}.log}"
SUMMARY_FILE="${SUMMARY_FILE:-$OUT_DIR/manual_test_${STAMP}.summary.json}"

resolve_model_path() {
  local input_path="$1"
  if [[ -f "$input_path/config.json" ]]; then
    printf '%s\n' "$input_path"
    return 0
  fi
  if [[ -f "$input_path/refs/main" ]]; then
    local ref_snapshot
    ref_snapshot="$(<"$input_path/refs/main")"
    if [[ -n "$ref_snapshot" && -f "$input_path/snapshots/$ref_snapshot/config.json" ]]; then
      printf '%s\n' "$input_path/snapshots/$ref_snapshot"
      return 0
    fi
  fi
  if [[ -d "$input_path/snapshots" ]]; then
    local latest_snapshot
    latest_snapshot="$(find "$input_path/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
    if [[ -n "$latest_snapshot" && -f "$latest_snapshot/config.json" ]]; then
      printf '%s\n' "$latest_snapshot"
      return 0
    fi
  fi
  return 1
}

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required but not installed." >&2
  exit 1
fi

if [[ "$DEVICE" != cuda* ]]; then
  echo "ERROR: this manual test is GPU-only and requires DEVICE=cuda:N. Got: $DEVICE" >&2
  exit 1
fi

if ! MODEL_PATH="$(resolve_model_path "$RAW_MODEL_PATH")"; then
  echo "ERROR: could not resolve a model snapshot from MODEL_PATH=$RAW_MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$PROMPT_PATH" ]]; then
  echo "ERROR: prompt file not found: $PROMPT_PATH" >&2
  exit 1
fi

if [[ ! "$MAX_NEW_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_NEW_TOKENS must be an integer." >&2
  exit 1
fi

if [[ ! "$REPEAT_DOC_COUNT" =~ ^[0-9]+$ ]] || (( REPEAT_DOC_COUNT < 1 )); then
  echo "ERROR: REPEAT_DOC_COUNT must be an integer >= 1." >&2
  exit 1
fi

echo "Running Qwen KV-direct manual test..."
echo "  model   : $MODEL_PATH"
echo "  prompt  : $PROMPT_PATH"
echo "  device  : $DEVICE"
echo "  output  : $OUT_FILE"
echo "  summary : $SUMMARY_FILE"
echo "  repeat  : $REPEAT_DOC_COUNT"
echo

export MODEL_PATH
export PROMPT_PATH
export DEVICE
export MAX_NEW_TOKENS
export TEMPERATURE
export SYSTEM_PROMPT
export USE_CHAT_TEMPLATE
export REPEAT_DOC_COUNT
export OUT_FILE
export SUMMARY_FILE

cd "$ROOT_DIR"

uv run --with accelerate --with compressed-tensors --with flash-linear-attention python - <<'PY' | tee "$OUT_FILE"
from __future__ import annotations

import json
import gc
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import accelerate
import torch
import torch.nn as _nn
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def install_qwen3_5_moe_per_expert_patch() -> str:
    """Replace the experts submodule inside Qwen3_5MoeSparseMoeBlock with a
    per-expert nn.ModuleList of Qwen3_5MoeMLP instances, so the NVFP4 checkpoint
    layout (experts.N.{gate,up,down}_proj.weight_packed) loads 1:1 into
    per-expert nn.Linear submodules and compressed-tensors can attach its
    NVFP4 dequant hooks. Must be called before from_pretrained.

    Why the block-level patch (and not replacing the module-level
    `Qwen3_5MoeExperts` symbol):
      * `Qwen3_5MoePreTrainedModel._init_weights` does
        `isinstance(module, Qwen3_5MoeExperts)` to initialise the fused
        gate_up_proj / down_proj 3D tensors. Our replacement has no such
        tensors, so we must NOT match that isinstance check. Leaving the
        original class alone and only patching `Qwen3_5MoeSparseMoeBlock.__init__`
        means no instance of the original class is ever constructed, the
        isinstance init branch is a no-op, and our per-expert Linears get the
        normal nn.Linear init branch instead — which is what we want.

    Upstream in-memory layout: fused 3D nn.Parameters
    (experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I]).
    That layout (a) does not match this checkpoint's per-expert weight_packed
    tensors, and (b) cannot host compressed-tensors NVFP4 hooks (those attach
    to nn.Linear modules). Without this patch the expert weights are silently
    random-initialised as bf16 at ~60 GB, beyond a 32 GB RTX 5090.
    """
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _m

    if getattr(_m, "_QWEN35_PER_EXPERT_PATCHED", False):
        return "already_patched"

    MLP = _m.Qwen3_5MoeMLP
    SparseBlock = _m.Qwen3_5MoeSparseMoeBlock
    original_block_init = SparseBlock.__init__

    class PerExpertQwen3_5MoeExperts(_nn.ModuleList):
        """nn.ModuleList of per-expert Qwen3_5MoeMLP submodules. Parameter
        names become experts.N.{gate_proj,up_proj,down_proj}.weight[_packed|_scale|...]
        — matching the Infatoshi NVFP4 checkpoint exactly."""

        def __init__(self, config):
            super().__init__([
                MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ])
            self.config = config
            self.num_experts = config.num_experts

        def forward(
            self,
            hidden_states: torch.Tensor,
            top_k_index: torch.Tensor,
            top_k_weights: torch.Tensor,
        ) -> torch.Tensor:
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts
                ).permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0
                ).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                expert_mlp = self[int(expert_idx)]
                current_hidden_states = expert_mlp(current_state)
                current_hidden_states = (
                    current_hidden_states
                    * top_k_weights[token_idx, top_k_pos, None]
                )
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states

    def patched_block_init(self, config):
        # Build gate / shared_expert / shared_expert_gate exactly like upstream,
        # but substitute our per-expert ModuleList for the fused experts module.
        _nn.Module.__init__(self)
        self.gate = _m.Qwen3_5MoeTopKRouter(config)
        self.experts = PerExpertQwen3_5MoeExperts(config)
        self.shared_expert = _m.Qwen3_5MoeMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = torch.nn.Linear(
            config.hidden_size, 1, bias=False
        )

    SparseBlock.__init__ = patched_block_init
    _m._QWEN35_PER_EXPERT_PATCHED = True
    _m._QWEN35_ORIGINAL_SPARSE_INIT = original_block_init
    return "installed:per_expert_module_list_via_sparse_block_init_override"


_qwen35_moe_patch_status = install_qwen3_5_moe_per_expert_patch()


def install_blackwell_causal_conv1d_workaround() -> str:
    """Disable causal-conv1d on Blackwell consumer GPUs.

    The prebuilt causal-conv1d wheel can import successfully yet still lack an
    sm_120 kernel image, which crashes at runtime on RTX 5090 with:
    `CUDA error: no kernel image is available for execution on the device`.

    Qwen3.5-MoE already has a safe torch fallback for the convolution path, and
    FLA provides the gated-delta kernels independently, so we explicitly null
    the causal-conv1d entrypoints while keeping the rest of the fast path.
    """
    if not torch.cuda.is_available():
        return "skipped:no_cuda"

    capability = torch.cuda.get_device_capability(0)
    if capability < (12, 0):
        return f"skipped:sm_{capability[0]}{capability[1]}"

    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as _m

    _m.causal_conv1d_fn = None
    _m.causal_conv1d_update = None
    if _m.chunk_gated_delta_rule is not None and _m.fused_recurrent_gated_delta_rule is not None:
        _m.is_fast_path_available = True
        return "installed:disable_causal_conv1d_on_sm120_keep_fla"

    _m.is_fast_path_available = False
    return "installed:disable_causal_conv1d_on_sm120"


_blackwell_causal_conv1d_workaround = install_blackwell_causal_conv1d_workaround()

from chuk_lazarus.inference.backends.torch_runtime import TorchInferenceRuntime
from chuk_lazarus.inference.generation import GenerationConfig


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def print_kv_table(items: list[tuple[str, object]]) -> None:
    for key, value in items:
        print(f"{key:24} {value}")


def mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def resolve_device_index(device_str: str) -> int:
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, got {device_str!r}")
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def gpu_snapshot(device_index: int) -> dict[str, float | int | str]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return {
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "cuda_runtime": str(torch.version.cuda),
        "total_mib": round(mib(total_bytes), 2),
        "free_mib": round(mib(free_bytes), 2),
        "used_mib": round(mib(total_bytes - free_bytes), 2),
        "torch_allocated_mib": round(mib(torch.cuda.memory_allocated(device_index)), 2),
        "torch_reserved_mib": round(mib(torch.cuda.memory_reserved(device_index)), 2),
        "torch_max_allocated_mib": round(
            mib(torch.cuda.max_memory_allocated(device_index)), 2
        ),
        "torch_max_reserved_mib": round(
            mib(torch.cuda.max_memory_reserved(device_index)), 2
        ),
    }


def nvidia_smi_snapshot(device_index: int) -> dict[str, object] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
        "-i",
        str(device_index),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "available": False,
            "error": completed.stderr.strip() or completed.stdout.strip() or "unknown error",
        }
    raw = completed.stdout.strip()
    if not raw:
        return {"available": False, "error": "empty output"}
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) < 7:
        return {"available": False, "error": f"unexpected output: {raw}"}
    return {
        "available": True,
        "index": int(parts[0]),
        "name": parts[1],
        "temperature_c": float(parts[2]),
        "gpu_util_percent": float(parts[3]),
        "memory_total_mib": float(parts[4]),
        "memory_used_mib": float(parts[5]),
        "memory_free_mib": float(parts[6]),
    }


def detect_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def normalize_token_ids(raw_value: object) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return [int(token_id) for token_id in raw_value if token_id is not None]
    return [int(raw_value)]


def detect_llmcompressor() -> bool:
    return importlib.util.find_spec("llmcompressor") is not None


def collect_tensor_devices(model) -> list[str]:
    devices = {str(param.device) for param in model.parameters()}
    devices.update(str(buffer.device) for buffer in model.buffers())
    return sorted(devices)


def verify_single_device(model, target_device: str) -> list[str]:
    devices = collect_tensor_devices(model)
    normalized_target = str(torch.device(target_device))
    if devices != [normalized_target]:
        raise RuntimeError(
            f"Model tensors are not fully resident on {normalized_target!r}: {devices!r}"
        )
    return devices


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def make_failure_payload(
    *,
    stage: str,
    exc: BaseException,
    summary_payload: dict[str, object],
) -> dict[str, object]:
    payload = dict(summary_payload)
    payload["overall_status"] = "FAIL"
    payload["failure"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    return payload


def build_prompt(
    tokenizer,
    user_text: str,
    system_prompt: str,
    use_chat_template: bool,
) -> tuple[str, str, str | None]:
    if (
        use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **template_kwargs,
                )
                strategy = "tokenizer_chat_template_thinking_disabled"
            except TypeError:
                rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
                strategy = "tokenizer_chat_template"
            return (
                rendered,
                strategy,
                None,
            )
        except Exception as exc:
            fallback = (
                f"System: {system_prompt}\n\n"
                f"User:\n{user_text}\n\n"
                "Assistant:\n"
            )
            return (
                fallback,
                "fallback_after_chat_template_error",
                f"{type(exc).__name__}: {exc}",
            )
    if use_chat_template:
        return (
            (
                f"System: {system_prompt}\n\n"
                f"User:\n{user_text}\n\n"
                "Assistant:\n"
            ),
            "plain_text_chat_fallback",
            None,
        )
    return user_text, "raw_document", None


def resolve_model_class(config) -> tuple[type, str]:
    for architecture_name in list(getattr(config, "architectures", []) or []):
        model_class = getattr(transformers, architecture_name, None)
        if model_class is not None and hasattr(model_class, "from_pretrained"):
            return model_class, f"declared_architecture:{architecture_name}"
    return AutoModelForCausalLM, "auto_causal_lm_fallback"


def compressed_tensors_quant_method(config) -> str | None:
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and isinstance(config, object):
        quantization_config = getattr(config, "to_dict", lambda: {})().get("quantization_config")
    if isinstance(quantization_config, dict):
        method = quantization_config.get("quant_method")
        return str(method) if method is not None else None
    method = getattr(quantization_config, "quant_method", None)
    return str(method) if method is not None else None


def load_model(
    *,
    model_path: Path,
    config,
    device: str,
    accelerate_available: bool,
    llmcompressor_available: bool,
) -> tuple[object, str, dict[str, object], str]:
    normalized_device = str(torch.device(device))
    if not normalized_device.startswith("cuda"):
        raise RuntimeError(
            f"GPU-only manual test requires a CUDA device, got {normalized_device!r}."
        )
    if not accelerate_available:
        raise RuntimeError(
            "GPU-only direct load requires accelerate in the project venv. "
            "Run `uv pip install accelerate` and re-run the script."
        )

    quant_method = compressed_tensors_quant_method(config)
    model_class, class_strategy = resolve_model_class(config)
    if quant_method == "compressed-tensors":
        import compressed_tensors  # noqa: F401 - fail fast if missing

        strategy = f"transformers_native_compressed_tensors:{class_strategy}"
    else:
        strategy = f"accelerate_device_map_gpu_only:{class_strategy}"
    load_kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "device_map": {"": normalized_device},
    }
    model = model_class.from_pretrained(model_path, **load_kwargs)
    return model, strategy, load_kwargs, model_class.__name__


def build_stop_tokens(model) -> list[int]:
    stop_tokens: list[int] = []
    stop_tokens.extend(normalize_token_ids(getattr(model.generation_config, "eos_token_id", None)))
    stop_tokens.extend(normalize_token_ids(getattr(model.config, "eos_token_id", None)))
    return list(dict.fromkeys(stop_tokens))


def parse_first_json_line(text: str) -> tuple[dict[str, object] | None, str | None]:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        if not isinstance(payload, dict):
            return None, f"Expected a JSON object on the first non-empty line, got {type(payload).__name__}."
        return payload, None
    return None, "Model output did not contain a non-empty first line."


model_path = Path(os.environ["MODEL_PATH"])
prompt_path = Path(os.environ["PROMPT_PATH"])
device = os.environ["DEVICE"]
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
temperature = float(os.environ["TEMPERATURE"])
system_prompt = os.environ["SYSTEM_PROMPT"]
use_chat_template = os.environ["USE_CHAT_TEMPLATE"] == "1"
repeat_doc_count = int(os.environ["REPEAT_DOC_COUNT"])
summary_file = Path(os.environ["SUMMARY_FILE"])
accelerate_available = detect_accelerate()
llmcompressor_available = detect_llmcompressor()

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available on this host.", file=sys.stderr)
    raise SystemExit(1)

device_index = resolve_device_index(device)
torch.cuda.set_device(device_index)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
model_type = str(getattr(config, "model_type", "") or "")
architectures = list(getattr(config, "architectures", []) or [])
config_class = type(config).__name__

if "qwen" not in model_type.lower() and not any("qwen" in str(a).lower() for a in architectures):
    print(
        f"ERROR: model does not look like a Qwen family checkpoint: "
        f"model_type={model_type!r} architectures={architectures!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
document = prompt_path.read_text()
if repeat_doc_count > 1:
    document = "\n\n".join(
        f"[DOCUMENT COPY {i + 1}/{repeat_doc_count}]\n{document}"
        for i in range(repeat_doc_count)
    )

rendered_prompt, prompt_render_strategy, prompt_warning = build_prompt(
    tokenizer, document, system_prompt, use_chat_template
)
prompt_tokens = int(tokenizer(rendered_prompt, return_tensors="pt")["input_ids"].shape[1])
prompt_chars = len(rendered_prompt)

expected = {
    "early_anchor": "amber-lathe-5090",
    "middle_anchor": "copper-window-bound",
    "late_anchor": "obsidian-residual-lantern",
    "project_codename": "vermilion-archive",
    "function_name": "merge_layers",
}

preload_torch_snapshot = gpu_snapshot(device_index)
preload_smi_snapshot = nvidia_smi_snapshot(device_index)

summary_base = {
    "overall_status": "RUNNING",
    "model": {
        "path": str(model_path),
        "config_class": config_class,
        "model_type": model_type,
        "architectures": architectures,
        "quant_method": compressed_tensors_quant_method(config),
        "tokenizer_class": type(tokenizer).__name__,
        "accelerate_available": accelerate_available,
        "llmcompressor_available": llmcompressor_available,
        "accelerate_version": accelerate.__version__,
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
    },
    "prompt": {
        "path": str(prompt_path),
        "chars": prompt_chars,
        "tokens": prompt_tokens,
        "repeat_doc_count": repeat_doc_count,
        "use_chat_template": use_chat_template,
        "prompt_render_strategy": prompt_render_strategy,
        "prompt_warning": prompt_warning,
    },
    "gpu": {
        "before_load": preload_torch_snapshot,
        "before_load_nvidia_smi": preload_smi_snapshot,
    },
}

print_section("Run Configuration")
print_kv_table(
    [
        ("model_path", model_path),
        ("prompt_path", prompt_path),
        ("summary_file", summary_file),
        ("device", device),
        ("max_new_tokens", max_new_tokens),
        ("temperature", temperature),
        ("use_chat_template", use_chat_template),
        ("repeat_doc_count", repeat_doc_count),
        ("accelerate_available", accelerate_available),
        ("llmcompressor_available", llmcompressor_available),
        ("accelerate_version", accelerate.__version__),
        ("torch_version", torch.__version__),
        ("python_executable", sys.executable),
        ("qwen35_moe_patch", _qwen35_moe_patch_status),
        ("blackwell_causal_conv1d_workaround", _blackwell_causal_conv1d_workaround),
    ]
)

print_section("Model Metadata")
print_kv_table(
    [
        ("config_class", config_class),
        ("model_type", model_type),
        ("architectures", architectures),
        ("quant_method", compressed_tensors_quant_method(config)),
        ("tokenizer_class", type(tokenizer).__name__),
    ]
)

print_section("Prompt Stats")
print_kv_table(
    [
        ("prompt_chars", prompt_chars),
        ("prompt_tokens", prompt_tokens),
        ("doc_source_bytes", prompt_path.stat().st_size),
        ("render_strategy", prompt_render_strategy),
        ("expected_anchor_1", expected["early_anchor"]),
        ("expected_anchor_2", expected["middle_anchor"]),
        ("expected_anchor_3", expected["late_anchor"]),
    ]
)
if prompt_warning:
    print(f"prompt_warning           {prompt_warning}")

print_section("GPU Before Load")
print_kv_table(list(preload_torch_snapshot.items()))
if preload_smi_snapshot is not None:
    print("nvidia_smi:")
    print_kv_table(list(preload_smi_snapshot.items()))

try:
    load_start = time.time()
    model, load_strategy, load_kwargs, load_model_class = load_model(
        model_path=model_path,
        config=config,
        device=device,
        accelerate_available=accelerate_available,
        llmcompressor_available=llmcompressor_available,
    )
    load_elapsed = time.time() - load_start
    model.eval()
    tensor_devices = verify_single_device(model, device)
    torch.cuda.synchronize(device_index)
    postload_torch_snapshot = gpu_snapshot(device_index)
    postload_smi_snapshot = nvidia_smi_snapshot(device_index)
except Exception as exc:
    failure_payload = make_failure_payload(
        stage="load_model",
        exc=exc,
        summary_payload={
            **summary_base,
            "loader": {
                "strategy": "failed_before_completion",
            },
        },
    )
    write_summary(summary_file, failure_payload)
    print_section("Failure")
    print_kv_table(
        [
            ("stage", failure_payload["failure"]["stage"]),
            ("type", failure_payload["failure"]["type"]),
            ("message", failure_payload["failure"]["message"]),
            ("summary_json", summary_file),
        ]
    )
    print(failure_payload["failure"]["traceback"])
    raise SystemExit(2)

print_section("Model Load")
print_kv_table(
    [
        ("load_seconds", f"{load_elapsed:.2f}"),
        ("load_strategy", load_strategy),
        ("loader_model_class", load_model_class),
        ("torch_dtype", getattr(model, "dtype", "unknown")),
        ("model_class", type(model).__name__),
        ("tensor_devices", tensor_devices),
    ]
)
print(f"load_kwargs              {load_kwargs}")

print_section("GPU After Load")
print_kv_table(list(postload_torch_snapshot.items()))
if postload_smi_snapshot is not None:
    print("nvidia_smi:")
    print_kv_table(list(postload_smi_snapshot.items()))

runtime = TorchInferenceRuntime(
    model,
    tokenizer,
    device=device,
    engine="kv_direct",
    family_type="qwen",
)

generation_stop_tokens = build_stop_tokens(model)

try:
    torch.cuda.reset_peak_memory_stats(device_index)
    generate_start = time.time()
    result = runtime.generate(
        rendered_prompt,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=generation_stop_tokens,
        ),
    )
    generate_elapsed = time.time() - generate_start
    torch.cuda.synchronize(device_index)

    postgen_torch_snapshot = gpu_snapshot(device_index)
    postgen_smi_snapshot = nvidia_smi_snapshot(device_index)
    response_text = result.text
    path_value = runtime.last_generation_path
except Exception as exc:
    failure_payload = make_failure_payload(
        stage="generate",
        exc=exc,
        summary_payload={
            **summary_base,
            "loader": {
                "strategy": load_strategy,
                "load_kwargs": load_kwargs,
                "load_seconds": load_elapsed,
                "tensor_devices": tensor_devices,
            },
            "gpu": {
                **summary_base["gpu"],
                "after_load": postload_torch_snapshot,
                "after_load_nvidia_smi": postload_smi_snapshot,
            },
        },
    )
    write_summary(summary_file, failure_payload)
    print_section("Failure")
    print_kv_table(
        [
            ("stage", failure_payload["failure"]["stage"]),
            ("type", failure_payload["failure"]["type"]),
            ("message", failure_payload["failure"]["message"]),
            ("summary_json", summary_file),
        ]
    )
    print(failure_payload["failure"]["traceback"])
    raise SystemExit(4)

json_payload, json_error = parse_first_json_line(response_text)
expected_json_keys = [
    "early_anchor",
    "middle_anchor",
    "late_anchor",
    "project_codename",
    "target_language",
    "read_whole_document",
    "summary",
]

checks = {
    "path_is_kv_direct": path_value == "torch.kv_direct.past_key_values",
    "has_early_anchor": expected["early_anchor"] in response_text,
    "has_middle_anchor": expected["middle_anchor"] in response_text,
    "has_late_anchor": expected["late_anchor"] in response_text,
    "has_project_codename": expected["project_codename"] in response_text,
    "has_merge_layers_function": f"def {expected['function_name']}" in response_text,
    "has_python_fence": "```python" in response_text.lower(),
    "mentions_rtx_5090": "rtx 5090" in response_text.lower(),
    "mentions_kv_direct": "kv-direct" in response_text.lower(),
    "json_first_line_parseable": json_payload is not None,
    "json_keys_exact": list(json_payload.keys()) == expected_json_keys if json_payload else False,
    "json_early_anchor_match": json_payload.get("early_anchor") == expected["early_anchor"]
    if json_payload
    else False,
    "json_middle_anchor_match": json_payload.get("middle_anchor") == expected["middle_anchor"]
    if json_payload
    else False,
    "json_late_anchor_match": json_payload.get("late_anchor") == expected["late_anchor"]
    if json_payload
    else False,
    "json_project_codename_match": json_payload.get("project_codename") == expected["project_codename"]
    if json_payload
    else False,
    "json_target_language_match": json_payload.get("target_language") == "python"
    if json_payload
    else False,
    "json_read_whole_document_true": json_payload.get("read_whole_document") is True
    if json_payload
    else False,
    "json_summary_mentions_rtx_5090": "rtx 5090" in str(json_payload.get("summary", "")).lower()
    if json_payload
    else False,
    "json_summary_mentions_kv_direct": "kv-direct" in str(json_payload.get("summary", "")).lower()
    if json_payload
    else False,
}

content_required = [
    "has_early_anchor",
    "has_middle_anchor",
    "has_late_anchor",
    "has_project_codename",
    "has_merge_layers_function",
    "has_python_fence",
    "mentions_rtx_5090",
    "mentions_kv_direct",
    "json_first_line_parseable",
    "json_keys_exact",
    "json_early_anchor_match",
    "json_middle_anchor_match",
    "json_late_anchor_match",
    "json_project_codename_match",
    "json_target_language_match",
    "json_read_whole_document_true",
    "json_summary_mentions_rtx_5090",
    "json_summary_mentions_kv_direct",
]
content_pass = all(checks[name] for name in content_required)
overall_status = "PASS" if checks["path_is_kv_direct"] and content_pass else "FAIL"

print_section("Generation Metrics")
print_kv_table(
    [
        ("generation_seconds", f"{generate_elapsed:.2f}"),
        ("path", path_value),
        ("stop_reason", result.stop_reason),
        ("output_tokens", result.stats.output_tokens),
        ("tokens_per_second", f"{result.stats.tokens_per_second:.2f}"),
    ]
)

print_section("GPU After Generate")
print_kv_table(list(postgen_torch_snapshot.items()))
if postgen_smi_snapshot is not None:
    print("nvidia_smi:")
    print_kv_table(list(postgen_smi_snapshot.items()))

print_section("Verification Checks")
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
if json_error:
    print(f"json_error               {json_error}")
print(f"OVERALL_STATUS={overall_status}")

print_section("Model Output")
print(response_text)

summary_payload = {
    "overall_status": overall_status,
    "model": {
        "path": str(model_path),
        "config_class": config_class,
        "model_type": model_type,
        "architectures": architectures,
        "quant_method": compressed_tensors_quant_method(config),
        "tokenizer_class": type(tokenizer).__name__,
        "accelerate_available": accelerate_available,
        "llmcompressor_available": llmcompressor_available,
        "accelerate_version": accelerate.__version__,
        "torch_version": torch.__version__,
        "python_executable": sys.executable,
    },
    "prompt": {
        "path": str(prompt_path),
        "chars": prompt_chars,
        "tokens": prompt_tokens,
        "repeat_doc_count": repeat_doc_count,
        "use_chat_template": use_chat_template,
        "prompt_render_strategy": prompt_render_strategy,
        "prompt_warning": prompt_warning,
    },
    "loader": {
        "strategy": load_strategy,
        "load_kwargs": load_kwargs,
        "tensor_devices": tensor_devices,
    },
    "generation": {
        "device": device,
        "path": path_value,
        "stop_reason": str(result.stop_reason),
        "output_tokens": result.stats.output_tokens,
        "tokens_per_second": result.stats.tokens_per_second,
        "generation_seconds": generate_elapsed,
        "load_seconds": load_elapsed,
        "stop_tokens": generation_stop_tokens,
    },
    "checks": checks,
    "json_error": json_error,
    "json_output": json_payload,
    "gpu": {
        "before_load": preload_torch_snapshot,
        "after_load": postload_torch_snapshot,
        "after_generate": postgen_torch_snapshot,
        "before_load_nvidia_smi": preload_smi_snapshot,
        "after_load_nvidia_smi": postload_smi_snapshot,
        "after_generate_nvidia_smi": postgen_smi_snapshot,
    },
    "response_excerpt": response_text[:1200],
}
summary_file.write_text(json.dumps(summary_payload, indent=2))

print_section("Artifacts")
print_kv_table(
    [
        ("summary_json", summary_file),
        ("log_file", os.environ.get("OUT_FILE", "captured by shell tee")),
    ]
)

if path_value != "torch.kv_direct.past_key_values":
    print(
        f"ERROR: expected PATH=torch.kv_direct.past_key_values, got {path_value!r}",
        file=sys.stderr,
    )
    summary_payload["overall_status"] = "FAIL"
    write_summary(summary_file, summary_payload)
    raise SystemExit(3)
if overall_status != "PASS":
    print("ERROR: model output failed manual verification checks.", file=sys.stderr)
    write_summary(summary_file, summary_payload)
    raise SystemExit(5)
PY
