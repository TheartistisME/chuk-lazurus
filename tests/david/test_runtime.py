from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.david import assert_path_field, require_attr, require_module, value_at


def _write_validation_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_name": "lazarus.model_config_validation_report",
                "schema_version": 1,
                "validation_status": "accepted",
                "confidence": "high",
                "auto_load_allowed": True,
                "selected_config": {
                    "adapter_config_id": "gemma-e2b-test",
                    "route_layer": 11,
                    "boundary_layer": 17,
                    "kv_source_layer": 21,
                    "kv_target_layer": 23,
                    "insertion_family": "kv_direct",
                },
                "source_report_summary": {
                    "model_identity": "gemma-e2b-test",
                    "tokenizer_identity": "gemma-tokenizer-test",
                    "adapter_family": "gemma",
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _boot_session(workspace: Path) -> SimpleNamespace:
    return SimpleNamespace(
        session_id="boot-test",
        workspace_path=str(workspace),
        validation_status="accepted",
        can_auto_load=True,
        jit_required=False,
        jit_actions=(),
        index_readiness=SimpleNamespace(state="ready"),
        user_memory=SimpleNamespace(state="ready"),
        task_memory=SimpleNamespace(state="ready"),
        selected_methodology=None,
        warnings=(),
    )


def _config_kwargs(tmp_path: Path) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    model = tmp_path / "model"
    workspace.mkdir()
    model.mkdir()
    return {
        "workspace_path": workspace,
        "model_path": model,
        "validation_report_path": _write_validation_report(tmp_path / "validation.json"),
        "offline": True,
        "allow_shell": False,
        "trace_root": tmp_path / "traces",
    }


def test_runtime_boot_uses_harness_gate_without_loading_model_backends(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    runtime_module = require_module("chuk_lazarus.david.runtime")
    Config = require_attr(runtime_module, "DavidRuntimeConfig", "runtime configuration")
    Runtime = require_attr(runtime_module, "DavidRuntime", "boot-gated terminal agent runtime")

    kwargs = _config_kwargs(tmp_path)
    kwargs["offline"] = False
    config = Config(**kwargs)
    boot_calls: list[dict[str, Any]] = []

    def fake_boot_harness(
        model_path: str,
        workspace_path: str,
        validation_report_path: str | None = None,
        require_validated_model: bool = True,
    ) -> SimpleNamespace:
        boot_calls.append(
            {
                "model_path": model_path,
                "workspace_path": workspace_path,
                "validation_report_path": validation_report_path,
                "require_validated_model": require_validated_model,
            }
        )
        return _boot_session(Path(workspace_path))

    runtime = Runtime(config, boot_harness=fake_boot_harness)
    returned = runtime.boot()
    status = runtime.status()

    assert returned is runtime
    assert runtime.harness_session.session_id == "boot-test"
    assert status.harness_booted is True
    assert status.model_load_allowed is True
    assert len(boot_calls) == 1
    assert_path_field(boot_calls[0], "workspace_path", kwargs["workspace_path"])
    assert_path_field(boot_calls[0], "model_path", kwargs["model_path"])
    assert_path_field(boot_calls[0], "validation_report_path", kwargs["validation_report_path"])
    assert boot_calls[0]["require_validated_model"] is True


def test_runtime_turn_loop_executes_model_tool_calls_and_records_trace(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    runtime_module = require_module("chuk_lazarus.david.runtime")
    Config = require_attr(runtime_module, "DavidRuntimeConfig", "runtime configuration")
    Runtime = require_attr(runtime_module, "DavidRuntime", "tool-using runtime loop")

    kwargs = _config_kwargs(tmp_path)
    workspace = kwargs["workspace_path"]
    (workspace / "README.md").write_text("alpha project\n", encoding="utf-8")
    config = Config(**kwargs)

    class ScriptedDecoder:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def _reply(self, *args: Any, **kwargs: Any) -> str:
            self.calls.append((args, kwargs))
            if len(self.calls) == 1:
                return (
                    '<tool_call>{"name":"read_file",'
                    '"arguments":{"path":"README.md"}}</tool_call>'
                )
            payload = json.dumps(args, default=str) + json.dumps(kwargs, default=str)
            assert "TOOL_RESULT" in payload
            assert "alpha project" in payload
            return "README.md says alpha project."

        def complete(self, *args: Any, **kwargs: Any) -> str:
            return self._reply(*args, **kwargs)

        def generate(self, *args: Any, **kwargs: Any) -> str:
            return self._reply(*args, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any) -> str:
            return self._reply(*args, **kwargs)

    decoder = ScriptedDecoder()

    runtime = Runtime(
        config,
        streams=SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(), stderr=io.StringIO()),
        decoder=decoder,
    )
    assert hasattr(runtime, "run_once"), (
        "DavidRuntime should expose run_once(prompt) so CLI and TUI can drive a "
        "single tool-using agent turn with injected decoders and streams."
    )

    result = runtime.run_once("inspect README.md")

    assert len(decoder.calls) >= 2
    assert "alpha project" in value_at(result, "answer", str(result))
    trace_files = list(Path(kwargs["trace_root"]).glob("*.jsonl"))
    assert trace_files, "tool calls should be traced as append-only JSONL"
    trace_payload = trace_files[0].read_text(encoding="utf-8")
    assert "read_file" in trace_payload
    assert "README.md" in trace_payload


def test_runtime_detects_product_methodologies_without_benchmark_names() -> None:
    runtime_module = require_module("chuk_lazarus.david.runtime")
    Runtime = require_attr(runtime_module, "DavidRuntime", "benchmark-free product task detection")
    runtime = Runtime({"offline": True})

    patch_task = runtime.detect_task_methodology(
        "Fix the failing pytest by updating src/chuk_lazarus/runtime.py."
    )
    dependency_task = runtime.detect_task_methodology(
        "Find the source file and dependency path for LocalCodingToolRunner."
    )
    temporal_task = runtime.detect_task_methodology(
        "Recall the earlier deadline from yesterday."
    )

    patch_mode = value_at(
        patch_task,
        "capability_mode",
        value_at(patch_task, "task_type", value_at(patch_task, "methodology", str(patch_task))),
    )
    dependency_mode = value_at(
        dependency_task,
        "capability_mode",
        value_at(
            dependency_task,
            "task_type",
            value_at(dependency_task, "methodology", str(dependency_task)),
        ),
    )
    temporal_mode = value_at(
        temporal_task,
        "capability_mode",
        value_at(
            temporal_task,
            "task_type",
            value_at(temporal_task, "methodology", str(temporal_task)),
        ),
    )

    assert patch_mode in {"patch_target", "repo_patch"}
    assert dependency_mode in {
        "dependency_source",
        "source_dependency",
        "source_dependency_reasoning",
    }
    assert temporal_mode in {"temporal_ordinal", "temporal_recall"}

    primary_names = " ".join(
        str(value_at(item, "methodology", value_at(item, "capability_mode", "")))
        for item in (patch_task, dependency_task, temporal_task)
    ).lower()
    for benchmark_name in ("swe_", "loco_", "mrcr", "ruler"):
        assert benchmark_name not in primary_names
