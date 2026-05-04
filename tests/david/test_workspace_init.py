from __future__ import annotations

import json
from pathlib import Path

import pytest

from chuk_lazarus.david.workspace_init import (
    CONFIG_SCHEMA_NAME,
    CONFIG_SCHEMA_VERSION,
    WorkspaceInitError,
    initialize_workspace,
    init_workspace,
)


def test_initialize_workspace_creates_layout_and_defaults(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"

    result = initialize_workspace(workspace)

    assert result.workspace_root == workspace.resolve()
    assert result.david_dir == workspace / ".david"
    assert result.config_path == workspace / ".david" / "config.json"
    assert result.next_steps_path == workspace / ".david" / "NEXT_STEPS.md"
    for relative in ("memory", "indexes", "model_validation", "sessions"):
        assert (workspace / ".david" / relative).is_dir()

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["schema_name"] == CONFIG_SCHEMA_NAME
    assert config["schema_version"] == CONFIG_SCHEMA_VERSION
    assert config["model"] is None
    assert config["validation_report"] == ".david/model_validation/model_validation_report.json"
    assert config["model_attestation"] == ".david/model_validation/model_attestation.json"
    assert config["model_backend"] == "transformers"
    assert config["model_device"] is None
    assert config["model_dtype"] == "auto"
    assert config["model_max_new_tokens"] == 160
    assert config["auto_jit_index"] is False
    assert config["paths"] == {
        "state_dir": ".david",
        "memory_dir": ".david/memory",
        "indexes_dir": ".david/indexes",
        "model_validation_dir": ".david/model_validation",
        "sessions_dir": ".david/sessions",
    }
    assert "Next steps:" in result.next_steps_summary
    assert result.next_steps_path.read_text(encoding="utf-8") == result.next_steps_summary
    assert result.changed is True


def test_initialize_workspace_accepts_model_runtime_defaults(tmp_path: Path) -> None:
    result = init_workspace(
        tmp_path,
        model="google/gemma-e2b",
        backend="torch-runtime",
        device="cuda:0",
        dtype="bfloat16",
        max_new_tokens=88,
        auto_jit_index=True,
    )

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["model"] == "google/gemma-e2b"
    assert config["model_backend"] == "torch-runtime"
    assert config["model_device"] == "cuda:0"
    assert config["model_dtype"] == "bfloat16"
    assert config["model_max_new_tokens"] == 88
    assert config["auto_jit_index"] is True
    assert "google/gemma-e2b" in result.next_steps_summary


def test_initialize_workspace_is_idempotent_without_overwriting(tmp_path: Path) -> None:
    first = initialize_workspace(tmp_path, model="first-model")
    first.config_path.write_text('{"custom": true}\n', encoding="utf-8")
    first.next_steps_path.write_text("operator notes\n", encoding="utf-8")

    second = initialize_workspace(tmp_path, model="second-model")

    assert json.loads(second.config_path.read_text(encoding="utf-8")) == {"custom": True}
    assert second.next_steps_path.read_text(encoding="utf-8") == "operator notes\n"
    assert second.overwritten_paths == ()
    assert second.changed is False
    assert second.config_path in second.existing_paths
    assert second.next_steps_path in second.existing_paths


def test_initialize_workspace_force_overwrites_managed_files(tmp_path: Path) -> None:
    first = initialize_workspace(tmp_path, model="first-model")
    first.config_path.write_text('{"custom": true}\n', encoding="utf-8")

    result = initialize_workspace(tmp_path, force=True, model="second-model")

    config = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert config["model"] == "second-model"
    assert result.config_path in result.overwritten_paths
    assert result.next_steps_path in result.overwritten_paths


def test_initialize_workspace_refuses_file_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace-file"
    workspace.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(WorkspaceInitError, match="not a directory"):
        initialize_workspace(workspace)

    assert not (tmp_path / "workspace-file" / ".david").exists()


def test_initialize_workspace_refuses_david_path_that_resolves_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside-state"
    workspace.mkdir()
    outside.mkdir()
    try:
        (workspace / ".david").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("filesystem does not allow directory symlinks")

    with pytest.raises(WorkspaceInitError, match="escapes workspace"):
        initialize_workspace(workspace)

    assert not (outside / "config.json").exists()
