"""Targeted tests for tokenizer asset download patterns."""

from __future__ import annotations

from unittest.mock import patch

from chuk_lazarus.inference.loader import DownloadConfig, HFLoader


def test_download_config_includes_text_tokenizer_assets():
    config = DownloadConfig(model_id="dummy/model")
    assert "*.txt" in config.allow_patterns
    assert "*.bin" in config.allow_patterns


@patch("huggingface_hub.snapshot_download")
@patch("huggingface_hub.list_repo_files")
def test_hf_loader_download_includes_text_files(mock_list_repo_files, mock_snapshot_download):
    mock_list_repo_files.return_value = ["config.json", "vocab.json", "merges.txt"]
    mock_snapshot_download.return_value = "/tmp/model"

    HFLoader.download("dummy/model", verbose=False)

    allow_patterns = mock_snapshot_download.call_args.kwargs["allow_patterns"]
    assert "*.txt" in allow_patterns
    assert "*.bin" in allow_patterns
