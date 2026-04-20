"""Tests for the standalone token counting script."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "count_tokens.py"
SPEC = importlib.util.spec_from_file_location("count_tokens_tool", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeStdin(io.StringIO):
    def isatty(self) -> bool:
        return False


def fake_counter(target: str = "tokenizer", label: str = "fake"):
    return MODULE.CounterSpec(target=target, label=label, count=lambda text: len(text.split()))


def test_direct_tokenizer_mode_counts_text_and_multiple_files(tmp_path, monkeypatch, capsys):
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_text("alpha beta", encoding="utf-8")
    file_two.write_text("gamma delta epsilon", encoding="utf-8")
    monkeypatch.setattr(MODULE, "build_direct_tokenizer_counter", lambda *_: fake_counter())

    exit_code = MODULE.main(
        [
            "--tokenizer",
            "fake-tokenizer",
            "--text",
            "one two three four",
            str(file_one),
            str(file_two),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "tokenizer"
    assert payload["counter"] == "fake"
    assert [item["tokens"] for item in payload["inputs"]] == [4, 2, 3]
    assert payload["total"]["tokens"] == 9


def test_codex_mode_defaults_to_gpt5_codex(monkeypatch, capsys):
    class FakeEncoding:
        name = "o200k_base"

        def encode(self, text: str):
            return list(range(len(text.split()) + 1))

    fake_tiktoken = types.SimpleNamespace(encoding_for_model=lambda model: FakeEncoding())
    monkeypatch.setitem(sys.modules, "tiktoken", fake_tiktoken)

    exit_code = MODULE.main(["--to", "codex", "--text", "alpha beta", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "codex"
    assert payload["counter"] == "gpt-5-codex (o200k_base)"
    assert payload["total"]["tokens"] == 3


def test_anthropic_mode_uses_latest_model_and_system_role(monkeypatch, capsys):
    calls: list[tuple[str, dict | None]] = []

    def fake_request(path: str, api_key: str, payload: dict | None = None) -> dict:
        calls.append((path, payload))
        if path == "/models":
            return {"data": [{"id": "claude-opus-4-1-20250805"}]}
        return {"input_tokens": 42}

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(MODULE, "anthropic_request", fake_request)

    exit_code = MODULE.main(["--to", "anthropic", "--role", "system", "--text", "hello", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["target"] == "anthropic"
    assert payload["counter"] == "claude-opus-4-1-20250805"
    assert payload["total"]["tokens"] == 42
    assert calls[0] == ("/models", None)
    assert calls[1][1] == {
        "model": "claude-opus-4-1-20250805",
        "messages": [],
        "system": "hello",
    }


def test_stdin_is_supported_in_new_interface(monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "build_direct_tokenizer_counter", lambda *_: fake_counter())
    monkeypatch.setattr(MODULE.sys, "stdin", FakeStdin("alpha beta gamma"))

    exit_code = MODULE.main(["--tokenizer", "fake-tokenizer", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["inputs"][0]["source"] == "<stdin>"
    assert payload["inputs"][0]["tokens"] == 3


def test_special_tokens_is_rejected_with_target_mode():
    with pytest.raises(SystemExit):
        MODULE.main(["--to", "codex", "--special-tokens", "--text", "hello"])
