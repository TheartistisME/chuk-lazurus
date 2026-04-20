"""Tests for the standalone token counting script."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "count_tokens.py"
SPEC = importlib.util.spec_from_file_location("count_tokens_tool", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    """Simple tokenizer stub for CLI tests."""

    def encode(self, text: str, add_special_tokens: bool = False):
        count = len(text.split())
        if add_special_tokens:
            count += 1
        return list(range(count))


class FakeStdin(io.StringIO):
    """StringIO with a predictable TTY flag."""

    def isatty(self) -> bool:
        return False


def test_count_tokens_json_supports_text_and_multiple_files(tmp_path, monkeypatch, capsys):
    file_one = tmp_path / "one.txt"
    file_two = tmp_path / "two.txt"
    file_one.write_text("alpha beta", encoding="utf-8")
    file_two.write_text("gamma delta epsilon", encoding="utf-8")

    monkeypatch.setattr(MODULE, "load_counter", lambda _: FakeTokenizer())

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
    assert payload["tokenizer"] == "fake-tokenizer"
    assert [item["tokens"] for item in payload["inputs"]] == [4, 2, 3]
    assert payload["total"]["tokens"] == 9


def test_count_tokens_total_only_uses_special_tokens(monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "load_counter", lambda _: FakeTokenizer())

    exit_code = MODULE.main(
        [
            "--tokenizer",
            "fake-tokenizer",
            "--text",
            "one two three",
            "--special-tokens",
            "--total-only",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "4"


def test_count_tokens_reads_stdin_when_piped(monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "load_counter", lambda _: FakeTokenizer())
    monkeypatch.setattr(MODULE.sys, "stdin", FakeStdin("alpha beta gamma"))

    exit_code = MODULE.main(["--tokenizer", "fake-tokenizer", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["inputs"][0]["source"] == "<stdin>"
    assert payload["inputs"][0]["tokens"] == 3
