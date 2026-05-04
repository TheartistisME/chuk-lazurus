from __future__ import annotations

import importlib.util

from chuk_lazarus.david.model_backend import OfflineModelBackend, TransformersCausalLMBackend


def test_offline_backend_is_deterministic_and_applies_stop() -> None:
    backend = OfflineModelBackend(prefix="test")

    first = backend.generate("alpha beta gamma", max_new_tokens=2)
    second = backend.generate("alpha beta gamma", max_new_tokens=2)
    stopped = backend.generate("alpha STOP beta", stop=["STOP"])

    assert backend.status().loaded is True
    assert first == second
    assert first.text == "test: alpha beta"
    assert stopped.text == "test: alpha "


def test_transformers_backend_reports_missing_optional_packages(monkeypatch) -> None:
    def fake_find_spec(name: str):
        if name in {"torch", "transformers"}:
            return None
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    backend = TransformersCausalLMBackend("example/local-model")

    status = backend.status()
    result = backend.generate("hello")

    assert status.available is False
    assert "missing optional packages" in status.reason
    assert result.ok is False
    assert result.text == ""
    assert result.metadata["local_files_only"] is True


def test_transformers_backend_never_downloads_by_default() -> None:
    backend = TransformersCausalLMBackend("example/local-model")

    assert backend.local_files_only is True
    assert backend.status().metadata["local_files_only"] is True
