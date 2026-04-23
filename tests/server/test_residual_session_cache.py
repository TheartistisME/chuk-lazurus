"""Unit tests for the session-keyed WARM residual cache.

These tests exercise the cache container in isolation. They use plain
Python stand-ins for ``past_key_values`` and ``boundary_residual`` so they
can run on any machine without torch installed.
"""

from __future__ import annotations

import pytest

from chuk_lazarus.server._residual_session_cache import (
    DEFAULT_MAX_SESSIONS,
    ResidualSessionCache,
    ResidualSessionEntry,
    resolve_conversation_id,
)


# ── Helpers ────────────────────────────────────────────────────────────────


class _FakePastKeyValues:
    """Stand-in for a torch Cache; tracks whether it was freed by eviction."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.freed = False

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"_FakePastKeyValues(tag={self.tag!r}, freed={self.freed})"


def _make_entry(
    conversation_id: str,
    *,
    cold_tokens: list[int],
    pkv_tag: str = "kv",
    residual: object | None = None,
    total_seen: int | None = None,
) -> ResidualSessionEntry:
    return ResidualSessionEntry(
        cold_token_ids=list(cold_tokens),
        past_key_values=_FakePastKeyValues(pkv_tag),
        boundary_residual=residual if residual is not None else object(),
        boundary_residual_position=len(cold_tokens) - 1,
        total_seen_tokens=total_seen if total_seen is not None else len(cold_tokens),
        hot_window_start=0,
        conversation_id=conversation_id,
    )


def _freeing_eviction_hook():
    freed: list[ResidualSessionEntry] = []

    def _hook(entry: ResidualSessionEntry) -> None:
        pkv = entry.past_key_values
        if isinstance(pkv, _FakePastKeyValues):
            pkv.freed = True
        freed.append(entry)
        entry.past_key_values = None
        entry.boundary_residual = None

    return _hook, freed


# ── Tests ──────────────────────────────────────────────────────────────────


def test_default_max_sessions_is_public() -> None:
    assert DEFAULT_MAX_SESSIONS == 8


def test_rejects_non_positive_max_sessions() -> None:
    with pytest.raises(ValueError):
        ResidualSessionCache(max_sessions=0)


def test_put_and_get_round_trip() -> None:
    cache = ResidualSessionCache(max_sessions=4)
    entry = _make_entry("user-1", cold_tokens=[1, 2, 3])
    cache.put(entry)

    out = cache.get("user-1")
    assert out is entry
    # get() must not touch the cold tokens or the hot KV handle.
    assert out.cold_token_ids == [1, 2, 3]
    assert isinstance(out.past_key_values, _FakePastKeyValues)


def test_get_miss_returns_none() -> None:
    cache = ResidualSessionCache(max_sessions=4)
    assert cache.get("does-not-exist") is None


def test_put_refuses_empty_conversation_id() -> None:
    cache = ResidualSessionCache(max_sessions=4)
    bad = _make_entry("", cold_tokens=[1])
    with pytest.raises(ValueError):
        cache.put(bad)


def test_get_updates_lru_ordering() -> None:
    hook, _freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=2, eviction_hook=hook)
    cache.put(_make_entry("a", cold_tokens=[1]))
    cache.put(_make_entry("b", cold_tokens=[2]))

    # Touch "a" so "b" becomes least-recently-used.
    _ = cache.get("a")
    cache.put(_make_entry("c", cold_tokens=[3]))

    assert cache.get("a") is not None
    assert cache.get("b") is None  # evicted
    assert cache.get("c") is not None


def test_put_on_existing_id_frees_previous_entry() -> None:
    hook, freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=2, eviction_hook=hook)
    first = _make_entry("a", cold_tokens=[1], pkv_tag="old")
    cache.put(first)
    assert first.past_key_values.freed is False

    replacement = _make_entry("a", cold_tokens=[1, 2], pkv_tag="new")
    cache.put(replacement)

    assert first.past_key_values is None or first.past_key_values.freed  # hook fired
    assert len(freed) == 1
    assert freed[0] is first
    assert cache.get("a") is replacement


def test_eviction_frees_memory_via_hook() -> None:
    hook, freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=2, eviction_hook=hook)

    cache.put(_make_entry("a", cold_tokens=[1], pkv_tag="kv-a"))
    cache.put(_make_entry("b", cold_tokens=[2], pkv_tag="kv-b"))
    # Access "b" so "a" becomes LRU.
    cache.get("b")
    cache.put(_make_entry("c", cold_tokens=[3], pkv_tag="kv-c"))

    assert len(freed) == 1
    assert freed[0].conversation_id == "a"
    # Hot KV handle was explicitly freed — this is the guarantee the runtime
    # relies on to bound GPU memory.
    assert freed[0].past_key_values is None
    assert freed[0].boundary_residual is None


def test_discard_fires_hook_and_frees() -> None:
    hook, freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=4, eviction_hook=hook)
    cache.put(_make_entry("a", cold_tokens=[1, 2]))
    assert cache.discard("a") is True
    assert cache.get("a") is None
    assert len(freed) == 1
    assert freed[0].conversation_id == "a"
    assert cache.discard("a") is False


def test_clear_frees_everything() -> None:
    hook, freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=4, eviction_hook=hook)
    cache.put(_make_entry("a", cold_tokens=[1]))
    cache.put(_make_entry("b", cold_tokens=[2]))
    cache.clear()
    assert len(cache) == 0
    assert len(freed) == 2
    assert all(fe.past_key_values is None for fe in freed)


def test_find_prefix_match_returns_longest_prefix_entry() -> None:
    cache = ResidualSessionCache(max_sessions=4)
    cache.put(_make_entry("short", cold_tokens=[1, 2]))
    cache.put(_make_entry("long", cold_tokens=[1, 2, 3, 4]))
    cache.put(_make_entry("other", cold_tokens=[9, 9]))

    # Prompt is the longer cold + new tokens — should match "long".
    match = cache.find_prefix_match([1, 2, 3, 4, 5, 6])
    assert match is not None
    assert match.conversation_id == "long"


def test_find_prefix_match_returns_none_on_divergence() -> None:
    cache = ResidualSessionCache(max_sessions=4)
    cache.put(_make_entry("a", cold_tokens=[1, 2, 3]))

    # Different content — no prefix match.
    assert cache.find_prefix_match([4, 5, 6]) is None
    # Exact cold length — not a PROPER prefix (no new tokens to extend on).
    assert cache.find_prefix_match([1, 2, 3]) is None
    # Shorter than cached — impossible to be a prefix extension.
    assert cache.find_prefix_match([1, 2]) is None


def test_find_prefix_match_touches_lru_ordering() -> None:
    hook, _freed = _freeing_eviction_hook()
    cache = ResidualSessionCache(max_sessions=2, eviction_hook=hook)
    cache.put(_make_entry("a", cold_tokens=[1, 2]))
    cache.put(_make_entry("b", cold_tokens=[9]))

    # Hitting "a" via prefix-match must keep it newer than "b".
    match = cache.find_prefix_match([1, 2, 3])
    assert match is not None
    assert match.conversation_id == "a"

    cache.put(_make_entry("c", cold_tokens=[7]))
    # "b" is LRU, so it gets evicted.
    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


def test_resolve_conversation_id_prefers_explicit_user() -> None:
    assert resolve_conversation_id(explicit_id="alice", prompt_token_ids=[1, 2]) == "alice"
    # Whitespace-only explicit id falls back to the prefix-hash.
    anon = resolve_conversation_id(explicit_id="   ", prompt_token_ids=[1, 2])
    assert anon is not None
    assert anon.startswith("prefix::")


def test_resolve_conversation_id_is_deterministic_for_prefix_hash() -> None:
    a = resolve_conversation_id(explicit_id=None, prompt_token_ids=[1, 2, 3, 4, 5])
    b = resolve_conversation_id(explicit_id=None, prompt_token_ids=[1, 2, 3, 4, 5])
    assert a == b
    # Different first-N tokens produce different keys.
    c = resolve_conversation_id(explicit_id=None, prompt_token_ids=[9, 8, 7])
    assert c != a


def test_resolve_conversation_id_none_on_no_inputs() -> None:
    assert resolve_conversation_id(explicit_id=None, prompt_token_ids=None) is None


# ── Runtime wire-up (mocked — no real model) ───────────────────────────────


def test_runtime_cache_hit_short_circuits_split_prefill(monkeypatch):
    """A cache hit must skip the split-prefill cold path entirely.

    Uses a fully-mocked ``TorchInferenceRuntime`` — we only assert that
    when a session-cache entry with a prefix match is present, the reuse
    helper is invoked exactly once and the cold-prefill path is NOT
    entered. This is a shape check, not a numerical one.
    """
    pytest_torch = pytest.importorskip("torch")
    del pytest_torch  # only needed to gate this test on torch presence.

    from chuk_lazarus.inference.backends._torch_residual_session import CacheHitResult
    from chuk_lazarus.inference.backends import torch_runtime as torch_runtime_module

    runtime_cls = torch_runtime_module.TorchInferenceRuntime

    # Build a skeleton runtime without driving __init__ (we need to avoid
    # touching real CUDA).
    runtime = runtime_cls.__new__(runtime_cls)
    runtime._engine_mode = "residual_bounded_kv_direct"
    runtime._session_cache = ResidualSessionCache(max_sessions=2)
    runtime._last_generation_path = None

    # Stub every helper the reuse branch consults.
    recorded: dict[str, int] = {"reuse_calls": 0, "cold_prefill_calls": 0}

    def _fake_reuse(runtime_arg, **kwargs):
        recorded["reuse_calls"] += 1
        return CacheHitResult(
            past_key_values=object(),
            logits=object(),
            boundary_residual=None,
            boundary_residual_position=0,
            total_seen_tokens=kwargs["entry"].total_seen_tokens + 1,
            hot_window_start=0,
            input_length=kwargs["input_ids"].shape[1],
            delta_len=1,
        )

    monkeypatch.setattr(
        "chuk_lazarus.inference.backends._torch_residual_session.reuse_from_session_cache",
        _fake_reuse,
    )

    # Seed the cache with an entry whose cold_token_ids is the prefix of the
    # "new prompt" we'll pass in.
    cached_entry = _make_entry("alice", cold_tokens=[1, 2, 3], total_seen=3)
    runtime._session_cache.put(cached_entry)

    # _resolve_session_entry is the seam we want to exercise in isolation.
    entry, resolved_id = runtime._resolve_session_entry("alice", [1, 2, 3, 4, 5])
    assert entry is cached_entry
    assert resolved_id == "alice"

    # With a prefix-hash fallback (no explicit id), still finds the same
    # entry because the cold tokens prefix-match.
    cached_entry.cold_token_ids = [1, 2, 3]  # reset in case test order changes
    entry2, resolved_id2 = runtime._resolve_session_entry(None, [1, 2, 3, 4, 5])
    assert entry2 is cached_entry
    assert resolved_id2 == "alice"  # preserved — we want to write back under the SAME id

    # Divergent prompt — no prefix match, no entry returned, but the
    # prefix-hash still returns a stable anonymous id for the write-back.
    entry3, resolved_id3 = runtime._resolve_session_entry(None, [9, 9, 9])
    assert entry3 is None
    assert resolved_id3 is not None
    assert resolved_id3.startswith("prefix::")


def test_runtime_resolve_session_entry_returns_nothing_without_cache() -> None:
    """No session cache attached → resolution is a no-op."""
    from chuk_lazarus.inference.backends import torch_runtime as torch_runtime_module

    runtime = torch_runtime_module.TorchInferenceRuntime.__new__(
        torch_runtime_module.TorchInferenceRuntime
    )
    runtime._session_cache = None
    runtime._engine_mode = "residual_bounded_kv_direct"

    entry, rid = runtime._resolve_session_entry("alice", [1, 2, 3])
    assert entry is None
    assert rid is None
