# /euclid axis-chat-save-emit (run-4 Axis-3)

LEAD: kv-memory-implementation-lead-axis-chat-glue
LEAD session: ve-ses-0moedyvk500003e7028
Mission: chuk-lazurus-zfs (Axis-3)
Branch: impl/kv-memory-finalize-run-4

## CLAIM

A unified `emit_store(self, *, force: bool = False) -> bool` method on `MemoryChat` (scripts/interactive_memory_chat.py:1656) is the single encoder for both `/save` and session-end (atexit / quit-command / EOF) per Q3 supervisor decision; it is `.dirty`-flag-gated per Addition 2, idempotent across crashes, and clears the flag only on successful encode.

## SUB-CLAIMS

### SUB-CLAIM 1 — CONFIRMED
`emit_store` reads `<store_root>/.dirty`. When absent and `force=False`, prints `"no changes since last save"` and returns False without invoking `save_current_session`. PASS test: `test_emit_store_no_op_when_clean` (lines 100-118 of tests/integration/test_chat_save_emit_emit_store.py). FAIL behavior: any branch that calls `save_current_session` despite the flag being clean would regress.

### SUB-CLAIM 2 — CONFIRMED
`force=True` bypasses the `.dirty` check. PASS test: `test_emit_store_force_bypasses_dirty_check`.

### SUB-CLAIM 3 — CONFIRMED
On successful `save_current_session()`, the `.dirty` flag is unlinked. PASS test: `test_emit_store_clears_dirty_on_success`.

### SUB-CLAIM 4 — CONFIRMED
On `save_current_session()` failure (returns False), `.dirty` is preserved. Crash-safety invariant: a failed encode does NOT lose the user's intent to save. PASS test: `test_emit_store_keeps_dirty_on_failure`.

### SUB-CLAIM 5 — CONFIRMED
Repeat invocations are idempotent: first call succeeds + clears `.dirty`; subsequent calls hit the no-op-when-clean branch. `save_current_session` is invoked exactly once across N calls. PASS test: `test_emit_store_idempotent_repeat_calls`.

### SUB-CLAIM 6 — CONFIRMED
Empty session (no `session.turns`) clears any stale `.dirty` and returns False without invoking `save_current_session`. PASS test: `test_emit_store_empty_session_clears_stale_flag`.

### SUB-CLAIM 7 — CONFIRMED
`_mark_dirty()` writes `<store_root>/.dirty` as a 0-byte sentinel; idempotent. PASS test: `test_mark_dirty_writes_flag`.

### SUB-CLAIM 8 — CONFIRMED
`force=True` does NOT bypass the empty-session guard. PASS test: `test_emit_store_force_with_empty_session_returns_false`.

### SUB-CLAIM 9 — CONFIRMED (CUDA-gated full prefill smoke)
End-to-end: real Gemma-4-E2B-it bf16 on RTX 5090 + `start_new_session` + `begin_turn`/`finish_turn` + `_capture_turn_text_live` populates LiveIndexer + `_mark_dirty` + `emit_store()` writes:
- `<store_root>/checkpoints/<sid>/torch_store/manifest.json`
- `<store_root>/checkpoints/<sid>/save-state.json`
PASS test: `test_emit_store_full_prefill_path_smoke`. Wallclock ~30-60s on RTX 5090.

## UNKNOWN edges

- Behaviour under simultaneous `/save` + atexit fire (race condition): the second emit_store call will see `.dirty` cleared by the first and no-op. Idempotent by construction.
- Behaviour when `<store_root>` is read-only at session-end: `_mark_dirty` and `emit_store` both swallow OSErrors with `info(...)` logs. The user is informed but the chat does not crash.
- Behaviour when `.dirty` exists on first launch (stale from prior crashed session): `emit_store` honours the flag and emits, treating any extant session.turns as the truth. Without session.turns, the flag is cleaned up.

## adaptation-status

ACHIEVED+VERIFIED. All 9 tests GREEN on RTX 5090 / Gemma-4-E2B-it bf16. Wallclock 61.6s for the full suite. Diagnostic JSONL at `prod/validation/diagnostic_axis_chat_save_emit_20260425T142352Z-a38e9b71.jsonl`.

## Test path

- Test file: `tests/integration/test_chat_save_emit_emit_store.py` (365 lines, 9 tests).
- Validator command: `uv run pytest tests/integration/test_chat_save_emit_emit_store.py -v --tb=short --no-header`
- Final verdict (run #3, post-fixes): 9 passed, 0 failed, 0 skipped.
- Iteration history:
  - Run #1 (validator first invocation): 8/9 — test 9 ChatHistory.append_user AttributeError.
  - Run #2 (post chathistory-api-fix): 8/9 — test 9 TurnRecord.model_dump AttributeError on duck-typed turns.
  - Run #3 (post turn-api-fix): 8/9 — test 9 LiveIndexer._compact: no windows captured.
  - Run #4 (post indexer-populate-fix): 9/9 PASS.

## Cross-refs

- Mission (beads): chuk-lazurus-zfs
- Axis-3 end-state: ve-ins-0moebmfex00005de840
- Mission proposal: ve-ins-0moebpfew00000db813
- Scope manifest: ve-ins-0moecnemd0000340b9c
- Baseline-of-absence: ve-ins-0moeedj8z0000ce96a3
- Recipe: ve-ins-0modtwi7v0000ff6d88
- Q3 supervisor decision: BOTH /save AND session-end via single emit_store (binding)
- Addition 2 supervisor binding: <store_dir>/.dirty flag lifecycle
- Code surgery file: scripts/interactive_memory_chat.py
- New methods: `MemoryChat.emit_store` (line 1656), `MemoryChat._mark_dirty` (line 681)
- New constants: `DIRTY_FLAG_FILENAME` (line 153)
- atexit registration: `run_repl()` line 1781
- Wired entry points: `/save` (line 1835), `/quit` + `/exit` (line 1826), EOF/Ctrl-C (line 1797)
