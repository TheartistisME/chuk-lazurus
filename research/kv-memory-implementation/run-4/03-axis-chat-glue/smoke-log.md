# smoke-log: axis-chat-glue (run-4 Axis-3+4 BUNDLED)

LEAD session: ve-ses-0moedyvk500003e7028
Branch: impl/kv-memory-finalize-run-4
Hardware: RTX 5090 + Gemma-4-E2B-it bf16 (snapshot b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf)
Python 3.13.11; torch 2.9.1+cu128; pytest 9.0.2

## Final verdict (post all fixes)

| Axis | Test file | Tests | Wallclock | Diagnostic JSONL |
|---|---|---|---|---|
| Axis-3 | tests/integration/test_chat_save_emit_emit_store.py | 9/9 PASS | 61.6s | prod/validation/diagnostic_axis_chat_save_emit_20260425T142352Z-a38e9b71.jsonl |
| Axis-4 | tests/integration/test_chat_session_route_inject.py | 8/8 PASS | 66.9s | prod/validation/diagnostic_axis_chat_session_route_20260425T141316Z-8cef4b57.jsonl |

## Run-by-run history (Axis-3 only — Axis-4 was 8/8 first try)

### Iteration 1 — initial run
- 8/9 PASS
- Failure: `test_emit_store_full_prefill_path_smoke` — `AttributeError: 'ChatHistory' object has no attribute 'append_user'`
- Root cause: test wrote `chat.history.append_user(...)`; real API is `chat.history.add_user(...)` (src/chuk_lazarus/inference/chat.py:54).
- JSONL: prod/validation/diagnostic_axis_chat_save_emit_20260425T141316Z-8cef4b57.jsonl (FAIL)

### Iteration 2 — post ChatHistory API fix
- Fix: tests/integration/test_chat_save_emit_emit_store.py:320-321 — `append_user` → `add_user`, `append_assistant` → `add_assistant`.
- Result: 8/9 PASS (different failure).
- New failure: `AttributeError: '_T' object has no attribute 'model_dump'` at scripts/interactive_memory_chat.py:1490.
- Root cause: test synthesized turns via `type("_T", ...)()`; ChatLoopSession's TurnRecord (Pydantic BaseModel) is required for the transcript serializer's `t.model_dump(mode="json")` call.
- JSONL: prod/validation/diagnostic_axis_chat_save_emit_20260425T141637Z-17365bce.jsonl (FAIL)

### Iteration 3 — post TurnRecord API fix
- Fix: tests/integration/test_chat_save_emit_emit_store.py:322-334 — replaced type-synthesis fallback with real `chat.session.begin_turn(role, text)` + `chat.session.finish_turn(turn)` API calls (matches plain_chat_turn lines 906-908).
- Result: 8/9 PASS.
- New failure: `RuntimeError: indexer flush FAILED: LiveIndexer._compact: no windows were captured — refusing to write a manifest for an empty store.`
- Root cause: begin_turn/finish_turn create a TurnRecord but do NOT populate the LiveIndexer; the production path also calls `_capture_turn_text_live(turn)` which routes the turn text through StreamingWindower → indexer.
- JSONL: prod/validation/diagnostic_axis_chat_save_emit_20260425T142018Z-020b38bb.jsonl (FAIL)

### Iteration 4 — post indexer-populate fix
- Fix: tests/integration/test_chat_save_emit_emit_store.py — added `chat._capture_turn_text_live(user_turn)` and `chat._capture_turn_text_live(assistant_turn)` after each finish_turn; lengthened planted texts to ~70 tokens to ensure StreamingWindower flush emits at least a tail boundary.
- Result: **9/9 PASS**.
- JSONL: prod/validation/diagnostic_axis_chat_save_emit_20260425T142352Z-a38e9b71.jsonl (PASS)

## Per-test wallclock (final iteration)

### Axis-3 (61.6s total, RTX 5090 / Gemma-4 E2B bf16)

| Test | Result |
|---|---|
| test_emit_store_no_op_when_clean | PASS |
| test_emit_store_force_bypasses_dirty_check | PASS |
| test_emit_store_clears_dirty_on_success | PASS |
| test_emit_store_keeps_dirty_on_failure | PASS |
| test_emit_store_idempotent_repeat_calls | PASS |
| test_emit_store_empty_session_clears_stale_flag | PASS |
| test_mark_dirty_writes_flag | PASS |
| test_emit_store_force_with_empty_session_returns_false | PASS |
| test_emit_store_full_prefill_path_smoke (CUDA-gated) | PASS |

### Axis-4 (66.9s total, RTX 5090 / Gemma-4 E2B bf16)

| Test | Result |
|---|---|
| test_apply_token_budget_empty_pass_through | PASS |
| test_apply_token_budget_under_budget_keeps_all | PASS |
| test_apply_token_budget_over_budget_truncates_from_bottom | PASS |
| test_apply_token_budget_sorts_by_score_descending | PASS |
| test_apply_token_budget_custom_budget | PASS |
| test_apply_token_budget_falls_back_when_config_missing | PASS |
| test_amd11_global_attention_set_invariant | PASS |
| test_kv_query_turn_smoke_with_real_store_and_model (CUDA-gated) | PASS |

## Hardware profile

```
Device: cuda (RTX 5090, 32GB VRAM)
Torch: 2.9.1+cu128
Python: 3.13.11
Model snapshot: b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf (Gemma-4-E2B-it)
Dtype: bfloat16
CUDA available: True (verified at module-import time in both test modules)
```

## AMD compliance

- AMD 1 (sqlite3): every record fetch via `sqlite3 .vee/state.db "..."`.
- AMD 3 (read-only backends): no edits to chuk_lazurus/inference/backends/.
- AMD 10 (no README.md in 03/04 subdirs): only `notes.md` + `asi-evolve-interface.md` + `euclid-*.md` + `smoke-log.md` written — README.md reserved for Axis-6.
- AMD 11 (target_layer ∈ global set): assert in chat-script line 1165-1168 + runtime guard at kv_direct_adapter.py:131.
- AMD 13 (feature branch only): commit lands on impl/kv-memory-finalize-run-4 only.
- AMD 14 (CUDA-only tests): heavy tests `pytestmark` CUDA-skip; tensors device='cuda' bf16; real Gemma loaded.
