# Changelog

## 2026-05-02 - Initial Central Router Harness

- Added the standalone `David/central router.py` centralized router harness with
  neutral request/window dataclasses, route candidates, evidence supports, tier
  assignments, materialization plans, and route metadata.
- Implemented routing modes for temporal ordinal lookup, symbolic chains,
  dependency source selection, SWE-style patch targets, durable chat memory, and
  general recall.
- Added smoke coverage in `David/smoke_test_central_router.py` for
  `temporal_ordinal`, `symbolic_chain`, `dependency_source`, `patch_target`,
  `durable_chat_memory`, and `general_recall`.
- Added `David/benchmark_row_validation.py` with one representative local row
  each for MRCR, RULER, LoCo, SWE, and Chat validation.
- Documented the harness in `David/README.md` and mapped benchmark capabilities
  in `David/index.md`.
- Guaranteed that all routing modes return HOT, WARM, and COLD tier windows
  whenever at least three eligible windows exist, across both tier assignments
  and materialization windows.
- Kept the harness standalone; it is not wired into product runtime yet.
