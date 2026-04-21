# Axis-3: `session_store` Design

## Purpose

`session_store` is a **zero-modification subprocess wrapper** around the
pre-existing clause-aligned pipeline at
`tools/build_clause_aligned_store.py`. The architectural thesis is simple:
the turn-aligned workflow reuses the already-validated clause-aligned build
engine unchanged, and axis-3 is the seam that drives it on a per-session
basis.

Nothing in this module edits, imports, or patches the build script. The
script is invoked via `subprocess.run`, and its on-disk artifacts are read
back for verification.

## Module Layout

```
src/chuk_lazarus/session_store/
  __init__.py      # Public API re-exports
  layout.py        # Path/filename conventions (mirrored constants)
  invoke.py        # Subprocess wrapper + SessionStoreResult dataclass
  verify.py        # Post-build validators
  cli.py           # `python -m chuk_lazarus.session_store.cli ...`
scripts/
  build_session_store.sh    # Convenience shell wrapper (CLI is authoritative)
docs/turn_aligned/
  session_store_design.md   # This document
```

## Public API

| Symbol | Purpose |
| ------ | ------- |
| `invoke_build(...)` | Spawn the build script for a single session; return a `SessionStoreResult`. |
| `per_session_checkpoint(root, session_id)` | Derive `<root>/<session_id>`. |
| `verify_checkpoint(checkpoint)` | Assert manifests are healthy; return summary dict. |
| `verify_metadata_preservation(input_dir)` | Assert secondary metadata still lives in the input dir. |
| `SessionStoreResult` | Dataclass: `session_id`, `checkpoint`, `returncode`, `stdout`, `stderr`. |

## Per-Session Checkpoint Convention

```
<checkpoint_root>/
  <session_id>/
    torch_store/
      manifest.json               <-- clause_aligned=True
      entries.npz
      window_tokens.npz
      window_token_lists.npz
      idf.json
      keywords.json
      boundaries/window_000.npy
      boundary_residual.npy
      window_metadata.json
    clause_aligned_build_manifest.json
    torch_prefill.json
    invoke.log                    <-- written by session_store.invoke
```

Axis-4 (retrieval) enumerates children of `<checkpoint_root>/` to discover
sessions. Per-session isolation means a broken session does not poison the
retrieval corpus.

### Filename constants (mirrored from the build pipeline)

| Constant | Value | Source of truth |
| -------- | ----- | --------------- |
| `TORCH_STORE_DIR` | `torch_store` | `chuk_lazarus.cli.commands.context.prefill._torch_sidecar` |
| `MANIFEST_FILE` | `manifest.json` | `chuk_lazarus.inference.context.knowledge.torch_build` |
| `BUILD_MANIFEST_FILE` | `clause_aligned_build_manifest.json` | `tools/build_clause_aligned_store.py` (line ~60) |
| `TORCH_PREFILL_FILE` | `torch_prefill.json` | `chuk_lazarus.cli.commands.context.prefill._torch_sidecar` |

> **Note on filename.** The axis-3 spec referred to the top-level build
> manifest as `build_manifest.json`. The authoritative build script emits
> `clause_aligned_build_manifest.json`. `layout.py` uses the true literal and
> the brief's name is ignored. Both manifests exist: the top-level one at the
> checkpoint root and the torch_store one inside `torch_store/`.

## Build Invocation

`invoke_build` assembles argv as:

```
<python> <repo>/tools/build_clause_aligned_store.py \
    --input-dir <input_dir> \
    --checkpoint <checkpoint_root>/<session_id> \
    --window-size 512 \
    --overlap-tokens 64 \
    --device cuda \
    [--model <path>] \
    [--force]
```

Direct-path invocation is used because the build script relies on
`if __name__ == "__main__":` and is not packaged as an importable module.
`sys.executable` is the default interpreter.

The subprocess's stdout/stderr are captured, written to
`<checkpoint>/invoke.log`, and returned inside the `SessionStoreResult`.
Non-zero returncodes are NOT raised; the caller decides.

### Dry Run

`invoke_build(..., dry_run=True)` skips the subprocess entirely and returns
a `SessionStoreResult` whose `stdout` is the shell-quoted argv. This enables
fast unit tests for the argv-assembly path without CUDA or model loading.

## Verification

### `verify_checkpoint(checkpoint)`

Reads:
- `<checkpoint>/clause_aligned_build_manifest.json`
- `<checkpoint>/torch_store/manifest.json`

Asserts:
- `manifest.clause_aligned is True`
- `manifest.num_windows >= 1`
- `manifest.num_entries >= 1`
- `build_manifest.num_windows == manifest.num_windows`
- `build_manifest.num_entries == manifest.num_entries`

Returns `{"ok": True, "num_windows": ..., "num_entries": ..., "num_tokens": ..., "checkpoint": ...}`.

### `verify_metadata_preservation(input_dir, secondary_keys=...)`

Scans the axis-2 input directory (left untouched by the build) and asserts
that each secondary key appears in at least one file. Default keys:

- `iso_timestamp`
- `speaker_role`
- `session_uuid`
- `topic_tags`

These keys are **not consumed** by the build script (confirmed by reading
`tools/build_clause_aligned_store.py`: it only references the five
`ClauseRecord` fields). Axis-4 reads them back directly from `input_dir`.

## Handoff Contract for Axis-4 (Retrieval-Lead)

Axis-4 can rely on these invariants:

1. **Enumeration.** Every immediate subdirectory of `<checkpoint_root>/`
   whose `torch_store/manifest.json` exists and has `clause_aligned=true`
   is a valid session store.
2. **Integrity check.** Axis-4 should call
   `session_store.verify_checkpoint(<checkpoint>)` before ingestion. A
   clean return means the store is loadable.
3. **Clause alignment flag.** `torch_store/manifest.json` carries
   `clause_aligned=true`; this disambiguates clause-aligned stores from any
   generic torch-prefill stores produced elsewhere.
4. **Secondary metadata.** Axis-4 reads secondary metadata
   (`iso_timestamp`, `speaker_role`, `session_uuid`, `topic_tags`) by
   opening the original per-turn JSON files under `input_dir`. The build
   pipeline does not ingest or rewrite these fields; they ride through by
   remaining in the source directory.
5. **Filename pattern.** Per-turn JSONs are named `<NNN>_<ti>_<ci>.json`
   (zero-padded `NNN`). `clause_id` in each file matches
   `^[0-9a-f]{32}\.\d+\.\d+$` and serves as the correlation key between
   the torch_store window metadata and the axis-2 input file.

## Zero-Modification Guarantee

The central invariant of axis-3:

- `tools/build_clause_aligned_store.py` is never edited by this module,
  its CLI, its tests, or any downstream consumer.
- The guarantee is enforced by a dedicated test (`test_zero_modification`)
  which asserts that `git diff HEAD -- tools/build_clause_aligned_store.py`
  produces no output.
- Any bug discovered in the build script HALTS axis-3 work and is routed
  upstream to the build-pipeline owner; axis-3 does not patch it.

## CUDA Gating

The default `--device cuda` requires a CUDA-capable host. End-to-end smoke
tests that actually spawn the build subprocess MUST be gated with
`pytest.mark.skipif(not torch.cuda.is_available(), ...)`. Unit tests for
argv assembly and path conventions use `dry_run=True` and do NOT require
CUDA.
