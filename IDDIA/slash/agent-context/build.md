---
description: Build an implementation context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:build

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage build --task "$ARGUMENTS" --next-steps "Implement idempotent writes, rebuildable indexes, provenance, and bounded interfaces." --output IDDIA/artifacts/ddia/packages/latest-build.md
```

Read the package, implement the scoped work, then continue with `/agent-context:verify`.
