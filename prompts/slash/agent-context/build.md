---
description: Build an implementation context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:build

Use the agent task in `$ARGUMENTS`.

Run:

```bash
lazarus agent-context package --stage build --task "$ARGUMENTS" --next-steps "Implement idempotent writes, rebuildable indexes, provenance, and bounded interfaces." --output artifacts/agent_context/ddia/packages/latest-build.md
```

Read the package, implement the scoped work, then continue with `/agent-context:verify`.
