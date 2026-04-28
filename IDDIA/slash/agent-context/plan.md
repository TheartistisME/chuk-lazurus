---
description: Build a planning context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:plan

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage plan --task "$ARGUMENTS" --next-steps "Turn the task into replayable stages, contracts, and validation gates." --output IDDIA/artifacts/ddia/packages/latest-plan.md
```

Read the package, make the implementation plan concrete, then continue with `/agent-context:build`.
