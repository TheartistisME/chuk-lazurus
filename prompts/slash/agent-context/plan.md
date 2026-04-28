---
description: Build a planning context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:plan

Use the agent task in `$ARGUMENTS`.

Run:

```bash
lazarus agent-context package --stage plan --task "$ARGUMENTS" --next-steps "Turn the task into replayable stages, contracts, and validation gates." --output artifacts/agent_context/ddia/packages/latest-plan.md
```

Read the package, make the implementation plan concrete, then continue with `/agent-context:build`.
