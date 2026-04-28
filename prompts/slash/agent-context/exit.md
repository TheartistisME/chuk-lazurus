---
description: Build an exit context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:exit

Use the agent task in `$ARGUMENTS`.

Run:

```bash
lazarus agent-context package --stage exit --task "$ARGUMENTS" --next-steps "File follow-ups, close issues, commit, sync, push, and leave replay notes." --output artifacts/agent_context/ddia/packages/latest-exit.md
```

Read the package, complete the repo closeout workflow, then loop to `/agent-context:onboard` for the next task.
