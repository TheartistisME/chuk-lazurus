---
description: Build a handoff context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:handoff

Use the agent task in `$ARGUMENTS`.

Run:

```bash
lazarus agent-context package --stage handoff --task "$ARGUMENTS" --next-steps "Summarize durable state, verification, remaining work, and the next command." --output artifacts/agent_context/ddia/packages/latest-handoff.md
```

Read the package, write the handoff, then continue with `/agent-context:exit`.
