---
description: Run the chainable agent-context lifecycle in order.
argument-hint: <agent task>
allowed-tools: Bash(lazarus agent-context*)
---

# /agent-context:loop

Use the agent task in `$ARGUMENTS`.

Run the lifecycle sequentially:

```bash
lazarus agent-context package --stage onboard --task "$ARGUMENTS" --next-steps "Read manifests and source-of-truth boundaries." --output artifacts/agent_context/ddia/packages/latest-onboard.md
lazarus agent-context package --stage plan --task "$ARGUMENTS" --next-steps "Create replayable stages and validation gates." --output artifacts/agent_context/ddia/packages/latest-plan.md
lazarus agent-context package --stage build --task "$ARGUMENTS" --next-steps "Implement idempotent writes and rebuildable indexes." --output artifacts/agent_context/ddia/packages/latest-build.md
lazarus agent-context package --stage verify --task "$ARGUMENTS" --next-steps "Run checks and compare derived state to sources." --output artifacts/agent_context/ddia/packages/latest-verify.md
lazarus agent-context package --stage handoff --task "$ARGUMENTS" --next-steps "Summarize durable state, evidence, and remaining work." --output artifacts/agent_context/ddia/packages/latest-handoff.md
lazarus agent-context package --stage exit --task "$ARGUMENTS" --next-steps "Close issues, commit, sync, push, and leave replay notes." --output artifacts/agent_context/ddia/packages/latest-exit.md
```

Use each generated package as the context handoff for that stage before moving to the next.
