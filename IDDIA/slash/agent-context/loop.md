---
description: Run the chainable agent-context lifecycle in order.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:loop

Use the agent task in `$ARGUMENTS`.

Run the lifecycle sequentially:

```bash
python -m IDDIA package --stage onboard --task "$ARGUMENTS" --next-steps "Read manifests and source-of-truth boundaries." --output IDDIA/artifacts/ddia/packages/latest-onboard.md
python -m IDDIA package --stage plan --task "$ARGUMENTS" --next-steps "Create replayable stages and validation gates." --output IDDIA/artifacts/ddia/packages/latest-plan.md
python -m IDDIA package --stage build --task "$ARGUMENTS" --next-steps "Implement idempotent writes and rebuildable indexes." --output IDDIA/artifacts/ddia/packages/latest-build.md
python -m IDDIA package --stage verify --task "$ARGUMENTS" --next-steps "Run checks and compare derived state to sources." --output IDDIA/artifacts/ddia/packages/latest-verify.md
python -m IDDIA package --stage handoff --task "$ARGUMENTS" --next-steps "Summarize durable state, evidence, and remaining work." --output IDDIA/artifacts/ddia/packages/latest-handoff.md
python -m IDDIA package --stage exit --task "$ARGUMENTS" --next-steps "Close issues, commit, sync, push, and leave replay notes." --output IDDIA/artifacts/ddia/packages/latest-exit.md
```

Use each generated package as the context handoff for that stage before moving to the next.
