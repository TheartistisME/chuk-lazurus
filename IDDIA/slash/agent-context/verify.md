---
description: Build a verification context package from the DDIA zvec index.
argument-hint: <agent task>
allowed-tools: Bash(python -m IDDIA*)
---

# /agent-context:verify

Use the agent task in `$ARGUMENTS`.

Run:

```bash
python -m IDDIA package --stage verify --task "$ARGUMENTS" --next-steps "Run quality gates and compare derived artifacts against durable sources." --output IDDIA/artifacts/ddia/packages/latest-verify.md
```

Read the package, run the relevant checks, then continue with `/agent-context:handoff`.
