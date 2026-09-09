---
name: weekly-preprint-check
description: Run or preview the AutoAxionLimits weekly preprint checker locally. Use to check existing limits for revised arXiv versions, publication transitions or withdrawals; supports Claude CLI and Anthropic API backends.
---

# Weekly preprint check

Read [the shared local-run procedure](../../../docs/local-pipelines.md) before
running. Preserve the pipeline's existing comparison and removal-review rules.

Run from the repository containing this skill:

```bash
bash scripts/run_pipeline.sh run weekly -- --dry-run
```

This checks the existing data files; it has no paper-count flag. Preview still
uses network/model calls but creates no science PRs and leaves version state
unchanged. For an authorized real run omit `--dry-run`; reuse a previous run with
`--run-dir /path/to/run` before `--`. Do not require a duplicate preview when the
user has already requested a real check.

Use `--init-only` only when the user wants to establish a version baseline. It
writes state without opening science PRs; combining it with `--dry-run` previews
that baseline without saving it. Do not use initialization as an update check.

Before `publish-state`, run `state-diff` and show the user the owned-state diff;
ask unless they explicitly authorized publishing state in this request, and never
publish a preview. Launcher commands need network and keychain access (outside
the shell sandbox, with the user's permission).
Never merge update or removal proposals. Report checked files, changes and
withdrawal/removal flags from the log, PR URLs from this run, exit status, state
publication status and recovery directory. An availability failure can follow
already completed work; inspect saved state before describing what changed.
