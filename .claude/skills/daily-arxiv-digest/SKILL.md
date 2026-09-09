---
name: daily-arxiv-digest
description: Run or preview the AutoAxionLimits daily arXiv digest, or process a specific arXiv paper locally. Use for new-limit monitoring and extraction requests in this repository; supports Claude CLI and Anthropic API backends.
---

# Daily arXiv digest

Read [the shared local-run procedure](../../../docs/local-pipelines.md) before
running. It defines environment setup, isolation, state recovery and publication.
Use the maintained pipeline; preserve its models, scientific guards and scope.

Run from the repository containing this skill:

```bash
bash scripts/run_pipeline.sh run daily -- --dry-run --max-papers 1
```

Choose `--days-back N`, `--max-papers N`, or `--arxiv-id ID` from the user's
request. A specific ID force-processes that paper even if previously handled.
Preview performs model extraction and review but creates no science PRs and
leaves durable state unchanged. Do not automatically double the extraction cost
with a preview if the user has already requested a real run.

For an authorized real run, omit `--dry-run`. If continuing a preview, reuse its
printed directory with `--run-dir /path/to/run` before `--`; keep the intended
paper arguments. Before `publish-state`, run `state-diff` and show the user the owned-state diff;
ask unless they explicitly authorized publishing state in this request, and never
publish a preview. Launcher commands need network and keychain access (outside
the shell sandbox, with the user's permission).
Never merge scientific proposals.

Report the exact paper scope, backend, exit status, PR URLs from this run,
skips/failures from its log, state publication status and recovery directory.
