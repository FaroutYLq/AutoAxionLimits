---
name: backfill-extraction
description: Discover or import historical dark-matter limit papers with the AutoAxionLimits INSPIRE-HEP backfill. Use for date-range and citation-filtered searches, bounded extraction batches, or resuming a saved local queue.
---

# Historical backfill

Read [the shared local-run procedure](../../../docs/local-pipelines.md) before
running. Keep the user's date range, citation threshold, coupling types and
batch limit; ask for a date range if discovery requires one and none is given.

Run from the repository containing this skill:

```bash
bash scripts/run_pipeline.sh run backfill -- --date-from 2023-01-01 --date-to 2023-12-31 --min-citations 10 --discover-only
```

`--discover-only` saves the candidate queue. It can make relevance-filter model
calls when a client is configured. For discovery without a model, choose the API
backend and unset `ANTHROPIC_API_KEY` for that command. Add `--dry-run` only if the
user wants an unsaved preview; such a preview cannot seed a later `--resume`.

Continue the saved queue in the SAME printed run directory:

```bash
bash scripts/run_pipeline.sh run backfill --run-dir /path/to/run -- --resume --max-papers 3
```

Use a small batch (three papers is a reasonable default) unless the user chooses
a limit. `--resume --dry-run` previews that batch without consuming queue items.
Avoid a second discovery in the same run: discovery replaces the queue. Pause on
subscription/auth failure; inspect the log and queue, then resume when available.
Do not schedule additional runs unless requested.

Before `publish-state`, run `state-diff` and show the user the owned-state diff;
ask unless they explicitly authorized publishing state in this request, and never
publish a preview. Launcher commands need network and keychain access (outside
the shell sandbox, with the user's permission).
Never merge scientific proposals. Report discovered/processed/skipped counts,
PR URLs from this run, remaining queue, backend, exit status, state publication
status and recovery directory. Retain unprocessed candidates and recovery files.
