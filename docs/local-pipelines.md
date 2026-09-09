# Local pipeline skills

Use the daily, weekly or backfill skill in Claude Code or Codex to find papers,
preview changes, or propose updates for review. Both agents use the same runner.
Convention triage remains a separate Claude/GPD workflow.

[Project overview](../README.md) · [Pipeline details](pipeline.md) · [Model backends](../pipeline/BACKENDS.md)

## Before running

- Run the shell helper from the repository containing the skills. It activates
  **straxion** before importing Python. Install pipeline dependencies there using
  `requirements_pipeline.txt`; real plot regeneration also needs LaTeX.
- Choose the backend explicitly if the user specified one: `--backend api` uses
  `ANTHROPIC_API_KEY`; `--backend claude-cli` uses the authenticated Claude CLI.
  The runner honors `AAL_BACKEND`, otherwise defaults to `claude-cli`. It does not
  change model selection, prompts or numerical guards. Check CLI login with
  `claude auth status`; use `gh auth status` for remote access and real PR runs.
  Never print credentials. In Claude Code, run every launcher command with the
  shell sandbox **disabled**: the Claude CLI reads its OAuth token from the macOS
  keychain and the run needs network access. A sandboxed run aborts at the
  preflight ping with exit 2 ("Not logged in"); re-run it non-sandboxed before
  treating exit 2 as a usage-window outage.
- The runner runs the **committed** revision it fetches as remote `master`, the
  production code, regardless of which branch the source checkout is on.
  Uncommitted code and data are never copied. `--ref <revision>` runs another
  committed revision for validation; a revision that is not already on remote
  master may only be previewed (`--dry-run`), because science PRs branch from the
  run's master and would carry its unmerged commits. `--allow-unmerged-ref`
  overrides that deliberately.
- The child does not inherit `AAL_*` or `EXTRACTOR_MODEL`/`REVIEWER_MODEL`
  overrides a benchmark shell may have left exported (operational knobs such as
  `AAL_CLI_TIMEOUT` and the download caches pass through). The runner prints what
  it ignored and the pipeline logs the resolved models at startup. Pass an
  explicit override with `--env NAME=VALUE` before `--`; it is recorded in
  `run.json`.
- A request for a real pipeline run includes its ordinary science proposal PRs.
  A request for a preview or discovery does not authorize extraction/publication
  beyond that scope. Preserve prior authorization; do not require repeated
  confirmation for actions the user has already requested. Never merge science
  PRs or push to master.

## Run and resume

```bash
bash scripts/run_pipeline.sh run daily --backend claude-cli -- --dry-run --max-papers 1
bash scripts/run_pipeline.sh run daily --run-dir /path/printed/by/runner -- --max-papers 1
```

Runner options go **before** `--`; existing pipeline arguments go **after** it.
`--run-dir` reuses an existing clone, baseline, and queue. Keep the same backend
when resuming (the runner enforces it). A fresh run restores its owned state
files (table below) from their state branches. A missing branch or file uses the
master baseline; network/auth errors stop setup instead of silently pretending
that the branch is absent.

Each run uses a separate local clone with its own `master`, because the existing
pipelines switch back to that branch between papers. Git objects are shared with
the source repository to avoid copying its large history. Keep the source object
store available while retaining runs. A local lock prevents concurrent use of the
same run directory; it does not coordinate independent runs or GitHub Actions.
The child always uses the clone's convention queue, overriding any inherited
`AAL_CONVENTION_QUEUE` from a benchmark, and does not write to `GITHUB_OUTPUT`.
Ctrl-C interrupts the pipeline (SIGINT to its process group, up to 30 s to
finish the current step; a second Ctrl-C kills it) and records the attempt as
`interrupted`; the pipeline withdraws the processed mark of a paper whose PR
was never created, so it is retried rather than silently retired.

By default runs live under `~/.local/state/autoaxionlimits/runs` (or
`$XDG_STATE_HOME/autoaxionlimits/runs`). The runner retains `checkout/`, `run.json`, per-attempt logs, and before/after JSON
state snapshots. It records PR URLs appearing in that attempt's log, rather than
listing every historical PR by the user. Use logs for scientific outcomes: exit
zero alone does not mean every paper succeeded. Additional extraction output
(including convention queue/cache changes) stays in the retained checkout and
snapshots; report it for separate review rather than silently discarding it.

`--dry-run` still performs network and model work. It suppresses durable pipeline
state writes, including processed IDs, preprint versions, backfill queue,
convention queue and derived-convention cache. It creates no science PRs.
Downloads/logs may still be written. `backfill --discover-only` normally saves a
queue; add `--dry-run` to preview discovery without saving it.

Exit 2 signals an availability failure in the pipeline; earlier completed work
may already be saved. Other nonzero exits require inspecting the log. Keep the
run directory, inspect saved progress, and resume after resolving the cause.
If a crash leaves the clone on a feature branch, the runner refuses reuse: inspect
and preserve the changes before returning that clone to its local master.

## Publish state

Publication pushes to a shared branch and opens a PR, so it is a separate
side-effect from the run. **Always show the owned-state diff and ask first**,
even when the run itself was authorized: a successful run does not prove its
bookkeeping is right (a paper recorded as processed without a PR would never be
retried). Never publish a preview.

```bash
bash scripts/run_pipeline.sh state-diff /path/printed/by/runner
bash scripts/run_pipeline.sh publish-state /path/printed/by/runner
```

| Pipeline | Owned state | Reused PR branch |
| --- | --- | --- |
| Daily | `processed.json`, `convention_queue.json` | `chore/update-pipeline-state` |
| Weekly | `preprint_versions.json` | `chore/update-preprint-state` |
| Backfill | `backfill_state.json` | `chore/update-backfill-state` |
| Backfill | `processed.json`, `convention_queue.json` | `chore/update-pipeline-state` |

For each branch, publication merges the owned files onto the branch's current tip
(three-way: the copy restored at setup, this run's copy, the tip's copy) and pushes
with a lease on that tip, so a GitHub Actions run that advanced the branch in the
meantime is absorbed rather than rejected. Lists merge as keyed sets: backfill
queue consumption is honoured, and no other file ever loses entries. The commit
contains **only** the owned files on top of the tip (or remote master when the
branch is new); science changes and other state files are excluded. It opens or
reuses a PR against master and never merges it. If the tip moves while
publication is in progress, it stops without pushing; simply retry. The runner
refuses to publish while the clone is off master (an interrupted paper) and
refuses to bypass the lease with an unconditional force push.

Keep recovery files until progress is published or intentionally retained
elsewhere. The runner never deletes a checkout automatically. Scheduling stays
with GitHub Actions or an explicitly requested automation.
