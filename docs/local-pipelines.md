# Local pipeline skills

The daily, weekly and backfill skills share `scripts/run_pipeline.sh` and
`pipeline/local_runner.py`. Claude reads their definitions in `.claude/skills`;
Codex discovers the same definitions through `.agents/skills` symlinks. Convention
triage remains a separate Claude/GPD workflow and is not ported by this runner.

## Before running

- Run the shell helper from the repository containing the skills. It activates
  **straxion** before importing Python. Install pipeline dependencies there using
  `requirements_pipeline.txt`; real plot regeneration also needs LaTeX.
- Choose the backend explicitly if the user specified one: `--backend api` uses
  `ANTHROPIC_API_KEY`; `--backend claude-cli` uses the authenticated Claude CLI.
  The runner honors `AAL_BACKEND`, otherwise defaults to `claude-cli`. It does not
  change model selection, prompts or numerical guards. Check CLI login with
  `claude auth status`; use `gh auth status` for remote access and real PR runs.
  Never print credentials. Network and, for CLI authentication, keychain access
  must be available under the host's execution policy.
- The runner copies the selected **committed** revision (`--ref HEAD` by default).
  Uncommitted code and data are not copied. To run updated production code, fetch
  `origin/master` and select `--ref origin/master`; to validate an implementation,
  commit it on its development branch and select that revision.
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
when resuming. A fresh run restores only its owned state file from its state
branch. A missing branch uses the source baseline; network/auth/read errors stop
setup instead of silently pretending that the branch is absent.

Each run uses a separate local clone with its own `master`, because the existing
pipelines switch back to that branch between papers. Git objects are shared with
the source repository to avoid copying its large history. Keep the source object
store available while retaining runs. A local lock prevents concurrent use of the
same run directory; it does not coordinate independent runs or GitHub Actions.
The child always uses the clone's convention queue, overriding any inherited
`AAL_CONVENTION_QUEUE` from a benchmark, and does not write to `GITHUB_OUTPUT`.

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

For an authorized real run, inspect the changed-state report and owned state diff
before publishing. Continue without another permission question when publication
is already in scope. For a preview-only request, do not publish.

```bash
bash scripts/run_pipeline.sh publish-state /path/printed/by/runner
```

| Pipeline | Owned state | Reused PR branch |
| --- | --- | --- |
| Daily | `processed.json` | `chore/update-pipeline-state` |
| Weekly | `preprint_versions.json` | `chore/update-preprint-state` |
| Backfill | `backfill_state.json` | `chore/update-backfill-state` |

Publication builds a commit from the remote master recorded at setup plus **only** the
owned state file. Science changes and other state files are excluded. It opens or
reuses a PR against master, and never merges it. A push lease requires the remote
state branch to match the baseline restored at setup (or the last successful
publication). If another run advances it, publication stops and retains local
progress. Inspect both states and reconcile deliberately; never retry with an
unconditional force push or just update the expected lease to bypass the check.

Keep recovery files until progress is published or intentionally retained
elsewhere. The runner never deletes a checkout automatically. Scheduling stays
with GitHub Actions or an explicitly requested automation.
