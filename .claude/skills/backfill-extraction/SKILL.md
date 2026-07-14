---
name: backfill-extraction
description: Run the AutoAxionLimits historical backfill on the Claude Code subscription (no ANTHROPIC_API_KEY). Searches INSPIRE-HEP for older dark-matter limit papers in a date range, filters by citations, and processes them through extraction, opening PRs. Use when the user asks to backfill/historical-import papers locally.
---

# Historical backfill (subscription backend)

Runs `pipeline.backfill` with the headless-`claude` transport
(`AAL_BACKEND=claude-cli`), billing to the Claude Code subscription. **Identical
methodology to the API run**; only the transport differs.

Backfill can process many papers, so it is the one pipeline most likely to hit
subscription rate windows. Keep batches small and use `--resume`.

## Guardrails
- Only *opens* PRs (many carry `[BACKFILL]`); never merge them — the user reviews.
- Confirm before the state-branch push (step 7).
- Run in a throwaway worktree, never the user's checkout.
- **Throttle:** `--max-papers 2-3` per session. If a run hits the subscription
  5-hour / weekly window mid-batch it aborts cleanly (exit 2, candidate
  re-queued); wait for the window to reset and `--resume`.

## Preconditions (stop if any fails)
1. `gh auth status` green.
2. `claude --version` works and is logged in to a Pro/Max subscription.
3. LaTeX installed (`which pdflatex`).
4. `git fetch origin` succeeds.

Note: **discovery needs no LLM** — `--discover-only` runs without a subscription
or key, so you can preview candidates even before checking auth.

## Procedure

**Important:** run every `python -m pipeline...` / `claude -p` command with the
shell sandbox **disabled** (needs network + keychain OAuth).

1. **Worktree:**
   ```bash
   git fetch origin master
   WT="$(mktemp -d)/aal-backfill"
   git worktree add "$WT" origin/master
   cd "$WT"
   ```
2. **Restore backfill-queue baseline** from its state branch (mirrors CI):
   ```bash
   git fetch origin chore/update-backfill-state \
     && git checkout FETCH_HEAD -- pipeline/state/backfill_state.json \
     || echo "no state branch yet; starting fresh"
   ```
3. **Discover candidates** (no LLM, no writes) — review before extracting:
   ```bash
   python -m pipeline.backfill --date-from 2020-01-01 --date-to 2024-12-31 \
     --min-citations 10 --discover-only
   ```
   Show the user the candidate count and let them confirm the date range /
   citation threshold / coupling types before spending subscription tokens.
4. **Dry run** one small batch to sanity-check extraction:
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.backfill --resume --max-papers 2 --dry-run
   ```
5. **Real run,** small batch:
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.backfill --resume --max-papers 3
   ```
   Repeat with `--resume` to drain the queue over several sessions. Exit code 2 =
   subscription/auth unavailable or window exhausted; the candidate is
   re-queued — stop and resume later.
6. **Collect results:** `gh pr list --repo FaroutYLq/AutoAxionLimits --author @me`.
7. **State push-back — ASK FIRST.** The run updated
   `pipeline/state/backfill_state.json`. Show the diff, then on approval:
   ```bash
   git add pipeline/state/backfill_state.json
   git commit -m "chore: update backfill state"
   git push -f origin HEAD:chore/update-backfill-state
   ```
   Open/reuse the single state PR. Skip if CI may be pushing the same branch.
8. **Cleanup:** `cd` back and `git worktree remove "$WT"`.

## Summary to give the user
Candidates discovered, papers processed this session, PRs opened (URLs), queue
remaining, and whether the state branch was pushed. Exit code 2 → subscription
window/auth issue; note how many remain and that `--resume` continues.
