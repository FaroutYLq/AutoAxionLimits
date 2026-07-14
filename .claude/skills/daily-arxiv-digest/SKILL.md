---
name: daily-arxiv-digest
description: Run the AutoAxionLimits daily arXiv digest pipeline on the Claude Code subscription (no ANTHROPIC_API_KEY). Monitors arXiv for new dark-matter limit papers, extracts data, and opens one PR per new limit. Use when the user asks to run the daily digest / arxiv monitor locally.
---

# Daily arXiv digest (subscription backend)

Runs `pipeline.orchestrator` with the headless-`claude` transport
(`AAL_BACKEND=claude-cli`), so extraction bills to the user's Claude Code
subscription instead of an API key. **Methodology is identical to the API/CI
run** — only the transport differs.

## Guardrails
- **Never merge or push science PRs.** This skill only *opens* PRs; the user reviews and merges.
- Confirm before the state-branch push (step 6) — it is a side-effectful `git push`.
- Do the whole run in a throwaway git worktree, never in the user's checkout: the pipeline switches branches and pushes, and a crash mid-run could otherwise strip the working tree.

## Preconditions (check first, stop if any fails)
1. `gh auth status` is green (PR creation uses the `gh` CLI, pushing to `origin` → `FaroutYLq/AutoAxionLimits`).
2. `claude --version` works and `claude` is logged in to a Pro/Max subscription (an unauthenticated CLI aborts the run with exit 2 at the preflight ping).
3. LaTeX is installed (`which pdflatex`) — needed for highlighted-plot regeneration.
4. `git fetch origin` succeeds.

## Procedure

**Important:** every `python -m pipeline...` and `claude -p` command must run with
the shell sandbox **disabled** (network + macOS keychain access are required to
reach arXiv and to read the subscription OAuth token). If a Bash call fails with
"Not logged in" or a network error, re-run it non-sandboxed.

1. **Worktree.** From the repo root:
   ```bash
   git fetch origin master
   WT="$(mktemp -d)/aal-daily"
   git worktree add "$WT" origin/master
   cd "$WT"
   ```
2. **Restore processed-papers baseline** from the state branch (so already-handled
   papers are not re-proposed — this mirrors the CI restore step exactly):
   ```bash
   git fetch origin chore/update-pipeline-state \
     && git checkout FETCH_HEAD -- pipeline/state/processed.json \
     || echo "no state branch yet; using master baseline"
   ```
3. **Dry run first** to preview what would be processed (no PRs, no writes):
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.orchestrator --dry-run
   ```
   Report the candidate papers to the user before the real run.
4. **Real run:**
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.orchestrator
   ```
   (Add `--arxiv-id 2412.12345` to force one paper, or `--max-papers N` to cap.)
   Exit code 2 = API/subscription unavailable (auth/usage-limit); nothing was
   marked processed — report and stop.
5. **Collect results:** list the branches/PRs the run opened (`gh pr list --repo FaroutYLq/AutoAxionLimits --author @me`).
6. **State push-back — ASK FIRST.** The run updated `pipeline/state/processed.json`.
   Show the diff, then ask the user before pushing:
   ```bash
   git add pipeline/state/processed.json
   git commit -m "chore: update pipeline state"
   git push -f origin HEAD:chore/update-pipeline-state
   ```
   Then open/reuse the single state PR (`gh pr create --base master --head chore/update-pipeline-state ...` if none is open). Skip this entirely if a CI run may be doing it — the branch is force-pushed and races are messy.
7. **Cleanup:** `cd` back to the repo root and `git worktree remove "$WT"`.

## Summary to give the user
Papers processed, PRs opened (with URLs), papers skipped/failed, and whether the
state branch was pushed. If exit code 2, say the subscription/auth was
unavailable and no state changed.
