---
name: weekly-preprint-check
description: Run the AutoAxionLimits weekly preprint checker on the Claude Code subscription (no ANTHROPIC_API_KEY). Scans existing data files for updated arXiv preprint versions with changed results and opens PRs; flags published papers that yield no data. Use when the user asks to run the weekly preprint/version check locally.
---

# Weekly preprint checker (subscription backend)

Runs `pipeline.preprint_checker` with the headless-`claude` transport
(`AAL_BACKEND=claude-cli`), billing to the Claude Code subscription. **Identical
methodology to the API/CI run**; only the transport differs.

## Guardrails
- Only *opens* PRs (`[NEEDS REVIEW]` flags and updated-limit proposals); never merge them — the user does.
- Confirm before the state-branch push (step 6).
- Run in a throwaway worktree, never the user's checkout.

## Preconditions (stop if any fails)
1. `gh auth status` green.
2. `claude --version` works and is logged in to a Pro/Max subscription.
3. LaTeX installed (`which pdflatex`).
4. `git fetch origin` succeeds.

## Procedure

**Important:** run every `python -m pipeline...` / `claude -p` command with the
shell sandbox **disabled** (needs network + keychain OAuth). Re-run
non-sandboxed on "Not logged in"/network errors.

1. **Worktree:**
   ```bash
   git fetch origin master
   WT="$(mktemp -d)/aal-weekly"
   git worktree add "$WT" origin/master
   cd "$WT"
   ```
2. **Restore preprint-version baseline** from its state branch (mirrors CI):
   ```bash
   git fetch origin chore/update-preprint-state \
     && git checkout FETCH_HEAD -- pipeline/state/preprint_versions.json \
     || echo "no state branch yet; using master baseline"
   ```
3. **Dry run first:**
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.preprint_checker --dry-run
   ```
   Report which files would be checked / updated before the real run.
4. **Real run:**
   ```bash
   AAL_BACKEND=claude-cli python -m pipeline.preprint_checker
   ```
   (First-ever setup only: `--init-only` populates the version baseline with no
   PRs.) Exit code 2 = subscription/auth unavailable; nothing changed — stop.
5. **Collect results:** `gh pr list --repo FaroutYLq/AutoAxionLimits --author @me`.
6. **State push-back — ASK FIRST.** The run updated
   `pipeline/state/preprint_versions.json`. Show the diff, then on approval:
   ```bash
   git add pipeline/state/preprint_versions.json
   git commit -m "chore: update preprint state"
   git push -f origin HEAD:chore/update-preprint-state
   ```
   Open/reuse the single state PR. Skip if CI may be pushing the same branch.
7. **Cleanup:** `cd` back and `git worktree remove "$WT"`.

## Summary to give the user
Files checked, updated-limit PRs opened (URLs), `[NEEDS REVIEW]` flags raised,
and whether the state branch was pushed. Exit code 2 → subscription/auth
unavailable, no state changed.
