---
name: convention-triage
description: Drain the AutoAxionLimits convention-escalation queue (pipeline/state/convention_queue.json). For each NEW convention token flagged [CONVENTION REVIEW] in production, derive the conversion offline with GPD (derivation + dimensional check + numeric spot-check + citation audit), then open ONE infrastructure PR promoting the vetted converter into the registry. Use when the user asks to drain / triage the convention queue.
---

# Convention triage (GPD drain of the escalation queue)

Implements Phase 3 of `pipeline/DESIGN_convention_escalation.md`. Production
extraction never converts an unknown coupling convention — it flags the paper
`[CONVENTION REVIEW]`, caps its confidence, and appends the token to
`pipeline/state/convention_queue.json`. This skill drains that queue **locally**,
running the same twice-proven offline workflow (rounds 1 & 2 in
`GPD/explanations/coupling-convention-conversions*`), once per convention token.

## Hard constraints
- **GPD never runs inside a production extraction.** This is the offline tier;
  it is invoked manually and its only output is a reviewable PR.
- **Never merge the PR yourself.** Convention conversions change scientific
  values — the user reviews and merges. (Per repo `CLAUDE.md`.)
- **Registry grows monotonically.** A token that has been `promoted` or marked
  `needs_human` is never re-derived — only counter-bumped in production.
- **One convention per derivation.** The steady-state marginal cost of a token
  already seen is a dict lookup; do not batch unrelated tokens into one PR.

## Preconditions (check first; stop if any fails)
1. `pipeline/state/convention_queue.json` exists and has ≥1 entry with
   `status: queued`. If none, report "queue empty — nothing to drain" and stop.
2. The GPD toolchain is available (the `gpd:*` skills and `gpd-bibliographer`
   agent resolve). This skill is local-only; it must not run in CI.
3. `gh auth status` is green and `git fetch origin` succeeds (PR target
   `FaroutYLq/AutoAxionLimits`, never `cajohare/AxionLimits`).

## Procedure

### 1. Read and group the queue
Load `convention_queue.json`. Select entries with `status == "queued"`. Group by
`cache_key` (the entries are already deduped by it, but confirm). Report the
list of tokens to drain: `cache_key`, `coupling_type`, `declared_convention`,
`count`, the `arxiv_ids`, `sample_points`, and `target_repo_file` when present.

### 2. Per NEW token — derive the conversion (GPD)
Run `gpd:explain` (or `gpd:derive-equation`) scoped to:
- the declaring paper(s) (`arxiv_ids`) and the `declared_convention` string;
- the coupling type's canonical variable (the registry's `_CANONICAL_TOKEN` /
  `evaluation/conventions.py::to_canonical`);
- the `sample_points` (paper-native values) and, when set, the
  `target_repo_file` plus its `PlotFuncs.py` plotting code (the repo-side
  storage quirks — in-code ×2m_N, mislabeled headers — are not in the paper).

Deliverable: a derivation note in `GPD/explanations/` containing
1. the closed-form conversion `declared → canonical`,
2. a **dimensional check** (both sides), and
3. a **numeric spot-check**: apply the conversion to a `sample_point` and
   confirm it lands on the `target_repo_file` / canonical scale.
Use the round-2 note as the template and the bar.

### 3. Citation audit
Run the `gpd-bibliographer` agent on the derivation note: every physical
constant / convention claim must trace to a real, attributed source. A failed
audit is a **FAIL** (step 5), not a warning.

### 4. PASS → open ONE promotion PR
On a new branch off `master` (never push to `master`), in a single
infrastructure PR:
- add the vetted converter to `evaluation/conventions.py::to_canonical` and the
  token to `classify_reported_convention`, plus the
  `pipeline/transform_guard.py` normalization if the flag is extractor-side;
- add tests mirroring `evaluation/tests` / `test_conventions_round2.py`
  (round-trip + the spot-check value);
- include the derivation note;
- flip the drained queue entries to `status: "promoted"` and set their
  `pr_url`;
- **Impact block** (required for any evaluation-affecting PR): local rescore of
  any cached snapshot in `evaluation/eval_runs/` that declared the token, before
  vs after. State the residual change.
Then monitor CI to green. An `UNCONVERTIBLE` verdict (e.g. oscillating-EDM
amplitudes with no constant conversion) is a valid PASS — record
"verified non-convertible" in the note and set the entry to `needs_human` with
that reason so it is never re-derived.

### 5. FAIL → mark needs_human
If no closed form exists, the conversion is ρ-dependent / model-locked without
GT support, or the citation audit fails: flip the entry to
`status: "needs_human"` with a one-line `failure_reason`. The production PR's
`[CONVENTION REVIEW]` cap remains the of-record flag. Commit the queue update on
the same branch/PR (queue-state change is infra, not science).

### 6. Report
Summarize per token: PASS (promoted, PR link) / FAIL (needs_human, reason) /
UNCONVERTIBLE. Never merge; hand the PR(s) to the user.
