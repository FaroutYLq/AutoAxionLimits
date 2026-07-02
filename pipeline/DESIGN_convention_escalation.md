# Design: GPD convention-escalation tier (#636)

**Status: design for review — no behavior changes in this PR.**
Implements Phase 3 of `evaluation/eval_runs/PLAN_post_full346.md`. Companion
analysis: `GPD/explanations/coupling-convention-conversions-EXPLAIN.md`
(round 1) and `...-round2-EXPLAIN.md` + `...-round2-CITATION-AUDIT.md`
(round 2), whose workflow this design productionizes.

## Problem

Convention alignment is a static, hand-curated registry
(`evaluation/conventions.py::to_canonical` / `classify_reported_convention`,
`pipeline/transform_guard.py::convention_review_needed`). It handles the
vetted families, but the long tail of published conventions is unbounded, and
two blind spots pass **silently** today:

1. **Undeclared conventions** — an empty `coupling_convention` is treated as
   canonical (deliberately, so the common case is not flagged), so a paper
   whose values are in a novel convention but whose declaration is empty is
   scored/PR'd raw.
2. **Same-unit / different-physics** — e.g. the AxionEDM *response*
   `C_G/(f_a·m_a)` vs the *operator* `g_{anγ}`: both `GeV^-2`, so string
   matching cannot distinguish them; this is the 4.6-dex error that sat
   inside the benchmark median until full346.

Twice now the fix has been the same **offline GPD workflow**: derive the
conversion with code-verified physics, dimension-check it, numerically
spot-check against the paper↔GT pair, citation-audit, then hand-encode the
result into the registry. That workflow is exactly what should run — but
**per NEW convention token, on demand**, not per incident months later.

## Design principles (from #636's constraints)

- **GPD never runs inside a production extraction.** The pipeline stays
  deterministic and cheap; the heavyweight multi-agent derivation runs
  *locally, asynchronously*, driven by a human-invoked skill.
- **Once per convention, not per paper.** The registry grows monotonically;
  the steady-state marginal cost of a convention that has been seen before is
  zero (a dict lookup).
- **Verification gates promotion.** A derivation enters the registry only
  after the same three checks the round-1/2 docs used: code-verified
  derivation + dimensional check, numeric spot-check against the target repo
  scale, and a citation audit. FAIL → stays flagged for the human.
- **Both sides.** For matching an existing repo file (weekly checker), the
  repo-side storage quirk (in-code ×2m_N, mislabeled headers) is not in the
  paper — the skill's inputs include the target repo file and `PlotFuncs.py`
  context, exactly as the round-2 audit did.

## Architecture

```
production run (deterministic, cheap)
────────────────────────────────────────────────────────────────────
 extractor → declared convention ──┬─ registry hit → convert, done
                                   ├─ UNCONVERTIBLE → exclude/flag (as today)
                                   └─ convention_review_needed
                                        │  PR keeps its [CONVENTION REVIEW] cap
                                        ▼
                        pipeline/state/convention_queue.json  (append)

local GPD drain (heavyweight, human-invoked, asynchronous)
────────────────────────────────────────────────────────────────────
 .claude/skills/convention-triage
   reads queue → groups by cache key (normalized token + coupling type)
   per NEW key: gpd:explain derivation → dimensional check → numeric
   spot-check vs target scale → gpd-bibliographer citation audit
     PASS → PR: promote converter into evaluation/conventions.py
             (+ pipeline/transform_guard.py normalization if extractor-side)
             + tests + queue entries cleared
     FAIL → queue entry marked needs_human; PR flag stands
```

### 1. Prerequisite — close the silent blind spots (code, small)

`convention_review_needed(coupling_type, declared)` gains two routes:

- **Undeclared + suspicious values**: when the declaration is empty BUT the
  emitted values violate the coupling type's canonical magnitude profile
  (reuse the per-token magnitude guards introduced in Phase 1d — e.g.
  dimensionless couplings > O(1), `GeV^-1` couplings > 1e3), flag review.
  An empty declaration with in-profile values stays unflagged (the common
  case must not regress).
- **Same-unit/different-physics sentinels**: a per-coupling list of known
  ambiguous unit strings (`AxionEDM` + `GeV^-2` is the first entry) whose
  *unit alone* cannot identify the variable; these flag review unless the
  declaration names the canonical variable explicitly (e.g. `g_angamma`).

Both routes only *flag* (confidence cap + queue append) — no new conversion
is attempted in production.

### 2. The queue — `pipeline/state/convention_queue.json`

Git-tracked, same lifecycle as `processed.json` (restored from the state
branch per #547, force-pushed back). Append-only from production; the drain
skill removes/annotates entries via its promotion PR.

```json
{
  "version": 1,
  "entries": [
    {
      "cache_key": "axionphoton::gamma_s^-1",          // normalized token + coupling
      "coupling_type": "AxionPhoton",
      "declared_convention": "decay rate Gamma in s^-1, not coupling",
      "arxiv_id": "2311.05476",
      "sample_points": [[4.1, 1.2e-27], [12.0, 4.5e-26]],   // ≤5 points
      "target_repo_file": "limit_data/AxionPhoton/DESI.txt", // when known
      "pr_url": "https://github.com/.../pull/NNN",
      "first_seen": "2026-07-02T09:00:00Z",
      "status": "queued"                                // queued | needs_human | promoted
    }
  ]
}
```

`cache_key` is the dedup unit: a token already `promoted` (now a registry
hit) or `needs_human` is appended only as a counter bump, never re-derived.
Normalization: lowercase, whitespace-collapsed declaration reduced by the
same `classify_reported_convention` tokenizer, plus the coupling type.

### 3. The drain skill — `.claude/skills/convention-triage`

A Claude Code skill (markdown instructions, no pipeline code) that runs the
exact twice-proven workflow:

1. Read the queue; group by `cache_key`; skip non-`queued` keys.
2. Per new key, run `gpd:explain` scoped to: the declaring paper (arXiv id),
   the sample points, the coupling type's canonical variable
   (`_CANONICAL_TOKEN`), the target repo file + its `PlotFuncs.py` plotting
   code. Deliverable: a derivation note in `GPD/explanations/` with the
   closed form, dimensional check, and a numeric spot-check landing on the
   target scale (the round-2 doc is the template and the bar).
3. Run the `gpd-bibliographer` citation audit on the note.
4. **PASS** → open ONE infrastructure PR per drain run: the vetted
   `to_canonical` branch + `classify_reported_convention` token + magnitude
   guard + tests (mirroring `test_conventions_round2.py`), the derivation
   note, and the queue entries flipped to `promoted`. Impact block: local
   rescore of any cached snapshots that declared the token.
5. **FAIL** (no closed form / ρ-dependent / model-locked without GT support
   / audit failure) → entry flipped to `needs_human` with the failure reason;
   the production PR's `[CONVENTION REVIEW]` cap remains the of-record flag.
   `UNCONVERTIBLE` sentinels (e.g. oscillating-EDM amplitudes) are a valid
   PASS outcome — "verified non-convertible" is knowledge too.

Human-in-the-loop is preserved twice over: the skill is invoked manually, and
its output is a reviewable PR (never merged by the pipeline).

### 4. Cost model

- Production marginal cost: one JSON append per flagged paper (≈0).
- Drain cost: one GPD derivation per NEW convention token. Round 2 covered
  6 families / 11 papers in one session; the full346 tail suggests the
  steady-state arrival rate of genuinely new tokens is ~1–2/month of daily
  operation, decaying as the registry saturates.
- Nothing is spent on tokens already known (registry hit) or already
  adjudicated (`promoted` / `needs_human`).

### 5. Metric (report-tracked)

Add to `evaluation/report.py` + `metrics_summary.json`:

- `[CONVENTION REVIEW]` flag rate per 100 processed papers (from the queue's
  counter bumps), split queued / promoted / needs_human;
- `convention_mismatch` + guard-refusal count in the benchmark (already
  reported since Phase 1c/1d) — the macro−micro residual gap is the
  headline number this tier should shrink.

## Implementation plan (follow-up PRs, in order)

1. **Blind-spot closure** (`transform_guard.py` + tests) — small, no queue
   yet; measures the true flag rate the queue will see.
2. **Queue plumbing** (`extractor.py` wiring + `monitor.py`-style state io +
   workflow state-branch restore line) — production writes, nothing reads.
3. **`convention-triage` skill** (`.claude/skills/`) — after 1–2 have run in
   production long enough to populate a few real entries.

## Alternatives considered

- **GPD inline in the pipeline** (rejected): breaks determinism, unbounded
  per-run cost/latency, and the Actions environment lacks the local GPD
  toolchain; #636 explicitly cautions against per-paper-always.
- **Extractor self-conversion via prompting** (rejected): conversions were
  made a deterministic layer precisely because prompt-time LLM arithmetic
  produced 1401.6460's ×1.5e6 error; the #594 contract instead makes the
  extractor *declare* truthfully and leave conversion to vetted code.
- **Registry-only growth by periodic failure analysis** (status quo): works —
  rounds 1/2 prove it — but reactively, months after the papers were
  mis-scored; the queue turns the same work item around per-PR-cycle.
