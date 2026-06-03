# P4 — CI regression gate (wire `subset_eval.py` as a required before/after check)

> Roadmap phase **P4** of issue **#566** (governs) / **#550** Phase "wire the subset eval into CI".
> Design/scoping only — no pipeline/CI code in this doc.
> Acceptance harness: `evaluation/subset_eval.py` + `evaluation/subset_compare.py`.
> Code designed against: `AAL-integration/pipeline/extractor.py`, `.../plot_calibration.py`,
> `.../config.py`; existing CI `.github/workflows/eval_tests.yml`, `arxiv_daily.yml`.

P4 is the **outermost guard** of the #566 roadmap. P0 (fail-safe contract),
P1 (axis OCR), P2 (selector), P3 (determinism/convention) each *improve* the
extractor; P4 makes the eval that **already caught the #550+#561 regression**
(`evaluation/eval_runs/comparison.md`) a **required, automated** before/after
check, so no future `pipeline/` change can silently re-introduce the same class
of regression. P4 adds **no extraction logic** — it adds one GitHub Actions
workflow, one small pure-stdlib gate script, and a committed baseline snapshot.

---

## 1. Problem & evidence

### 1.1 The regression was caught by a human running this harness, not by CI

The #550+#561 integration branch regressed the 82-paper subset hard, and we only
know that because someone *manually* ran `subset_eval.py` and read
`comparison.md`:

| Metric | Before (master) | After (#550+#561) | Δ |
|---|---|---|---|
| Overall median residual | 0.485 dex | 0.842 dex | **+0.357** ⚠️ |
| Zero-overlap papers | 14 | 32 | **+18** ⚠️ |
| Mean interp. coverage | 63.9% | 35.4% | **−28.5 pp** ⚠️ |
| Papers compared | 45 | 24 | **−21** |
| figure_vision median resid | 0.642 dex | 1.002 dex | **+0.360** ⚠️ |
| figure_vision ≤0.3 dex | 33.3% | 25.2% | **−8.1 pp** ⚠️ |
| Zero-overlap `unit_offset` cause | 0 | 12 | **+12** ⚠️ |

Source: `evaluation/eval_runs/comparison.md` (the exact table this gate must
reproduce automatically). The 41-agent fan-out
(`per_paper_findings.md`) showed **33/41 changed papers moved away from truth** —
and *every one of those 33 regressions would have been blocked by a CI gate on
these same numbers*. Concretely, the gate would have failed on:

- the **+18 zero-overlap** jump (1509.00476, 1709.00009, 2007.13071, 1905.13650,
  2102.06722, 2202.08858, 2303.03594, 2308.06339, 2311.16364, 2401.16747,
  2403.03004, 2410.10363, 1907.11485, 2207.11968, … all flipping
  `compared→zero_overlap`),
- the **+0.357 dex** overall-median regression,
- the **crash** `1607.06083` (`compared→error`) — a `status==error` paper that a
  gate trivially rejects.

P4's job is to make that detection **mandatory and pre-merge**, not post-hoc.

### 1.2 Why a *before/after* gate (not an absolute threshold)

The eval is intrinsically noisy: the run-to-run LLM extraction **noise floor is
~0.32 dex** (pinned in `evaluation/metrics.py:586-592`, PR #545; surfaced in the
determinism table header `subset_compare.py:293`). The
`comparison.md` determinism table shows real per-paper coupling-scale spread of
**0.368 dex mean (before)** and **0.230 dex (after)** across repeats — i.e. an
*absolute* threshold on median residual would either be too loose (miss real
regressions) or flap on noise. So P4 gates on **Δ relative to a baseline
snapshot**, with the **0.32 dex noise floor as the no-regression tolerance** on
the overall median, and **strict counts** (zero-overlap, errors) that the noise
floor does not move.

### 1.3 Why this must be cheap

The eval is **expensive and slow**: each 82-paper extraction pass took
**~58 min wallclock** (`before.log` 15:51:40→16:50:40 = 59 min; `after.log`
16:10:24→17:06:26 = 56 min) at **~42 s/paper**, calling
`CLAUDE_MODEL = "claude-opus-4-8"` (`extractor.py:27`, Opus per #563). A naive
"re-extract before *and* after on every PR" gate would cost **2×82 Opus
extractions (~2 hr, ~164 API calls × multi-stage)** per push — unaffordable and
flaky. P4's central design choice is therefore **the baseline ("before") is a
committed snapshot, never re-extracted**; only the "after" side runs the API,
halving cost to **one 82-paper pass (~58 min)** — and §2.4 bounds even that.

---

## 2. Design

### 2.1 Trigger: a `paths`-filtered workflow on `pipeline/**`

New workflow **`.github/workflows/extraction_regression.yml`**, modeled on the
existing `eval_tests.yml` (same checkout/setup-python/paths-filter shape) but
gated to the extraction surface:

```yaml
on:
  pull_request:
    paths:
      - "pipeline/extractor.py"
      - "pipeline/plot_calibration.py"
      - "pipeline/config.py"
      - "pipeline/transform_guard.py"      # P0's module
      - "pipeline/reviewer.py"
      - "evaluation/subset_eval.py"
      - "evaluation/subset_compare.py"
      - "evaluation/eval_runs/baseline/**" # baseline snapshot (re-pin)
      - ".github/workflows/extraction_regression.yml"
```

Rationale: the regression came **only** from extraction/calibration code
(`extractor.py`, the new `plot_calibration.py`), so the gate fires exactly when
those files (or P0's `transform_guard.py`, or the anchor table now living near
`config.py`) change. `eval_tests.yml` already establishes the "paths-filter +
additive, independent of the daily/weekly/backfill workflows" precedent
(`eval_tests.yml:6-18`); P4 mirrors it. This workflow is **separate** from the
no-API `eval_tests.yml` because it *does* call the API and is much slower — they
must not share a job.

It also supports `workflow_dispatch` (manual re-run / re-pin baseline) like the
daily/weekly workflows.

### 2.2 Baseline-snapshot strategy: commit the "before", re-extract only "after"

This is the affordability lever. Two committed artifacts:

1. **`evaluation/eval_runs/baseline/<id>.json`** — a frozen extraction snapshot
   of all 82 subset papers produced by running `subset_eval extract` once on the
   blessed `master` commit. Snapshots are **tiny** (`du -sh before` = **328 KB**
   for 82 JSONs) and are **not gitignored** (`git check-ignore` returns exit 1),
   so they commit cleanly into the repo. This is the `before` side and is **never
   re-extracted in CI**.
2. **`evaluation/eval_runs/baseline/META.json`** — `{ "baseline_commit": "<sha>",
   "model": "claude-opus-4-8", "subset_key": "union", "n_papers": 82,
   "created_utc": "...", "overall_median_resid": 0.485, "zero_overlap": 14,
   "figure_vision_median": 0.642 }` — records *which commit + model* the snapshot
   was extracted on, so the gate can warn if the baseline is stale (e.g. model
   bumped from the #563 Opus switch) and a human must re-pin.

CI flow on a PR touching `pipeline/`:

```
checkout PR head
restore baseline/*.json  (committed; load_ground_truth pairs against it)
subset_eval extract --key union --outdir $RUNNER_TEMP/after   # ONLY the after pass (API)
subset_eval compare  --before evaluation/eval_runs/baseline \
                     --after  $RUNNER_TEMP/after \
                     --out    $RUNNER_TEMP/comparison.md
gate.py $RUNNER_TEMP/after  evaluation/eval_runs/baseline  --thresholds …
post comparison.md as a PR comment (sticky)
```

The `subset_eval extract` cache logic already supports this: `cmd_extract`
skips a paper when `dest.exists() and not args.force`
(`subset_eval.py:60-62`). For the **after** dir in CI we start empty
(`$RUNNER_TEMP`), so all 82 re-extract; the **before** dir is the committed
baseline and is passed straight to `compare` without any `extract` call.

> **Re-pinning the baseline.** When extraction *legitimately* improves (P0–P3
> land), the PR author re-runs `subset_eval extract --key union --outdir
> evaluation/eval_runs/baseline --force` locally, updates `META.json`, and
> commits the new snapshot **in the same PR**. The `baseline/**` path in the
> trigger means that PR re-runs the gate against itself — but the gate compares
> *new after* vs *new baseline* (both = the improved code), so it passes by
> construction, and the diff to `baseline/*.json` is the human-auditable record
> of "the eval moved this much." This is the explicit, reviewed way to advance
> the bar — mirroring how `processed.json` is the git-tracked, human-merged
> pipeline baseline (CLAUDE.md "State persistence").

### 2.3 Pass/fail thresholds (computed by `evaluation/gate.py`)

New **`evaluation/gate.py`** — pure stdlib + the existing `subset_compare`
summary functions (no new deps; unit-testable like the #544 metric tests). It
loads `before`=baseline and `after`=PR snapshot, calls
`subset_compare._records_for` / `_summarize` (reuse, **no metric
reimplementation**), and asserts:

| # | Gate rule | Threshold | Evidence anchor |
|---|---|---|---|
| **G1** | overall median residual must not regress beyond the noise floor | `after.overall_median_resid ≤ before.overall_median_resid + 0.32` dex | noise floor 0.32 dex (`metrics.py:586`, `subset_compare.py:293`). The #550/#561 Δ was **+0.357 > 0.32 → FAIL** (exactly the regression we want blocked). |
| **G2** | zero-overlap count must not increase | `after.n_zero_overlap ≤ before.n_zero_overlap` (strict, no tolerance) | counts don't move under the noise floor. #550/#561: 14→32 (**+18 → FAIL**). |
| **G3** | no new `unit_offset` zero-overlap cause | `after.zo_causes["unit_offset"] ≤ before.zo_causes["unit_offset"]` | the dominant new failure mode: 0→12 (**FAIL**). Catches the soft-anchor snaps (1705.02290, 2102.06722, …) directly. |
| **G4** | figure_vision per-source residual must not regress beyond noise floor | `after.sources["figure_vision"].median_resid ≤ before + 0.32` dex | the #550 scoreboard row: 0.642→1.002 (**+0.360 → FAIL**). This is the per-source check the brief requires. |
| **G5** | figure_vision ≤0.3 dex fraction must not drop by > 5 pp | `after.figure_vision.frac_0_3 ≥ before − 0.05` | 33.3%→25.2% (**−8.1 pp → FAIL**). Guards "accurate, not just overlapping." |
| **G6** | zero papers with `status == error` | `after` has no `status==error` record | the crash floor. 1607.06083 went `compared→error` (**FAIL**). P0's `_safe_float` fix makes this pass; the gate keeps it passing forever. |
| **G7** | papers-compared must not collapse | `after.n_compared ≥ before.n_compared − ceil(0.32·... )` → in practice `≥ before − 3` | 45→24 (**−21 → FAIL**). A small slack (3) absorbs noise-driven status flips; −21 is far outside it. |

All Δ-thresholds tied to **0.32 dex** are the noise floor; the **count
thresholds (G2/G3/G6)** are strict because the noise floor moves *residuals*, not
*membership*. The `±0.05`/`−3` slacks in G5/G7 are set just above the observed
repeat-to-repeat flap (the determinism table's 0.230–0.368 dex spread implies
≤1–2 papers flip status across repeats on this subset) and far below the #550/#561
regression magnitudes.

**Exit contract:** `gate.py` exits **non-zero** on any failed rule, printing a
per-rule PASS/FAIL table; the workflow step fails the check. On pass it exits 0.
A `--soft` mode (for the daily pipeline, §3.3) prints the table but always exits 0.

### 2.4 Cost & runtime, and how P4 bounds them

| Lever | Value / mechanism |
|---|---|
| Subset size | **82 papers** (`subset.json` `union`); one 82-paper pass = **~58 min**, ~42 s/paper, Opus (`extractor.py:27`). |
| Baseline | committed (328 KB) → **before pass never runs** → cost halved to one pass. |
| Default CI subset | gate on the **`figure` key (56 papers)** for the *fast* required check, not the full `union`. The regression is figure-extraction-centric (#550/#561 hit figure_vision hardest: G4 row), so the 56-paper `figure` subset retains the binding signal at **~40 min / ~56 Opus extractions**. The full 82-paper `union` runs nightly (§3.3), not per-PR. |
| Caching | (a) pip cache via `actions/setup-python cache: 'pip'` (already in `arxiv_daily.yml:32`). (b) PDF/metadata cache: the extractor already memoizes `metadata_cache.json` (`evaluate.py:run_extraction`); persist `~/.cache` + `RESULTS_DIR/metadata_cache.json` across runs with `actions/cache` keyed on the subset list, so arXiv PDF downloads aren't repeated. (c) **after-dir reuse within a PR**: `subset_eval.py:60` `dest.exists() and not args.force` means a re-run of the same workflow (e.g. after a flaky-network retry) re-uses already-extracted papers in `$RUNNER_TEMP` if the runner is the same — but across fresh runners this is cold, so (b) is the real saver. |
| Concurrency | `concurrency: extraction-gate-${{ github.ref }}` with `cancel-in-progress: true` so successive pushes to a PR don't stack 40-min jobs (mirrors `backfill.yml`'s concurrency guard, CLAUDE.md "Backfill auto-scheduling"). |
| Timeout | `timeout-minutes: 90` (1.5× the observed 58-min union pass; comfortably bounds the 40-min figure pass) so a hung API call can't burn a runner. |
| Affordability budget | one figure-subset pass ≈ **56 Opus paper-extractions** per PR that touches `pipeline/`. The `paths` filter means typical doc/notebook/limit-data PRs **never** trigger it. |

### 2.5 Determinism handling

The gate must not flap on the 0.32 dex noise floor:

1. **Noise-floor tolerance** on all residual rules (G1/G4/G5) — §2.3.
2. **Median, not mean** for the overall metric (`subset_compare._summarize`
   already uses `statistics.median`, `subset_compare.py:143,162`) — robust to a
   single noisy paper.
3. **Optional `--repeats 2` smoothing on the *after* side for the residual
   rules**: `subset_eval extract --repeats 2` writes `<id>_r{k}.json`
   (`subset_eval.py:57-59`), and `determinism_report` (`subset_compare.py:179`)
   already computes per-paper scale spread; the gate can take the *median across
   repeats* of each paper's residual before summarizing, shrinking the effective
   floor toward the determinism table's 0.230 dex. **Default off** (doubles cost);
   enabled via `workflow_dispatch` input when a borderline result needs
   disambiguation.
4. **Repeat-budget escape**: if a single residual rule fails by **< 0.10 dex**
   (within the noise band but above the 0.32 tolerance is impossible by
   construction; this catches *exactly-at-threshold* flap), the gate prints
   `BORDERLINE` and the workflow re-extracts just the offending papers once more
   and re-checks — a bounded (≤ a few papers) retry, not a full re-run.

P4 itself is deterministic given a snapshot: `gate.py` is a pure function of the
two JSON dirs (the same property the #544 metric unit tests rely on), so the gate
*logic* is testable offline with fixture snapshots (e.g. the committed
`before/` and `after/` dirs already in `eval_runs/` make a perfect FAIL fixture,
and `before/` vs `before/` a perfect PASS fixture).

---

## 3. Integration points

### 3.1 Files added by P4 (no extraction code touched)
- **`.github/workflows/extraction_regression.yml`** — the gated workflow (§2.1).
  New file; sibling of `eval_tests.yml`. Does **not** modify any pipeline
  workflow (`arxiv_daily.yml` / `preprint_weekly.yml` / `backfill.yml`),
  honoring CLAUDE.md "additive, independent of the daily/weekly/backfill
  pipeline" and "PR separation."
- **`evaluation/gate.py`** — the pass/fail evaluator (§2.3). Imports
  `evaluation.subset_compare` (`run_compare`/`_summarize`/`_records_for`) and
  `evaluation.subset_eval` (`_load_subset`); **reuses** their metric logic, adds
  only the threshold comparison + exit code.
- **`evaluation/eval_runs/baseline/`** — committed snapshot + `META.json` (§2.2).
- **`evaluation/tests/test_gate.py`** — unit tests for the threshold logic using
  the existing `eval_runs/before` (PASS-vs-itself) and `before/after` (FAIL)
  fixtures; runs in the no-API `eval_tests.yml` (extend its `paths` to include
  `evaluation/gate.py` and `evaluation/tests/test_gate.py`).

### 3.2 Harness reuse (no reimplementation)
- `subset_eval.cmd_extract` (`subset_eval.py:43-66`) — used verbatim for the
  *after* pass (`--outdir $RUNNER_TEMP/after`).
- `subset_compare.run_compare` (`subset_compare.py:223`) — used verbatim to emit
  the PR-comment markdown (`--out`), so the human reviewer sees the *exact*
  `comparison.md` format the regression was found in.
- `subset_compare._summarize` / `_records_for` (`:128`,`:168`) — `gate.py` calls
  these to get the numeric dict it thresholds on. **The gate never recomputes a
  residual** — it only compares two summaries, so it can never disagree with the
  report a human reads.

### 3.3 Composition with P0–P3 and the production pipeline
- **P0–P3 are *measured* by P4.** Each of P0's six acceptance criteria
  (`P0_failsafe_contract.md` §4) maps onto a P4 gate rule: P0-§4.1↔G1, §4.2↔G2/G3,
  §4.3↔G6 (no `status==error`), §4.4↔G4/G5, §4.7 (determinism ≤0.230 dex)↔§2.5.
  When P0 lands it **re-pins the baseline** (§2.2) so its recovered numbers
  (median ≤0.485, ZO ≤14, crash gone) become the new floor; P1–P3 then can only
  improve from there. The **HARD floor (P0 R5)** additionally gets a
  *no-API* assertion in `gate.py`: scan every `after/*.json` for a
  `_sorted_median` mass/coupling outside strict `VALID_RANGES[ct]`
  (`config.py:320-334`) → fail (G6-adjacent). This makes P0's "zero HARD-floor
  violations" criterion a permanent CI invariant, runnable even on the *before*
  snapshot without the API.
- **Daily pipeline integration (soft gate).** Add an optional
  `extraction_regression.yml`-style step to a **nightly** `workflow_dispatch`/cron
  that runs the **full 82-paper `union`** with `gate.py --soft` (exit 0, comment
  only) and **auto-re-pins the baseline on green master**, so the committed
  baseline tracks the merged code and per-PR gates stay fast (figure subset). This
  is the same "git-tracked cumulative snapshot, advance periodically" model as
  `processed.json` (CLAUDE.md). It is a **separate workflow** from the per-PR gate
  and from the science pipelines (PR-separation rule).
- **P1/P2/P3 trigger inclusion.** When P1 adds an OCR module
  (`pipeline/plot_ocr.py` or similar) or P3 adds a convention normalizer, those
  filenames are added to the workflow `paths` so the gate fires on them too.

---

## 4. Acceptance criteria (measured on `subset_eval.py`)

P4 is accepted when:

1. **The gate reproduces the known FAIL.** Running `gate.py` with
   `before=evaluation/eval_runs/before` (master snapshot) and
   `after=evaluation/eval_runs/after` (#550+#561 snapshot) **exits non-zero** and
   fails **G1 (+0.357 > 0.32), G2 (14→32), G3 (0→12), G4 (+0.360), G5 (−8.1 pp),
   G6 (1607.06083 error), G7 (45→24)** — i.e. it would have blocked the merge.
2. **The gate reproduces the known PASS.** `gate.py` with `before=after=`
   `evaluation/eval_runs/before` **exits 0** (a no-op change never fails).
3. **Determinism / no-flap.** Running the gate with `before=`
   `evaluation/eval_runs/before_repeats` (r0) vs (r1) of the *same* code — i.e.
   two repeat snapshots of master — **exits 0**: every Δ is within the 0.32 dex
   tolerance and no count rule trips (validates the noise-floor tolerances against
   the measured 0.230–0.368 dex repeat spread).
4. **Cost bound met.** A dry timing run of the *figure*-key after pass completes
   in **≤ 90 min** (observed ~40 min) and issues **≤ 56 Opus extractions**; the
   `paths` filter is verified to **not** trigger on a docs-only / limit-data-only
   PR (e.g. editing `docs/dp.md`).
5. **HARD-floor invariant.** `gate.py`'s no-API VALID_RANGES scan reports **0
   violations** on a P0 baseline and **≥1** on the pre-P0 `after` snapshot
   (proving the check is live).
6. **Unit tests green.** `evaluation/tests/test_gate.py` passes in
   `eval_tests.yml` (no API), covering each G-rule's boundary at exactly
   `+0.32`/`+0.33` dex and at the count boundaries.

These are all measurable **today** against the four snapshot dirs already in
`evaluation/eval_runs/` (`before`, `after`, `before_repeats`, `after_repeats`) —
no new extraction needed to validate the gate logic itself.

---

## 5. Risks & open questions

- **Baseline staleness vs. model bumps.** The #563 Haiku→Opus switch would have
  invalidated any pre-existing snapshot. `META.json` records `model` and
  `baseline_commit`; the gate **warns (not fails)** if the running
  `CLAUDE_MODEL` ≠ baseline model, prompting a re-pin. Open question: should a
  model change *force* a re-pin (block until baseline regenerated) or just warn?
  Proposed: warn on PRs, **block** on the nightly soft gate so master never
  carries a stale-model baseline silently.
- **API flakiness → false FAIL.** A transient arXiv/Anthropic outage can drop a
  paper to `error` and trip G6. Mitigations: the extractor's existing exponential
  backoff (CLAUDE.md "API retry"); a `gate.py` distinction between
  `status==error` caused by *PDF-download* failure (infra → retry/skip, don't
  fail the gate) vs. extraction-logic error (real → fail). Open question:
  threshold for "too many infra errors → inconclusive, re-run" vs. hard fail.
- **Subset drift / overfitting.** Thresholds (0.32 floor, ±0.05, −3) are
  calibrated on 82 papers; as the subset grows the floor should be re-measured
  from `determinism_report`, not hard-coded. Open question: auto-derive the
  tolerance each run from an `after --repeats` spread instead of the fixed 0.32?
  (More robust, but doubles cost — deferred to a `--adaptive-floor` opt-in.)
- **figure vs union default.** Defaulting the *required* per-PR gate to the
  56-paper `figure` subset trades ~30% coverage of text/table papers for ~30% less
  cost. The text-path regressions (e.g. 2007.13071, 2401.16747 — soft-anchor
  snaps on `text` source) would be **missed** by a figure-only gate. Mitigation:
  the nightly `union` soft gate catches them, and G3 (`unit_offset` count) on the
  figure subset still catches most anchor snaps that *also* hit figures. Open
  question: is the right default `zero_overlap` key (32 papers, the hardest cases)
  rather than `figure`? It is smaller (32) and regression-dense, but less
  representative of "no regression on easy cases." Proposed: **`union` for the
  required gate with `--repeats 1`** if the ~58-min/§2.4 budget is acceptable to
  the maintainer; fall back to `figure` only if CI minutes are constrained.
- **Cost of the API in a public PR.** The gate calls a paid API on every
  `pipeline/`-touching PR, including from forks — a secret-exposure / cost-abuse
  surface. Mitigation: `pull_request` from forks does **not** get secrets by
  default; run the gate on `pull_request_target`-style trusted trigger **only
  after a maintainer label** (`run-eval`), so external PRs don't auto-burn the
  key. Open question: label-gated vs. maintainer-only `workflow_dispatch`.
- **Where the threshold table lives.** Proposed in `gate.py` as named constants
  (testable, single source of truth). Alternative: a `gate_thresholds.json` next
  to `META.json` so a maintainer can tune without code change. Leaning toward
  constants for the #544-style unit-test guarantee.

---

## 6. Draft sub-issue body

> **Title:** `[eval][P4] Wire subset_eval.py into CI as a required extraction-regression gate`

**Parent:** #566 (governs). Outermost guard for P0–P3. Implements #550's "wire
the subset eval into CI" item.

### Why
The #550+#561 regression (overall median residual **0.485→0.842 dex**,
zero-overlap **14→32**, figure_vision **0.642→1.002 dex**, plus a crash
`1607.06083`) was caught **only because a human ran `evaluation/subset_eval.py`
manually** and read `evaluation/eval_runs/comparison.md`. Nothing in CI would have
blocked the merge. P4 makes that before/after check **automated and required** for
any `pipeline/` extraction/calibration change, so the same class of regression
can never silently re-land.

### What
1. **Trigger** — new `.github/workflows/extraction_regression.yml`, `pull_request`
   with a `paths` filter on `pipeline/extractor.py`, `pipeline/plot_calibration.py`,
   `pipeline/config.py`, `pipeline/transform_guard.py` (P0), and the
   `evaluation/subset_*`/baseline files. Separate from the no-API `eval_tests.yml`
   and from all science pipelines (PR-separation rule). `workflow_dispatch` for
   manual re-pin.
2. **Baseline strategy** — commit a frozen `master` snapshot under
   `evaluation/eval_runs/baseline/<id>.json` (82 JSONs, **328 KB**, not
   gitignored) + `META.json` (`baseline_commit`, `model=claude-opus-4-8`,
   `subset_key`, headline numbers). CI **re-extracts only the "after" side** (one
   pass), never the baseline → halves API cost. Re-pin (with `--force`) is a
   reviewed commit in the improving PR — the diff to `baseline/*.json` is the
   audit trail (same model as git-tracked `processed.json`).
3. **Pass/fail thresholds** (in new pure-stdlib `evaluation/gate.py`, reusing
   `subset_compare._summarize`; unit-tested like the #544 metrics):
   - **G1** overall median residual: `after ≤ before + 0.32 dex` (the **0.32 dex
     noise floor**, `metrics.py:586`/#545). #550/#561 was +0.357 → FAIL.
   - **G2** zero-overlap count must not increase (strict). 14→32 → FAIL.
   - **G3** `unit_offset` zero-overlap cause must not increase. 0→12 → FAIL.
   - **G4** figure_vision per-source median residual `≤ before + 0.32 dex`.
     0.642→1.002 → FAIL.
   - **G5** figure_vision ≤0.3 dex fraction `≥ before − 5 pp`. 33.3%→25.2% → FAIL.
   - **G6** zero papers with `status==error` (crash floor) **and** zero P0
     HARD-floor (out-of-`VALID_RANGES`) medians (no-API scan). 1607.06083 → FAIL.
   - **G7** papers-compared `≥ before − 3`. 45→24 → FAIL.
   Count rules are strict (noise floor moves residuals, not membership); residual
   rules carry the 0.32 dex tolerance; G5/G7 slacks (±5 pp / −3) sit above the
   measured 0.230–0.368 dex repeat flap and far below the regression magnitudes.
   `gate.py` exits non-zero on any failure and posts `comparison.md` as a sticky
   PR comment.
4. **Cost / runtime bound** — one 82-paper Opus pass = **~58 min** (`before.log`
   59 min), ~42 s/paper. Bounds: committed baseline (no before pass); **default
   required gate on the 56-paper `figure` subset (~40 min)** with the full
   82-paper `union` on a **nightly soft gate** (`--soft`, comment-only,
   auto-re-pins on green master); pip + arXiv-PDF/metadata `actions/cache`;
   `concurrency: cancel-in-progress`; `timeout-minutes: 90`; `paths` filter so
   docs/notebook/limit-data PRs never trigger it; label-gated for fork PRs so
   secrets/cost aren't exposed.
5. **Determinism** — gate on **Δ vs baseline** (not absolute), median-based,
   0.32 dex tolerance on residuals; optional `--repeats 2` smoothing toward the
   0.230 dex determinism-table floor; `gate.py` is a pure function of two JSON
   dirs (offline-testable with the existing `before/`/`after/` fixtures).

### Acceptance (on `evaluation/subset_eval.py`, validated against existing snapshots)
1. `gate.py before=eval_runs/before after=eval_runs/after` → **exit non-zero**,
   fails G1–G7 (reproduces the blocked regression).
2. `gate.py before=after=eval_runs/before` → **exit 0** (no-op never fails).
3. `gate.py` on two master repeat snapshots (`before_repeats` r0 vs r1) → **exit
   0** (no flap within the noise floor).
4. Figure-key after pass ≤ **90 min**, ≤ **56 Opus extractions**; docs-only PR
   does **not** trigger the workflow.
5. No-API VALID_RANGES scan: **0** violations on a P0 baseline, **≥1** on the
   pre-P0 `after` snapshot.
6. `evaluation/tests/test_gate.py` green in `eval_tests.yml` (boundary tests at
   ±0.32 dex and the count edges).

### Non-goals
No extraction-logic changes (that's P0–P3). No new runtime deps (`gate.py` is
stdlib + reuse of `subset_compare`). Does not modify `arxiv_daily.yml` /
`preprint_weekly.yml` / `backfill.yml`.

### Artifacts
`evaluation/eval_runs/comparison.md` (the report this gate automates),
`evaluation/eval_runs/per_paper_findings.md`,
`evaluation/eval_runs/roadmap_design/P4_ci_gate.md`,
existing snapshot fixtures `evaluation/eval_runs/{before,after,before_repeats,after_repeats}`.
