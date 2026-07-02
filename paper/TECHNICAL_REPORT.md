# AutoAxionLimits — Technical Report

*A companion to the PAI26 paper "LLM-Powered Automation of a Dark Matter Constraint Repository." This document expands the architecture (paper §2), evaluation (§3), and lessons (§4) into a self-contained, human-readable account, and then catalogues the problems the evaluation still exposes. All numbers are from the full 346-paper benchmark run (`evaluation/report.md`, current code, N=3 read-vote, Claude Opus 4.8).*

---

## 0. Background and glossary

The goal of the system is to keep a **dark matter constraint repository** up to date automatically. Some terms used throughout:

- **Ultralight dark matter / axions / dark photons.** Hypothetical very light particles that could make up dark matter. They are searched for through a weak **coupling** (interaction strength) to ordinary matter. Different particles couple in different ways.
- **Coupling type.** A specific particle–interaction channel, e.g. *AxionPhoton* (axion ↔ photon, strength $g_{a\gamma}$), *DarkPhoton* (kinetic mixing $\chi$), *AxionNeutron* ($g_{an}$), *ScalarElectron* ($d_{m_e}$), etc. The benchmark spans 14 of these. Each has its own units and parameterisation.
- **Exclusion plot / constraint curve.** Experiments report which combinations of particle **mass** (x-axis, in eV) and **coupling** (y-axis) are ruled out, drawn as a curve in a log–log plane. The region above/inside the curve is excluded. The repository stores each such curve as a two-column text file `mass [eV], coupling`.
- **The repository.** [AxionLimits](https://github.com/cajohare/AxionLimits) (and our fork): ~600 limit curves and ~160 projected sensitivities, with Python (`PlotFuncs.py`) and Jupyter notebooks that render publication-quality figures. Maintained by one volunteer — the sustainability problem this project addresses.
- **dex.** "Decimal exponent" = one order of magnitude in $\log_{10}$. A **residual of 0.3 dex ≈ a factor of 2**; 1 dex = a factor of 10. Because couplings span many orders of magnitude, all error is measured in dex.
- **Convention.** The same physical bound can be written in different units/variables (e.g. dark-photon mixing as $\varepsilon$ or $\varepsilon^2$; axion–nucleon coupling as a dimensionless $g_{aN}$ or a GeV$^{-1}$ derivative coupling). Converting between them is exact but requires knowing each side's convention — a recurring theme below.
- **Haloscope.** A resonant-cavity experiment that searches for dark matter in the local galactic halo; relevant because its bound depends on the assumed local dark matter density.

The pipeline has four stages — **Discovery → Extraction → Integration → Review** — each detailed below.

---

# Part I — System architecture (expanding paper §2)

## I.1 Discovery

A daily job pulls new submissions from the arXiv RSS feed and narrows them in three steps of increasing cost:

1. **Keyword filter.** Cheap string matching against a curated keyword set (experiment names, coupling jargon) drops the obvious non-matches.
2. **LLM relevance classification.** A short LLM call on title + abstract decides whether the paper actually reports a new exclusion limit (vs. a theory paper, a recast, or an unrelated result).
3. **PDF download.** Survivors are fetched for extraction. (A real bug fixed during this work: old-style arXiv IDs like `hep-ph/0611223` carry a category-prefix slash that broke the local PDF path; the URL keeps the slash, the filename must not.)

Two sibling workflows reuse the same machinery: a **weekly preprint checker** that re-extracts tracked papers when a new arXiv version appears (it once caught a JWST dark-photon search that demoted a measured limit to a projection on publication), and a **historical backfill** that mines INSPIRE-HEP for older high-citation papers and feeds them through extraction in batches.

## I.2 Extraction — the two-stage core

Extraction turns a PDF into a structured `ExtractionResult`: coupling type, a list of `(mass, coupling)` data points, the declared units/convention, whether it is a new limit or a projection, an extraction-confidence score, and notes.

**Stage 1 — text and tables (preferred).** The PDF is parsed with PyMuPDF and the full text is sent to the LLM with a structured prompt that demands JSON output. Text/tables are preferred because they are cheaper (no image tokens) and, when a paper tabulates its bound, far more accurate than reading it off a plot. Prompt-injection defences strip control characters and wrap the paper text in sentinel delimiters so instructions embedded in a PDF cannot hijack the model.

**Stage 2 — vision fallback (only when needed).** Many papers publish their limit *only* as a figure. When Stage 1 is **not clearly dominant** — defined as failing any of: a valid mass window, ≥5 data points, extraction-confidence ≥0.6, and a non-degenerate (non-flat) curve — selected figure pages are rendered to images and the model reads axis labels, tick spacing, and curve positions off the log–log plot. This gating keeps the expensive vision path off the majority of papers (text wins on 166 of 346) while still covering plot-only results.

### I.2.1 Consensus across noisy reads (read-vote)

A single LLM read of a dense log–log plot is unreliable *and* irreproducible: re-running the same paper gives different curves. We therefore read each paper **N = 3 times independently** and reconcile the reads (`pipeline/read_vote.py`):

- **Coupling type:** majority vote across the 3 reads.
- **Curve:** the **medoid** — among the reads that agree on the modal coupling type and have ≥2 points, pick the one whose curve is *most central*, i.e. has the smallest **median pairwise distance** to the others. "Distance" between two curves is the median $|\Delta\log_{10}(\text{coupling})|$ evaluated at 25 log-spaced mass points in their overlapping range.

The medoid is deliberately **a real sample, not an average**. Averaging two incompatible curves would manufacture a bound no experiment reported (and possibly a non-physical one); the medoid returns an actual extracted curve while discarding one-off outliers. The run-to-run spread this damps is quantified in §II.5 (a 0.32 dex noise floor).

### I.2.2 Quality-tier source selection

When the surviving candidate curves still disagree (e.g. a sparse text bound vs. a dense figure curve), they are ranked by a **lexicographic quality tuple** (`pipeline/transform_guard.py`) rather than by raw point count. In priority order:

| Tier | Criterion | Why it matters |
|---|---|---|
| T0 | **In valid ranges** | Hard floor: coupling/mass must lie in physically possible ranges. A curve that fails is unusable. |
| T1 | **Non-degenerate** | A traced curve must span ≥1 dex in coupling; a flat horizontal line is usually a misread axis or a fill artefact (point-limits are exempt). |
| T2 | **Recoverable** | The values become valid under some power-of-ten correction (catches a clean unit slip). |
| T3 | **Source tier** | table (4) > text (3) > figure-vision (2) > raw CV trace (0). **A sparse text/table snippet of ≤3 points is demoted below a traceable multi-point figure curve** — so a single headline number quoted in prose cannot override an actual digitised curve. |
| T4 | **Corroborated** | A second read's spot-check value agrees within a factor of 3. |
| T5 | **Confidence** | The model's self-reported extraction confidence (rounded, to avoid float jitter). |
| T6 | **Point count** | Last-resort tie-breaker. |

This ordering encodes a hard-won lesson (§IV): "more points" is not "more trustworthy," and a confidently-quoted single number is often a worse source than a noisy-but-real figure trace.

### I.2.3 Convention canonicalization

This is the layer that most distinguishes a working pipeline from a naive one, and it is **pure physics, not perception**. The same bound is stored in incompatible $y$-variables across papers, and even across files *within one coupling type*, with the conversion living in the repository's plotting code rather than in the paper. Concretely (verified against `PlotFuncs.py` and the notebooks; see §IV.2):

| Coupling family | Repository's canonical variable | A common alternative in papers | Conversion (implemented in `evaluation/conventions.py`) |
|---|---|---|---|
| AxionNeutron / AxionProton | dimensionless $g_{aN}$ | GeV$^{-1}$ derivative coupling $g_{aNN}=C_N/2f_a$ | $g_{aN}=2m_N\cdot g_{aNN}$, with $m_n=0.93957$, $m_p=0.93828$ GeV |
| AxionNeutron (SNO file only) | dimensionless $g_{an}$ | $g_{an}/m_n$ [GeV$^{-1}$] | $g_{an}=m_n\cdot(\text{stored})$ — a **per-file** exception, not per-type |
| Scalar / dilaton | dimensionless $d_e$, $d_{m_e}$ | fifth-force strength $\alpha$ | $d=Q^{-1}\sqrt{\alpha}$, $Q^{-1}=500$ (photon/nucleon), $4000$ (electron) |
| DarkPhoton | kinetic mixing $\chi=\varepsilon$ | $\varepsilon^2$ | $\chi=\sqrt{\varepsilon^2}$ |
| AxionMass / AxionEDM | $g_{a\gamma\gamma\gamma}$ [GeV$^{-2}$] | $1/f_a$ [GeV$^{-1}$] | $g_{a\gamma\gamma\gamma}=3.7\times10^{-3}\cdot(1/f_a)$ |

Two further traps the layer handles: **sentinel values** — the rows valued $10^{20}$, $10^{30}$, $10^{99}$ are not data but the closing vertices of a filled polygon (`fill_between`), and are stripped (floor $10^{19}$); and **header-vs-value disagreement** — some scalar files carry a header claiming `d_e` while the magnitudes ($10^4$–$10^{13}$) are inconsistent with it, so classification uses the value range, not the header string.

The comparator (and the production pipeline) calls `to_canonical()` to map **both** the extracted curve and the reference curve to the canonical variable *before* any residual is computed. Anything outside the verified set — most importantly the AxionEDM *response* coupling $C_G/(f_a m_a)\propto 1/m_a$, which is not a constant rescale — is returned as `UNCONVERTIBLE` and surfaces as a `[CONVENTION REVIEW]` flag with capped confidence rather than a silent wrong number.

**Provenance of the conversion factors.** These were not guessed or prompted out of the model. They were derived with a guided physics-derivation assistant (**GPD**), which produced a reference document (`GPD/explanations/coupling-convention-conversions-EXPLAIN.md`) that is *code-verified* (every factor checked against the actual repository source and notebooks) and *citation-audited* (the underlying physics — Damour–Donoghue dilaton parameterisation, the axion derivative-coupling on-shell reduction giving the $2m_N$ factor — traced to real literature, 10 references checked). Getting these exactly right is a small but exacting derivation task, and treating it as such — rather than as a prompting problem — is what made the registry trustworthy enough to apply automatically.

## I.3 Integration

The reconciled, canonicalized result is turned into a code change:

- The LLM generates a `@staticmethod` for the relevant plotting class (a post-generation guard re-inserts the decorator if the model omits it).
- The method is inserted using Python's **AST** (abstract syntax tree) module, locating the last method in the target class — never by regex string-matching, which could corrupt the file.
- A notebook cell that calls the new method is inserted via `nbformat`, and the figure is regenerated headlessly via `nbconvert`.

## I.4 Review

Every change becomes a **pull request**, never an automatic merge. Low-confidence or unresolved-convention extractions are flagged in the PR title. The key reviewer aid is a **highlighted plot**: the full constraint figure is rendered with all existing limits greyed out and only the new proposed curve in red, so a domain expert can sanity-check the proposal visually in seconds without reading code.

## I.5 Cost model

The pipeline runs on a frontier model (Claude Opus 4.8) to maximise extraction quality. Per paper it makes roughly **5–9 model calls**: 3 read-vote passes of Stage 1, plus up to 3 vision passes when gating triggers, plus relevance/codegen calls. Text-only papers (the majority) avoid the expensive image tokens entirely. Per-paper cost is on the order of **tens of cents to a few dollars**, dominated by vision papers. This is not a cheap pipeline, but it is far below the expert time it substitutes for, and text-first gating bounds the spend. (For the full benchmark, parallelising the independent extractions across a thread pool cut a ~9 h sequential run to ~50 min.)

---

# Part II — Evaluation methodology and results (expanding paper §3)

## II.1 The benchmark

We evaluate on **346 papers** spanning 14 coupling types, chosen as every curated repository data file that carries a known arXiv ID. The **ground truth** for each paper is the *repository's own digitised curve* — never a curve digitised by our own model.

This choice avoids **circularity** (we are not grading the model against itself) but has an important consequence: the reference was itself digitised and rescaled from the same paper by the human maintainer, so **a perfect extraction still shows a nonzero residual** equal to the upstream digitisation/convention gap. Every residual we report is therefore an **upper bound** on the true extraction error. The size of that gap is measured independently in §II.5 (~0.03 dex for table/text sources).

## II.2 The primary metric: log–log interpolation residual

A curve is scored only against a ground-truth curve of the **same coupling type**. The residual is computed as follows (`evaluation/metrics.py`):

1. **Strip sentinels** from both curves (coupling ≥ $10^{-2}$, the polygon-closing fill vertices).
2. Build a log–log interpolation $\hat g(x)$ from the extracted points: `interp1d` over $\log_{10}(\text{mass}) \to \log_{10}(\text{coupling})$.
3. Evaluate $\hat g$ at each ground-truth mass $x_i$ **inside the extracted mass range** (no extrapolation).
4. **Residual** at each point: $\;r_i = |\log_{10}\hat g(x_i) - \log_{10} g_{\text{truth}}(x_i)|\;$ (in dex).
5. When several extracted points share a mass, keep the strongest constraint (lowest coupling).

Two distinct quantities come out of this, and conflating them hides failures:

- **Median residual** — how *accurate* the curve is where it overlaps the truth (the headline coupling-value error).
- **Interpolation coverage** — the fraction of ground-truth masses that fall inside the extracted mass range, i.e. how much of the curve the extraction actually *spans*. A curve can be accurate where it overlaps yet cover the wrong mass window; coverage catches that.

A **reverse pass** (interpolating the ground truth onto the *extracted* masses) flags curves that run *past* the true mass range — a large forward-vs-reverse gap means the extraction's extent disagrees with the truth.

## II.3 Honest disposition taxonomy

A major methodological point: **not every paper is a measurement of extraction quality.** Many are simply not comparable, and lumping them into the residual statistics would either flatter or punish the system unfairly. Each paper gets exactly one disposition:

| Disposition | N | Meaning | Counts toward residual? |
|---|---|---|---|
| **compared** | 271 | scored against a same-coupling GT curve | **yes** |
| no_comparable_gt | 30 | predicted coupling has no GT curve in the pool — usually a misclassification | no (counts against classification) |
| convention_mismatch | 10 | same coupling, but GT uses an incompatible parameterisation | no (a units gap, not extraction error) |
| gt_point_reference | 14 | GT is a single operating-point limit, not a curve | no (nothing to interpolate) |
| gt_unusable | 2 | GT curve has <2 usable points after sentinel stripping | no |
| no_extracted_points | 16 | pipeline returned a type but no data points | no |
| no_prediction | 3 | pipeline returned no coupling type | no |
| extraction_failed | 0 | download/parse/API error (the 5 old-ID failures are now fixed) | no |

Of 346 papers, **271 are comparable**; of those, **243 achieve mass-range overlap** and carry a residual. The residual statistics below are over those.

## II.4 Classification labels

Two kinds of labels are scored:

- **Coupling type** — graded against the repository's directory structure (ground truth by construction). **90.5% accurate** (N=346).
- **Property labels** — *is new limit* (vs. a recast of old data), *is projection* (a future-sensitivity forecast vs. a measured bound), and *data source* (text/table/figure). These have no repository ground truth, so they are graded against an **independent LLM labeler** — a *different* model and prompt whose only job is to classify paper properties. This is a fair cross-model check, not self-agreement. A 15-paper human audit found labeler↔human agreement of 15/15 (is_new_limit), 15/15 (is_projection), 14/15 (data_source). Scores: **is_new_limit 88.9%, is_projection 94.4%, data_source 61.1%** (N=36 labeled).

## II.5 Headline results

> **Provenance (2026-07-02).** These numbers are the post-full346 **benchmark repair** re-baseline (PRs #650/#652/#653): the *same* cached extraction snapshots (2026-06-08, Haiku) rescored after fixing the grading itself — documented GT exclusions (15 papers whose GT could grade no extraction), per-entry GT data files, header-declared unit ingestion, both-sides convention canonicalization, reverse-pass/single-point scoring, and closed-contour envelopes. The failure analysis behind each change: `evaluation/eval_runs/failure_analysis_full346.md` (57% of the failure tail was the benchmark's own fault). Extraction behavior is unchanged; these are the same extractions graded fairly.

| Aggregate metric | Value |
|---|---|
| Coupling-type accuracy | **90.9%** (N=331 scoreable; 15 papers GT-excluded with documented reasons) |
| Median coupling residual (overlap papers) | **0.245 dex** (IQR 0.094–0.562) |
| Mean residual (outlier-sensitive) | 0.723 dex |
| Within a factor of 2 (≤0.3 dex) | **55.1%** |
| Within a factor of 3 (≤0.5 dex) | 69.4% |
| Mean interpolation coverage | **83.7%** |
| Zero-overlap papers | 10 / 277 (**3.6%**) |
| Reverse-pass median residual | 0.252 dex (forward 0.245) |
| Reverse-pass mean coverage | 73.9% (forward 83.7%) |

Reading: the typical extracted curve is **within a factor of two for over half of all curves**, lands at a median ~0.25 dex (better than a factor of two), and covers ~84% of the true mass range. The reverse pass tracking the forward pass means the curves cover the *right extent*, not just the right values where they happen to overlap.

### Two yardsticks for interpreting the residual

- **Noise floor (0.32 dex).** Re-running the *same* extraction repeatedly gives slightly different curves; the 90th-percentile per-paper standard deviation of the median residual is 0.32 dex. A change below this is run-to-run LLM variance, not signal. The headline median (0.331) sits essentially *at* this floor — consensus voting is what keeps it there.
- **Digitisation floor (~0.03 dex).** Independently re-digitising a paper and comparing to the repository (table/text sources) gives ~0.034 dex. This is the irreducible "perfect extraction" residual from §II.1. Because it is far below the observed residuals, **what we report is real extractor error, not a yardstick artefact.**

### Breakdown by extraction source — the central finding

| Source | Papers | Compared | Zero-overlap | Median residual | ≤0.3 dex |
|---|---|---|---|---|---|
| Text | 164 | 153 | 1 | **0.243 dex** | 56.9% |
| Figure (vision) | 156 | 120 | 8 | **0.268 dex** | 51.6% |
| Table | 5 | 4 | 1 | 0.124 dex† | 93.0% |

†The table row is **N = 4** — small but no longer pathological: the old 4.7-dex value was one scalar-nucleon paper hitting the fifth-force-$\alpha$ convention gap, which the repaired benchmark now correctly excludes as a units mismatch rather than scoring as table-reading error.

**Figure-vision (0.268 dex) is on par with text (0.243 dex).** Reading dense log–log plots — historically *the* hard part of this task and the dominant failure mode in earlier iterations — is no longer the limiting factor. The combination of read-vote consensus, quality-tier selection, and convention canonicalization closed the gap.

### Breakdown by coupling type — where the difficulty actually lives

| Coupling type | N | Median residual | 95% CI | Note |
|---|---|---|---|---|
| AxionPhoton | 136 | **0.175** | [0.123, 0.218] | the dominant, well-covered type |
| DarkPhoton | 60 | 0.338 | [0.192, 0.406] | |
| AxionElectron | 19 | 0.235 | [0.090, 0.567] | |
| AxionNeutron | 14 | 0.467 | [0.220, 1.096] | convention-heavy ($2m_N$; per-file registry fixed a +0.27 dex GT-side bug) |
| VectorBL | 11 | 0.905 | [0.551, 2.940] | rare; tail = projection-vs-existing wrong-curve traces |
| AxionMass | 8 | 0.555 | [0.200, 0.897] | |
| ScalarPhoton | 8 | 0.486 | [0.274, 1.082] | |
| AxionProton | 3 | 0.045 | [0.015, 0.151] | small-sample |
| MonopoleDipole | 3 | 3.805 | [0.331, 4.905] | small-sample; poor vision traces |
| ScalarElectron | 3 | 0.176 | [0.127, 3.745] | small-sample |
| ScalarBaryon | 1 | 5.482 | — | single paper (wrong-curve vision trace) |
| AxionEDM | 1 | 6.311 | — | single paper (LLM e·cm arithmetic; deterministic-constant fix pending re-extraction) |

- **Micro-average** (per paper, 267 papers): **0.245 dex** — dominated by AxionPhoton, the most common type.
- **Macro-average** (equal weight per type, 12 types): **1.582 dex**.
- **Macro − micro gap: +1.337 dex.** The gap *rose* under the repaired benchmark — an honesty effect, not a regression: rare-type failures that were previously invisible (∞ residuals, unit-incommensurable comparisons) are now finite, visible numbers. It is concentrated: the three worst types have N ≤ 3, so ~4 known-fault papers control most of the gap (macro over the N ≥ 5 types is 0.45 dex). Per-paper causes and the fix pipeline are tracked in issue #658.

### Breakdown by difficulty

| Difficulty | Papers | Coupling acc. | Median residual | ≤0.3 dex |
|---|---|---|---|---|
| easy | 12 | 100.0% | 0.045 | 85.9% |
| medium | 278 | 90.6% | 0.245 | 55.0% |
| hard | 41 | 90.2% | 0.301 | 47.5% |

### Confidence calibration — mostly a scoring artifact, largely resolved

A paper is counted "accurate" only if **median residual < 0.32 dex AND coverage ≥ 50%** (both value and span correct; 0.32 dex is the run-to-run noise floor). Binning papers by the model's self-reported extraction confidence:

| Confidence bin | N | Mean confidence | Actual accuracy | Gap |
|---|---|---|---|---|
| [0.2–0.4) | 12 | 30.8% | 25.0% | +0.06 |
| [0.4–0.6) | 137 | 54.4% | 37.2% | +0.17 |
| [0.6–0.8) | 51 | 72.3% | 52.9% | +0.19 |
| [0.8–1.0) | 77 | 83.2% | 83.1% | **+0.00** |

**The historically-reported "large, systematic overconfidence" (+0.69 top-bin gap in earlier drafts of this report) was primarily a benchmark-scoring artifact**: correct extractions were being graded as failures (convention gaps, single-point auto-fails, GT mis-mappings), which deflated "accuracy" while confidence was in fact tracking real quality. Under the repaired benchmark the top bin is calibrated to within 0.1 pp, and PLACEHOLDER_NO

---

# Part III — Lessons learned (expanding paper §4)

**1. Treat extraction as a noisy measurement and vote.** A single read of a log–log plot is unreliable and irreproducible. Reading each paper three times and reconciling by a *medoid* consensus with source-quality tiering was the single change that moved figure-vision from the dominant failure mode to parity with text. The medoid (a real central sample, never an average) matters: averaging incompatible curves would fabricate a bound no experiment reported.

**2. The worst errors are deterministic physics, not perception — so solve them with verified derivation, not prompting.** The largest residuals were never about pixels. They were unit/convention mismatches — the axion–nucleon $2m_N$ factor, the scalar $d_e$-vs-$\sqrt\alpha$ 10–15 dex gap, dark-photon $\varepsilon$ vs $\varepsilon^2$, sentinel values mistaken for data — that are *fixed*, not stochastic, and depend on per-file repository conventions absent from the paper. No prompt fixes them. The high-leverage fix was a separate, auditable physics layer: a GPD-derived, code-verified, citation-audited per-file convention registry, with a `[CONVENTION REVIEW]` flag for anything outside the verified set. The general principle for automated scientific extraction: it couples a **stochastic perception problem** (reading plots → handle by sampling and voting) with a **deterministic domain-knowledge problem** (conventions → handle by verified derivation), and the two demand different tools.

**3. Score honestly.** Grounding truth in the upstream repository (never the model's own digitisation) avoids circularity and makes the residual a defensible upper bound. Separating non-comparability (wrong-coupling, point-reference, convention-mismatch) from genuine error keeps the headline meaningful. Scoring property labels against an independent LLM labeler restores those metrics as a fair cross-model test.

**4. Cheap engineering that paid off.** Text-first extraction with conditional vision keeps most papers off the expensive vision path; AST-based insertion produces syntactically valid methods every time; highlighted plots make review accessible to non-coders.

---

# Part IV — Remaining problems the evaluation exposes

The aggregate numbers are good, but the breakdowns point to specific, named, mostly *deterministic* failure modes. In rough priority order:

## IV.1 Rare coupling types are the real frontier (macro 1.115 ≫ micro 0.331 dex)

The single most important signal in the report is the **+0.784 dex macro−micro gap**. The system is strong on common, well-sampled couplings (AxionPhoton 0.21, DarkPhoton 0.41 dex) and weak on rare ones (AxionNeutron 0.77, VectorBL 0.91, AxionMass 0.94, AxionEDM 4.6 dex). These types share three compounding handicaps: small samples (so a single bad paper dominates the median and inflates the confidence interval — note ScalarPhoton's CI reaches 17.9 dex), idiosyncratic conventions, and a higher misclassification rate that sends them to the wrong reference (IV.4). **Improvements should be targeted per-coupling, not chased on the AxionPhoton-dominated headline**, which is already at the noise floor and cannot move much.

## IV.2 Convention gaps that are not yet auto-converted

Convention canonicalization (§I.2.3) handles the *verified* families, but several remain open and show up directly in the numbers:

- **AxionEDM operator vs. response (4.649 dex).** The operator coupling $g_{a\gamma\gamma\gamma}$ is flat in mass, while the experimentally reported *response* $C_G/(f_a m_a)\propto 1/m_a$ folds in the dark-matter field amplitude. This is **not a constant rescale**, so the registry correctly flags it `UNCONVERTIBLE` rather than mis-converting — but it is not yet handled, and it is the largest per-type residual.
- **Scalar fifth-force-$\alpha$ files** (the table-source 4.668 dex outlier, and ScalarPhoton's wide CI). Some scalar files store a fifth-force strength $\alpha$ that converts to $d_e$ via $Q^{-1}\sqrt\alpha$ only inside the notebook; two such files (CsCav, Holometer) could not be text-verified at the file level and are classified by magnitude rather than by a vetted per-file rule. This is the residual long tail of the convention work.
- **convention_mismatch (10 papers).** These are *correctly identified* as a same-coupling/incompatible-parameterisation gap and excluded from the residual — but excluding is not solving. Each is a candidate for a new registry entry.

The pattern: the convention layer has turned silent multi-dex errors into either correct comparisons or honest flags, but the flagged set is real future work, concentrated in scalars and EDM.

## IV.3 Mass-window failures: the 10.3% zero-overlap tail

28 of 271 comparable papers extract a curve that **misses the ground-truth mass range entirely** (coverage 0). These split into recognisable causes: **too few points** (1–2 extracted points cannot span the range), a constant-factor **unit offset** on the mass axis (a clean conversion slip, e.g. µeV vs eV, GHz vs eV), and a **wrong window** (the model traced the wrong panel or the wrong curve in a multi-curve figure). Mean coverage is 76.2% and the reverse pass (69.8%) is somewhat lower, indicating a secondary tendency for some curves to run *past* the true range. These are largely mechanical and addressable with axis-sanity checks against the paper's stated frequency/mass band.

## IV.4 Coupling-type misclassification (~1 in 10 proposals targets the wrong plot)

At 90.5% coupling accuracy, roughly one proposal in ten is compared against — and in production would be inserted into — the *wrong* coupling's plot. The errors are systematically concentrated in **rare and easily-confused types**, e.g. VectorBL → DarkPhoton (both are vector couplings), scalar sub-types confused with each other or with ScalarBaryon, and AxionMass/AxionEDM/AxionCPV confusions. This compounds with IV.1: a misclassified rare-type paper becomes a `no_comparable_gt` (30 papers) or is scored against the wrong reference. It is the strongest single argument that **human-in-the-loop review is mandatory, not optional**.

## IV.5 Confidence is informative but not sufficient to gate review

Earlier versions of this report claimed a +0.67–0.69 top-bin confidence gap ("the model does not know when it is wrong"). The post-full346 benchmark repair showed that claim was **mostly an artifact of unfair grading**: with the same extractions scored correctly, the top bin is calibrated to within 0.1 pp (83.2% claimed vs 83.1% strict accuracy) and accuracy is monotone in confidence (§II.5). What survives of the original concern: mid-confidence bins are still moderately overconfident (+0.17–0.19), and top-bin accuracy of 83% means roughly one in six high-confidence extractions is still wrong — so confidence can *prioritize* review (it is a genuine ranking signal) but cannot *replace* it. Agreement-based confidence (read-vote dispersion, already computed) remains a promising complement to the self-reported scalar.

## IV.6 Smaller, structural issues

- **Multi-coupling papers lose secondary results.** A paper reporting bounds on several couplings yields one extraction; secondary curves are dropped. This shows up as `no_comparable_gt` when the extracted (single) coupling differs from the GT entry being scored.
- **Single-point / projection references (14 `gt_point_reference`).** Operating-point limits and projections are not curves; they need a single-point comparison mode rather than interpolation, currently excluded.
- **Benchmark vs. extractor faults are ~50/50.** A hand audit of scored failures in an earlier run found roughly half were genuine extractor faults and half were ground-truth/benchmark issues (wrong convention in a GT entry, a single-point reference miscoded as a curve). The disposition taxonomy keeps the latter out of the residuals, but the benchmark itself needs ongoing hygiene.
- **Run-specific caveat.** This run injected a ground-truth-title fallback (empty abstract) for 128 newer papers because the arXiv metadata API was rate-limiting; the title+abstract pre-classifier was mildly degraded for those, though Stage 1 still reads the full PDF text. A clean re-warm of abstracts would remove this asterisk.

## IV.7 Governance (the non-technical blocker)

The pipeline is deployed and has generated limit proposals; **none have merged**, because no review process for AI-generated scientific data exists. The technical accuracy is now good enough that the binding constraint is social: who is authorised to vet and accept a machine-proposed limit into authoritative community infrastructure. Two recommendations: a community editorial board analogous to PDG reviewers, where collaborations validate their own limits; and, more fundamentally, **publishing machine-readable limit data alongside papers**, which would bypass the figure-extraction problem entirely — the single highest-leverage change the field could make.

---

## Appendix — one-line summary of the numbers

346 papers · 277 comparable · 267 with overlap (15 GT-excluded with documented reasons). Coupling-type accuracy **90.9%**. Median residual **0.245 dex** (factor-2 for **55.1%**, factor-3 for 69.4%), coverage **83.7%**, zero-overlap **3.6%**. Figure-vision **0.268 dex** ≈ text **0.243 dex**. Micro **0.245** vs macro **1.582 dex** (the gap is concentrated in three N ≤ 3 types — ~4 known-fault papers; macro over N ≥ 5 types is 0.45 dex). Noise floor 0.32 dex; digitisation floor 0.03 dex. Top-bin confidence calibrated (+0.00 gap; earlier +0.68 claim was a scoring artifact); mid bins +0.17–0.19. Largest open problems: wrong-curve vision traces on rare couplings, the declaration-contract mislabels (extractor-side fixes merged, re-extraction validation pending), the rare-coupling sample sizes (#658), and review governance.
