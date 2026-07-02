# Journal Submission Plan — JINST

*Target venue:* **Journal of Instrumentation (JINST)**, IOP + SISSA.
*Decision history:* CSBS was the original target; CSBS relaunched as *EPJ Research Infrastructures* (Jan 2026) with a broadened infrastructure/data/governance scope; user chose JINST (2026-06-19).
*Grounding rule:* every claim/number in the manuscript must trace to a repo artifact (`evaluation/report.md`, `paper/TECHNICAL_REPORT.md`, `evaluation/eval_runs/*`, code) or external literature. No invented content.

## Scope-fit caveat (honest)
JINST scope = "detector physics, accelerator science and associated experimental methods and techniques, theory, modelling and simulations" ([about](https://iopscience.iop.org/journal/1748-0221/page/about-the-journal)). An LLM data-curation pipeline is an *associated-methods* fit, not central. Mitigations: (a) consider the **Technical Report** article type; (b) pick keywords from JINST's controlled list (candidates: "Analysis and statistical methods", "Software architectures (event data models, frameworks and databases)", "Data processing methods"); (c) cover letter framing as community experimental-data infrastructure. Some desk-reject scope risk remains.

## JINST format requirements (from the JINST author manual + SISSA instructions)
- LaTeX: use **`jinstpub.sty`** (class `JINST.cls` is obsolete); TeXLive 2023; Overleaf "JINST" template. *(Not NeurIPS, not Springer.)*
- Front matter: full title, full author names + affiliations + corresponding-author email, abstract that **fits on the first page**, **2–4 keywords from the JINST list**. Line numbering recommended.
- References: **sequential numerical in square brackets**, BibTeX `JHEP.bst`. Cite data/code in external repos with **DOIs**.
- **Data/Software/Code Availability Statement required** at end; FAIR repositories (Zenodo / institutional) encouraged.
- **AI/LLM use must be declared in detail** in Methods or Acknowledgments.
- No hard length limit; conciseness encouraged. No page charges; gold OA optional (CC BY, APC).

---

## Overall status (2026-06-19)
Full first draft executed. Deliverables in `paper/`: **`jinst_autoaxionlimits.tex`** (complete JINST manuscript), **`refs.bib`** (26 verified citations, 0 hallucinated), **`make_journal_figures.py`** + **`figures/{residual_cdf,per_type_residual,confidence_calibration}.pdf`** (generated from `metrics.json`). Validated: compiles to 15 pp via a local shim (real build = JINST Overleaf template), BibTeX clean, **all 21 in-text citations resolve**, no missing figures. Remaining human/long-lead items: confirm JINST submit button + final keyword list; mint the **Zenodo DOI** (Step 8); optional fresh pipeline re-run if newer numbers wanted.

## Step status

### Step 0 — Verify headline numbers — ⚠️ REDO AGAINST THE REPAIRED BENCHMARK (2026-07-02)
The 2026-06-19 reconciliation (346/271/243; median 0.331; factor-2 48.4%; macro 1.115) is **superseded**: the post-full346 benchmark repair (#650/#652/#653, merged 2026-07-02) rescored the same extraction snapshots with fair grading. New source of truth: `evaluation/report.md` (regenerated) + `paper/TECHNICAL_REPORT.md` §II.5 (updated): **346/277/267 papers (15 GT-excluded, documented); coupling acc 90.9%; median 0.245 dex; factor-2 55.1%; factor-3 69.4%; coverage 83.7%; zero-overlap 3.6%; text 0.243 (N=153) vs figure-vision 0.268 (N=120); macro 1.582 vs micro 0.245 (gap concentrated in three N≤3 types — see #658)**. Every number in `jinst_autoaxionlimits.tex` must be re-reconciled (Step 4).

✅ **The 2026-06-19 calibration discrepancy is RESOLVED — in the opposite direction than assumed.** The +0.69 overconfidence was primarily a *scoring artifact* (correct extractions graded as failures). Repaired benchmark: top bin 83.2% conf → 83.1% accurate, gap **+0.00**; accuracy monotone in confidence; mid bins +0.17–0.19. TECHNICAL_REPORT §II.5 + IV.5 rewritten accordingly (2026-07-02). Step 7's governance argument must switch from "confidence is unusable" to "confidence ranks but cannot replace review". Submitted PAI paper unaffected (no calibration table in it).

⏳ **Optional final re-baseline:** the extractor-side fixes (#655/#657) merged un-validated (API key pending, #648). If the Haiku re-extraction + Phase-4A re-baseline runs before submission, regenerate all numbers once more; otherwise the 0.245-dex re-scored baseline above is the citable, reproducible state.

### Step 1 — Reformat to JINST template — TODO
`jinstpub.sty`, line numbers, abstract-on-first-page, keywords from JINST list, decide Technical Report vs research paper. Keep authors + ORCIDs (review is not double-blind).

### Step 2 — Expand architecture (Part I) — TODO
Source: `TECHNICAL_REPORT.md` Part I (Discovery, two-stage extraction, medoid consensus `pipeline/read_vote.py`, quality-tier tuple `transform_guard.py`, convention canon I.2.3, integration, review, cost). Keep TikZ architecture figure.

### Step 3 — Convention-canonicalization centerpiece (physicist-facing core) — TODO
Source: `TECHNICAL_REPORT.md:68–84` table + `evaluation/conventions.py` + `GPD/explanations/coupling-convention-conversions-EXPLAIN.md` (+ `-CITATION-AUDIT.md`). **Re-verify each factor against current `PlotFuncs.py`** (memory notes are point-in-time). Present per-file registry, sentinel rule, `[CONVENTION REVIEW]`/`UNCONVERTIBLE` escalation.

### Step 4 — Full evaluation (Part II) — TODO
Source: `TECHNICAL_REPORT.md` Part II + `report.md`. Benchmark/circularity, metric + reverse pass, disposition taxonomy (346→271→243), classification + human audit, per-type table w/ 95% CIs, source breakdown. **Calibration table must be regenerated (see Step 0 discrepancy).**

### Step 5 — Reliability/ablation (isolation methodology) — TODO
Sources: `evaluation/eval_runs/comparison_*.md` (P0→roadmap chain), `failure_analysis.md` (46-paper, 93% extractor faults), determinism runs (`repeat_voted/`, `repeat1/`, `after_561fix_repeats/`). **Use per-paper attribution / monotonicity / same-snapshot canon-on-vs-off, NOT naive subset medians** (memory: subset headlines are N=3-drift-confounded). Report noise floor ~0.32 dex (PR #545) + digitization floor ~0.034 dex (PR #558). Frame baseline→after_roadmap (median 0.530→0.391, ZO 15→9, cov 56.6→72.0%) as cumulative.

### Step 6 — Related Work (genuinely new) — TODO
Run **gpd-bibliographer** agent. Real cites only: LLM/document extraction from scientific PDFs, figure digitization (WebPlotDigitizer etc.), AI4Science automation, existing physics anchors. JHEP.bst.

### Step 7 — Limitations + Governance — TODO
Source: `TECHNICAL_REPORT.md` Part IV (rare-coupling frontier, open convention gaps, zero-overlap tail, misclassification, **overconfidence — REWRITE per Step 0**, structural issues, governance/PDG-board/FAIR-data).

### Step 8 — Availability + AI disclosure (JINST) — TODO (long-lead)
**Archive code + 346-paper benchmark on Zenodo → DOI** (GitHub link alone not FAIR-persistent). Add Data/Software/Code Availability Statement. Expand AI footnote (`pai26_autoaxionlimits.tex:41`) into detailed Methods/Acknowledgments declaration.

### Step 9 — Figures — TODO
Keep highlighted plot (`figures/DarkPhoton_highlighted.png`). Add residual CDF/histogram, per-coupling-type bar chart, 2–3 overlay-validation plots (pipeline vs upstream) from real eval data.
