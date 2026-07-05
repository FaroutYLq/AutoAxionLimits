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

## Overall status (2026-07-04)
**Manuscript rewritten from `paper/JINST_SKELETON_v2.md` on the definitive benchmark** (PR #692). Deliverables in `paper/`: **`jinst_autoaxionlimits.tex`** (skeleton-v2 rewrite), **`refs.bib`** (26 verified citations), **`make_paper_numbers.py`** → `numbers.json`/`numbers.md` (single source of truth for every cited number, pinned definitions), **`make_journal_figures.py`** + `figures/` (regenerated from the final2 two-arm metrics). Validated: compiles to 14 pp via a local shim (real build = JINST Overleaf template), all 24 citations resolve, no overfull boxes. Remaining: case studies (§5.9 stub, pending showcase re-extractions), optional F4 anatomy figure, title decision, Zenodo DOI (Step 8), keyword confirmation.

**Content authority order:** `JINST_SKELETON_v2.md` (structure + grounding map) > `numbers.json` (values) > this plan (format/venue rules only). Steps below are kept for the format rules and long-lead items; their content instructions are superseded where they conflict with the skeleton.

## Step status

### Step 0 — Verify headline numbers — ✅ SUPERSEDED BY THE PINNED-NUMBERS SCRIPT (2026-07-04)
The 0.245-era reconciliation below is **superseded** by the definitive two-arm benchmark (master ad8ecc1a, both models on the fixed pipeline, N=1, 346 fresh papers each; `evaluation/eval_runs/final2_definitive_report.md`): **Opus 0.2646 dex micro / 1.156 macro, 266 compared, 19 catastrophic; Haiku 0.6605 / 4.947, 231, 44; paired model gap +0.223 dex over 217 papers (139/38/40)**. Every number in the manuscript is now emitted by **`paper/make_paper_numbers.py`** with definitions pinned in code — re-run it rather than reconciling by hand. Old reference points (still citable as history): old-code N=3 baseline 0.245/1.582 (`evaluation/results/metrics.json`); 77-paper code A/B −0.016 dex.

✅ **Calibration:** the "+0.00 top-bin gap" was a property of the 2026-07-02 rescored snapshot and does **not** transfer: the definitive Opus arm shows top bin 0.840 conf → 0.723 acc (+0.12), monotone. The manuscript computes calibration from `final2_opus_n1` and claims "ranks but cannot replace review" (TECHNICAL_REPORT §II.5 carries the same does-not-transfer note).

### Step 1 — Reformat to JINST template — ✅ DONE (2026-07-04)
`jinstpub.sty`, line numbers, abstract-on-first-page, keywords from JINST list, decide Technical Report vs research paper. Keep authors + ORCIDs (review is not double-blind).

### Step 2 — Expand architecture (Part I) — ✅ DONE (2026-07-04, per skeleton v2)
Written around the **current** architecture: five channels with trust tiers, validity-first selection, deterministic guards (#683), convention contracts (#684), N=1. Medoid consensus appears only as a retired lesson (+0.003 dex probe) — do NOT restore it as architecture. Source: `TECHNICAL_REPORT.md` Part I (updated 2026-07-04). TikZ architecture figure updated (layout polish still welcome).

### Step 3 — Convention-canonicalization centerpiece (physicist-facing core) — ✅ DONE (2026-07-04; incl. #683/#684 contracts + foreign-quantity screen)
Source: `TECHNICAL_REPORT.md:68–84` table + `evaluation/conventions.py` + `GPD/explanations/coupling-convention-conversions-EXPLAIN.md` (+ `-CITATION-AUDIT.md`). **Re-verify each factor against current `PlotFuncs.py`** (memory notes are point-in-time). Present per-file registry, sentinel rule, `[CONVENTION REVIEW]`/`UNCONVERTIBLE` escalation.

### Step 4 — Full evaluation (Part II) — ✅ DONE (2026-07-04, per skeleton §3–§4; calibration from final2_opus_n1)
Source: `TECHNICAL_REPORT.md` Part II + `report.md`. Benchmark/circularity, metric + reverse pass, disposition taxonomy (346→271→243), classification + human audit, per-type table w/ 95% CIs, source breakdown. **Calibration table must be regenerated (see Step 0 discrepancy).**

### Step 5 — Reliability/ablation (isolation methodology) — ✅ SUPERSEDED by skeleton §3.3/§4 (2026-07-04)
The manuscript's confound-control section now tells this story with the definitive artifacts: the silent-model incident + retraction, the paired two-arm design, the **aggregate noise floor ~0.02–0.07 dex** (routing memo), and the N-probe. ⚠️ The old **0.32 dex** figure here is the *per-paper* repeat spread measured on OLD code at N=3 — do not cite it near the 0.2646 headline (it invites "the result sits at the noise floor"); the two floors are different quantities. Digitization floor ~0.034 dex (PR #558) is cited in §GT. The P0→roadmap cumulative chain is development history, not manuscript content.

### Step 6 — Related Work (genuinely new) — ✅ DONE (carried from prior draft; 26 verified refs; optional bibliographer refresh before submission)
Run **gpd-bibliographer** agent. Real cites only: LLM/document extraction from scientific PDFs, figure digitization (WebPlotDigitizer etc.), AI4Science automation, existing physics anchors. JHEP.bst.

### Step 7 — Limitations + Governance — ✅ DONE (2026-07-04; "ranks but cannot replace review" framing)
Source: `TECHNICAL_REPORT.md` Part IV (rare-coupling frontier, open convention gaps, zero-overlap tail, misclassification, **overconfidence — REWRITE per Step 0**, structural issues, governance/PDG-board/FAIR-data).

### Step 8 — Availability + AI disclosure (JINST) — TODO (long-lead)
**Archive code + 346-paper benchmark on Zenodo → DOI** (GitHub link alone not FAIR-persistent). Add Data/Software/Code Availability Statement. Expand AI footnote (`pai26_autoaxionlimits.tex:41`) into detailed Methods/Acknowledgments declaration.

### Step 9 — Figures — ✅ regenerated from final2 (two-arm CDF, per-type, calibration); PENDING: case-study highlighted plots (showcases), optional F4 anatomy figure
Keep highlighted plot (`figures/DarkPhoton_highlighted.png`). Add residual CDF/histogram, per-coupling-type bar chart, 2–3 overlay-validation plots (pipeline vs upstream) from real eval data.
