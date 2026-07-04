# JINST paper skeleton v2 (2026-07-04) — supersedes the 0.245-era draft structure

Working title: *AutoAxionLimits: an LLM-agent pipeline with deterministic
guards for automated curation of axion and dark-photon exclusion limits*
(keep/adjust the existing `jinst_autoaxionlimits.tex` title if preferred).

Article type: **JINST Technical Report**. Format per `JOURNAL_PLAN.md`
(jinstpub.sty, numbered refs/JHEP.bst, abstract on first page, 2–4 keywords,
line numbers, Data/Code Availability statement, **detailed AI/LLM-use
declaration** — mandatory).

**Grounding rule (unchanged):** every number below traces to a repo artifact;
the artifact path is given inline. Do not carry any number from the old draft
without re-verifying — the 2026-07-04 definitive benchmark supersedes the
0.245-era reconciliation (JOURNAL_PLAN Step 0 is superseded again).

**Editorial constraints:**
- Do NOT cite counts/volumes of pipeline-generated PRs. The deployment story
  is told through the hand-picked case studies (§4.8) re-extracted with the
  release version, plus the human-in-the-loop design description.
- Do not cite the pre-fix Haiku headline (0.604/0.975) without its confound
  framing; the retraction/confound story is a *methodology contribution*
  (§3.3), not a result.

---

## 0. Front matter

- **Abstract skeleton** (one paragraph, first page): problem (community limit
  compilations are hand-curated; results published as figures) → system
  (multi-channel LLM extraction with deterministic guards + single-owner
  convention canonicalization + human-in-the-loop PRs) → evaluation (346-paper
  benchmark against curator ground truth; confound-controlled) → headline
  (median residual 0.26 dex, ~5% catastrophic, at one LLM read per paper;
  model-tier gap quantified at +0.22 dex median) → lessons (keep the LLM to
  perception; keep arithmetic, conversion, and selection deterministic).
- Keywords (JINST list): "Analysis and statistical methods"; "Data processing
  methods"; "Software architectures (event data models, frameworks and
  databases)".
- AI-use declaration: models used (claude-opus-4-8, claude-haiku-4-5) for
  extraction/review agents AND for manuscript drafting assistance; human
  review of all scientific content.

## 1. Introduction

- 1.1 The curation problem: `cajohare/AxionLimits` as the community reference;
  limits published as plotted exclusion curves in heterogeneous conventions;
  manual digitization is slow, error-prone, and unbounded in scope.
- 1.2 Why this is an instrumentation/methods problem: reproducible pipelines,
  quantified accuracy, auditable provenance — not "LLM demo".
- 1.3 Contributions (bulleted, each pointing to a section):
  1. A multi-channel extraction architecture with validity-first candidate
     selection and deterministic guard rails (§2).
  2. A convention-canonicalization system: truthful-declaration + plotted-
     values contracts, a single-owner vetted conversion registry, fail-closed
     escalation to human review (§2.4).
  3. A 346-paper benchmark against curator ground truth with confound-
     controlled methodology and a measured run-to-run noise floor (§3).
  4. Quantified design lessons: what helped, what did not — including
     consensus-voting and model-tier economics (§5).
  5. Open-source, human-in-the-loop deployment on the live literature (§2.5,
     case studies §4.8).
- 1.4 Related work: HEPData/data preservation; plot digitization tools
  (WebPlotDigitizer etc.); LLM information extraction; agentic pipelines.

## 2. System overview (the workflow)

*Figure 1: architecture diagram (TikZ exists in the old draft — update to add
the guard/corroboration boxes and the convention registry).*

- 2.1 **Discovery & triage.** Daily arXiv monitor; weekly preprint-version
  checker (re-extract on changed results; flag published-versions that yield
  no data); historical backfill via INSPIRE-HEP (citation-count filter,
  keyword classifier, LLM relevance check). State persistence in git.
- 2.2 **Extraction agent.** Stage structure: pre-classifier → stage-1
  text/table read → stage-2a axis identification (cheap) → stage-2 vision
  trace. Candidate channels with semantic trust tiers (table):

  | channel | tier | nature |
  |---|---|---|
  | source_data | 5 | deterministic numeric files from the paper's own e-print (pgfplots/.dat/ancillary) |
  | table | 4 | LLM table read |
  | vector_trace | 3.5 | deterministic vector-path geometry from figure PDFs; curve identity by one cheap LLM call |
  | text | 3 | LLM prose read (stated bounds) |
  | figure_vision | 2 | LLM plot trace (rendered pages) |

  Validity-first selector (`pipeline/transform_guard.py` quality tuple):
  in-valid-ranges → non-degenerate → recoverable → convention-known → tier →
  corroborated → confidence → point count LAST (sparse point-limit demotion
  and its corroboration counterweight, §2.3).
- 2.3 **Deterministic guards.** Wrong-curve gates A–D (reject/demote on the
  model's own notes, abstract-stated mass windows, regime checks);
  **text–vision corroboration gate** (vision rejected when >2 dex from an
  in-range text anchor over shared mass support); range validation with
  snap-suppression on non-canonical declarations. Design principle: guards
  are pure functions over model outputs — no extra API calls, fully testable.
- 2.4 **Convention canonicalization (physicist-facing core).**
  - The problem: the same physics is published as g, g², g²/4π, g²/ħc, 1/f_a,
    Λ, ε², decay rates, lifetimes… Cross-convention comparison creates
    multi-decade *apparent* errors that are not extraction errors.
  - Truthful-declaration contract: every channel declares the convention of
    the values it EMITS; **plotted-values contract**: the vision stage never
    converts — it emits raw axis values and declares the plotted quantity;
    axis read-back overrides canonical *claims*.
  - Single-owner conversions: a vetted registry (with citation-audited
    conversion factors) is the only place conversions happen. Foreign-quantity
    screen: declarations naming physics outside a coupling's vetted vocabulary
    fail CLOSED — eval side excludes as a convention gap; runtime flags
    `[CONVENTION REVIEW]` for a human. `UNCONVERTIBLE` semantics.
  - *Table: the conversion registry families with their vetted factors
    (source: `evaluation/conventions.py` + GPD explain/citation-audit docs;
    re-verify against current `PlotFuncs.py`).*
- 2.5 **Integration & human-in-the-loop.** Reviewer agent produces repo
  artifacts: two-column data file, AST-based `PlotFuncs.py` method insertion,
  notebook wiring with automatic axis-window extension, and a highlighted
  plot (new limit in red over greyed compilation). Every output is a pull
  request for expert review; nothing merges automatically. Confidence tiers
  (`[LOW CONFIDENCE]`, `[NEEDS REVIEW]`, `[CONVENTION REVIEW]`) route
  reviewer attention.
- 2.6 **Robustness & operations.** Availability fail-fast (billing/auth
  errors abort the run without marking papers processed — "a property of the
  run, not the paper"); runtime config assertion (the driver asserts the
  resolved model, never trusts env alone); prompt-injection defenses
  (sanitized, delimited paper content); prompt caching (20–36% measured);
  API retry semantics.

## 3. Evaluation methodology

- 3.1 **Ground truth.** Curator repo data files ONLY (`limit_data/`) — never
  model-digitized data (circularity rule). 346 papers / 433 channel-entries;
  per-entry header-declared unit conversions; documented, reversible
  exclusions used only when the GT itself cannot grade any extraction
  (`evaluation/ground_truth/EXCLUSIONS.md`).
- 3.2 **Metrics.** Log-space residuals by interpolation over shared mass
  support (forward + reverse); micro-median (per-paper) and macro-median
  (per-coupling-type — the tail-sensitive statistic); catastrophic (>3 dex)
  and >1 dex counts; papers-compared/coverage; channel mixes. Convention-gap
  papers are excluded, not scored raw (scoring a units mismatch as extraction
  error is the larger sin).
- 3.3 **Confound control** (methodology contribution — tell it honestly):
  - The model×code confound incident: an early full-pool comparison silently
    changed model and code together; it was retracted, and a same-code
    model-isolation design plus a fixed-model code A/B replaced it.
  - The definitive design: both models on identical code, identical 346
    papers, identical transport, identical N — paired per-paper comparison.
  - The measured run-to-run noise floor (~0.02–0.07 dex on aggregate medians;
    channel-routing flips as the dominant source) and what it means for
    single-run precision claims (report the first decimal, not the third).
  - Vote-count probe design (N=1 vs N=3, transport-matched).
  *Artifacts: `evaluation/eval_runs/final_full346_haiku_report.md`,
  `haiku_regression_causal_digest.md`, `nprobe_and_routing_memo.md`.*

## 4. Results

*(All numbers below are the citable current state — artifact:
`evaluation/eval_runs/final2_definitive_report.md` + `final2_*/metrics.json`
unless noted.)*

- 4.1 **Headline (definitive two-arm benchmark, fixed pipeline, N=1, 346
  fresh papers each):**

  | | Opus 4.8 | Haiku 4.5 |
  |---|---|---|
  | micro-median | 0.2646 dex | 0.6605 dex |
  | macro-median | 1.156 dex | 4.947 dex |
  | catastrophic (>3 dex) | 18 | 41 |
  | papers compared | 266 | 231 |

  Channel mixes; guard in-vivo activity (19 corroboration rejections, 20
  convention-review flags in the Opus arm).
  *Figures: residual CDF per arm; per-type residual (regenerate
  `make_journal_figures.py` from the new metrics.json).*
- 4.2 **Code effect at fixed model.** Old-code N=3 baseline 0.245/1.582
  (`evaluation/results/metrics.json`) → fixed-code N=1 0.2646/1.156: headline
  parity within the noise floor at one-third the LLM reads per paper, tail
  improved; grading is now stricter (registry excludes convention gaps the
  old scoring accepted). Supporting: 77-paper paired code A/B at fixed Opus
  (−0.016 dex, +4 coverage; `final_full346_haiku_report.md`).
- 4.3 **Model effect, measured cleanly.** Paired over 206 papers scored in
  both arms: **Haiku − Opus = +0.220 dex median** (Opus better 132, Haiku 36,
  tied 38). Tail and coverage gaps are the real story (41 vs 18 catastrophic;
  231 vs 266 compared). Contrast with the tail-biased earlier estimate
  (+0.95 dex) and explain the shrinkage: sampling + the plane-bookkeeping
  amplifier (→ §4.5). Haiku's own operating point moved 0.975 → 0.6605 from
  code fixes alone.
- 4.4 **Vote-count economics.** Transport-matched probe: N=3 − N=1 =
  +0.003 dex on Opus — consensus voting retired for production (it stabilizes
  the traced curve, not the channel choice, which is where the variance
  lives). *Artifact: `nprobe_and_routing_memo.md`.*
- 4.5 **Anatomy of decade-scale errors** (the paper's most transferable
  finding): vision models read plots to ~0.2 dex; the multi-dex failures were
  units/plane accounting. Case anatomy (1508.02463): double conversion
  (model self-sqrt + registry sqrt) and zero conversion (plane-less axis
  read-back) both produced ~8 dex on correct readings; the plotted-values
  contract fixed it end-to-end (8.5 → 0.22–0.27 dex, reproduced twice).
  Error-mass concentration: vision-routed papers held 60% of total residual
  mass pre-fix.
- 4.6 **Channel value.** source_data: exact where present (survey median
  ~0.055 dex); vector_trace: oracle-level rescues (24 papers <0.3 dex in the
  channel study); per-channel residual table from the definitive Opus arm.
- 4.7 **Cost & calibration.** Per-paper cost by model and N; operational
  guidance (production on Opus — at daily volume the 5× price delta is
  negligible against reviewer time; bulk evaluation on Haiku). Confidence
  calibration: repaired benchmark shows top-bin confidence ≈ accuracy (gap
  ~0.000) — confidence *ranks* extractions but does not replace review
  (TECHNICAL_REPORT §II.5/IV.5, 2026-07-02 rewrite).
- 4.8 **Case studies (SHOWCASES — placeholders until re-extraction).**
  Hand-picked papers re-run end-to-end on the release version; each with the
  highlighted plot and the audit trail (declaration, gates fired, selector
  reason). Suggested selection criteria (one each):
  1. clean text/table extraction (baseline behavior);
  2. non-canonical-axis vision paper (squared coupling — the §4.5 anatomy,
     e.g. 1508.02463);
  3. vector_trace win (exact geometry, LLM identity selection);
  4. wrong-curve gate rejection with graceful fallback;
  5. `[CONVENTION REVIEW]` escalation (unvetted convention, fail-closed).

## 5. Lessons learned (Discussion)

- 5.1 **What helped** (each with its measured evidence):
  - validity-first selection over point-count routing;
  - deterministic channels (source_data, vector_trace) wherever determinism
    is available;
  - deterministic guards over prompt exhortations (pure-function gates on
    model outputs; 19+20 firings in one 346-paper run);
  - single-owner conversions + truthful-declaration/plotted-values contracts
    (the 8-dex anatomy);
  - fail-closed escalation on unknown conventions (human review beats
    guessing);
  - cross-channel numeric corroboration (text anchor vs vision trace);
  - human-in-the-loop PR gate as the safety backstop for everything above;
  - availability fail-fast + runtime config assertion;
  - confound-controlled evaluation with a measured noise floor;
  - prompt caching.
- 5.2 **What did not help / anti-patterns** (equally valuable):
  - consensus voting on a strong model (+0.003 dex for 3× cost);
  - asking the model to convert units ("helpful conversion" is poison —
    it creates double/zero conversion downstream);
  - substring/keyword matching of free-text convention declarations
    (false convertibles silently suppress review flags);
  - Message-Batches transport for heterogeneous agentic loops with large
    vision payloads (single-dispatcher serialization; fine for homogeneous
    small-payload jobs);
  - trusting environment configuration without runtime assertion (the
    silent-model incident, framed as a config-assertion lesson);
  - single-run benchmark deltas below the noise floor.
- 5.3 **Cross-cutting principles:** keep the LLM to perception and semantics;
  keep arithmetic, unit conversion, and selection deterministic and testable;
  make every model claim auditable (declarations, notes, gate flags);
  measure the noise floor before claiming improvements.

## 6. Limitations and future work

- Channel-routing stability (stage-1 point-count variance) — corroboration
  mitigates, does not remove.
- OCR-calibration fallback for figures with outlined/vector text.
- Text-value corroboration generalization (abstract-stated anchors).
- Registry growth loop: offline vetted convention escalation (queue → expert/
  assistant vetting → registry release) rather than runtime guessing.
- Multi-run distributional benchmarks (2–3 repeats) given the noise floor.
- Population of classification metrics (coupling-type/new-limit accuracy) in
  the current scoring config.
- Per-stage model tiering (cheap pre-classification, strong tracing).
- Upstream adoption loop with the curator community.

## 7. Conclusions

- The pipeline curates real literature at ~0.26 dex median accuracy with ~5%
  catastrophic rate at one LLM read per paper, every output human-reviewed.
- The decisive engineering was NOT the model: it was deterministic guards,
  convention ownership, and confound-controlled measurement.

## Back matter

- Data/Software Availability: repo + Zenodo DOI (mint before submission —
  JOURNAL_PLAN Step 8); benchmark artifacts paths.
- AI/LLM use declaration (detailed, per JINST policy).

---

## Figure & table plan

| # | item | source |
|---|---|---|
| F1 | architecture diagram (update guards/registry) | old draft TikZ |
| F2 | residual CDF, both arms | `make_journal_figures.py` on final2 metrics |
| F3 | per-coupling-type residual | same |
| F4 | 8-dex anatomy (plotted plane vs canonical; before/after contract) | §4.5 case data |
| F5+ | case-study highlighted plots | re-extraction showcases |
| T1 | channel/tier table | §2.2 |
| T2 | conversion-registry families | `evaluation/conventions.py` + GPD docs |
| T3 | definitive two-arm results | `final2_definitive_report.md` |
| T4 | paired model comparison | same |
| T5 | lessons (helped/didn't, one-line evidence each) | §5 |

## Number-grounding map (re-verify at write time)

| claim | value | artifact |
|---|---|---|
| definitive Opus micro/macro | 0.2646 / 1.156 dex | `final2_opus_n1/metrics.json` |
| definitive Haiku micro/macro | 0.6605 / 4.947 dex | `final2_haiku_n1/metrics.json` |
| paired model gap | +0.220 dex (206 papers) | `final2_definitive_report.md` |
| old-code baseline | 0.245 / 1.582 dex | `evaluation/results/metrics.json` |
| code A/B fixed-model | −0.016 dex, +4 coverage | `final_full346_haiku_report.md` |
| N-probe | +0.003 dex | `nprobe_and_routing_memo.md` |
| 8-dex anatomy fix | 8.5 → 0.22/0.27 dex | PR #684 body + probe logs |
| guard activity | 19 + 20 firings | `final2_definitive_report.md` |
| noise floor | ~0.02–0.07 dex | routing memo + digest |
| caching saving | 20–36% | PR #668 |
| calibration top-bin gap | ~0.000 | TECHNICAL_REPORT §II.5 (2026-07-02) |
