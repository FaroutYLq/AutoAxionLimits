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
0.245-era reconciliation (JOURNAL_PLAN Step 0 is superseded again; Steps 2
and 5 are also stale — Step 2 centers the retired medoid consensus, Step 5
prescribes the old-code 0.32 dex noise-floor framing — update pointers in
the docs pass).

**Editorial constraints:**
- Do NOT cite counts/volumes of pipeline-generated PRs. The deployment story
  is told through the hand-picked case studies (§4.8) re-extracted with the
  release version, plus the human-in-the-loop design description.
- Do not cite the pre-fix Haiku headline (0.604/0.975) without its confound
  framing; the retraction/confound story is a *methodology contribution*
  (§3.3), not a result.
- The old draft's "none have merged, as governance … is unsolved" abstract
  line is dropped: it undersells the deployment and edges toward the
  PR-volume constraint. Keep the human-in-the-loop governance point in §5/§7.

**Write-time warnings (from the 2026-07-04 skeleton audit):**
- `jinst_autoaxionlimits.tex` is **0.331-era** (older than assumed) and its
  narrative is *inverted*: consensus voting is presented as a headline
  contribution in its abstract, while this paper retires voting as an
  anti-pattern (§5.2). Write §3–§5 fresh from this skeleton; quarry the old
  draft only for §1–§2 prose, the TikZ figure, related work, and `refs.bib`.
- **Pin the numbers before prose**: a small checked-in script must emit every
  §4 number from the two `final2_*/metrics.json` files (catastrophic counts
  with a fixed residual definition, calibration bins, per-channel table,
  classification accuracy). The report's catastrophic counts (18/47, 41/89)
  do NOT reproduce under the naive forward-interpolation definition
  (19/55, 44/95) — the paper cites whatever the pinned script produces.
- `make_journal_figures.py` is hardcoded to the OLD baseline
  (`evaluation/results/metrics.json`) and draws a single-arm CDF — it needs
  parameterized input, a two-arm F2, and calibration re-pointed at
  `final2_opus_n1`. F4 (8-dex anatomy) has no generation path yet — locate
  the 1508.02463 before/after probe data and write a bespoke script;
  treat F4 as the highest-risk figure.
- The AI-use declaration must name the manuscript-*drafting* model(s) in
  addition to the pipeline models (JINST wants the detailed declaration).

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
  NOTE: state the catastrophic denominator explicitly — 18/346 processed
  ≈ 5.2% but 18/266 compared ≈ 6.8%; pick one and use it consistently.
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
  model-digitized data (circularity rule). 346 papers / ~433 channel-entries
  (re-verify: `papers.json` has 434 top-level entries — one may be excluded);
  per-entry header-declared unit conversions; documented, reversible
  exclusions used only when the GT itself cannot grade any extraction
  (`evaluation/ground_truth/EXCLUSIONS.md`). State the GT's own error floor:
  the curator files are themselves digitized, with a measured digitization
  floor ~0.034 dex (PR #558) — residuals below ~0.03 dex are unresolvable by
  construction. This preempts the "why trust GT?" referee objection.
- 3.2 **Metrics.** Log-space residuals by interpolation over shared mass
  support (forward + reverse); micro-median (per-paper) and macro-median
  (per-coupling-type — the tail-sensitive statistic); catastrophic (>3 dex)
  and >1 dex counts; papers-compared/coverage; channel mixes. Convention-gap
  papers are excluded, not scored raw (scoring a units mismatch as extraction
  error is the larger sin). Define "papers compared" precisely: it is
  `n_finite` (comparison_status == compared AND finite forward residual;
  Opus 270 compared − 4 zero-overlap = 266 finite), not `n_compared`.
- 3.3 **Confound control** (methodology contribution — tell it honestly):
  - The model×code confound incident: an early full-pool comparison silently
    changed model and code together; it was retracted, and a same-code
    model-isolation design plus a fixed-model code A/B replaced it.
  - The definitive design: both models on identical code, identical 346
    papers, identical transport, identical N — paired per-paper comparison.
  - The measured run-to-run noise floor (~0.02–0.07 dex on aggregate medians;
    channel-routing flips as the dominant source) and what it means for
    single-run precision claims (report the first decimal, not the third).
    DISAMBIGUATION: this is NOT the 0.32 dex figure in TECHNICAL_REPORT/
    JOURNAL_PLAN Step 5 — that is the 90th-percentile *per-paper* repeat
    spread, measured on OLD code at N=3. Use only the aggregate floor for
    delta claims; do not cite 0.32 dex near the 0.2646 headline (invites the
    misreading "the result sits at the noise floor"), and re-measure the
    per-paper spread on current code if it is used at all.
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

  Catastrophic/>1-dex counts: use the pinned-script definition (see
  write-time warnings — the report's 18/47 and 41/89 don't reproduce under
  the naive forward-interpolation definition, which gives 19/55 and 44/95).
  Channel mixes — pick ONE definition for T3 and say which: the report's
  winning-`data_source`-per-snapshot mix (text 141 / vision 156 / …) differs
  from metrics.json's scoring-time `source_breakdown` (139/154/…).
  Classification accuracy IS populated in `final2_opus_n1/metrics.json`
  (contrary to the definitive report's caveat — presumably a later rescore):
  coupling-type 90.9% (331 papers), is_new_limit 85.7% (35), is_projection
  97.1% (35) — cite it here. Guard in-vivo activity (19 corroboration
  rejections, 20 convention-review flags in the Opus arm).
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
  0.055 dex; `source_survey.md`); vector_trace: oracle-level rescues (24
  papers <0.3 dex, oracle selection; `vector_ceiling.md`); per-channel
  residual table from the definitive Opus arm. CAVEAT: the scorer's
  `source_breakdown` only has text (0.276) / figure_vision (0.292) /
  table (0.13) rows — vector_trace and source_data must be computed custom
  over `per_paper`, or the claims stay attributed to the channel-study
  artifacts above. Text 0.276 vs vision 0.292 in the definitive arm is
  itself a datum for "vision is no longer the bottleneck".
- 4.7 **Cost & calibration.** Per-paper cost by model and N; operational
  guidance (production on Opus — at daily volume the 5× price delta is
  negligible against reviewer time; bulk evaluation on Haiku). Confidence
  calibration: compute from `final2_opus_n1/metrics.json` like every other
  §4 number — the TECHNICAL_REPORT §II.5 "top-bin gap ~0.000" claim is from
  the 0.245-era master rescore (old code, N=3) and does NOT transfer: the
  definitive arm's top bin is 0.840 mean confidence vs 0.723 accuracy
  (+0.12 overconfident; mid bins 0.53→0.32, 0.72→0.47). What survives:
  accuracy is *monotone* in confidence — it ranks extractions but does not
  replace review; the historically-reported +0.69 gap was a scoring
  artifact. State the accuracy definition (resid <0.32 dex & coverage ≥50%).
  Claiming ~0.000 here would be internally inconsistent with the published
  artifacts a referee can recompute from.
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
| F2 | residual CDF, both arms | `make_journal_figures.py` on final2 metrics (script rework needed: currently hardcoded to old baseline, single-arm) |
| F3 | per-coupling-type residual | same, Opus arm |
| F4 | 8-dex anatomy (plotted plane vs canonical; before/after contract) | §4.5 case data — NO generation path yet; bespoke script + locate 1508.02463 probe data; highest-risk figure |
| F5+ | case-study highlighted plots | re-extraction showcases |
| F? | confidence calibration | `final2_opus_n1/metrics.json` (NOT the old baseline — see §4.7) |
| T1 | channel/tier table | §2.2 — verify tier values against `pipeline/transform_guard.py` |
| T2 | conversion-registry families | `evaluation/conventions.py` + GPD docs |
| T3 | definitive two-arm results | `final2_definitive_report.md`, counts re-pinned by script |
| T4 | paired model comparison | same |
| T5 | lessons (helped/didn't, one-line evidence each) | §5 |

## Number-grounding map (re-verify at write time)

| claim | value | artifact | verified 2026-07-04 |
|---|---|---|---|
| definitive Opus micro/macro | 0.2646 / 1.156 dex | `final2_opus_n1/metrics.json` | ✅ |
| definitive Haiku micro/macro | 0.6605 / 4.947 dex | `final2_haiku_n1/metrics.json` | ✅ |
| papers compared (finite) | 266 / 231 | `interpolation_aggregate.n_finite` per arm | ✅ |
| paired model gap | +0.220 dex (206 papers) | `final2_definitive_report.md` | report only |
| old-code baseline | 0.2454 / 1.5817 dex | `evaluation/results/metrics.json` | ✅ |
| code A/B fixed-model | −0.016 dex, +4 coverage | `final_full346_haiku_report.md` | report only |
| N-probe | +0.003 dex | `nprobe_and_routing_memo.md` | report only |
| catastrophic / >1 dex counts | PIN BY SCRIPT | report says 18/47 & 41/89; naive recompute gives 19/55 & 44/95 | ⚠️ definition unpinned |
| 8-dex anatomy fix | 8.5 → 0.22/0.27 dex | PR #684 body + probe logs | not re-verified |
| guard activity | 19 + 20 firings | `final2_definitive_report.md` | report only |
| noise floor (aggregate) | ~0.02–0.07 dex | routing memo + digest | report only |
| caching saving | 20–36% | PR #668 | not re-verified |
| calibration (definitive arm) | top bin 0.840 conf → 0.723 acc; monotone | `final2_opus_n1/metrics.json` `confidence_calibration` | ✅ (supersedes "~0.000", which was the 0.245-era rescore) |
| classification accuracy | coupling 90.9% (331); new-limit 85.7%; projection 97.1% | `final2_opus_n1/metrics.json` `classification` | ✅ |
| per-channel (definitive arm) | text 0.276 / vision 0.292 / table 0.13 | `final2_opus_n1/metrics.json` `source_breakdown` | ✅ |
| source_data survey | 0.055 dex median | `evaluation/eval_runs/source_survey.md` | ✅ |
| vector_trace ceiling | 24 papers <0.3 dex (oracle) | `evaluation/eval_runs/vector_ceiling.md` | ✅ |
| GT digitization floor | ~0.034 dex | PR #558 | not re-verified |
| GT pool size | 346 papers / ~433 entries | `evaluation/ground_truth/papers.json` | ⚠️ file has 434 entries — reconcile |
| channel tier values | 5 / 4 / 3.5 / 3 / 2 | `pipeline/transform_guard.py` | not re-verified |
