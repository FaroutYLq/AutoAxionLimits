# Projection-vs-limit confusion audit (limits-only pool)

**Question.** Now that the pipeline scope is limits-only (#698, `e9a1c463`), is
there still projection-vs-limit confusion in the current evaluation pool — and
would a "reject projections" extraction directive measurably help a re-benchmark?

**Answer.** No measurable win. The GT-side projection problem is already fully
removed by #698; the residual extraction-side confusion is possible in ~8 of 328
limit papers and is the *plausible cause* of the residual in **at most 2** — below
the 0.32 dex headline noise floor. **Do not re-benchmark for this.** A limits-only
directive is still worth shipping as forward-looking production-scope alignment
(daily/weekly genuinely only want limits), but **without** claiming a benchmark delta.

Run audited: `final2_opus_n1` (Opus 4.8, N=1, master `ad8ecc1a`), limits-only
metrics (`metrics_noproj.json`).

## Two levels of "confusion" — keep them separate

1. **GT-side (paper's GT *is* a projection).** Already handled by #698. Switching
   to `metrics_noproj` dropped exactly 4 papers out of the catastrophic tail
   (23→19 on Opus), all `is_projection=True` under `Projections/`:
   2102.11740 (LZ), 1606.07001 (DARWIN), 2007.04899 (OptomechanicalMembranes),
   2209.12901 (PolarisationHaloscope). `is_projection` classification accuracy is
   0.97 (34/35) — the model reliably knows *which papers* are forecasts.

2. **Extraction-side (limit paper, but the extractor traces a projection curve
   shown in the same figure).** This is the only thing left, and it is a
   figure-level curve-selection error, not a paper-level classification error.
   The current extractor does **not** forbid it: Stage-1 (`extractor.py:507`)
   extracts a measured limit *or* a projection; Stage-2 vision (`extractor.py:620`)
   just traces "the LOWER boundary … on any limit plot", with no projection guard
   (only the vector-channel selector at `:1558` treats projections as distractors).

## Method

Filtered the 328-paper limits-only pool to curve-traced (`figure_vision` /
`vector_trace`) papers with median residual > 1 dex (29). Free local pymupdf
caption scan eliminated 12 with **zero** projection language (their residual is
unit/convention/anchor/wrong-panel — e.g. 1008.3536 correctly shows `no-proj`;
its error is tracing the *existing-exclusion envelope*, a different distractor a
projection directive would not fix). The remaining 17 were read semantically
(abstract + figure captions + results) by four parallel agents.

## Result (17 semantically scanned)

**Tier 0 — projection confusion impossible (11):** 2006.09721 (mathematical
"projection onto S1-S2 plane", not a forecast), 2410.02218 (text-only mention),
2412.03655 (theory benchmarks only), 2407.10618 (signal-preferred contours),
2302.09096 (dashed = other papers' limits), 1606.07494 (lattice-QCD theory, no
exclusion curve), 1401.6460 / 2403.03004 / 2104.12772 (projection in a *separate*
panel/figure, not the result plot) — plus the 12 zero-language papers pre-filtered.

**Tier 1 — projection present, but cannot explain the residual (directive won't
move the score):**

| arXiv | resid (dex) | coupling | why not the cause |
|---|---|---|---|
| 2301.06560 | 13.5 | AxionPhoton | miss is the lifetime/decay-rate axis convention (Γ→g), not tracing |
| 2207.11968 | 6.5 | AxionElectron | only the "−1σ expected limit" (~0.1–0.3 dex from observed) |
| 1503.06886 | 1.9 | ScalarPhoton | "expected limit" thin line, close to observed — digitization noise |
| 1902.04644 | 1.5 | AxionNeutron | CASPEr Phase II projection is ~10⁵ (5 dex) away; 1.5 dex ⇒ traced the *right* curve |
| 2008.08773 | 1.2 | ScalarPhoton | projected curve present but residual only 1.2 dex (no mistrace) |

**Tier 2 — projection present AND plausibly the cause (the actionable set):**

| arXiv | resid (dex) | coupling | evidence | caveat |
|---|---|---|---|---|
| 1708.08464 | 10.5 | AxionMass | "can be probed by indirect measurement" future-reach curves on the f_a–m_a plane alongside the Earth/Sun excluded regions | — |
| 1709.00009 | ∞ | AxionPhoton | Belle II projected reach overlaid on the beam-dump limit | older diagnosis tags the ∞ as a #561 anchor-snap regression, so cause is contested |

## Sizing verdict

Best case, a limits-only extraction directive recovers **1–2 papers** (1708.08464;
1709.00009 confounded). Everything else in the tail is unit / convention / anchor /
wrong-panel and untouched by a projection guard. 1–2 papers is below the noise
floor (0.32 dex) — it will not register on the micro-median in a re-benchmark.

**Recommendation.** Ship the limits-only directive (if desired) as production-scope
alignment, not a benchmarked improvement. Do not re-run the 346-paper benchmark for
this effect. The separate `1008.3536`-style "existing-exclusion envelope" mistrace
is a *different* problem (generalize the vector-channel "trace only THIS paper's
curve" selector to all channels) and is not addressed by rejecting projections.
