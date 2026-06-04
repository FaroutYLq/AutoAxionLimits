# Extraction Pipeline Failure-Mode Analysis

*Lead-analyst report over 46 per-paper failure diagnoses (of an 82-paper benchmark). Each diagnosis was produced by an independent agent that compared our extraction against verified ground truth.*

---

## 1. Executive Summary

Of the **82-paper benchmark, 46 papers (56%) fail or materially underperform.** This report dissects all 46.

**Category split (46 failing papers):**

| Category | Count | Share of failures |
|---|---:|---:|
| HIGH_RESID (extracted curve overlaps in mass but coupling is off by ≥1 dex) | 20 | 43% |
| ZERO_OVERLAP (no usable overlap with GT curve) | 16 | 35% |
| NO_POINTS (extractor emitted nothing) | 9 | 20% |
| NO_PREDICTION (recast/prediction not recognized) | 1 | 2% |

**Fault attribution:** 43 of 46 (93%) are genuine **extractor faults**; only **3 (7%) are inherent to the source/benchmark** and not the extractor's fault (`1707.07921`, `2111.09892`, `2401.17253`).

**Fixability:** `easy` 1, `moderate` 27, `hard` 6, `needs_chart_model` 9, `unextractable` 3.

### Dominant root-cause themes

The 46 failures collapse into **four** recurring mechanisms that account for almost everything:

1. **Source misrouting toward text / single points (the #1 theme, ~17 papers).** The selector repeatedly chose a single text-quoted headline value, a single tabulated benchmark, or "text mode → give up" over tracing the figure curve that GT was built from. This drives nearly all ZERO_OVERLAP-by-too-few-points and most NO_POINTS cases. The extracted single point is frequently *physically correct* (`1806.00310`, `2110.03679`, `1207.3275`) — it just cannot overlap a 15–129-point GT curve.

2. **Vision-trace inaccuracy (coupling-axis drift / undertracing, ~7 papers).** When figure_vision *was* used, it read the log-y axis 1–7 decades off, or traced only the flat top of a descending curve. This is the bulk of the `needs_chart_model` bucket.

3. **Coupling-type / convention / observable confusion (~11 papers).** Multi-channel and multi-panel papers (CROWS, MICROSCOPE, DAMNED, DarkSide, XENON1T) led the extractor to the wrong physical quantity (chi vs g, d_e vs d_me, alpha vs d_e, EDM amplitude vs g_an), or to the right quantity in the wrong unit convention (eV⁻¹ vs dimensionless, d_e_large vs small-Lambda). These produce the multi-decade HIGH_RESID outliers (13–19 dex).

4. **Truncated-PDF / no-vision-fallback (~3 papers) and representation mismatches (exclusion-box vs line, mass-independent line collapsed to one mass, ~4 papers).** A handful of NO_POINTS cases are simply the results section being cut off the input text with no fallback; a handful of ZERO_OVERLAP cases are benchmark encoding artifacts (box ceiling vertices, placeholder mass) where the physics was actually correct.

**Headline takeaway:** the failure rate is *not* benchmark noise. Only 3/46 (~3.7% of the 82) are genuinely unextractable. The remaining 43 are real opportunity, and roughly **half are recoverable without a better chart model** — purely by fixing source routing (prefer figure_vision/curve over single text points), adding a vision fallback on truncated text, and normalizing conventions/box-encodings in the comparator.

---

## 2. Symptom Taxonomy

Grouped into families (raw symptom strings collapsed where they describe the same mechanism).

### A. Single-point / single-benchmark instead of full curve — 9 papers
*One physically-plausible value extracted (from text or a single table row) where GT is a multi-point curve; no overlap possible.* **All extractor fault.**

| id | source of the lone point | GT size |
|---|---|---|
| 1207.3275 | text annotation (dip minimum) | 11-pt curve |
| 1310.8098 | text point, also wrong coupling type | 150-pt curve |
| 1806.00310 | text best-limit (correct value) | 15-pt polygon |
| 1907.11485 | single table benchmark row | 80-pt curve |
| 2110.03679 | text benchmark (correct value) | 129-pt curve |
| 2211.12699 | text inline value (correct scale) | 80-pt curve |
| 2305.01002 | two text benchmark points | 41-pt curve |
| 2311.16364 | text best-sensitivity (+mass typo) | 123-pt curve |
| 1506.08082 | text flat plateau, missed turn-up | 13-pt table |

### B. NO_POINTS — figure-only curve, vision never attempted — 4 papers
*Extractor identified the correct figure and coupling type but stayed text-only and emitted 0 points.* **All extractor fault, mostly `needs_chart_model`.**
`1110.2895`, `1606.07001`, `1810.04602`, plus `2006.09721` (also picked wrong interpretation).

### C. NO_POINTS — truncated PDF, no vision fallback — 3 papers
*Results section cut off the input text; no fallback to vision on the results page.* **Extractor fault, `moderate`.**
`2007.04990`, `2112.12116`, `2208.07293`.

### D. NO_POINTS / NO_PREDICTION — prediction/recast not emitted — 4 papers
*Paper gives a mass-window prediction or a recastable bound, not a (mass,coupling) curve; GT stores a synthetic point.* Mixed fault.
- Extractor fault: `1606.03145` (lattice f_a prediction not converted), `2112.03439` (radio-line g_aγγ recast rejected).
- Not extractor fault / unextractable: `2401.17253` (string-sim mass window; GT is a degenerate placeholder).

### E. Vision coupling-axis drift / undertracing — 7 papers
*figure_vision used, mass axis fine, coupling read 1–7 dex off or only the curve's flat top traced.* **All extractor fault, mostly `needs_chart_model`.**
`1403.1290` (3.2 dex, traced only flat top), `1509.00476` (~2 dex low), `1604.06800` (coarse 4-pt eyeball), `1905.13650` (~6.8 dex axis misread), `2007.04899` (admitted hand-estimate, ~3 dex low), `2403.03004` (~1 dex high, traced noise body not floor), `2410.10363` (~2 dex low, multi-coupling figure).

### F. Coupling-type / panel / observable confusion — 8 papers
*Wrong physical quantity selected from a multi-channel/multi-panel paper.* **All extractor fault.**
`1310.8098` (chi vs g), `1708.06367` (EDM amplitude vs g_an), `1709.00009` (SN1987A envelope vs BeamDump wedge), `1712.00483` (alpha vs d_e), `1508.01798` (EP envelope vs DUAL dip), `2006.07055` (d_me vs d_e), `2006.09721` (AxionElectron vs DarkPhoton), `2207.11968` (g_Ae vs chi), `2401.18076` (ScalarPhoton vs ScalarElectron panel).

### G. Convention / unit conversion offset — 6 papers
*Right quantity, wrong unit system or a botched conversion factor; flat multi-decade coupling offset.* **All extractor fault.**
`1410.7267` (alpha→coupling never applied, ~16–18 dex), `1902.04246` (Ce/Fa→g_ae headline, ~6 dex), `2004.02733` (1e6 conversion blunder, exactly −6 dex), `2204.01454` (divided by m_a, ~13 dex), `2209.13588` (eV⁻¹ vs dimensionless, non-constant 3–9 dex), `2410.19902` (f_a_norm vs GeV⁻¹ + spurious ×10).

### H. Mass-axis unit / scale errors — 3 papers
*Coupling fine; mass axis collapsed or unconverted.* **Extractor fault.**
`2008.05355` (spurious Hz→eV auto-correct, −14.4 dex, **`easy`**), `1607.06083` (GeV→eV never applied, ~12 dex), `2311.16364` (single-point mass typo, in family A too).

### I. Representation / encoding mismatch (physics correct) — 4 papers
*Extraction physically right; mismatch is box-vs-line or placeholder-mass artifact.* Mixed.
- Extractor-fault but benchmark-encoding-driven: `2102.08764` (box bottom edge vs ceiling vertices), `2007.03694` (mass-independent line collapsed to m=1 eV).
- Not extractor fault / unextractable: `2111.09892` (hand-built NS-cooling box; spurious Fig.3 scrape filtered to 0), `1707.07921` (GT auto-selector pulled wrong experiment-era file).

---

## 3. Extractor-Fault vs Unextractable Split

This is the decisive question for whether the 56% failure rate is real opportunity or benchmark noise.

| Bucket | Count | % of 46 | % of 82 |
|---|---:|---:|---:|
| Genuine extractor faults | 43 | 93% | 52% |
| Not extractor fault / unextractable | 3 | 7% | 3.7% |

**The 3 genuinely-not-our-fault cases:**
- `1707.07921` — GT auto-selector matched "PandaX" to the **2024** PandaX file while the paper is the **2017** PandaX-II analysis. Wrong ground truth, not wrong extraction. → prune/fix GT.
- `2111.09892` — GT is a hand-built rectangular NS-cooling box derived from a single scalar bound; the paper has no curve. Extractor correctly noted this but the vision selector scraped an unrelated DFSZ figure (filtered to 0). → prune.
- `2401.17253` — string-simulation mass-window prediction stored as a degenerate synthetic AxionMass point. No (mass,coupling) data exists. → prune.

**Borderline "fault but driven by benchmark encoding" (2 papers):** `2102.08764` and `2007.03694` are scored as extractor faults but the *physics is correct* — the coupling value (`2102.08764`: 2.6e-6 matches GT bottom edge exactly) and the bound value (`2007.03694`: 1.3e-13 exact) are right. The zero-overlap is purely a box-ceiling-vertices vs placeholder-mass comparator artifact. These are **comparator/representation fixes**, not extraction fixes.

**Conclusion:** the 56% is overwhelmingly real opportunity. At most ~3.7% of the benchmark (3 papers) is irreducible noise; another ~2 papers are comparator-encoding artifacts. The other ~41 papers represent genuine recoverable extraction quality.

---

## 4. Analysis by Failure Category

### HIGH_RESID — 20 papers (43%)
Mass windows overlap but the coupling axis is wrong. Three distinct drivers:
- **Convention/unit offsets (6):** the multi-decade tail (`2401.18076` 18.9, `2006.07055` 18.6, `1410.7267` 16.2, `2204.01454` 13.1 dex). These are arithmetic/normalization bugs, not perception — see §5.
- **Wrong panel/coupling type (4–5):** `1508.01798`, `1709.00009`, `2207.11968`, `1708.06367` — 2 dex offsets because a different physical quantity was traced.
- **Vision drift (7):** `1403.1290`, `1509.00476`, `1905.13650`, `2007.04899`, `2403.03004`, `2410.10363`, `1604.06800` — 1–7 dex from misreading the log-y axis or tracing the wrong feature of a noisy/multi-curve plot.

HIGH_RESID is the category where a **chart-extraction model would help most** (the drift subset), but the *largest* errors here are convention bugs that a chart model would not fix.

### ZERO_OVERLAP — 16 papers (35%)
By `zo_cause`:
- **too_few_points (8):** family A above. The extracted point is usually *correct*; the failure is source routing, not perception. Highest-value, lowest-effort fix: stop preferring single text/table points over figure curves.
- **wrong_window (5):** `1506.08082`, `1607.06083`, `1712.00483`, `2305.01002`, `2311.16364` — mass axis unconverted (GeV→eV), wrong observable's mass scale, or a mass typo.
- **unit_offset (2):** `2008.05355` (Hz auto-correct), `2410.19902` (f_a_norm vs GeV⁻¹).
- **encoding artifact (1):** `2102.08764` (+`2007.03694` collapse).

### NO_POINTS — 9 papers (20%)
- Figure-only curve, vision not attempted (4): `1110.2895`, `1606.07001`, `1810.04602`, `2006.09721`.
- Truncated PDF, no fallback (3): `2007.04990`, `2112.12116`, `2208.07293`.
- Prediction not emitted (2): `1606.03145`, plus the unextractable `2401.17253`.

Every text-mode NO_POINTS case explicitly **named the correct figure and numeric anchors** in its notes, then declined to trace. This is a routing/policy failure, not a capability ceiling.

### NO_PREDICTION — 1 paper
`2112.03439` — Breakthrough Listen radio-line search; the standard g_aγγ recast was rejected because the extractor anchored on the decay-rate observable. GT is 2 points at g_aγγ=4.1e-8. Recoverable with a recast-recognition rule.

---

## 5. The Worst Offenders (multi-decade HIGH_RESID)

All four extreme outliers are **convention/quantity bugs, not perception failures** — the mass axis was read correctly in every case.

- **`2401.18076` — 18.9 dex.** GT `ScalarElectron/LIGO.txt` is the **d_e_large** convention (~0.27 up to 1e20). Extractor traced Fig.3 in the paper's small-Lambda convention (~1e-18) **and** picked the wrong panel (ScalarPhoton/Λ_γ instead of ScalarElectron/Λ_e). Two stacked errors: convention + panel.
- **`2006.07055` — 18.6 dex.** DAMNED reports both d_e (ScalarPhoton) and d_me (ScalarElectron). Extractor read the d_me panel in physics convention (~1e-15); GT is d_e in the large-d_e convention (~1e3–1e30). Wrong type + wrong convention.
- **`1410.7267` — 16.2 dex.** Correct figure, correct coupling type (ScalarNucleon), good mass coverage (0.86) — but reported the raw figure y-axis **alpha (relative to gravity)** instead of converting to the repo's dimensionless coupling (IUPUI.txt, ~1e3–1e17). The α→coupling transform was simply never applied; the extractor's notes flagged the ambiguity but did not resolve it.
- **`2204.01454` — 13.1 dex.** Traced the gluon coupling C_G/(f_a·m_a) in GeV⁻² and labeled it AxionEDM; GT is 1/f_a in GeV⁻¹ (AxionMass, f_a_norm). The ~13 dex gap is exactly division by m_a (~1e-24 GeV).

**Honorable mention — `2004.02733` (4.6 dex):** an *exactly* 1e6 conversion blunder (g_an = g·m_n with m_n in the wrong units), confirmed by the ratio 999930 ≈ 1e6. GT was already in GeV⁻¹ so no conversion was needed at all.

**Lesson:** the worst residuals are deterministic, diagnosable unit/convention bugs — precisely the class addressed by the P3 coupling-convention normalizer. They are *not* arguments for a chart model.

---

## 6. Prioritized Recommendations

Ranked by papers recovered per unit effort.

### P1 — Source-routing policy: prefer the figure curve over single text/table points (recovers ~12–15 papers)
The single largest lever. Affects family A (9 ZERO_OVERLAP too-few-points) + family B (4 NO_POINTS figure-only). In nearly all of these the extracted single point is *physically correct* — the pipeline just refused to trace the curve. Change the best-extraction selector so that **a single text/table point never wins over a traceable figure curve** when GT-class evidence (an exclusion contour) is present; force figure_vision when the read-vote finds "no consensus curve, took most-points."
Papers: `1207.3275, 1310.8098, 1806.00310, 1907.11485, 2110.03679, 2211.12699, 2305.01002, 2311.16364, 1506.08082, 1110.2895, 1606.07001, 1810.04602, 2006.09721, 2211.12699`.

### P2 — Vision fallback on truncated / results-missing text (recovers 3, possibly more)
When `data_source=none` because text lacks a results section (`2007.04990, 2112.12116, 2208.07293`), automatically fall back to figure_vision on the results pages. Cheap, deterministic, no model change. Also catches the "named the figure then gave up" NO_POINTS cases.

### P3 — Convention/unit normalization + sanity guards (recovers ~6–8, eliminates the worst residuals)
Extend the coupling-convention normalizer (P3 read-layer) to:
- detect and apply d_e_large vs small-Lambda, eV⁻¹ vs dimensionless, alpha-relative-to-gravity, and C_G/(f_a·m_a)→1/f_a transforms (`2401.18076, 2006.07055, 1410.7267, 2204.01454, 2209.13588, 2410.19902`);
- add a **±1e6 / m_n round-number guard** to catch conversion blunders (`2004.02733`);
- **disable or sanity-bound the Hz→eV auto-correction** when masses are already in GeV/eV range (`2008.05355` — this one is `easy` and a pure regression-guard fix).
Add a GeV→eV check for collider papers whose stated mass is in GeV but extracted mass lands ~12 dex low (`1607.06083`).

### P4 — Comparator/encoding normalization (recovers ~2, removes false negatives)
- Treat exclusion **boxes** as fill regions: compare against the bottom/limit edge, not the ceiling-1.0 vertices (`2102.08764`).
- Expand mass-independent bounds to a horizontal segment over the plot mass range instead of a placeholder single mass (`2007.03694`).
These are scoring fixes, not extraction fixes, but they remove genuine false negatives.

### P5 — Coupling-type/panel disambiguation for multi-channel papers (recovers ~4–5, hard)
For dual-channel and multi-panel papers, route by **matching the GT-expected coupling type / panel** rather than easiest source. Affects `1709.00009, 1508.01798, 1708.06367, 2207.11968, 1712.00483`. Harder because it needs the extractor to enumerate candidate channels and pick deliberately; partly mitigated once P1 forces curve tracing.

### Is a chart-extraction model warranted?
**Partially, and second-order.** A dedicated chart model would directly recover the 9 `needs_chart_model` vision-drift papers (`1110.2895, 1403.1290, 1509.00476, 1606.07001, 1709.00009, 1810.04602, 1905.13650, 2305.01002, 2410.10363`) — ~20% of failures. But note:
- Several of those (`1110.2895, 1606.07001, 1810.04602`) are recovered *first* by P1/P2 (the issue is "vision not attempted," not "vision inaccurate") — they only need a model if tracing-once-invoked is still poor.
- The **largest** errors (13–19 dex) are convention bugs a chart model would not touch.
So: pursue P1–P4 first (cheap, recover ~half the failures, deterministic). A chart model is justified only after, to close the residual ~7–9 true vision-accuracy cases. Do not lead with it.

### Should the benchmark be pruned?
**Yes — prune exactly 3 papers, fix-don't-prune a 4th.**
- Remove `2401.17253` (degenerate mass-window placeholder) and `2111.09892` (hand-built box, no curve in paper) — they are unextractable by construction and only generate noise.
- **Fix, not prune,** `1707.07921`: the GT auto-selector pulled the wrong experiment-era file (2024 PandaX vs the 2017 paper). Repoint GT to the correct file; this is a benchmark data bug, not an unextractable paper.
After pruning the 3 noise papers, the corrected failure denominator is ~43/79 ≈ 54%, of which **~95% is genuinely recoverable.** The benchmark is sound; the failures are real.

---

## Appendix — per-paper findings (all 46)

| arXiv | category | symptom | fault | fixability | root cause / evidence |
|---|---|---|:--:|---|---|
| 1604.06800 | HIGH_RESID | `coarse_eyeballed_figure_curve_misses_shape` | yes | moderate | Extractor routed to text but read only ~4 rough by-eye points off Fig 4 instead of densely tracing the curve. The flat 4-point approximation fails to capture the GT curve's steep low-mass rise and overall shape, yielding ~1.1 dex median residual. — Coupling type and projection status both correct. GT has 155 points ... |
| 1905.13650 | HIGH_RESID | `coupling_axis_scale_misread` | yes | needs_chart_model | The figure_vision trace read the g_aNN exclusion curve at ~1e-10 to 1e-11, but the GT CASPEr_Comagnetometer curve has g_an values of order 1e-3 to 1.0. The vision extractor misread the coupling (y) axis scale/decade calibration by ~7 dex while the mass (x) axis was read correctly, producing strong mass-coverage (0.8... |
| 2209.13588 | HIGH_RESID | `coupling_convention_mismatch_dimensionless_vs_eVinv` | yes | moderate | Extractor read approximate Fig. 2 (gaPP) values and reported the proton coupling as C_N/F_a in eV^-1 (~1e-10), whereas the GT NASDUCK-SERF.txt file is the dimensionless g_ap convention (~1e-6 up to 1.0). The two conventions differ by f_a-dependent factors, so the offset is non-constant and the dimensionless GT curve... |
| 2401.18076 | HIGH_RESID | `coupling_convention_offset_and_panel_confusion` | yes | hard | The GT curve (limit_data/ScalarElectron/LIGO.txt) is stored in a "d_e_large" convention with coupling values ~0.27 up to 1e20, while the extractor traced Fig.3 in the paper's small-coupling Lambda convention (~1e-18 to 5e-20 eV^-1-style values), giving a ~18-20 dex offset on the coupling axis. Compounding this, the ... |
| 2204.01454 | HIGH_RESID | `coupling_quantity_confusion_fa_vs_gluon_per_ma` | yes | moderate | The extractor traced the wrong y-axis quantity: it read the gluon coupling C_G/(f_a·m_a) in GeV^-2 (~1e-29..7e-23) and labeled it AxionEDM, whereas the GT curve is the normalized inverse decay constant 1/f_a in GeV^-1 (coupling_type AxionMass, f_a_norm convention, ~2e-13..1e0). Dividing by m_a (~1e-24 GeV) shifts th... |
| 1708.06367 | HIGH_RESID | `coupling_type_confusion` | yes | moderate | The extractor read the oscillating-EDM amplitude (d_n ~ few x10^-25 e.cm) from Fig.2 as the deliverable and classified it as AxionEDM, when the GT curve is the derived dimensionless axion-neutron coupling g_an (= C_N/f_a recast). It extracted the wrong physical observable on the y-axis: EDM amplitude in e.cm vs dime... |
| 2006.07055 | HIGH_RESID | `coupling_type_confusion_and_convention_mismatch` | yes | moderate | DAMNED reports limits on BOTH d_e (ScalarPhoton) and d_me (ScalarElectron); the extractor picked the ScalarElectron panel and read tiny physics-convention coupling values (~1e-15), while GT is the ScalarPhoton d_e curve in the "large d_e" convention (~1e3-1e30). Wrong coupling type plus wrong convention produce a hu... |
| 2207.11968 | HIGH_RESID | `coupling_type_confusion_wrong_panel` | yes | hard | The paper reports several couplings (g_Ae axioelectric, dark-photon kinetic mixing chi, DM-electron sigma_e, \|Ue4\|^2) across multiple Fig. 2 panels. The vision extractor traced the red g_Ae (AxionElectron) exclusion curve, but the verified GT for this DarkSide entry is the dark-photon kinetic-mixing (chi) limit, s... |
| 2004.02733 | HIGH_RESID | `coupling_unit_conversion_offset` | yes | moderate | The extractor correctly identified the physical exclusion band (2e-5 to 1e-3 GeV^-1 isovector/derivative coupling) but then converted GeV^-1 -> dimensionless g_an by multiplying by neutron mass and botched the factor by exactly 1e6 (used m_n in wrong units / extra 1e-6). The GT file (SNO.txt) is actually stored in G... |
| 2007.04899 | HIGH_RESID | `eyeballed_curve_coupling_offset` | yes | moderate | Extractor self-reported it did NOT trace Fig.2 but instead "approximate readings... estimated from the plotted curves" and routed through data_source=text (selector chose text over figure_vision). These 6 hand-estimated points are systematically too low in coupling and miss the actual curve shape, producing a large ... |
| 2410.10363 | HIGH_RESID | `figure_curve_drift_coupling_offset` | yes | needs_chart_model | Vision-traced a curve from a multi-coupling FASER ALP figure where g_aWW (the paper's primary target) is plotted alongside the secondary g_agamma reinterpretation; the extractor read coupling values ~2 dex too low and a mass window shifted/extended to higher masses, so the traced AxionPhoton curve does not match the... |
| 1509.00476 | HIGH_RESID | `figure_curve_drift_coupling_too_low` | yes | needs_chart_model | The extractor digitized Fig. 4 by eye via figure_vision and read the g_agamma exclusion curve systematically too low on the log y-axis. Mass axis was traced well (good overlap/coverage), but the coupling magnitudes were misread by roughly 2 decades, producing a near-constant downward offset. — Mass window matches GT... |
| 1403.1290 | HIGH_RESID | `figure_curve_drift_undertraced_descent` | yes | needs_chart_model | Vision tracing of the ARIADNE monopole-dipole projection (Fig. 2, \|g_s^N g_p^N\| vs force range) captured only the upper/flat portion of the sensitivity curve and missed its steep descent. The extractor even noted the curve "reach[es] down to ~1e-36 to 1e-39" but its 21 traced points stayed pinned near 1e-29-1e-30,... |
| 2403.03004 | HIGH_RESID | `figure_curve_drift_upward` | yes | moderate | Vision tracing of FIG. 5 captured the noisy PRCL/MICH exclusion band roughly one decade too high. The extractor reported tracing "the lower envelope / floor" of the noisy curves, but in practice landed on the visual body of the noise band rather than its true minimum, so every traced g_BL point sits systematically a... |
| 2212.01139 | HIGH_RESID | `flat_text_value_vs_sloped_figure_curve` | yes | moderate | The best-extraction selector chose the text source (a single rough ~1e-8 sensitivity value quoted in the paper body) over figure_vision, yielding a flat 2-point line at chi=1e-8. The actual STAX projection is a steeply sloped 134-point curve that lives only in Figs 6-7; the extractor's own notes admit the detailed c... |
| 1707.07921 | HIGH_RESID | `ground_truth_source_misrouting_wrong_experiment_era` | no | unextractable | The GT auto-selector matched the experiment name "PandaX" and pulled limit_data/AxionElectron/PandaX.txt, which actually holds the 2024 PandaX DM-axion absorption curve (sourced from arXiv 2408.07641 / 2409.00773), not the 2017 PandaX-II solar/galactic axion-electron limits that paper 1707.07921 reports. The extract... |
| 1902.04246 | HIGH_RESID | `text_headline_flat_plateau_with_coupling_conversion_offset` | yes | moderate | Extractor routed to the text headline (Fa/Ce >= 2e15 eV) instead of digitizing the mass-resolved Fig.5 curve, then converted Ce/Fa to g_ae via g_ae ~ 2*m_e*Ce/Fa and stamped a flat ~5.2e-4 plateau across 4 masses. The real limit is a 62-point curve varying with mass, and the conversion/headline-depth is ~6 dex too s... |
| 1508.01798 | HIGH_RESID | `wrong_curve_in_multicurve_figure` | yes | hard | Fig.1 of this paper overlays many curves (EP/spectroscopy existing bounds, seismology, plus projected resonant-detector sensitivities for Cu-Si sphere, AURIGA, DUAL, quartz). GT wants the specific "DUAL projected scalar-electron sensitivity" — a narrow resonant dip — but the extractor traced the broad horizontal exi... |
| 1709.00009 | HIGH_RESID | `wrong_curve_traced_astro_vs_beamdump` | yes | needs_chart_model | Vision extractor traced the SN 1987A astrophysical lower-boundary envelope (weak-coupling diagonal) in Figure 2, but the ground truth is the BeamDump exclusion wedge (strong-coupling, MeV-GeV mass region). It picked the wrong constraint curve within the right ALP-photon figure, so the coupling values are systematica... |
| 1410.7267 | HIGH_RESID | `yukawa_alpha_not_converted_to_coupling` | yes | hard | The extractor correctly traced Fig. 4 (Yukawa strength \|alpha\| relative to gravity vs lambda) and even correctly identified coupling_type=ScalarNucleon and source=figure_vision, with good mass coverage (0.86). But it reported the raw figure y-axis (alpha, relative-to-gravity, ~0.1 down to 1e-13) instead of convert... |
| 1607.06083 | ZERO_OVERLAP | `axis_scale_misread_gev_to_ev_unconverted` | yes | moderate | The paper is a collider ALP search (mass 5-100 GeV); GT is the LHC_pp table in GeV-scale mass (~1e7-6e9 eV) and g [GeV^-1]. The vision extractor traced Fig.4 but emitted mass values of ~1e-5 to 1e-4 eV and couplings ~1e-4 — it failed to read/convert the log GeV axis to eV (and never applied the GeV->eV factor), plac... |
| 2410.19902 | ZERO_OVERLAP | `coupling_convention_mismatch_plus_bad_vision_calibration` | yes | moderate | GT sample_points are expressed in a normalized "f_a_norm" convention (~7.4e-11) while our extraction reports raw 1/f_a in GeV^-1 (~9e-8) read from Fig. 5; the two live in different unit systems (~3 dex apart), giving zero overlap. The offset is worsened because the vision reader saw the correct floor 1/f_a~9e-9 GeV^... |
| 1310.8098 | ZERO_OVERLAP | `coupling_type_confusion_plus_single_text_point` | yes | moderate | CROWS is a dual-channel LSW paper reporting both an ALP (AxionPhoton g_aγγ) and a hidden-sector-photon (DarkPhoton chi) limit. The extractor's read-vote chose DarkPhoton (3/3) and quoted only the single most-sensitive text point for that channel, while the verified GT is the AxionPhoton g_aγγ exclusion curve (CROWS.... |
| 1712.00483 | ZERO_OVERLAP | `coupling_type_confusion_wrong_observable` | yes | moderate | MICROSCOPE reports multiple distinct observables (Yukawa fifth-force alpha for B / B-L coupling, and the dimensionless dilaton-EM coupling d_e). The extractor routed to ScalarBaryon and quoted the two text summary alpha bounds, but the ground truth is the ScalarPhoton d_e constraint (a different physical quantity on... |
| 2102.08764 | ZERO_OVERLAP | `exclusion_region_not_curve` | yes | moderate | The GT is a closed exclusion rectangle encoded as 4 vertices (two mass endpoints crossed with coupling values {2.6e-6, 1.0}), where 1.0 is just the fill-to-ceiling top edge. Our extractor correctly read the physical limit (g_ae < 2.6e-6 over the magnon mass band) and emitted 3 points all at coupling=2.6e-6, i.e. onl... |
| 2111.09892 | ZERO_OVERLAP | `exclusion_region_not_curve` | no | unextractable | The GT is a hand-built rectangular exclusion box for neutron-star cooling (4 corner points spanning mass 1e-23 to 9e3 eV, coupling 1.3e-9 to 1.9e-4), derived from the paper's single g_an bound — not a curve in the paper. The extractor correctly recognized (in its notes) that the paper gives only single-point bounds ... |
| 2007.03694 | ZERO_OVERLAP | `mass_independent_bound_collapsed_to_single_mass` | yes | moderate | The extractor correctly read the mass-independent stellar red-giant bound (g_ae<1.3e-13) from Table I but, lacking a real mass axis, placed both points at a single nominal placeholder mass of 1.0 eV instead of emitting a horizontal line spanning the plot's full mass range. The GT represents the same constant bound a... |
| 2211.12699 | ZERO_OVERLAP | `single_point_text_read_instead_of_figure_curve` | yes | moderate | The extractor routed to text and pulled the single inline value "g_agamma > 3e-4 GeV^-1 excluded at m_a ~ 0.25 GeV" instead of tracing the full exclusion contour in Fig 6. With only 1 point, there is no curve to overlap the 80-point GT, so coverage collapses to 0. The extractor's own notes admit the Fig 6 curve exis... |
| 1806.00310 | ZERO_OVERLAP | `single_point_text_read_vs_curve` | yes | hard | Extractor took the single best-limit value quoted in the paper's text (g_aee < 4.9e-10 at m_a = 58 ueV) instead of tracing the multi-segment exclusion outline in Fig. 8, which it explicitly judged "not numerically tabulated." With n_ext=1 the metric cannot form any overlap/coverage against the 15-point GT polygon, s... |
| 1207.3275 | ZERO_OVERLAP | `single_point_text_read_vs_full_curve` | yes | moderate | The extractor read only the single text-annotated characteristic exclusion point (m=1.1e-5 eV, chi=7.6e-9 at the operating frequency 2.9565 GHz from the Fig. 2 annotation) instead of tracing the full Fig. 2 exclusion curve. It explicitly noted "full curve shape only available as plot" and the read-vote found "no con... |
| 1907.11485 | ZERO_OVERLAP | `single_table_point_vs_full_figure_curve` | yes | moderate | The selector routed to the `table` source, which contains only one tabulated dark-photon benchmark (m=0.5 keV, kappa=2.2e-16 from Table I). The full 80-point exclusion curve exists only in panel F of Fig. 5, so a 1-point extraction cannot overlap the GT curve (coverage=0, resid=inf). Source misrouting: figure_vision... |
| 2110.03679 | ZERO_OVERLAP | `single_text_point_vs_full_curve` | yes | moderate | Extractor routed to text-only and quoted the single benchmark 95% CL limit (g<=3.76e-11 GeV^-1 for m_a<<1e-11 eV) instead of tracing the figure. The GT (DSNALP.txt) is a 129-point exclusion curve over mass 1e-23 to 1e-8 eV that is flat at ~3.75e-11 then rises to ~1e-9 GeV^-1. One point cannot cover a 129-point curve... |
| 2311.16364 | ZERO_OVERLAP | `single_text_point_vs_full_curve` | yes | moderate | Extractor routed to data_source=text and read only the single best-sensitivity number quoted in prose (min \|g_p\| = 1.2e-7 at one mass), instead of digitizing the full exclusion curve in Fig 5 (GT expected_source=table, 123 points). The lone point also has a self-admitted mass-conversion error. — Extracted n=1 poin... |
| 2008.05355 | ZERO_OVERLAP | `spurious_hz_to_ev_mass_autocorrection` | yes | easy | The extractor correctly read the ATLAS Pb+Pb ALP masses as 6-100 GeV and converted to eV (6 GeV=6e9 eV), and its couplings were right. But a post-processing "auto-correct masses Hz->eV" anchoring rule misidentified the already-correct GeV/eV masses as Hz and multiplied them by 4.136e-15 (h in eV*s), collapsing the e... |
| 2305.01002 | ZERO_OVERLAP | `text_benchmark_points_instead_of_figure_curve` | yes | needs_chart_model | Extractor took the source path of least resistance: it quoted two illustrative ALP benchmark points stated in the text (m_a=200/398 MeV) rather than tracing the 41-point Fig. 4 exclusion boundary. Its own notes admit these benchmarks "may not lie on the exclusion boundary" and are "approximate representatives only,"... |
| 1506.08082 | ZERO_OVERLAP | `text_quoted_flat_segment_instead_of_table_contour` | yes | moderate | The extractor chose the text route and reconstructed OSQAR as a 2-point flat line at the quoted bound g=3.5e-8 GeV^-1 over m<2e-4 eV, ignoring the digitized exclusion-contour table (expected_source=table, 13 points) where the limit degrades steeply (coupling rising 3.5e-8 -> 1.0) between m=2.2e-4 and the 4.98e-4 eV ... |
| 1110.2895 | NO_POINTS | `exclusion_region_declined_no_figure_trace` | yes | needs_chart_model | The extractor stayed in text mode (data_source=none), correctly identified the coupling type (AxionPhoton) but concluded that because the bounds appear only as exclusion regions in Figs 1/2/6 with no numeric tables, nothing was extractable, and returned 0 points. It never invoked figure_vision to digitize the very e... |
| 1606.07001 | NO_POINTS | `figure_vision_not_attempted_source_misrouting` | yes | needs_chart_model | The extractor recognized the DARWIN sensitivity curves live in Figure 4 but stayed in text-only mode and concluded the data "cannot be reliably extracted from text alone," returning data_source=none / 0 points. It never invoked figure_vision tracing, which is exactly the source GT used to obtain the curve. — Our ext... |
| 2401.17253 | NO_POINTS | `mass_range_prediction_not_a_curve` | no | unextractable | The paper is a theory/string-simulation prediction of the post-inflationary QCD axion mass window (~95-450 ueV), not an exclusion limit with a coupling-vs-mass curve. GT stores this as a single synthetic AxionMass point [9.5e-5, 4.5e-4] (the mass-range endpoints, not a real coupling). The extractor correctly recogni... |
| 1606.03145 | NO_POINTS | `single_prediction_point_not_emitted` | yes | hard | The extractor correctly recognized this lattice-QCD paper presents f_a vs oscillation-temperature (no mass-coupling plane) and so emitted zero points (data_source=none). But the GT is a single AxionMass *prediction* point derived from the topological-susceptibility result (the predicted QCD-axion mass window), store... |
| 1810.04602 | NO_POINTS | `source_misrouting_figure_only_curve` | yes | needs_chart_model | The exclusion limit on g_agamma vs m_a lives only in Fig. 7 (a closed exclusion contour). The extractor stayed text-only: it correctly identified the coupling type, source figure, and even the cross-section numbers, but never invoked figure_vision to trace the curve, so it emitted n_points=0. The read-vote found no ... |
| 2007.04990 | NO_POINTS | `stated_mass_bound_not_captured_text_truncated` | yes | moderate | The paper's single numeric deliverable is a QCD axion mass lower bound (~0.5 meV) stated in prose/table, but the extractor's PDF text was truncated at Section 4 where the bound appears, and the extractor only searched for a 2D coupling-vs-mass exclusion curve (the figures show string-network parameters xi/q, not a l... |
| 2208.07293 | NO_POINTS | `truncated_pdf_no_results_section` | yes | moderate | The supplied PDF text was cut off before the Section IV results tables/figures containing the numeric EDM/ALP coupling limits, so the extractor identified the experiment and mass scan window but had no data to extract; it emitted data_source=none with 0 points instead of falling back to vision on the results page. —... |
| 2112.12116 | NO_POINTS | `truncated_pdf_no_results_section_no_vision_fallback` | yes | moderate | The PDF text fed to the extractor was cut off before Section VI (the limit-setting results), so no numeric (mass, chi) values appeared in the extractable text; the pipeline never fell back to figure_vision to trace the published exclusion curve, yielding data_source="none" and n_points=0. — Extraction: data_source=n... |
| 2006.09721 | NO_POINTS | `wrong_interpretation_and_source_misrouting` | yes | moderate | XENON1T excess paper reports two interpretations; the extractor locked onto the AxionElectron solar-axion reading (whose bounds are single mass-independent values, not a curve) and read from text, missing the DarkPhoton bosonic-DM exclusion region (kinetic mixing chi vs mass) that is the verified GT. It self-noted t... |
| 2112.03439 | NO_PREDICTION | `coupling_recast_not_recognized` | yes | moderate | The extractor anchored on the paper's primary model-independent observables (DM decay rate lambda ~1e-32 s^-1 and annihilation cross-section <sigma v> ~1e-30 cm^3/s) and concluded no axion-photon coupling was presented, so it emitted data_source=none with 0 points. It missed that this Breakthrough Listen radio-line ... |