# Failure Analysis — full 346-paper benchmark (`full346`, report.md of 2026-06-08)

Per-paper study of the 67 significantly failed cases: 28 zero_overlap, 20 compared with median residual ≥ 3 dex, 16 no_extracted_points, 3 no_prediction. Each paper was diagnosed locally (no API extraction) by a dedicated agent reading the cached extraction JSON (`evaluation/results/`), the ground-truth entry + data file, the run log (`full346_extract.log`), and the locally cached paper PDF. Structured per-paper verdicts: `failure_analysis_full346_detail.md` (same directory). Confidence: 65/67 high, 2 medium.

Benchmark context: extractor ran on Haiku via `EXTRACTOR_MODEL`; the working tree at run time carried at least part of the uncommitted #613 lever-D draft (now committed on `fix/613-leverd-curve-selection` @ f931446e), so the surviving wrong-curve failures below have to be re-validated against that branch before further prompt work.

## 1. Headline

- **Total failures analyzed:** 67
- **Genuine extractor faults:** 29 (43%)
- **Benchmark/scoring issues (not extractor fault):** 38 (57%)
- **Recoverable:** 64 (96%); unfixable: 3 (2005.14694 — d_e limit only in the published Nature version; 2011.08693 — GT digitizes private-communication data absent from the paper; 2112.03439 — paper reports decay rates, not couplings)

The single biggest finding: **more than half of the failure tail is the benchmark grading itself, not the extractor** — and within that, two large mechanical clusters (AxionMass prediction-band GT files, and `evaluate.py` missing scoring logic that `subset_compare.py` already has). The reported headline metrics materially understate true pipeline quality; e.g. 2103.03783 rescores 18.3 → 1.06 dex and 2401.18076 rescores 20.8 → 2.3 dex once conventions are applied in scoring, and 1406.6053's "zero_overlap" extraction is actually accurate to 0.003 dex.

## 2. Failures by root cause

| Root cause | Papers | Extractor fault | Notes |
|---|---|---|---|
| gt_benchmark_issue | 24 | no | AxionMass prediction bands, wrong GT↔paper mapping, shared/composite GT files, λ[m] axes |
| comparator_encoding | 11 | no | evaluate.py scoring gaps vs subset_compare.py (conventions, mass-independent bounds, single points, contours, sentinels) |
| convention_units | 9 | yes | missing converters: f_a inversion, Γ→g, squared axes, ξ, g_d, Λ |
| wrong_curve_vision | 9 | yes | traced existing-exclusion/compilation/wrong-panel curve |
| anchor_snap | 5 | yes | AxionMass VALID_RANGES floor too high → spurious snaps of ultralight windows |
| text_extraction_error | 4 | yes | excerpt windowing, analytic reconstruction, flat-bound encoding, unit-factor misuse |
| sparse_points | 4 | mixed | n_ext==1 auto-scored ∞; R4_MIN_SPAN_DEX too strict for narrow-band figures |
| source_misrouting | 1 | yes | approximate text read beat figure |

## 3. Development levers (ranked)

### Lever 1 — Port subset_compare.py scoring logic into evaluate.py (recovers ~14, zero extractor risk) ★ do first
**Papers:** 2103.03783, 2401.18076, 2208.07293, 2007.03694, 1406.6053, 0910.5914, 1906.08814, 2111.09892, 2306.01048, 2410.19902, 2102.08764 + the sparse-points trio 1504.00118, 2208.03183, 2305.09711.
**Fix (all in `evaluation/` — the extractor is untouched):**
- Wire `classify_reported_convention()`/`to_canonical()` into the `evaluate.py --metrics` pairing path (subset_compare.py already does this; the full-pool report does not). Verified: 2103.03783 18.3→1.06 dex, 2401.18076 20.8→2.3 dex.
- Port `_expand_mass_independent` (flat bounds `[(0,g)…]` → wide 2-point line) into the scoring path (2007.03694, 1406.6053 @0.003 dex, 0910.5914 @0.08 dex).
- Score `n_ext == 1` extractions via the reverse pass (GT interpolated onto the extracted mass) instead of returning ∞ residual/zero_overlap (`metrics.py` ~L363). Five papers whose single headline point is *correct* currently score as total failures.
- Pre-process closed-contour/wedge GT curves to their lower envelope before interpolation (2306.01048: 3.51→0.02 dex).
- Add the UNCONVERTIBLE guard so unit-incommensurable pairs become `convention_mismatch`, not a 15-dex "residual" (2208.07293).

### Lever 2 — GT hygiene: AxionMass prediction bands + broken mappings (recovers ~21, mechanical)
**Papers:** 1202.5851, 1505.07455, 1509.00026, 1606.03145, 1608.05414, 1705.00676, 1906.00967, 2007.04990, 2108.05368, 2206.11598, 2412.08699 (prediction bands); 1712.00483, 2011.07100, 2302.09096 (λ[m] axes); 1003.0964, 1410.5244, 2411.13701, 2312.11608 (wrong file/mapping); 1110.2895, 2112.12116, 1806.00310 (shared/composite GT files).
**Fix (in `evaluation/build_ground_truth.py` + `papers.json`):**
- Exclude `limit_data/AxionMass/*` **prediction-band** files (headers "Predicted axion masses"; both columns are masses, no coupling) from GT auto-selection, or score them as mass-band overlap. 11 papers where a *correct* extractor refusal or mass-window answer is graded as failure.
- Convert MonopoleDipole GT x-columns given as λ[m] to eV (`m = 1.973e-7/λ`) at GT-ingestion time.
- Re-key wrong mappings: LSW_UWA.txt belongs to 1105.6169 not 1003.0964; 1410.5244 docs/dp.md refs swapped; 2411.13701's file stores ε (×1e-11 offset); 2312.11608 should point at `Projections/21cm.txt`.
- Write **per-entry GT data files** when multiple repo files share one arXiv id (currently all collapse onto one `<id>.txt`, structurally breaking 2nd/3rd entries).
- Tag the 3 unfixable papers (2005.14694, 2011.08693, 2112.03439) as excluded-with-reason.

### Lever 3 — Convention converters, round 2 (recovers ~9)
**Papers:** 2211.02661, 2105.13963, 2301.06560, 2311.05476, 0809.4700, 1508.02463, astro-ph/0611502, 1708.06367, 2112.10618.
**Fix (in `evaluation/conventions.py` + extractor prompt):**
- `f_a` [GeV] ↔ `1/f_a` inversion (two AxionMass/AxionPhoton papers plot f_a increasing = coupling decreasing).
- Decay-rate planes: Γ[s⁻¹] → g_aγγ via `g = sqrt(64π·Γ/m³)`.
- Squared-coupling axes (`g²/4π`, `g²`) → `sqrt` branch, mirroring the existing DarkPhoton ε² case.
- One-off converters: ξ→g_aγγ (astro-ph/0611502), d_n vs g_d for AxionEDM (1708.06367), Λ→d_e (2112.10618).
- **Guard interaction:** suppress the anchor snap whenever the declared convention is flagged non-canonical — several convention failures were then compounded by a spurious snap.

### Lever 4 — AxionMass VALID_RANGES + anchor (recovers 5, tiny diff)
**Papers:** 2406.10337, 2412.20932, 2012.12790, 2303.09865, 2302.00685 (all ultralight/superradiance, masses 1e-21–1e-11 eV).
**Fix (in `pipeline/config.py` + `extractor.py`):** widen `VALID_RANGES['AxionMass']` mass floor to ~1e-24 eV; add an explicit `_EXPECTED_MASS_ANCHOR_EV['AxionMass']`; and make `_validate_extracted_range` **revert** a snap that still leaves the window invalid (currently it commits the corruption anyway).

### Lever 5 — Vision multi-curve disambiguation, beyond the prompt (recovers ~9, highest effort)
**Papers:** 1512.06165, 2309.07995, 1508.01798 (projection vs existing exclusion), 1008.3536, 1207.3275 (compilation envelope), 1708.02111, 1903.12190, 1808.02340 (wrong panel/regime), 1912.07751.
These survived the run despite the draft #613 prompt guidance being (at least partly) active — prompt-only instruction is insufficient at Haiku tier. **Fix (extends #613):** add deterministic post-hoc gates rather than more prose: (a) when `is_projection=true`, reject a traced curve whose shape/range matches a *named other experiment* in the legend; (b) reject curves whose axis units or mass regime contradict the declared coupling and the abstract-stated mass window (1708.02111, 1903.12190); (c) panel targeting by abstract mass window before tracing (1808.02340). Backstop remains the dedicated chart-extraction model (#606 note).

### Lever 6 — Text-path hygiene (recovers ~5)
**Papers:** 2402.00741 (excerpt window cuts off the equation line after the keyword lead-in — widen `_result_excerpts` context), 1401.6460 + 2407.10618 (demote text candidates whose notes admit analytic reconstruction/approximate figure reads below a real vision curve, in `read_vote`/`transform_guard` quality tiers), 1906.11844 (Stage 1 should emit mass-independent bounds as a flat 2-point line — reuse the Lever 1 encoding), 2109.11734 (prompt: the 4.136e-15 Hz→eV factor applies to *frequencies only*, never to values already in μeV).

### Lever 7 — Sparse/narrow-band extraction (recovers 1–2)
**Paper:** 2110.10497. Scale `R4_MIN_SPAN_DEX` to the figure's actual axis extent so narrow resonance searches aren't rejected for spanning "too few" decades; single-point scoring from Lever 1 covers the rest.

**Priority: 1 → 2 → 4 → 3 → 6 → 5 → 7.** Levers 1+2 are pure benchmark/eval-side (no extractor behavior change, no re-extraction needed to validate — just re-run `--metrics`), recover ~35 papers, and will visibly move every headline number: fewer zero-overlaps, lower macro-average residual, and honest per-type medians. Lever 4 is a 5-line diff. Levers 3/6 are moderate extractor changes needing a subset re-run. Lever 5 is the open research problem and should be validated against the committed #613 branch first.

## 4. Per-paper appendix

| arXiv | status | root_cause | fault | fixable | fix (summary) |
|---|---|---|---|---|---|
| 2012.12790 | zero_overlap | anchor_snap | y | y | Widen VALID_RANGES['AxionMass'] in pipeline/config.py to cover ultralight superradiance masses (mass_lo ~1e-22 eV, matching the fa-plot data in limit_data/fa... |
| 2302.00685 | zero_overlap | anchor_snap | y | y | Widen VALID_RANGES['AxionMass'] mass lower bound in pipeline/config.py from 1e-12 to ~1e-22 eV (fa-plane cosmology limits are ultralight; the repo's own GT f... |
| 2303.09865 | zero_overlap | anchor_snap | y | y | Two-part: (1) widen VALID_RANGES['AxionMass'] mass floor in pipeline/config.py from 1e-12 to 1e-24 eV — the O'Hare fa plot includes ultralight/fuzzy-DM f_a b... |
| 2406.10337 | zero_overlap | anchor_snap | y | y | Widen VALID_RANGES['AxionMass'] in pipeline/config.py to cover the fa-plot superradiance domain (mass floor ~1e-22 eV, coupling floor ~1e-30 GeV^-1 for fa^-1... |
| 2412.20932 | zero_overlap | anchor_snap | y | y | Two-part: (1) widen VALID_RANGES['AxionMass'] mass window in pipeline/config.py from (1e-12,1e18) to include the ultralight fa-plane (~1e-24 lower edge; limi... |
| 0910.5914 | zero_overlap | comparator_encoding | n | y | Same fix as 1406.6053: in evaluation/metrics.py compute_interpolation_metrics, when the forward pass has 0 interpolatable GT vertices, fall back to the rever... |
| 1406.6053 | zero_overlap | comparator_encoding | n | y | In evaluation/metrics.py compute_interpolation_metrics, do not early-return _empty when num_interpolatable==0: still compute the reverse pass (GT interpolate... |
| 1906.08814 | zero_overlap | comparator_encoding | n | y | In evaluation/metrics.py compute_interpolation_metrics, when n_ext == 1 fall back to the reverse pass (interpolate the GT curve at the extracted mass and rep... |
| 2007.03694 | zero_overlap | comparator_encoding | n | y | Port _expand_mass_independent from evaluation/subset_compare.py into the evaluate.py scoring path (cleanest: move it to evaluation/metrics.py and apply to ex... |
| 2102.08764 | zero_overlap | comparator_encoding | n | y | In evaluation/metrics.py compute_interpolation_metrics: when the forward pass yields num_interpolatable=0 but the log-mass intervals overlap (mass_jaccard>0)... |
| 2103.03783 | high_resid | comparator_encoding | n | y | Wire classify_reported_convention + to_canonical into evaluate.py's comparison path (mirror subset_compare.py:144-147) so an extraction that declares a GeV^-... |
| 2111.09892 | zero_overlap | comparator_encoding | n | y | In evaluate.py/metrics.py add a single-point fallback: when n_ext == 1 and the point's mass lies inside the GT mass range, score /log10(c_ext) - log10(gt_int... |
| 2208.07293 | zero_overlap | comparator_encoding | n | y | Port the subset_compare.py UNCONVERTIBLE guard (_maybe_canonicalize / conventions.classify_reported_convention) into evaluate.py's candidate/pairing block (~... |
| 2306.01048 | high_resid | comparator_encoding | n | y | In evaluation/metrics.py compute_interpolation_metrics: detect multivalued/closed-contour GT (non-monotonic mass sequence or multiple couplings per mass diff... |
| 2401.18076 | high_resid | comparator_encoding | n | y | Make the full346 scoring/manifest residual computation call evaluation.conventions.classify_reported_convention + to_canonical (as evaluation/subset_compare.... |
| 2410.19902 | zero_overlap | comparator_encoding | n | y | In evaluation/metrics.py compute_interpolation_metrics, when the forward pass yields 0 interpolatable GT points but the mass ranges intersect (typical for 2-... |
| 0809.4700 | high_resid | convention_units | y | y | Two-layer fix: (1) evaluation/conventions.py classify_convention/to_canonical: detect squared-coupling tokens ('^2/(4pi)', 'g^2', 'squared') for axion coupli... |
| 1508.02463 | high_resid | convention_units | y | y | Add an AxionElectron (and generically axion-fermion) branch to classify_reported_convention() in evaluation/conventions.py detecting '^2'/'squared' in the de... |
| 1708.06367 | high_resid | convention_units | y | y | Two options: (a) add a vetted mass-dependent converter in evaluation/conventions.py for oscillating-EDM amplitude, g_d [GeV^-2] = d_n[e*cm] * 5.06e13 * m_a /... |
| 2105.13963 | high_resid | convention_units | y | y | Add an AxionMass branch to classify_reported_convention() in evaluation/conventions.py detecting a declared 'f_a in GeV' (values >> 1) and a to_canonical() p... |
| 2112.10618 | no_extracted_points | convention_units | y | y | Teach _STAGE2A_AXIS_SYSTEM/_run_stage2 in pipeline/extractor.py to recognize exponent-labeled axes (log10(m/eV), log10(GeV/Lambda)) and to extract curves in ... |
| 2211.02661 | high_resid | convention_units | y | y | Add a vetted f_a[GeV] -> 1/f_a[GeV^-1] inversion for AxionMass/f_a_norm in evaluation/conventions.py to_canonical; and in pipeline/extractor.py suppress the ... |
| 2301.06560 | high_resid | convention_units | y | y | Handle decay-rate/lifetime y-axes: when vision reports the axis as Gamma_{a->gamma gamma} [s^-1] (or tau), convert to g_agamma via g = sqrt(64*pi*Gamma/m_a^3... |
| 2311.05476 | no_extracted_points | convention_units | y | y | Add a decay-rate convention to evaluation/conventions.py to_canonical (Gamma [s^-1] -> g_agamma [GeV^-1] = sqrt(64*pi*Gamma*hbar)/m^1.5 with hbar=6.582e-25 G... |
| astro-ph/0611502 | high_resid | convention_units | y | y | Add an AxionPhoton branch in evaluation/conventions.py classify_reported_convention()/to_canonical() for a declared dimensionless xi (lifetime tau = 6.8e24 x... |
| 1003.0964 | zero_overlap | gt_benchmark_issue | n | y | Fix the benchmark mapping in evaluation/ground_truth/papers.json: LSW_UWA.txt does not match 1003.0964's reported limit (verify against arXiv:1105.6169 and r... |
| 1110.2895 | no_extracted_points | gt_benchmark_issue | n | y | Fix the GT side in evaluation/build_ground_truth.py: when multiple repo files share one arXiv id, write per-entry GT data files instead of collapsing to one ... |
| 1202.5851 | no_extracted_points | gt_benchmark_issue | n | y | In evaluation/ground_truth/papers.json auto-selection, exclude (or re-encode) AxionMass theory-band files consumed by PlotTheoryMass whose two columns are (m... |
| 1410.5244 | high_resid | gt_benchmark_issue | n | y | Fix the evaluation/ground_truth/papers.json entry for 1410.5244 to reference_repo_file limit_data/DarkPhoton/LSW_UWA.txt (and regenerate ground_truth/data/14... |
| 1505.07455 | no_extracted_points | gt_benchmark_issue | n | y | In evaluation/metrics.py (or ground_truth/papers.json auto-selection), treat AxionMass single-point placeholder entries as mass-only: either exclude them fro... |
| 1509.00026 | no_extracted_points | gt_benchmark_issue | n | y | Exclude AxionMass single-point theoretical mass-prediction entries (placeholder couplings) from evaluation/ground_truth/papers.json, or score AxionMass entri... |
| 1606.03145 | no_extracted_points | gt_benchmark_issue | n | y | In the GT auto-expansion (evaluation/ground_truth/papers.json generation), treat limit_data/AxionMass/* files as mass-window predictions, not (mass, coupling... |
| 1608.05414 | no_extracted_points | gt_benchmark_issue | n | y | In evaluation/ground_truth/papers.json remove or flag AxionMass prediction-band entries (Ballesteros16 and similar) as non-comparable; alternatively teach GT... |
| 1705.00676 | no_extracted_points | gt_benchmark_issue | n | y | In the GT auto-expansion (evaluation/ground_truth/papers.json generation), exclude or specially tag limit_data/AxionMass/ theory mass-prediction files (both ... |
| 1712.00483 | zero_overlap | gt_benchmark_issue | n | y | In GT data generation (evaluation/ground_truth populate_data_from_repo), detect the '# lambda [m]' header of limit_data/ScalarBaryon/ScalarNucleon/MICROSCOPE... |
| 1806.00310 | zero_overlap | gt_benchmark_issue | n | y | Fix the benchmark: give 1806.00310 its own GT containing only the 2018 notch (or tag the entry so it routes to gt_point_reference / a single-point nearest-ma... |
| 1906.00967 | no_extracted_points | gt_benchmark_issue | n | y | Exclude AxionMass prediction-band files (header 'Predicted axion masses (in eV)'; both columns are masses) from GT auto-selection in evaluation/ground_truth/... |
| 2005.14694 | no_prediction | gt_benchmark_issue | n | n | Not recoverable from the arXiv PDF. Benchmark fix: tag the entry in evaluation/ground_truth/papers.json as published-version-only (preprint/published content... |
| 2007.04990 | no_extracted_points | gt_benchmark_issue | n | y | Exclude AxionMass prediction-band files (repo headers say 'Predicted axion masses (in eV)'; both columns are masses) from GT auto-selection in evaluation/gro... |
| 2011.07100 | zero_overlap | gt_benchmark_issue | n | y | In the GT build/ingestion path (evaluation/gold_build.py auto-copy or the GT loader in evaluation/evaluate.py), detect fifth-force files whose header declare... |
| 2011.08693 | no_prediction | gt_benchmark_issue | n | n | No extractor fix possible - the 178-point BHSR compilation exists only via private communication with the authors. Benchmark fix: tag/remove the 2011.08693 e... |
| 2108.05368 | no_extracted_points | gt_benchmark_issue | n | y | Exclude AxionMass prediction-band files (header 'Predicted axion masses (in eV)'; both columns are masses) from GT auto-selection in evaluation/ground_truth/... |
| 2112.03439 | no_prediction | gt_benchmark_issue | n | n | Not recoverable by extraction (requires lambda->g_agamma physics conversion with halo/stimulated-emission assumptions). Benchmark fix: tag/remove the 2112.03... |
| 2112.12116 | high_resid | gt_benchmark_issue | n | y | Store per-entry GT data files in evaluation/ground_truth/data (e.g. <safeid>__<paper_title>.txt) instead of one file per arXiv id, so multi-limit papers comp... |
| 2206.11598 | no_extracted_points | gt_benchmark_issue | n | y | Exclude limit_data/AxionMass/ theory mass-prediction files (two mass columns, no coupling axis) from GT auto-expansion in evaluation/build_ground_truth.py / ... |
| 2302.09096 | zero_overlap | gt_benchmark_issue | n | y | In GT ingestion / evaluation loader, detect 'lambda [m]' headers on MonopoleDipole repo files (most NucleonNucleon files: Moon, Sun, Mainz, SMILE, Grenoble, ... |
| 2312.11608 | zero_overlap | gt_benchmark_issue | n | y | Repoint the 2312.11608 entry in evaluation/ground_truth/papers.json to limit_data/AxionPhoton/Projections/21cm.txt with is_projection=true (or drop the paper... |
| 2411.13701 | high_resid | gt_benchmark_issue | n | y | Fix the benchmark, not the extractor: correct the coupling_convention for this entry in evaluation/ground_truth/papers.json and add a per-file override in ev... |
| 2412.08699 | no_extracted_points | gt_benchmark_issue | n | y | Exclude AxionMass prediction-band files (header 'Predicted axion masses (in eV)'; both columns are masses) from GT auto-selection in evaluation/ground_truth/... |
| 2407.10618 | high_resid | source_misrouting | y | y | In pipeline/read_vote.py quality-tier selection (lever A) / pipeline/transform_guard.py: demote a 'text' candidate to vision tier (or below) when its notes a... |
| 1504.00118 | zero_overlap | sparse_points | n | y | Primary: evaluation/metrics.py compute_interpolation_metrics - score n_ext==1 via the reverse pass (GT interp at extracted mass), giving ~0.08 dex here. Seco... |
| 2110.10497 | zero_overlap | sparse_points | y | y | Scale the R4 span floor in pipeline/transform_guard.py to the figure's actual x-axis extent (or lower it for narrow-band haloscope figures) so a faithful sub... |
| 2208.03183 | zero_overlap | sparse_points | n | y | Same metrics.py fix (score n_ext==1 via GT-interp reverse pass) plus a small nearest-log-mass tolerance (~0.01 dex) when GT is a near-degenerate spike, in ev... |
| 2305.09711 | zero_overlap | sparse_points | n | y | In evaluation/metrics.py compute_interpolation_metrics, drop the 'n_ext < 2' early return and score single-point extractions via the existing reverse pass (i... |
| 1401.6460 | high_resid | text_extraction_error | y | y | Don't trust LLM arithmetic for analytically-reconstructed text curves: in read_vote.py source selection, demote text points whose notes say 'reconstructed/de... |
| 1906.11844 | no_extracted_points | text_extraction_error | y | y | In pipeline/extractor.py Stage 1 prompt (_run_stage1), instruct that a stated mass-independent coupling bound (typical of astrophysical/SN limits) should be ... |
| 2109.11734 | zero_overlap | text_extraction_error | y | y | Tighten the mass-conversion block of the Stage-1/2 prompts in pipeline/extractor.py (~L505/615): state that the 4.136e-15*f[Hz] rule applies ONLY when the pa... |
| 2402.00741 | no_extracted_points | text_extraction_error | y | y | In pipeline/extractor.py _result_excerpts(), widen the context window around keyword lines from +/-1 line to several following lines (e.g. i-1..i+6) or exten... |
| 1008.3536 | high_resid | wrong_curve_vision | y | y | In pipeline/extractor.py Stage-2 vision: when the candidate figure is a compilation/summary plot (caption keywords like 'compilation'/'summary of current bou... |
| 1207.3275 | high_resid | wrong_curve_vision | y | y | In pipeline/extractor.py vision prompt, instruct tracing ONLY the region newly claimed by this paper (named/highlighted in caption) and explicitly forbid mer... |
| 1508.01798 | high_resid | wrong_curve_vision | y | y | In pipeline/extractor.py Stage-2 vision curve-selection prompt: when Stage-1 context says the paper's main result is a projection (is_projection=true / 'prop... |
| 1512.06165 | high_resid | wrong_curve_vision | y | y | In pipeline/extractor.py Stage-2 vision prompt, condition the trace target on the paper's own result: when the classifier/notes mark is_projection=true (prop... |
| 1708.02111 | zero_overlap | wrong_curve_vision | y | y | In pipeline/extractor.py vision path, reject (or re-prompt/re-route to another panel) when the vision response itself reports that the traced y-axis quantity... |
| 1808.02340 | zero_overlap | wrong_curve_vision | y | y | Improve vision figure/panel targeting in pipeline/extractor.py (Stage 2 prompt + _run_vision_verify): steer to the figure whose axis window matches the abstr... |
| 1903.12190 | zero_overlap | wrong_curve_vision | y | y | In pipeline/extractor.py Stage-2 figure/curve selection: reject a candidate figure whose x-axis mass regime (GeV-scale m_chi, 'millicharge'/'epsilon' axis) i... |
| 1912.07751 | zero_overlap | wrong_curve_vision | y | y | Treat an above-ceiling text read whose MASS window is in-range as recoverable rather than hard-rejecting it (VALID_RANGES coupling ceiling 1e-3 in pipeline/c... |
| 2309.07995 | high_resid | wrong_curve_vision | y | y | In pipeline/extractor.py vision stage: when is_projection/is_new_limit is targeted, instruct the model to trace the paper's OWN new-result curve (typically t... |