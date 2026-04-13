# AutoAxionLimits Extraction Pipeline — Evaluation Report

**Date**: 2026-04-13
**Model**: claude-haiku-4-5-20251001
**Evaluation set**: 100 unique papers (124 ground-truth entries including multi-coupling)

---

## Executive Summary

The extraction pipeline was benchmarked on 100 papers spanning 13 coupling types. Key results:

| Metric | Value |
|--------|-------|
| **Coupling type accuracy** | **86.0%** (86/100) |
| is_new_limit accuracy | 90.0% |
| is_projection accuracy | 96.0% |
| Coupling errors | 14 |
| None returns | 4 |

On the original 87-paper subset (before adding 13 new papers), coupling_type accuracy was **93.1%** (6 errors). The 13 newly added papers introduced 8 additional errors, mostly from underrepresented coupling types where the extractor lacks strong disambiguation signals.

---

## Classification Accuracy

### Coupling Type: 86.0% (14 errors out of 100)

| arXiv ID | Experiment | Predicted | Expected | Category |
|----------|-----------|-----------|----------|----------|
| 2006.04809 | Co20 | AxionPhoton | AxionMass | Ambiguous: cosmological paper |
| 2307.08577 | MnCO3 | None | AxionProton | Projection-only, no data extracted |
| 1310.8098 | CROWS | AxionPhoton | DarkPhoton | LSW experiment, debatable GT |
| 1611.05852 | RedGiants | None | ScalarElectron/ScalarNucleon | Multi-coupling, None return |
| 1708.06367 | nEDM | AxionNeutron | AxionEDM/AxionMass | Multi-coupling, wrong primary |
| 1609.00667 | NuSTAR | None | AxionPhoton | Sterile neutrino paper, GT debatable |
| 2102.11740 | LZ | None | AxionElectron | WIMP-focused paper, no ALP data |
| 0807.2926 | SolarNu | AxionPhoton | AxionElectron | Solar neutrino → wrong coupling |
| 1508.02463 | TorsionPendulum | MonopoleDipole | AxionElectron | Spin-dependent → wrong coupling |
| 1604.06800 | Superconductors | DarkPhoton | AxionElectron | Multi-target proposal → wrong primary |
| 1508.01798 | DUAL | ScalarPhoton | ScalarElectron | Scalar confusion: photon vs electron |
| 1512.06165 | TorsionBalance | ScalarNucleon | VectorBL | Fifth force → wrong coupling type |
| 2004.02733 | SNO | AxionNeutron | AxionProton | Nucleon confusion: neutron vs proton |
| 2111.09892 | NeutronStars | AxionMass | AxionProton | NS cooling → wrong coupling type |

### Error Breakdown

| Category | Count | Papers |
|----------|-------|--------|
| Baseline noise (original 87) | 6 | Co20, MnCO3, CROWS, RedGiants, nEDM, NuSTAR |
| New paper: None return | 1 | LZ |
| New paper: wrong coupling | 7 | SolarNu, TorsionPendulum, Superconductors, DUAL, TorsionBalance, SNO, NeutronStars |

The new errors cluster in **AxionElectron** (3 papers misclassified) and **AxionProton** (2 papers misclassified) — coupling types where the pipeline's disambiguation prompts are weakest. The pre-classifier receives only the paper title (no abstract) for these papers, contributing to misclassification.

### Other Classification Metrics

| Field | Accuracy | Notes |
|-------|----------|-------|
| is_new_limit | 90.0% | 10 false negatives |
| is_projection | 96.0% | 4 errors |
| data_source | 31.0% | text vs figure_vision mismatch (not actionable) |

---

## Extraction Quality

### Interpolation Metric (primary)

Builds log-log interpolation from extracted points and evaluates at ground-truth mass values.

| Statistic | Value |
|-----------|-------|
| Papers with curve comparison | 108 |
| Mean interpolation coverage | 49.7% |
| Mean median residual | 3.73 dex |
| Mean P90 residual | 4.49 dex |
| Mean fraction within 0.3 dex (factor 2) | 19.5% |
| Mean fraction within 0.5 dex (factor 3) | 28.2% |

The high residuals are dominated by papers with zero coverage or axis-scale misreads. Papers where the pipeline gets non-trivial coverage tend to perform much better.

### Top Performing Extractions (median residual < 0.5 dex)

| arXiv ID | Experiment | Coupling | Med. Resid. | <=0.3 dex | Points |
|----------|-----------|----------|-------------|-----------|--------|
| 2208.12670 | QUAX3 | AxionPhoton | 0.000 dex | 100% | 6/2 |
| 2408.15227 | ADMX2024 | AxionPhoton | 0.033 dex | 100% | 8/230 |
| 2109.11734 | BulletCluster | AxionPhoton | 0.072 dex | 81.8% | 6/12 |
| 0807.2926 | SolarNu | AxionElectron | 0.131 dex | 100% | 14/39 |
| 2303.11792 | NeutronStars_Battye2 | AxionPhoton | 0.141 dex | 100% | 3/20 |
| 2407.03828 | NuSTAR_Sun | AxionPhoton | 0.171 dex | 71.4% | 6/116 |
| 1806.05120 | Tokyo-Knirck | DarkPhoton | 0.192 dex | 81.3% | 50/94 |
| 2011.07100 | QUAX_2020 | MonopoleDipole | 0.200 dex | 59.0% | 11/79 |
| 1606.07001 | DARWIN | AxionElectron | 0.226 dex | 60.9% | 10/39 |
| 2308.09077 | CAPP-7 | AxionPhoton | 0.229 dex | 85.2% | 25/54 |
| 2308.14656 | DOSUE-RR-2 | DarkPhoton | 0.301 dex | 49.3% | 8/75 |
| 2406.00387 | Mrk421-Fermi-HAWC | AxionPhoton | 0.329 dex | 48.2% | 22/108 |
| 1207.3275 | LSW_CERN | DarkPhoton | 0.337 dex | 40.0% | 20/9 |
| 2408.02368 | MADMAX | DarkPhoton | 0.416 dex | 39.4% | 5/647 |
| 2110.06096 | ADMX2021 | AxionPhoton | 0.460 dex | 0.0% | 6/242 |
| 2504.12377 | MWDPolarisation | AxionPhoton | 0.456 dex | 33.3% | 30/15 |
| 2503.14582 | JWST_Saha | AxionPhoton | 0.468 dex | 23.1% | 7/224122 |

---

## Breakdown by Difficulty

| Difficulty | Papers | Coupling Acc. | Med. Resid. | <=0.3 dex |
|------------|--------|---------------|-------------|-----------|
| easy | 23 | 87.0% | 6.27 dex | 25.0% |
| medium | 30 | 80.0% | 5.69 dex | 21.7% |
| hard | 71 | 90.1% | 2.48 dex | 17.9% |

Hard papers (figure_vision extraction) have the best coupling accuracy because they tend to be well-known experiments with clear exclusion plots. Medium papers have the lowest accuracy, often involving ambiguous multi-coupling theory papers.

## Breakdown by Extraction Source

| Source | Papers | Med. Resid. | <=0.3 dex |
|--------|--------|-------------|-----------|
| table | 2 | 0.072 dex | 81.8% |
| text | 52 | 2.18 dex | 21.7% |
| figure_vision | 53 | 5.83 dex | 14.8% |

Table extraction is by far the most accurate when available. Text extraction outperforms vision, consistent with the text-first design.

## Breakdown by Coupling Type

| Coupling Type | Papers | Coupling Acc. |
|---------------|--------|---------------|
| AxionPhoton | 24 | 95.8% |
| DarkPhoton | 12 | 91.7% |
| AxionElectron | 12 | 75.0% |
| ScalarElectron | 9 | 77.8% |
| AxionMass | 7 | 85.7% |
| AxionEDM | 6 | 100% |
| AxionNeutron | 6 | 100% |
| AxionProton | 6 | 50.0% |
| VectorBL | 6 | 83.3% |
| ScalarPhoton | 5 | 100% |
| MonopoleDipole | 3 | 100% |
| ScalarBaryon | 2 | 100% |
| ScalarNucleon | 2 | 100% |

Weakest types: **AxionProton** (50% — confused with AxionNeutron/AxionMass), **AxionElectron** (75% — confused with AxionPhoton/DarkPhoton/MonopoleDipole).

## Confidence Calibration

| Bin | N | Mean Conf. | Actual Acc. | Gap |
|-----|---|------------|-------------|-----|
| [0.4-0.6) | 2 | 57.5% | 0.0% | +0.57 |
| [0.6-0.8) | 67 | 72.4% | 1.5% | +0.71 |
| [0.8-1.0) | 39 | 87.0% | 17.9% | +0.69 |

The pipeline is significantly **overconfident** — extraction_confidence does not correlate well with actual data quality. This is a known limitation: the confidence reflects coupling-type certainty, not curve accuracy.

---

## Improvement History

| Round | Date | PRs | coupling_type | N papers |
|-------|------|-----|---------------|----------|
| Baseline | 2026-04-12 | — | 59.6% | 87 |
| Round 2 | 2026-04-12 | #423-#427 | 78.2% | 87 |
| Round 3 | 2026-04-13 | #428-#429 | 93.1% | 87 |
| Round 3 (expanded) | 2026-04-13 | #428-#429 | **86.0%** | **100** |

The drop from 93.1% to 86.0% when expanding from 87 to 100 papers reflects the harder distribution of new papers (underrepresented coupling types with weaker disambiguation signals).

---

## Remaining Issues and Next Steps

### High-priority fixes (8 new errors)
1. **AxionElectron disambiguation**: 3 papers (SolarNu, TorsionPendulum, Superconductors) misclassified. The pre-classifier receives empty abstracts for these older papers — consider fetching abstracts from arXiv API.
2. **AxionProton vs AxionNeutron**: 2 papers (SNO, NeutronStars) confused. Add stronger proton-specific signals (hydrogen, proton spin, nucleon cooling with proton emphasis).
3. **ScalarElectron vs ScalarPhoton**: DUAL paper misclassified. These scalar couplings need clearer disambiguation.

### Systemic issues
- **Pre-classifier with empty abstracts**: Many papers pass only the title to the classifier, yielding None/0.0 confidence. The fallback to full-text extraction then makes its own coupling decision without the classifier's guidance.
- **Confidence calibration**: extraction_confidence is severely overconfident relative to actual data quality. Consider recalibrating or separating coupling confidence from curve confidence.
- **Vision axis misreads**: Large residuals (>3 dex) in vision extraction stem from axis-scale misinterpretation. The calibration/verification stage catches some but not all.

---

## Methodology

### Interpolation metric (primary)
1. Filter boundary-closure sentinel points (coupling >= 1e-2) from both extracted and GT data
2. Build `scipy.interpolate.interp1d` from extracted points in log10(mass) -> log10(coupling) space
3. Evaluate the interpolation at each ground-truth mass value
4. Compute residual = |log10(g_interpolated) - log10(g_ground_truth)| at each GT point
5. Only GT points inside the extracted mass range are used (no extrapolation)

### Confidence calibration
- A paper is "accurate" if median residual < 0.3 dex AND interpolation coverage >= 50%
- Papers binned by extraction_confidence; actual accuracy computed per bin
