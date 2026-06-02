# AutoAxionLimits Extraction Pipeline — Evaluation Report

## Summary

- **Papers evaluated**: 211
- **Papers with curve comparison**: 154

## Curve-Comparison Coverage

A curve is scored only against a ground-truth curve of the **same coupling**. Papers whose extracted coupling has no matching GT curve are not comparable and are excluded from residual statistics (this is not an extraction failure).

| Status | Papers | Meaning |
|--------|--------|---------|
| compared | 154 | scored against a same-coupling GT curve |
| no_comparable_gt | 16 | extracted coupling has no GT curve in the pool |
| gt_unusable | 23 | GT curve has <2 usable points after boundary filtering |
| no_extracted_points | 10 | pipeline returned no data points |
| no_prediction | 4 | pipeline returned no coupling type |
| extraction_failed | 4 | extraction errored (download/parse/API) |

## Classification Accuracy

| Field | Accuracy | N |
|-------|----------|---|
| coupling_type | 90.3% | 207 |
| is_new_limit | N/A — no human-verified labels | 0 |
| is_projection | N/A — no human-verified labels | 0 |
| data_source | N/A — no human-verified labels | 0 |

> `is_new_limit`, `is_projection`, and `data_source` are scored only against human-verified ground-truth entries. The current pool is entirely repo-sourced (placeholder labels), so these are reported as N/A rather than against placeholders.

### Coupling Type Misclassifications

| arXiv ID | Predicted | Expected |
|----------|-----------|----------|
| 2209.12901 | AxionPhoton | ['AxionEDM'] |
| 1608.01994 | DarkPhoton | ['AxionElectron'] |
| 1711.08999 | AxionNeutron | ['AxionMass'] |
| 2211.08439 | AxionElectron | ['AxionNeutron'] |
| 2307.08577 | AxionNeutron | ['AxionProton'] |
| 1607.07327 | ScalarBaryon | ['ScalarElectron', 'ScalarPhoton'] |
| 1611.05852 | None | ['ScalarElectron', 'ScalarNucleon'] |
| 2205.03617 | DarkPhoton | ['VectorBL'] |
| 2401.18076 | ScalarBaryon | ['ScalarElectron', 'ScalarPhoton'] |
| 1207.2442 | ScalarBaryon | ['VectorBL'] |
| 1609.00667 | None | ['AxionPhoton'] |
| 2306.01048 | AxionNeutron | ['AxionProton'] |
| 2102.11740 | None | ['AxionElectron'] |
| 0807.2926 | AxionPhoton | ['AxionElectron'] |
| 1512.06165 | ScalarBaryon | ['VectorBL'] |
| 2111.09892 | AxionMass | ['AxionNeutron', 'AxionProton'] |
| 2410.02218 | AxionCPV | ['AxionEDM', 'AxionMass'] |
| 2207.11330 | AxionPhoton | ['AxionElectron'] |
| 2312.11608 | AxionMass | ['AxionPhoton'] |
| 2111.08025 | None | ['AxionPhoton'] |

## Extraction Quality — Interpolation Metric (primary)

Build log-log interpolation from extracted points, evaluate at ground-truth masses.

- **Papers compared**: 154 (123 with mass-range overlap, 31 with zero overlap)

**Coupling-value accuracy** (papers with mass-range overlap):
- **Median residual across papers**: 0.696 dex (IQR 0.239–1.912)
- **Mean residual across papers** (outlier-sensitive): 2.058 dex
- **Mean fraction within 0.3 dex (factor 2)**: 29.8%
- **Mean fraction within 0.5 dex (factor 3)**: 40.5%

**Mass-range coverage** (a separate failure mode):
- **Mean interpolation coverage**: 62.9%
- **Zero-overlap papers**: 31/154 (20.1%) — extracted masses miss the GT range entirely (usually 1–2 extracted points or the wrong mass window)

## Per-Paper Results

| arXiv ID | Coupling | Conf. | Interp. Cov. | Med. Resid. | ≤0.3 dex | Points |
|----------|----------|-------|--------------|-------------|----------|--------|
| 2208.07293 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 2/2 |
| 2212.04413 | ✓ | 0.45 | no_extracted_points | — | — | — |
| 2410.19902 | ✓ | 0.75 | 50.0% | 0.608 | 0.0% | 15/2 |
| 2209.12901 | ✗ (AxionPhoton) | 0.75 | no_comparable_gt | — | — | — |
| 1907.03767 | ✓ | 0.85 | 97.0% | 2.459 | 0.0% | 11/164 |
| 2209.06216 | ✓ | 0.85 | 36.6% | 0.696 | 15.5% | 2/2597 |
| 2005.14184 | ✓ | 0.85 | 99.7% | 0.825 | 14.6% | 12/891 |
| 1608.01994 | ✗ (DarkPhoton) | 0.75 | no_comparable_gt | — | — | — |
| 2504.00720 | ✓ | 0.65 | 88.8% | 3.154 | 0.0% | 8/80 |
| 1608.05414 | ✓ | 0.75 | gt_unusable | — | — | — |
| 2006.04809 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 1711.08999 | ✗ (AxionNeutron) | 0.75 | no_comparable_gt | — | — | — |
| 2408.02668 | ✓ | 0.85 | 100.0% | 0.947 | 12.6% | 12/413 |
| 1905.13650 | ✓ | 0.65 | 100.0% | 5.421 | 1.2% | 30/243 |
| 2211.08439 | ✗ (AxionElectron) | 0.65 | no_comparable_gt | — | — | — |
| 2110.03679 | ✓ | 0.75 | 60.5% | 0.130 | 76.9% | 13/129 |
| 2303.07370 | ✓ | 0.65 | 100.0% | 2.672 | 1.9% | 6/107 |
| 1912.05733 | ✓ | 0.75 | 71.7% | 1.837 | 9.1% | 12/46 |
| 2309.16600 | ✓ | 0.85 | 39.3% | 1.350 | 0.0% | 23/196 |
| 2504.16044 | ✓ | 0.92 | gt_unusable | — | — | — |
| 2307.08577 | ✗ (AxionNeutron) | 0.35 | no_comparable_gt | — | — | — |
| 2312.06746 | ✓ | 0.92 | 0.6% | 0.615 | 0.0% | 2/171 |
| 1310.8098 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 2/43 |
| 2408.02368 | ✓ | 0.95 | 100.0% | 0.176 | 87.8% | 3/647 |
| 2212.01139 | ✓ | 0.75 | 98.5% | 1.082 | 0.0% | 23/134 |
| 2011.07100 | ✓ | 0.92 | 98.7% | 0.200 | 59.0% | 8/79 |
| 1403.1290 | ✓ | 0.85 | 97.4% | 21.135 | 0.0% | 70/38 |
| 2302.09096 | ✓ | 0.75 | 58.5% | 24.536 | 0.0% | 9/41 |
| 1410.7267 | ✓ | 0.85 | gt_unusable | — | — | — |
| 1712.00483 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 3/119 |
| 1607.07327 | ✗ (ScalarBaryon) | 0.75 | no_comparable_gt | — | — | — |
| 1611.05852 | ✗ (None) | 0.25 | no_prediction | — | — | — |
| 2111.06883 | ✓ | 0.65 | gt_unusable | — | — | — |
| 2112.10618 | ✓ | 0.35 | no_extracted_points | — | — | — |
| 0802.2350 | ✓ | 0.95 | gt_unusable | — | — | — |
| 2009.04517 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 2010.08107 | ✓ | 0.75 | gt_unusable | — | — | — |
| 2201.02042 | ✓ | 0.75 | gt_unusable | — | — | — |
| 2403.03004 | ✓ | 0.85 | 92.3% | 1.507 | 0.0% | 30/414 |
| 2205.03617 | ✗ (DarkPhoton) | 0.65 | no_comparable_gt | — | — | — |
| 2310.06017 | ✓ | 0.75 | 46.9% | 1.005 | 5.7% | 3/113 |
| 2007.04899 | ✓ | 0.85 | 1.7% | 2.662 | 2.2% | 30/24406 |
| 2102.08764 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 2/2 |
| 2308.14656 | ✓ | 0.92 | 98.7% | 0.413 | 1.4% | 6/75 |
| 2207.03102 | ✓ | 0.92 | gt_unusable | — | — | — |
| 1505.07455 | ✓ | 0.85 | gt_unusable | — | — | — |
| 2105.04603 | ✓ | 0.85 | 98.3% | 0.699 | 27.1% | 7/707 |
| 2303.11792 | ✓ | 0.92 | 90.0% | 0.141 | 100.0% | 2/20 |
| 1607.06083 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 8/2 |
| 2108.04746 | ✓ | 0.65 | gt_unusable | — | — | — |
| 2208.12670 | ✓ | 0.95 | 100.0% | 0.000 | 100.0% | 6/2 |
| 1806.05120 | ✓ | 0.92 | 97.9% | 0.192 | 81.5% | 2/94 |
| 2101.01241 | ✓ | 0.92 | gt_unusable | — | — | — |
| 2404.14476 | ✓ | 0.95 | 51.0% | 0.071 | 88.5% | 3/51 |
| 2504.12377 | ✓ | 0.92 | 13.3% | 0.003 | 100.0% | 8/15 |
| 2408.15227 | ✓ | 0.85 | 100.0% | 0.624 | 0.0% | 8/230 |
| 2209.09917 | ✓ | 0.85 | 80.0% | 0.536 | 25.0% | 10/5 |
| 2406.00387 | ✓ | 0.95 | 40.7% | 0.241 | 59.1% | 3/108 |
| 1207.3275 | ✓ | 0.75 | 40.0% | 0.880 | 0.0% | 30/10 |
| 2205.01079 | ✓ | 0.65 | 78.6% | 0.523 | 27.3% | 5/14 |
| 2212.02403 | ✓ | 0.85 | 100.0% | 2.154 | 2.4% | 4/42 |
| 1903.12190 | ✓ | 0.65 | 66.7% | 6.526 | 0.0% | 14/3 |
| 2305.00890 | ✓ | 0.85 | 100.0% | 7.821 | 1.4% | 19/4991 |
| 1708.06367 | ✓ | 0.85 | 100.0% | 6.147 | 5.6% | 39/36 |
| 2312.13723 | ✓ | 0.75 | gt_unusable | — | — | — |
| 2410.10363 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 6/367 |
| 1202.5851 | ✓ | 0.75 | gt_unusable | — | — | — |
| 2208.06519 | ✓ | 0.92 | 100.0% | 0.432 | 0.0% | 30/2 |
| 2401.16747 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 26/66 |
| 2401.18076 | ✗ (ScalarBaryon) | 0.75 | no_comparable_gt | — | — | — |
| 2503.13653 | ✓ | 0.75 | 74.1% | 1.843 | 5.0% | 9/54 |
| 2308.06339 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 1/76 |
| 2109.11734 | ✓ | 0.92 | 75.0% | 0.026 | 66.7% | 7/12 |
| 2308.09077 | ✓ | 0.85 | 83.3% | 1.366 | 0.0% | 2/54 |
| 2110.06096 | ✓ | 0.92 | 97.5% | 0.570 | 0.0% | 6/242 |
| 2205.03679 | ✓ | 0.92 | 100.0% | 0.223 | 68.9% | 2/1525 |
| 2202.08858 | ✓ | 0.75 | 99.5% | 3.398 | 0.0% | 15/203 |
| 2503.14582 | ✓ | 0.75 | 100.0% | 0.226 | 63.3% | 13/224122 |
| 1207.2442 | ✗ (ScalarBaryon) | 0.35 | no_comparable_gt | — | — | — |
| 1609.00667 | ✗ (None) | 0.25 | no_prediction | — | — | — |
| 2407.03828 | ✓ | 0.92 | 42.2% | 0.215 | 61.2% | 6/116 |
| 1810.04602 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 18/55 |
| 2110.10262 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 1/88 |
| 2306.05934 | ✓ | 0.92 | 99.5% | 0.111 | 99.0% | 11/2636 |
| 2306.01048 | ✗ (AxionNeutron) | 0.85 | no_comparable_gt | — | — | — |
| 2008.08773 | ✓ | 0.92 | 100.0% | 0.589 | 28.2% | 20/543 |
| 2007.04990 | ✓ | 0.25 | gt_unusable | — | — | — |
| 2207.11968 | ✓ | 0.45 | 0.0% | ∞ | 0.0% | 30/154 |
| 2102.11740 | ✗ (None) | 0.15 | no_prediction | — | — | — |
| 0807.2926 | ✗ (AxionPhoton) | 0.85 | no_comparable_gt | — | — | — |
| 1508.02463 | ✓ | 0.75 | 88.4% | 15.251 | 0.0% | 5/43 |
| 1604.06800 | ✓ | 0.75 | 100.0% | 2.281 | 6.5% | 19/155 |
| 1606.07001 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 4/39 |
| 0809.4700 | ✓ | 0.85 | 57.1% | 5.334 | 0.0% | 7/35 |
| 2309.07995 | ✓ | 0.65 | 100.0% | 6.151 | 0.0% | 51/12 |
| 2006.07055 | ✓ | 0.65 | gt_unusable | — | — | — |
| 1508.01798 | ✓ | 0.75 | 100.0% | 0.949 | 16.0% | 37/50 |
| 1512.06165 | ✗ (ScalarBaryon) | 0.45 | no_comparable_gt | — | — | — |
| 2004.02733 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 6/34 |
| 2111.09892 | ✗ (AxionMass) | 0.85 | no_comparable_gt | — | — | — |
| 1401.6460 | ✓ | 0.65 | 50.0% | 6.338 | 0.0% | 5/4 |
| 2204.01454 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 0/44 |
| 2410.02218 | ✗ (AxionCPV) | 0.92 | no_comparable_gt | — | — | — |
| 1808.02340 | ✓ | 0.75 | 1.1% | 0.126 | 100.0% | 28/265 |
| 2311.16364 | ✓ | 0.75 | 100.0% | 2.318 | 0.8% | 30/121 |
| 1704.02297 | ✓ | 0.92 | 96.6% | 0.338 | 35.7% | 5/58 |
| 1707.07921 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 10/191 |
| 1806.00310 | ✓ | 0.95 | 0.0% | ∞ | 0.0% | 1/9 |
| 2007.03694 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 0/2 |
| 1911.11905 | ✓ | 0.85 | 24.2% | 0.481 | 43.5% | 4/256 |
| 1902.04246 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 0/61 |
| 1708.02111 | — | — | FAILED | — | — | — |
| 2006.09721 | ✓ | 0.75 | 100.0% | 1.449 | 7.4% | 31/148 |
| 1907.11485 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 23/111 |
| 2112.12116 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 5/67 |
| 2006.12431 | ✓ | 0.72 | 38.9% | 1.777 | 0.0% | 4/90 |
| 2207.11330 | ✗ (AxionPhoton) | 0.85 | no_comparable_gt | — | — | — |
| 2412.08699 | ✓ | 0.85 | gt_unusable | — | — | — |
| 1512.06746 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 1606.07494 | ✓ | 0.85 | gt_unusable | — | — | — |
| 1906.00967 | ✓ | 0.85 | gt_unusable | — | — | — |
| 2108.05368 | ✓ | 0.92 | gt_unusable | — | — | — |
| 1705.00676 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 1509.00026 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 2402.00741 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 1708.07521 | ✓ | 0.92 | gt_unusable | — | — | — |
| 1606.03145 | ✓ | 0.85 | gt_unusable | — | — | — |
| 2401.17253 | ✓ | 0.35 | gt_unusable | — | — | — |
| 1412.0789 | ✓ | 0.25 | no_extracted_points | — | — | — |
| 2206.11598 | ✓ | 0.65 | gt_unusable | — | — | — |
| 1902.04644 | ✓ | 0.65 | 100.0% | 1.453 | 0.0% | 4/73 |
| 2306.08039 | ✓ | 0.92 | 72.4% | 1.080 | 26.4% | 18/1158 |
| 2102.01448 | ✓ | 0.92 | 100.0% | 1.488 | 0.0% | 13/70 |
| 2209.03289 | ✓ | 0.92 | 46.4% | 0.076 | 94.1% | 5/110 |
| 2209.13588 | ✓ | 0.85 | 0.0% | ∞ | 0.0% | 22/251 |
| 1906.11844 | ✓ | 0.25 | no_extracted_points | — | — | — |
| hep-ph/0611223 | — | — | FAILED | — | — | — |
| 1810.12257 | ✓ | 0.95 | 100.0% | 0.304 | 49.6% | 2/3214 |
| 2102.06722 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 3/391 |
| 2404.12517 | ✓ | 0.95 | 99.6% | 0.321 | 45.6% | 7/284 |
| 0910.5914 | ✓ | 0.75 | 7.4% | 0.521 | 0.0% | 2/27 |
| 1804.05750 | ✓ | 0.75 | 100.0% | 0.147 | 84.1% | 34/145 |
| 1910.08638 | ✓ | 0.85 | 97.1% | 0.026 | 100.0% | 5/172 |
| 2504.07279 | ✓ | 0.85 | 84.6% | 0.572 | 1.0% | 15/234 |
| 1911.05772 | ✓ | 0.75 | 100.0% | 0.331 | 40.9% | 6/22 |
| 1901.00920 | ✓ | 0.85 | 99.1% | 0.878 | 0.9% | 15/117 |
| 1004.1313 | ✓ | 0.85 | 99.6% | 0.937 | 15.9% | 6/228 |
| 2008.05355 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 0/49 |
| 2302.10206 | ✓ | 0.85 | 90.0% | 1.223 | 0.0% | 11/10 |
| 2101.11290 | ✓ | 0.75 | 100.0% | 0.460 | 0.0% | 36/2 |
| 2002.08370 | — | — | FAILED | — | — | — |
| 2211.12699 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 9/75 |
| 2108.03316 | ✓ | 0.92 | 100.0% | 1.120 | 13.7% | 8/321 |
| 1709.00009 | ✓ | 0.85 | 0.0% | ∞ | 0.0% | 10/37 |
| 2007.13071 | ✓ | 0.85 | 0.0% | ∞ | 0.0% | 7/318 |
| 2009.09059 | ✓ | 0.92 | 3.5% | 0.304 | 50.0% | 2/114 |
| 2112.03439 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 29/2 |
| 2001.05102 | ✓ | 0.85 | 94.8% | 0.616 | 0.0% | 3/77 |
| 2008.10141 | ✓ | 0.85 | 98.2% | 1.099 | 0.0% | 20/222 |
| 2012.10764 | ✓ | 0.85 | 100.0% | 1.655 | 0.0% | 4/125 |
| 2206.08845 | ✓ | 0.92 | 99.1% | 1.784 | 0.0% | 6/2959 |
| 2207.13597 | ✓ | 0.75 | 100.0% | 5.241 | 0.0% | 18/49 |
| 2210.10961 | ✓ | 0.95 | 94.8% | 0.009 | 100.0% | 3/251 |
| 2312.11003 | ✓ | 0.92 | 91.2% | 2.097 | 0.0% | 29/181 |
| 2403.13390 | ✓ | 0.85 | 100.0% | 1.913 | 0.0% | 43/70 |
| 2402.12892 | ✓ | 0.85 | 100.0% | 2.824 | 0.0% | 17/362 |
| 2211.02902 | ✓ | 0.92 | 100.0% | 0.447 | 11.2% | 2/169 |
| 1705.02290 | ✓ | 0.92 | 99.8% | 0.869 | 17.9% | 31/436 |
| hep-ex/0702006 | — | — | FAILED | — | — | — |
| 1704.05189 | ✓ | 0.92 | 100.0% | 0.439 | 34.4% | 38/61 |
| 2411.13701 | ✓ | 0.75 | 92.6% | 8.717 | 0.0% | 10/81 |
| 2109.03261 | ✓ | 0.95 | 31.0% | 0.006 | 100.0% | 8/29 |
| 1304.0989 | ✓ | 0.92 | 44.8% | 0.030 | 100.0% | 4/29 |
| 1703.07354 | ✓ | 0.95 | 38.6% | 0.151 | 100.0% | 2/44 |
| 1907.05475 | ✓ | 0.85 | 94.1% | 0.238 | 56.2% | 40/17 |
| 2104.12772 | ✓ | 0.85 | 22.8% | 1.911 | 0.0% | 3/79 |
| 2407.10618 | ✓ | 0.75 | 100.0% | 2.244 | 7.5% | 10/2612 |
| 2303.03594 | ✓ | 0.85 | 68.5% | 0.691 | 23.0% | 32/89 |
| 2311.05476 | ✓ | 0.72 | 65.3% | 13.446 | 0.0% | 8/121 |
| 2201.09890 | ✓ | 0.85 | 95.5% | 0.587 | 19.0% | 4/22 |
| 1110.2895 | ✓ | 0.65 | 100.0% | 11.796 | 0.0% | 13/42 |
| 2412.02232 | ✓ | 0.85 | 28.2% | 0.940 | 25.7% | 3/124 |
| 2504.07559 | ✓ | 0.92 | 58.1% | 0.119 | 79.1% | 3/198 |
| 2404.17333 | ✓ | 0.92 | 96.0% | 0.262 | 54.2% | 5/25 |
| 2405.08059 | ✓ | 0.75 | 87.5% | 2.661 | 0.0% | 2/40 |
| 2211.03414 | ✓ | 0.72 | 84.4% | 0.332 | 43.5% | 30/109 |
| 1603.06978 | ✓ | 0.92 | 10.5% | 0.231 | 77.1% | 3/333 |
| 2305.10327 | ✓ | 0.75 | 100.0% | 0.732 | 17.6% | 4/51 |
| 2305.01002 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 5/41 |
| 2208.13794 | ✓ | 0.85 | 94.4% | 0.541 | 29.8% | 6/89 |
| 2312.11608 | ✗ (AxionMass) | 0.25 | no_comparable_gt | — | — | — |
| 2501.17119 | ✓ | 0.75 | 100.0% | 1.301 | 0.0% | 5/799 |
| 1406.6053 | ✓ | 0.75 | 0.0% | ∞ | 0.0% | 0/27 |
| 2110.14406 | ✓ | 0.85 | 100.0% | 1.350 | 0.0% | 24/2 |
| 2203.04332 | ✓ | 0.92 | 45.9% | 0.030 | 88.2% | 4/148 |
| 1610.02580 | ✓ | 0.92 | 54.5% | 0.048 | 98.5% | 10/123 |
| 2008.01853 | ✓ | 0.92 | 46.8% | 1.961 | 0.0% | 4/111 |
| 2409.08998 | ✓ | 0.92 | 0.0% | ∞ | 0.0% | 0/324 |
| 1311.3148 | ✓ | 0.92 | 59.1% | 0.058 | 100.0% | 3/22 |
| 2301.06560 | ✓ | 0.72 | 68.0% | 13.450 | 0.0% | 5/97 |
| 2412.02543 | ✓ | 0.78 | 99.0% | 0.409 | 41.4% | 11/100 |
| 2209.06299 | ✓ | 0.75 | 100.0% | 2.870 | 0.0% | 9/45 |
| 2310.15395 | ✓ | 0.92 | 79.0% | 0.139 | 84.0% | 3/300 |
| 2503.11753 | ✓ | 0.75 | 95.9% | 2.639 | 0.0% | 10/1844 |
| 1509.00476 | ✓ | 0.65 | 0.0% | ∞ | 0.0% | 5/21 |
| 2307.01365 | ✓ | 0.95 | 57.4% | 0.132 | 100.0% | 3/94 |
| 2111.08025 | ✗ (None) | 0.25 | no_prediction | — | — | — |
| 2412.03660 | ✓ | 0.85 | 67.0% | 0.368 | 45.5% | 8/115 |
| 2409.11777 | ✓ | 0.95 | 100.0% | 0.182 | 75.6% | 4/86 |
| 2401.07798 | ✓ | 0.92 | 100.0% | 0.416 | 38.6% | 12/44 |
| 1811.10997 | ✓ | 0.85 | 97.9% | 0.177 | 68.8% | 3/144 |

## Breakdown by Extraction Source

Median residual is over papers with mass-range overlap; zero-overlap papers are listed separately.

| Source | Papers | Compared | Zero-overlap | Med. Resid. | ≤0.3 dex |
|--------|--------|----------|--------------|-------------|----------|
| table | 3 | 2 | 0 | 0.144 dex | 60.4% |
| figure_vision | 49 | 36 | 5 | 1.350 dex | 15.3% |
| text | 136 | 115 | 25 | 0.588 dex | 34.1% |

## Breakdown by Difficulty

> Difficulty is a placeholder label for the repo-sourced pool (nearly all `medium`); this table is informational only.

| Difficulty | Papers | Coupling Acc. | Med. Resid. | ≤0.3 dex |
|------------|--------|---------------|-------------|----------|
| easy | 18 | 88.9% | 0.520 dex | 25.0% |
| medium | 135 | 90.1% | 0.602 dex | 32.7% |
| hard | 58 | 91.4% | 0.947 dex | 23.9% |

## Confidence Calibration

| Bin | N | Mean Conf. | Actual Acc. | Gap |
|-----|---|------------|-------------|-----|
| [0.4–0.6) | 1 | 45.0% | 0.0% | +0.45 |
| [0.6–0.8) | 59 | 72.3% | 5.1% | +0.67 |
| [0.8–1.0) | 94 | 89.5% | 21.3% | +0.68 |

> **Interpretation**: Gap > 0 means the pipeline is overconfident; Gap < 0 means underconfident.

## Methodology

### Curve selection (what each extraction is compared against)
- A paper usually produces several repo curves (one per coupling). The single extraction is compared **only** against the GT curve whose coupling matches the extracted coupling type (taken from the data file's `limit_data/<dir>/`).
- Papers whose extracted coupling has no matching GT curve, or whose GT curve has <2 usable points, are reported under Curve-Comparison Coverage and excluded from residual statistics — they do not measure extraction quality.

### Caveats on the residual floor
- The ground truth `g(x_i)` is the **upstream-curated** repo curve (itself digitised and rescaled from the same papers), not the paper's raw numbers. A perfect extraction still shows a nonzero residual equal to the upstream digitisation/convention gap, so the ~0.5–0.7 dex typical residual is an upper bound on true extraction error.
- `is_new_limit`, `is_projection`, `data_source`, and `difficulty` are placeholder labels in the repo-sourced pool and are not scored (shown as N/A / informational).

### Interpolation metric (primary)
1. Filter boundary-closure sentinel points (coupling >= 1e-2) from both extracted and GT data
2. Build `scipy.interpolate.interp1d` from extracted points in log10(mass) → log10(coupling) space
3. Evaluate the interpolation at each ground-truth mass value
4. Compute residual = |log10(g_interpolated) - log10(g_ground_truth)| at each GT point
5. Only GT points inside the extracted mass range are used (no extrapolation)

**Key statistics:**
- **Interpolation coverage**: fraction of GT points inside the extracted mass range
- **Median/P90 residual**: summary of coupling errors in dex (0.3 dex ≈ factor 2)
- **Fraction within threshold**: what % of GT points have residual below 0.1/0.3/0.5/1.0 dex

When multiple extracted points share the same mass, the strongest constraint (lowest coupling) is kept.

### Confidence calibration
- A paper is "accurate" if median residual < 0.3 dex AND interpolation coverage ≥ 50%
- Papers binned by extraction_confidence; actual accuracy computed per bin
- Perfect calibration: actual accuracy = mean confidence in each bin
