# Subset before/after comparison — issues #550 & #561

Subset: 82 papers. Before snapshot: `evaluation/eval_runs/before` (82 loaded). After snapshot: `evaluation/eval_runs/after` (82 loaded).

## Headline

| Metric | Before | After | Δ |
|---|---|---|---|
| Overall median residual (dex) | 0.485 | 0.842 | +0.357 ⚠️ |
| Zero-overlap papers | 14 | 32 | +18 ⚠️ |
| Mean interp. coverage | 63.9% | 35.4% | -0.285 ⚠️ |
| Papers compared | 45 | 24 | -21 |

## Zero-overlap causes

| Cause | Before | After |
|---|---|---|
| too_few_points | 7 | 7 |
| unit_offset | 0 | 12 |
| wrong_window | 7 | 13 |

## Per-source breakdown (#550 scoreboard)

| Source | Papers (B/A) | Compared (B/A) | Zero-ovl (B/A) | Med.Resid B→A | ≤0.3dex B→A |
|---|---|---|---|---|---|
| figure_vision | 44/43 | 32/16 | 1/15 | 0.642 → 1.002 | 33.3% → 25.2% |
| none | 11/13 | 0/0 | 0/0 | — → — | — → — |
| text | 27/26 | 13/8 | 13/17 | 0.375 → 0.465 | 47.8% → 45.0% |

## Determinism (#561) — coupling-scale spread across repeats

Scale = median(log10 coupling) per run; lower std = more stable. Noise floor = 0.32 dex.

| arXiv | Before std (dex) | After std (dex) | Before range | After range |
|---|---|---|---|---|
| 1709.00009 | 1.000 | 0.480 | 2.000 | 1.111 |
| 1907.11485 | 0.115 | 0.000 | 0.272 | 0.000 |
| 2007.13071 | 0.055 | 0.000 | 0.116 | 0.000 |
| 2111.06883 | 1.148 | 0.471 | 2.680 | 1.000 |
| 2207.11968 | 0.116 | 0.000 | 0.280 | 0.000 |
| 2209.13588 | 0.260 | 0.188 | 0.637 | 0.460 |
| 2211.12699 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2401.16747 | 0.263 | 0.138 | 0.637 | 0.293 |
| 2410.10363 | 0.357 | 0.793 | 0.810 | 1.943 |

Mean before std: 0.368 dex
  Mean after std: 0.230 dex

## Per-paper detail

| arXiv | Source B→A | Status B→A | Med.Resid B→A | Cov B→A |
|---|---|---|---|---|
| 0807.2926 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1110.2895 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1207.3275 | figure_vision→figure_vision | compared→compared | 3.403→2.134 | 50.0%→90.0% |
| 1310.8098 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1403.1290 | figure_vision→figure_vision | compared→compared | 3.699→4.951 | 65.8%→65.8% |
| 1406.6053 | text→text | zero_overlap→compared | ∞→0.003 | 0.0%→3.7% |
| 1410.7267 | figure_vision→figure_vision | compared→zero_overlap | 9.588→∞ | 100.0%→0.0% |
| 1506.08082 | figure_vision→figure_vision | compared→compared | 0.043→1.030 | 91.7%→91.7% |
| 1508.01798 | figure_vision→figure_vision | compared→compared | 2.402→0.974 | 100.0%→100.0% |
| 1509.00476 | figure_vision→figure_vision | compared→zero_overlap | 0.555→∞ | 90.5%→0.0% |
| 1604.06800 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1606.03145 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1606.07001 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1607.06083 | figure_vision→None | zero_overlap→error | ∞→— | 0.0%→— |
| 1607.07327 | figure_vision→text | compared→zero_overlap | 0.180→∞ | 88.6%→0.0% |
| 1704.05189 | figure_vision→text | compared→zero_overlap | 0.730→∞ | 100.0%→0.0% |
| 1705.02290 | text→text | compared→zero_overlap | 0.066→∞ | 21.8%→0.0% |
| 1706.00209 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1707.07921 | text→text | compared→zero_overlap | 0.817→∞ | 35.6%→0.0% |
| 1708.06367 | text→figure_vision | compared→compared | 4.763→2.514 | 61.1%→100.0% |
| 1709.00009 | figure_vision→figure_vision | compared→zero_overlap | 2.025→∞ | 83.8%→0.0% |
| 1711.08999 | none→none | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1712.00483 | text→figure_vision | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1804.05750 | text→text | zero_overlap→compared | ∞→0.252 | 0.0%→99.3% |
| 1806.00310 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1808.02340 | text→figure_vision | compared→zero_overlap | 1.003→∞ | 100.0%→0.0% |
| 1810.04602 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1902.04246 | text→text | compared→compared | 0.234→5.790 | 91.8%→91.8% |
| 1905.13650 | figure_vision→figure_vision | compared→zero_overlap | 0.342→∞ | 8.2%→0.0% |
| 1907.05475 | figure_vision→figure_vision | compared→compared | 0.073→1.065 | 94.1%→94.1% |
| 1907.11485 | figure_vision→figure_vision | compared→zero_overlap | 2.290→∞ | 94.6%→0.0% |
| 2004.02733 | text→text | compared→compared | 0.547→1.000 | 94.1%→5.9% |
| 2006.07055 | figure_vision→figure_vision | compared→zero_overlap | 3.949→∞ | 59.5%→0.0% |
| 2006.09721 | text→none | zero_overlap→no_extracted_points | ∞→— | 0.0%→— |
| 2007.03694 | text→none | zero_overlap→no_extracted_points | ∞→— | 0.0%→— |
| 2007.04899 | figure_vision→text | compared→zero_overlap | 2.437→∞ | 57.3%→0.0% |
| 2007.04990 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2007.13071 | text→text | compared→zero_overlap | 0.245→∞ | 99.4%→0.0% |
| 2008.05355 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2008.10141 | figure_vision→figure_vision | compared→compared | 0.118→0.710 | 97.3%→99.1% |
| 2101.01241 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2101.11290 | text→text | compared→zero_overlap | 0.000→∞ | 50.0%→0.0% |
| 2102.06722 | text→text | compared→zero_overlap | 0.375→∞ | 98.7%→0.0% |
| 2102.08764 | text→figure_vision | zero_overlap→compared | ∞→10.455 | 0.0%→100.0% |
| 2110.03679 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.10262 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.13636 | figure_vision→figure_vision | compared→compared | 0.228→0.103 | 95.4%→83.1% |
| 2110.14406 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2111.06883 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2111.09892 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2112.03439 | none→none | no_prediction→no_prediction | —→— | —→— |
| 2112.12116 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2202.08858 | figure_vision→figure_vision | compared→zero_overlap | 0.454→∞ | 100.0%→0.0% |
| 2203.12152 | figure_vision→figure_vision | compared→compared | 0.462→0.141 | 100.0%→100.0% |
| 2204.01454 | figure_vision→text | compared→zero_overlap | 3.816→∞ | 90.9%→0.0% |
| 2207.11968 | figure_vision→figure_vision | compared→zero_overlap | 2.115→∞ | 96.1%→0.0% |
| 2207.13597 | text→text | compared→compared | 0.209→0.679 | 93.9%→93.9% |
| 2208.06519 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2208.07293 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2209.12901 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2209.13588 | text→text | compared→compared | 0.571→1.351 | 99.6%→99.6% |
| 2211.03414 | figure_vision→figure_vision | compared→compared | 0.089→0.075 | 88.1%→86.2% |
| 2211.08439 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2211.12699 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2212.01139 | figure_vision→figure_vision | compared→compared | 1.262→2.281 | 97.8%→39.6% |
| 2303.03594 | figure_vision→figure_vision | compared→zero_overlap | 0.117→∞ | 84.3%→0.0% |
| 2305.01002 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2307.03878 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2308.06339 | text→text | compared→zero_overlap | 0.145→∞ | 96.1%→0.0% |
| 2309.07995 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2311.16364 | figure_vision→figure_vision | compared→zero_overlap | 0.908→∞ | 100.0%→0.0% |
| 2312.11003 | figure_vision→text | compared→compared | 0.072→0.007 | 96.1%→91.2% |
| 2401.16747 | text→text | compared→zero_overlap | 0.430→∞ | 92.4%→0.0% |
| 2401.17253 | none→figure_vision | no_extracted_points→no_comparable_gt | —→— | —→— |
| 2401.18076 | figure_vision→figure_vision | compared→zero_overlap | 20.733→∞ | 100.0%→0.0% |
| 2402.12892 | figure_vision→figure_vision | compared→compared | 0.853→0.102 | 100.0%→100.0% |
| 2403.03004 | figure_vision→figure_vision | compared→zero_overlap | 0.814→∞ | 100.0%→0.0% |
| 2403.13390 | figure_vision→figure_vision | compared→compared | 0.052→0.524 | 97.1%→100.0% |
| 2407.18586 | figure_vision→figure_vision | compared→compared | 0.054→0.369 | 95.8%→100.0% |
| 2409.08998 | figure_vision→text | compared→compared | 0.356→0.120 | 100.0%→97.2% |
| 2410.10363 | figure_vision→figure_vision | compared→zero_overlap | 0.485→∞ | 13.4%→0.0% |
| 2410.19902 | text→figure_vision | zero_overlap→compared | ∞→2.131 | 0.0%→50.0% |
