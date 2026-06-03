# Subset before/after comparison — issues #550 & #561

Subset: 82 papers. Before snapshot: `evaluation/eval_runs/before` (82 loaded). After snapshot: `evaluation/eval_runs/after_561fix` (82 loaded).

## Headline

| Metric | Before | After | Δ |
|---|---|---|---|
| Overall median residual (dex) | 0.485 | 0.435 | -0.051 ✅ |
| Zero-overlap papers | 14 | 14 | +0 ▪️ |
| Mean interp. coverage | 63.9% | 63.6% | -0.003 ⚠️ |
| Papers compared | 45 | 44 | -1 |

## Zero-overlap causes

| Cause | Before | After |
|---|---|---|
| too_few_points | 7 | 7 |
| unit_offset | 0 | 2 |
| wrong_window | 7 | 5 |

## Per-source breakdown (#550 scoreboard)

| Source | Papers (B/A) | Compared (B/A) | Zero-ovl (B/A) | Med.Resid B→A | ≤0.3dex B→A |
|---|---|---|---|---|---|
| figure_vision | 44/40 | 32/28 | 1/4 | 0.642 → 0.585 | 33.3% → 37.6% |
| none | 11/14 | 0/0 | 0/0 | — → — | — → — |
| text | 27/28 | 13/16 | 13/10 | 0.375 → 0.223 | 47.8% → 56.3% |

## Determinism (#561) — coupling-scale spread across repeats

Scale = median(log10 coupling) per run; lower std = more stable. Noise floor = 0.32 dex.

| arXiv | Before std (dex) | After std (dex) | Before range | After range |
|---|---|---|---|---|
| 1709.00009 | 1.000 | 1.621 | 2.000 | 3.562 |
| 1907.11485 | 0.115 | 0.070 | 0.272 | 0.163 |
| 2007.13071 | 0.055 | 0.000 | 0.116 | 0.000 |
| 2111.06883 | 1.148 | 0.107 | 2.680 | 0.254 |
| 2207.11968 | 0.116 | 0.036 | 0.280 | 0.076 |
| 2209.13588 | 0.260 | 0.523 | 0.637 | 1.228 |
| 2211.12699 | 0.000 | 0.123 | 0.000 | 0.261 |
| 2401.16747 | 0.263 | 0.060 | 0.637 | 0.140 |
| 2410.10363 | 0.357 | 0.455 | 0.810 | 0.991 |

Mean before std: 0.368 dex
  Mean after std: 0.333 dex

## Per-paper detail

| arXiv | Source B→A | Status B→A | Med.Resid B→A | Cov B→A |
|---|---|---|---|---|
| 0807.2926 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1110.2895 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1207.3275 | figure_vision→figure_vision | compared→compared | 3.403→5.968 | 50.0%→90.0% |
| 1310.8098 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1403.1290 | figure_vision→figure_vision | compared→compared | 3.699→8.400 | 65.8%→65.8% |
| 1406.6053 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1410.7267 | figure_vision→figure_vision | compared→zero_overlap | 9.588→∞ | 100.0%→0.0% |
| 1506.08082 | figure_vision→figure_vision | compared→compared | 0.043→0.041 | 91.7%→91.7% |
| 1508.01798 | figure_vision→figure_vision | compared→compared | 2.402→1.865 | 100.0%→100.0% |
| 1509.00476 | figure_vision→figure_vision | compared→compared | 0.555→0.609 | 90.5%→90.5% |
| 1604.06800 | figure_vision→figure_vision | no_comparable_gt→compared | —→0.106 | —→100.0% |
| 1606.03145 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1606.07001 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1607.06083 | figure_vision→figure_vision | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1607.07327 | figure_vision→text | compared→compared | 0.180→0.195 | 88.6%→100.0% |
| 1704.05189 | figure_vision→text | compared→compared | 0.730→0.930 | 100.0%→72.1% |
| 1705.02290 | text→text | compared→compared | 0.066→0.066 | 21.8%→21.8% |
| 1706.00209 | figure_vision→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1707.07921 | text→text | compared→zero_overlap | 0.817→∞ | 35.6%→0.0% |
| 1708.06367 | text→text | compared→compared | 4.763→4.534 | 61.1%→61.1% |
| 1709.00009 | figure_vision→figure_vision | compared→compared | 2.025→2.995 | 83.8%→100.0% |
| 1711.08999 | none→none | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1712.00483 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1804.05750 | text→text | zero_overlap→compared | ∞→0.186 | 0.0%→100.0% |
| 1806.00310 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1808.02340 | text→figure_vision | compared→compared | 1.003→0.087 | 100.0%→1.1% |
| 1810.04602 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1902.04246 | text→text | compared→compared | 0.234→5.790 | 91.8%→91.8% |
| 1905.13650 | figure_vision→figure_vision | compared→compared | 0.342→0.462 | 8.2%→96.3% |
| 1907.05475 | figure_vision→figure_vision | compared→compared | 0.073→0.064 | 94.1%→88.2% |
| 1907.11485 | figure_vision→figure_vision | compared→compared | 2.290→2.226 | 94.6%→94.6% |
| 2004.02733 | text→text | compared→compared | 0.547→0.547 | 94.1%→94.1% |
| 2006.07055 | figure_vision→figure_vision | compared→compared | 3.949→19.208 | 59.5%→93.7% |
| 2006.09721 | text→none | zero_overlap→no_extracted_points | ∞→— | 0.0%→— |
| 2007.03694 | text→none | zero_overlap→no_extracted_points | ∞→— | 0.0%→— |
| 2007.04899 | figure_vision→figure_vision | compared→compared | 2.437→3.052 | 57.3%→57.3% |
| 2007.04990 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2007.13071 | text→text | compared→compared | 0.245→0.245 | 99.4%→99.4% |
| 2008.05355 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2008.10141 | figure_vision→text | compared→compared | 0.118→0.286 | 97.3%→98.2% |
| 2101.01241 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2101.11290 | text→text | compared→compared | 0.000→0.000 | 50.0%→50.0% |
| 2102.06722 | text→text | compared→compared | 0.375→0.195 | 98.7%→90.3% |
| 2102.08764 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.03679 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.10262 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.13636 | figure_vision→figure_vision | compared→compared | 0.228→0.211 | 95.4%→100.0% |
| 2110.14406 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2111.06883 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2111.09892 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2112.03439 | none→none | no_prediction→no_prediction | —→— | —→— |
| 2112.12116 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2202.08858 | figure_vision→figure_vision | compared→compared | 0.454→0.560 | 100.0%→100.0% |
| 2203.12152 | figure_vision→figure_vision | compared→compared | 0.462→0.141 | 100.0%→100.0% |
| 2204.01454 | figure_vision→figure_vision | compared→compared | 3.816→12.209 | 90.9%→65.9% |
| 2207.11968 | figure_vision→figure_vision | compared→compared | 2.115→1.005 | 96.1%→96.1% |
| 2207.13597 | text→text | compared→compared | 0.209→0.202 | 93.9%→93.9% |
| 2208.06519 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2208.07293 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2209.12901 | figure_vision→none | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2209.13588 | text→text | compared→compared | 0.571→0.988 | 99.6%→99.6% |
| 2211.03414 | figure_vision→figure_vision | compared→compared | 0.089→0.063 | 88.1%→88.1% |
| 2211.08439 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2211.12699 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2212.01139 | figure_vision→figure_vision | compared→compared | 1.262→1.146 | 97.8%→98.5% |
| 2303.03594 | figure_vision→figure_vision | compared→compared | 0.117→0.316 | 84.3%→41.6% |
| 2305.01002 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2307.03878 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2308.06339 | text→text | compared→compared | 0.145→0.138 | 96.1%→96.1% |
| 2309.07995 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2311.16364 | figure_vision→figure_vision | compared→compared | 0.908→0.754 | 100.0%→100.0% |
| 2312.11003 | figure_vision→figure_vision | compared→zero_overlap | 0.072→∞ | 96.1%→0.0% |
| 2401.16747 | text→text | compared→compared | 0.430→0.648 | 92.4%→92.4% |
| 2401.17253 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2401.18076 | figure_vision→figure_vision | compared→compared | 20.733→20.369 | 100.0%→72.4% |
| 2402.12892 | figure_vision→figure_vision | compared→compared | 0.853→0.197 | 100.0%→100.0% |
| 2403.03004 | figure_vision→figure_vision | compared→compared | 0.814→1.001 | 100.0%→99.0% |
| 2403.13390 | figure_vision→figure_vision | compared→compared | 0.052→0.247 | 97.1%→97.1% |
| 2407.18586 | figure_vision→figure_vision | compared→compared | 0.054→0.051 | 95.8%→95.8% |
| 2409.08998 | figure_vision→text | compared→compared | 0.356→0.162 | 100.0%→97.2% |
| 2410.10363 | figure_vision→figure_vision | compared→compared | 0.485→0.407 | 13.4%→8.2% |
| 2410.19902 | text→figure_vision | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
