# Subset before/after comparison — issues #550 & #561

Subset: 80 papers. Before snapshot: `evaluation/eval_runs/baseline` (80 loaded). After snapshot: `evaluation/eval_runs/after_roadmap` (80 loaded).

## Headline

| Metric | Before | After | Δ |
|---|---|---|---|
| Overall median residual (dex) | 0.530 | 0.391 | -0.139 ✅ |
| Zero-overlap papers | 15 | 9 | -6 ✅ |
| Mean interp. coverage | 56.6% | 72.0% | +0.154 ✅ |
| Papers compared | 43 | 54 | +11 |

## Zero-overlap causes

| Cause | Before | After |
|---|---|---|
| too_few_points | 7 | 4 |
| unit_offset | 2 | 0 |
| wrong_window | 6 | 5 |

## Per-source breakdown (#550 scoreboard)

| Source | Papers (B/A) | Compared (B/A) | Zero-ovl (B/A) | Med.Resid B→A | ≤0.3dex B→A |
|---|---|---|---|---|---|
| figure_vision | 23/42 | 15/32 | 2/1 | 1.876 → 0.614 | 17.7% → 33.0% |
| none | 10/4 | 0/0 | 0/0 | — → — | — → — |
| table | 2/1 | 0/0 | 2/1 | — → — | — → — |
| text | 45/33 | 28/22 | 11/7 | 0.276 → 0.264 | 48.3% → 49.0% |

## Per-paper detail

| arXiv | Source B→A | Status B→A | Med.Resid B→A | Cov B→A |
|---|---|---|---|---|
| 0807.2926 | text→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1110.2895 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1207.3275 | text→figure_vision | zero_overlap→compared | ∞→0.421 | 0.0%→50.0% |
| 1310.8098 | text→figure_vision | zero_overlap→compared | ∞→0.544 | 0.0%→97.7% |
| 1403.1290 | figure_vision→figure_vision | compared→compared | 3.225→3.160 | 65.8%→65.8% |
| 1406.6053 | text→text | compared→zero_overlap | 0.003→∞ | 3.7%→0.0% |
| 1410.7267 | figure_vision→figure_vision | convention_mismatch→convention_mismatch | —→— | —→— |
| 1506.08082 | text→figure_vision | zero_overlap→compared | ∞→0.073 | 0.0%→91.7% |
| 1508.01798 | figure_vision→figure_vision | compared→compared | 1.876→1.995 | 100.0%→100.0% |
| 1509.00476 | figure_vision→figure_vision | compared→compared | 2.121→1.664 | 90.5%→90.5% |
| 1604.06800 | text→text | compared→compared | 1.088→1.011 | 72.9%→72.9% |
| 1606.03145 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 1606.07001 | none→figure_vision | no_extracted_points→compared | —→1.968 | —→97.4% |
| 1607.06083 | figure_vision→figure_vision | zero_overlap→compared | ∞→1.303 | 0.0%→50.0% |
| 1607.07327 | text→text | compared→compared | 0.210→0.195 | 94.3%→100.0% |
| 1704.05189 | figure_vision→figure_vision | compared→compared | 0.703→0.664 | 68.9%→96.7% |
| 1705.02290 | text→text | compared→compared | 0.066→0.066 | 21.8%→36.2% |
| 1706.00209 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1707.07921 | text→text | compared→compared | 0.646→0.518 | 96.7%→50.0% |
| 1708.06367 | text→figure_vision | compared→compared | 4.173→3.703 | 75.0%→100.0% |
| 1709.00009 | figure_vision→figure_vision | compared→compared | 2.255→1.617 | 100.0%→100.0% |
| 1711.08999 | none→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 1712.00483 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1804.05750 | text→text | compared→zero_overlap | 0.221→∞ | 100.0%→0.0% |
| 1806.00310 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 1808.02340 | figure_vision→text | compared→zero_overlap | 0.913→∞ | 1.1%→0.0% |
| 1810.04602 | none→figure_vision | no_extracted_points→compared | —→0.323 | —→96.4% |
| 1902.04246 | text→figure_vision | compared→compared | 6.228→0.831 | 91.8%→62.3% |
| 1905.13650 | figure_vision→figure_vision | compared→compared | 6.790→0.364 | 87.2%→100.0% |
| 1907.05475 | text→figure_vision | compared→compared | 0.054→0.059 | 23.5%→94.1% |
| 1907.11485 | table→figure_vision | zero_overlap→compared | ∞→2.386 | 0.0%→94.6% |
| 2004.02733 | text→text | compared→compared | 4.612→0.246 | 94.1%→94.1% |
| 2006.07055 | figure_vision→figure_vision | convention_mismatch→convention_mismatch | —→— | —→— |
| 2006.09721 | none→figure_vision | no_extracted_points→compared | —→1.075 | —→100.0% |
| 2007.03694 | table→table | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2007.04899 | text→text | compared→compared | 2.966→3.216 | 5.0%→4.4% |
| 2007.04990 | none→none | no_extracted_points→no_extracted_points | —→— | —→— |
| 2007.13071 | text→text | compared→compared | 0.245→0.245 | 99.4%→99.4% |
| 2008.05355 | text→text | zero_overlap→compared | ∞→0.340 | 0.0%→98.0% |
| 2008.10141 | text→text | compared→compared | 0.306→0.117 | 98.2%→98.2% |
| 2101.01241 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2101.11290 | text→text | compared→compared | 0.000→0.000 | 50.0%→50.0% |
| 2102.06722 | text→text | compared→compared | 0.308→0.128 | 98.7%→90.3% |
| 2102.08764 | text→text | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
| 2110.03679 | text→figure_vision | zero_overlap→compared | ∞→0.056 | 0.0%→99.2% |
| 2110.10262 | text→text | compared→compared | 0.530→0.530 | 94.3%→94.3% |
| 2110.13636 | text→figure_vision | compared→compared | 0.206→0.181 | 80.0%→100.0% |
| 2110.14406 | text→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2111.06883 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2112.03439 | none→none | no_prediction→no_prediction | —→— | —→— |
| 2112.12116 | none→figure_vision | no_extracted_points→compared | —→3.555 | —→97.0% |
| 2202.08858 | figure_vision→figure_vision | compared→compared | 0.335→0.563 | 100.0%→100.0% |
| 2203.12152 | text→text | compared→compared | 0.188→0.188 | 98.4%→98.4% |
| 2204.01454 | figure_vision→figure_vision | compared→compared | 13.090→14.222 | 56.8%→93.2% |
| 2207.11968 | figure_vision→figure_vision | compared→compared | 2.012→2.153 | 92.2%→98.7% |
| 2207.13597 | text→text | compared→compared | 0.327→0.327 | 93.9%→93.9% |
| 2208.06519 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2208.07293 | none→text | no_extracted_points→zero_overlap | —→∞ | —→0.0% |
| 2209.12901 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2209.13588 | text→text | compared→compared | 4.419→0.981 | 99.6%→99.6% |
| 2211.03414 | text→figure_vision | compared→compared | 0.115→0.099 | 72.5%→86.2% |
| 2211.08439 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2211.12699 | text→figure_vision | zero_overlap→compared | ∞→0.105 | 0.0%→100.0% |
| 2212.01139 | text→figure_vision | compared→compared | 1.465→1.151 | 73.9%→98.5% |
| 2303.03594 | text→text | compared→compared | 0.459→0.459 | 25.8%→25.8% |
| 2305.01002 | text→text | zero_overlap→compared | ∞→0.892 | 0.0%→14.6% |
| 2307.03878 | text→text | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2308.06339 | text→text | compared→compared | 0.121→0.282 | 96.1%→96.1% |
| 2309.07995 | figure_vision→figure_vision | no_comparable_gt→no_comparable_gt | —→— | —→— |
| 2311.16364 | text→figure_vision | zero_overlap→compared | ∞→0.344 | 0.0%→98.3% |
| 2312.11003 | figure_vision→text | compared→compared | 0.053→0.394 | 99.4%→91.2% |
| 2401.16747 | text→text | compared→compared | 0.577→0.242 | 92.4%→97.0% |
| 2401.18076 | figure_vision→figure_vision | compared→compared | 18.928→2.282 | 58.6%→65.5% |
| 2402.12892 | figure_vision→figure_vision | compared→compared | 0.077→0.081 | 100.0%→100.0% |
| 2403.03004 | figure_vision→figure_vision | compared→compared | 1.023→0.200 | 100.0%→100.0% |
| 2403.13390 | text→text | compared→compared | 0.013→0.013 | 98.6%→98.6% |
| 2407.18586 | text→text | compared→compared | 0.043→0.043 | 100.0%→100.0% |
| 2409.08998 | text→figure_vision | compared→compared | 0.149→0.098 | 97.2%→98.5% |
| 2410.10363 | figure_vision→figure_vision | compared→compared | 1.457→0.389 | 13.4%→8.2% |
| 2410.19902 | figure_vision→figure_vision | zero_overlap→zero_overlap | ∞→∞ | 0.0%→0.0% |
