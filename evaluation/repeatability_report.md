# Extraction Repeatability & Benchmark Noise Floor (issue #545)

Extraction is LLM-based and non-deterministic. This study re-runs the **same** extraction path used by `evaluation/evaluate.py` (5 repeats per paper) on a representative subset and measures the run-to-run spread of the primary metric (median interpolation residual, in dex) so that real metric changes can be told apart from noise.

## Benchmark noise floor

**NOISE FLOOR = 0.320 dex.** A change in a paper's headline median residual smaller than this is within run-to-run LLM variance and should NOT be read as a real improvement or regression.

Basis and supporting spread statistics:

- Derivation: 90th percentile of per-paper median-residual std, well-behaved papers only
- Well-behaved papers contributing to the floor: 5
- Eligible papers (>=2 finite medians): 13
- Median per-paper std across repeats: 0.000 dex
- Max per-paper std across repeats: 0.386 dex
- Median per-paper min-max range: 0.000 dex
- Max per-paper min-max range: 0.900 dex

## Scale-instability outliers (separate failure mode)

These papers have a mean median-residual above 1.5 dex — i.e. the extracted curve lands many orders of magnitude off the ground truth, and the offset varies run-to-run. This is NOT the small run-to-run jitter the noise floor bounds; it is a coupling-value SCALE error (typically the extractor's run-to-run order-of-magnitude auto-correction). They are excluded from the noise floor and flagged here as a known instability:

- 1708.06367: mean median residual 5.910 dex, std 0.748 dex, range 4.90–6.72 dex
- 0809.4700: mean median residual 5.325 dex, std 0.019 dex, range 5.29–5.33 dex
- 1403.1290: mean median residual 17.370 dex, std 5.767 dex, range 8.49–22.88 dex
- 1410.7267: mean median residual 10.498 dex, std 5.083 dex, range 7.51–16.37 dex
- 2008.08773: mean median residual 1.630 dex, std 0.765 dex, range 1.07–2.81 dex
- 1401.6460: mean median residual 6.213 dex, std 1.529 dex, range 3.52–7.09 dex
- 1902.04644: mean median residual 2.122 dex, std 0.782 dex, range 1.67–3.03 dex
- 0801.1527: mean median residual 4.488 dex, std 1.321 dex, range 3.36–6.30 dex

## Aggregate stability

- Papers studied: 20
- Repeats per paper: 5
- Coupling-type classification stable across all repeats: 18/20
- Papers with UNSTABLE coupling classification: 1401.6460, 1902.04644
- Extracted point-count std (median across papers): 1.791; max 33.410
- Extracted point-count min-max range (median across papers): 4.000; max 68.000

## Per-paper detail

| arXiv | coupling(s) | runs | finite | inf | err | median resid mean | std | IQR | min-max | pts std | coupling stable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0801.1527 | DarkPhoton | 5 | 5 | 0 | 0 | 4.488 | 1.321 | 2.011 | 3.36-6.30 | 2.17 | yes |
| 0807.2926 | AxionElectron | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 9.93 | yes |
| 0809.4700 | AxionNeutron | 5 | 5 | 0 | 0 | 5.325 | 0.019 | 0.000 | 5.29-5.33 | 0.45 | yes |
| 0910.5914 | AxionPhoton | 5 | 5 | 0 | 0 | 0.454 | 0.221 | 0.125 | 0.07-0.61 | 0.00 | yes |
| 1202.5851 | AxionMass | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.89 | yes |
| 1207.2442 | VectorBL | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.89 | yes |
| 1310.8098 | AxionPhoton, DarkPhoton | 5 | 5 | 0 | 0 | 1.267 | 0.000 | 0.000 | 1.27-1.27 | 0.00 | yes |
| 1401.6460 | AxionEDM, AxionMass | 5 | 5 | 0 | 0 | 6.213 | 1.529 | 0.606 | 3.52-7.09 | 6.54 | NO |
| 1403.1290 | MonopoleDipole | 5 | 5 | 0 | 0 | 17.370 | 5.767 | 6.242 | 8.49-22.88 | 12.25 | yes |
| 1410.7267 | ScalarBaryon, ScalarNucleon | 5 | 3 | 2 | 0 | 10.498 | 5.083 | 4.428 | 7.51-16.37 | 7.64 | yes |
| 1508.01798 | ScalarElectron | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 33.41 | yes |
| 1609.00667 | AxionPhoton | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.00 | yes |
| 1704.02297 | AxionElectron | 5 | 5 | 0 | 0 | 0.338 | 0.000 | 0.000 | 0.34-0.34 | 1.41 | yes |
| 1708.06367 | AxionEDM, AxionMass, AxionNeutron | 5 | 5 | 0 | 0 | 5.910 | 0.748 | 0.936 | 4.90-6.72 | 12.97 | yes |
| 1902.04644 | AxionNeutron | 5 | 3 | 0 | 0 | 2.122 | 0.782 | 0.678 | 1.67-3.03 | 1.10 | NO |
| 2004.02733 | AxionNeutron, AxionProton | 5 | 2 | 3 | 0 | 0.547 | 0.000 | 0.000 | 0.55-0.55 | 3.58 | yes |
| 2008.08773 | ScalarElectron, ScalarPhoton | 5 | 5 | 0 | 0 | 1.630 | 0.765 | 0.917 | 1.07-2.81 | 3.51 | yes |
| 2009.04517 | ScalarNucleon | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 0.00 | yes |
| 2111.09892 | AxionNeutron, AxionProton | 5 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | 5.68 | yes |
| 2504.00720 | AxionMass | 5 | 5 | 0 | 0 | 1.304 | 0.386 | 0.030 | 1.09-1.99 | 0.55 | yes |

## How to read this table

- `finite` = repeats that produced a finite median residual (a real same-curve comparison). `inf` = repeats with zero mass-range overlap. `N/A` rows are papers whose extracted coupling matched no comparable GT curve in every repeat (not scored).
- `pts std` is the run-to-run std of the extracted point count: even when the residual is stable, the number of digitised points can swing (e.g. 1508.01798: std 33 points), so point count is itself noisy and a poor stability signal on its own.
- A median-residual change below the **noise floor (0.32 dex)** for a well-behaved paper is run-to-run noise, not signal.

