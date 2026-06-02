# Zero-overlap diagnostic (issue #540)

Papers where the extracted curve and the ground-truth curve share **zero mass-range overlap** (interpolation coverage = 0%, median residual = inf). Pairing, boundary filtering, and the interpolation metric are reproduced exactly from `evaluate.py` / `metrics.py`.

**Total zero-overlap papers:** 32

## Breakdown by cause

| Cause | Count |
|---|---|
| `unit_offset` | 11 |
| `wrong_window` | 12 |
| `too_few_points` | 9 |
| **total** | **32** |

## Recurring conversion factors (unit_offset cases)

| Inferred factor | Papers |
|---|---|
| 2.418e+08 = MHz per eV (inverse) | 2 |
| 1.000e+09 = GeV per eV (1e9) | 2 |
| 4.136e-15 = eV per Hz (frequency->energy: E = h*f) | 2 |
| 1.000e+03 = 1e3 power-of-ten (keV<->eV / wrong exponent) | 2 |
| 1.000e+06 = 1e6 power-of-ten (MeV<->eV / wrong exponent) | 1 |
| 1.000e-06 = 1e-6 power-of-ten (eV<->MeV / micro prefix) | 1 |
| 4.136e-06 = eV per GHz (GHz->eV) | 1 |
## Per-paper detail

| arxiv_id | coupling | extracted mass [eV] | GT mass [eV] | offset (GT/ext) | n_ext | classification | note |
|---|---|---|---|---|---|---|---|
| 1406.6053 | AxionPhoton | — – — | 8.680e-13 – 3.886e+05 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 1806.00310 | AxionElectron | 5.800e-05 – 5.800e-05 | 4.137e-05 – 5.956e-05 | x8.56e-01 (-0.07 dex) | 1 | `too_few_points` | only 1 usable extracted point(s); interp1d needs >= 2 |
| 1902.04246 | AxionElectron | — – — | 1.082e-22 – 1.238e-18 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 2007.03694 | AxionElectron | — – — | 1.000e-30 – 1.000e+04 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 2008.05355 | AxionPhoton | — – — | 5.970e+09 – 9.254e+10 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 2110.10262 | AxionPhoton | 1.984e-05 – 1.984e-05 | 1.983e-05 – 1.985e-05 | x1.00e+00 (-0.00 dex) | 1 | `too_few_points` | only 1 usable extracted point(s); interp1d needs >= 2 |
| 2204.01454 | AxionEDM | — – — | 9.724e-20 – 3.311e-12 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 2308.06339 | AxionPhoton | 3.500e-01 – 3.500e-01 | 1.998e+08 – 4.505e+08 | x8.57e+08 (+8.93 dex) | 1 | `too_few_points` | only 1 usable extracted point(s); interp1d needs >= 2 |
| 2409.08998 | AxionPhoton | — – — | 1.726e-05 – 1.947e-05 | — | 0 | `too_few_points` | only 0 usable extracted point(s); interp1d needs >= 2 |
| 1709.00009 | AxionPhoton | 1.000e-04 – 2.000e+00 | 1.895e+04 – 4.982e+08 | x2.17e+08 (+8.34 dex) | 10 | `unit_offset` | offset ~ 2.418e+08 = MHz per eV (inverse) |
| 1810.04602 | AxionPhoton | 5.000e+03 – 9.000e+04 | 4.963e+09 – 7.018e+10 | x8.80e+05 (+5.94 dex) | 18 | `unit_offset` | offset ~ 1.000e+06 = 1e6 power-of-ten (MeV<->eV / wrong exponent) |
| 1907.11485 | DarkPhoton | 2.000e+08 – 5.000e+09 | 1.876e+02 – 5.458e+03 | x1.01e-06 (-5.99 dex) | 23 | `unit_offset` | offset ~ 1.000e-06 = 1e-6 power-of-ten (eV<->MeV / micro prefix) |
| 2007.13071 | AxionPhoton | 2.000e-01 – 9.700e+00 | 1.960e+08 – 8.742e+09 | x9.40e+08 (+8.97 dex) | 7 | `unit_offset` | offset ~ 1.000e+09 = GeV per eV (1e9) |
| 2111.06883 | ScalarElectron | 1.000e+01 – 1.000e+08 | 4.224e-14 – 4.002e-07 | x4.11e-15 (-14.39 dex) | 8 | `unit_offset` | offset ~ 4.136e-15 = eV per Hz (frequency->energy: E = h*f) |
| 2112.03439 | AxionPhoton | 1.317e+09 – 1.867e+09 | 1.431e-05 – 1.514e-05 | x9.39e-15 (-14.03 dex) | 29 | `unit_offset` | offset ~ 4.136e-15 = eV per Hz (frequency->energy: E = h*f) |
| 2207.11968 | AxionElectron | 1.000e+07 – 3.000e+09 | 3.066e+01 – 1.963e+04 | x4.48e-06 (-5.35 dex) | 30 | `unit_offset` | offset ~ 4.136e-06 = eV per GHz (GHz->eV) |
| 2209.13588 | AxionNeutron | 2.068e-15 – 2.068e-13 | 1.387e-12 – 1.997e-10 | x8.05e+02 (+2.91 dex) | 22 | `unit_offset` | offset ~ 1.000e+03 = 1e3 power-of-ten (keV<->eV / wrong exponent) |
| 2211.12699 | AxionPhoton | 1.650e-01 – 2.840e+00 | 1.812e+08 – 2.804e+09 | x1.04e+09 (+9.02 dex) | 9 | `unit_offset` | offset ~ 1.000e+09 = GeV per eV (1e9) |
| 2401.16747 | AxionPhoton | 1.800e+00 – 1.800e+01 | 1.797e+03 – 1.803e+04 | x1.00e+03 (+3.00 dex) | 26 | `unit_offset` | offset ~ 1.000e+03 = 1e3 power-of-ten (keV<->eV / wrong exponent) |
| 2410.10363 | AxionPhoton | 5.000e-02 – 5.000e-01 | 1.017e+07 – 7.235e+07 | x1.72e+08 (+8.23 dex) | 6 | `unit_offset` | offset ~ 2.418e+08 = MHz per eV (inverse) |
| 1310.8098 | AxionPhoton | 7.200e-06 – 7.200e-06 | 2.754e-06 – 5.900e-05 | x1.77e+00 (+0.25 dex) | 2 | `wrong_window` | ranges differ by x1.771e+00 (+0.25 dex); no clean conversion match |
| 1509.00476 | AxionPhoton | 1.000e-03 – 1.000e+01 | 2.033e+07 – 1.085e+11 | x1.49e+10 (+10.17 dex) | 5 | `wrong_window` | ranges differ by x1.485e+10 (+10.17 dex); no clean conversion match |
| 1606.07001 | AxionElectron | 1.000e-05 – 4.000e+01 | 1.000e-10 – 3.897e+04 | x9.87e-02 (-1.01 dex) | 4 | `wrong_window` | ranges differ by x9.871e-02 (-1.01 dex); no clean conversion match |
| 1607.06083 | AxionPhoton | 5.000e+03 – 1.000e+05 | 1.008e+07 – 6.120e+09 | x1.11e+04 (+4.05 dex) | 8 | `wrong_window` | ranges differ by x1.111e+04 (+4.05 dex); no clean conversion match |
| 1707.07921 | AxionElectron | 1.000e-05 – 2.500e+01 | 1.005e+03 – 9.898e+05 | x1.99e+06 (+6.30 dex) | 10 | `wrong_window` | ranges differ by x1.995e+06 (+6.30 dex); no clean conversion match |
| 1712.00483 | ScalarBaryon | 1.000e-14 – 1.000e-12 | 5.535e+04 – 1.000e+10 | x2.35e+20 (+20.37 dex) | 3 | `wrong_window` | ranges differ by x2.353e+20 (+20.37 dex); no clean conversion match |
| 2004.02733 | AxionNeutron | 1.000e-06 – 5.500e-03 | 1.000e-23 – 5.500e+06 | x1.00e-04 (-4.00 dex) | 6 | `wrong_window` | ranges differ by x1.000e-04 (-4.00 dex); no clean conversion match |
| 2102.06722 | AxionPhoton | 4.136e-10 – 4.136e-10 | 4.115e-10 – 8.472e-09 | x4.51e+00 (+0.65 dex) | 3 | `wrong_window` | ranges differ by x4.515e+00 (+0.65 dex); no clean conversion match |
| 2102.08764 | AxionElectron | 3.312e-05 – 3.313e-05 | 3.312e-05 – 3.313e-05 | x1.00e+00 (-0.00 dex) | 2 | `wrong_window` | ranges differ by x1.000e+00 (-0.00 dex); no clean conversion match |
| 2112.12116 | AxionElectron | 1.000e-03 – 1.000e+01 | 1.132e+01 – 4.056e+02 | x6.77e+02 (+2.83 dex) | 5 | `wrong_window` | ranges differ by x6.775e+02 (+2.83 dex); no clean conversion match |
| 2208.07293 | AxionEDM | 4.950e-09 – 5.020e-09 | 4.373e-10 – 5.718e-10 | x1.00e-01 (-1.00 dex) | 2 | `wrong_window` | ranges differ by x1.003e-01 (-1.00 dex); no clean conversion match |
| 2305.01002 | AxionPhoton | 1.000e-03 – 1.000e+00 | 8.011e+03 – 1.088e+08 | x2.95e+07 (+7.47 dex) | 5 | `wrong_window` | ranges differ by x2.952e+07 (+7.47 dex); no clean conversion match |
